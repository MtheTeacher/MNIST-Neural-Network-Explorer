import * as tf from '@tensorflow/tfjs';
import type { ModelConfig } from '../types';
import { getLearningRate } from './lrSchedulerService';

const MODEL_STORAGE_KEY = 'localstorage://mnist-model';
const MODEL_DOWNLOAD_KEY = 'downloads://mnist-model';

export function createModel(config: ModelConfig): tf.Sequential {
    const model = tf.sequential();

    if (config.architecture === 'cnn') {
        // CNN Architecture: Reshape -> Conv -> Pool -> Conv -> Pool -> Flatten -> Dense
        model.add(tf.layers.reshape({ inputShape: [784], targetShape: [28, 28, 1] }));

        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: 10,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));

    } else {
        // Dense Architecture (default)
        model.add(tf.layers.dense({
            inputShape: [784],
            units: config.layers[0].units,
            activation: config.layers[0].activation,
        }));
        if (config.dropoutRate > 0) {
            model.add(tf.layers.dropout({ rate: config.dropoutRate }));
        }

        for (let i = 1; i < config.layers.length; i++) {
            const layerConfig = config.layers[i];
            model.add(tf.layers.dense({
                units: layerConfig.units,
                activation: layerConfig.activation,
            }));
            if (config.dropoutRate > 0) {
                model.add(tf.layers.dropout({ rate: config.dropoutRate }));
            }
        }

        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));
    }

    // Fix: `tf.optimizers` is deprecated. Use `tf.train` to create optimizers.
    const optimizer = tf.train.adam(config.learningRate);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

export async function trainModel(
    model: tf.Sequential,
    trainImages: tf.Tensor,
    trainLabels: tf.Tensor,
    testImages: tf.Tensor,
    testLabels: tf.Tensor,
    config: ModelConfig,
    onEpochEndCallback: (epoch: number, logs: tf.Logs, lr: number) => Promise<void>
): Promise<tf.History> {

    const { learningRate: initialLr, epochs, lrSchedule } = config;
    let currentLr = initialLr;
    const valAccHistory: number[] = [];

    const lrCallback = new tf.CustomCallback({
        onEpochBegin: async (epoch) => {
            currentLr = getLearningRate(
                lrSchedule,
                epoch,
                epochs,
                initialLr,
                valAccHistory
            );
            // The `setLearningRate` method is not consistently available at runtime.
            // Directly setting the `learningRate` property is a more robust workaround.
            // We cast to 'any' to bypass TypeScript's protected property access check.
            (model.optimizer as any).learningRate = currentLr;
        },
    });

    const appCallback = new tf.CustomCallback({
         onEpochEnd: async (epoch, logs) => {
            if (logs) {
                 if (logs.val_acc) {
                    valAccHistory.push(logs.val_acc as number);
                }
                await onEpochEndCallback(epoch, logs, currentLr);
            }
        }
    });


    return model.fit(trainImages, trainLabels, {
        batchSize: config.batchSize,
        validationData: [testImages, testLabels],
        epochs: config.epochs,
        shuffle: true,
        callbacks: [lrCallback, appCallback],
    });
}


export async function saveModel(model: tf.Sequential): Promise<void> {
    await model.save(MODEL_STORAGE_KEY);
}

export async function downloadModel(model: tf.Sequential): Promise<void> {
    await model.save(MODEL_DOWNLOAD_KEY);
}

export async function loadModel(): Promise<tf.Sequential> {
    const loadedModel = await tf.loadLayersModel(MODEL_STORAGE_KEY);
    return loadedModel as tf.Sequential;
}

export async function checkForSavedModel(): Promise<boolean> {
    const models = await tf.io.listModels();
    return models[MODEL_STORAGE_KEY] !== undefined;
}

export async function deleteSavedModel(): Promise<void> {
    await tf.io.removeModel(MODEL_STORAGE_KEY);
}

/**
 * Applies global magnitude-based weight pruning to a model.
 * @param model The trained tf.Sequential model to prune.
 * @param targetSparsity The desired fraction of weights to be zeroed out (e.g., 0.9 for 90%).
 * @returns A new, pruned model and the actual sparsity achieved.
 */
export async function pruneModel(
    model: tf.Sequential,
    targetSparsity: number
): Promise<{ prunedModel: tf.Sequential; actualSparsity: number }> {
    return tf.tidy(() => {
        // Step 1: Collect all weight values from trainable layers (Dense, Conv2D)
        const allWeights: tf.Tensor[] = [];
        for (const layer of model.layers) {
            if (layer.getWeights().length > 0 && (layer.getClassName() === 'Dense' || layer.getClassName() === 'Conv2d')) {
                // We only prune the kernel (weights), not the bias
                allWeights.push(layer.getWeights()[0]);
            }
        }
        if (allWeights.length === 0) {
            throw new Error("No trainable weights found to prune.");
        }

        // Step 2: Find the magnitude threshold for the target sparsity
        const allValues = tf.concat(allWeights.map(w => w.flatten())).abs();
        const k = Math.ceil(allValues.size * (1 - targetSparsity));
        const topKValues = tf.topk(allValues, k, true).values;
        const threshold = topKValues.min(); // The smallest of the top k values is our threshold

        // Step 3: Create new weights with values below the threshold zeroed out
        const originalWeights = model.getWeights();
        const prunedWeights: tf.Tensor[] = [];
        let totalWeightCount = 0;
        let zeroWeightCount = 0;

        for (const originalWeight of originalWeights) {
            // Only prune kernels (weights), not biases. Biases are typically the odd-indexed tensors.
            const isKernel = originalWeight.rank > 1; 

            if (isKernel) {
                const mask = originalWeight.abs().greaterEqual(threshold);
                const prunedWeight = originalWeight.mul(mask);
                
                // For calculating actual sparsity
                // FIX: `tf.countNonZero` does not exist in tfjs. Use `notEqual(0).sum()` instead.
                const nonZeroCount = prunedWeight.notEqual(tf.scalar(0)).sum().dataSync()[0];
                const zeroValues = prunedWeight.size - nonZeroCount;
                zeroWeightCount += zeroValues;
                totalWeightCount += prunedWeight.size;

                prunedWeights.push(prunedWeight);
            } else {
                prunedWeights.push(originalWeight.clone()); // Keep biases as they are
                totalWeightCount += originalWeight.size; // Biases are non-zero
            }
        }

        // Step 4: Create a new model with the same architecture and set the pruned weights
        const config = model.getConfig();
        const prunedModel = tf.sequential(config as tf.SequentialArgs);
        
        // The `model.inputs` property provides the most reliable way to get the
        // model's input shape, as it's derived directly from the built model graph.
        // This avoids issues where layer configs might not contain the batchInputShape.
        const batchInputShape = model.inputs[0].shape;

        if (!batchInputShape || batchInputShape.length < 2) {
            throw new Error(`Could not determine a valid input shape from the model's inputs. Shape found: ${JSON.stringify(batchInputShape)}`);
        }
        
        // The build method expects the shape for a single sample, without the batch dimension.
        // The shape from model.inputs is [null, 784], so we slice it to get [784].
        const inputShapeForBuild = batchInputShape.slice(1);
        prunedModel.build(inputShapeForBuild);

        prunedModel.setWeights(prunedWeights);

        // Re-compile the model for fine-tuning
        prunedModel.compile({
            optimizer: tf.train.adam(0.0001), // A low LR is good for fine-tuning
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
        
        const actualSparsity = totalWeightCount > 0 ? zeroWeightCount / totalWeightCount : 0;
        
        // Clean up tensors that are no longer needed
        tf.dispose(allWeights);
        tf.dispose(allValues);
        tf.dispose(topKValues);
        tf.dispose(threshold);
        tf.dispose(originalWeights);
        // The tensors inside prunedWeights are now managed by the new model, so they shouldn't be disposed here.

        return { prunedModel, actualSparsity };
    });
}