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
 * This function is now robust, correctly managing tensor memory and reconstructing
 * the model architecture reliably.
 * @param model The trained tf.Sequential model to prune.
 * @param targetSparsity The desired fraction of weights to be zeroed out.
 * @param inputShape (No longer used for model reconstruction, but kept for signature compatibility).
 * @param modelJSON The model's architecture, captured after training.
 * @returns A new, pruned model and the actual sparsity achieved.
 */
export async function pruneModel(
    model: tf.Sequential,
    targetSparsity: number,
    inputShape: (number | null)[],
    modelJSON: object
): Promise<{ prunedModel: tf.Sequential; actualSparsity: number }> {
    
    // Use a tidy scope for the complex weight calculations, but `keep` the
    // resulting pruned weights so they don't get disposed.
    const { prunedWeights, actualSparsity } = tf.tidy(() => {
        // Step 1: Collect all weight values from trainable layers (Dense, Conv2D)
        const allWeights: tf.Tensor[] = [];
        for (const layer of model.layers) {
            if (layer.getWeights().length > 0 && (layer.getClassName() === 'Dense' || layer.getClassName() === 'Conv2d')) {
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
        const threshold = topKValues.min();

        // Step 3: Create new weights with values below the threshold zeroed out
        const originalWeights = model.getWeights();
        const newPrunedWeights: tf.Tensor[] = [];
        let totalWeightCount = 0;
        let zeroWeightCount = 0;

        for (const originalWeight of originalWeights) {
            // Only prune kernels (weights), not biases.
            const isKernel = originalWeight.rank > 1; 

            if (isKernel) {
                const mask = originalWeight.abs().greaterEqual(threshold);
                const prunedWeight = originalWeight.mul(mask);
                
                const nonZeroCount = prunedWeight.notEqual(tf.scalar(0)).sum().dataSync()[0];
                zeroWeightCount += (prunedWeight.size - nonZeroCount);
                totalWeightCount += prunedWeight.size;

                newPrunedWeights.push(prunedWeight);
            } else {
                newPrunedWeights.push(originalWeight); // Pass original tensor; will be cloned by `setWeights` later
                totalWeightCount += originalWeight.size;
            }
        }
        
        const newActualSparsity = totalWeightCount > 0 ? zeroWeightCount / totalWeightCount : 0;
        
        // This is CRITICAL: `tf.keep` prevents tidy() from disposing the tensors
        // that we need to return and set on the new model.
        tf.keep(newPrunedWeights);

        return {prunedWeights: newPrunedWeights, actualSparsity: newActualSparsity};
    });
    
    // Step 4: Create a new model from the saved JSON architecture.
    // This is the most robust way to create an identical, built model.
    if (!modelJSON) {
        throw new Error("Cannot prune model: valid modelJSON was not provided.");
    }
    const modelArtifacts: tf.io.ModelArtifacts = { modelTopology: modelJSON as tf.io.ModelJSON };
    const prunedModel = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts)) as tf.Sequential;

    // Set the new sparse weights. `setWeights` clones the tensors, so we can dispose ours after.
    prunedModel.setWeights(prunedWeights);
    tf.dispose(prunedWeights);

    // Re-compile the model for fine-tuning
    prunedModel.compile({
        optimizer: tf.train.adam(0.0001), // A low LR is good for fine-tuning
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return { prunedModel, actualSparsity };
}