import * as tf from '@tensorflow/tfjs';
import type { ModelConfig } from '../types';

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

        for (let i = 1; i < config.layers.length; i++) {
            const layerConfig = config.layers[i];
            model.add(tf.layers.dense({
                units: layerConfig.units,
                activation: layerConfig.activation,
            }));
        }

        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));
    }

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
    onEpochEnd: (epoch: number, logs: any) => void
): Promise<tf.History> {

    return model.fit(trainImages, trainLabels, {
        batchSize: config.batchSize,
        validationData: [testImages, testLabels],
        epochs: config.epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd,
        },
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