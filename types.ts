import type * as tf from '@tensorflow/tfjs';

export interface LayerConfig {
    units: number;
    activation: 'relu' | 'sigmoid' | 'softmax' | 'tanh';
}

export type LRSchedulerId =
    | 'constant'
    | 'step'
    | 'exponential'
    | 'cosine'
    | 'warmup-cosine'
    | 'plateau'
    | 'one-cycle'
    | 'cosine-restarts';

export interface LRSchedulerConfig {
    id: LRSchedulerId;
    name: string;
    description: string;
}


export interface ModelConfig {
    layers: LayerConfig[];
    learningRate: number; // This is the initial/max learning rate
    lrSchedule: LRSchedulerId;
    epochs: number;
    batchSize: number;
    architecture: 'dense' | 'cnn';
    dropoutRate: number;
}

export interface TrainingLog {
    epoch: number;
    loss: number;
    accuracy: number;
    val_loss?: number;
    val_accuracy?: number;
    lr?: number;
}

export interface MnistSample {
    tensor: tf.Tensor; // Shape [1, 784]
    label: number;
    id: number; // index in the original test set
}

export interface LayerCalc {
    name: string;
    type: string;
    outputShape: string;
    params: number;
    calculation: string;
}

export interface ModelInfo {
    name: string;
    totalParams: number;
    layerCalcs: LayerCalc[];
}

export interface PruningInfo {
    fromRunId: number;
    sparsity: number;
}
