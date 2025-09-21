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
    | 'one-cycle';

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
}

export interface TrainingLog {
    epoch: number;
    loss: number;
    accuracy: number;
    lr?: number;
}

export interface MnistSample {
    tensor: tf.Tensor; // Shape [1, 784]
    label: number;
    id: number; // index in the original test set
}
