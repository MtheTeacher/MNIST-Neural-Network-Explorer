import type * as tf from '@tensorflow/tfjs';

export interface LayerConfig {
    units: number;
    activation: 'relu' | 'sigmoid' | 'softmax' | 'tanh';
}

export interface ModelConfig {
    layers: LayerConfig[];
    learningRate: number;
    epochs: number;
    batchSize: number;
    architecture: 'dense' | 'cnn';
}

export interface TrainingLog {
    epoch: number;
    loss: number;
    accuracy: number;
}
