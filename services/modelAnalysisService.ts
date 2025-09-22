
import * as tf from '@tensorflow/tfjs';
import type { ModelConfig, ModelInfo, LayerCalc } from '../types';

function generateModelName(config: ModelConfig): string {
    if (config.architecture === 'cnn') {
        return 'Convolutional Neural Network';
    }
    const layerUnits = config.layers.map(l => l.units).join(', ');
    return `Dense (${layerUnits})`;
}

export function analyzeModel(model: tf.Sequential, config: ModelConfig): ModelInfo {
    const layerCalcs: LayerCalc[] = [];
    let totalParams = 0;

    // Add input layer pseudo-info
    layerCalcs.push({
        name: 'input_layer',
        type: 'Input',
        outputShape: '[-1, 784]',
        params: 0,
        calculation: 'Input data (28x28 pixels flattened)',
    });

    model.layers.forEach((layer) => {
        const weights = layer.getWeights();
        if (weights.length === 0) { // Non-trainable layers like Flatten, Reshape, Pooling
            layerCalcs.push({
                name: layer.name,
                type: layer.getClassName(),
                outputShape: JSON.stringify(layer.outputShape),
                params: 0,
                calculation: 'No trainable parameters',
            });
            return;
        }

        const paramCount = layer.countParams();
        totalParams += paramCount;
        let calculation = 'N/A';
        
        if (layer.getClassName() === 'Dense') {
            const wShape = weights[0].shape; // [input_units, output_units]
            const bShape = weights[1].shape; // [output_units]
            const wCount = wShape[0] * wShape[1];
            const bCount = bShape[0];
            calculation = `Weights: (${wShape[0]} × ${wShape[1]}) + Biases: ${bShape[0]} = ${wCount.toLocaleString()} + ${bCount.toLocaleString()}`;
        } else if (layer.getClassName() === 'Conv2d') {
            const wShape = weights[0].shape; // [h, w, in, out]
            const bShape = weights[1].shape; // [out]
            const wCount = wShape[0] * wShape[1] * wShape[2] * wShape[3];
            const bCount = bShape[0];
            calculation = `Filters: (${wShape[0]}×${wShape[1]}×${wShape[2]})×${wShape[3]} + Biases: ${bShape[0]} = ${wCount.toLocaleString()} + ${bCount.toLocaleString()}`;
        }

        layerCalcs.push({
            name: layer.name,
            type: layer.getClassName(),
            outputShape: JSON.stringify(layer.outputShape).replace('null', '-1'),
            params: paramCount,
            calculation,
        });
    });

    return {
        name: generateModelName(config),
        totalParams,
        layerCalcs,
    };
}
