

import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Activation } from './ModelVisualizer';
import { EyeIcon } from '../constants';

const VISUALIZATION_SIZE = 200; // px

const renderActivationToCanvas = (
    canvas: HTMLCanvasElement,
    activation: Activation,
    layerClassName: string
) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    tf.tidy(() => {
        const tensor = tf.tensor(activation.data, activation.shape);

        // Normalize tensor values to [0, 1] for visualization.
        // Add a small epsilon to avoid division by zero if all activations are the same.
        const maxVal = tensor.max();
        const minVal = tensor.min();
        const normalizedTensor = tensor.sub(minVal).div(maxVal.sub(minVal).add(tf.backend().epsilon()));

        if (tensor.isDisposed || normalizedTensor.isDisposed) return;
        
        let canvasContent: tf.Tensor;
        
        if (layerClassName === 'Input') {
            canvas.width = 28;
            canvas.height = 28;
            canvasContent = normalizedTensor.reshape([28, 28, 1]);
        } else if (layerClassName.includes('Dense')) {
            const units = activation.shape[1];
            const gridSide = Math.ceil(Math.sqrt(units));
            canvas.width = gridSide;
            canvas.height = gridSide;
            
            const normalizedPaddedData = new Float32Array(gridSide * gridSide).fill(0);
            const dataSync = normalizedTensor.dataSync();
            normalizedPaddedData.set(dataSync);
            canvasContent = tf.tensor2d(normalizedPaddedData, [gridSide, gridSide]).expandDims(-1);
        } else if (layerClassName.includes('Conv2d') || layerClassName.includes('Pooling')) {
            const [_, height, width, channels] = activation.shape as [number, number, number, number];
            const numCols = Math.ceil(Math.sqrt(channels));
            const numRows = Math.ceil(channels / numCols);

            const montage = tf.tidy(() => {
                const featureMaps = normalizedTensor.transpose([3, 0, 1, 2]);
                const unstacked = tf.unstack(featureMaps);
                
                const rows = [];
                for (let i = 0; i < numRows; i++) {
                    const rowChannels = unstacked.slice(i * numCols, (i * numCols) + numCols);
                    while (rowChannels.length < numCols) {
                        rowChannels.push(tf.zerosLike(rowChannels[0]));
                    }
                    rows.push(tf.concat(rowChannels, 1));
                }
                return tf.concat(rows, 0).expandDims(-1);
            });
            canvas.width = montage.shape[1];
            canvas.height = montage.shape[0];
            canvasContent = montage;
        } else {
            return;
        }

        // FIX: Cast `canvasContent` to a compatible rank for tf.browser.toPixels.
        // All logic paths produce a Tensor3D, so this cast is safe.
        tf.browser.toPixels(canvasContent as tf.Tensor3D, canvas);
    });
};

const LayerActivation: React.FC<{ activation: Activation, layerClassName: string, layerName: string }> = ({ activation, layerClassName, layerName }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
        if (canvasRef.current) {
            renderActivationToCanvas(canvasRef.current, activation, layerClassName);
        }
    }, [activation, layerClassName]);

    return (
        <div className="flex flex-col items-center flex-shrink-0 space-y-2 w-56 text-center">
            <h4 className="font-semibold text-sm truncate w-full" title={layerName}>{layerName}</h4>
            <canvas
                ref={canvasRef}
                className="bg-black border border-white/20 rounded-md"
                style={{ width: VISUALIZATION_SIZE, height: VISUALIZATION_SIZE, imageRendering: 'pixelated' }}
            />
        </div>
    );
};

export const LayerActivationHeatmaps: React.FC<{ model: tf.Sequential, activations: Activation[] }> = ({ model, activations }) => {
    if (activations.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center bg-black/20 rounded-lg">
                <EyeIcon className="w-16 h-16 text-gray-500 mb-4" />
                <h4 className="font-semibold text-lg">Awaiting Input</h4>
                <p className="text-gray-400 max-w-sm">Use the controls on the left to provide an input image, then click "See Activation Flow" to populate this view.</p>
            </div>
        );
    }

    const layers = [{ name: 'Input', getClassName: () => 'Input' }, ...model.layers];

    return (
        <div className="flex flex-row space-x-6 overflow-x-auto p-4 bg-black/20 rounded-lg border border-white/20 min-h-[260px]">
            {activations.map((activation, i) => {
                const layer = layers[i];
                const layerClassName = layer.getClassName();

                if (layerClassName.includes('Flatten') || layerClassName.includes('Reshape')) {
                    return null;
                }

                return (
                    <LayerActivation
                        key={`${layer.name}-${i}`}
                        activation={activation}
                        layerClassName={layerClassName}
                        layerName={layer.name}
                    />
                );
            })}
        </div>
    );
};
