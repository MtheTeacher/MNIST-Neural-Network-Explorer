import React from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Activation } from './ModelVisualizer';

interface NetworkActivationDiagramProps {
    model: tf.Sequential;
    activations: Activation[];
    highlightedLayer: number | null;
}

const MAX_UNITS_TO_DRAW = 32;

export const NetworkActivationDiagram: React.FC<NetworkActivationDiagramProps> = ({ model, activations, highlightedLayer }) => {
    
    if (!model) return null;

    const layers = [{ name: 'Input', getClassName: () => 'Input', getConfig: () => ({ units: 784 }) }, ...model.layers];

    const getLayerActivation = (layerIndex: number): Activation | null => {
        if (!activations || activations.length <= layerIndex) return null;
        return activations[layerIndex];
    };

    // --- RENDER HELPERS ---
    const renderInputLayer = (activation: Activation | null) => {
        const canvasSize = 60;
        const canvasRef = React.useRef<HTMLCanvasElement>(null);

        React.useEffect(() => {
            if (canvasRef.current && activation) {
                const ctx = canvasRef.current.getContext('2d');
                if (!ctx) return;

                const [height, width] = activation.shape.length > 2 ? [activation.shape[1], activation.shape[2]] : [28,28];
                canvasRef.current.width = width;
                canvasRef.current.height = height;
                const imageData = ctx.createImageData(width, height);
                const data = activation.data;
                for (let i = 0; i < data.length; i++) {
                    const value = data[i] * 255;
                    imageData.data[i * 4] = value;
                    imageData.data[i * 4 + 1] = value;
                    imageData.data[i * 4 + 2] = value;
                    imageData.data[i * 4 + 3] = 255;
                }
                ctx.putImageData(imageData, 0, 0);
            }
        }, [activation]);

        return (
            <div className="flex flex-col items-center">
                <canvas ref={canvasRef} style={{ width: canvasSize, height: canvasSize, imageRendering: 'pixelated' }} className="bg-black rounded-sm border border-white/20"/>
                <span className="text-xs font-bold mt-2">Input Image</span>
            </div>
        );
    };
    
    const renderDenseLayer = (layer: tf.layers.Layer, activation: Activation | null, isHighlighted: boolean) => {
        const config = layer.getConfig() as { units: number };
        const units = config.units;
        const displayUnits = Math.min(units, MAX_UNITS_TO_DRAW);
        const activationData = activation?.data;

        const maxActivation = activationData ? Math.max(...Array.from(activationData)) : 1;
        
        return (
            <div className={`flex flex-col items-center p-2 rounded-lg transition-all ${isHighlighted ? 'bg-cyan-500/20' : ''}`}>
                <div className="flex flex-col items-center space-y-1">
                    {Array.from({ length: displayUnits }).map((_, i) => {
                        const neuronIndex = Math.floor(i * (units / displayUnits));
                        const value = activationData ? activationData[neuronIndex] / (maxActivation || 1) : 0;
                        const isOutput = units === 10;
                        const isPredicted = isOutput && activationData && activationData[neuronIndex] === maxActivation;

                        return (
                             <div key={i} className="flex items-center space-x-2">
                                {isOutput && <span className={`text-xs w-3 text-right font-mono ${isPredicted ? 'text-cyan-300 font-bold' : 'text-gray-400'}`}>{neuronIndex}</span>}
                                <div
                                    className={`w-4 h-4 rounded-full bg-cyan-400 transition-opacity duration-150 ${isPredicted ? 'border-2 border-white' : ''}`}
                                    style={{ opacity: value }}
                                    title={`Neuron ${neuronIndex}: ${activationData ? activationData[neuronIndex]?.toFixed(3) : 'N/A'}`}
                                />
                             </div>
                        );
                    })}
                </div>
                {units > MAX_UNITS_TO_DRAW && <span className="text-xs text-gray-400 mt-1">...</span>}
                <span className="text-xs font-bold mt-2">{layer.name} ({units})</span>
            </div>
        );
    };

    const renderConvLayer = (layer: tf.layers.Layer, activation: Activation | null, isHighlighted: boolean) => {
         const activationShape = activation?.shape;
         if (!activationShape || activationShape.length < 4) return null; // expecting [batch, height, width, channels]

         const channels = activationShape[3];
         const displayChannels = Math.min(channels, 16);
         const featureMapSize = 32;

         return (
             <div className={`flex flex-col items-center p-2 rounded-lg transition-all ${isHighlighted ? 'bg-cyan-500/20' : ''}`}>
                 <div className="grid grid-cols-4 gap-1">
                     {Array.from({ length: displayChannels }).map((_, i) => {
                         const canvasRef = React.useRef<HTMLCanvasElement>(null);
                         React.useEffect(() => {
                             if (canvasRef.current && activation) {
                                 tf.tidy(() => {
                                     // Fix: Explicitly cast the shape to help TypeScript infer a Tensor4D.
                                     const tensor = tf.tensor(activation.data, activation.shape as [number, number, number, number]);
                                     const featureMap = tensor.slice([0, 0, 0, i], [1, -1, -1, 1]);
                                     // FIX: Cast featureMap to Tensor4D, as tf.image.resizeBilinear expects a specific tensor rank.
                                     const resizedMap = tf.image.resizeBilinear(featureMap as tf.Tensor4D, [featureMapSize, featureMapSize]);
                                     // FIX: Cast the squeezed tensor to Tensor2D for tf.browser.toPixels.
                                     tf.browser.toPixels(resizedMap.squeeze() as tf.Tensor2D, canvasRef.current!);
                                 });
                             }
                         }, [activation]);
                         return <canvas key={i} ref={canvasRef} width={featureMapSize} height={featureMapSize} className="bg-black rounded-sm" />;
                     })}
                 </div>
                 <span className="text-xs font-bold mt-2">{layer.name} ({channels} filters)</span>
             </div>
         );
    };

    return (
        <div className="w-full flex items-center justify-around space-x-2 overflow-x-auto p-4">
           {layers.map((layer, i) => {
                const activation = getLayerActivation(i);
                const isHighlighted = highlightedLayer === i;
                const className = layer.getClassName();

                let layerVisual;
                if (className === 'Input') {
                    layerVisual = renderInputLayer(activation);
                } else if (className.includes('Dense')) {
                    // Fix: Cast layer to tf.layers.Layer to satisfy the function signature.
                    // TypeScript's control flow analysis doesn't recognize that if className is 'Dense',
                    // the layer must be a tf.layers.Layer and not the custom input object.
                    layerVisual = renderDenseLayer(layer as tf.layers.Layer, activation, isHighlighted);
                } else if (className.includes('Conv2d') || className.includes('Pooling')) {
                    // Fix: Cast layer to tf.layers.Layer for the same reason as above.
                    layerVisual = renderConvLayer(layer as tf.layers.Layer, activation, isHighlighted);
                } else if (className.includes('Flatten') || className.includes('Reshape')) {
                    layerVisual = (
                        <div className={`flex flex-col items-center p-2 transition-all ${isHighlighted ? 'bg-cyan-500/20 rounded-lg' : ''}`}>
                            <div className="w-2 h-10 bg-gray-600 rounded-full"/>
                            <span className="text-xs font-bold mt-2">{layer.name}</span>
                        </div>
                    );
                }

                return (
                    <React.Fragment key={i}>
                        {layerVisual}
                        {i < layers.length - 1 && (
                            <div className="flex-1 min-w-[20px] h-px bg-gradient-to-r from-gray-600 to-gray-800" />
                        )}
                    </React.Fragment>
                );
           })}
        </div>
    );
};