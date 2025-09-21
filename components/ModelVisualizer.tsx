import React, { useState, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { XIcon, ChevronDownIcon, SparklesIcon, EraserIcon } from '../constants';
import { WeightHeatmap } from './WeightHeatmap';
import { DrawingCanvas, type DrawingCanvasRef } from './DrawingCanvas';
import { NetworkActivationDiagram } from './NetworkActivationDiagram';


interface ModelVisualizerProps {
    model: tf.Sequential;
    onClose: () => void;
}

export interface Activation {
    data: Float32Array;
    shape: number[];
}

const LayerCard: React.FC<{ layer: tf.layers.Layer, index: number }> = ({ layer, index }) => {
    const weights = layer.getWeights();
    const layerClassName = layer.getClassName();

    const renderVisuals = () => {
        if (weights.length === 0) {
            return <p className="text-gray-400 italic text-center col-span-full">This layer has no trainable weights.</p>;
        }

        const weightTensor = weights[0];
        
        if (layerClassName === 'Conv2d') {
            const outChannels = weightTensor.shape[3];
            const filters = tf.tidy(() => {
                const transposed = weightTensor.transpose([3,0,1,2]);
                const splitFilters = tf.split(transposed, outChannels, 0);
                return splitFilters.map(f => f.squeeze());
            });
            
            return (
                 <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                    {filters.map((filter, i) => (
                        <div key={i} className="flex flex-col items-center">
                            <WeightHeatmap tensor={filter} canvasSize={60} />
                            <span className="text-xs mt-1 text-gray-400">Filter {i+1}</span>
                        </div>
                    ))}
                </div>
            );
        }

        if (layerClassName === 'Dense' && weightTensor.shape[0] === 784) {
            const neurons = tf.tidy(() => {
                const splitNeurons = tf.split(weightTensor, weightTensor.shape[1], 1);
                return splitNeurons.map(n => n.reshape([28, 28]));
            });

            return (
                <div className="grid grid-cols-8 sm:grid-cols-12 lg:grid-cols-16 gap-2">
                    {neurons.slice(0, 128).map((neuron, i) => (
                        <div key={i} className="flex flex-col items-center">
                            <WeightHeatmap tensor={neuron} canvasSize={40} />
                        </div>
                    ))}
                </div>
            )
        }

        if (layerClassName === 'Dense') {
            return (
                <div className="w-full flex justify-center items-center overflow-x-auto">
                    <WeightHeatmap tensor={weightTensor.clone()} canvasSize={Math.min(400, weightTensor.shape[1])} />
                </div>
            );
        }

        return null;
    };

    return (
         <div className="bg-white/5 border border-white/20 rounded-2xl p-6 shadow-xl mb-8">
            <h3 className="text-xl font-bold mb-1 text-cyan-300">Layer {index + 1}: {layer.name}</h3>
            <div className="text-sm text-gray-300 mb-4 flex flex-wrap gap-x-4">
                <span>Type: <span className="font-mono text-pink-300">{layerClassName}</span></span>
                {weights.length > 0 && <span>Weights Shape: <span className="font-mono text-pink-300">{`[${weights[0].shape.join(', ')}]`}</span></span>}
                {weights.length > 1 && <span>Biases Shape: <span className="font-mono text-pink-300">{`[${weights[1].shape.join(', ')}]`}</span></span>}
            </div>
            {renderVisuals()}
        </div>
    );
};

const AccordionItem: React.FC<{
    title: string;
    name: 'weights' | 'activations';
    activePanel: string | null;
    setActivePanel: (name: 'weights' | 'activations' | null) => void;
    children: React.ReactNode;
}> = ({ title, name, activePanel, setActivePanel, children }) => {
    const isOpen = activePanel === name;
    return (
        <div className="bg-white/5 border border-white/20 rounded-2xl mb-4 overflow-hidden transition-all duration-300">
            <button
                onClick={() => setActivePanel(isOpen ? null : name)}
                className="w-full flex justify-between items-center p-4 text-left text-lg font-bold hover:bg-white/10"
                aria-expanded={isOpen}
            >
                <span>{title}</span>
                <ChevronDownIcon className={`w-6 h-6 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
            </button>
            {isOpen && (
                <div className="p-4 sm:p-6 border-t border-white/20">
                    {children}
                </div>
            )}
        </div>
    );
};


export const ModelVisualizer: React.FC<ModelVisualizerProps> = ({ model, onClose }) => {
    const [activePanel, setActivePanel] = useState<'weights' | 'activations' | null>('weights');
    const [activations, setActivations] = useState<Activation[]>([]);
    const [highlightedLayer, setHighlightedLayer] = useState<number | null>(null);
    const [isVisualizing, setIsVisualizing] = useState(false);
    const canvasRef = useRef<DrawingCanvasRef>(null);

    const handleVisualizeActivations = useCallback(async () => {
        setIsVisualizing(true);
        setActivations([]);
        
        const cleanupTensors: tf.Tensor[] = [];

        try {
            const inputTensor = canvasRef.current?.getTensor();
            if (!inputTensor) {
                setIsVisualizing(false);
                return;
            };
            cleanupTensors.push(inputTensor);

            const outputs = model.layers.map(layer => layer.output);
            // Fix: tf.tidy is for tensors, not models. Models are JS objects that don't need disposal.
            const intermediateModels = outputs.map(output => 
                tf.model({ inputs: model.inputs, outputs: output as tf.SymbolicTensor })
            );
            // Fix: Do not push models to the tensor cleanup array. This was causing a type error
            // and is conceptually incorrect.
            
            const newActivations: Activation[] = [];
            // Add input tensor as the first "activation"
            newActivations.push({ data: await inputTensor.data() as Float32Array, shape: inputTensor.shape });

            // Fix: The loop variable 'intermediateModels' is now correctly typed as tf.LayersModel[]
            for (const intermediateModel of intermediateModels) {
                const activationTensor = intermediateModel.predict(inputTensor) as tf.Tensor;
                cleanupTensors.push(activationTensor);
                newActivations.push({
                    data: await activationTensor.data() as Float32Array,
                    shape: activationTensor.shape
                });
            }
            setActivations(newActivations);

            for (let i = 0; i < newActivations.length; i++) {
                setHighlightedLayer(i);
                await new Promise(res => setTimeout(res, 150));
            }

        } catch (error) {
            console.error("Error during activation visualization:", error);
        } finally {
            tf.dispose(cleanupTensors);
            setHighlightedLayer(null);
            setIsVisualizing(false);
        }
    }, [model]);

    const handleClear = () => {
        canvasRef.current?.clearCanvas();
        setActivations([]);
        setHighlightedLayer(null);
    };

    return (
        <div className="min-h-screen w-full bg-black/80 backdrop-blur-md flex flex-col items-center p-4 sm:p-8">
            <header className="w-full max-w-7xl mx-auto flex justify-between items-center mb-8">
                <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-500">
                    Model Visualizer
                </h1>
                <button onClick={onClose} className="p-2 bg-white/10 hover:bg-red-500/50 rounded-full transition-colors">
                    <XIcon className="w-6 h-6"/>
                </button>
            </header>
            <main className="w-full max-w-7xl mx-auto overflow-y-auto" style={{maxHeight: 'calc(100vh - 120px)'}}>
                <AccordionItem title="Layer Weight Heatmaps" name="weights" activePanel={activePanel} setActivePanel={setActivePanel}>
                    {model.layers.map((layer, i) => <LayerCard key={layer.name + i} layer={layer} index={i} />)}
                </AccordionItem>
                <AccordionItem title="Live Activation Viewer" name="activations" activePanel={activePanel} setActivePanel={setActivePanel}>
                    <div className="flex flex-col lg:flex-row gap-8">
                        <div className="lg:w-1/4 space-y-4">
                            <h3 className="text-lg font-semibold">1. Draw a Digit</h3>
                             <DrawingCanvas ref={canvasRef} />
                             <div className="space-y-2">
                                <button
                                    onClick={handleVisualizeActivations}
                                    disabled={isVisualizing}
                                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-wait"
                                >
                                    <SparklesIcon className="w-5 h-5"/>
                                    <span>{isVisualizing ? 'Visualizing...' : 'See Activation Flow'}</span>
                                </button>
                                <button
                                    onClick={handleClear}
                                    disabled={isVisualizing}
                                    className="w-full bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300"
                                >
                                    <EraserIcon className="w-5 h-5"/>
                                    <span>Clear</span>
                                </button>
                             </div>
                        </div>
                        <div className="lg:w-3/4">
                             <h3 className="text-lg font-semibold mb-4">2. Activation Propagation</h3>
                             <div className="bg-black/20 p-4 rounded-lg border border-white/20 min-h-[300px]">
                                <NetworkActivationDiagram 
                                    model={model} 
                                    activations={activations}
                                    highlightedLayer={highlightedLayer}
                                />
                             </div>
                        </div>
                    </div>
                </AccordionItem>
            </main>
        </div>
    );
};
