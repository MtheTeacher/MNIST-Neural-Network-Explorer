import React from 'react';
import type { ModelConfig, LayerConfig } from '../types';
import { PlayIcon, StopIcon, PlusIcon, XIcon, LayersIcon, InfoIcon } from '../constants';
import { LEARNING_RATE_SCHEDULES } from '../constants';
import { LRScheduleVisualizer } from './LRScheduleVisualizer';

interface TrainingConfiguratorProps {
    config: ModelConfig;
    setConfig: React.Dispatch<React.SetStateAction<ModelConfig>>;
    onStartTraining: () => void;
    onStopTraining: () => void;
    isTraining: boolean;
    onShowInfo: (topic: string) => void;
}

const InfoButton: React.FC<{onClick: () => void; className?: string}> = ({ onClick, className }) => (
    <button onClick={onClick} className={`ml-2 text-gray-400 hover:text-cyan-300 transition-colors ${className}`} aria-label="More information">
        <InfoIcon className="w-4 h-4" />
    </button>
);

const Label: React.FC<{htmlFor: string, children: React.ReactNode, tooltip: string}> = ({htmlFor, children, tooltip}) => (
    <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-300 mb-1 group relative">
        <div className="flex items-center">
            {children}
        </div>
        <span className="absolute left-0 bottom-full mb-2 w-max max-w-xs bg-gray-800 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none shadow-lg z-10">
            {tooltip}
        </span>
    </label>
);


export const TrainingConfigurator: React.FC<TrainingConfiguratorProps> = ({ config, setConfig, onStartTraining, onStopTraining, isTraining, onShowInfo }) => {

    const updateLayer = (index: number, key: keyof LayerConfig, value: string | number) => {
        const newLayers = [...config.layers];
        if (key === 'units') {
            newLayers[index] = { ...newLayers[index], [key]: Math.max(1, Number(value)) };
        } else {
            newLayers[index] = { ...newLayers[index], [key]: value as LayerConfig['activation'] };
        }
        setConfig(prev => ({ ...prev, layers: newLayers }));
    };
    
    const addLayer = () => {
        setConfig(prev => ({
            ...prev,
            layers: [...prev.layers, { units: 64, activation: 'relu' }]
        }));
    };

    const removeLayer = (index: number) => {
        if (config.layers.length > 1) {
            const newLayers = config.layers.filter((_, i) => i !== index);
            setConfig(prev => ({ ...prev, layers: newLayers }));
        }
    };
    
    const applyPreset = (preset: 'simple' | 'deep' | 'wide' | 'cnn') => {
        if (preset === 'cnn') {
            setConfig(prev => ({ 
                ...prev, 
                architecture: 'cnn',
                learningRate: 0.001,
                epochs: 15,
                batchSize: 128,
                dropoutRate: 0, // Dropout is not applied to this CNN preset
            }));
            return;
        }

        let newLayers: LayerConfig[];
        switch (preset) {
            case 'simple':
                newLayers = [{ units: 128, activation: 'relu' }];
                break;
            case 'deep':
                newLayers = [
                    { units: 128, activation: 'relu' },
                    { units: 64, activation: 'relu' },
                    { units: 32, activation: 'relu' }
                ];
                break;
            case 'wide':
                newLayers = [{ units: 512, activation: 'relu' }];
                break;
        }
        setConfig(prev => ({ ...prev, layers: newLayers, architecture: 'dense' }));
    };


    return (
        <div className="bg-white/10 border border-white/20 rounded-2xl p-6 space-y-6 shadow-2xl">
            <h2 className="text-2xl font-bold text-white">1. Configure Model</h2>
            
             {/* Architecture Presets */}
            <div>
                <Label htmlFor="presets" tooltip="Quickly load common neural network architectures.">
                    <span>Architecture Presets</span>
                    <InfoButton onClick={() => onShowInfo('architecture')} />
                </Label>
                <div className="grid grid-cols-2 gap-2 mt-2">
                    <button onClick={() => applyPreset('simple')} className="bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 font-semibold py-2 px-2 rounded-lg transition duration-300 text-sm">Simple</button>
                    <button onClick={() => applyPreset('deep')} className="bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 font-semibold py-2 px-2 rounded-lg transition duration-300 text-sm">Deep</button>
                    <button onClick={() => applyPreset('wide')} className="bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 font-semibold py-2 px-2 rounded-lg transition duration-300 text-sm">Wide</button>
                    <button onClick={() => applyPreset('cnn')} className="bg-pink-500/20 text-pink-300 hover:bg-pink-500/40 font-semibold py-2 px-2 rounded-lg transition duration-300 text-sm">High-Accuracy CNN</button>
                </div>
            </div>

            {/* Layers Configuration */}
            {config.architecture === 'dense' ? (
                <div>
                    <Label htmlFor="layers" tooltip="Define the hidden layers of your neural network. More layers can capture more complex patterns but increase training time.">
                        <span className="flex items-center space-x-2">
                            <LayersIcon className="w-4 h-4"/>
                            <span>Hidden Layers</span>
                        </span>
                        <InfoButton onClick={() => onShowInfo('layers')} />
                    </Label>
                    <div className="space-y-3 mt-2">
                        {config.layers.map((layer, index) => (
                            <div key={index} className="flex items-center space-x-2 bg-black/20 p-2 rounded-lg">
                                <div className="flex-1">
                                    <Label htmlFor={`units-${index}`} tooltip="Number of neurons in this layer.">Units</Label>
                                    <input
                                        id={`units-${index}`}
                                        type="number"
                                        value={layer.units}
                                        onChange={(e) => updateLayer(index, 'units', e.target.value)}
                                        className="w-full bg-gray-900/50 border border-gray-600 rounded-md px-2 py-1 text-white focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition"
                                        min="1"
                                        step="16"
                                        disabled={isTraining}
                                    />
                                </div>
                                <div className="flex-1">
                                    <Label htmlFor={`activation-${index}`} tooltip="The activation function introduces non-linearity. 'ReLU' is a common choice.">Activation</Label>
                                    <select
                                        id={`activation-${index}`}
                                        value={layer.activation}
                                        onChange={(e) => updateLayer(index, 'activation', e.target.value)}
                                        className="w-full bg-gray-900/50 border border-gray-600 rounded-md px-2 py-1 text-white focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition"
                                        disabled={isTraining}
                                    >
                                        <option>relu</option>
                                        <option>sigmoid</option>
                                        <option>tanh</option>
                                    </select>
                                </div>
                                <button onClick={() => removeLayer(index)} disabled={config.layers.length <= 1 || isTraining} className="p-2 text-gray-400 hover:text-red-500 disabled:text-gray-600 disabled:cursor-not-allowed transition self-end mb-1">
                                    <XIcon className="w-5 h-5"/>
                                </button>
                            </div>
                        ))}
                    </div>
                     <button onClick={addLayer} disabled={isTraining} className="mt-3 w-full flex items-center justify-center space-x-2 bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 font-semibold py-2 px-4 rounded-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed">
                        <PlusIcon className="w-5 h-5"/>
                        <span>Add Layer</span>
                    </button>
                </div>
            ) : (
                <div className="bg-black/20 p-4 rounded-lg text-center border border-pink-500/30">
                  <p className="font-semibold text-pink-300">High-Accuracy CNN Selected</p>
                  <p className="text-sm text-gray-400 mt-1">This uses a powerful, pre-defined architecture. Layer configuration is disabled.</p>
                </div>
              )}


            {/* Hyperparameters */}
            <div className="space-y-4">
                <div>
                    <Label htmlFor="lrSchedule" tooltip="Determines how the learning rate changes during training.">
                        <span>Learning Rate Schedule</span>
                        <InfoButton onClick={() => onShowInfo('lr')} />
                    </Label>
                    <select
                        id="lrSchedule"
                        value={config.lrSchedule}
                        onChange={(e) => setConfig(prev => ({...prev, lrSchedule: e.target.value as ModelConfig['lrSchedule']}))}
                        className="w-full bg-gray-900/50 border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition"
                        disabled={isTraining}
                    >
                        {LEARNING_RATE_SCHEDULES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                    </select>
                </div>

                 <LRScheduleVisualizer schedule={config.lrSchedule} epochs={config.epochs} initialLr={config.learningRate} />

                <div>
                    <Label htmlFor="learningRate" tooltip="Controls how much to change the model in response to the estimated error each time the weights are updated. Smaller values require more training epochs.">
                        Learning Rate: {config.learningRate.toExponential(2)}
                    </Label>
                    <input
                        id="learningRate"
                        type="range"
                        min="0.0001"
                        max="0.02"
                        step="0.0001"
                        value={config.learningRate}
                        onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-thumb-cyan"
                        disabled={isTraining}
                    />
                </div>
            </div>

            {/* Regularization */}
            <div>
                <Label htmlFor="dropoutRate" tooltip="Fraction of neurons to randomly drop during training to prevent overfitting.">
                    Dropout Rate: {config.dropoutRate.toFixed(2)}
                </Label>
                <input
                    id="dropoutRate"
                    type="range"
                    min="0"
                    max="0.75"
                    step="0.05"
                    value={config.dropoutRate}
                    onChange={(e) => setConfig(prev => ({ ...prev, dropoutRate: parseFloat(e.target.value) }))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-thumb-cyan"
                    disabled={isTraining || config.architecture === 'cnn'}
                />
                 {config.architecture === 'cnn' && <p className="text-xs text-gray-400 mt-1">Dropout is disabled for the CNN preset.</p>}
            </div>
            
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <Label htmlFor="epochs" tooltip="One epoch is one full pass through the entire training dataset.">
                        <span>Epochs</span>
                        <InfoButton onClick={() => onShowInfo('epochs-batch')} />
                    </Label>
                    <input
                        id="epochs"
                        type="number"
                        value={config.epochs}
                        onChange={(e) => setConfig(prev => ({ ...prev, epochs: Math.max(1, parseInt(e.target.value, 10) || 1) }))}
                        className="w-full bg-gray-900/50 border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition"
                        min="1"
                        disabled={isTraining}
                    />
                </div>
                <div>
                    <Label htmlFor="batchSize" tooltip="The number of training examples utilized in one iteration.">
                        <span>Batch Size</span>
                        <InfoButton onClick={() => onShowInfo('epochs-batch')} />
                    </Label>
                    <input
                        id="batchSize"
                        type="number"
                        value={config.batchSize}
                        onChange={(e) => setConfig(prev => ({ ...prev, batchSize: Math.max(1, parseInt(e.target.value, 10) || 1) }))}
                        className="w-full bg-gray-900/50 border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 transition"
                        min="1"
                        step="32"
                        disabled={isTraining}
                    />
                </div>
            </div>

            {isTraining ? (
                <button
                    onClick={onStopTraining}
                    aria-label="Stop current training run"
                    className="w-full bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-400 hover:to-pink-400 text-white font-bold py-3 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                    <StopIcon className="w-5 h-5"/>
                    <span>Stop Training</span>
                </button>
            ) : (
                <button
                    onClick={onStartTraining}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-3 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                    <PlayIcon className="w-5 h-5"/>
                    <span>Start Training</span>
                </button>
            )}
        </div>
    );
};