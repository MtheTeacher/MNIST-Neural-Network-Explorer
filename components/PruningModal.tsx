import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { ModelInfo, ModelConfig, TrainingLog, PruningInfo } from '../types';
import { XIcon, CutIcon } from '../constants';

interface TrainingRun {
    id: number;
    config: ModelConfig;
    log: TrainingLog[];
    modelInfo: ModelInfo | null;
    pruning?: PruningInfo;
}

interface PruningModalProps {
    run: TrainingRun;
    onClose: () => void;
    onStartFinetuning: (run: TrainingRun, targetSparsity: number) => void;
}

export const PruningModal: React.FC<PruningModalProps> = ({ run, onClose, onStartFinetuning }) => {
    const [targetSparsity, setTargetSparsity] = useState(0.8);

    if (!run.modelInfo) {
        // This should not happen if the button is rendered correctly
        return (
             <div 
                className="fixed inset-0 bg-black/70 backdrop-blur-md flex justify-center items-center z-50 p-4"
                onClick={onClose}
                role="dialog" aria-modal="true"
            >
                <div 
                    className="bg-gray-900 border border-red-500/50 rounded-2xl p-8 shadow-2xl max-w-lg w-full text-gray-300"
                    onClick={(e) => e.stopPropagation()}
                >
                    <h2 className="text-xl font-bold text-red-400">Error</h2>
                    <p>Model information is not available for pruning.</p>
                     <button onClick={onClose} className="mt-4 bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-4 rounded-full">Close</button>
                </div>
            </div>
        )
    }

    const { totalParams, layerCalcs, name } = run.modelInfo;
    const prunedParams = Math.round(totalParams * (1 - targetSparsity));
    
    const layerData = layerCalcs
        .filter(l => l.params > 0)
        .map(l => ({ name: l.name, params: l.params }));

    return (
        <div 
            className="fixed inset-0 bg-black/70 backdrop-blur-md flex justify-center items-center z-50 p-4 transition-opacity duration-300"
            onClick={onClose}
            role="dialog"
            aria-modal="true"
            aria-labelledby="pruning-modal-title"
        >
            <div 
                className="bg-gray-900 border border-white/20 rounded-2xl p-8 shadow-2xl max-w-2xl w-full text-gray-300 relative transform transition-all max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
            >
                <button 
                    onClick={onClose} 
                    className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors z-10"
                    aria-label="Close modal"
                >
                    <XIcon className="w-6 h-6" />
                </button>
                
                <h2 id="pruning-modal-title" className="text-2xl font-bold text-white mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-500">Prune & Fine-Tune Model</h2>
                <p className="text-gray-400 mb-6">Create a smaller, more efficient model by removing weights with the lowest magnitude.</p>
                
                <div className="space-y-6">
                    <div>
                        <label htmlFor="sparsity-slider" className="block text-lg font-medium text-gray-300 mb-2">
                            Target Sparsity: <span className="font-bold text-cyan-300">{Math.round(targetSparsity * 100)}%</span>
                        </label>
                        <input
                            id="sparsity-slider"
                            type="range"
                            min="0.1"
                            max="0.98"
                            step="0.01"
                            value={targetSparsity}
                            onChange={(e) => setTargetSparsity(parseFloat(e.target.value))}
                            className="w-full h-3 bg-gray-700 rounded-lg appearance-none cursor-pointer range-thumb-cyan"
                        />
                         <p className="text-sm text-gray-400 mt-2">This means you want to make {Math.round(targetSparsity * 100)}% of the weights equal to zero.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                        <div className="bg-black/20 p-4 rounded-lg">
                            <p className="text-sm text-gray-400">Original Parameters</p>
                            <p className="text-2xl font-bold text-white">{totalParams.toLocaleString()}</p>
                        </div>
                        <div className="bg-black/20 p-4 rounded-lg border border-cyan-500/50">
                            <p className="text-sm text-gray-400">New Non-Zero Parameters</p>
                            <p className="text-2xl font-bold text-cyan-300">~{prunedParams.toLocaleString()}</p>
                        </div>
                    </div>
                    
                     <div>
                        <h3 className="text-lg font-semibold text-gray-300 mb-2">Original Parameter Distribution</h3>
                        <div className="bg-black/20 p-4 rounded-lg" style={{ height: '250px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={layerData} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                                    <XAxis type="number" stroke="rgba(255, 255, 255, 0.5)" tickFormatter={(val) => val.toLocaleString()} />
                                    <YAxis type="category" dataKey="name" stroke="rgba(255, 255, 255, 0.7)" width={80} tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.1)' }}
                                        contentStyle={{
                                            backgroundColor: 'rgba(30, 41, 59, 0.9)',
                                            borderColor: 'rgba(255, 255, 255, 0.3)',
                                        }}
                                        formatter={(value: number) => [value.toLocaleString(), 'Parameters']}
                                    />
                                    <Bar dataKey="params" fill="#22d3ee" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t border-white/20">
                        <button 
                            onClick={() => onStartFinetuning(run, targetSparsity)}
                            className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-3 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg"
                        >
                            <CutIcon className="w-5 h-5" />
                            <span>Start Fine-Tuning</span>
                        </button>
                        <button 
                            onClick={onClose}
                            className="flex-1 sm:flex-initial bg-gray-600 hover:bg-gray-500 text-white font-bold py-3 px-4 rounded-full transition"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};