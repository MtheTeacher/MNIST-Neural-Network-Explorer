import React from 'react';
import type { TrainingLog, ModelConfig, ModelInfo, PruningInfo } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LEARNING_RATE_SCHEDULES } from '../constants';
import { RocketIcon, EyeIcon, InfoIcon, ChevronDownIcon, CutIcon } from '../constants';
import type { TrainingRun as FullTrainingRun } from '../App';


interface TrainingRun {
    id: number;
    config: ModelConfig;
    log: TrainingLog[];
    modelInfo: ModelInfo | null;
    pruning?: PruningInfo;
}

interface TrainingDashboardProps {
    isLive: boolean;
    status?: string;
    // Props for live run
    trainingLog?: TrainingLog[];
    config?: ModelConfig;
    modelInfo?: ModelInfo | null;
    // Props for completed run
    run?: TrainingRun;
    parentRun?: FullTrainingRun;
    onTestModel?: () => void;
    onVisualizeModel?: () => void;
    onPruneModel?: (run: FullTrainingRun) => void;
    isModelInTest?: boolean;
}

const getScheduleName = (id: string) => LEARNING_RATE_SCHEDULES.find(s => s.id === id)?.name || 'Unknown';

export const TrainingDashboard: React.FC<TrainingDashboardProps> = (props) => {
    const { isLive, status, onTestModel, onVisualizeModel, onPruneModel, isModelInTest, parentRun } = props;
    
    // Consolidate props for live vs completed runs
    const runData = isLive
        ? { log: props.trainingLog!, config: props.config!, modelInfo: props.modelInfo, pruning: undefined }
        : { log: props.run!.log, config: props.run!.config, modelInfo: props.run!.modelInfo, pruning: props.run!.pruning };
    
    const { log: trainingLog, config, modelInfo, pruning } = runData;

    const progress = trainingLog.length > 0 ? (trainingLog[trainingLog.length - 1].epoch / config.epochs) * 100 : (isLive ? 0 : 100);
    
    const lastLog = !isLive && trainingLog.length > 0 ? trainingLog[trainingLog.length - 1] : null;
    const finalAccuracy = lastLog ? (lastLog.val_accuracy ?? lastLog.accuracy) : null;
    const hasValidationData = lastLog && lastLog.val_accuracy !== undefined;
    
    let effectiveParams: number | null = null;
    let accuracyDiff: number | null = null;

    if (pruning && modelInfo) {
        effectiveParams = Math.round(modelInfo.totalParams * (1 - pruning.sparsity));
    }

    if (pruning && parentRun && finalAccuracy && parentRun.log.length > 0) {
        const parentLastLog = parentRun.log[parentRun.log.length - 1];
        if (parentLastLog) {
            const parentAccuracy = parentLastLog.val_accuracy ?? parentLastLog.accuracy;
            accuracyDiff = finalAccuracy - parentAccuracy;
        }
    }

    const title = pruning
        ? `Pruned Model (${(pruning.sparsity * 100).toFixed(0)}% sparse)`
        : (modelInfo?.name || (isLive ? "Live Training" : "Completed Run"));

    return (
        <div className={`bg-white/10 border rounded-2xl p-6 shadow-2xl transition-all duration-300 ${isLive ? 'border-cyan-400/50' : 'border-white/20'}`}>
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4 mb-4">
                <div>
                    <h2 className="text-xl font-bold text-white">{title}</h2>
                    <div className="text-xs text-gray-300 flex flex-wrap gap-x-3 gap-y-1 mt-1">
                        <span>Arch: <span className="font-semibold text-gray-200">{config.architecture.toUpperCase()}</span></span>
                        <span>LR Schedule: <span className="font-semibold text-gray-200">{getScheduleName(config.lrSchedule)}</span></span>
                        <span>Initial LR: <span className="font-semibold text-gray-200">{config.learningRate}</span></span>
                        <span>Epochs: <span className="font-semibold text-gray-200">{config.epochs}</span></span>
                        {config.dropoutRate > 0 && <span>Dropout: <span className="font-semibold text-gray-200">{config.dropoutRate}</span></span>}
                        {modelInfo && <span>Params: <span className="font-semibold text-gray-200">{modelInfo.totalParams.toLocaleString()}</span></span>}
                        {pruning && <span>Sparsity: <span className="font-semibold text-gray-200">{(pruning.sparsity * 100).toFixed(0)}%</span></span>}
                        {effectiveParams !== null && (
                            <span>Effective Params: <span className="font-semibold text-cyan-300">{effectiveParams.toLocaleString()}</span></span>
                        )}
                        {accuracyDiff !== null && (
                             <span>Validation Acc. vs Unpruned: <span className={`font-semibold ${accuracyDiff >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {accuracyDiff >= 0 ? '+' : ''}{(accuracyDiff * 100).toFixed(2)}%
                            </span></span>
                        )}
                    </div>
                </div>
                {!isLive && (
                     <div className="flex flex-col sm:flex-row gap-2">
                        {onTestModel && (
                            <button
                                onClick={onTestModel}
                                className={`font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 text-sm ${isModelInTest ? 'bg-pink-500 text-white' : 'bg-white/20 hover:bg-white/30 text-white'}`}
                            >
                                <RocketIcon className="w-4 h-4"/>
                                <span>{isModelInTest ? 'Currently Testing' : 'Test Model'}</span>
                            </button>
                        )}
                        {onVisualizeModel && (
                             <button
                                onClick={onVisualizeModel}
                                className="font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 text-sm bg-white/20 hover:bg-white/30 text-white"
                            >
                                <EyeIcon className="w-4 h-4"/>
                                <span>Visualize</span>
                            </button>
                        )}
                        {onPruneModel && config.architecture === 'dense' && (
                             <button
                                onClick={() => onPruneModel(props.run as FullTrainingRun)}
                                className="font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 text-sm bg-white/20 hover:bg-white/30 text-white"
                            >
                                <CutIcon className="w-4 h-4"/>
                                <span>Prune & Fine-Tune</span>
                            </button>
                        )}
                    </div>
                )}
            </div>

             {isLive && <div className="mb-6">
                <div className="flex justify-between mb-1">
                    <span className="text-base font-medium text-cyan-300">{status}</span>
                    <span className="text-sm font-medium text-cyan-300">{trainingLog.length} / {config.epochs} Epochs</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                    <div className="bg-gradient-to-r from-cyan-400 to-pink-500 h-2.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
                </div>
            </div>}
            
            {!isLive && finalAccuracy !== null && (
                <div className="mb-4 text-center bg-black/20 p-2 rounded-lg">
                    <span className="text-lg font-semibold text-gray-200">{hasValidationData ? 'Final Validation Accuracy' : 'Final Accuracy'}: </span>
                    <span className="text-lg font-bold text-cyan-300">{(finalAccuracy * 100).toFixed(2)}%</span>
                </div>
            )}

            {modelInfo && (
                <details className="mb-6 bg-black/20 rounded-lg border border-white/10 group">
                    <summary className="p-3 flex justify-between items-center cursor-pointer list-none text-gray-300 hover:text-white">
                        <span className="font-semibold flex items-center space-x-2">
                            <InfoIcon className="w-4 h-4" />
                            <span>Model Architecture & Parameters</span>
                        </span>
                        <ChevronDownIcon className="w-5 h-5 transition-transform transform group-open:rotate-180" />
                    </summary>
                    <div className="p-4 border-t border-white/10">
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm">
                                <thead className="text-xs text-gray-400 uppercase">
                                    <tr>
                                        <th className="py-2 px-3">Layer (Type)</th>
                                        <th className="py-2 px-3">Output Shape</th>
                                        <th className="py-2 px-3">Param #</th>
                                        <th className="py-2 px-3">Calculation</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-700">
                                    {modelInfo.layerCalcs.map((layer, index) => (
                                        <tr key={index}>
                                            <td className="py-2 px-3 font-mono">{layer.name} ({layer.type})</td>
                                            <td className="py-2 px-3 font-mono">{layer.outputShape}</td>
                                            <td className="py-2 px-3 font-mono">{layer.params.toLocaleString()}</td>
                                            <td className="py-2 px-3 text-gray-400 text-xs">{layer.calculation}</td>
                                        </tr>
                                    ))}
                                </tbody>
                                <tfoot>
                                    <tr className="font-bold border-t-2 border-white/20">
                                        <td colSpan={2} className="py-2 px-3 text-right">Total Parameters</td>
                                        <td className="py-2 px-3 font-mono">{modelInfo.totalParams.toLocaleString()}</td>
                                        <td></td>
                                    </tr>
                                </tfoot>
                            </table>
                        </div>
                    </div>
                </details>
            )}


            <div className="flex flex-col gap-8">
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Performance vs. Learning Rate</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        <LineChart data={trainingLog} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis 
                                dataKey="epoch" 
                                stroke="rgba(255, 255, 255, 0.7)" 
                                name="Epoch" 
                                type="number"
                                domain={[1, config.epochs]}
                                allowDataOverflow={true}
                            />
                            <YAxis yAxisId="left" stroke="rgba(255, 255, 255, 0.7)" domain={[0, 0.55]} allowDataOverflow={true} />
                            <YAxis yAxisId="right" orientation="right" stroke="rgba(255, 255, 255, 0.7)" tickFormatter={(val) => val.toExponential(1)} domain={[0, config.learningRate * 1.1]} allowDataOverflow={true} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                            />
                            <Legend />
                            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#f472b6" strokeWidth={2} dot={false} name="Training Loss" isAnimationActive={false}/>
                            <Line yAxisId="left" type="monotone" dataKey="val_loss" stroke="#f97316" strokeWidth={2} dot={false} name="Validation Loss" isAnimationActive={false} connectNulls />
                            <Line yAxisId="right" type="monotone" dataKey="lr" stroke="#8884d8" strokeWidth={2} dot={false} name="Learning Rate" isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Accuracy</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        <LineChart data={trainingLog} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis 
                                dataKey="epoch" 
                                stroke="rgba(255, 255, 255, 0.7)" 
                                type="number"
                                domain={[1, config.epochs]}
                                allowDataOverflow={true}
                            />
                            <YAxis 
                                stroke="rgba(255, 255, 255, 0.7)" 
                                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                                domain={[0.65, 1]}
                                allowDataOverflow={true}
                             />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                                formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={2} dot={false} name="Training Accuracy" isAnimationActive={false} />
                            <Line type="monotone" dataKey="val_accuracy" stroke="#4ade80" strokeWidth={2} dot={false} name="Validation Accuracy" isAnimationActive={false} connectNulls />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};