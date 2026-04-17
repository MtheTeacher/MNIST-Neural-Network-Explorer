import React from 'react';

/**
 * RECHARTS IMPLEMENTATION NOTE:
 * We use custom wrappers and sanitization. Do NOT use ResponsiveContainer.
 * Please read `/docs/RECHARTS_GUIDE.md` for in-depth chart rendering docs.
 */

import type { TrainingLog, ModelConfig, ModelInfo, PruningInfo } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { useChartDimensions } from '../hooks/useChartDimensions';
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

const ChartWrapper = ({ children, heightClass = "h-[490px]" }: { children: (width: number, height: number) => React.ReactNode, heightClass?: string }) => {
    const [ref, size] = useChartDimensions();
    return (
        <div ref={ref} className={`w-full ${heightClass} relative`}>
            {size.width > 0 && size.height > 0 ? children(size.width, size.height) : (
                <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
                    Initializing chart...
                </div>
            )}
        </div>
    );
};

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-900/95 border border-white/20 p-3 rounded-lg shadow-xl text-xs z-50">
                <p className="text-gray-300 font-bold mb-2">Epoch {label}</p>
                {payload.map((entry: any, index: number) => (
                    <div key={index} className="flex items-center gap-2 mb-1">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                        <span className="text-gray-400">{entry.name}:</span>
                        <span className="text-white font-mono">
                            {entry.name === 'Learning Rate' 
                                ? entry.value.toExponential(4) 
                                : entry.name.includes('Accuracy') 
                                    ? `${(entry.value * 100).toFixed(2)}%`
                                    : entry.value.toFixed(4)}
                        </span>
                    </div>
                ))}
            </div>
        );
    }
    return null;
};

export const TrainingDashboard: React.FC<TrainingDashboardProps> = (props) => {
    const { isLive, status, onTestModel, onVisualizeModel, onPruneModel, isModelInTest, parentRun } = props;
    
    // Consolidate data references
    const runData = isLive
        ? { log: props.trainingLog ?? [], config: props.config!, modelInfo: props.modelInfo, pruning: undefined }
        : { log: props.run!.log, config: props.run!.config, modelInfo: props.run!.modelInfo, pruning: props.run!.pruning };
    
    const { log: trainingLog, config, modelInfo, pruning } = runData;

    const lastLog = trainingLog.length > 0 ? trainingLog[trainingLog.length - 1] : null;
    const progress = lastLog ? (lastLog.epoch / config.epochs) * 100 : 0;
    
    const trainAccuracy = lastLog ? lastLog.accuracy : null;
    const valAccuracy = lastLog ? (lastLog.val_accuracy ?? null) : null;
    
    let effectiveParams: number | null = null;
    let accuracyDiff: number | null = null;

    if (pruning && modelInfo) {
        effectiveParams = Math.round(modelInfo.totalParams * (1 - pruning.sparsity));
    }

    if (pruning && parentRun && valAccuracy && parentRun.log.length > 0) {
        const parentLastLog = parentRun.log[parentRun.log.length - 1];
        if (parentLastLog) {
            const parentAccuracy = parentLastLog.val_accuracy ?? parentLastLog.accuracy;
            accuracyDiff = valAccuracy - parentAccuracy;
        }
    }

    const title = pruning
        ? `Pruned Model (${(pruning.sparsity * 100).toFixed(0)}% sparse)`
        : (modelInfo?.name || (isLive ? "Training Progress" : "Training Run"));

    return (
        <div className={`bg-white/10 border rounded-2xl p-6 shadow-2xl transition-all duration-300 mb-6 ${isLive ? 'border-cyan-400/50 ring-1 ring-cyan-400/20' : 'border-white/20'}`}>
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4 mb-4">
                <div>
                    <h2 className="text-xl font-bold text-white">{title}</h2>
                    <div className="text-xs text-gray-300 flex flex-wrap gap-x-3 gap-y-1 mt-1">
                        <span>Arch: <span className="font-semibold text-gray-200">{config.architecture.toUpperCase()}</span></span>
                        <span>Epochs: <span className="font-semibold text-gray-200">{config.epochs}</span></span>
                        {modelInfo && <span>Params: <span className="font-semibold text-gray-200">{modelInfo.totalParams.toLocaleString()}</span></span>}
                    </div>
                </div>
                {!isLive && (
                     <div className="flex flex-col sm:flex-row gap-2">
                        {onTestModel && (
                            <button onClick={onTestModel} className={`font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition transform hover:scale-105 text-sm ${isModelInTest ? 'bg-pink-500 text-white' : 'bg-white/20 text-white'}`}>
                                <RocketIcon className="w-4 h-4"/><span>{isModelInTest ? 'Testing' : 'Test'}</span>
                            </button>
                        )}
                        {onVisualizeModel && (
                             <button onClick={onVisualizeModel} className="bg-white/20 hover:bg-white/30 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 text-sm transition transform hover:scale-105">
                                <EyeIcon className="w-4 h-4"/><span>Visualize</span>
                            </button>
                        )}
                        {onPruneModel && config.architecture === 'dense' && !pruning && (
                             <button onClick={() => onPruneModel(props.run as FullTrainingRun)} className="bg-white/20 hover:bg-white/30 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 text-sm transition transform hover:scale-105">
                                <CutIcon className="w-4 h-4"/><span>Prune</span>
                            </button>
                        )}
                    </div>
                )}
            </div>

             {isLive && (
                <div className="mb-6 bg-black/30 p-4 rounded-xl border border-white/5">
                    <div className="flex justify-between mb-2">
                        <span className="text-sm font-semibold text-cyan-300 uppercase tracking-wider">{status}</span>
                        <span className="text-sm font-mono text-cyan-300">{trainingLog.length} / {config.epochs} EPOCHS</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                        <div className="bg-gradient-to-r from-cyan-400 to-pink-500 h-2 rounded-full transition-all duration-700 ease-out" style={{ width: `${progress}%` }}></div>
                    </div>
                </div>
            )}
            
            <div className="mb-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                 <div className="bg-black/20 p-3 rounded-lg border border-white/5 flex flex-col items-center justify-center">
                    <span className="text-xs text-gray-400 uppercase font-bold tracking-tight">Train Accuracy</span>
                    <span className="text-2xl font-bold text-cyan-400">{trainAccuracy !== null ? `${(trainAccuracy * 100).toFixed(2)}%` : '--'}</span>
                </div>
                 <div className="bg-black/20 p-3 rounded-lg border border-white/5 flex flex-col items-center justify-center">
                    <span className="text-xs text-gray-400 uppercase font-bold tracking-tight">Validation Accuracy</span>
                    <span className="text-2xl font-bold text-emerald-400">{valAccuracy !== null ? `${(valAccuracy * 100).toFixed(2)}%` : '--'}</span>
                </div>
            </div>

            <div className="flex flex-col gap-8 min-w-0">
                <div className="bg-black/30 p-6 rounded-2xl border border-white/5 shadow-inner">
                    <h3 className="text-base font-bold text-white mb-6 text-center">Performance vs. Learning Rate</h3>
                    <ChartWrapper>
                        {(width, height) => (
                            <LineChart width={width} height={height} data={trainingLog} margin={{ top: 10, right: 50, left: 10, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={true} horizontal={false} />
                                <XAxis 
                                    dataKey="epoch" 
                                    stroke="#888" 
                                    tick={{ fill: '#888', fontSize: 12 }}
                                    axisLine={{ stroke: '#444' }}
                                    tickLine={false}
                                    domain={[1, config.epochs]}
                                    type="number"
                                />
                                <YAxis 
                                    yAxisId="loss"
                                    stroke="#888" 
                                    tick={{ fill: '#888', fontSize: 12 }}
                                    axisLine={{ stroke: '#444' }}
                                    tickLine={false}
                                    domain={[0, 'auto']}
                                    tickFormatter={(val) => val.toFixed(2)}
                                />
                                <YAxis 
                                    yAxisId="lr"
                                    orientation="right"
                                    stroke="#888" 
                                    tick={{ fill: '#888', fontSize: 12 }}
                                    axisLine={{ stroke: '#444' }}
                                    tickLine={false}
                                    domain={[0, 'auto']}
                                    tickFormatter={(val) => val.toExponential(1)}
                                />
                                <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2 }} />
                                <Legend 
                                    verticalAlign="bottom" 
                                    height={36} 
                                    iconType="circle"
                                    wrapperStyle={{ fontSize: '12px', color: '#888', paddingTop: '20px' }}
                                />
                                <Line yAxisId="lr" type="monotone" dataKey="lr" name="Learning Rate" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <Line yAxisId="loss" type="monotone" dataKey="loss" name="Training Loss" stroke="#ec4899" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="Validation Loss" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                            </LineChart>
                        )}
                    </ChartWrapper>
                </div>

                <div className="bg-black/30 p-6 rounded-2xl border border-white/5 shadow-inner">
                    <h3 className="text-base font-bold text-white mb-6 text-center">Accuracy History</h3>
                    <ChartWrapper>
                        {(width, height) => (
                            <LineChart width={width} height={height} data={trainingLog} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={true} horizontal={false} />
                                <XAxis 
                                    dataKey="epoch" 
                                    stroke="#888" 
                                    tick={{ fill: '#888', fontSize: 12 }}
                                    axisLine={{ stroke: '#444' }}
                                    tickLine={false}
                                    domain={[1, config.epochs]}
                                    type="number"
                                />
                                <YAxis 
                                    stroke="#888" 
                                    tick={{ fill: '#888', fontSize: 12 }}
                                    axisLine={{ stroke: '#444' }}
                                    tickLine={false}
                                    domain={[
                                        (dataMin: number) => {
                                            const min = isFinite(dataMin) ? dataMin : 0;
                                            return Math.max(0, Math.floor(min * 20) / 20);
                                        }, 
                                        1
                                    ]}
                                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                                />
                                <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2 }} />
                                <Legend 
                                    verticalAlign="bottom" 
                                    height={36} 
                                    iconType="circle"
                                    wrapperStyle={{ fontSize: '12px', color: '#888', paddingTop: '20px' }}
                                />
                                <Line type="monotone" dataKey="accuracy" name="Training Accuracy" stroke="#06b6d4" strokeWidth={2} dot={false} isAnimationActive={false} />
                                <Line type="monotone" dataKey="val_accuracy" name="Validation Accuracy" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                            </LineChart>
                        )}
                    </ChartWrapper>
                </div>
            </div>
        </div>
    );
};
