
import React, { useEffect, useRef } from 'react';
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
    
    const chart1Ref = useRef<HTMLDivElement>(null);
    const chart2Ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (isLive) {
            const logDimensions = (name: string, ref: React.RefObject<HTMLDivElement>) => {
                if (ref.current) {
                    const { width, height } = ref.current.getBoundingClientRect();
                    console.log(`[Chart Instrumentation] ${name} dimensions:`, { width, height });
                    if (width === 0 || height === 0) {
                        console.warn(`[Chart Warning] ${name} has zero dimension! Recharts may fail to render.`);
                    }
                }
            };
            logDimensions('Performance Chart', chart1Ref);
            logDimensions('Accuracy Chart', chart2Ref);
        }
    }, [isLive, props.trainingLog?.length]);

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
                <div className="bg-black/30 p-4 rounded-xl border border-white/5 min-w-0">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 text-center">Performance / LR</h3>
                    <div ref={chart1Ref} className="h-[350px] w-full relative min-w-0" style={{ height: 350, width: '100%', minWidth: 0 }}>
                        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                            <LineChart data={trainingLog} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                                <XAxis 
                                    dataKey="epoch" 
                                    stroke="rgba(255, 255, 255, 0.4)" 
                                    fontSize={10}
                                    type="number"
                                    domain={[1, config.epochs]}
                                    allowDataOverflow={true}
                                />
                                <YAxis yAxisId="left" stroke="#f472b6" fontSize={10} domain={[0, 'auto']} />
                                <YAxis yAxisId="right" orientation="right" stroke="#8884d8" fontSize={10} tickFormatter={(val) => val?.toExponential(1) ?? '0'} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                        borderColor: 'rgba(255, 255, 255, 0.2)',
                                        borderRadius: '8px',
                                        fontSize: '11px',
                                    }}
                                />
                                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                                <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#f472b6" strokeWidth={3} dot={false} name="Train Loss" isAnimationActive={false}/>
                                <Line yAxisId="left" type="monotone" dataKey="val_loss" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Val Loss" isAnimationActive={false} connectNulls />
                                <Line yAxisId="right" type="stepAfter" dataKey="lr" stroke="#8884d8" strokeWidth={1.5} dot={false} name="LR" isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-black/30 p-4 rounded-xl border border-white/5 min-w-0">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4 text-center">Accuracy History</h3>
                    <div ref={chart2Ref} className="h-[350px] w-full relative min-w-0" style={{ height: 350, width: '100%', minWidth: 0 }}>
                        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                            <LineChart data={trainingLog} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                                <XAxis 
                                    dataKey="epoch" 
                                    stroke="rgba(255, 255, 255, 0.4)" 
                                    fontSize={10}
                                    type="number"
                                    domain={[1, config.epochs]}
                                    allowDataOverflow={true}
                                />
                                <YAxis 
                                    stroke="rgba(255, 255, 255, 0.6)" 
                                    fontSize={10}
                                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                                    domain={[0, 1]}
                                 />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                        borderColor: 'rgba(255, 255, 255, 0.2)',
                                        borderRadius: '8px',
                                        fontSize: '11px'
                                    }}
                                    formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                                />
                                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                                <Line type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={3} dot={false} name="Train Acc" isAnimationActive={false} />
                                <Line type="monotone" dataKey="val_accuracy" stroke="#4ade80" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Val Acc" isAnimationActive={false} connectNulls />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
};
