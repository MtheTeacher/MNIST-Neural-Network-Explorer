import React from 'react';
import type { TrainingLog, ModelConfig } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LEARNING_RATE_SCHEDULES } from '../constants';
import { RocketIcon } from '../constants';

interface TrainingDashboardProps {
    trainingLog: TrainingLog[];
    isLive: boolean;
    status: string;
    config: ModelConfig;
    onTestModel?: () => void;
    isModelInTest?: boolean;
}

const getScheduleName = (id: string) => LEARNING_RATE_SCHEDULES.find(s => s.id === id)?.name || 'Unknown';

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ trainingLog, isLive, status, config, onTestModel, isModelInTest }) => {
    const progress = trainingLog.length > 0 ? (trainingLog[trainingLog.length - 1].epoch / config.epochs) * 100 : (isLive ? 0 : 100);
    const finalAccuracy = !isLive && trainingLog.length > 0 ? trainingLog[trainingLog.length - 1].accuracy : null;

    return (
        <div className={`bg-white/10 border rounded-2xl p-6 shadow-2xl transition-all duration-300 ${isLive ? 'border-cyan-400/50' : 'border-white/20'}`}>
            <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2 mb-4">
                <div>
                    <h2 className="text-xl font-bold text-white">{isLive ? "Live Training" : "Completed Run"}</h2>
                    <div className="text-xs text-gray-300 flex flex-wrap gap-x-3 gap-y-1 mt-1">
                        <span>Arch: <span className="font-semibold text-gray-200">{config.architecture.toUpperCase()}</span></span>
                        <span>LR Schedule: <span className="font-semibold text-gray-200">{getScheduleName(config.lrSchedule)}</span></span>
                        <span>Initial LR: <span className="font-semibold text-gray-200">{config.learningRate}</span></span>
                        <span>Epochs: <span className="font-semibold text-gray-200">{config.epochs}</span></span>
                    </div>
                </div>
                {!isLive && onTestModel && (
                     <button
                        onClick={onTestModel}
                        className={`font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 text-sm ${isModelInTest ? 'bg-pink-500 text-white' : 'bg-white/20 hover:bg-white/30 text-white'}`}
                    >
                        <RocketIcon className="w-4 h-4"/>
                        <span>{isModelInTest ? 'Currently Testing' : 'Test This Model'}</span>
                    </button>
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
            
            {!isLive && finalAccuracy && (
                <div className="mb-4 text-center bg-black/20 p-2 rounded-lg">
                    <span className="text-lg font-semibold text-gray-200">Final Accuracy: </span>
                    <span className="text-lg font-bold text-cyan-300">{(finalAccuracy * 100).toFixed(2)}%</span>
                </div>
            )}


            <div className="flex flex-col gap-8">
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Loss & Learning Rate</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        {/* Fix: isAnimationActive is not a prop of LineChart. Moved to Line components. */}
                        <LineChart data={trainingLog} margin={{ top: 5, right: 5, left: -15, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis 
                                dataKey="epoch" 
                                stroke="rgba(255, 255, 255, 0.7)" 
                                name="Epoch" 
                                type="number"
                                domain={[1, config.epochs]}
                                allowDataOverflow={true}
                            />
                            <YAxis yAxisId="left" stroke="#f472b6" domain={['auto', 'auto']} />
                            <YAxis yAxisId="right" orientation="right" stroke="#8884d8" tickFormatter={(val) => val.toExponential(1)} domain={['auto', 'auto']} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                            />
                            <Legend />
                            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#f472b6" strokeWidth={4} dot={false} activeDot={{ r: 6 }} name="Loss" isAnimationActive={false}/>
                            <Line yAxisId="right" type="monotone" dataKey="lr" stroke="#8884d8" strokeWidth={4} dot={false} activeDot={{ r: 6 }} name="LR" isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Accuracy</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        {/* Fix: isAnimationActive is not a prop of LineChart. Moved to Line component. */}
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
                                domain={['auto', 'auto']}
                             />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                                formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={4} dot={false} activeDot={{ r: 6 }} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};