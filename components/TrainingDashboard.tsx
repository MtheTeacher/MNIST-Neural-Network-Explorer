
import React from 'react';
import type { TrainingLog } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingDashboardProps {
    trainingLog: TrainingLog[];
    isTraining: boolean;
    status: string;
    epochs: number;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ trainingLog, isTraining, status, epochs }) => {
    const progress = trainingLog.length > 0 ? (trainingLog[trainingLog.length - 1].epoch / epochs) * 100 : 0;
    
    return (
        <div className="bg-white/10 border border-white/20 rounded-2xl p-6 shadow-2xl">
            <h2 className="text-2xl font-bold text-white mb-4">2. Training Progress</h2>

            <div className="mb-6">
                <div className="flex justify-between mb-1">
                    <span className="text-base font-medium text-cyan-300">{status}</span>
                    <span className="text-sm font-medium text-cyan-300">{trainingLog.length} / {epochs} Epochs</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                    <div className="bg-gradient-to-r from-cyan-400 to-pink-500 h-2.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-80">
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Loss (Gradient Descent)</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <LineChart data={trainingLog} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis dataKey="epoch" stroke="rgba(255, 255, 255, 0.7)" name="Epoch" />
                            <YAxis stroke="rgba(255, 255, 255, 0.7)" domain={[0, 'dataMax + 0.2']} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="loss" stroke="#f472b6" strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 6 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="bg-black/20 p-4 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-200 mb-4 text-center">Accuracy</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <LineChart data={trainingLog} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis dataKey="epoch" stroke="rgba(255, 255, 255, 0.7)" />
                            <YAxis stroke="rgba(255, 255, 255, 0.7)" domain={[0, 1]} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                }}
                                formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="accuracy" stroke="#22d3ee" strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 6 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};
