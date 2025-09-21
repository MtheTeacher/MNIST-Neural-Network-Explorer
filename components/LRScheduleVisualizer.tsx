import React from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import type { LRSchedulerId } from '../types';
import { LEARNING_RATE_SCHEDULES } from '../constants';
import { generateScheduleData } from '../services/lrSchedulerService';

interface LRScheduleVisualizerProps {
    schedule: LRSchedulerId;
    epochs: number;
    initialLr: number;
}

export const LRScheduleVisualizer: React.FC<LRScheduleVisualizerProps> = ({ schedule, epochs, initialLr }) => {
    const data = generateScheduleData(schedule, epochs, initialLr);
    const scheduleInfo = LEARNING_RATE_SCHEDULES.find(s => s.id === schedule);

    return (
        <div className="bg-black/20 p-3 rounded-lg border border-white/20">
            <p className="text-sm font-semibold text-gray-300 mb-2">{scheduleInfo?.description}</p>
            <div style={{ width: '100%', height: 100 }}>
                <ResponsiveContainer>
                    <LineChart data={data} margin={{ top: 5, right: 10, left: -35, bottom: -10 }}>
                        <XAxis dataKey="epoch" stroke="rgba(255, 255, 255, 0.5)" fontSize={10} tick={false} />
                        <YAxis stroke="rgba(255, 255, 255, 0.5)" fontSize={10} domain={['dataMin', 'dataMax']} tickFormatter={(val) => val.toExponential(0)} />
                        <Tooltip 
                             contentStyle={{
                                backgroundColor: 'rgba(30, 41, 59, 0.8)',
                                borderColor: 'rgba(255, 255, 255, 0.3)',
                                fontSize: '12px',
                                padding: '4px 8px',
                            }}
                            labelFormatter={(label) => `Epoch: ${label}`}
                            formatter={(value:number) => [value.toExponential(3), 'LR']}
                        />
                        <Line type="monotone" dataKey="lr" stroke="#22d3ee" strokeWidth={2} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
