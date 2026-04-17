import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import type { LRSchedulerId } from '../types';
import { LEARNING_RATE_SCHEDULES } from '../constants';
import { generateScheduleData } from '../services/lrSchedulerService';
import { useChartDimensions } from '../hooks/useChartDimensions';

interface LRScheduleVisualizerProps {
    schedule: LRSchedulerId;
    epochs: number;
    initialLr: number;
}

const ChartWrapper = ({ children }: { children: (width: number, height: number) => React.ReactNode }) => {
    const [ref, size] = useChartDimensions();
    return (
        <div ref={ref} className="w-full h-[150px] relative">
            {size.width > 0 && size.height > 0 ? children(size.width, size.height) : null}
        </div>
    );
};

export const LRScheduleVisualizer: React.FC<LRScheduleVisualizerProps> = ({ schedule, epochs, initialLr }) => {
    const data = generateScheduleData(schedule, epochs, initialLr);
    const scheduleInfo = LEARNING_RATE_SCHEDULES.find(s => s.id === schedule);

    return (
        <div className="bg-black/20 p-4 rounded-xl border border-white/5">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Learning Rate Schedule</h3>
            <p className="text-sm font-semibold text-gray-300 mb-4">{scheduleInfo?.description}</p>
            <ChartWrapper>
                {(width, height) => (
                    <LineChart width={width} height={height} data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={true} horizontal={false} />
                        <XAxis dataKey="epoch" stroke="#888" fontSize={10} tickLine={false} axisLine={{ stroke: '#444' }} />
                        <YAxis 
                            stroke="#888" 
                            fontSize={10} 
                            tickLine={false} 
                            axisLine={{ stroke: '#444' }}
                            tickFormatter={(val) => val.toExponential(1)}
                            domain={[0, 'auto']}
                        />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                            formatter={(value: number) => value.toExponential(4)}
                            labelStyle={{ color: '#94a3b8' }}
                            cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2 }}
                        />
                        <Line 
                            type="monotone" 
                            dataKey="lr" 
                            stroke="#8b5cf6" 
                            strokeWidth={2} 
                            dot={false}
                            isAnimationActive={false}
                        />
                    </LineChart>
                )}
            </ChartWrapper>
        </div>
    );
};
