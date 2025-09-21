

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { BrainCircuitIcon } from '../constants';

interface NetworkGraphVisualizerProps {
    model: tf.Sequential;
}

interface LayerInfo {
    name: string;
    units: number;
}

interface NeuronPosition {
    x: number;
    y: number;
}

const MAX_NEURONS_TO_DRAW = 48;
const NEURON_RADIUS = 5;
const LAYER_GAP = 150;

export const NetworkGraphVisualizer: React.FC<NetworkGraphVisualizerProps> = ({ model }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [threshold, setThreshold] = useState(0.4);

    const denseLayers = useMemo(() => 
        // Fix: Cast the filtered layers to the specific `tf.layers.Dense` type for type-safe access to weights.
        model.layers.filter(l => l.getClassName() === 'Dense') as tf.layers.Dense[],
    [model]);
    
    const weights = useMemo(() => 
        denseLayers.slice(1).map(l => l.getWeights()[0]),
    [denseLayers]);

    const draw = useCallback(async () => {
        const canvas = canvasRef.current;
        if (!canvas || weights.length === 0) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        const { width, height } = canvas;
        ctx.clearRect(0, 0, width, height);

        // --- 1. Calculate Layout ---
        const layerLayouts: NeuronPosition[][] = [];
        const horizontalPadding = 50;
        const totalLayersWidth = (denseLayers.length - 1) * LAYER_GAP;
        const startX = (width - totalLayersWidth) / 2;

        denseLayers.forEach((layer, i) => {
            const layerNeurons: NeuronPosition[] = [];
            const config = layer.getConfig() as { units: number };
            const numUnits = config.units;
            const numToDraw = Math.min(numUnits, MAX_NEURONS_TO_DRAW);
            
            const x = startX + i * LAYER_GAP;
            const layerHeight = numToDraw * NEURON_RADIUS * 3;
            const startY = (height - layerHeight) / 2;

            for (let j = 0; j < numToDraw; j++) {
                const y = startY + j * (NEURON_RADIUS * 3);
                layerNeurons.push({ x, y });
            }
            layerLayouts.push(layerNeurons);
        });

        // --- 2. Draw Connections ---
        ctx.lineWidth = 1.5;
        for (let i = 0; i < weights.length; i++) {
            const tensor = weights[i];
            const [inputUnits, outputUnits] = tensor.shape;
            const weightData = await tensor.data() as Float32Array;
            
            const fromLayerLayout = layerLayouts[i];
            const toLayerLayout = layerLayouts[i+1];
            
            let maxAbsWeight = 0;
            for(const w of weightData) {
                if(Math.abs(w) > maxAbsWeight) maxAbsWeight = Math.abs(w);
            }

            for (let fromIdx = 0; fromIdx < fromLayerLayout.length; fromIdx++) {
                for (let toIdx = 0; toIdx < toLayerLayout.length; toIdx++) {
                    
                    const fromUnitOriginalIndex = Math.floor(fromIdx * (inputUnits / fromLayerLayout.length));
                    const toUnitOriginalIndex = Math.floor(toIdx * (outputUnits / toLayerLayout.length));

                    const weight = weightData[fromUnitOriginalIndex * outputUnits + toUnitOriginalIndex];
                    const absWeight = Math.abs(weight);

                    if (absWeight / maxAbsWeight > threshold) {
                        ctx.beginPath();
                        ctx.globalAlpha = Math.max(0.1, (absWeight / maxAbsWeight - threshold) / (1 - threshold));
                        ctx.strokeStyle = weight > 0 ? '#22d3ee' : '#f472b6'; // Cyan for positive, Pink for negative
                        ctx.moveTo(fromLayerLayout[fromIdx].x, fromLayerLayout[fromIdx].y);
                        ctx.lineTo(toLayerLayout[toIdx].x, toLayerLayout[toIdx].y);
                        ctx.stroke();
                    }
                }
            }
        }
        ctx.globalAlpha = 1.0;

        // --- 3. Draw Neurons ---
        layerLayouts.forEach((layer, i) => {
            layer.forEach(pos => {
                ctx.beginPath();
                ctx.fillStyle = '#1f2937'; // gray-800
                ctx.strokeStyle = '#9ca3af'; // gray-400
                ctx.lineWidth = 2;
                ctx.arc(pos.x, pos.y, NEURON_RADIUS, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            });
             // Draw layer label
            if (layer.length > 0) {
                ctx.fillStyle = 'white';
                ctx.textAlign = 'center';
                ctx.font = '12px sans-serif';
                const layerName = denseLayers[i].name;
                const units = (denseLayers[i].getConfig() as {units: number}).units;
                ctx.fillText(`${layerName} (${units})`, layer[0].x, layer[0].y - NEURON_RADIUS * 4);
            }
        });

    }, [weights, threshold, denseLayers]);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const resizeObserver = new ResizeObserver(() => {
            canvas.width = container.clientWidth;
            canvas.height = 500; // Fixed height
            draw();
        });

        resizeObserver.observe(container);
        return () => resizeObserver.disconnect();
    }, [draw]);

    if (denseLayers.length < 2) {
        return (
            <div className="flex flex-col items-center justify-center h-48 text-center bg-black/20 rounded-lg">
                <BrainCircuitIcon className="w-12 h-12 text-gray-500 mb-4" />
                <h4 className="font-semibold text-lg">Not Enough Layers to Visualize</h4>
                <p className="text-gray-400">This visualization requires at least two dense layers to show connections.</p>
            </div>
        );
    }

    return (
        <div ref={containerRef} className="space-y-4">
            <div>
                <p className="text-sm text-gray-300 mb-2">
                    This diagram visualizes the connections (weights) between neurons. Blue lines are positive weights, red are negative. Use the slider to hide weaker connections and reveal the network's core structure.
                </p>
                <label htmlFor="weight-threshold" className="block text-sm font-medium text-gray-300">
                    Connection Strength Threshold: {Math.round(threshold * 100)}%
                </label>
                <input
                    id="weight-threshold"
                    type="range"
                    min="0"
                    max="0.95"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-thumb-cyan mt-1"
                />
            </div>
            <canvas ref={canvasRef} className="w-full rounded-lg bg-black/20" />
        </div>
    );
};
