import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

interface WeightHeatmapProps {
    tensor: tf.Tensor;
    canvasSize: number;
}

export const WeightHeatmap: React.FC<WeightHeatmapProps> = ({ tensor, canvasSize }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const draw = async () => {
            if (!canvasRef.current || !tensor) return;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            // Ensure tensor is 2D
            const readyTensor = tensor.rank === 1 ? tensor.reshape([tensor.shape[0], 1]) : tensor;
            if (readyTensor.rank !== 2) {
                console.error("Heatmap only supports 2D tensors.");
                return;
            }

            const [height, width] = readyTensor.shape;
            canvas.width = width;
            canvas.height = height;

            const data = await readyTensor.data() as Float32Array;
            
            // Find the maximum absolute value for symmetric color scaling
            let absMax = 0;
            for (let i = 0; i < data.length; i++) {
                if (Math.abs(data[i]) > absMax) {
                    absMax = Math.abs(data[i]);
                }
            }
            
            if (absMax === 0) { // Handle all-zero weights
                ctx.fillStyle = 'rgb(20, 20, 20)'; // Dark gray
                ctx.fillRect(0, 0, width, height);
                return;
            }

            const imageData = ctx.createImageData(width, height);
            
            for (let i = 0; i < data.length; i++) {
                const value = data[i];
                const normalized = value / absMax; // Range [-1, 1]

                let r, g, b;
                if (normalized > 0) {
                    // White (0) to Red (1)
                    r = 255;
                    g = 255 * (1 - normalized);
                    b = 255 * (1 - normalized);
                } else {
                    // Blue (-1) to White (0)
                    r = 255 * (1 + normalized);
                    g = 255 * (1 + normalized);
                    b = 255;
                }
                
                const pixelIndex = i * 4;
                imageData.data[pixelIndex] = r;
                imageData.data[pixelIndex + 1] = g;
                imageData.data[pixelIndex + 2] = b;
                imageData.data[pixelIndex + 3] = 255; // Alpha
            }
            ctx.putImageData(imageData, 0, 0);
        };
        
        draw().finally(() => {
            tf.dispose(tensor); // Clean up the tensor passed as a prop
        });

    }, [tensor]); // Redraw when tensor changes
    
    return (
        <canvas 
            ref={canvasRef} 
            className="rounded-sm border border-white/20"
            style={{ 
                width: `${canvasSize}px`, 
                height: `${canvasSize * (tensor.shape[0] / tensor.shape[1]) || canvasSize}px`, // Maintain aspect ratio
                imageRendering: 'pixelated' 
            }} 
        />
    );
};
