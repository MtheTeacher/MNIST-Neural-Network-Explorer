import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { EraserIcon } from '../constants';

const CANVAS_WIDTH = 200;
const CANVAS_HEIGHT = 200;
const LINE_WIDTH = 10;

export interface DrawingCanvasRef {
  getTensor: () => tf.Tensor | null;
  clearCanvas: () => void;
}

interface DrawingCanvasProps {
    onDrawStart?: () => void;
}

export const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(({ onDrawStart }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const lastPos = useRef<{ x: number, y: number } | null>(null);

    const getCanvasContext = () => canvasRef.current?.getContext('2d');

    useEffect(() => {
        const ctx = getCanvasContext();
        if (ctx) {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        }
    }, []);

    const getPosition = (e: React.MouseEvent | React.TouchEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        // Scale coordinates from display size to canvas internal resolution
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    };

    const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
        e.preventDefault();
        onDrawStart?.();
        const pos = getPosition(e);
        if (pos) {
            setIsDrawing(true);
            lastPos.current = pos;
        }
    };
    
    const draw = (e: React.MouseEvent | React.TouchEvent) => {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPosition(e);
        if (pos && lastPos.current) {
            const ctx = getCanvasContext();
            if (ctx) {
                // To create a softer, more pencil-like stroke with grayscale falloff at the edges,
                // we use a combination of a thin line and a blurred shadow of the same color.
                // This simulates a fuzzy, anti-aliased brush.
                ctx.shadowColor = 'white';
                ctx.shadowBlur = 3; // The amount of blur controls the softness of the edge.

                ctx.beginPath();
                ctx.strokeStyle = 'white';
                // The line has a 4px core, and the shadow creates the rest of the apparent width (total ~10px).
                ctx.lineWidth = 4;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.moveTo(lastPos.current.x, lastPos.current.y);
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
                ctx.closePath();

                // It's crucial to reset shadow properties after each stroke.
                // This prevents the shadow from affecting subsequent drawing operations.
                ctx.shadowBlur = 0;

                lastPos.current = pos;
            }
        }
    };

    const endDrawing = () => {
        setIsDrawing(false);
        lastPos.current = null;
    };

    const clearCanvas = () => {
        const ctx = getCanvasContext();
        if (ctx) {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        }
    };
    
    const getTensor = (): tf.Tensor | null => {
        const ctx = getCanvasContext();
        const canvas = canvasRef.current;
        if (!ctx || !canvas) return null;

        const imageData = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        const { data, width, height } = imageData;

        let minX = width, minY = height, maxX = -1, maxY = -1;
        let isBlank = true;

        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 0) { // Check the red channel for non-black pixels.
                isBlank = false;
                const x = (i / 4) % width;
                const y = Math.floor((i / 4) / width);
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }

        if (isBlank) return null;

        const padding = LINE_WIDTH * 1.5;
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(width, maxX + padding);
        maxY = Math.min(height, maxY + padding);

        const digitWidth = maxX - minX;
        const digitHeight = maxY - minY;

        // Create the initial 28x28 canvas for scaling and initial centering.
        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = 28;
        scaledCanvas.height = 28;
        const scaledCtx = scaledCanvas.getContext('2d', { willReadFrequently: true });
        if (!scaledCtx) return null;
        
        scaledCtx.fillStyle = "black";
        scaledCtx.fillRect(0, 0, 28, 28);

        // Calculate scaling to fit the digit into a 20x20 box (bounding box centering).
        const longestSide = Math.max(digitWidth, digitHeight);
        const scale = 20 / longestSide;
        
        const scaledWidth = digitWidth * scale;
        const scaledHeight = digitHeight * scale;
        const dx = (28 - scaledWidth) / 2;
        const dy = (28 - scaledHeight) / 2;
        
        // Apply a 1px Gaussian blur. This makes the drawn digit more closely resemble
        // the fuzzy digits in the original MNIST dataset, but keeps it sharper than before.
        // This filter is applied to the subsequent drawImage operation.
        scaledCtx.filter = 'blur(1px)';

        // Draw the cropped image from the source canvas directly to the scaled canvas.
        // The browser's scaling algorithm provides some anti-aliasing, and the blur
        // filter enhances this effect to match the training data.
        scaledCtx.drawImage(
            canvas,
            minX, minY, digitWidth, digitHeight, // Source rectangle
            dx, dy, scaledWidth, scaledHeight    // Destination rectangle
        );
        
        // It's good practice to reset the filter after use.
        scaledCtx.filter = 'none';
        
        // Now, calculate the center of mass on this scaled and blurred image to re-center it.
        const scaledImageData = scaledCtx.getImageData(0, 0, 28, 28);
        const pixelData = scaledImageData.data;
        
        let totalMass = 0;
        let xMass = 0;
        let yMass = 0;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const index = (y * 28 + x) * 4;
                const mass = pixelData[index]; // Red channel as intensity
                if (mass > 0) {
                    totalMass += mass;
                    xMass += x * mass;
                    yMass += y * mass;
                }
            }
        }

        if (totalMass === 0) {
            // Failsafe in case of a blank image after processing, return bounding-box centered tensor.
             const tensorData = new Float32Array(28 * 28);
             for (let i = 0; i < pixelData.length / 4; i++) {
                 tensorData[i] = pixelData[i * 4] / 255.0;
             }
             return tf.tensor2d(tensorData, [1, 784]);
        }

        const centerX = xMass / totalMass;
        const centerY = yMass / totalMass;
        
        const shiftX = Math.round(14 - centerX); // 14 is the center pixel
        const shiftY = Math.round(14 - centerY);

        // Create the final canvas and draw the shifted image.
        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = 28;
        finalCanvas.height = 28;
        const finalCtx = finalCanvas.getContext('2d', { willReadFrequently: true });
        if (!finalCtx) return null;

        finalCtx.fillStyle = "black";
        finalCtx.fillRect(0, 0, 28, 28);
        finalCtx.drawImage(scaledCanvas, shiftX, shiftY);

        // Normalize brightness to ensure the drawn digit uses the full dynamic range [0, 255].
        // This makes the input brighter and improves contrast, which can help model performance.
        const imageDataForNormalization = finalCtx.getImageData(0, 0, 28, 28);
        const pixelDataForNormalization = imageDataForNormalization.data;
        let maxVal = 0;
        for (let i = 0; i < pixelDataForNormalization.length; i += 4) {
            if (pixelDataForNormalization[i] > maxVal) {
                maxVal = pixelDataForNormalization[i];
            }
        }

        // Avoid division by zero and scale the pixel values.
        if (maxVal > 0) {
            const scaleFactor = 255 / maxVal;
            for (let i = 0; i < pixelDataForNormalization.length; i += 4) {
                const val = pixelDataForNormalization[i] * scaleFactor;
                pixelDataForNormalization[i] = val;
                pixelDataForNormalization[i + 1] = val;
                pixelDataForNormalization[i + 2] = val;
            }
        }
        finalCtx.putImageData(imageDataForNormalization, 0, 0);

        // Create the tensor from the final, center-of-mass adjusted image.
        const finalImageData = finalCtx.getImageData(0, 0, 28, 28);
        const tensorData = new Float32Array(28 * 28);
        
        for (let i = 0; i < finalImageData.data.length / 4; i++) {
            tensorData[i] = finalImageData.data[i * 4] / 255.0;
        }
        
        return tf.tensor2d(tensorData, [1, 784]);
    };

    useImperativeHandle(ref, () => ({
        getTensor,
        clearCanvas,
    }));
    
    return (
        <div className="relative w-full aspect-square">
            <canvas
                ref={canvasRef}
                width={CANVAS_WIDTH}
                height={CANVAS_HEIGHT}
                className="w-full h-full bg-black rounded-xl border-2 border-gray-600 cursor-crosshair touch-none"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={endDrawing}
                onMouseLeave={endDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={endDrawing}
            />
            <button 
              onClick={clearCanvas} 
              className="absolute top-2 right-2 p-1.5 bg-gray-700/50 hover:bg-red-500/50 rounded-full text-white transition-colors"
              aria-label="Clear canvas"
            >
                <EraserIcon className="w-4 h-4" />
            </button>
        </div>
    );
});