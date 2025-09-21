import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { EraserIcon } from '../constants';

const CANVAS_WIDTH = 200;
const CANVAS_HEIGHT = 200;
const LINE_WIDTH = 14;

export interface DrawingCanvasRef {
  getTensor: () => tf.Tensor | null;
  clearCanvas: () => void;
}

export const DrawingCanvas = forwardRef<DrawingCanvasRef>((props, ref) => {
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
                ctx.beginPath();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = LINE_WIDTH;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.moveTo(lastPos.current.x, lastPos.current.y);
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
                ctx.closePath();
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

        // Find the bounding box of the drawn digit to remove whitespace.
        const imageData = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        const { data, width, height } = imageData;

        let minX = width, minY = height, maxX = -1, maxY = -1;
        let isBlank = true;

        // Iterate through pixels to find the bounds of the drawing
        for (let i = 0; i < data.length; i += 4) {
            // Check the alpha channel; a non-zero value means something was drawn.
            if (data[i + 3] > 0) { 
                isBlank = false;
                const x = (i / 4) % width;
                const y = Math.floor((i / 4) / width);
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }

        // If the canvas is blank, return null.
        if (isBlank) return null;

        // Add some padding to the bounding box
        const padding = LINE_WIDTH * 1.5; // Add slightly more padding
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(width, maxX + padding);
        maxY = Math.min(height, maxY + padding);

        const digitWidth = maxX - minX;
        const digitHeight = maxY - minY;

        // Create a temporary 28x28 canvas to scale down the image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
        if (!tempCtx) return null;
        
        // Fill the temporary canvas with black. This will be the background (value 0).
        tempCtx.fillStyle = "black";
        tempCtx.fillRect(0,0,28,28);

        // Calculate the scaling factor and position to center the digit.
        // We want to fit the digit inside a 20x20 box within the 28x28 canvas,
        // which mimics the MNIST data format.
        const longestSide = Math.max(digitWidth, digitHeight);
        const scale = 20 / longestSide;
        
        const scaledWidth = digitWidth * scale;
        const scaledHeight = digitHeight * scale;
        const dx = (28 - scaledWidth) / 2;
        const dy = (28 - scaledHeight) / 2;
        
        // Draw the cropped, scaled-down digit (which is white) onto the black background.
        tempCtx.drawImage(canvas, minX, minY, digitWidth, digitHeight, dx, dy, scaledWidth, scaledHeight);
        
        // Get the pixel data from the 28x28 canvas.
        const tensorImageData = tempCtx.getImageData(0, 0, 28, 28);
        const tensorData = new Float32Array(28 * 28);
        
        // Convert the RGBA data to a grayscale float array.
        // Since we drew white on black, the R, G, and B channels are all the same.
        // We can just use the red channel. This results in digit pixels being ~1.0 and background 0.0.
        for (let i = 0; i < tensorImageData.data.length / 4; i++) {
            tensorData[i] = tensorImageData.data[i * 4] / 255.0;
        }
        
        // Create the final tensor in the shape the model expects [1, 784].
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
