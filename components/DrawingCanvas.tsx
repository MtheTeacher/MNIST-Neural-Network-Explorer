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
        if ('touches' in e) {
            return {
                x: e.touches[0].clientX - rect.left,
                y: e.touches[0].clientY - rect.top,
            };
        }
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
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

        const imageData = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        const { data, width, height } = imageData;

        let minX = width, minY = height, maxX = -1, maxY = -1;
        let isBlank = true;

        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 0) { // Check red channel for white pixel
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

        // Add padding to bounding box
        const padding = LINE_WIDTH;
        minX = Math.max(0, minX - padding);
        minY = Math.max(0, minY - padding);
        maxX = Math.min(width, maxX + padding);
        maxY = Math.min(height, maxY + padding);

        const digitWidth = maxX - minX;
        const digitHeight = maxY - minY;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return null;
        
        tempCtx.fillStyle = "black";
        tempCtx.fillRect(0,0,28,28);

        const scale = Math.min(20 / digitWidth, 20 / digitHeight);
        const scaledWidth = digitWidth * scale;
        const scaledHeight = digitHeight * scale;
        const dx = (28 - scaledWidth) / 2;
        const dy = (28 - scaledHeight) / 2;

        tempCtx.drawImage(canvas, minX, minY, digitWidth, digitHeight, dx, dy, scaledWidth, scaledHeight);
        
        const tensorImageData = tempCtx.getImageData(0, 0, 28, 28);
        const tensorData = new Float32Array(28 * 28);
        for (let i = 0; i < tensorImageData.data.length / 4; i++) {
            tensorData[i] = tensorImageData.data[i * 4] / 255.0; // Use red channel
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
