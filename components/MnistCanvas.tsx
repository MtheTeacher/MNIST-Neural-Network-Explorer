import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

// A small component to render a 28x28 MNIST image tensor to a canvas
export const MnistCanvas: React.FC<{ tensor: tf.Tensor, size: number }> = ({ tensor, size }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const draw = () => {
            if (!canvasRef.current || !tensor || tensor.isDisposed) return;
            const canvas = canvasRef.current;
            tf.tidy(() => {
                const imageTensor = tensor.reshape([28, 28, 1]);
                // Cast imageTensor to Tensor3D, as tf.browser.toPixels requires a more specific tensor type than the inferred one.
                tf.browser.toPixels(imageTensor as tf.Tensor3D, canvas);
            });
        };
        draw();
    }, [tensor]);

    return (
        <canvas
            ref={canvasRef}
            width={28}
            height={28}
            style={{ width: `${size}px`, height: `${size}px`, imageRendering: 'pixelated' }}
            className="bg-black border border-gray-600 rounded-md block"
        />
    );
};
