import React, { useState, useCallback, useRef, createRef } from 'react';
// Fix: `tf` is used as a value for `tf.stack` and `tf.dispose`, so it needs a regular import, not a type-only import.
import * as tf from '@tensorflow/tfjs';
import { DrawingCanvas, type DrawingCanvasRef } from './DrawingCanvas';
import { SparklesIcon, EraserIcon } from '../constants';

interface InferenceTesterProps {
    model: tf.Sequential;
}

export const InferenceTester: React.FC<InferenceTesterProps> = ({ model }) => {
    const [predictions, setPredictions] = useState<(number | null)[]>(Array(10).fill(null));
    const [isProcessing, setIsProcessing] = useState(false);
    const canvasRefs = useRef<React.RefObject<DrawingCanvasRef>[]>(
        Array(10).fill(null).map(() => createRef<DrawingCanvasRef>())
    );

    const handlePredictAll = useCallback(async () => {
        setIsProcessing(true);
        const tensorsAndIndices: { tensor: tf.Tensor; index: number }[] = [];

        canvasRefs.current.forEach((ref, index) => {
            const tensor = ref.current?.getTensor();
            if (tensor) {
                tensorsAndIndices.push({ tensor, index });
            }
        });

        if (tensorsAndIndices.length > 0) {
            const batch = tf.stack(tensorsAndIndices.map(item => item.tensor.reshape([1, 28, 28, 1])));
            const predictionsTensor = model.predict(batch) as tf.Tensor;
            const predictionsArray = await predictionsTensor.argMax(-1).data();

            const newPredictions = [...predictions];
            predictionsArray.forEach((pred, i) => {
                const originalIndex = tensorsAndIndices[i].index;
                newPredictions[originalIndex] = pred as number;
            });
            setPredictions(newPredictions);
            
            tf.dispose([batch, predictionsTensor, ...tensorsAndIndices.map(item => item.tensor)]);
        }
        
        setIsProcessing(false);
    }, [model, predictions]);

    const handleClearAll = () => {
        canvasRefs.current.forEach(ref => ref.current?.clearCanvas());
        setPredictions(Array(10).fill(null));
    };

    return (
        <div>
            <div className="flex flex-col sm:flex-row gap-4 mb-6">
                 <button
                    onClick={handlePredictAll}
                    disabled={isProcessing}
                    className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-3 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-wait"
                >
                    <SparklesIcon className="w-5 h-5"/>
                    <span>{isProcessing ? 'Predicting...' : 'Predict All Digits'}</span>
                </button>
                 <button
                    onClick={handleClearAll}
                    disabled={isProcessing}
                    className="flex-1 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-500 hover:to-gray-600 text-white font-bold py-3 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                    <EraserIcon className="w-5 h-5"/>
                    <span>Clear All</span>
                </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                {Array.from({ length: 10 }).map((_, i) => (
                    <div key={i} className="flex flex-col items-center text-center space-y-2">
                        <p className="font-bold text-lg text-gray-300">Draw a "{i}"</p>
                        <DrawingCanvas ref={canvasRefs.current[i]} />
                        <div className="h-12 flex items-center justify-center">
                            {predictions[i] !== null && (
                                 <p className={`text-4xl font-bold bg-clip-text text-transparent ${predictions[i] === i ? 'bg-gradient-to-r from-green-400 to-emerald-500' : 'bg-gradient-to-r from-red-500 to-pink-500'}`}>
                                     {predictions[i]}
                                 </p>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};