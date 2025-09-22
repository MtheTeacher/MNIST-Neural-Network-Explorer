import React, { useState, useCallback, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { DrawingCanvas, type DrawingCanvasRef } from './DrawingCanvas';
import { SparklesIcon, EraserIcon, PencilIcon, ImageIcon, CheckCircleIcon, XCircleIcon, XIcon } from '../constants';
import { MnistData } from '../services/mnistData';
import type { MnistSample } from '../types';
import { MnistCanvas } from './MnistCanvas';

interface InferenceTesterProps {
    model: tf.Sequential;
}

type DigitSlot = { thumbnail: string; tensor: tf.Tensor } | null;

export const InferenceTester: React.FC<InferenceTesterProps> = ({ model }) => {
    // Shared state
    const [isProcessing, setIsProcessing] = useState(false);
    const [testMode, setTestMode] = useState<'draw' | 'mnist'>('draw');

    // State for 'draw' mode
    const mainCanvasRef = useRef<DrawingCanvasRef>(null);
    const [digitSlots, setDigitSlots] = useState<DigitSlot[]>(Array(10).fill(null));
    const [drawPredictions, setDrawPredictions] = useState<(number | null)[]>(Array(10).fill(null));

    // State for 'mnist' mode
    const [mnistSamples, setMnistSamples] = useState<MnistSample[]>([]);
    const [isLoadingMnist, setIsLoadingMnist] = useState(false);
    const [droppedImages, setDroppedImages] = useState<(MnistSample | null)[]>(Array(10).fill(null));
    const [mnistPredictions, setMnistPredictions] = useState<(number | null)[]>(Array(10).fill(null));
    
    // Load MNIST test data when switching to that mode
    useEffect(() => {
        if (testMode === 'mnist' && mnistSamples.length === 0) {
            const loadMnistData = async () => {
                setIsLoadingMnist(true);
                try {
                    const data = await MnistData.getInstance();
                    const samples = data.getTestSamplesForInference(100);
                    setMnistSamples(samples);
                } catch (error) {
                    console.error("Failed to load MNIST data for testing:", error);
                } finally {
                    setIsLoadingMnist(false);
                }
            };
            loadMnistData();
        }
    }, [testMode, mnistSamples.length]);

    // Cleanup tensors from draw slots on unmount
    useEffect(() => {
        // This cleanup function should only run when the component unmounts.
        // By providing an empty dependency array, we prevent it from running
        // on every state change, which was causing tensors to be disposed prematurely.
        return () => {
            digitSlots.forEach(slot => {
                if (slot && slot.tensor && !slot.tensor.isDisposed) {
                  slot.tensor.dispose();
                }
            });
        };
    }, []);

    const handleAssignDigit = (digitIndex: number) => {
        if (!mainCanvasRef.current) return;

        const tensor = mainCanvasRef.current.getTensor();
        const thumbnail = mainCanvasRef.current.getDataURL();

        if (tensor && thumbnail) {
            // Dispose the old tensor for this slot if it exists
            digitSlots[digitIndex]?.tensor.dispose();

            const newSlots = [...digitSlots];
            newSlots[digitIndex] = { thumbnail, tensor };
            setDigitSlots(newSlots);

            const newPredictions = [...drawPredictions];
            newPredictions[digitIndex] = null;
            setDrawPredictions(newPredictions);

            mainCanvasRef.current.clearCanvas();
        } else {
            // Maybe add a small toast/notification later
        }
    };

    const handleClearSlot = (digitIndex: number) => {
        digitSlots[digitIndex]?.tensor.dispose();
        
        const newSlots = [...digitSlots];
        newSlots[digitIndex] = null;
        setDigitSlots(newSlots);

        const newPredictions = [...drawPredictions];
        newPredictions[digitIndex] = null;
        setDrawPredictions(newPredictions);
    };

    const handlePredictAll = useCallback(async () => {
        setIsProcessing(true);
        if (testMode === 'draw') {
            const tensorsAndIndices: { tensor: tf.Tensor; index: number }[] = [];
            digitSlots.forEach((slot, index) => {
                if (slot) tensorsAndIndices.push({ tensor: slot.tensor, index });
            });

            if (tensorsAndIndices.length > 0) {
                const tensorsToBatch = tensorsAndIndices.map(item => item.tensor);
                const batch = tf.concat(tensorsToBatch, 0);
                const predictionsTensor = model.predict(batch) as tf.Tensor;
                const predictionsArray = await predictionsTensor.argMax(-1).data();
                
                const newPredictions = [...drawPredictions];
                predictionsArray.forEach((pred, i) => {
                    newPredictions[tensorsAndIndices[i].index] = pred as number;
                });
                setDrawPredictions(newPredictions);
                tf.dispose([batch, predictionsTensor]);
            }
        } else { // 'mnist' mode
            const tensorsAndIndices: { tensor: tf.Tensor; index: number }[] = [];
            droppedImages.forEach((sample, index) => {
                if (sample) tensorsAndIndices.push({ tensor: sample.tensor, index });
            });

            if (tensorsAndIndices.length > 0) {
                const tensorsToBatch = tensorsAndIndices.map(item => item.tensor);
                const batch = tf.concat(tensorsToBatch, 0);
                const predictionsTensor = model.predict(batch) as tf.Tensor;
                const predictionsArray = await predictionsTensor.argMax(-1).data();

                const newPredictions = [...mnistPredictions];
                predictionsArray.forEach((pred, i) => {
                    newPredictions[tensorsAndIndices[i].index] = pred as number;
                });
                setMnistPredictions(newPredictions);
                tf.dispose([batch, predictionsTensor]);
            }
        }
        setIsProcessing(false);
    }, [model, testMode, digitSlots, drawPredictions, droppedImages, mnistPredictions]);

    const handleClearAll = () => {
        if (testMode === 'draw') {
            mainCanvasRef.current?.clearCanvas();
            digitSlots.forEach(slot => {
                if (slot) slot.tensor.dispose();
            });
            setDigitSlots(Array(10).fill(null));
            setDrawPredictions(Array(10).fill(null));
        } else {
            setDroppedImages(Array(10).fill(null));
            setMnistPredictions(Array(10).fill(null));
        }
    };
    
    // --- Drag and Drop Handlers for MNIST mode ---
    const handleDragStart = (e: React.DragEvent, sampleId: number) => {
        e.dataTransfer.setData('application/mnist-sample-id', sampleId.toString());
    };
    const handleDragOver = (e: React.DragEvent) => e.preventDefault();
    const handleDrop = (e: React.DragEvent, slotIndex: number) => {
        e.preventDefault();
        const sampleId = parseInt(e.dataTransfer.getData('application/mnist-sample-id'), 10);
        const sample = mnistSamples.find(s => s.id === sampleId);
        if (sample) {
            const newDroppedImages = [...droppedImages];
            newDroppedImages[slotIndex] = sample;
            setDroppedImages(newDroppedImages);

            const newPredictions = [...mnistPredictions];
            newPredictions[slotIndex] = null;
            setMnistPredictions(newPredictions);
        }
    };

    return (
        <div>
            <div className="flex justify-center mb-6 bg-black/20 p-1 rounded-full w-full max-w-sm mx-auto">
                <button 
                    onClick={() => setTestMode('draw')} 
                    className={`flex-1 flex items-center justify-center space-x-2 text-sm font-semibold p-2 rounded-full transition-colors ${testMode === 'draw' ? 'bg-cyan-500 text-white shadow' : 'text-gray-300 hover:bg-white/10'}`}
                >
                    <PencilIcon className="w-4 h-4" />
                    <span>Draw Digits</span>
                </button>
                <button 
                    onClick={() => setTestMode('mnist')}
                    className={`flex-1 flex items-center justify-center space-x-2 text-sm font-semibold p-2 rounded-full transition-colors ${testMode === 'mnist' ? 'bg-cyan-500 text-white shadow' : 'text-gray-300 hover:bg-white/10'}`}
                >
                    <ImageIcon className="w-4 h-4" />
                    <span>Test on MNIST Images</span>
                </button>
            </div>
            
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

            {testMode === 'draw' && (
                <div className="flex flex-col lg:flex-row gap-8 items-start">
                    <div className="w-full lg:max-w-xs space-y-4">
                        <h3 className="text-lg font-semibold text-center text-white">1. Draw a Digit</h3>
                        <DrawingCanvas ref={mainCanvasRef} />
                        <h3 className="text-lg font-semibold text-center text-white">2. Assign to a Slot</h3>
                        <div className="grid grid-cols-5 gap-2">
                            {Array.from({ length: 10 }).map((_, i) => (
                                <button
                                    key={i}
                                    onClick={() => handleAssignDigit(i)}
                                    className="bg-white/10 text-white hover:bg-cyan-500/50 font-semibold py-2 px-2 rounded-lg transition duration-300 text-sm"
                                >
                                    Slot {i}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="flex-1 w-full">
                        <h3 className="text-lg font-semibold text-center text-white mb-4">3. Prediction Slots</h3>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                            {digitSlots.map((slot, i) => {
                                const prediction = drawPredictions[i];
                                const isCorrect = slot && prediction !== null && i === prediction;
                                return (
                                    <div key={i} className="flex flex-col items-center text-center space-y-2">
                                        <div className="relative w-full aspect-square bg-black/30 border-2 border-dashed border-gray-600 rounded-xl flex items-center justify-center">
                                            {slot ? (
                                                <>
                                                    <img src={slot.thumbnail} alt={`Drawing of ${i}`} className="w-full h-full rounded-lg" />
                                                    <button onClick={() => handleClearSlot(i)} className="absolute top-1 right-1 p-1 bg-gray-900/50 hover:bg-red-500/80 rounded-full text-white transition-colors" aria-label={`Clear slot ${i}`}>
                                                        <XIcon className="w-3 h-3" />
                                                    </button>
                                                </>
                                            ) : (
                                                <span className="text-4xl font-bold text-gray-500">{i}</span>
                                            )}
                                        </div>
                                        <div className="h-12 flex flex-col items-center justify-center">
                                            {prediction !== null && slot && (
                                                <div className="flex items-center space-x-2">
                                                    <p className={`text-3xl font-bold ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>{prediction}</p>
                                                    <div className="flex flex-col text-xs text-left">
                                                        <span className="text-gray-400">(Is {i})</span>
                                                        {isCorrect 
                                                            ? <CheckCircleIcon className="w-5 h-5 text-green-400" /> 
                                                            : <XCircleIcon className="w-5 h-5 text-red-400" />
                                                        }
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>
            )}
            
            {testMode === 'mnist' && (
                <div className="space-y-8">
                    <div>
                        <p className="text-center text-gray-300 mb-4">Drag images from the gallery below and drop them into these slots.</p>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                            {Array.from({ length: 10 }).map((_, i) => {
                                const dropped = droppedImages[i];
                                const prediction = mnistPredictions[i];
                                const isCorrect = dropped && prediction !== null && dropped.label === prediction;

                                return (
                                    <div key={i} className="flex flex-col items-center text-center space-y-2">
                                        <div 
                                            onDragOver={handleDragOver} 
                                            onDrop={(e) => handleDrop(e, i)}
                                            className="w-full aspect-square bg-black/30 border-2 border-dashed border-gray-600 rounded-xl flex items-center justify-center"
                                        >
                                            {dropped && <MnistCanvas tensor={dropped.tensor} size={80} />}
                                        </div>
                                        <div className="h-12 flex flex-col items-center justify-center">
                                            {prediction !== null && dropped && (
                                                <div className="flex items-center space-x-2">
                                                    <p className={`text-3xl font-bold ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>{prediction}</p>
                                                    <div className="flex flex-col text-xs text-left">
                                                        <span className="text-gray-400">(Actual: {dropped.label})</span>
                                                        {isCorrect 
                                                            ? <CheckCircleIcon className="w-5 h-5 text-green-400" /> 
                                                            : <XCircleIcon className="w-5 h-5 text-red-400" />
                                                        }
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold text-center mb-4">MNIST Test Image Gallery</h3>
                        {isLoadingMnist ? <p className="text-center text-gray-400">Loading test images...</p> : (
                             <div className="bg-black/30 p-4 rounded-xl border border-white/20 h-72 overflow-y-auto">
                                <div className="grid grid-cols-5 sm:grid-cols-10 md:grid-cols-12 gap-2">
                                {mnistSamples.map(sample => (
                                    <div 
                                        key={sample.id} 
                                        draggable 
                                        onDragStart={(e) => handleDragStart(e, sample.id)}
                                        className="cursor-grab active:cursor-grabbing p-1 hover:bg-cyan-500/20 rounded-md"
                                        title={`Digit: ${sample.label}`}
                                    >
                                        <MnistCanvas tensor={sample.tensor} size={40} />
                                    </div>
                                ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};