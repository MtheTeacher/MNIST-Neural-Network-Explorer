import React, { useState, useCallback, useRef, createRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { DrawingCanvas, type DrawingCanvasRef } from './DrawingCanvas';
import { SparklesIcon, EraserIcon, PencilIcon, ImageIcon, CheckCircleIcon, XCircleIcon } from '../constants';
import { MnistData } from '../services/mnistData';

interface InferenceTesterProps {
    model: tf.Sequential;
}

interface MnistSample {
    tensor: tf.Tensor; // Shape [1, 784]
    label: number;
    id: number; // index in the original test set
}

// A small component to render a 28x28 MNIST image tensor to a canvas
const MnistCanvas: React.FC<{ tensor: tf.Tensor, size: number }> = ({ tensor, size }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const draw = () => {
            if (!canvasRef.current || !tensor) return;
            const canvas = canvasRef.current;
            tf.tidy(() => {
                const imageTensor = tensor.reshape([28, 28, 1]);
                // FIX: Cast imageTensor to Tensor3D, as tf.browser.toPixels requires a more specific tensor type than the inferred one.
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
            className="bg-black border border-gray-600 rounded-md"
        />
    );
};

export const InferenceTester: React.FC<InferenceTesterProps> = ({ model }) => {
    // Shared state
    const [isProcessing, setIsProcessing] = useState(false);
    const [testMode, setTestMode] = useState<'draw' | 'mnist'>('draw');

    // State for 'draw' mode
    const [drawPredictions, setDrawPredictions] = useState<(number | null)[]>(Array(10).fill(null));
    const canvasRefs = useRef<React.RefObject<DrawingCanvasRef>[]>(
        Array(10).fill(null).map(() => createRef<DrawingCanvasRef>())
    );

    // State for 'mnist' mode
    // FIX: Corrected typo from MmistSample to MnistSample.
    const [mnistSamples, setMnistSamples] = useState<MnistSample[]>([]);
    const [isLoadingMnist, setIsLoadingMnist] = useState(false);
    // FIX: Corrected typo from MmistSample to MnistSample.
    const [droppedImages, setDroppedImages] = useState<(MnistSample | null)[]>(Array(10).fill(null));
    const [mnistPredictions, setMnistPredictions] = useState<(number | null)[]>(Array(10).fill(null));
    const samplesRef = useRef(mnistSamples);
    samplesRef.current = mnistSamples;
    
    // Load MNIST test data when switching to that mode
    useEffect(() => {
        if (testMode === 'mnist' && mnistSamples.length === 0) {
            const loadMnistData = async () => {
                setIsLoadingMnist(true);
                try {
                    const data = new MnistData();
                    await data.load();
                    const { images: testImages, labels: testLabels } = data.getTestData();
                    const testLabelsArray = await testLabels.argMax(-1).data() as Uint8Array;
                    
                    const samples: MnistSample[] = [];
                    // Slice the first 100 images
                    const imageSlices = tf.split(testImages.slice([0, 0], [100, 784]), 100);
                    
                    for (let i = 0; i < 100; i++) {
                        samples.push({
                            tensor: imageSlices[i],
                            label: testLabelsArray[i],
                            id: i,
                        });
                    }
                    setMnistSamples(samples);
                    tf.dispose([testImages, testLabels]);

                } catch (error) {
                    console.error("Failed to load MNIST data for testing:", error);
                } finally {
                    setIsLoadingMnist(false);
                }
            };
            loadMnistData();
        }
    }, [testMode, mnistSamples.length]);

    // Cleanup tensors on unmount
    useEffect(() => {
        return () => {
            samplesRef.current.forEach(sample => sample.tensor.dispose());
        };
    }, []);

    const handlePredictAll = useCallback(async () => {
        setIsProcessing(true);
        if (testMode === 'draw') {
            const tensorsAndIndices: { tensor: tf.Tensor; index: number }[] = [];
            canvasRefs.current.forEach((ref, index) => {
                const tensor = ref.current?.getTensor();
                if (tensor) tensorsAndIndices.push({ tensor, index });
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
                tf.dispose([batch, predictionsTensor, ...tensorsToBatch]);
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
    }, [model, testMode, drawPredictions, droppedImages, mnistPredictions]);

    const handleClearAll = () => {
        if (testMode === 'draw') {
            canvasRefs.current.forEach(ref => ref.current?.clearCanvas());
            setDrawPredictions(Array(10).fill(null));
        } else {
            setDroppedImages(Array(10).fill(null));
            setMnistPredictions(Array(10).fill(null));
        }
    };
    
    // --- Drag and Drop Handlers ---
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
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                    {Array.from({ length: 10 }).map((_, i) => (
                        <div key={i} className="flex flex-col items-center text-center space-y-2">
                            <DrawingCanvas ref={canvasRefs.current[i]} />
                            <div className="h-12 flex items-center justify-center">
                                {drawPredictions[i] !== null && (
                                    <p className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
                                        {drawPredictions[i]}
                                    </p>
                                )}
                            </div>
                        </div>
                    ))}
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