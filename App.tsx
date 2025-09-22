
import React, { useState, useCallback, useEffect, useRef } from 'react';
import type * as tf from '@tensorflow/tfjs';
import { TrainingConfigurator } from './components/TrainingConfigurator';
import { TrainingDashboard } from './components/TrainingDashboard';
import { InferenceTester } from './components/InferenceTester';
import { Header } from './components/Header';
import { ModelVisualizer } from './components/ModelVisualizer';
import { InfoModal } from './components/InfoModal';
import { WaveScape } from './components/WaveScape';
import { createModel, trainModel, saveModel, loadModel, checkForSavedModel, deleteSavedModel, downloadModel } from './services/modelService';
import { MnistData } from './services/mnistData';
import type { ModelConfig, TrainingLog, ModelInfo } from './types';
import { BrainCircuitIcon, PlayIcon, RocketIcon, SaveIcon, FolderDownIcon, TrashIcon, DownloadIcon, XIcon } from './constants';
import { analyzeModel } from './services/modelAnalysisService';
// Info Modal Content
import { AboutInfo } from './components/info-content/AboutInfo';
import { ArchitectureInfo } from './components/info-content/ArchitectureInfo';
import { LayersInfo } from './components/info-content/LayersInfo';
import { LearningRateInfo } from './components/info-content/LearningRateInfo';
import { EpochsBatchSizeInfo } from './components/info-content/EpochsBatchSizeInfo';

interface TrainingRun {
    id: number;
    config: ModelConfig;
    log: TrainingLog[];
    model: tf.Sequential | null;
    modelInfo: ModelInfo | null;
}

const MAX_COMPLETED_RUNS = 3;

type Page = 'mnist' | 'wavescape';

const App: React.FC = () => {
    const [page, setPage] = useState<Page>('mnist');
    const [config, setConfig] = useState<ModelConfig>({
        layers: [{ units: 128, activation: 'relu' }],
        learningRate: 0.005,
        lrSchedule: 'cosine',
        epochs: 10,
        batchSize: 512,
        architecture: 'dense',
    });
    const [isTraining, setIsTraining] = useState(false);
    const [trainingLog, setTrainingLog] = useState<TrainingLog[]>([]);
    const [modelForTesting, setModelForTesting] = useState<tf.Sequential | null>(null);
    const [modelToVisualize, setModelToVisualize] = useState<tf.Sequential | null>(null);
    const [trainingStatus, setTrainingStatus] = useState('Ready to train.');
    const [savedModelExists, setSavedModelExists] = useState(false);
    const [completedRuns, setCompletedRuns] = useState<TrainingRun[]>([]);
    const [activeInfoModal, setActiveInfoModal] = useState<string | null>(null);
    const [currentModelInfo, setCurrentModelInfo] = useState<ModelInfo | null>(null);
    const stopTrainingRef = useRef(false);

    useEffect(() => {
        const checkModel = async () => {
            const exists = await checkForSavedModel();
            setSavedModelExists(exists);
        };
        checkModel();
    }, []);

    const handleStopTraining = useCallback(() => {
        stopTrainingRef.current = true;
        setTrainingStatus('Stopping training...');
    }, []);

    const handleStartTraining = useCallback(async () => {
        setIsTraining(true);
        setTrainingLog([]);
        setModelForTesting(null);
        setTrainingStatus('Initializing...');
        stopTrainingRef.current = false;
        
        const currentRunLog: TrainingLog[] = [];
        let newModel: tf.Sequential | null = null;
        let modelInfo: ModelInfo | null = null;

        try {
            newModel = createModel(config);
            newModel.summary();

            modelInfo = analyzeModel(newModel, config);
            setCurrentModelInfo(modelInfo);

            setTrainingStatus('Loading MNIST dataset...');
            const data = await MnistData.getInstance();
            const { images: trainImages, labels: trainLabels } = data.getTrainData();
            const { images: testImages, labels: testLabels } = data.getValidationData();
            
            setTrainingStatus('Starting training...');
            await trainModel(
                newModel,
                trainImages,
                trainLabels,
                testImages,
                testLabels,
                config,
                async (epoch, logs, lr) => {
                    if (stopTrainingRef.current && newModel) {
                        newModel.stopTraining = true;
                    }
                    const newLogEntry: TrainingLog = {
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        val_loss: logs.val_loss,
                        val_accuracy: logs.val_acc,
                        lr
                    };
                    currentRunLog.push(newLogEntry);

                    const isLastEpoch = epoch + 1 === config.epochs;
                    const isUpdateEpoch = (epoch + 1) % 3 === 0;

                    if (isUpdateEpoch || isLastEpoch) {
                        setTrainingLog([...currentRunLog]);
                        // Yield to the browser's render cycle. requestAnimationFrame ensures that the main thread
                        // is free for the browser to process the UI update and paint the new chart data before
                        // the next heavy training epoch begins. This is more reliable than setTimeout.
                        await new Promise(resolve => requestAnimationFrame(resolve));
                    }
                    
                    if (!stopTrainingRef.current) {
                        setTrainingStatus(`Epoch ${epoch + 1}/${config.epochs} complete...`);
                    }
                }
            );
            
            const finalStatus = stopTrainingRef.current 
                ? 'Training stopped by user.'
                : 'Training complete!';
            
            setTrainingStatus(finalStatus);

            // Add to completed runs
            setCompletedRuns(prev => {
                const newRun: TrainingRun = {
                    id: Date.now(),
                    config: { ...config },
                    log: currentRunLog,
                    model: newModel,
                    modelInfo: modelInfo,
                };
                const updatedRuns = [newRun, ...prev];
                return updatedRuns.slice(0, MAX_COMPLETED_RUNS);
            });

        } catch (error) {
            console.error('Training failed:', error);
            setTrainingStatus(`Error during training: ${error instanceof Error ? error.message : 'Unknown error'}`);
        } finally {
            setIsTraining(false);
            setTrainingLog([]); // Clear live log
            setCurrentModelInfo(null);
        }
    }, [config]);

    const handleSaveModel = useCallback(async () => {
        if (!modelForTesting) return;
        setTrainingStatus('Saving model to local storage...');
        try {
            await saveModel(modelForTesting);
            setSavedModelExists(true);
            setTrainingStatus('Model saved successfully!');
        } catch (error) {
            console.error('Failed to save model:', error);
            setTrainingStatus(`Error saving model: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }, [modelForTesting]);

    const handleDownloadModel = useCallback(async () => {
        if (!modelForTesting) return;
        setTrainingStatus('Preparing model for download...');
        try {
            await downloadModel(modelForTesting);
            setTrainingStatus('Model download started!');
        } catch (error) {
            console.error('Failed to download model:', error);
            setTrainingStatus(`Error downloading model: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }, [modelForTesting]);

    const handleLoadModel = useCallback(async () => {
        setTrainingStatus('Loading model from local storage...');
        try {
            const loadedModel = await loadModel();
            setModelForTesting(loadedModel);
            setTrainingStatus('Model loaded! Ready for testing.');
        } catch (error) {
            console.error('Failed to load model:', error);
            setTrainingStatus(`Error loading model: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }, []);
    
    const handleDeleteModel = useCallback(async () => {
        if (window.confirm('Are you sure you want to delete the saved model? This action cannot be undone.')) {
            setTrainingStatus('Deleting saved model...');
            try {
                await deleteSavedModel();
                setSavedModelExists(false);
                setTrainingStatus('Saved model deleted.');
                if(modelForTesting?.name === 'localstorage://mnist-model') {
                    setModelForTesting(null);
                }
            } catch (error) {
                console.error('Failed to delete model:', error);
                setTrainingStatus(`Error deleting model: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
        }
    }, [modelForTesting]);

    const handleClearRuns = () => {
        setCompletedRuns([]);
        setModelForTesting(null);
        setCurrentModelInfo(null);
    }

    const handleShowInfo = (topic: string) => setActiveInfoModal(topic);
    const handleCloseInfo = () => setActiveInfoModal(null);

    const renderInfoModal = () => {
        if (!activeInfoModal) return null;

        let title = '';
        let content: React.ReactNode = null;

        switch (activeInfoModal) {
            case 'about':
                title = 'About This App';
                content = <AboutInfo />;
                break;
            case 'architecture':
                title = 'Understanding Model Architectures';
                content = <ArchitectureInfo />;
                break;
            case 'layers':
                title = 'Hidden Layers, Units, and Activations';
                content = <LayersInfo />;
                break;
            case 'lr':
                title = 'The Learning Rate & Schedules';
                content = <LearningRateInfo />;
                break;
            case 'epochs-batch':
                title = 'Epochs & Batch Size';
                content = <EpochsBatchSizeInfo />;
                break;
            default:
                return null;
        }

        return (
            <InfoModal title={title} onClose={handleCloseInfo}>
                {content}
            </InfoModal>
        );
    };

    const isIdle = !isTraining && trainingLog.length === 0 && completedRuns.length === 0;

    const renderMnistExplorer = () => (
        <main className="w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
            <div className="lg:col-span-1 space-y-8">
                <TrainingConfigurator 
                    config={config} 
                    setConfig={setConfig} 
                    onStartTraining={handleStartTraining} 
                    onStopTraining={handleStopTraining}
                    isTraining={isTraining} 
                    onShowInfo={handleShowInfo}
                />
                {savedModelExists && (
                     <div className="bg-white/10 border border-white/20 rounded-2xl p-6 space-y-4 shadow-2xl">
                        <h2 className="text-xl font-bold text-white">Manage Saved Model</h2>
                        <p className="text-sm text-gray-300">A trained model is available in your browser's local storage.</p>
                        <div className="flex flex-col sm:flex-row gap-4">
                            <button 
                                onClick={handleLoadModel} 
                                disabled={isTraining}
                                className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-400 hover:to-emerald-400 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-wait"
                            >
                                <FolderDownIcon className="w-5 h-5"/>
                                <span>Load Model</span>
                            </button>
                            <button 
                                onClick={handleDeleteModel} 
                                disabled={isTraining}
                                className="flex-1 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-400 hover:to-pink-400 text-white font-bold py-2 px-4 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105 disabled:opacity-50"
                            >
                                <TrashIcon className="w-5 h-5"/>
                                <span>Delete Model</span>
                            </button>
                        </div>
                    </div>
                )}
            </div>
            <div className="lg:col-span-2 space-y-8">
                {isIdle ? (
                    <div className="bg-white/10 border border-white/20 rounded-2xl p-8 h-full flex flex-col justify-center items-center text-center shadow-2xl">
                        <BrainCircuitIcon className="w-24 h-24 text-cyan-300 mb-6" />
                        <h2 className="text-3xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-500">Welcome to the ML Explorer</h2>
                        <p className="text-gray-300 max-w-md">Configure your neural network on the left, then click 'Start Training' to see it learn{savedModelExists && ", or load a previously saved model"}!</p>
                        {!isTraining && (
                            <button
                                onClick={handleStartTraining}
                                className="mt-8 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-bold py-3 px-8 rounded-full flex items-center space-x-2 transition-all duration-300 transform hover:scale-105 shadow-lg"
                            >
                                <PlayIcon className="w-5 h-5" />
                                <span>Start First Training</span>
                            </button>
                        )}
                    </div>
                ) : (
                  <>
                    {completedRuns.length > 0 && (
                        <div className="flex justify-between items-center">
                            <h2 className="text-2xl font-bold text-white">Comparison Runs</h2>
                            <button onClick={handleClearRuns} className="flex items-center space-x-2 text-sm text-gray-400 hover:text-red-400 transition">
                                <TrashIcon className="w-4 h-4" />
                                <span>Clear All Runs</span>
                            </button>
                        </div>
                    )}

                    {(isTraining || trainingLog.length > 0) && (
                        <TrainingDashboard 
                            isLive={true}
                            trainingLog={trainingLog} 
                            config={config}
                            status={trainingStatus}
                            modelInfo={currentModelInfo}
                        />
                    )}
                    
                    {completedRuns.map(run => (
                        <TrainingDashboard 
                            key={run.id}
                            isLive={false}
                            trainingLog={run.log}
                            config={run.config}
                            status="Completed"
                            onTestModel={() => setModelForTesting(run.model)}
                            onVisualizeModel={() => setModelToVisualize(run.model)}
                            isModelInTest={modelForTesting === run.model}
                            modelInfo={run.modelInfo}
                        />
                    ))}
                  </>
                )}
                {modelForTesting && (
                   <div className="bg-white/10 border border-white/20 rounded-2xl p-8 shadow-2xl">
                     <div className="flex flex-col sm:flex-row items-center justify-between space-y-4 sm:space-y-0 sm:space-x-4 mb-6">
                       <div className="flex items-center space-x-4">
                         <RocketIcon className="w-10 h-10 text-pink-400" />
                         <div>
                           <h2 className="text-2xl font-bold text-white">Test Your Model</h2>
                           <p className="text-gray-300">Draw digits on the canvases below to see your model in action.</p>
                         </div>
                       </div>
                       <div className="flex flex-col sm:flex-row gap-3">
                        <button 
                            onClick={handleSaveModel}
                            className="w-full sm:w-auto bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-400 hover:to-fuchsia-400 text-white font-bold py-2 px-6 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105"
                        >
                            <SaveIcon className="w-5 h-5" />
                            <span>Save Model</span>
                        </button>
                        <button 
                            onClick={handleDownloadModel}
                            className="w-full sm:w-auto bg-gradient-to-r from-blue-500 to-teal-500 hover:from-blue-400 hover:to-teal-400 text-white font-bold py-2 px-6 rounded-full flex items-center justify-center space-x-2 transition-all duration-300 transform hover:scale-105"
                        >
                            <DownloadIcon className="w-5 h-5" />
                            <span>Download Model</span>
                        </button>
                        <button onClick={() => setModelForTesting(null)} className="w-full sm:w-auto bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-6 rounded-full flex items-center justify-center space-x-2 transition-all duration-300">
                            <XIcon className="w-5 h-5" />
                            <span>Close</span>
                        </button>
                       </div>
                     </div>
                     <InferenceTester model={modelForTesting} />
                   </div>
                )}
            </div>
        </main>
    );

    const renderWaveScape = () => (
         <main className="w-full max-w-7xl mx-auto flex flex-col items-center gap-8 mt-8 px-4">
            <div className="text-center">
                <h2 className="text-3xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-sky-500">Interactive Gradient Descent</h2>
                <p className="text-gray-300 max-w-3xl">
                    Explore how an optimizer finds the minimum of a complex loss function. The ball represents the model's parameters, and it follows the slope (gradient) of the "wavescape" to find the lowest point. Adjust the learning rate, momentum, and schedule to see how they affect its path.
                </p>
            </div>
            <WaveScape />
          </main>
    );

    return (
        <div 
            className="min-h-screen w-full bg-black bg-cover bg-center bg-fixed text-white font-sans" 
            style={{ backgroundImage: `url('https://files.catbox.moe/w544w8.webp')` }}
        >
            {renderInfoModal()}

            {modelToVisualize ? (
                <ModelVisualizer model={modelToVisualize} onClose={() => setModelToVisualize(null)} />
            ) : (
                <div className="min-h-screen w-full bg-black/60 backdrop-blur-sm flex flex-col">
                    <div className="flex-grow flex flex-col items-center p-4 sm:p-8">
                        <Header onShowInfo={handleShowInfo} page={page} setPage={setPage} />
                        {page === 'mnist' ? renderMnistExplorer() : renderWaveScape()}
                    </div>
                    <footer className="w-full text-center p-4 text-xs text-gray-400">
                        App shared under a CC-BY-SA license by Morgan Andreasson
                    </footer>
                </div>
            )}
        </div>
    );
};

export default App;
