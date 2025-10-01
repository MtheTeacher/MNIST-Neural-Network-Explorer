import * as tf from '@tensorflow/tfjs';
import { MNIST_IMAGES_BASE64, MNIST_LABELS_URL } from './embeddedData';

// Constants for the full dataset stored in the sprite file.
const FILE_NUM_TRAIN_ELEMENTS = 55000;
const FILE_NUM_TEST_ELEMENTS = 10000;
const FILE_NUM_DATASET_ELEMENTS = FILE_NUM_TRAIN_ELEMENTS + FILE_NUM_TEST_ELEMENTS;

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const NUM_CLASSES = 10;

/**
 * A singleton class that fetches, parses, and provides the MNIST dataset from a sprite sheet.
 * It ensures the data is loaded only once and is accessible throughout the app.
 */
export class MnistData {
    private static instance: MnistData;

    private trainImages: tf.Tensor | null = null;
    private trainLabels: tf.Tensor | null = null;
    private validationImages: tf.Tensor | null = null;
    private validationLabels: tf.Tensor | null = null;
    
    private testImageSamples: tf.Tensor[] = [];
    private testLabelSamples: number[] = [];
    
    private constructor() {}

    public static async getInstance(): Promise<MnistData> {
        if (!MnistData.instance) {
            MnistData.instance = new MnistData();
            await MnistData.instance.load();
        }
        return MnistData.instance;
    }
    
    private async load() {
        console.log('Loading MNIST data from embedded source for maximum reliability...');
        
        try {
             // Fetch the labels and create an in-memory image element from the Base64 data URI.
            const [labelsResponse, img] = await Promise.all([
                fetch(MNIST_LABELS_URL),
                new Promise<HTMLImageElement>((resolve, reject) => {
                    const image = new Image();
                    image.crossOrigin = 'anonymous'; // Still needed if the base64 string is a URL
                    image.onload = () => resolve(image);
                    image.onerror = (err) => reject(new Error('Failed to load image from data source.'));
                    image.src = MNIST_IMAGES_BASE64; // This is a URL but could be a giant base64 string
                })
            ]);

            if (!labelsResponse.ok) throw new Error(`Failed to fetch labels: ${labelsResponse.statusText}`);
            
            const labelsBuffer = await labelsResponse.arrayBuffer();
            
            // Draw the image sprite onto a canvas to access its raw pixel data.
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not create canvas context');
            
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const datasetBytesBuffer = ctx.getImageData(0, 0, img.width, img.height).data;
            
            // The sprite is grayscale, so R, G, and B channels are identical. We only need one.
            // Normalize the pixel values from [0, 255] to [0, 1].
            const datasetBytes = new Float32Array(FILE_NUM_DATASET_ELEMENTS * IMAGE_SIZE);
            for (let i = 0; i < datasetBytesBuffer.length / 4; i++) {
                datasetBytes[i] = datasetBytesBuffer[i * 4] / 255;
            }

            // The labels file is one-hot encoded. We need to decode it for the UI
            // and use it directly for the tensors.
            const labelsUint8 = new Uint8Array(labelsBuffer);
            const labelIndices = new Uint8Array(FILE_NUM_DATASET_ELEMENTS);
            for (let i = 0; i < FILE_NUM_DATASET_ELEMENTS; i++) {
                const offset = i * NUM_CLASSES;
                for (let j = 0; j < NUM_CLASSES; j++) {
                    if (labelsUint8[offset + j] === 1) {
                        labelIndices[i] = j;
                        break;
                    }
                }
            }

            // Create Tensors from the raw data and split into training and test sets.
            tf.tidy(() => {
                const datasetImages = tf.tensor2d(datasetBytes, [FILE_NUM_DATASET_ELEMENTS, IMAGE_SIZE]);
                const datasetLabels = tf.tensor2d(labelsUint8, [FILE_NUM_DATASET_ELEMENTS, NUM_CLASSES]);
                
                const fullTrainImages = datasetImages.slice([0, 0], [FILE_NUM_TRAIN_ELEMENTS, IMAGE_SIZE]);
                const fullTrainLabels = datasetLabels.slice([0, 0], [FILE_NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
                
                const shuffledIndicesArray = tf.util.createShuffledIndices(FILE_NUM_TRAIN_ELEMENTS);
                const shuffledIndices = tf.tensor1d(new Int32Array(shuffledIndicesArray), 'int32');

                this.trainImages = tf.keep(fullTrainImages.gather(shuffledIndices));
                this.trainLabels = tf.keep(fullTrainLabels.gather(shuffledIndices));

                const testImages = datasetImages.slice([FILE_NUM_TRAIN_ELEMENTS, 0], [FILE_NUM_TEST_ELEMENTS, IMAGE_SIZE]);
                const testLabels = datasetLabels.slice([FILE_NUM_TRAIN_ELEMENTS, 0], [FILE_NUM_TEST_ELEMENTS, NUM_CLASSES]);

                this.validationImages = tf.keep(testImages);
                this.validationLabels = tf.keep(testLabels);
                
                this.testImageSamples = tf.split(testImages, FILE_NUM_TEST_ELEMENTS).map(t => tf.keep(t));
            });

            this.testLabelSamples = Array.from(
                labelIndices.slice(FILE_NUM_TRAIN_ELEMENTS, FILE_NUM_DATASET_ELEMENTS)
            );
            
            console.log(`Successfully loaded MNIST data from source.`);
            console.log(`- Training samples: ${FILE_NUM_TRAIN_ELEMENTS}`);
            console.log(`- Validation samples (from test set): ${FILE_NUM_TEST_ELEMENTS}`);
            console.log(`- Test samples (for UI): ${FILE_NUM_TEST_ELEMENTS}`);

        } catch (error) {
            console.error(`Fatal error: Failed to load or parse MNIST data:`, error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to process MNIST data. The application may be offline or the data source is unavailable. Error: ${errorMessage}`);
        }
    }
    
    getTrainData() {
        if (!this.trainImages || !this.trainLabels) {
            throw new Error("Data not loaded yet. Call `await MnistData.getInstance()` first.");
        }
        return { images: this.trainImages, labels: this.trainLabels };
    }

    getValidationData() {
        if (!this.validationImages || !this.validationLabels) {
            throw new Error("Data not loaded yet. Call `await MnistData.getInstance()` first.");
        }
        return { images: this.validationImages, labels: this.validationLabels };
    }
    
    getTestSamplesForInference(numSamples: number) {
        if (this.testImageSamples.length === 0 || this.testLabelSamples.length === 0) {
             throw new Error("Data not loaded yet. Call `await MnistData.getInstance()` first.");
        }
        const count = Math.min(numSamples, this.testLabelSamples.length);
        return Array.from({ length: count }).map((_, i) => ({
            tensor: this.testImageSamples[i],
            label: this.testLabelSamples[i],
            id: i,
        }));
    }
}
