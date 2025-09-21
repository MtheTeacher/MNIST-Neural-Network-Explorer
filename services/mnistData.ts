import * as tf from '@tensorflow/tfjs';

// Paths to the MNIST dataset files. Assuming they are served from the root.
const TRAIN_IMAGES_URL = '/train-images.idx3-ubyte';
const TRAIN_LABELS_URL = '/train-labels.idx1-ubyte';
const TEST_IMAGES_URL = '/test10k-images.idx3-ubyte';
const TEST_LABELS_URL = '/test10k-labels.idx1-ubyte';

// MNIST file format constants.
const IMAGE_HEADER_BYTES = 16;
const LABEL_HEADER_BYTES = 8;
const IMAGE_SIZE = 784; // 28x28 pixels
const NUM_CLASSES = 10;

// Magic numbers to verify file integrity.
const IMAGE_MAGIC_NUMBER = 2051;
const LABEL_MAGIC_NUMBER = 2049;

/**
 * A singleton class that fetches, parses, and provides the MNIST dataset.
 * It ensures the data is loaded only once and is accessible throughout the app.
 */
export class MnistData {
    private static instance: MnistData;

    private trainImages: tf.Tensor | null = null;
    private testImages: tf.Tensor | null = null;
    private trainLabels: tf.Tensor | null = null;
    private testLabels: tf.Tensor | null = null;
    
    private testImageSamples: tf.Tensor[] = [];
    private testLabelSamples: number[] = [];
    
    private numTrainElements: number = 0;
    private numTestElements: number = 0;

    private constructor() {}

    public static async getInstance(): Promise<MnistData> {
        if (!MnistData.instance) {
            MnistData.instance = new MnistData();
            await MnistData.instance.load();
        }
        return MnistData.instance;
    }
    
    private async load() {
        console.log('Loading MNIST data...');

        // Helper function to fetch a file and throw a clear error on failure.
        const fetchAndValidate = async (url: string, expectedMagicNumber: number) => {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
            }
            const buffer = await response.arrayBuffer();
            const magicNumber = new DataView(buffer).getInt32(0);
            if (magicNumber !== expectedMagicNumber) {
                throw new Error(`Invalid magic number for file ${url}. Expected ${expectedMagicNumber}, got ${magicNumber}`);
            }
            return buffer;
        };

        const [trainImagesBuffer, trainLabelsBuffer, testImagesBuffer, testLabelsBuffer] = await Promise.all([
            fetchAndValidate(TRAIN_IMAGES_URL, IMAGE_MAGIC_NUMBER),
            fetchAndValidate(TRAIN_LABELS_URL, LABEL_MAGIC_NUMBER),
            fetchAndValidate(TEST_IMAGES_URL, IMAGE_MAGIC_NUMBER),
            fetchAndValidate(TEST_LABELS_URL, LABEL_MAGIC_NUMBER),
        ]);
        
        // Use DataView to read metadata from the headers (big-endian format).
        this.numTrainElements = new DataView(trainLabelsBuffer).getInt32(4);
        this.numTestElements = new DataView(testLabelsBuffer).getInt32(4);

        // Slice the buffers to get only the pixel and label data, skipping the headers.
        const trainImagesData = new Uint8Array(trainImagesBuffer, IMAGE_HEADER_BYTES);
        const testImagesData = new Uint8Array(testImagesBuffer, IMAGE_HEADER_BYTES);
        const trainLabelsData = new Uint8Array(trainLabelsBuffer, LABEL_HEADER_BYTES);
        const testLabelsData = new Uint8Array(testLabelsBuffer, LABEL_HEADER_BYTES);

        // Now, create the TensorFlow.js tensors.
        tf.tidy(() => {
            const trainImagesTensor = tf.tensor2d(trainImagesData, [this.numTrainElements, IMAGE_SIZE]);
            const testImagesTensor = tf.tensor2d(testImagesData, [this.numTestElements, IMAGE_SIZE]);
            const trainLabelsTensor = tf.tensor1d(trainLabelsData, 'int32');
            const testLabelsTensor = tf.tensor1d(testLabelsData, 'int32');

            // Normalize images to [0, 1] range.
            this.trainImages = trainImagesTensor.cast('float32').div(255);
            this.testImages = testImagesTensor.cast('float32').div(255);
            
            this.trainLabels = tf.oneHot(trainLabelsTensor, NUM_CLASSES);
            this.testLabels = tf.oneHot(testLabelsTensor, NUM_CLASSES);
            
            this.testLabelSamples = Array.from(testLabelsData);
            this.testImageSamples = tf.split(this.testImages, this.numTestElements);
        });
        
        console.log('MNIST data loaded successfully.');
        console.log(`- Training samples: ${this.numTrainElements}`);
        console.log(`- Test samples: ${this.numTestElements}`);
    }
    
    getTrainData() {
        if (!this.trainImages || !this.trainLabels) {
            throw new Error("Data not loaded yet. Call `await MnistData.getInstance()` first.");
        }
        return { images: this.trainImages, labels: this.trainLabels };
    }

    getTestData() {
        if (!this.testImages || !this.testLabels) {
            throw new Error("Data not loaded yet. Call `await MnistData.getInstance()` first.");
        }
        return { images: this.testImages, labels: this.testLabels };
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