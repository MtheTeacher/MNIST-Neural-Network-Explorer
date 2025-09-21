import * as tf from '@tensorflow/tfjs';

// Define multiple sources for the MNIST dataset to add resilience.
// The sources are ordered by preference, starting with the most stable mirrors.
const DATA_SOURCES = [
    {
        name: 'TFJS Examples GCS (model-builder)',
        images: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
        labels: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'
    },
    {
        name: 'TFJS Examples GCS (common)',
        images: 'https://storage.googleapis.com/learnjs-data/common/mnist_images.png',
        labels: 'https://storage.googleapis.com/learnjs-data/common/mnist_labels_uint8'
    },
    {
        name: 'jsDelivr CDN',
        images: 'https://cdn.jsdelivr.net/gh/tensorflow/tfjs-examples-data@master/mnist/mnist_images.png',
        labels: 'https://cdn.jsdelivr.net/gh/tensorflow/tfjs-examples-data@master/mnist/mnist_labels_uint8'
    },
    {
        name: 'GitHub Raw Content',
        images: 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data/mnist_images.png',
        labels: 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data/mnist_labels_uint8'
    }
];

// Dataset constants for this specific sprite version.
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = 10000;
const NUM_DATASET_ELEMENTS = NUM_TRAIN_ELEMENTS + NUM_TEST_ELEMENTS;

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
    private testImages: tf.Tensor | null = null;
    private trainLabels: tf.Tensor | null = null;
    private testLabels: tf.Tensor | null = null;
    
    private testImageSamples: tf.Tensor[] = [];
    private testLabelSamples: number[] = [];
    
    private numTrainElements: number = NUM_TRAIN_ELEMENTS;
    private numTestElements: number = NUM_TEST_ELEMENTS;

    private constructor() {}

    public static async getInstance(): Promise<MnistData> {
        if (!MnistData.instance) {
            MnistData.instance = new MnistData();
            await MnistData.instance.load();
        }
        return MnistData.instance;
    }
    
    private async load() {
        console.log('Loading MNIST data from sprite...');
        
        let lastError: Error | null = null;

        for (const source of DATA_SOURCES) {
            try {
                console.log(`Attempting to load data from ${source.name}...`);
                
                // Fetch the image sprite and the labels file concurrently.
                const [imgResponse, labelsResponse] = await Promise.all([
                    fetch(source.images, { mode: 'cors' }),
                    fetch(source.labels, { mode: 'cors' }),
                ]);

                if (!imgResponse.ok) throw new Error(`Failed to fetch ${source.images}: ${imgResponse.statusText}`);
                if (!labelsResponse.ok) throw new Error(`Failed to fetch ${source.labels}: ${labelsResponse.statusText}`);

                const imgBlob = await imgResponse.blob();
                const labelsBuffer = await labelsResponse.arrayBuffer();

                // Create an in-memory image element from the fetched sprite.
                const img = await new Promise<HTMLImageElement>((resolve, reject) => {
                    const image = new Image();
                    image.crossOrigin = 'anonymous';
                    image.onload = () => resolve(image);
                    image.onerror = reject;
                    image.src = URL.createObjectURL(imgBlob);
                });

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
                const datasetBytes = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);
                for (let i = 0; i < datasetBytesBuffer.length / 4; i++) {
                    datasetBytes[i] = datasetBytesBuffer[i * 4] / 255;
                }

                const labels = new Uint8Array(labelsBuffer);

                // Create Tensors from the raw data and split into training and test sets.
                // Use tf.tidy to automatically dispose intermediate tensors, but use
                // tf.keep to prevent the final dataset tensors from being disposed.
                tf.tidy(() => {
                    const datasetImages = tf.tensor2d(datasetBytes, [NUM_DATASET_ELEMENTS, IMAGE_SIZE]);
                    const datasetLabels = tf.tensor1d(labels, 'int32');
                    
                    this.trainImages = tf.keep(datasetImages.slice([0, 0], [NUM_TRAIN_ELEMENTS, IMAGE_SIZE]));
                    this.testImages = tf.keep(datasetImages.slice([NUM_TRAIN_ELEMENTS, 0], [NUM_TEST_ELEMENTS, IMAGE_SIZE]));
                    
                    this.trainLabels = tf.keep(tf.oneHot(datasetLabels.slice([0], [NUM_TRAIN_ELEMENTS]), NUM_CLASSES));
                    const testLabelsTensor = datasetLabels.slice([NUM_TRAIN_ELEMENTS], [NUM_TEST_ELEMENTS]);
                    this.testLabels = tf.keep(tf.oneHot(testLabelsTensor, NUM_CLASSES));
                    
                    // Store raw test labels and split test images for the inference tester.
                    this.testLabelSamples = Array.from(labels.slice(NUM_TRAIN_ELEMENTS));
                    const testImageSamplesTensors = tf.split(this.testImages, NUM_TEST_ELEMENTS);
                    // Keep each tensor in the split array.
                    testImageSamplesTensors.forEach(t => tf.keep(t));
                    this.testImageSamples = testImageSamplesTensors;
                });
                
                console.log(`Successfully loaded MNIST data from ${source.name}.`);
                console.log(`- Training samples: ${this.numTrainElements}`);
                console.log(`- Test samples: ${this.numTestElements}`);

                // If we got this far, the data is loaded. Exit the method.
                return;

            } catch (error) {
                console.warn(`Failed to load data from ${source.name}:`, error);
                lastError = error as Error;
            }
        }
        
        // If the loop completes without returning, all sources failed.
        if (lastError) {
            throw new Error(`Failed to load MNIST data from all sources. Last error: ${lastError.message}`);
        } else {
            throw new Error('Failed to load MNIST data from any source.');
        }
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