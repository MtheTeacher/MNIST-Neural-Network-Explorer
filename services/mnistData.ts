import * as tf from '@tensorflow/tfjs';

// Per the recommendation in docs/DATA_SOURCES.md, we will exclusively use the
// Google-hosted source. It is known to be reliable and configured with the
// correct CORS headers for in-browser fetching, which has been the root
// of the recent loading failures.
const DATA_SOURCE = {
    name: 'Google TFJS',
    images: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
    labels: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'
};

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
        console.log('Loading MNIST data from the recommended Google TFJS source...');
        
        try {
            console.log(`Attempting to load data from ${DATA_SOURCE.name}...`);
            
            // Fetch the image sprite and the labels file concurrently.
            const [imgResponse, labelsResponse] = await Promise.all([
                fetch(DATA_SOURCE.images, { mode: 'cors' }),
                fetch(DATA_SOURCE.labels, { mode: 'cors' }),
            ]);

            if (!imgResponse.ok) throw new Error(`Failed to fetch ${DATA_SOURCE.images}: ${imgResponse.statusText}`);
            if (!labelsResponse.ok) throw new Error(`Failed to fetch ${DATA_SOURCE.labels}: ${labelsResponse.statusText}`);

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
            // Use tf.tidy to automatically dispose intermediate tensors, but use
            // tf.keep to prevent the final dataset tensors from being disposed.
            tf.tidy(() => {
                const datasetImages = tf.tensor2d(datasetBytes, [FILE_NUM_DATASET_ELEMENTS, IMAGE_SIZE]);
                const datasetLabels = tf.tensor2d(labelsUint8, [FILE_NUM_DATASET_ELEMENTS, NUM_CLASSES]);
                
                // Get the full 55,000 training images and shuffle them.
                const fullTrainImages = datasetImages.slice([0, 0], [FILE_NUM_TRAIN_ELEMENTS, IMAGE_SIZE]);
                const fullTrainLabels = datasetLabels.slice([0, 0], [FILE_NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
                
                // FIX: `tf.util.createShuffledIndices` returns a Uint32Array. The `tf.gather`
                // operation requires an int32 tensor for indices. Directly creating a tensor
                // from the Uint32Array can cause a type error, so we explicitly convert to
                // an Int32Array before creating the tensor.
                const shuffledIndicesArray = tf.util.createShuffledIndices(FILE_NUM_TRAIN_ELEMENTS);
                const shuffledIndices = tf.tensor1d(new Int32Array(shuffledIndicesArray), 'int32');

                // The training set is the complete, shuffled 55,000 images.
                this.trainImages = tf.keep(fullTrainImages.gather(shuffledIndices));
                this.trainLabels = tf.keep(fullTrainLabels.gather(shuffledIndices));

                // The validation set is the official 10,000-image test set, ensuring no data leakage.
                const testImages = datasetImages.slice([FILE_NUM_TRAIN_ELEMENTS, 0], [FILE_NUM_TEST_ELEMENTS, IMAGE_SIZE]);
                const testLabels = datasetLabels.slice([FILE_NUM_TRAIN_ELEMENTS, 0], [FILE_NUM_TEST_ELEMENTS, NUM_CLASSES]);

                this.validationImages = tf.keep(testImages);
                this.validationLabels = tf.keep(testLabels);

                // The UI test samples will also come from the same official test set.
                this.testImageSamples = tf.split(testImages, FILE_NUM_TEST_ELEMENTS).map(t => tf.keep(t));
            });

            // Store the decoded integer test labels for the inference tester drag-and-drop UI.
            this.testLabelSamples = Array.from(
                labelIndices.slice(FILE_NUM_TRAIN_ELEMENTS, FILE_NUM_DATASET_ELEMENTS)
            );
            
            console.log(`Successfully loaded MNIST data from ${DATA_SOURCE.name}.`);
            console.log(`- Training samples: ${FILE_NUM_TRAIN_ELEMENTS}`);
            console.log(`- Validation samples (from test set): ${FILE_NUM_TEST_ELEMENTS}`);
            console.log(`- Test samples (for UI): ${FILE_NUM_TEST_ELEMENTS}`);

        } catch (error) {
            console.error(`Failed to load data from ${DATA_SOURCE.name}:`, error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to load MNIST data. Please check the network connection and console for errors. Last error: ${errorMessage}`);
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