import * as tf from '@tensorflow/tfjs';

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const LABELS_HEADER_BYTES = 8;

export class MnistData {
    private datasetImages: Float32Array | null = null;
    private datasetLabels: Uint8Array | null = null;
    private trainImages: tf.Tensor | null = null;
    private testImages: tf.Tensor | null = null;
    private trainLabels: tf.Tensor | null = null;
    private testLabels: tf.Tensor | null = null;

    async load() {
        // Create promises for both the image sprite and the labels file.
        const imgRequest = new Promise<void>((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    return reject(new Error("Could not get 2d context"));
                }
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                // Create a buffer for all the image data.
                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const datasetBytesView = new Float32Array(datasetBytesBuffer);
                
                // Iterate over the image data, extracting the grayscale value for each pixel.
                for (let j = 0; j < imageData.data.length / 4; j++) {
                    // All channels (R,G,B) hold an equal value since the image is grayscale.
                    // We only need to read the red channel.
                    datasetBytesView[j] = imageData.data[j * 4] / 255;
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };
            img.onerror = reject;
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });

        const labelsRequest = fetch(MNIST_LABELS_PATH);

        // Wait for both promises to resolve.
        const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

        // Load the labels, skipping the 8-byte header.
        const labelsBuffer = await (labelsResponse as Response).arrayBuffer();
        this.datasetLabels = new Uint8Array(labelsBuffer, LABELS_HEADER_BYTES);
        
        // Use tf.tidy to manage memory of intermediate tensors.
        // The tensors we want to keep are returned from the tidy function.
        const { trainImages, testImages, trainLabels, testLabels } = tf.tidy(() => {
            if (!this.datasetImages || !this.datasetLabels) {
                throw new Error("Dataset not loaded");
            }
            const allImagesTensor = tf.tensor2d(this.datasetImages, [NUM_DATASET_ELEMENTS, IMAGE_SIZE]);
            const allLabelsTensor = tf.tensor1d(this.datasetLabels, 'int32');

            // Slice the full dataset into training and test sets.
            const trainImagesT = allImagesTensor.slice([0, 0], [NUM_TRAIN_ELEMENTS, IMAGE_SIZE]);
            const testImagesT = allImagesTensor.slice([NUM_TRAIN_ELEMENTS, 0], [NUM_TEST_ELEMENTS, IMAGE_SIZE]);
            const trainLabelsT = allLabelsTensor.slice([0], [NUM_TRAIN_ELEMENTS]);
            const testLabelsT = allLabelsTensor.slice([NUM_TRAIN_ELEMENTS], [NUM_TEST_ELEMENTS]);
            
            // One-hot encode the labels.
            const trainLabelsOneHot = tf.oneHot(trainLabelsT, NUM_CLASSES);
            const testLabelsOneHot = tf.oneHot(testLabelsT, NUM_CLASSES);

            // Return the tensors that we want to keep. The others will be disposed.
            return { 
                trainImages: trainImagesT, 
                testImages: testImagesT, 
                trainLabels: trainLabelsOneHot,
                testLabels: testLabelsOneHot
            };
        });

        this.trainImages = trainImages;
        this.testImages = testImages;
        this.trainLabels = trainLabels;
        this.testLabels = testLabels;

        // Nullify the large TypedArrays to free up memory
        this.datasetImages = null;
        this.datasetLabels = null;
    }

    getTrainData() {
        if (!this.trainImages || !this.trainLabels) throw new Error("Data not loaded yet");
        return { images: this.trainImages, labels: this.trainLabels };
    }

    getTestData() {
        if (!this.testImages || !this.testLabels) throw new Error("Data not loaded yet");
        return { images: this.testImages, labels: this.testLabels };
    }
}