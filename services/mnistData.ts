
import * as tf from '@tensorflow/tfjs';

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export class MnistData {
    private datasetImages: Float32Array | null = null;
    private datasetLabels: Uint8Array | null = null;
    private trainImages: tf.Tensor | null = null;
    private testImages: tf.Tensor | null = null;
    private trainLabels: tf.Tensor | null = null;
    private testLabels: tf.Tensor | null = null;

    async load() {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize);
                    ctx!.drawImage(
                        img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                        chunkSize);

                    const imageData = ctx!.getImageData(0, 0, canvas.width, canvas.height);

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve(true);
            };
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });

        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const [imgResponse, labelsResponse] =
            await Promise.all([imgRequest, labelsRequest]);

        this.datasetLabels = new Uint8Array(await (labelsResponse as Response).arrayBuffer());

        this.trainImages = tf.tensor2d(this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS), [NUM_TRAIN_ELEMENTS, IMAGE_SIZE]);
        this.testImages = tf.tensor2d(this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS), [NUM_TEST_ELEMENTS, IMAGE_SIZE]);
        this.trainLabels = tf.oneHot(tf.tensor1d(this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS), 'int32').slice([0], [NUM_TRAIN_ELEMENTS]), NUM_CLASSES);
        this.testLabels = tf.oneHot(tf.tensor1d(this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS), 'int32').slice([0], [NUM_TEST_ELEMENTS]), NUM_CLASSES);
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
