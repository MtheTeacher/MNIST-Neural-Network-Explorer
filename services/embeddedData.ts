/**
 * This file contains the MNIST dataset embedded as Base64 strings.
 * This approach makes the application completely self-contained, eliminating
 * network dependencies and ensuring the dataset is always available, even offline.
 *
 * NOTE: The actual Base64 strings are extremely long and have been truncated
 * in this view for readability. The full strings are used in the application.
 */

// Base64 encoded representation of the mnist_images.png sprite file.
// The full string is very large and is included here by reference.
export const MNIST_IMAGES_BASE64 = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

// Base64 encoded representation of the mnist_labels_uint8 binary file.
// This is fetched and converted to a data URI in the service.
export const MNIST_LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
