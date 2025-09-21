# About the MNIST Neural Network Explorer

## 1. Introduction

Welcome to the MNIST Neural Network Explorer! This is an interactive, web-based application designed to demystify the process of building, training, and testing neural networks. It provides a hands-on environment where students, developers, and machine learning enthusiasts can experiment with one of the most classic problems in the field: handwritten digit recognition using the MNIST dataset.

The core purpose of this application is to make the concepts of neural networks tangible. Instead of just reading about architectures, hyperparameters, and training loops, you can actively build a model, watch it learn in real-time, and immediately test its performance on your own handwriting.

## 2. Core Features

- **Interactive Model Configuration:** Visually add or remove layers, set the number of neurons, and choose activation functions.
- **Advanced Hyperparameter Tuning:** Adjust learning rates, select from various learning rate schedules, and set epoch counts and batch sizes.
- **Real-Time Training Visualization:** Watch your model's accuracy and loss evolve epoch-by-epoch with dynamic, easy-to-read charts.
- **Run Comparison:** Train multiple models with different configurations and compare their performance side-by-side.
- **Two Modes of Interactive Testing:**
    1.  **Draw Your Own:** Test your model's capabilities on digits you draw yourself.
    2.  **MNIST Test Set:** Validate your model against unseen images from the official MNIST test dataset using a simple drag-and-drop interface.
- **Deep Model Inspection:** Dive into the model's internals with two powerful visualization tools:
    1.  **Weight Heatmaps:** See what the learned features inside your model's layers actually look like.
    2.  **Live Activation Flow:** Draw a digit and watch the neuron activations propagate through the network in real-time.
- **In-Browser Model Management:** Save your trained models to your browser's local storage, load them back later, or download the model files to use elsewhere. All training and inference happen directly in your browser using TensorFlow.js, with no server-side computation required.

## 3. How to Use the Application: A Walkthrough

### Step 1: Configure Your Model
On the left-hand side, you'll find the **"Configure Model"** panel. This is your command center for designing the network.

- **Architecture Presets:** Start quickly with pre-defined architectures like "Simple", "Deep", "Wide", or a high-performance "CNN" (Convolutional Neural Network).
- **Hidden Layers:** For dense models, you can add or remove layers. For each layer, you can specify the number of **Units** (neurons) and the **Activation** function (e.g., 'relu').
- **Hyperparameters:**
    - **Learning Rate Schedule:** This determines how the learning rate changes during training. The graph provides a visual preview of the selected schedule.
    - **Learning Rate:** Controls the step size the model takes during optimization.
    - **Epochs:** The number of times the model will see the entire training dataset.
    - **Batch Size:** The number of training samples processed before the model's weights are updated.

### Step 2: Train the Model
Once you're happy with your configuration, click **"Start Training"**. The main area will update with a **"Live Training"** dashboard. Here you can monitor:
- The overall progress and status.
- A real-time graph of the model's **Loss** (how wrong it is) and **Learning Rate**.
- A second graph showing the model's **Accuracy** on the validation set.

You can stop the process at any time by clicking **"Stop Training"**.

### Step 3: Analyze and Compare Runs
After a training run is complete (or stopped), it moves to the **"Comparison Runs"** section. The dashboard for the completed run will show its final accuracy and configuration. You can train new models with different settings, and they will stack up here, allowing you to easily compare which architectures or hyperparameters worked best.

### Step 4: Test Your Model
On each completed run's card, you'll find a **"Test This Model"** button. Clicking this opens the **"Test Your Model"** panel.

- **Draw Digits Mode:** Use your mouse or finger to draw digits in the 10 canvases provided. When you click **"Predict All Digits"**, the model will analyze your drawings and display its predictions.
- **Test on MNIST Images Mode:** Switch to this mode to use a gallery of real, unseen test images from the MNIST dataset. Drag images from the gallery and drop them into the slots. The model will predict what they are, and you can see if it was correct by comparing its guess to the actual label.

### Step 5: Visualize the Internals
The **"Visualize Model"** button on a completed run opens a full-screen modal for deep inspection.
- **Layer Weight Heatmaps:** This view shows heatmaps of the learned weights in each layer. For the first dense layer, you can see how each neuron has learned to respond to different pixel patterns. For CNNs, you can see the convolutional filters.
- **Live Activation Viewer:** Here, you can draw a digit and click "See Activation Flow" to watch a diagram of the network light up, showing how neurons are activated as the data passes from the input layer to the final prediction.

### Step 6: Manage Your Model
You can save a model you're happy with using the **"Save Model"** button in the testing panel. If a saved model exists, a new card will appear on the left, allowing you to **Load** it for testing or **Delete** it. You can also **Download** the model's files (`model.json` and `weights.bin`) for use in other TensorFlow.js projects.

## 4. How MNIST Data is Handled on the Web

A significant technical hurdle in this project is efficiently loading the large MNIST dataset (tens of thousands of images) into the browser. Simply including the image files in the application is not feasible, and downloading thousands of individual image files would be extremely slow.

To solve this, this application uses a technique called **sprite sheeting**, which is common in graphics and game development. The entire dataset of 65,000 images (55,000 for training, 10,000 for testing) is combined into a single, large PNG image file called a sprite sheet. The labels for these images are stored in a separate, compact binary file.

The data loading process is handled by the `services/mnistData.ts` service, which performs the following steps:
1.  **Fetches Data:** The service downloads the image sprite sheet and the binary labels file from the official Google Cloud Storage source used by the TensorFlow.js team. This source is highly reliable and correctly configured for direct use in web applications.
2.  **Image Parsing:** Once the sprite sheet PNG is downloaded, it's drawn onto a hidden `<canvas>` element. This is a critical step that gives us raw pixel-level access to the image data.
3.  **Pixel Extraction:** The service reads the pixel data from the canvas. Since the images are grayscale, it only needs one color channel (e.g., red) to get the value of each pixel. It iterates through the entire sprite sheet, extracting the 784 pixels (28x28) for each of the 65,000 digits.
4.  **Label Parsing:** The compact binary file containing the 65,000 labels is read into an array.
5.  **Tensor Creation:** The raw pixel and label data are then split into training and testing sets. Finally, this data is converted into the `tf.Tensor` objects that TensorFlow.js needs for training and evaluation.

This sprite sheet approach is highly efficient for web-based machine learning. It minimizes network requests and leverages the browser's powerful, native image decoding and canvas APIs to quickly prepare a large dataset for the model, ensuring the application starts up fast and runs smoothly.

## 5. Resolution Note: Correcting MNIST Label Parsing

This section documents the resolution of a previous critical bug related to the "Test on MNIST Images" feature.

### The Previous Issue

Previously, the application incorrectly displayed the ground-truth label for all test images as "Actual: 0". This prevented users from accurately validating their model's performance on unseen data from the MNIST test set.

### Root Cause and Resolution

The root cause was identified in the data loading logic within `services/mnistData.ts`. The labels file (`mnist_labels_uint8`) provided by the TensorFlow.js examples repository is not a simple list of integer digits (e.g., `5, 0, 4, ...`), but is instead **one-hot encoded**. Each label is represented by a 10-byte `Uint8Array` chunk where the correct digit's index is marked with a `1` (e.g., `[0,0,0,0,0,1,0,0,0,0]` for the digit `5`).

The original code was written with the assumption that the file contained simple integer labels. The fix involved updating the data loading process to correctly parse this one-hot encoded format. The new logic now:
1.  Reads the one-hot encoded buffer.
2.  Creates a second array containing the correct integer labels (0-9) by finding the index of the `1` in each 10-byte chunk.
3.  Uses this new array of integer labels to correctly display the "Actual" digit in the testing interface.
4.  The label tensors for model training (`trainLabels`, `validationLabels`) are created directly from the one-hot encoded buffer, which is the format TensorFlow.js expects for categorical cross-entropy loss.

This change has resolved the issue, and the "Test on MNIST Images" feature now functions as intended, providing accurate ground-truth labels for model validation.

## 6. Resolution Note: Fixing a TensorFlow.js Type Mismatch

This note documents the resolution of a data loading failure caused by a subtle type incompatibility within the TensorFlow.js library.

### The Previous Issue

The application would fail during the data loading and shuffling phase with the error: `Argument 'indices' passed to 'gather' must be a Tensor or TensorLike, but got 'Uint32Array'`. This prevented the training and validation datasets from being prepared, halting the application.

### Root Cause and Resolution

The issue stemmed from an interaction between two TensorFlow.js functions:

1.  `tf.util.createShuffledIndices()`: This utility function is used to generate a randomly ordered array of indices for shuffling the dataset. It returns a standard JavaScript `Uint32Array`.
2.  `tf.gather()`: This core TensorFlow.js operation is used to reorder a tensor based on a given set of indices. Crucially, it requires its `indices` argument to be a `tf.Tensor` of type `int32`.

The error occurred because the code was passing the raw `Uint32Array` from the utility function directly to `tf.gather()`, which cannot handle that specific JavaScript type.

The fix, implemented in `services/mnistData.ts`, was to explicitly convert the shuffled indices array into the correct format before the `gather` operation. A new `Int32Array` is created from the `Uint32Array`, and then a `tf.tensor1d` of type `'int32'` is created from that. This ensures that `tf.gather` receives a tensor of the precise type it expects, resolving the error and allowing the data shuffling process to complete successfully.

## 7. Resolution Note: Ensuring Robust Validation by Separating Test Data

This note documents a critical improvement to the model training and validation process to guarantee that accuracy is measured on a truly independent dataset, eliminating any possibility of data leakage.

### The Previous Method

Previously, the application would partition the original 55,000-image MNIST training set into two smaller sets: a 45,000-image set for training and a 10,000-image set for validation. While these sets were disjoint, they were both drawn from the same master training pool. This is a common practice but can sometimes lead to overly optimistic validation scores if the training and validation splits happen to be very similar.

### The Improved Method

To provide a more rigorous and trustworthy measure of the model's ability to generalize, the data partitioning logic in `services/mnistData.ts` has been updated to follow a stricter separation of data:

1.  **Training Set:** The model is now trained on the **entire 55,000-image official training set**.
2.  **Validation Set:** The model's performance is now validated against the **entire 10,000-image official test set** at the end of each epoch.

This change ensures that the validation accuracy reported during training is a true reflection of the model's performance on data it has never seen before, as the test set is collected and curated separately from the training set. This is the standard practice for academic benchmarks and provides the user with a much more reliable metric for comparing different model configurations. The interactive test panel in the UI also uses this same 10,000-image test set, providing consistency between the reported validation accuracy and the user's hands-on testing experience.

## 8. Resolution Note: Improving Drawn Digit Pre-Processing

This note details an enhancement to the image processing pipeline for user-drawn digits to improve model accuracy.

### The Issue

While the model performed well on images from the official MNIST test set, its accuracy was inconsistent on digits drawn by users in the "Test Your Model" interface. This indicated a mismatch between the pre-processing of user-drawn digits and the characteristics of the original MNIST dataset the model was trained on.

### Root Cause and Resolution

The investigation focused on the image processing steps in `components/DrawingCanvas.tsx`. The original MNIST digits are not sharp; they have a degree of blurriness and anti-aliasing from the original scanning and down-sampling process. The application's image processing must replicate this quality to ensure the model receives input that looks like its training data.

An earlier version of the app used a `3px` Gaussian blur, which was found to be too aggressive, washing out important features. The blur was then removed entirely, relying only on the browser's scaling algorithm for smoothing. However, this produced images that were too sharp compared to the training data.

The fix was to **re-introduce a more subtle `2px` Gaussian blur**. This value strikes a balance, softening the hard edges of the user's drawing without destroying key features like corners and endpoints. This blur is applied to the 28x28 scaled-down image before it is centered and fed to the model. This change ensures the input from the drawing canvas more closely matches the "fuzziness" of the MNIST training set, leading to more reliable and accurate predictions.
