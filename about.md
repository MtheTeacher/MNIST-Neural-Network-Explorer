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

The data loading process is handled by our `services/mnistData.ts` service, which performs the following steps:
1.  **Fetches Data:** To ensure reliability, it attempts to download the single image sprite sheet and the binary labels file from a list of proven, highly-available sources, starting with the official Google Cloud Storage bucket used by the TensorFlow.js team. If the primary source fails, it automatically tries several backup mirrors.
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

The root cause was identified in the data loading logic within `services/mnistData.ts`. The labels file (`mnist_labels_uint8`) provided by the TensorFlow.js examples repository is not a simple list of integer digits (e.g., `5, 0, 4, ...`), but is instead **one-hot encoded**. Each label is represented by a 10-byte array where the correct digit's index is marked with a `1` (e.g., `[0,0,0,0,0,1,0,0,0,0]` for the digit `5`).

The original code was written with the assumption that the file contained simple integer labels. When retrieving a label for the UI, it was reading only the first byte of each 10-byte one-hot vector, which was almost always `0`.

The fix involved updating the data loading process to correctly parse this one-hot encoded format. The new logic now:
1.  Reads the one-hot encoded buffer.
2.  Creates a second array containing the correct integer labels (0-9) by finding the index of the `1` in each 10-byte chunk.
3.  Uses this new array of integer labels to correctly display the "Actual" digit in the testing interface.
4.  The label tensors for model training (`trainLabels`, `testLabels`) are now created directly from the one-hot encoded buffer, removing a redundant `tf.oneHot()` operation and improving efficiency.

This change has resolved the issue, and the "Test on MNIST Images" feature now functions as intended, providing accurate ground-truth labels for model validation.