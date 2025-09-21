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
1.  **Fetches Data:** It simultaneously downloads the single image sprite sheet and the binary labels file from a highly available Google Cloud Storage bucket.
2.  **Image Parsing:** Once the sprite sheet PNG is downloaded, it's drawn onto a hidden `<canvas>` element. This is a critical step that gives us raw pixel-level access to the image data.
3.  **Pixel Extraction:** The service reads the pixel data from the canvas. Since the images are grayscale, it only needs one color channel (e.g., red) to get the value of each pixel. It iterates through the entire sprite sheet, extracting the 784 pixels (28x28) for each of the 65,000 digits.
4.  **Label Parsing:** The compact binary file containing the 65,000 labels is read into an array.
5.  **Tensor Creation:** The raw pixel and label data are then split into training and testing sets. Finally, this data is converted into the `tf.Tensor` objects that TensorFlow.js needs for training and evaluation.

This sprite sheet approach is highly efficient for web-based machine learning. It minimizes network requests and leverages the browser's powerful, native image decoding and canvas APIs to quickly prepare a large dataset for the model, ensuring the application starts up fast and runs smoothly.

## 5. A Known Issue: Incorrect "Actual" Labels in MNIST Test Mode

We are currently investigating a frustrating bug within the "Test Your Model" panel and would appreciate any insights from the community.

### The Intended Behavior

The "Test on MNIST Images" mode is designed to be a crucial validation step. A user should be able to drag any digit image from the gallery and drop it into one of the ten test slots. After clicking "Predict All Digits," the UI should display the model's prediction alongside the true, ground-truth label of that image. For example, if the user drags an image of a "7", the interface should show something like "Prediction: 7 (Actual: 7)". This allows for a direct and accurate assessment of the model's performance on unseen data.

### The Problem

Currently, the feature is not working as intended. While the model provides a prediction, the ground-truth label displayed for any dragged image is **always "Actual: 0"**. This is incorrect and misleading, as it prevents users from knowing if their model's predictions are right or wrong for any digit other than zero.

### Our Working Theories

We have a few theories as to why this might be happening, primarily centered around how the data is being loaded and processed.

1.  **Primary Theory: Label File Header Issue.** Our leading hypothesis is that there's an error in how we parse the binary file containing the MNIST labels (`mnist_labels_uint8`). Many raw data files, including this one, contain a metadata "header" at the beginning of the file that needs to be skipped. Our current data loading logic in `services/mnistData.ts` reads the file from the very first byte. If a header exists, we are misinterpreting that metadata as label data. When we later slice the data array to get the 10,000 test labels, we could be pointing to a section of the buffer that is either part of the header or an incorrect offset, which happens to contain zero values.

2.  **Secondary Theory: Incorrect Data Slicing.** It's also possible that the logic for splitting the full 65,000-item dataset into the 55,000 training examples and 10,000 test examples is flawed. An "off-by-one" error or an incorrect starting index for the slice that extracts the test labels could cause us to read the wrong data segment.

3.  **Less Likely Theory: State Management Bug.** While less probable, there could be an issue in the React component (`InferenceTester.tsx`) itself. The logic that transfers the image's metadata (including its true label) during the drag-and-drop operation, or the state update that follows, might be losing or corrupting the label information. However, the consistency of the bug (always displaying "0") strongly suggests a more fundamental, systemic issue with the source data rather than a flaw in the UI logic.

We are actively working to resolve this issue and any help or suggestions would be greatly appreciated.
