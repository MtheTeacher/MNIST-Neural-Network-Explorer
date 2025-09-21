
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

## 4. The Challenge of Handling MNIST Data

A significant technical hurdle in this project, and for anyone working with the raw MNIST dataset, is correctly downloading, parsing, and using the data files.

The MNIST data is not provided in a standard image format like JPEG or PNG. Instead, it's stored in a custom binary format called `.idx3-ubyte` for images and `.idx1-ubyte` for labels. These files are essentially compressed archives that must be carefully parsed.

**The file structure consists of:**
1.  **A Magic Number:** A 32-bit integer at the very beginning of the file that identifies its contents (e.g., `2051` for images, `2049` for labels).
2.  **Metadata:** Additional 32-bit integers describing the data, such as the number of images, and their height and width.
3.  **The Data:** A flat, contiguous block of all the pixel values (for images) or label values (for labels) for the entire dataset.

We have faced challenges in the past ensuring the correct and robust parsing of these binary files. The data is stored in a **big-endian** byte order, which requires specific handling when reading the multi-byte integers in the header. An error in reading these headers—or a misalignment when pairing an image from the image file with its corresponding label from the label file—can be catastrophic for the model. It would lead to the model being trained on a dataset where the labels are mismatched with the images (e.g., an image of a '5' being labeled as a '2'). This would make it impossible for the model to learn correctly and would invalidate any accuracy measurements.

To solve this, we created the `services/mnistData.ts` service. This class acts as a singleton that encapsulates all the complex logic. It:
- Fetches the raw binary `.ubyte` files.
- Reads the headers using a `DataView` to correctly handle the big-endian byte format.
- Validates the magic numbers to ensure the files are not corrupted.
- Slices the raw byte buffers to separate the metadata from the pixel and label data.
- Constructs perfectly aligned pairs of images and labels.
- Finally, converts this data into the `tf.Tensor` objects that TensorFlow.js needs for training and evaluation.

This robust data-loading pipeline is crucial for the integrity of the entire application, as it guarantees that the model is learning from and being tested on a clean, correctly-labeled dataset.
