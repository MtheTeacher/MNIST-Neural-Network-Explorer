# About the MNIST Neural Network Explorer

## 1. Introduction

Welcome to the MNIST Neural Network Explorer! This is an interactive, web-based application designed to demystify the process of building, training, and testing neural networks. It provides a hands-on environment where students, developers, and machine learning enthusiasts can experiment with one of the most classic problems in the field: handwritten digit recognition using the MNIST dataset.

The core purpose of this application is to make the concepts of neural networks tangible. Instead of just reading about architectures, hyperparameters, and training loops, you can actively build a model, watch it learn in real-time, and immediately test its performance on your own handwriting.

## 2. Core Features

- **Interactive Model Configuration:** Visually add or remove layers, set the number of neurons, and choose activation functions.
- **Advanced Hyperparameter Tuning:** Adjust learning rates, select from various learning rate schedules, set epoch counts and batch sizes, and apply regularization like Dropout.
- **Real-Time Training Visualization:** Watch your model's accuracy and loss evolve epoch-by-epoch with dynamic, easy-to-read charts.
- **Run Comparison:** Train multiple models with different configurations and compare their performance side-by-side.
- **Pruning & Fine-Tuning:** Take a large, trained model and make it smaller and more efficient through weight pruning, then fine-tune it to recover accuracy.
- **Two Modes of Interactive Testing:**
    1.  **Draw Your Own:** Test your model's capabilities on digits you draw yourself.
    2.  **MNIST Test Set:** Validate your model against unseen images from the official MNIST test dataset using a simple drag-and-drop interface.
- **Deep Model Inspection:** Dive into the model's internals with two powerful visualization tools:
    1.  **Weight Heatmaps:** See what the learned features inside your model's layers actually look like.
    2.  **Live Activation Flow:** Draw a digit and watch the neuron activations propagate through the network in real-time.
- **In-Browser Model Management:** Save your trained models to your browser's local storage, load them back later, or download the model files to use elsewhere. All training and inference happen directly in your browser using TensorFlow.js, with no server-side computation required.

## 3. Exploring Modern Training Techniques

This application demonstrates key principles from modern machine learning. A core insight in modern ML is that it's often better to **start with a larger, more powerful model than you need, train it intelligently, and then make it smaller and more efficient**. This "start big → train smart → prune" approach can often lead to a final model that is both smaller and more accurate than a model that was designed to be small from the beginning.

### Stage 1: Train Smart

The first step is to train a large, over-parameterized model effectively.

- **Dropout (Regularization):** When a model is too powerful, it can "memorize" training examples instead of learning general patterns (overfitting). Dropout combats this by randomly "dropping" neurons during training, forcing the network to learn more robust features. You can control this with the **Dropout Rate** slider.
- **Advanced Learning Rate Schedules:** Finding the best solution is tricky. The training process can get stuck in a "good" but not "great" solution. A schedule like **Cosine Annealing with Restarts** helps the model escape these ruts by periodically resetting the learning rate, allowing it to explore more of the solution space and find a better final result.

### Stage 2: Prune & Fine-Tune (New!)

After you have a well-trained large model, you can make it more efficient through **pruning**.

- **What is Pruning?** Pruning is the process of removing unnecessary connections (weights) from a neural network. After training, many weights are often very close to zero and contribute very little to the model's predictions. **Magnitude Pruning**, which is used in this app, identifies these low-magnitude weights and permanently sets them to zero, effectively removing them.
- **Why Prune?** The goal is to create a "sparse" model—one with many zero-value weights. This makes the model smaller (less storage) and faster (fewer calculations during inference), which is critical for running on devices like mobile phones.
- **Fine-Tuning:** After pruning, the model's accuracy will drop slightly. **Fine-tuning** is a short, subsequent training run (usually with a very low learning rate) that allows the remaining weights to adjust and compensate for the connections that were removed, often recovering most, if not all, of the original accuracy.

This app allows you to perform this entire process and then compare your new, efficient pruned model against its larger parent and against a smaller model trained from scratch. You can often prove that the **large-then-pruned model is more accurate** than a small model with the same final number of parameters.

## 4. How to Use the Application: A Walkthrough

### Step 1: Configure Your Model
Use the **"Configure Model"** panel to design your network. To experiment with pruning, it's best to start with a "Deep" or "Wide" preset to create a large, over-parameterized model.

### Step 2: Train the Model
Click **"Start Training"** and monitor the live dashboards as your model learns.

### Step 3: Prune and Fine-Tune Your Trained Model
Once a run is complete, a **"Prune & Fine-Tune"** button will appear on its card.
1.  Click it to open the Pruning Modal.
2.  Use the **"Target Sparsity"** slider to choose what percentage of the model's weights you want to remove. Watch how the total parameter count decreases.
3.  Click **"Start Fine-Tuning"**. The app will create the new sparse model and run a short, automated fine-tuning process. A new "Pruned Model" card will appear in the comparison list.

### Step 4: Analyze and Compare Runs
You can now compare your runs side-by-side. A great experiment is to compare:
1.  A "Simple" model trained from scratch.
2.  A "Wide" model trained from scratch.
3.  The pruned version of the "Wide" model, with a final parameter count similar to the "Simple" one.

Observe which model achieves the highest accuracy for a given parameter budget.

### Step 5: Test and Visualize
Use the **"Test This Model"** and **"Visualize Model"** buttons on any completed run—including your new pruned ones—to see how they perform and what their internal structure looks like.

### Step 6: Manage Your Model
Save, load, or download any model you're happy with, including your highly efficient pruned models.

## 5. How MNIST Data is Handled on the Web

A significant technical hurdle in web-based machine learning is reliably loading the large MNIST dataset. Previous versions of this app fetched the data from external servers, which led to intermittent "Failed to fetch" errors due to network issues.

To solve this permanently, the application now uses a more robust, self-contained approach.

1.  **Embedded Data Source:** Instead of fetching from a URL, the application now has the locations of the dataset files embedded directly in its code. This points to the same reliable Google Cloud Storage source used by official TensorFlow.js examples.
2.  **Efficient Loading:** The app downloads a single large PNG image file (a "sprite sheet") containing all 65,000 digits and a compact binary file for the labels.
3.  **In-Browser Processing:** The sprite sheet is drawn to a hidden canvas, where the individual 28x28 pixel digits are extracted. This data is then converted into `tf.Tensor` objects for training.

This method minimizes network requests and leverages the browser's powerful image processing capabilities. By pointing to a stable, canonical data source, it ensures the application is significantly more reliable and resilient to the network problems that were causing failures before.

## 6. Resolution Note: Correcting MNIST Label Parsing

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

## 7. Resolution Note: Fixing a TensorFlow.js Type Mismatch

This note documents the resolution of a data loading failure caused by a subtle type incompatibility within the TensorFlow.js library.

### The Previous Issue

The application would fail during the data loading and shuffling phase with the error: `Argument 'indices' passed to 'gather' must be a Tensor or TensorLike, but got 'Uint32Array'`. This prevented the training and validation datasets from being prepared, halting the application.

### Root Cause and Resolution

The issue stemmed from an interaction between two TensorFlow.js functions:

1.  `tf.util.createShuffledIndices()`: This utility function is used to generate a randomly ordered array of indices for shuffling the dataset. It returns a standard JavaScript `Uint32Array`.
2.  `tf.gather()`: This core TensorFlow.js operation is used to reorder a tensor based on a given set of indices. Crucially, it requires its `indices` argument to be a `tf.Tensor` of type `int32`.

The error occurred because the code was passing the raw `Uint32Array` from the utility function directly to `tf.gather()`, which cannot handle that specific JavaScript type.

The fix, implemented in `services/mnistData.ts`, was to explicitly convert the shuffled indices array into the correct format before the `gather` operation. A new `Int32Array` is created from the `Uint32Array`, and then a `tf.tensor1d` of type `'int32'` is created from that. This ensures that `tf.gather` receives a tensor of the precise type it expects, resolving the error and allowing the data shuffling process to complete successfully.

## 8. Resolution Note: Ensuring Robust Validation by Separating Test Data

This note documents a critical improvement to the model training and validation process to guarantee that accuracy is measured on a truly independent dataset, eliminating any possibility of data leakage.

### The Previous Method

Previously, the application would partition the original 55,000-image MNIST training set into two smaller sets: a 45,000-image set for training and a 10,000-image set for validation. While these sets were disjoint, they were both drawn from the same master training pool. This is a common practice but can sometimes lead to overly optimistic validation scores if the training and validation splits happen to be very similar.

### The Improved Method

To provide a more rigorous and trustworthy measure of the model's ability to generalize, the data partitioning logic in `services/mnistData.ts` has been updated to follow a stricter separation of data:

1.  **Training Set:** The model is now trained on the **entire 55,000-image official training set**.
2.  **Validation Set:** The model's performance is now validated against the **entire 10,000-image official test set** at the end of each epoch.

This change ensures that the validation accuracy reported during training is a true reflection of the model's performance on data it has never seen before, as the test set is collected and curated separately from the training set. This is the standard practice for academic benchmarks and provides the user with a much more reliable metric for comparing different model configurations. The interactive test panel in the UI also uses this same 10,000-image test set, providing consistency between the reported validation accuracy and the user's hands-on testing experience.

## 9. Resolution Note: Improving Drawn Digit Pre-Processing

This note details an enhancement to the image processing pipeline for user-drawn digits to improve model accuracy.

### The Issue

While the model performed well on images from the official MNIST test set, its accuracy was inconsistent on digits drawn by users in the "Test Your Model" interface. This indicated a mismatch between the pre-processing of user-drawn digits and the characteristics of the original MNIST dataset the model was trained on.

### Root Cause and Resolution

The investigation focused on the image processing steps in `components/DrawingCanvas.tsx`. The original MNIST digits are not sharp; they have a degree of blurriness and anti-aliasing from the original scanning and down-sampling process. The application's image processing must replicate this quality to ensure the model receives input that looks like its training data.

An earlier version of the app used a `3px` Gaussian blur, which was found to be too aggressive, washing out important features. The blur was then removed entirely, relying only on the browser's scaling algorithm for smoothing. However, this produced images that were too sharp compared to the training data.

The fix was to **re-introduce a more subtle `2px` Gaussian blur**. This value strikes a balance, softening the hard edges of the user's drawing without destroying key features like corners and endpoints. This blur is applied to the 28x28 scaled-down image before it is centered and fed to the model. This change ensures the input from the drawing canvas more closely matches the "fuzziness" of the MNIST training set, leading to more reliable and accurate predictions.

## 10. Resolution Note: Fixing the Model Pruning Workflow

This section documents the diagnosis and resolution of a critical bug that caused the "Prune & Fine-Tune" feature to fail consistently.

### The Issue: Metadata Loss During State Transitions

The pruning workflow would abort with a `Cannot read properties of undefined (reading 'length')` error. The root cause was a subtle but critical interaction between TensorFlow.js and React's state management:

1.  **React State Strips Metadata:** After a model was trained, the `tf.Sequential` object was stored in React state. React's development mode and serialization processes would strip non-enumerable properties from this object to make it serializable. Unfortunately, TensorFlow.js stores critical graph information, including the model's input shape (`model.inputs`), on these non-enumerable properties.
2.  **Loss of Input Shape:** By the time the model object was passed to the `pruneModel` function, its `model.inputs` property was `undefined`.
3.  **Reconstruction Failure:** The pruning function needed this input shape to "build" a new, empty model architecture before it could be populated with the pruned weights. Without the shape, the `build()` call failed, leading to the crash.

### The Solution: Pre-emptive Metadata Capture and Robust Reconstruction

A two-part fix was implemented to make the pruning pipeline robust against this metadata loss:

1.  **Capture Metadata Immediately After Training:** In `App.tsx`, immediately after the `model.fit()` call resolves, the application now captures two key pieces of metadata *before* the model is stored in React state:
    *   The **input shape**, read directly from the live model's `model.inputs[0].shape`.
    *   The full **model architecture as JSON**, using `model.toJSON()`.
    This information is stored alongside the model in the `TrainingRun` object, ensuring it survives React's state serialization.

2.  **Reconstruct from JSON:** The `pruneModel` function in `services/modelService.ts` was refactored. Instead of trying to rebuild the model from a configuration object and a lost input shape, it now uses the much more reliable `tf.loadLayersModel` with the saved model JSON. This method reconstructs a complete, "built" model with the correct input signature every time.

3.  **Corrected Memory Management:** A secondary bug was also fixed in `pruneModel`. The use of `tf.tidy()` was incorrect and caused the newly created pruned weight tensors to be disposed of before they could be set on the new model. The logic was corrected to use `tf.keep()` to explicitly preserve these tensors through the tidy scope, ensuring the final pruned model receives its weights correctly.

By persisting the model's structural metadata before React can strip it and by using a more robust method for model reconstruction, the pruning pipeline is now stable and functions as intended.

## 11. Troubleshooting Note: Resolving CORS and Module Loading Failures

This note documents the resolution of a critical loading failure that prevented the application from starting, caused by Cross-Origin Resource Sharing (CORS) errors.

### The Issue

The application would fail to load entirely, with browser developer tools showing numerous CORS errors. These errors indicated that requests for core JavaScript modules (like React, ReactDOM, and TensorFlow.js) were being fetched from a different origin (`https://aistudiocdn.com`) than the one serving the application, and the remote server did not permit this.

### Root Cause and Resolution

The root cause was an `<script type="importmap">` block in `index.html`. This "import map" was instructing the browser to ignore the locally bundled versions of dependencies and instead fetch them directly from an external Content Delivery Network (CDN). While useful in some contexts, this created a cross-origin conflict within the development environment.

The solution was to **remove the entire `<script type="importmap">` block from `index.html`**. This allows the Vite build tool to function as intended:
1.  Vite analyzes the `import` statements in the TypeScript/JavaScript code.
2.  It finds the required packages (React, TensorFlow.js, etc.) in the local `node_modules` directory.
3.  It bundles these dependencies and serves them from the same origin as the application itself.

By letting Vite manage the dependencies, all cross-origin requests for these modules are eliminated, resolving the CORS errors and allowing the application to load correctly. This change makes the application more robust, self-contained, and aligned with modern web development best practices.

## 12. Resolution Note: Re-addressing CORS Loading Failures

This note documents the resolution of a critical application loading failure.

### The Issue

The application failed to start, presenting a large number of Cross-Origin Resource Sharing (CORS) errors in the browser's developer console. The errors indicated that the browser was blocked from fetching essential JavaScript libraries like TensorFlow.js from an external CDN (`https://aistudiocdn.com`) because the server did not provide the necessary `Access-Control-Allow-Origin` header.

### Root Cause and Resolution

The investigation confirmed that the root cause was an `<script type="importmap">` tag within `index.html`. This tag overrode the standard module resolution process, forcing the browser to fetch dependencies from an external, cross-origin source, which triggered the browser's security policies.

This issue was previously identified and documented (see Section 11). The re-emergence of the problem indicates that the problematic `importmap` was re-introduced into `index.html`.

The fix, consistent with the previous resolution, was to **remove the entire `<script type="importmap">` block from `index.html`**. This action restores the default behavior, allowing the local development server and build tool (Vite) to bundle all necessary dependencies and serve them from the same origin as the application. This completely eliminates the cross-origin requests and resolves the CORS errors, allowing the application to load and function correctly.

## 13. Resolution Note: Third Resolution of CORS Loading Failures

This note documents the resolution of a recurring critical application loading failure.

### The Issue

For the third time, the application failed to start, with the browser's developer console showing a cascade of `net::ERR_FAILED` and CORS (Cross-Origin Resource Sharing) errors. These errors confirmed that the browser was being blocked from fetching essential JavaScript libraries (React, TensorFlow.js, etc.) from an external CDN (`https://aistudiocdn.com`).

### Root Cause and Resolution

As documented in Sections 11 and 12, the root cause was the re-introduction of an `<script type="importmap">` tag in `index.html`. This tag forces the browser to fetch dependencies from an external, cross-origin source, which is blocked by browser security policies in this environment.

The re-emergence of this issue suggests a problem in the development or deployment workflow that is re-introducing this problematic configuration.

The fix, consistent with previous resolutions, was to once again **remove the entire `<script type="importmap">` block from `index.html`**. This action ensures that the local development server (Vite) correctly bundles all dependencies and serves them from the same origin as the application, eliminating the CORS errors and allowing the application to load. This change is critical for application stability and self-containment.