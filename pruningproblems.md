# Analysis of Model Pruning Failures

This document details a persistent and critical bug within the "Prune & Fine-Tune" functionality of the MNIST Neural Network Explorer. The goal is to provide a comprehensive overview for an external consultant to aid in troubleshooting.

## 1. Objective: Implementing Model Pruning and Fine-Tuning

The application aims to demonstrate a modern machine learning workflow: training a large, potentially over-parameterized model and then making it smaller and more efficient through **magnitude-based weight pruning**.

The intended user flow is as follows:
1.  A user configures and trains a neural network.
2.  Upon completion, the user can select this trained "parent" model for pruning.
3.  The user specifies a "target sparsity" (e.g., 90%), indicating the desired percentage of weights to be turned into zeros.
4.  The application then creates a new, sparse model which is significantly smaller and faster for inference.
5.  This new model undergoes a short "fine-tuning" training process with a low learning rate to recover any accuracy lost during pruning.
6.  The final, efficient model is available for testing and comparison.

This feature is critical for demonstrating how large, accurate models can be optimized for deployment on resource-constrained devices.

## 2. Current Implementation (`services/modelService.ts`)

The core logic resides in the `pruneModel` function. It attempts to perform the following steps:

1.  **Collect Weights:** It iterates through the layers of the trained parent model and collects all trainable weight tensors (kernels), ignoring biases.
2.  **Calculate Threshold:** It concatenates all weight values into a single tensor, takes their absolute values, and calculates the magnitude threshold required to achieve the target sparsity. This is done by finding the k-th smallest value, where `k` is the number of weights to keep.
3.  **Create Pruned Weights:** It creates a new set of weight tensors. For each kernel in the original model, it creates a binary mask by comparing weight magnitudes against the threshold. It then multiplies the original weights by this mask, effectively zeroing out any value below the threshold. Biases are cloned without modification.
4.  **Reconstruct Model Architecture:** This is the failing step. To create a new model to hold the pruned weights, the implementation does the following:
    a. Gets the configuration of the original model via `model.getConfig()`.
    b. Creates a new, empty `tf.sequential` model from this configuration.
    c. **Crucially, this new model is not yet "built".** It has no defined input or output shapes. Before weights can be set with `setWeights()`, the model must be built.
    d. To build the model, the code attempts to retrieve the input shape from the original, trained model.
    e. It then calls `prunedModel.build(inputShapeForBuild)` to finalize the new model's architecture.
5.  **Set Weights & Compile:** It calls `prunedModel.setWeights()` with the new sparse weights and compiles the model for fine-tuning.

## 3. The Persistent Error

The process consistently fails at **step 4e**. The application logs the following error:

```
Error: Pruning failed: Cannot read properties of undefined (reading 'length')
```

This error originates in the `pruneModel` function. It occurs when the code attempts to prepare the input shape for the `.build()` method. The specific line that fails is `batchInputShape.length`, which implies that `batchInputShape` is `undefined`.

`batchInputShape` is derived from `model.inputs[0].shape`. Therefore, the root cause is that `model.inputs[0]` or `model.inputs[0].shape` is `undefined` on the trained model object that is passed into the `pruneModel` function.

## 4. Diagnostic Steps and Failed Attempts

We have identified that the core of the issue is an inability to reliably retrieve the input shape from a model that has already been trained.

### Attempt 1: Using Layer Configuration (Avoided)

A common but brittle approach is to inspect the configuration of the first layer: `model.layers[0].getConfig().batchInputShape`. This was considered but avoided because this property is not guaranteed to be present in the configuration of all layer types (e.g., a `Reshape` layer, as used in our CNN, might not have it). This method is not robust.

### Attempt 2: Using `model.inputs` (Current Failing Method)

The current implementation uses `model.inputs[0].shape`. According to TensorFlow.js documentation and best practices, the `.inputs` property of a model is the canonical and most reliable source for its input shape definition **after the model has been built**.

A model is built implicitly when `.fit()` (training) is called. Since the models being pruned have all been successfully trained, the `.inputs` property **should be populated**. The fact that it is not is the central mystery of this bug.

## 5. Core Problem Hypothesis

The fundamental problem is that the `tf.Sequential` model object, despite having been successfully trained, appears to lose or not expose its input shape definition by the time it is passed to the `pruneModel` function.

Potential causes include:
1.  **Model State Preservation:** There might be an issue with how the model object is stored in React state and passed between components. It's possible that some non-enumerable or internal state related to the model's "built" status is being lost, although a simple reference pass should not cause this.
2.  **TensorFlow.js Memory Management:** Although the use of `tf.tidy()` appears correct (it should not dispose of the input `model`), there could be a more subtle memory management issue causing parts of the model's internal graph definition to be disposed of prematurely.
3.  **TensorFlow.js Version Subtlety:** It is possible that this is an undocumented behavior or a bug in the specific version of `@tensorflow/tfjs` being used, where the `.inputs` property is not consistently available on a trained `Sequential` model under all circumstances.

## 6. Request for Consultation

We have exhausted the standard, documented methods for model reconstruction. We are seeking expert advice to diagnose why a trained and functional `tf.Sequential` model object would have an undefined `inputs` property. We need to understand the exact conditions under which a model is considered "built" by TensorFlow.js and how to ensure that this state is preserved so that we can reliably clone its architecture for the pruning process.
