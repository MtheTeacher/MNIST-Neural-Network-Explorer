
# About the MNIST Neural Network Explorer

## Overview
An interactive web application for learning neural networks using the MNIST dataset.

## Core Features
- Custom Dense and CNN architecture configuration.
- Real-time training visualization with Recharts.
- Magnitude-based model pruning and fine-tuning.
- Interactive gradient descent sandbox (WaveScape).

## Resolution Note: Live Dashboard & Chart Visibility (v2.1)

### The Issue
Graphs were not appearing immediately or were perceived as missing during the active training phase. Metrics for training and validation sets were also appearing identical.

### The Fix
1. **State Eagerness**: `setIsTraining` is now triggered immediately upon the "Start Training" click, ensuring the `TrainingDashboard` component is mounted before any async model preparation.
2. **Axis Pre-rendering**: Updated Recharts configurations in `TrainingDashboard` to render axes even with empty data arrays. This provides a visual "staging area" for the incoming data.
3. **Smooth State Transfers**: Transitioned the training log updates to use functional state updates and reference cloning (`[...]`) to ensure React consistently detects and renders the data arriving from the TensorFlow.js worker thread.
4. **Metric Integrity**: Explicitly split "Train Accuracy" and "Validation Accuracy" logic in the UI badges to provide a true reflection of the model's generalization capabilities.
5. **UI Thread Yielding**: Implemented `requestAnimationFrame` yielding within the training loop to ensure chart re-renders are processed smoothly without blocking the background training.
