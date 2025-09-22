
import React from 'react';

export const LayersInfo: React.FC = () => (
    <>
        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">What are Hidden Layers?</h3>
        <p>These are the layers between the input (the 784 pixels of the image) and the output (the 10 possible digits). This is where the "thinking" happens. The model learns to recognize patterns in these layers, and more layers can learn more complex patterns.</p>

        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">Units (Neurons)</h3>
        <p>Each "unit" is a neuron. It receives inputs from the previous layer, performs a calculation, and passes its output to the next layer. The number of units determines the "width" of a layer.</p>
        <ul>
            <li><strong>More Units:</strong> Can learn more features at that layer.
                <ul>
                    <li><strong>Pro:</strong> Increases the model's "capacity" to learn complex patterns.</li>
                    <li><strong>Con:</strong> More computationally expensive, and a higher risk of <strong>overfitting</strong> (memorizing the training data instead of learning general patterns).</li>
                </ul>
            </li>
            <li><strong>Fewer Units:</strong> A simpler model.
                <ul>
                    <li><strong>Pro:</strong> Faster to train and a lower risk of overfitting.</li>
                    <li><strong>Con:</strong> May not be powerful enough to learn the task, leading to <strong>underfitting</strong>.</li>
                </ul>
            </li>
        </ul>
        <p>A common practice is to have more units in the earlier layers and gradually decrease the number in deeper layers, like a funnel, to distill information down to the final prediction.</p>

        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">Activation Functions</h3>
        <p>An activation function decides whether a neuron should be "activated" or not. It introduces essential non-linearity into the model, allowing it to learn complex data patterns. Without it, a deep neural network would behave just like a simple linear model.</p>
        <ul>
            <li><strong>ReLU (Rectified Linear Unit):</strong> The most common choice. It's simple and efficient. It outputs the input directly if it's positive; otherwise, it outputs zero. (<code>f(x) = max(0, x)</code>).</li>
            <li><strong>Sigmoid:</strong> Squeezes numbers into a range between 0 and 1. It was popular in the past but is less common in hidden layers now, partly due to the "vanishing gradient" problem.</li>
            <li><strong>Tanh (Hyperbolic Tangent):</strong> Similar to sigmoid, but squeezes values into a range between -1 and 1. It is zero-centered and often performs better than sigmoid in hidden layers.</li>
        </ul>
    </>
);
