
import React from 'react';

export const ArchitectureInfo: React.FC = () => (
    <>
        <p>The <strong>architecture</strong> (or topology) of a neural network is how its layers are structured. It's the blueprint of your model, defining the path that data takes from input to output.</p>

        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">Dense (Fully-Connected) Networks</h3>
        <p>In a dense network, each neuron in a layer is connected to <em>every</em> neuron in the next layer. It's a versatile architecture for many types of data.</p>
        <ul>
            <li><strong>Simple:</strong> A good baseline with one hidden layer. It's surprisingly effective for many problems and is a great starting point.</li>
            <li><strong>Deep:</strong> Multiple hidden layers allow the model to learn more complex, hierarchical features. For example, the first layer might learn edges, the second shapes, and the third parts of digits.
                <ul>
                    <li><strong>Pro:</strong> Can be very powerful and achieve high accuracy.</li>
                    <li><strong>Con:</strong> Harder and slower to train, and has a higher risk of "overfitting" (memorizing the training data).</li>
                </ul>
            </li>
            <li><strong>Wide:</strong> A single hidden layer with many neurons. This allows the model to learn many different features in parallel.
                <ul>
                    <li><strong>Pro:</strong> Can be very effective and is often faster to train than a deep model.</li>
                    <li><strong>Con:</strong> May not capture hierarchical relationships between features as well as a deep network.</li>
                </ul>
            </li>
        </ul>

        <h3 className="text-xl font-bold mt-4 mb-2 text-pink-300">Convolutional Neural Networks (CNN)</h3>
        <p>The "High-Accuracy CNN" preset uses a specialized architecture that is extremely effective for image data. Instead of looking at all pixels at once, CNNs use "filters" to scan over the image and detect specific low-level patterns like edges, curves, and textures. Subsequent layers combine these patterns into more complex shapes.</p>
        <ul>
            <li><strong>Pro:</strong> State-of-the-art performance on image tasks. They are more parameter-efficient than dense networks and are naturally "translation invariant" (they don't care where in the image a feature is located).</li>
            <li><strong>Con:</strong> More conceptually complex to understand and configure from scratch.</li>
        </ul>
    </>
);
