
import React from 'react';

export const EpochsBatchSizeInfo: React.FC = () => (
    <>
        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">What is an Epoch?</h3>
        <p>One epoch is one complete pass through the <em>entire</em> training dataset. For MNIST, one epoch means the model has seen all 55,000 training images once.</p>
        <ul>
            <li><strong>Too Few Epochs:</strong> The model may not have had enough time to learn the patterns in the data. This is called <strong>underfitting</strong>. Its accuracy will be low on both training and test data.</li>
            <li><strong>Too Many Epochs:</strong> The model might start to "memorize" the training data, including its noise and quirks, instead of learning the general digit shapes. This will make it perform poorly on new, unseen data. This is called <strong>overfitting</strong>. You can often see this when the training accuracy keeps going up, but the validation accuracy flattens or starts to decrease.</li>
        </ul>

        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">What is a Batch Size?</h3>
        <p>The model doesn't process the entire dataset at once. Instead, it processes it in small chunks called "batches". The batch size is the number of training samples in one batch. The model's weights are updated after each batch is processed.</p>
        <p>Total iterations per epoch = (Total Training Samples) / (Batch Size)</p>
        <ul>
            <li><strong>Small Batch Size (e.g., 32):</strong>
                <ul>
                    <li><strong>Pro:</strong> Less memory required. The weight updates are frequent and noisy, which can sometimes help the model escape suboptimal solutions and generalize better.</li>
                    <li><strong>Con:</strong> Training can be slow because there are many updates per epoch.</li>
                </ul>
            </li>
            <li><strong>Large Batch Size (e.g., 512, 1024):</strong>
                <ul>
                    <li><strong>Pro:</strong> Faster training as the computations are more efficient on modern hardware (like GPUs). The loss decrease is more stable.</li>
                    <li><strong>Con:</strong> Requires more memory. Can sometimes converge to solutions that don't generalize as well as those found with smaller batch sizes.</li>
                </ul>
            </li>
        </ul>
    </>
);
