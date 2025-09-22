
import React from 'react';

export const LearningRateInfo: React.FC = () => (
    <>
        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">What is a Learning Rate?</h3>
        <p>The learning rate is arguably the most important hyperparameter to tune. It controls how much the model's internal weights are adjusted with respect to the calculated error. Think of it as the <strong>step size</strong> the model takes on its path to finding the best solution.</p>
        <ul>
            <li><strong>High Learning Rate:</strong> The model learns quickly with large steps.
                <ul>
                    <li><strong>Pro:</strong> Faster training time.</li>
                    <li><strong>Con:</strong> Can easily overshoot the optimal solution, causing the loss to jump around erratically and fail to converge.</li>
                </ul>
            </li>
            <li><strong>Low Learning Rate:</strong> The model learns slowly with tiny, careful steps.
                <ul>
                    <li><strong>Pro:</strong> More likely to find a good solution.</li>
                    <li><strong>Con:</strong> Training takes a very long time, and the model might get stuck in a suboptimal solution (a "local minimum").</li>
                </ul>
            </li>
        </ul>

        <h3 className="text-xl font-bold mt-4 mb-2 text-cyan-300">Learning Rate Schedules</h3>
        <p>Instead of using one fixed learning rate, it's often better to change it during training. This is what a "schedule" does. The common strategy is to start with a larger learning rate to make big progress quickly, and then decrease it as training progresses to make finer, more precise adjustments.</p>
        <ul>
            <li><strong>Constant:</strong> The LR never changes. Simple, but usually not the best performing.</li>
            <li><strong>Decay (Step, Exponential):</strong> Gradually lowers the LR after a set number of epochs. A reliable and common strategy.</li>
            <li><strong>Cosine Annealing:</strong> A popular and effective schedule that smoothly decreases the LR following the shape of a cosine curve.</li>
            <li><strong>Warm-up & Cosine:</strong> Starts with a very low LR, "warms up" to the maximum LR over a few epochs, and then decays. This can help stabilize training in the beginning, especially for very deep models.</li>
            <li><strong>One-Cycle:</strong> An advanced policy that goes from a low LR up to the max and back down to a very low LR, all within the training run. It can lead to faster training and better performance.</li>
        </ul>
    </>
);
