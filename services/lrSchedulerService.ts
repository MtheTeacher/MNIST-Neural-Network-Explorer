import type { LRSchedulerId } from '../types';

/**
 * Calculates the learning rate for a given epoch based on the selected schedule.
 */
export function getLearningRate(
    schedule: LRSchedulerId,
    epoch: number, // current epoch (0-indexed)
    totalEpochs: number,
    initialLr: number,
    valAccHistory: number[] = []
): number {
    const minLr = initialLr / 100;

    switch (schedule) {
        case 'constant':
            return initialLr;

        case 'step': {
            const dropRate = 0.5;
            const epochsPerDrop = Math.ceil(totalEpochs / 4);
            return initialLr * Math.pow(dropRate, Math.floor((epoch + 1) / epochsPerDrop));
        }

        case 'exponential': {
            const decayRate = 0.96;
            return initialLr * Math.pow(decayRate, epoch);
        }

        case 'cosine': {
            const decayEpochs = totalEpochs;
            const cosDecay = 0.5 * (1 + Math.cos(Math.PI * epoch / decayEpochs));
            return minLr + (initialLr - minLr) * cosDecay;
        }
        
        case 'cosine-restarts': {
            const numRestarts = 3;
            const cycleLength = Math.ceil(totalEpochs / numRestarts);
            const currentEpochInCycle = epoch % cycleLength;
            const cosDecay = 0.5 * (1 + Math.cos(Math.PI * currentEpochInCycle / cycleLength));
            return minLr + (initialLr - minLr) * cosDecay;
        }

        case 'warmup-cosine': {
            const warmupEpochs = Math.max(1, Math.floor(totalEpochs * 0.1));
            if (epoch < warmupEpochs) {
                return initialLr * (epoch + 1) / warmupEpochs;
            }
            const decayEpochs = totalEpochs - warmupEpochs;
            const currentDecayEpoch = epoch - warmupEpochs;
            const cosDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayEpoch / decayEpochs));
            return minLr + (initialLr - minLr) * cosDecay;
        }
        
        case 'one-cycle': {
            const peakEpoch = Math.floor(totalEpochs * 0.4);
            if (epoch < peakEpoch) {
                // Ramp up
                const cosAnn = 0.5 * (1 - Math.cos(Math.PI * epoch / peakEpoch));
                return minLr + (initialLr - minLr) * cosAnn;
            } else {
                // Ramp down
                const decayEpochs = totalEpochs - peakEpoch;
                const currentDecayEpoch = epoch - peakEpoch;
                const cosAnn = 0.5 * (1 + Math.cos(Math.PI * currentDecayEpoch / decayEpochs));
                return minLr + (initialLr - minLr) * cosAnn;
            }
        }
        
        case 'plateau': {
            const patience = 3;
            const factor = 0.2;
            if (valAccHistory.length < patience + 1) {
                return initialLr;
            }
            const recentHistory = valAccHistory.slice(-patience -1);
            const currentAcc = recentHistory[recentHistory.length - 1];
            const plateauAcc = recentHistory[0];

            if (currentAcc - plateauAcc < 0.001) { // If accuracy hasn't improved by 0.1%
                const lastLr = getLearningRate('plateau', epoch - 1, totalEpochs, initialLr, valAccHistory.slice(0, -1));
                return Math.max(minLr, lastLr * factor);
            }
            // Return previous LR if no plateau
            return getLearningRate('plateau', epoch - 1, totalEpochs, initialLr, valAccHistory.slice(0, -1));
        }

        default:
            return initialLr;
    }
}


/**
 * Generates an array of learning rates over all epochs for visualization.
 */
export function generateScheduleData(schedule: LRSchedulerId, epochs: number, initialLr: number): { epoch: number, lr: number }[] {
    const data: { epoch: number, lr: number }[] = [];
    let valAccHistory: number[] = [];
    
    // Simulate a reasonable accuracy progression for plateau visualization
    if (schedule === 'plateau') {
         for (let i = 0; i < epochs; i++) {
            if (i < 5) valAccHistory.push(0.1 * i);
            else if (i > 5 && i < 10) valAccHistory.push(0.5); // Plateau
            else if (i > 10 && i < 15) valAccHistory.push(0.5 + 0.05 * (i-10));
            else valAccHistory.push(0.75); // Plateau
        }
    }


    for (let i = 0; i < epochs; i++) {
        data.push({
            epoch: i + 1,
            lr: getLearningRate(schedule, i, epochs, initialLr, valAccHistory.slice(0, i))
        });
    }
    return data;
}