/*:
# Perceptron Training

This page demonstrates the complete perceptron training process step-by-step.

---
*/

import Foundation

// dataset for AND-gate
let andGateDataset: [([Double], Int)] = [
    ([0.0, 0.0], -1),
    ([0.0, 1.0], -1),
    ([1.0, 0.0], -1),
    ([1.0, 1.0],  1),
]

// dataset for OR-gate
let orGateDataset: [([Double], Int)] = [
    ([0.0, 0.0], -1),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 1),
]

// dataset for XOR-gate (single-layer perceptron will never converge here)
let xorGateDataset: [([Double], Int)] = [
    ([0.0, 0.0], -1),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], -1),
]

struct Perceptron {
    var weights: [Double]
    var bias: Double
        
    func predict(_ inputs: [Double]) -> Int {
        precondition(inputs.count == weights.count, "Perceptron expects \(weights.count) inputs but received \(inputs.count)")
        // Calculate Dot Product
        let weightedDotProduct = zip(inputs, weights)
            .map { $0 * $1 }
            .reduce(0, +)
        
        // Add bias
        let weightedSum = weightedDotProduct + bias
        
        // Activation (above or below threshold?)
        return weightedSum >= 0 ? 1 : -1
    }
    
    /// Train the perceptron on a labeled dataset
    /// Labels must be -1 or +1
    mutating func train(
        dataset: [([Double], Int)],
        learningRate: Double = 0.5,
        epochs: Int = 10,
    ) {
        for epoch in 1...epochs {
            print("\n=== Epoch \(epoch) ===")
            var mistakes = 0

            for (inputs, expected) in dataset {
                // Step 1: Run prediction
                let prediction = predict(inputs)
                
                // Step 2: Calculate error
                let error = Double(expected) - Double(prediction)

                print("  inputs=\(inputs) expected=\(expected)  pred=\(prediction) error=\(error)")

                // Step 2: Check if wrong
                if error != 0 {
                    mistakes += 1

                    // Step 4: Apply weight updates
                    for i in weights.indices {
                        let newWeight = weights[i] + (error * learningRate * inputs[i])
                        print("  Updating w\(i) from \(weights[i]) to \(newWeight)")
                        weights[i] = newWeight
                    }
                    
                    // Step 5: Apply bias update
                    let newBias = bias + (error * learningRate * 1.0)
                    print("  Updating bias from \(bias) to \(newBias)")
                    bias = newBias

                    print("  ❌ updated → weights=\(weights) bias=\(bias)")
                } else {
                    print("  ✅ correct (no update)")
                }
            }

            print("Epoch \(epoch) summary → mistakes: \(mistakes), weights=\(weights), bias=\(bias)")

            if mistakes == 0 {
                print("Converged ✅")
                break
            }
        }

        print("\n==> Training finished: weights=\(weights), bias=\(bias)\n")
    }
}

var p = Perceptron(weights: [0.0, 0.0], bias: 0.0)

// Choose dataset (AND, OR, XOR)
let dataset = andGateDataset

p.train(dataset: dataset, learningRate: 0.3, epochs: 10)
for (inputs, expected) in dataset {
    let prediction = p.predict(inputs)
    print("final check → inputs=\(inputs) expected=\(expected) pred=\(prediction) \(expected == prediction ? "✅" : "❌")")
}
