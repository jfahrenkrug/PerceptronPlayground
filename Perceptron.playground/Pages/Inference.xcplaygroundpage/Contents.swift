/*:
# Perceptron Inference

This page demonstrates how a perceptron makes predictions on new inputs.

---
*/

import Foundation

struct Perceptron {
    var weights: [Double]
    var bias: Double
        
    func predict(_ inputs: [Double]) -> Int {
        // Calculate Dot Product
        let weightedDotProduct = zip(inputs, weights)
            .map { $0 * $1 }
            .reduce(0, +)
        
        // Add bias
        let weightedSum = weightedDotProduct + bias
        
        print("inputs: \(inputs) -> dotProduct: \(weightedDotProduct), weighted sum: \(weightedSum)")
        
        // Activation (above or below threshold?)
        return weightedSum >= 0 ? 1 : -1
    }
}

let p = Perceptron(weights: [0.0, 0.0], bias: 0.0)
p.predict([0.0, 0.0])
p.predict([0.0, 1.0])
p.predict([1.0, 0.0])
p.predict([1.0, 1.0])

// Working weights for AND gate:
// 1.0, 1.0, -1.5
