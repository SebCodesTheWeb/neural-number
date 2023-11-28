import { expect } from 'chai'
import { describe, it } from 'mocha'
import { NeuralNetwork } from '../neural-network'

describe('NeuralNetwork', () => {
  it('should compute correct average bias and weight gradients for a training set', () => {
    const network = new NeuralNetwork('./test/network-test-config.json')

    const inputSet = [
      [0.05, 0.1],
      [0.5, 0.9],
    ]
    const expectedSet = [
      [0.01, 0.99],
      [0.8, 0.2],
    ]

    const averageGradients = network.getAverageGradient(inputSet, expectedSet)

    const gradientOne = network.backPropgataion(inputSet[0], expectedSet[0])
    const gradientTwo = network.backPropgataion(inputSet[1], expectedSet[1])

    expect(averageGradients.weightGradients[0][0][0]).to.equal(
      (gradientOne.weightGradients[0][0][0] +
        gradientTwo.weightGradients[0][0][0]) /
        2
    )
  })
})
