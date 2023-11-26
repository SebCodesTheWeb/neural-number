import { expect } from 'chai'
import { describe, it } from 'mocha'
import { NeuralNetwork } from '../neural-network'
import initialConfig from './network-test-config.json'

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))
const derivativeOfNormalizingFunction = (n: number) => {
  return Math.exp(-n) / Math.pow(1 + Math.exp(-n), 2)
}

describe.only('NeuralNetwork', () => {
  it('should compute correct bias and weight gradients after one backpropagation iteration', () => {
    const network = new NeuralNetwork('./test/network-test-config.json')
    const input = [0.05, 0.1]
    const expected = [0.01, 0.99]

    const gradients = network.backPropgataion(input, expected)
    const { activations, zVectors } =
      network.forwardPassWithSavedActivations(input)

    const outputActivations = activations[activations.length - 1]
    const outputZVectors = zVectors[zVectors.length - 1]

    const biasGradients = outputActivations.map((activation, i) => {
      return (
        (activation - expected[i]) *
        derivativeOfNormalizingFunction(outputZVectors[i])
      )
    })

    const previousActivations = activations[activations.length - 2]
    const weightGradientsTwo = biasGradients.map((biasGradient) => {
      return previousActivations.map((activation) => biasGradient * activation)
    })
    const weightGradients = [
      [
        biasGradients[0] * activations[0][0],
        biasGradients[0] * activations[0][1],
      ],
      [
        biasGradients[1] * activations[0][0],
        biasGradients[1] * activations[0][1],
      ],
    ]

    console.log({
      weightGradientsTwo,
      weightGradients,
      ref: JSON.stringify(gradients.weightGradients),
    })

    expect(gradients.weightGradients).to.deep.equal([weightGradients])
    // expect(gradients.biasGradients).to.deep.equal([biasGradients])
  })
})
