import { expect } from 'chai'
import { describe, it } from 'mocha'
import { NeuralNetwork } from '../neural-network' 
import initialConfig from './network-test-config.json' 

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))
const derivativeOfNormalizingFunction = (n: number) => {
  return Math.exp(-n) / Math.pow(1 + Math.exp(-n), 2)
}

describe.skip('NeuralNetwork', () => {
  it('should compute correct bias and weight gradients after one backpropagation iteration', () => {
    const network = new NeuralNetwork('./test/network-test-config.json')
    const input = [0.05, 0.1]
    const expected = [0.01, 0.99]

    const gradients = network.backPropgataion(input, expected)


    console.log(gradients.weightGradients)
    // expect(gradients.weightGradients).to.deep.equal(expectedWeightGradients)
    // expect(gradients.biasGradients).to.deep.equal(expectedBiasGradients)
  })
})
