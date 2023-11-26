import { expect } from 'chai'
import { describe, it } from 'mocha'
import { NeuralNetwork } from '../neural-network'
import initialConfig from './network-test-config.json'

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))

describe('ForwardPass', () => {
  it('Correct fowardPass', () => {
    const network = new NeuralNetwork('./test/network-test-config.json')
    const input = [0.05, 0.1]

    const preActivation = [
      0.15 * 0.05 + 0.1 * 0.2 + 0.35,
      0.25 * 0.05 + 0.3 * 0.1 + 0.35,
    ]

    const postActivation = preActivation.map(normalizingFunction)

    const output = network.forwardPass(input)
    const { activations, zVectors } =
      network.forwardPassWithSavedActivations(input)
    expect(output).to.deep.equal(postActivation)
    expect(activations[1]).to.deep.equal(postActivation)
    expect(zVectors[0]).to.deep.equal(preActivation)
  })
})
