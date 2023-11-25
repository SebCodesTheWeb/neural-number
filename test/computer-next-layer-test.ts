import { expect } from 'chai'
import { describe, it } from 'mocha'
import { computeNextLayer } from '../lib/compute-next-layer'

describe('computeNextLayer', () => {
  it('correctly computes the activations for the next layer', () => {
    const previousLayer = [1, 2]
    const weights = [
      [0.5, -0.5],
      [0.3, 0.3],
    ]
    const biases = [0.1, -0.2]
    const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n)) 

    const expectedLayer = [
      normalizingFunction(0.5 * 1 + -0.5 * 2 + 0.1),
      normalizingFunction(0.3 * 1 + 0.3 * 2 - 0.2),
    ]

    const actualLayer = computeNextLayer(
      previousLayer,
      weights,
      biases,
      normalizingFunction
    )
    expect(actualLayer).to.deep.equal(expectedLayer)
  })
})
