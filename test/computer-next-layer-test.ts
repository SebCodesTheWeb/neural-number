import { expect } from 'chai'
import { describe, it } from 'mocha'
import { computeNextLayer } from '../lib/compute-next-layer'
import { transposeMatrix } from '../utils'

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))

describe('computeNextLayer', () => {
  it('correctly computes the activations for the next layer', () => {
    const previousLayer = [1, 2]
    const weights = [
      [0.5, 0.3],
      [-0.5, 0.3],
    ]
    const biases = [0.1, -0.2]

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

  it('correctly computes the activations for the next layer with a 10d to 3d shape change', () => {
    const previousLayer: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const weights: number[][] = transposeMatrix([
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2],
      [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3],
    ])
    const biases: number[] = [0.1, 0.2, 0.3]

    const expectedLayer: number[] = [
      normalizingFunction(
        weights[0].reduce((sum, w, i) => sum + w * previousLayer[i], biases[0])
      ),
      normalizingFunction(
        weights[1].reduce((sum, w, i) => sum + w * previousLayer[i], biases[1])
      ),
      normalizingFunction(
        weights[2].reduce((sum, w, i) => sum + w * previousLayer[i], biases[2])
      ),
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
