import { expect } from 'chai'
import { describe, it } from 'mocha'
import { BP4 } from '../lib/BP4'

describe('BP4', () => {
  it('calculates the correct weight gradient matrix given πVector and activations', () => {
    const piVector = [0.5, 0.1]
    const activations = [0.2, 0.8, 0.4]

    const weightGradientMatrix = BP4(piVector, activations)

    const expectedWeightGradientMatrix = [
      [0.2 * 0.5, 0.2 * 0.1],
      [0.8 * 0.5, 0.8 * 0.1],
      [0.4 * 0.5, 0.4 * 0.1],
    ]

    expect(weightGradientMatrix).to.deep.equal(expectedWeightGradientMatrix)
  })
})
