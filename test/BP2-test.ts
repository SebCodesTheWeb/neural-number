import { expect } from 'chai'
import { describe, it } from 'mocha'
import { BP2 } from '../lib/BP2'

describe('BP2', () => {
  it('calculates the correct hidden layer deltas given the next layer delta', () => {
    const weights = [
      [0.5, 1],
      [0.1, 0.3],
    ]
    const nextLayerDelta = [0.9, 0.2]
    const zVector = [0.1, 0.2]

    const derivativeOfActivationFn = (n: number) =>
      Math.exp(-n) / Math.pow(1 + Math.exp(-n), 2)

    const hiddenLayerDelta = BP2(
      weights,
      nextLayerDelta,
      derivativeOfActivationFn,
      zVector
    )

    const expectedHiddenLayerDelta = [
      (0.9 * 0.5 + 1 * 0.2) * derivativeOfActivationFn(0.1),
      (0.9 * 0.1 + 0.3 * 0.2) * derivativeOfActivationFn(0.2),
    ]

    expect(hiddenLayerDelta).to.deep.equal(expectedHiddenLayerDelta)
  })
})
