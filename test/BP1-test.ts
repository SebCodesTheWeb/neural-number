import { expect } from 'chai'
import { describe, it } from 'mocha'
import { BP1 } from '../lib/BP1'

describe('BP1', () => {
  it('calculates the correct output layer deltas for a sample set', () => {
    const outputActivations = [0.8, 0.7]
    const expected = [1.0, 0.0]
    const zValues = [1.0, -1.0]
    const sigmoidDerivative = (n: number) =>
      Math.exp(-n) / Math.pow(1 + Math.exp(-n), 2)

    const outputLayerDeltas = BP1(
      outputActivations,
      expected,
      sigmoidDerivative,
      zValues
    )

    const expectedDeltas = [
      (0.8 - 1.0) * sigmoidDerivative(1.0),
      (0.7 - 0.0) * sigmoidDerivative(-1.0),
    ]

    expect(outputLayerDeltas).to.deep.equal(expectedDeltas)
  })
})
