import { scalarMultiplication } from '../utils/scalar-multiplication'
import { expect } from 'chai'

describe('scalarMultiplicatoin', () => {
  it('Scales a vector accoringly', () => {
    const baseVector = [1, 0, -2]
    const scalar = -1.5

    const scaledVector = scalarMultiplication(baseVector, scalar)

    expect(scaledVector).to.deep.equal([-1.5, 0, 3])
  })
})
