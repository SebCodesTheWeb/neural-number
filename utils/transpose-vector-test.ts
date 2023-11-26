import { transposeVector } from './transpose-vector'
import { expect } from 'chai'

describe('transposeVector', () => {
  it('Can transpose', () => {
    const testVector = [0.2, 0.5, 0.3, 0.6]
    const expected = [[0.2], [0.5], [0.3], [0.6]]
    const transpose = transposeVector(testVector)
    expect(transpose).to.deep.equal(expected)
  })
})
