import { transposeMatrix } from '../utils/transpose-matrix'
import { expect } from 'chai'

describe('transposeMatrix', () => {
  it('Correctly transposes matrix', () => {
    const matrix = [
      [1, 4, 7],
      [2, 5, 8],
      [3, 6, 9],
    ]

    const turnedMatrix = transposeMatrix(matrix)
    expect(turnedMatrix).to.deep.equal([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ])
  })
})
