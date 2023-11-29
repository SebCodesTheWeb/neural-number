import { expect } from 'chai'
import { describe, it } from 'mocha'
import { matrixAdd } from '../utils'

describe.only('matrixAdd', () => {
  it('correctly adds two matrices of the same dimensions', () => {
    const matrixA = [
      [1, 2],
      [3, 4],
    ]
    const matrixB = [
      [5, 6],
      [7, 8],
    ]
    const expected = [
      [6, 8],
      [10, 12],
    ]
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })

  it('correctly adds two row vectors', () => {
    const matrixA = [[1, 2, 3]]
    const matrixB = [[4, 5, 6]]
    const expected = [[5, 7, 9]]
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })

  it('correctly adds two column vectors', () => {
    const matrixA = [[1], [2], [3]]
    const matrixB = [[4], [5], [6]]
    const expected = [[5], [7], [9]]
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })

  it('throws an error when adding matrices of different row counts', () => {
    const matrixA = [[1, 2]]
    const matrixB = [
      [3, 4],
      [5, 6],
    ]
    expect(() => matrixAdd(matrixA, matrixB)).to.throw(
      Error,
      /Row count mismatch/
    )
  })

  it('throws an error when adding matrices of different column counts within a row', () => {
    const matrixA = [[1, 2, 3]]
    const matrixB = [[4, 5]]
    expect(() => matrixAdd(matrixA, matrixB)).to.throw(
      Error,
      /Column count mismatch/
    )
  })

  it('correctly adds two empty matrices', () => {
    const matrixA: number[][] = []
    const matrixB: number[][] = []
    const expected: number[][] = []
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })

  it('correctly adds when one matrix is empty and other is not', () => {
    const matrixA: number[][] = []
    const matrixB = [
      [1, 2],
      [3, 4],
    ]
    const expected = [
      [1, 2],
      [3, 4],
    ]
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })

  it('throws an error when adding matrices with empty rows', () => {
    const matrixA = [[]]
    const matrixB = [[]]
    const expected = [[]]
    expect(matrixAdd(matrixA, matrixB)).to.deep.equal(expected)
  })
})
