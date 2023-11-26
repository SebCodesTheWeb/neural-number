import { matrixMultiplication } from '../utils/matrix-multiplication'
import { expect } from 'chai'

describe('matrixMultiplication', () => {
  it('Should transform the first matrix by the second', () => {
    const matrix1 = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]
    const matrix2 = [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]

    const result = [
      [2, 4, 6],
      [8, 10, 12],
      [14, 16, 18],
    ]

    const transformedMatrix = matrixMultiplication(matrix1, matrix2)
    expect(transformedMatrix).to.deep.equal(result)
  })

  it('Can multiply a vector', () => {
    const baseVector = [[1, 0, 0]]

    const rotationMatrix = () => [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]

    const rotatedMatrix = matrixMultiplication(baseVector, rotationMatrix())

    expect(rotatedMatrix).to.deep.equal([[2, 0, 0]])
  })

  it('Can handles this matrix', () => {
    const matrix1 = [[1, 2, 3]]
    const matrix2 = [
      [1, 4, 7],
      [2, 5, 8],
      [3, 6, 9],
    ]

    const result = matrixMultiplication(matrix1, matrix2)

    expect(result).to.deep.equal([[14, 32, 50]])
  })
})
