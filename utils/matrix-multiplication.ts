import { transposeMatrix } from './transpose-matrix'
import { dotProduct } from './dot-product'

export const matrixMultiplication = (
  maxtrix1: number[][],
  matrix2: number[][]
) => {
  const matrix2Tranpose = transposeMatrix(matrix2)

  const transformedMatrix = maxtrix1.map((vector) =>
    matrix2Tranpose.map((vec2) => dotProduct(vec2, vector))
  )

  return transformedMatrix
}
