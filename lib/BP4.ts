import {
  matrixMultiplication,
  transposeMatrix,
} from '../utils'

export const BP4 = (πVector: number[], activation: number[]): number[][] =>
  matrixMultiplication([πVector], transposeMatrix([activation]))
