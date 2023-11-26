import { scalarMultiplication, transposeVector } from '../utils'

export const BP4 = (πVector: number[], activation: number[]): number[][] =>
  transposeVector(activation).map((vec) =>
    scalarMultiplication(πVector, vec[0])
  )