import { scalarMultiplication, transposeVector } from '../utils'

//@ts-ignore
export const BP4 = (πVector: number[], activation: number[]): number[][] => console.log({πLen: πVector.length, actLen: activation.length}) ||
  transposeVector(activation).map((vec) =>
    scalarMultiplication(πVector, vec[0])
  )