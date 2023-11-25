import { scalarMultiplication, vecAdd, hadamardProduct } from '../utils'

export const BP1 = (
  activation: number[],
  expectedVector: number[],
  normalizationFnDerivative: (n: number) => number,
  zVector: number[]
) =>
  hadamardProduct(
    vecAdd(activation, scalarMultiplication(expectedVector, -1)),
    zVector.map(normalizationFnDerivative)
  )
