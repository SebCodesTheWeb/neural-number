import {
  hadamardProduct,
  transposeMatrix,
  vectorTransformation,
} from '../utils'

export const BP2 = (
  weights: number[][],
  πVector: number[],
  normalizationFnDerivative: (n: number) => number,
  zVector: number[]
) =>
  hadamardProduct(
    vectorTransformation(πVector, transposeMatrix(weights)),
    zVector.map(normalizationFnDerivative)
  )
