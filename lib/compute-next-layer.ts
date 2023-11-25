import { vectorTransformation, vecAdd } from '../utils'

export const computeNextLayer = (
  previousLayer: number[],
  weights: number[][],
  biases: number[],
  normalizingFunction: (n: number) => number
) => {
  const primaryActivation = vecAdd(
    vectorTransformation(previousLayer, weights),
    biases
  )

  const normalizedActivation = primaryActivation.map(normalizingFunction)

  return normalizedActivation
}
