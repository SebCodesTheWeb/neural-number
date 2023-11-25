import { matrixMultiplication } from './matrix-multiplication'

export const vectorTransformation = (
  vec: number[],
  transformationMatrix: number[][]
): number[] => {
  const placeholderMatrix = [vec]
  const transformedMatrix = matrixMultiplication(
    placeholderMatrix,
    transformationMatrix
  )

  return transformedMatrix[0]
}
