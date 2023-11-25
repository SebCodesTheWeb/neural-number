import { vectorTransformation } from './vector-transformation'

export const transformLines = (
  listOfLines: number[][][],
  transformationMatrix: number[][]
): number[][][] => {
  const transformedLines = listOfLines.map((line) =>
    line.map((point) => vectorTransformation(point, transformationMatrix))
  )

  return transformedLines
}
