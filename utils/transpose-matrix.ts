export const transposeMatrix = (matrix: number[][]) => {
  const turnedMatrix = matrix.map((_, index) =>
    matrix.flatMap((vec) => vec[index])
  )

  return turnedMatrix
}
