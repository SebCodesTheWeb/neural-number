export const transposeMatrix = (matrix: number[][]): number[][] => {
  const rows = matrix.length
  const cols = matrix.reduce((max, row) => Math.max(max, row.length), 0)
  const transposed = new Array(cols)

  for (let i = 0; i < cols; i++) {
    transposed[i] = new Array(rows)
    for (let j = 0; j < rows; j++) {
      if (matrix[j][i] !== undefined) {
        transposed[i][j] = matrix[j][i]
      }
    }
  }

  return transposed
}
