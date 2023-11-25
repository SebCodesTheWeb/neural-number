export const matrixAdd = (
  matrixA: number[][],
  matrixB: number[][]
): number[][] => {
  return matrixA.map((row, rowIndex) =>
    row.map((value, colIndex) => value + matrixB[rowIndex][colIndex])
  )
}
