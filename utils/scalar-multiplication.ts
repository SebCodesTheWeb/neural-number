// Gets rid of -0 as result of multiplying with zero
const adjustedMultiplication = (num1: number, num2: number) => {
  if (num1 === 0 || num2 === 0) return 0
  return num1 * num2
}

export const scalarMultiplication = (vector: number[], scalar: number) => {
  return vector.map((element) => adjustedMultiplication(element, scalar))
}
