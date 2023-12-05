export const hadamardProduct = (vec1: number[], vec2: number[]): number[] => {
  return vec1.map((element, index) => element * (vec2[index] ?? 0))
}
