import { reduce, add } from 'ramda'
import { hadamardProduct } from './hadamard-product'

export const dotProduct = (vec1: number[], vec2: number[]) =>
  reduce(add, 0, hadamardProduct(vec1, vec2))