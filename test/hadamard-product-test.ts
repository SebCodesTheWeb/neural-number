import { hadamardProduct } from '../utils/hadamard-product'
import { expect } from 'chai'

describe('hadamardProduct', () => {
  it('Should form appropiate vector from two input vectors', () => {
    const vec1 = [1, 2, 3]
    const vec2 = [2, 0, 3]
    const product = hadamardProduct(vec1, vec2)
    expect(product).to.deep.equal([2, 0, 9])
  })
  it('is symmetric', () => {
    const vec1 = [1, 2, 3]
    const vec2 = [2, 0, 3]
    const productOne = hadamardProduct(vec1, vec2)
    const productTwo = hadamardProduct(vec2, vec1)
    expect(productOne).to.deep.equal(productTwo)
  })
})
