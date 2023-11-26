import { dotProduct } from '../utils/dot-product'
import { expect } from 'chai'

describe('dotProduct', () => {
  it('Should take the dot product between two numbers', () => {
    const vec1 = [1, 2, 3]
    const vec2 = [2, 3, 5]
    const product = dotProduct(vec1, vec2)
    expect(product).to.equal(23)
  })
  it('is symmetric', () => {
    const vec1 = [1, 2, 3]
    const vec2 = [2, 3, 5]
    const productOne = dotProduct(vec1, vec2)
    const productTwo = dotProduct(vec2, vec1)
    expect(productOne).to.equal(productTwo)
  })
})
