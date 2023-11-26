import { vecAdd } from '../utils/vec-add'
import { expect } from 'chai'

describe('vecAdd', () => {
  it('Should correctly add two vectors together', () => {
    const vec1 = [1, 2, 3]
    const vec2 = [2, 3, 4]
    const sum = vecAdd(vec1, vec2)

    expect(sum).to.deep.equal([3, 5, 7])
  })
})
