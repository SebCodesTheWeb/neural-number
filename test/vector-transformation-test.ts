import { vectorTransformation } from '../utils/vector-transformation'
import { expect } from 'chai'

describe('vectorTransformation', () => {
  it('Can multiply a vector', () => {
    const baseVector = [1, 0, 0]

    const scaleMatrix = [
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]

    const rotatedMatrix = vectorTransformation(baseVector, scaleMatrix)

    expect(rotatedMatrix).to.deep.equal([2, 0, 0])
  })

  it('Handles this random matrix', () => {
    const baseVector = [3, 7, 5]

    const transformationMatrix = [
      [0.32, -0.08, 0.95],
      [0.46, 0.89, -0.08],
      [-0.83, 0.46, 0.32],
    ]

    const rotatedVector = vectorTransformation(baseVector, transformationMatrix)
    const roundedVector = rotatedVector.map((num) => +num.toFixed(2))

    expect(roundedVector).to.deep.equal([0.03, 8.29, 3.89])
  })
})
