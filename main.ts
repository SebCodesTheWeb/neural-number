import { NeuralNetwork } from './neural-network'
import { images } from './process-images'

const cnn = new NeuralNetwork('./neural-network-config.json')

const output = cnn.forwardPass(images[0])

const indexOfMaxValue = output.reduce(
  (bestIndexSoFar, currentValue, currentIndex, array) => {
    return currentValue > array[bestIndexSoFar] ? currentIndex : bestIndexSoFar
  },
  0
)

console.log(indexOfMaxValue)
