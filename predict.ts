import { NeuralNetwork } from './neural-network'
import { images } from './process-images'
import { testExpected } from './process-labels'

const cnn = new NeuralNetwork('./neural-network-config.json')

const result = cnn.forwardPass(images[2])


const highestIndex = result.reduce((acc, curr, i) => {
    if (curr > result[acc]) {
        acc = i
    }
    return acc
}, 0)

console.log(`Guess: ${highestIndex} with ${result[highestIndex] * 100}% accuracy`)
console.log({result})
console.log(testExpected[1])
