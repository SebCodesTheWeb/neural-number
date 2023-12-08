import { NeuralNetwork } from './neural-network'
import { images, testImages } from './process-images'
import { testExpected } from './process-labels'

const cnn = new NeuralNetwork('./trained-network.json')

let nbrCorrect  = 0
let total = 0
testImages.forEach((_, i) => {
    console.log(i)
    const result = cnn.forwardPass(testImages[i])
    const guess = result.reduce((acc, curr, i) => {
        if (curr > result[acc]) {
            acc = i
        }
        return acc
    }, 0)
    const expected = testExpected[i].reduce((acc, curr, i) => {
        if (curr > result[acc]) {
            acc = i
        }
        return acc
    }, 0)

    if(guess === expected) {
        nbrCorrect += 1
    }
    total += 1
})

console.log({nbrCorrect, total}, nbrCorrect/total)

// const result = cnn.forwardPass(testImages[1])


// const highestIndex = result.reduce((acc, curr, i) => {
//     if (curr > result[acc]) {
//         acc = i
//     }
//     return acc
// }, 0)

// console.log(`Guess: ${highestIndex} with ${result[highestIndex] * 100}% accuracy`)
// console.log({result})
// console.log(testExpected[1])
