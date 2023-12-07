import { NeuralNetwork } from './neural-network'
import { images } from './process-images'
import { trainExpected } from './process-labels'
import { shuffleArray } from './utils/shuffle-array'

const cnn = new NeuralNetwork('./neural-network-config.json')
cnn.saveNetworkConfig('./neural-network-config.json')

function train(nbrEpochs: number, miniBatchSize: number, stepSize: number) {
  for (let epoch = 0; epoch < nbrEpochs; epoch++) {
    console.log("epoch: ", epoch)
    const shuffledIndices = images.map((_, idx) => idx)
    shuffleArray(shuffledIndices)

    for (let i = 0; i < images.length; i += miniBatchSize) {
      console.log("miniBatch: ", i / miniBatchSize)
      const miniBatchIndices = shuffledIndices.slice(i, i + miniBatchSize)
      const miniBatchImages = miniBatchIndices.map((index) => images[index])
      const miniBatchLabels = miniBatchIndices.map((index) => trainExpected[index])

      const averageGradient = cnn.getAverageGradient(
        miniBatchImages,
        miniBatchLabels
      )

      // console.log("here", averageGradient.weightGradients[0])

      cnn.updateParameters(averageGradient, stepSize)
      cnn.saveNetworkConfig('./neural-network-config.json')
    }
  }
}

// optimal: 100, 80, 7
// const t1 = performance.now()
train(5, 80, 1)
// const t2 = performance.now()
// console.log(`Process finished, exited in ${t2 - t1}ms`)
