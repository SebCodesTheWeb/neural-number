import { NeuralNetwork } from './neural-network'
import { images } from './process-images'
import { expected } from './process-labels'
import { shuffleArray } from './utils/shuffle-array'

const cnn = new NeuralNetwork('./neural-network-config.json')
cnn.saveNetworkConfig('./neural-network-config.json')

function train(nbrEpochs: number, miniBatchSize: number, stepSize: number) {
  for (let epoch = 0; epoch < nbrEpochs; epoch++) {
    const shuffledIndices = images.map((_, idx) => idx)
    shuffleArray(shuffledIndices)

    for (let i = 0; i < images.length; i += miniBatchSize) {
      const miniBatchIndices = shuffledIndices.slice(i, i + miniBatchSize)
      const miniBatchImages = miniBatchIndices.map((index) => images[index])
      const miniBatchLabels = miniBatchIndices.map((index) => expected[index])

      const averageGradient = cnn.getAverageGradient(
        miniBatchImages,
        miniBatchLabels
      )

      cnn.updateParameters(averageGradient, stepSize)
    }
  }
}

train(1, 1, 1)
