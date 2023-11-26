import { readFileSync, writeFileSync } from 'fs'
import { computeNextLayer } from './lib/compute-next-layer'
import { join } from 'path'
import { BP1 } from './lib/BP1'
import { last } from 'ramda'
import { BP4 } from './lib/BP4'
import { BP2 } from './lib/BP2'
import { vecAdd, matrixAdd } from './utils'

type LayerConfig = {
  weights: number[][]
  biases: number[]
}

type NeuralNetworkConfig = {
  layers: number[]
  layerConfigs?: LayerConfig[]
}

type NetworkGradient = {
  weightGradients: number[][][]
  biasGradients: number[][]
}

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))
const derivativeOfNormalizingFunction = (n: number) =>
  Math.exp(-n) / Math.pow(1 + Math.exp(-n), 2)

export class NeuralNetwork {
  public layers: number[]
  public layerConfigs: LayerConfig[] = []

  constructor(configPath: string) {
    const config = this.loadNetworkConfig(configPath)
    this.layers = config.layers
    if (!config.layerConfigs) {
      this.initializeNetwork()
    } else {
      this.layerConfigs = config.layerConfigs
    }
  }

  private initializeNetwork() {
    this.layerConfigs = this.layers.slice(1).map((layerSize, index) => ({
      weights: Array.from({ length: layerSize }, () =>
        Array.from({ length: this.layers[index] }, Math.random)
      ),
      biases: Array.from({ length: layerSize }, Math.random),
    }))
  }

  private loadNetworkConfig(configPath: string): NeuralNetworkConfig {
    const configFile = readFileSync(join(__dirname, configPath), {
      encoding: 'utf-8',
    })
    return JSON.parse(configFile)
  }

  public saveNetworkConfig(savePath: string): void {
    const data = {
      layers: this.layers,
      layerConfigs: this.layerConfigs,
    }
    writeFileSync(join(__dirname, savePath), JSON.stringify(data, null, 2), {
      encoding: 'utf-8',
    })
  }

  public forwardPassWithSavedActivations(input: number[]): {
    activations: number[][]
    zVectors: number[][]
  } {
    const activations = [input]
    const zVectors: number[][] = []

    let activation = input

    for (const { weights, biases } of this.layerConfigs) {
      const z = computeNextLayer(activation, weights, biases, (x) => x)
      zVectors.push(z)
      activation = z.map(normalizingFunction)
      activations.push(activation)
    }

    return { activations, zVectors }
  }

  public backPropgataion(input: number[], expected: number[]): NetworkGradient {
    let biasGradients: number[][] = []
    let weightGradients: number[][][] = []

    const { activations, zVectors } =
      this.forwardPassWithSavedActivations(input)
    const πVector = BP1(
      last(activations) as number[],
      expected,
      derivativeOfNormalizingFunction,
      last(zVectors) as number[]
    )

    const weightGradient = BP4(πVector, last(activations) as number[])

    biasGradients.push(πVector)
    weightGradients.push(weightGradient)

    for (let i = this.layerConfigs.length - 1; i > 0; i -= 1) {
      const { weights } = this.layerConfigs[i]
      const newπVector = BP2(
        weights,
        πVector,
        derivativeOfNormalizingFunction,
        zVectors[i]
      )
      const newWeightGradient = BP4(newπVector, activations[i])

      biasGradients.push(newπVector)
      weightGradients.push(newWeightGradient)
    }

    return {
      biasGradients,
      weightGradients,
    }
  }

  public updateParameters(gradient: NetworkGradient, stepSize: number) {
    for (let i = 0; i < this.layerConfigs.length; i += 1) {
      this.layerConfigs[i].biases = vecAdd(
        this.layerConfigs[i].biases,
        gradient.biasGradients[i].map((bias) => -stepSize * bias)
      )
      this.layerConfigs[i].weights = matrixAdd(
        this.layerConfigs[i].weights,
        gradient.weightGradients[i].map((row) =>
          row.map((weight) => -stepSize * weight)
        )
      )
    }
  }

  public forwardPass(input: number[], layerIndex = 0): number[] {
    if (layerIndex >= this.layerConfigs.length) {
      return input
    }

    const { weights, biases } = this.layerConfigs[layerIndex]
    const activation = computeNextLayer(
      input,
      weights,
      biases,
      normalizingFunction
    )

    return this.forwardPass(activation, layerIndex + 1)
  }

  // uses 1/2n sum (a-y)^2
  private lossFunction(output: number[], expected: number[]): number {
    return (
      output.reduce(
        (sum, outputVal, i) => sum + (outputVal - expected[i]) ** 2,
        0
      ) / output.length
    )
  }
}
