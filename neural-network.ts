import { readFileSync, writeFileSync } from 'fs'
import { computeNextLayer } from './lib/compute-next-layer'
import { join } from 'path'
import { BP1 } from './lib/BP1'
import { last } from 'ramda'
import { BP4 } from './lib/BP4'

type LayerConfig = {
  weights: number[][]
  biases: number[]
}

type NeuralNetworkConfig = {
  layers: number[]
  layerConfigs?: LayerConfig[] // Make layerConfigs optional
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

  private forwardPassWithSavedActivations(input: number[]): {
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

    return {
      biasGradients,
      weightGradients,
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

  private lossFunction(output: number[], expected: number[]): number {
    return (
      output.reduce(
        (sum, outputVal, i) => sum + (outputVal - expected[i]) ** 2,
        0
      ) / output.length
    )
  }
}
