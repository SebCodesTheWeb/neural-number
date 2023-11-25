import { readFileSync, writeFileSync } from 'fs'
import { computeNextLayer } from './lib/compute-next-layer'
import { join } from 'path'

type LayerConfig = {
  weights: number[][]
  biases: number[]
}

type NeuralNetworkConfig = {
  layers: number[]
  layerConfigs?: LayerConfig[] // Make layerConfigs optional
}

const normalizingFunction = (n: number) => 1 / (1 + Math.exp(-n))

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
}
