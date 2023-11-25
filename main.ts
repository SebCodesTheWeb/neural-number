
import { NeuralNetwork } from "./neural-network";

const cnn = new NeuralNetwork('./neural-network-config.json')

const input = new Array(728).fill(0.5)
const output = cnn.forwardPass(input)

console.log({output})