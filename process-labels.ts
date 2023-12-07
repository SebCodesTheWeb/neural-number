import * as fs from 'fs'
import * as path from 'path'

function readLabels(filePath: string): number[][] {
  const fileBuffer: Buffer = fs.readFileSync(filePath)
  const numberOfLabels: number = fileBuffer.readInt32BE(4)
  let offset: number = 8 
  const labels: number[][] = []

  for (let i = 0; i < numberOfLabels; i++) {
    const label: number = fileBuffer.readUInt8(offset)
    const oneHotLabel: number[] = new Array(10).fill(0)
    oneHotLabel[label] = 1
    labels.push(oneHotLabel)
    offset += 1
  }
  return labels
}

const testingLabelsPath: string = path.join(
  __dirname,
  'archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
)

const trainingLabelsPath: string = path.join(
  __dirname,
  'archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
)

export const trainExpected: number[][] = readLabels(trainingLabelsPath)
export const testExpected: number[][] = readLabels(testingLabelsPath)