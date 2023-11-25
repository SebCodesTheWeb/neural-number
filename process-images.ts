import * as fs from "fs";
import * as path from "path";

function readImages(filePath: string): number[][] {
  const fileBuffer: Buffer = fs.readFileSync(filePath);
  const numberOfImages: number = fileBuffer.readInt32BE(4);
  const numberOfRows: number = fileBuffer.readInt32BE(8);
  const numberOfColumns: number = fileBuffer.readInt32BE(12);
  let offset: number = 16;
  const images: number[][] = [];

  for (let i = 0; i < numberOfImages; i++) {
    const image: number[] = [];
    for (let j = 0; j < numberOfRows * numberOfColumns; j++) {
      const pixel: number = fileBuffer.readUInt8(offset);
      image.push(pixel / 255.0);
      offset += 1;
    }
    images.push(image);
  }
  return images;
}

const imagesPath: string = path.join(__dirname, 'archive/train-images-idx3-ubyte/train-images-idx3-ubyte');

const images: number[][] = readImages(imagesPath);
console.log(images[0])