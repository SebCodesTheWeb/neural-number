import * as fs from "fs";
import * as path from "path";

function readImages(filePath: string): number[][] {
  const fileBuffer: Buffer = fs.readFileSync(filePath);
  const numberOfImages: number = fileBuffer.readInt32BE(4);
  const numberOfRows: number = fileBuffer.readInt32BE(8);
  const numberOfColumns: number = fileBuffer.readInt32BE(12);
  let offset: number = 16;
  const images: number[][] = [];

  // console.log(`Number of images: ${numberOfImages}`);
  // console.log(`Number of rows: ${numberOfRows}`);
  // console.log(`Number of columns: ${numberOfColumns}`);

  for (let i = 0; i < numberOfImages; i++) {
    const image: number[] = [];
    for (let j = 0; j < numberOfRows * numberOfColumns; j++) {
      const pixel: number = fileBuffer.readUInt8(offset);
      image.push(pixel / 255.0);
      offset += 1;
    }
    images.push(image);
  }

  // if (images.length > 0 && images[0].length > 0) {
  //   console.log(`First few pixels of the first image: ${images[0].slice(0, 784)}`);
  // }

  return images;
}

const imagesPath: string = path.join(__dirname, 'archive/train-images-idx3-ubyte/train-images-idx3-ubyte');

export const images: number[][] = readImages(imagesPath);

// console.log(images[0][624])

// images[0].forEach((pixel, i) => {
//   if(pixel !== 0) {
//     console.log(i)
//   }
// })

function printImage(image: number[], rows: number, columns: number) {
  for (let i = 0; i < rows; i++) {
    let row = '';
    for (let j = 0; j < columns; j++) {
      const pixelValue = image[i * columns + j];
      row += pixelValue > 0.5 ? '██' : '  ';
    }
    console.log(row);
  }
}

// images.reduce((acc, curr))

if (images.length > 0) {
  const firstImage = images[2];
  // console.log(JSON.stringify(firstImage))
  const rows = 28; 
  const columns = 28; 
  
  console.log('First image:');
  printImage(firstImage, rows, columns);
}