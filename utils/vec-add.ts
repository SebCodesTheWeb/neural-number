import { zipWith, add } from 'ramda'

export const vecAdd = zipWith<number, number, number>(add)