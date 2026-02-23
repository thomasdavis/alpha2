export { Variable, Tape, type TapeEntry } from "./tape.js";
export {
  add, sub, mul, div, neg, scale,
  matmul,
  sum, mean,
  exp, log, sqrt, relu, gelu, clamp,
  embedding, layerNorm, softmax, crossEntropy,
  reshape, transpose,
} from "./ops.js";
