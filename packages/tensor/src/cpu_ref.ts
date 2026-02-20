/**
 * cpu_ref -- Reference CPU backend for the alpha tensor system.
 *
 * Every operation is implemented with straightforward loops over typed arrays.
 * The goal is correctness, not speed.
 */

import {
  type Backend,
  type TensorData,
  type Dtype,
  type Shape,
  shapeSize,
  shapeStrides,
  dtypeArray,
  SeededRng,
} from "@alpha/core";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTensor(shape: Shape, dtype: Dtype, data: Float32Array | Float64Array | Int32Array): TensorData {
  return { shape, dtype, data };
}

function allocTensor(shape: Shape, dtype: Dtype): TensorData {
  const Ctor = dtypeArray(dtype);
  return makeTensor(shape, dtype, new Ctor(shapeSize(shape)));
}

/** Normalise a possibly-negative axis to [0, ndim). */
function normalizeAxis(axis: number, ndim: number): number {
  const a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim) throw new Error(`axis ${axis} out of range for ndim ${ndim}`);
  return a;
}

/** Resolve the common dtype for a binary op (promote to wider float if mixed). */
function commonDtype(a: Dtype, b: Dtype): Dtype {
  if (a === b) return a;
  if (a === "f64" || b === "f64") return "f64";
  if (a === "f32" || b === "f32") return "f32";
  return "i32";
}

/**
 * Broadcast two shapes and return [resultShape, stridesA, stridesB].
 * Supports: same shape, scalar broadcast, and trailing-dimension broadcast.
 */
function broadcastShapes(sa: Shape, sb: Shape): [Shape, number[], number[]] {
  const ndim = Math.max(sa.length, sb.length);
  const result: number[] = new Array(ndim);
  const padA = ndim - sa.length;
  const padB = ndim - sb.length;

  for (let i = 0; i < ndim; i++) {
    const da = i < padA ? 1 : sa[i - padA];
    const db = i < padB ? 1 : sb[i - padB];
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes [${sa}] and [${sb}]`);
    }
    result[i] = Math.max(da, db);
  }

  // Build strides: if a dimension is 1 (and needs broadcasting), stride = 0.
  const stridesA = new Array(ndim);
  const stridesB = new Array(ndim);
  let strA = 1;
  let strB = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    const da = i < padA ? 1 : sa[i - padA];
    const db = i < padB ? 1 : sb[i - padB];
    stridesA[i] = da === 1 && result[i] !== 1 ? 0 : strA;
    stridesB[i] = db === 1 && result[i] !== 1 ? 0 : strB;
    strA *= da;
    strB *= db;
  }

  return [result, stridesA, stridesB];
}

/** Convert a flat index in the result to flat indices in a and b using broadcast strides. */
function broadcastIndices(
  flatIdx: number,
  resultShape: Shape,
  stridesA: number[],
  stridesB: number[],
): [number, number] {
  const ndim = resultShape.length;
  let idxA = 0;
  let idxB = 0;
  let remainder = flatIdx;
  for (let d = ndim - 1; d >= 0; d--) {
    const coord = remainder % resultShape[d];
    remainder = (remainder - coord) / resultShape[d];
    idxA += coord * stridesA[d];
    idxB += coord * stridesB[d];
  }
  return [idxA, idxB];
}

function binaryOp(
  a: TensorData,
  b: TensorData,
  fn: (x: number, y: number) => number,
): TensorData {
  const dtype = commonDtype(a.dtype, b.dtype);
  const [resultShape, stridesA, stridesB] = broadcastShapes(a.shape, b.shape);
  const size = shapeSize(resultShape);
  const Ctor = dtypeArray(dtype);
  const out = new Ctor(size);
  for (let i = 0; i < size; i++) {
    const [ia, ib] = broadcastIndices(i, resultShape, stridesA, stridesB);
    out[i] = fn(a.data[ia], b.data[ib]);
  }
  return makeTensor(resultShape, dtype, out);
}

function unaryOp(a: TensorData, fn: (x: number) => number, dtype?: Dtype): TensorData {
  const d = dtype ?? a.dtype;
  const Ctor = dtypeArray(d);
  const out = new Ctor(a.data.length);
  for (let i = 0; i < a.data.length; i++) {
    out[i] = fn(a.data[i]);
  }
  return makeTensor(a.shape, d, out);
}

/** Multi-index <-> flat index conversion helpers. */
function flatToMulti(flat: number, shape: Shape): number[] {
  const ndim = shape.length;
  const coords = new Array(ndim);
  let rem = flat;
  for (let d = ndim - 1; d >= 0; d--) {
    coords[d] = rem % shape[d];
    rem = (rem - coords[d]) / shape[d];
  }
  return coords;
}

function multiToFlat(coords: number[], strides: number[]): number {
  let idx = 0;
  for (let d = 0; d < coords.length; d++) {
    idx += coords[d] * strides[d];
  }
  return idx;
}

// ---------------------------------------------------------------------------
// CpuRefBackend
// ---------------------------------------------------------------------------

export class CpuRefBackend implements Backend {
  readonly name = "cpu_ref";
  private readonly rng = new SeededRng(42);

  // ── creation ────────────────────────────────────────────────────────────

  zeros(shape: Shape, dtype: Dtype = "f32"): TensorData {
    return allocTensor(shape, dtype);
  }

  ones(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const t = allocTensor(shape, dtype);
    t.data.fill(1);
    return t;
  }

  full(shape: Shape, value: number, dtype: Dtype = "f32"): TensorData {
    const t = allocTensor(shape, dtype);
    t.data.fill(value);
    return t;
  }

  randn(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const t = allocTensor(shape, dtype);
    for (let i = 0; i < t.data.length; i++) {
      t.data[i] = this.rng.nextGauss();
    }
    return t;
  }

  fromArray(data: number[], shape: Shape, dtype: Dtype = "f32"): TensorData {
    const size = shapeSize(shape);
    if (data.length !== size) {
      throw new Error(`Data length ${data.length} does not match shape size ${size}`);
    }
    const Ctor = dtypeArray(dtype);
    return makeTensor(shape, dtype, Ctor.from(data));
  }

  // ── math ────────────────────────────────────────────────────────────────

  add(a: TensorData, b: TensorData): TensorData {
    return binaryOp(a, b, (x, y) => x + y);
  }

  sub(a: TensorData, b: TensorData): TensorData {
    return binaryOp(a, b, (x, y) => x - y);
  }

  mul(a: TensorData, b: TensorData): TensorData {
    return binaryOp(a, b, (x, y) => x * y);
  }

  div(a: TensorData, b: TensorData): TensorData {
    return binaryOp(a, b, (x, y) => x / y);
  }

  matmul(a: TensorData, b: TensorData): TensorData {
    const aShape = a.shape;
    const bShape = b.shape;
    const aNdim = aShape.length;
    const bNdim = bShape.length;

    if (aNdim < 2 || bNdim < 2) {
      throw new Error(`matmul requires at least 2D tensors, got [${aShape}] x [${bShape}]`);
    }

    const M = aShape[aNdim - 2];
    const K = aShape[aNdim - 1];
    const N = bShape[bNdim - 1];

    if (bShape[bNdim - 2] !== K) {
      throw new Error(`matmul shape mismatch: [${aShape}] x [${bShape}]`);
    }

    // Compute batch dimensions
    const aBatch = aShape.slice(0, aNdim - 2);
    const bBatch = bShape.slice(0, bNdim - 2);

    // Broadcast batch dimensions
    const maxBatchDims = Math.max(aBatch.length, bBatch.length);
    const batchShape: number[] = [];
    for (let i = 0; i < maxBatchDims; i++) {
      const ai = i < aBatch.length ? aBatch[aBatch.length - 1 - i] : 1;
      const bi = i < bBatch.length ? bBatch[bBatch.length - 1 - i] : 1;
      if (ai !== bi && ai !== 1 && bi !== 1) {
        throw new Error(`matmul batch shape mismatch: [${aShape}] x [${bShape}]`);
      }
      batchShape.unshift(Math.max(ai, bi));
    }

    let batchSize = 1;
    for (const d of batchShape) batchSize *= d;

    const outShape = [...batchShape, M, N];
    const dtype = commonDtype(a.dtype, b.dtype);
    const Ctor = dtypeArray(dtype);
    const out = new Ctor(batchSize * M * N);

    const aMK = M * K;
    const bKN = K * N;
    const oMN = M * N;

    // For each batch index, compute the corresponding a and b offsets
    // (handling broadcasting)
    const aBatchStrides: number[] = [];
    const bBatchStrides: number[] = [];
    {
      let aStride = aMK;
      for (let i = aBatch.length - 1; i >= 0; i--) {
        aBatchStrides.unshift(aBatch[i] === 1 ? 0 : aStride);
        aStride *= aBatch[i];
      }
      let bStride = bKN;
      for (let i = bBatch.length - 1; i >= 0; i--) {
        bBatchStrides.unshift(bBatch[i] === 1 ? 0 : bStride);
        bStride *= bBatch[i];
      }
    }

    for (let batch = 0; batch < batchSize; batch++) {
      // Decompose batch index into batch coordinates
      let rem = batch;
      let aOff = 0;
      let bOff = 0;
      for (let d = batchShape.length - 1; d >= 0; d--) {
        const coord = rem % batchShape[d];
        rem = (rem - coord) / batchShape[d];
        const aIdx = d - (batchShape.length - aBatch.length);
        const bIdx = d - (batchShape.length - bBatch.length);
        if (aIdx >= 0) aOff += coord * aBatchStrides[aIdx];
        if (bIdx >= 0) bOff += coord * bBatchStrides[bIdx];
      }

      const oOff = batch * oMN;
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            sum += a.data[aOff + m * K + k] * b.data[bOff + k * N + n];
          }
          out[oOff + m * N + n] = sum;
        }
      }
    }

    return makeTensor(outShape, dtype, out);
  }

  sum(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) {
      // Reduce over all elements -> scalar
      let s = 0;
      for (let i = 0; i < a.data.length; i++) s += a.data[i];
      const Ctor = dtypeArray(a.dtype);
      const shape: Shape = keepdims ? a.shape.map(() => 1) : [];
      return makeTensor(shape, a.dtype, Ctor.from([s]));
    }

    const ax = normalizeAxis(axis, a.shape.length);
    const dimSize = a.shape[ax];
    const outShape: number[] = [];
    for (let d = 0; d < a.shape.length; d++) {
      if (d === ax) {
        if (keepdims) outShape.push(1);
      } else {
        outShape.push(a.shape[d]);
      }
    }
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);

    const inStrides = shapeStrides(a.shape);
    const axStride = inStrides[ax];

    // For each element in the output, sum over the axis dimension
    for (let i = 0; i < outSize; i++) {
      // Map output flat index -> multi-index in input (skip the reduced axis)
      const outCoords = flatToMulti(i, outShape);
      // Insert the axis dimension back
      const inCoords: number[] = [];
      let oi = 0;
      for (let d = 0; d < a.shape.length; d++) {
        if (d === ax) {
          inCoords.push(0); // placeholder
        } else {
          inCoords.push(outCoords[oi++]);
        }
      }

      let s = 0;
      const baseFlat = multiToFlat(inCoords, inStrides);
      for (let j = 0; j < dimSize; j++) {
        s += a.data[baseFlat + j * axStride];
      }
      out[i] = s;
    }

    return makeTensor(outShape, a.dtype, out);
  }

  mean(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) {
      let s = 0;
      for (let i = 0; i < a.data.length; i++) s += a.data[i];
      const Ctor = dtypeArray(a.dtype);
      const shape: Shape = keepdims ? a.shape.map(() => 1) : [];
      return makeTensor(shape, a.dtype, Ctor.from([s / a.data.length]));
    }

    const sumT = this.sum(a, axis, keepdims);
    const ax = normalizeAxis(axis, a.shape.length);
    const dimSize = a.shape[ax];
    const out = dtypeArray(sumT.dtype).from(sumT.data);
    for (let i = 0; i < out.length; i++) out[i] /= dimSize;
    return makeTensor(sumT.shape, sumT.dtype, out);
  }

  // ── element-wise ────────────────────────────────────────────────────────

  neg(a: TensorData): TensorData {
    return unaryOp(a, (x) => -x);
  }

  exp(a: TensorData): TensorData {
    return unaryOp(a, Math.exp);
  }

  log(a: TensorData): TensorData {
    return unaryOp(a, Math.log);
  }

  sqrt(a: TensorData): TensorData {
    return unaryOp(a, Math.sqrt);
  }

  pow(a: TensorData, exponent: number): TensorData {
    return unaryOp(a, (x) => Math.pow(x, exponent));
  }

  scale(a: TensorData, s: number): TensorData {
    return unaryOp(a, (x) => x * s);
  }

  // ── nn ──────────────────────────────────────────────────────────────────

  embedding(weight: TensorData, indices: TensorData): TensorData {
    // weight: [vocabSize, dim], indices: arbitrary shape of int indices
    // output: [...indices.shape, dim]
    const dim = weight.shape[1];
    const outShape = [...indices.shape, dim];
    const Ctor = dtypeArray(weight.dtype);
    const out = new Ctor(shapeSize(outShape));

    const numIndices = indices.data.length;
    for (let i = 0; i < numIndices; i++) {
      const idx = indices.data[i];
      const srcOff = idx * dim;
      const dstOff = i * dim;
      for (let d = 0; d < dim; d++) {
        out[dstOff + d] = weight.data[srcOff + d];
      }
    }

    return makeTensor(outShape, weight.dtype, out);
  }

  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    // x: [..., dim], weight: [dim], bias: [dim]
    // Normalize over the last dimension.
    const shape = x.shape;
    const dim = shape[shape.length - 1];
    const outer = shapeSize(shape) / dim;
    const Ctor = dtypeArray(x.dtype);
    const out = new Ctor(x.data.length);

    for (let i = 0; i < outer; i++) {
      const off = i * dim;
      // Compute mean
      let mean = 0;
      for (let j = 0; j < dim; j++) mean += x.data[off + j];
      mean /= dim;
      // Compute variance
      let variance = 0;
      for (let j = 0; j < dim; j++) {
        const d = x.data[off + j] - mean;
        variance += d * d;
      }
      variance /= dim;
      const invStd = 1 / Math.sqrt(variance + eps);
      // Normalize, scale, shift
      for (let j = 0; j < dim; j++) {
        out[off + j] = (x.data[off + j] - mean) * invStd * weight.data[j] + bias.data[j];
      }
    }

    return makeTensor(shape, x.dtype, out);
  }

  gelu(a: TensorData): TensorData {
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    return unaryOp(a, (x) => {
      return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
    });
  }

  relu(a: TensorData): TensorData {
    return unaryOp(a, (x) => (x > 0 ? x : 0));
  }

  silu(a: TensorData): TensorData {
    return unaryOp(a, (x) => x / (1 + Math.exp(-x)));
  }

  softmax(a: TensorData, axis?: number): TensorData {
    // Default: last axis
    const ndim = a.shape.length;
    const ax = normalizeAxis(axis ?? ndim - 1, ndim);
    const dimSize = a.shape[ax];
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);

    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const totalSize = shapeSize(a.shape);

    // Iterate over all "lines" along the axis
    // Build outer iteration: all indices except the axis dimension
    const outerSize = totalSize / dimSize;

    for (let outer = 0; outer < outerSize; outer++) {
      // Compute the base flat index for this outer position.
      // Map outer index to coordinates, skipping the axis dimension.
      let rem = outer;
      const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) {
        if (d === ax) {
          coords[d] = 0;
          continue;
        }
        coords[d] = rem % a.shape[d];
        rem = (rem - coords[d]) / a.shape[d];
      }
      const base = multiToFlat(coords, strides);

      // Find max for numerical stability
      let max = -Infinity;
      for (let j = 0; j < dimSize; j++) {
        const v = a.data[base + j * axStride];
        if (v > max) max = v;
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let j = 0; j < dimSize; j++) {
        const e = Math.exp(a.data[base + j * axStride] - max);
        out[base + j * axStride] = e;
        sumExp += e;
      }

      // Normalize
      for (let j = 0; j < dimSize; j++) {
        out[base + j * axStride] /= sumExp;
      }
    }

    return makeTensor(a.shape, a.dtype, out);
  }

  logSoftmax(a: TensorData, axis?: number): TensorData {
    // log(softmax(x)) = x - max - log(sum(exp(x - max)))
    const ndim = a.shape.length;
    const ax = normalizeAxis(axis ?? ndim - 1, ndim);
    const dimSize = a.shape[ax];
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);

    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const totalSize = shapeSize(a.shape);
    const outerSize = totalSize / dimSize;

    for (let outer = 0; outer < outerSize; outer++) {
      let rem = outer;
      const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) {
        if (d === ax) {
          coords[d] = 0;
          continue;
        }
        coords[d] = rem % a.shape[d];
        rem = (rem - coords[d]) / a.shape[d];
      }
      const base = multiToFlat(coords, strides);

      // max
      let max = -Infinity;
      for (let j = 0; j < dimSize; j++) {
        const v = a.data[base + j * axStride];
        if (v > max) max = v;
      }

      // sum(exp(x - max))
      let sumExp = 0;
      for (let j = 0; j < dimSize; j++) {
        sumExp += Math.exp(a.data[base + j * axStride] - max);
      }
      const logSumExp = max + Math.log(sumExp);

      // x - logSumExp
      for (let j = 0; j < dimSize; j++) {
        out[base + j * axStride] = a.data[base + j * axStride] - logSumExp;
      }
    }

    return makeTensor(a.shape, a.dtype, out);
  }

  crossEntropy(logits: TensorData, targets: TensorData): TensorData {
    // logits: [N, C], targets: [N] of class indices. Returns scalar loss.
    const N = logits.shape[0];
    const C = logits.shape[1];

    // First compute log-softmax of logits
    const logProbs = this.logSoftmax(logits, 1);

    // Pick the log-probability of the correct class for each sample
    let loss = 0;
    for (let i = 0; i < N; i++) {
      const cls = targets.data[i];
      loss -= logProbs.data[i * C + cls];
    }
    loss /= N;

    const Ctor = dtypeArray(logits.dtype);
    return makeTensor([], logits.dtype, Ctor.from([loss]));
  }

  // ── reshape / slice ─────────────────────────────────────────────────────

  reshape(a: TensorData, shape: Shape): TensorData {
    const newSize = shapeSize(shape);
    if (newSize !== shapeSize(a.shape)) {
      throw new Error(`Cannot reshape [${a.shape}] to [${shape}]: size mismatch`);
    }
    // Data is contiguous, just reinterpret with new shape
    const Ctor = dtypeArray(a.dtype);
    const out = Ctor.from(a.data);
    return makeTensor(shape, a.dtype, out);
  }

  transpose(a: TensorData, dim0: number, dim1: number): TensorData {
    const ndim = a.shape.length;
    const d0 = normalizeAxis(dim0, ndim);
    const d1 = normalizeAxis(dim1, ndim);

    // Build new shape
    const newShape = [...a.shape];
    newShape[d0] = a.shape[d1];
    newShape[d1] = a.shape[d0];

    const srcStrides = shapeStrides(a.shape);
    const totalSize = shapeSize(a.shape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(totalSize);
    const dstStrides = shapeStrides(newShape);

    for (let i = 0; i < totalSize; i++) {
      // Decompose flat index in source
      const coords = flatToMulti(i, a.shape);
      // Swap dimensions
      const tmp = coords[d0];
      coords[d0] = coords[d1];
      coords[d1] = tmp;
      const dstIdx = multiToFlat(coords, dstStrides);
      out[dstIdx] = a.data[i];
    }

    return makeTensor(newShape, a.dtype, out);
  }

  slice(a: TensorData, starts: number[], ends: number[]): TensorData {
    const ndim = a.shape.length;
    const outShape: number[] = new Array(ndim);
    for (let d = 0; d < ndim; d++) {
      outShape[d] = ends[d] - starts[d];
    }
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);

    const srcStrides = shapeStrides(a.shape);
    const dstStrides = shapeStrides(outShape);

    for (let i = 0; i < outSize; i++) {
      const coords = flatToMulti(i, outShape);
      // Offset coordinates by starts
      let srcFlat = 0;
      for (let d = 0; d < ndim; d++) {
        srcFlat += (coords[d] + starts[d]) * srcStrides[d];
      }
      out[i] = a.data[srcFlat];
    }

    return makeTensor(outShape, a.dtype, out);
  }

  cat(tensors: TensorData[], axis: number): TensorData {
    if (tensors.length === 0) throw new Error("cat: empty tensor list");
    const ndim = tensors[0].shape.length;
    const ax = normalizeAxis(axis, ndim);
    const dtype = tensors[0].dtype;

    // Compute output shape
    const outShape = [...tensors[0].shape];
    for (let t = 1; t < tensors.length; t++) {
      for (let d = 0; d < ndim; d++) {
        if (d === ax) {
          outShape[d] += tensors[t].shape[d];
        } else if (tensors[t].shape[d] !== outShape[d]) {
          throw new Error(`cat: shape mismatch at dim ${d}`);
        }
      }
    }

    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(dtype);
    const out = new Ctor(outSize);
    const outStrides = shapeStrides(outShape);

    let axOffset = 0;
    for (let t = 0; t < tensors.length; t++) {
      const src = tensors[t];
      const srcStrides = shapeStrides(src.shape);
      const srcSize = shapeSize(src.shape);

      for (let i = 0; i < srcSize; i++) {
        const coords = flatToMulti(i, src.shape);
        coords[ax] += axOffset;
        const dstIdx = multiToFlat(coords, outStrides);
        out[dstIdx] = src.data[i];
      }
      axOffset += src.shape[ax];
    }

    return makeTensor(outShape, dtype, out);
  }

  // ── utility ─────────────────────────────────────────────────────────────

  argmax(a: TensorData, axis?: number): TensorData {
    if (axis === undefined) {
      // Global argmax -> scalar
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let i = 0; i < a.data.length; i++) {
        if (a.data[i] > maxVal) {
          maxVal = a.data[i];
          maxIdx = i;
        }
      }
      return makeTensor([], "i32", Int32Array.from([maxIdx]));
    }

    const ndim = a.shape.length;
    const ax = normalizeAxis(axis, ndim);
    const dimSize = a.shape[ax];

    // Output shape: remove the axis dimension
    const outShape: number[] = [];
    for (let d = 0; d < ndim; d++) {
      if (d !== ax) outShape.push(a.shape[d]);
    }
    if (outShape.length === 0) outShape.push(1);

    const outSize = shapeSize(outShape);
    const out = new Int32Array(outSize);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];

    for (let i = 0; i < outSize; i++) {
      const outCoords = flatToMulti(i, outShape);
      // Build input coords (insert axis dim = 0)
      const inCoords: number[] = [];
      let oi = 0;
      for (let d = 0; d < ndim; d++) {
        if (d === ax) {
          inCoords.push(0);
        } else {
          inCoords.push(outCoords[oi++]);
        }
      }
      const base = multiToFlat(inCoords, strides);

      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let j = 0; j < dimSize; j++) {
        const v = a.data[base + j * axStride];
        if (v > maxVal) {
          maxVal = v;
          maxIdx = j;
        }
      }
      out[i] = maxIdx;
    }

    return makeTensor(outShape, "i32", out);
  }

  topk(
    a: TensorData,
    k: number,
    axis?: number,
  ): { values: TensorData; indices: TensorData } {
    const ndim = a.shape.length;
    const ax = normalizeAxis(axis ?? ndim - 1, ndim);
    const dimSize = a.shape[ax];

    if (k > dimSize) throw new Error(`topk: k=${k} exceeds axis size ${dimSize}`);

    // Output shape: same as input but with axis dim = k
    const outShape = [...a.shape];
    outShape[ax] = k;
    const outSize = shapeSize(outShape);

    const Ctor = dtypeArray(a.dtype);
    const valuesOut = new Ctor(outSize);
    const indicesOut = new Int32Array(outSize);

    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const outerSize = shapeSize(a.shape) / dimSize;
    const outStrides = shapeStrides(outShape);

    for (let outer = 0; outer < outerSize; outer++) {
      // Get base coords for this line
      let rem = outer;
      const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) {
        if (d === ax) {
          coords[d] = 0;
          continue;
        }
        coords[d] = rem % a.shape[d];
        rem = (rem - coords[d]) / a.shape[d];
      }
      const base = multiToFlat(coords, strides);

      // Collect (value, index) pairs
      const pairs: Array<[number, number]> = new Array(dimSize);
      for (let j = 0; j < dimSize; j++) {
        pairs[j] = [a.data[base + j * axStride], j];
      }
      // Sort descending by value
      pairs.sort((x, y) => y[0] - x[0]);

      // Write top-k into output
      const outBase = multiToFlat(coords, outStrides);
      const outAxStride = outStrides[ax];
      for (let j = 0; j < k; j++) {
        valuesOut[outBase + j * outAxStride] = pairs[j][0];
        indicesOut[outBase + j * outAxStride] = pairs[j][1];
      }
    }

    return {
      values: makeTensor(outShape, a.dtype, valuesOut),
      indices: makeTensor(outShape, "i32", indicesOut),
    };
  }

  gather(a: TensorData, axis: number, indices: TensorData): TensorData {
    // Like PyTorch gather: output[i][j][k] = input[i][indices[i][j][k]][k] (for axis=1)
    const ndim = a.shape.length;
    const ax = normalizeAxis(axis, ndim);
    const outShape = [...indices.shape];
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);

    const aStrides = shapeStrides(a.shape);
    const idxStrides = shapeStrides(indices.shape);

    for (let i = 0; i < outSize; i++) {
      const coords = flatToMulti(i, outShape);
      // In the source tensor, replace the axis coordinate with the index value
      const srcCoords = [...coords];
      srcCoords[ax] = indices.data[i];
      const srcFlat = multiToFlat(srcCoords, aStrides);
      out[i] = a.data[srcFlat];
    }

    return makeTensor(outShape, a.dtype, out);
  }

  clone(a: TensorData): TensorData {
    const Ctor = dtypeArray(a.dtype);
    return makeTensor(a.shape, a.dtype, Ctor.from(a.data));
  }

  // ── comparison ──────────────────────────────────────────────────────────

  equal(a: TensorData, b: TensorData): boolean {
    if (a.shape.length !== b.shape.length) return false;
    for (let d = 0; d < a.shape.length; d++) {
      if (a.shape[d] !== b.shape[d]) return false;
    }
    if (a.dtype !== b.dtype) return false;
    for (let i = 0; i < a.data.length; i++) {
      if (a.data[i] !== b.data[i]) return false;
    }
    return true;
  }

  allClose(a: TensorData, b: TensorData, atol = 1e-5, rtol = 1e-8): boolean {
    if (a.shape.length !== b.shape.length) return false;
    for (let d = 0; d < a.shape.length; d++) {
      if (a.shape[d] !== b.shape[d]) return false;
    }
    for (let i = 0; i < a.data.length; i++) {
      const diff = Math.abs(a.data[i] - b.data[i]);
      if (diff > atol + rtol * Math.abs(b.data[i])) return false;
    }
    return true;
  }

  // ── mask ─────────────────────────────────────────────────────────────────

  causalMask(size: number): TensorData {
    // [size, size] with 0 where attend (lower triangle + diagonal), -Infinity upper triangle
    const out = new Float32Array(size * size);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        out[i * size + j] = j > i ? -Infinity : 0;
      }
    }
    return makeTensor([size, size], "f32", out);
  }

  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData {
    const Ctor = dtypeArray(a.dtype);
    const out = Ctor.from(a.data);
    const maskSize = mask.data.length;
    const aSize = a.data.length;

    if (maskSize === aSize) {
      // Same size — direct
      for (let i = 0; i < out.length; i++) {
        if (mask.data[i] !== 0) out[i] = value;
      }
    } else {
      // Broadcasting: mask is smaller, tile it over leading dims
      for (let i = 0; i < aSize; i++) {
        if (mask.data[i % maskSize] !== 0) out[i] = value;
      }
    }
    return makeTensor(a.shape, a.dtype, out);
  }
}
