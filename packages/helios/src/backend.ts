/**
 * backend.ts — HeliosBackend: GPU compute via our Vulkan native addon.
 *
 * Implements the @alpha/core Backend interface using:
 *   - native/helios_vk.node for Vulkan device/buffer/pipeline/dispatch
 *   - src/kernels.ts for SPIR-V shader generation (from scratch in TS)
 *
 * Strategy:
 *   - Operations above a size threshold run on GPU
 *   - Small operations fall back to CPU (transfer overhead > compute savings)
 *   - The GPU threshold is tunable via MIN_GPU_SIZE
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

import { getNative, initDevice, type NativeAddon } from "./device.js";
import { getKernelSpirv } from "./kernels.js";

// ── Config ──────────────────────────────────────────────────────────────────

/**
 * Minimum number of elements to use GPU. Below this, CPU is faster.
 * Currently set high because per-dispatch overhead (descriptor pool/set/cmd buffer
 * creation) dominates for small element-wise ops. GPU matmul will lower this.
 */
const MIN_GPU_SIZE = 1_000_000;

const WG_SIZE = 256;

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeTensor(shape: Shape, dtype: Dtype, data: Float32Array | Float64Array | Int32Array): TensorData {
  return { shape, dtype, data };
}

function toF32(td: TensorData): Float32Array {
  if (td.data instanceof Float32Array) return td.data;
  return Float32Array.from(td.data);
}

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
  for (let d = 0; d < coords.length; d++) idx += coords[d] * strides[d];
  return idx;
}

// ── Pipeline cache ──────────────────────────────────────────────────────────

const pipelineCache = new Map<string, number>();

function getPipeline(vk: NativeAddon, name: string, numBindings: number): number {
  const key = `${name}:${numBindings}`;
  let handle = pipelineCache.get(key);
  if (handle !== undefined) return handle;

  const spirv = getKernelSpirv(name, WG_SIZE);
  handle = vk.createPipeline(spirv, numBindings);
  pipelineCache.set(key, handle);
  return handle;
}

// ── HeliosBackend ───────────────────────────────────────────────────────────

export class HeliosBackend implements Backend {
  readonly name = "helios";
  private readonly rng = new SeededRng(42);
  private initialized = false;

  private init(): NativeAddon {
    if (!this.initialized) {
      initDevice();
      this.initialized = true;
    }
    return getNative();
  }

  // ── GPU binary ops ──────────────────────────────────────────────────────

  private gpuBinaryOp(a: TensorData, b: TensorData, kernelName: string): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);

    const pipeline = getPipeline(vk, kernelName, 4);
    const bufA = vk.createBuffer(size * 4);
    const bufB = vk.createBuffer(size * 4);
    const bufC = vk.createBuffer(size * 4);

    // Pack params: len as f32 (GPU shader reads as f32, compares in float space)
    const paramsData = new Float32Array([size]);
    const bufP = vk.createBuffer(4);

    vk.uploadBuffer(bufA, toF32(a));
    vk.uploadBuffer(bufB, toF32(b));
    vk.uploadBuffer(bufP, paramsData);

    vk.dispatch(pipeline, [bufA, bufB, bufC, bufP], Math.ceil(size / WG_SIZE));

    const result = vk.readBuffer(bufC);
    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    vk.destroyBuffer(bufC);
    vk.destroyBuffer(bufP);

    return makeTensor(a.shape, "f32", result.slice(0, size));
  }

  private gpuUnaryOp(a: TensorData, kernelName: string, scalar = 0): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);

    const numBindings = kernelName === "scale" ? 3 : 3;
    const pipeline = getPipeline(vk, kernelName, numBindings);
    const bufA = vk.createBuffer(size * 4);
    const bufC = vk.createBuffer(size * 4);

    // Pack params: len as f32, scalar as f32
    const paramsData = new Float32Array([size, scalar]);
    const bufP = vk.createBuffer(8);

    vk.uploadBuffer(bufA, toF32(a));
    vk.uploadBuffer(bufP, paramsData);

    vk.dispatch(pipeline, [bufA, bufC, bufP], Math.ceil(size / WG_SIZE));

    const result = vk.readBuffer(bufC);
    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufC);
    vk.destroyBuffer(bufP);

    return makeTensor(a.shape, "f32", result.slice(0, size));
  }

  // ── Backend interface: creation ─────────────────────────────────────────

  zeros(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const Ctor = dtypeArray(dtype);
    return makeTensor(shape, dtype, new Ctor(shapeSize(shape)));
  }

  ones(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const Ctor = dtypeArray(dtype);
    const data = new Ctor(shapeSize(shape));
    data.fill(1);
    return makeTensor(shape, dtype, data);
  }

  full(shape: Shape, value: number, dtype: Dtype = "f32"): TensorData {
    const Ctor = dtypeArray(dtype);
    const data = new Ctor(shapeSize(shape));
    data.fill(value);
    return makeTensor(shape, dtype, data);
  }

  randn(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const Ctor = dtypeArray(dtype);
    const data = new Ctor(shapeSize(shape));
    for (let i = 0; i < data.length; i++) data[i] = this.rng.nextGauss();
    return makeTensor(shape, dtype, data);
  }

  fromArray(data: number[], shape: Shape, dtype: Dtype = "f32"): TensorData {
    const size = shapeSize(shape);
    if (data.length !== size) throw new Error(`Data length ${data.length} != shape size ${size}`);
    const Ctor = dtypeArray(dtype);
    return makeTensor(shape, dtype, Ctor.from(data));
  }

  // ── Backend interface: binary math ──────────────────────────────────────

  add(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= MIN_GPU_SIZE && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "add");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x + y);
  }

  sub(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= MIN_GPU_SIZE && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "sub");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x - y);
  }

  mul(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= MIN_GPU_SIZE && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "mul");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x * y);
  }

  div(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= MIN_GPU_SIZE && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "div");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x / y);
  }

  // ── Backend interface: element-wise ─────────────────────────────────────

  neg(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= MIN_GPU_SIZE) return this.gpuUnaryOp(a, "neg");
    return this.cpuUnary(a, (x) => -x);
  }

  exp(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= MIN_GPU_SIZE) return this.gpuUnaryOp(a, "exp");
    return this.cpuUnary(a, Math.exp);
  }

  log(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= MIN_GPU_SIZE) return this.gpuUnaryOp(a, "log");
    return this.cpuUnary(a, Math.log);
  }

  sqrt(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= MIN_GPU_SIZE) return this.gpuUnaryOp(a, "sqrt");
    return this.cpuUnary(a, Math.sqrt);
  }

  pow(a: TensorData, exponent: number): TensorData {
    // GPU pow kernel not yet implemented, CPU fallback
    return this.cpuUnary(a, (x) => Math.pow(x, exponent));
  }

  scale(a: TensorData, s: number): TensorData {
    if (shapeSize(a.shape) >= MIN_GPU_SIZE) return this.gpuUnaryOp(a, "scale", s);
    return this.cpuUnary(a, (x) => x * s);
  }

  gelu(a: TensorData): TensorData {
    // GELU kernel not yet generated (needs tanh), CPU fallback
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    return this.cpuUnary(a, (x) =>
      0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
    );
  }

  relu(a: TensorData): TensorData {
    return this.cpuUnary(a, (x) => (x > 0 ? x : 0));
  }

  // ── Backend interface: matmul (CPU for now, GPU matmul kernel coming) ───

  matmul(a: TensorData, b: TensorData): TensorData {
    return this.cpuMatmul(a, b);
  }

  // ── Backend interface: reductions ───────────────────────────────────────

  sum(a: TensorData, axis?: number, keepdims = false): TensorData {
    return this.cpuSum(a, axis, keepdims);
  }

  mean(a: TensorData, axis?: number, keepdims = false): TensorData {
    return this.cpuMean(a, axis, keepdims);
  }

  // ── Backend interface: nn ops ───────────────────────────────────────────

  embedding(weight: TensorData, indices: TensorData): TensorData {
    const dim = weight.shape[1];
    const outShape = [...indices.shape, dim];
    const Ctor = dtypeArray(weight.dtype);
    const out = new Ctor(shapeSize(outShape));
    for (let i = 0; i < indices.data.length; i++) {
      const idx = indices.data[i];
      for (let d = 0; d < dim; d++) out[i * dim + d] = weight.data[idx * dim + d];
    }
    return makeTensor(outShape, weight.dtype, out);
  }

  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    return this.cpuLayerNorm(x, weight, bias, eps);
  }

  softmax(a: TensorData, axis?: number): TensorData {
    return this.cpuSoftmax(a, axis);
  }

  logSoftmax(a: TensorData, axis?: number): TensorData {
    return this.cpuLogSoftmax(a, axis);
  }

  crossEntropy(logits: TensorData, targets: TensorData): TensorData {
    const N = logits.shape[0];
    const C = logits.shape[1];
    const logProbs = this.logSoftmax(logits, 1);
    let loss = 0;
    for (let i = 0; i < N; i++) loss -= logProbs.data[i * C + targets.data[i]];
    loss /= N;
    return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([loss]));
  }

  // ── Backend interface: reshape / slice ──────────────────────────────────

  reshape(a: TensorData, shape: Shape): TensorData {
    if (shapeSize(shape) !== shapeSize(a.shape)) throw new Error(`Cannot reshape [${a.shape}] to [${shape}]`);
    return makeTensor(shape, a.dtype, dtypeArray(a.dtype).from(a.data));
  }

  transpose(a: TensorData, dim0: number, dim1: number): TensorData {
    const ndim = a.shape.length;
    const d0 = dim0 < 0 ? dim0 + ndim : dim0;
    const d1 = dim1 < 0 ? dim1 + ndim : dim1;
    const newShape = [...a.shape]; newShape[d0] = a.shape[d1]; newShape[d1] = a.shape[d0];
    const srcStrides = shapeStrides(a.shape);
    const dstStrides = shapeStrides(newShape);
    const size = shapeSize(a.shape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(size);
    for (let i = 0; i < size; i++) {
      const c = flatToMulti(i, a.shape);
      const tmp = c[d0]; c[d0] = c[d1]; c[d1] = tmp;
      out[multiToFlat(c, dstStrides)] = a.data[i];
    }
    return makeTensor(newShape, a.dtype, out);
  }

  slice(a: TensorData, starts: number[], ends: number[]): TensorData {
    const ndim = a.shape.length;
    const outShape = starts.map((s, d) => ends[d] - s);
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);
    const srcStrides = shapeStrides(a.shape);
    for (let i = 0; i < outSize; i++) {
      const coords = flatToMulti(i, outShape);
      let srcFlat = 0;
      for (let d = 0; d < ndim; d++) srcFlat += (coords[d] + starts[d]) * srcStrides[d];
      out[i] = a.data[srcFlat];
    }
    return makeTensor(outShape, a.dtype, out);
  }

  cat(tensors: TensorData[], axis: number): TensorData {
    if (tensors.length === 0) throw new Error("cat: empty");
    const ndim = tensors[0].shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const outShape = [...tensors[0].shape];
    for (let t = 1; t < tensors.length; t++) {
      for (let d = 0; d < ndim; d++) {
        if (d === ax) outShape[d] += tensors[t].shape[d];
        else if (tensors[t].shape[d] !== outShape[d]) throw new Error(`cat: shape mismatch at dim ${d}`);
      }
    }
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(tensors[0].dtype);
    const out = new Ctor(outSize);
    const outStrides = shapeStrides(outShape);
    let axOffset = 0;
    for (const src of tensors) {
      const srcStrides = shapeStrides(src.shape);
      const srcSize = shapeSize(src.shape);
      for (let i = 0; i < srcSize; i++) {
        const coords = flatToMulti(i, src.shape);
        coords[ax] += axOffset;
        out[multiToFlat(coords, outStrides)] = src.data[i];
      }
      axOffset += src.shape[ax];
    }
    return makeTensor(outShape, tensors[0].dtype, out);
  }

  // ── Backend interface: utility ──────────────────────────────────────────

  argmax(a: TensorData, axis?: number): TensorData {
    if (axis === undefined) {
      let maxVal = -Infinity, maxIdx = 0;
      for (let i = 0; i < a.data.length; i++) if (a.data[i] > maxVal) { maxVal = a.data[i]; maxIdx = i; }
      return makeTensor([], "i32", Int32Array.from([maxIdx]));
    }
    const ndim = a.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const dimSize = a.shape[ax];
    const outShape: number[] = [];
    for (let d = 0; d < ndim; d++) if (d !== ax) outShape.push(a.shape[d]);
    if (outShape.length === 0) outShape.push(1);
    const outSize = shapeSize(outShape);
    const out = new Int32Array(outSize);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    for (let i = 0; i < outSize; i++) {
      const outCoords = flatToMulti(i, outShape);
      const inCoords: number[] = []; let oi = 0;
      for (let d = 0; d < ndim; d++) inCoords.push(d === ax ? 0 : outCoords[oi++]);
      const base = multiToFlat(inCoords, strides);
      let maxVal = -Infinity, maxIdx = 0;
      for (let j = 0; j < dimSize; j++) { const v = a.data[base + j * axStride]; if (v > maxVal) { maxVal = v; maxIdx = j; } }
      out[i] = maxIdx;
    }
    return makeTensor(outShape, "i32", out);
  }

  topk(a: TensorData, k: number, axis?: number): { values: TensorData; indices: TensorData } {
    const ndim = a.shape.length;
    const ax = axis !== undefined ? (axis < 0 ? axis + ndim : axis) : ndim - 1;
    const dimSize = a.shape[ax];
    if (k > dimSize) throw new Error(`topk: k=${k} > axis size ${dimSize}`);
    const outShape = [...a.shape]; outShape[ax] = k;
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const valuesOut = new Ctor(outSize);
    const indicesOut = new Int32Array(outSize);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const outerSize = shapeSize(a.shape) / dimSize;
    const outStrides = shapeStrides(outShape);
    for (let outer = 0; outer < outerSize; outer++) {
      let rem = outer; const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) { if (d === ax) { coords[d] = 0; continue; } coords[d] = rem % a.shape[d]; rem = (rem - coords[d]) / a.shape[d]; }
      const base = multiToFlat(coords, strides);
      const pairs: [number, number][] = new Array(dimSize);
      for (let j = 0; j < dimSize; j++) pairs[j] = [a.data[base + j * axStride], j];
      pairs.sort((x, y) => y[0] - x[0]);
      const outBase = multiToFlat(coords, outStrides);
      const outAxStride = outStrides[ax];
      for (let j = 0; j < k; j++) { valuesOut[outBase + j * outAxStride] = pairs[j][0]; indicesOut[outBase + j * outAxStride] = pairs[j][1]; }
    }
    return { values: makeTensor(outShape, a.dtype, valuesOut), indices: makeTensor(outShape, "i32", indicesOut) };
  }

  gather(a: TensorData, axis: number, indices: TensorData): TensorData {
    const ndim = a.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const outSize = shapeSize(indices.shape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);
    const aStrides = shapeStrides(a.shape);
    for (let i = 0; i < outSize; i++) {
      const coords = flatToMulti(i, indices.shape);
      const srcCoords = [...coords]; srcCoords[ax] = indices.data[i];
      out[i] = a.data[multiToFlat(srcCoords, aStrides)];
    }
    return makeTensor([...indices.shape], a.dtype, out);
  }

  clone(a: TensorData): TensorData {
    return makeTensor(a.shape, a.dtype, dtypeArray(a.dtype).from(a.data));
  }

  equal(a: TensorData, b: TensorData): boolean {
    if (a.shape.length !== b.shape.length) return false;
    for (let d = 0; d < a.shape.length; d++) if (a.shape[d] !== b.shape[d]) return false;
    if (a.dtype !== b.dtype) return false;
    for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) return false;
    return true;
  }

  allClose(a: TensorData, b: TensorData, atol = 1e-5, rtol = 1e-8): boolean {
    if (a.shape.length !== b.shape.length) return false;
    for (let d = 0; d < a.shape.length; d++) if (a.shape[d] !== b.shape[d]) return false;
    for (let i = 0; i < a.data.length; i++) {
      if (Math.abs(a.data[i] - b.data[i]) > atol + rtol * Math.abs(b.data[i])) return false;
    }
    return true;
  }

  causalMask(size: number): TensorData {
    const out = new Float32Array(size * size);
    for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) out[i * size + j] = j > i ? -Infinity : 0;
    return makeTensor([size, size], "f32", out);
  }

  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData {
    const Ctor = dtypeArray(a.dtype);
    const out = Ctor.from(a.data);
    const maskSize = mask.data.length;
    for (let i = 0; i < out.length; i++) if (mask.data[i % maskSize] !== 0) out[i] = value;
    return makeTensor(a.shape, a.dtype, out);
  }

  // ── Private CPU fallbacks ───────────────────────────────────────────────

  private shapesEqual(a: Shape, b: Shape): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  private cpuUnary(a: TensorData, fn: (x: number) => number): TensorData {
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);
    for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]);
    return makeTensor(a.shape, a.dtype, out);
  }

  private cpuBinaryOp(a: TensorData, b: TensorData, fn: (x: number, y: number) => number): TensorData {
    const size = shapeSize(a.shape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(size);
    if (this.shapesEqual(a.shape, b.shape)) {
      for (let i = 0; i < size; i++) out[i] = fn(a.data[i], b.data[i]);
      return makeTensor(a.shape, a.dtype, out);
    }
    const bSize = shapeSize(b.shape);
    for (let i = 0; i < size; i++) out[i] = fn(a.data[i], b.data[i % bSize]);
    return makeTensor(a.shape, a.dtype, out);
  }

  private cpuMatmul(a: TensorData, b: TensorData): TensorData {
    const aNdim = a.shape.length, bNdim = b.shape.length;
    if (aNdim < 2 || bNdim < 2) throw new Error("matmul requires 2D+");
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
    if (b.shape[bNdim - 2] !== K) throw new Error("matmul shape mismatch");
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(batchSize * M * N);
    for (let batch = 0; batch < batchSize; batch++) {
      const aOff = batch * M * K, bOff = batch * K * N, oOff = batch * M * N;
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) sum += a.data[aOff + m * K + k] * b.data[bOff + k * N + n];
        out[oOff + m * N + n] = sum;
      }
    }
    return makeTensor([...aBatch, M, N], a.dtype, out);
  }

  private cpuSoftmax(a: TensorData, axis?: number): TensorData {
    const ndim = a.shape.length;
    const ax = axis !== undefined ? (axis < 0 ? axis + ndim : axis) : ndim - 1;
    const dimSize = a.shape[ax];
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const outerSize = shapeSize(a.shape) / dimSize;
    for (let outer = 0; outer < outerSize; outer++) {
      let rem = outer; const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) { if (d === ax) { coords[d] = 0; continue; } coords[d] = rem % a.shape[d]; rem = (rem - coords[d]) / a.shape[d]; }
      const base = multiToFlat(coords, strides);
      let mx = -Infinity;
      for (let j = 0; j < dimSize; j++) mx = Math.max(mx, a.data[base + j * axStride]);
      let s = 0;
      for (let j = 0; j < dimSize; j++) { const e = Math.exp(a.data[base + j * axStride] - mx); out[base + j * axStride] = e; s += e; }
      for (let j = 0; j < dimSize; j++) out[base + j * axStride] /= s;
    }
    return makeTensor(a.shape, a.dtype, out);
  }

  private cpuLogSoftmax(a: TensorData, axis?: number): TensorData {
    const ndim = a.shape.length;
    const ax = axis !== undefined ? (axis < 0 ? axis + ndim : axis) : ndim - 1;
    const dimSize = a.shape[ax];
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    const outerSize = shapeSize(a.shape) / dimSize;
    for (let outer = 0; outer < outerSize; outer++) {
      let rem = outer; const coords = new Array(ndim);
      for (let d = ndim - 1; d >= 0; d--) { if (d === ax) { coords[d] = 0; continue; } coords[d] = rem % a.shape[d]; rem = (rem - coords[d]) / a.shape[d]; }
      const base = multiToFlat(coords, strides);
      let mx = -Infinity;
      for (let j = 0; j < dimSize; j++) mx = Math.max(mx, a.data[base + j * axStride]);
      let s = 0;
      for (let j = 0; j < dimSize; j++) s += Math.exp(a.data[base + j * axStride] - mx);
      const lse = mx + Math.log(s);
      for (let j = 0; j < dimSize; j++) out[base + j * axStride] = a.data[base + j * axStride] - lse;
    }
    return makeTensor(a.shape, a.dtype, out);
  }

  private cpuLayerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    const dim = x.shape[x.shape.length - 1];
    const outer = shapeSize(x.shape) / dim;
    const Ctor = dtypeArray(x.dtype);
    const out = new Ctor(x.data.length);
    for (let i = 0; i < outer; i++) {
      const off = i * dim;
      let mean = 0;
      for (let j = 0; j < dim; j++) mean += x.data[off + j];
      mean /= dim;
      let variance = 0;
      for (let j = 0; j < dim; j++) { const d = x.data[off + j] - mean; variance += d * d; }
      variance /= dim;
      const invStd = 1 / Math.sqrt(variance + eps);
      for (let j = 0; j < dim; j++) out[off + j] = (x.data[off + j] - mean) * invStd * weight.data[j] + bias.data[j];
    }
    return makeTensor(x.shape, x.dtype, out);
  }

  private cpuSum(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) {
      let s = 0; for (let i = 0; i < a.data.length; i++) s += a.data[i];
      const Ctor = dtypeArray(a.dtype);
      return makeTensor(keepdims ? a.shape.map(() => 1) : [], a.dtype, Ctor.from([s]));
    }
    const ndim = a.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const dimSize = a.shape[ax];
    const outShape: number[] = [];
    for (let d = 0; d < ndim; d++) { if (d === ax) { if (keepdims) outShape.push(1); } else outShape.push(a.shape[d]); }
    const outSize = shapeSize(outShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(outSize);
    const strides = shapeStrides(a.shape);
    const axStride = strides[ax];
    for (let i = 0; i < outSize; i++) {
      const outCoords = flatToMulti(i, outShape);
      const inCoords: number[] = []; let oi = 0;
      for (let d = 0; d < ndim; d++) inCoords.push(d === ax ? 0 : outCoords[oi++]);
      const base = multiToFlat(inCoords, strides);
      let s = 0; for (let j = 0; j < dimSize; j++) s += a.data[base + j * axStride];
      out[i] = s;
    }
    return makeTensor(outShape, a.dtype, out);
  }

  private cpuMean(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) {
      let s = 0; for (let i = 0; i < a.data.length; i++) s += a.data[i];
      const Ctor = dtypeArray(a.dtype);
      return makeTensor(keepdims ? a.shape.map(() => 1) : [], a.dtype, Ctor.from([s / a.data.length]));
    }
    const sumT = this.cpuSum(a, axis, keepdims);
    const ax = axis < 0 ? axis + a.shape.length : axis;
    const out = dtypeArray(sumT.dtype).from(sumT.data);
    for (let i = 0; i < out.length; i++) out[i] /= a.shape[ax];
    return makeTensor(sumT.shape, sumT.dtype, out);
  }
}
