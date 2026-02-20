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

import { getNative, initDevice, getDeviceInfo, type NativeAddon } from "./device.js";
import { getKernelSpirv } from "./kernels.js";

// ── Config ──────────────────────────────────────────────────────────────────

/**
 * Default minimum number of elements to use GPU. Below this, CPU is faster.
 * Currently set high because per-dispatch overhead (descriptor pool/set/cmd buffer
 * creation) dominates for small element-wise ops. GPU matmul will lower this.
 */
const DEFAULT_MIN_GPU_SIZE = 1_000_000;

const WG_CANDIDATES = [64, 128, 256, 512] as const;
let WG_SIZE = 256;  // default, overridden by auto-tuning
let wgAutoTuned = false;

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

// ── Auto-tune workgroup size ────────────────────────────────────────────────

function autoTuneWgSize(vk: NativeAddon): void {
  if (wgAutoTuned) return;
  wgAutoTuned = true;

  try {
    // Benchmark with a 256K element add kernel (large enough to saturate, small enough to be fast)
    const testSize = 262144;
    const byteSize = testSize * 4;
    const bufA = vk.createBuffer(byteSize, 0);
    const bufB = vk.createBuffer(byteSize, 0);
    const bufC = vk.createBuffer(byteSize, 0);
    const pushBuf = new Float32Array(2);

    let bestTime = Infinity;
    let bestWg = 256;

    for (const wg of WG_CANDIDATES) {
      // Create a pipeline with this WG size
      const spirv = getKernelSpirv("add_vec4", wg);
      const pipe = vk.createPipeline(spirv, 3, PUSH_SIZE);

      const vecSize = testSize >> 2;
      pushBuf[0] = vecSize;
      pushBuf[1] = 0;
      const groups = Math.ceil(vecSize / wg);

      // Warm up
      vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf);

      // Timed runs
      let total = 0;
      const iters = 5;
      for (let i = 0; i < iters; i++) {
        total += vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf);
      }
      const avg = total / iters;

      if (avg < bestTime) {
        bestTime = avg;
        bestWg = wg;
      }
    }

    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    vk.destroyBuffer(bufC);

    WG_SIZE = bestWg;
  } catch {
    // If timestamp queries aren't supported, keep default WG_SIZE=256
    WG_SIZE = 256;
  }
}

// ── Pipeline cache ──────────────────────────────────────────────────────────

const pipelineCache = new Map<string, number>();

function getPipeline(vk: NativeAddon, name: string, numBindings: number, pushSize = PUSH_SIZE): number {
  const key = `${name}:${numBindings}:${pushSize}`;
  let handle = pipelineCache.get(key);
  if (handle !== undefined) return handle;

  const spirv = getKernelSpirv(name, WG_SIZE);
  handle = vk.createPipeline(spirv, numBindings, pushSize);
  pipelineCache.set(key, handle);
  return handle;
}

// ── Buffer pool (device-local) ──────────────────────────────────────────────

const POOL_MAX_PER_SIZE = 8;
const bufferPool = new Map<number, number[]>();

function acquireBuffer(vk: NativeAddon, byteSize: number): number {
  const pool = bufferPool.get(byteSize);
  if (pool && pool.length > 0) return pool.pop()!;
  return vk.createBuffer(byteSize, 0); // device-local (staging handled in C)
}

function releaseBuffer(vk: NativeAddon, handle: number, byteSize: number): void {
  let pool = bufferPool.get(byteSize);
  if (!pool) { pool = []; bufferPool.set(byteSize, pool); }
  if (pool.length < POOL_MAX_PER_SIZE) {
    pool.push(handle);
  } else {
    vk.destroyBuffer(handle);
  }
}

// Push constant data — reusable typed array (8 bytes = 2 x f32: [len, scalar])
const pushData = new Float32Array(2);
const PUSH_SIZE = 8;  // bytes — all kernels use 2 x f32 push constants

// ── GPU residence tracking ──────────────────────────────────────────────────

interface GpuHandle { handle: number; byteSize: number }

/** Maps TensorData → its GPU buffer. Keyed on the object identity. */
const gpuResidence = new WeakMap<object, GpuHandle>();

/** Auto-release GPU buffers when TensorData is garbage collected. */
const gpuCleanup = new FinalizationRegistry<GpuHandle>((info) => {
  try {
    releaseBuffer(getNative(), info.handle, info.byteSize);
  } catch { /* device may be destroyed */ }
});

/** Get or create a GPU buffer for a TensorData. Returns the buffer handle. */
function ensureGpu(vk: NativeAddon, td: TensorData): number {
  const existing = gpuResidence.get(td);
  if (existing) return existing.handle;
  // Upload to a new device-local buffer
  const byteSize = td.data.length * 4;
  const handle = acquireBuffer(vk, byteSize);
  vk.uploadBuffer(handle, toF32(td));
  const info: GpuHandle = { handle, byteSize };
  gpuResidence.set(td, info);
  gpuCleanup.register(td, info);
  return handle;
}

// ── Timeline-aware output buffer pool ────────────────────────────────────────

interface OutputRegion {
  handle: number;
  byteSize: number;
  readyValue: number;  // timeline value when this region becomes available
}

const outputPool = new Map<number, OutputRegion[]>();

function acquireOutputRegion(vk: NativeAddon, byteSize: number): OutputRegion {
  const completed = vk.getCompleted();
  const pool = outputPool.get(byteSize);
  if (pool) {
    for (let i = 0; i < pool.length; i++) {
      if (pool[i].readyValue <= completed) {
        return pool.splice(i, 1)[0];
      }
    }
  }
  return { handle: acquireBuffer(vk, byteSize), byteSize, readyValue: 0 };
}

function releaseOutputRegion(region: OutputRegion, submitValue: number): void {
  region.readyValue = submitValue;
  let pool = outputPool.get(region.byteSize);
  if (!pool) { pool = []; outputPool.set(region.byteSize, pool); }
  if (pool.length < POOL_MAX_PER_SIZE) {
    pool.push(region);
  }
  // Don't destroy — let buffer pool handle lifecycle
}

/**
 * Create a TensorData with lazy readback. The C layer waits for the buffer's
 * timeline value on readBuffer, so we don't need to track it in TS.
 */
function lazyTensor(vk: NativeAddon, shape: Shape, region: OutputRegion, timelineValue: number): TensorData {
  let cached: Float32Array | null = null;
  const td: TensorData = {
    shape: [...shape],
    dtype: "f32",
    get data(): Float32Array {
      if (!cached) cached = vk.readBuffer(region.handle);
      return cached;
    },
  };
  // Release the output region back to the pool for future reuse
  // (it won't actually be reused until the timeline reaches this value)
  releaseOutputRegion(region, timelineValue);
  return td;
}

// ── Compute graph / lazy evaluation ──────────────────────────────────────────

const MAX_PENDING_OPS = 64; // auto-flush when this many ops are pending

type PendingOpKind = "binary" | "unary" | "softmax" | "layernorm" | "matmul" | "reduce_sum";

interface PendingOp {
  kind: PendingOpKind;
  kernel: string;
  pipeline: number;
  inputBufs: number[];     // GPU buffer handles for inputs
  outputRegion: OutputRegion;
  groups: [number, number, number];
  push: Float32Array;       // snapshot of push constants (must be a copy!)
  pushSize: number;
  shape: Shape;             // output shape
}

/**
 * The compute graph accumulates GPU operations and flushes them as a
 * single batch (one command buffer submit) when results are needed.
 * This eliminates per-op submit+wait overhead: N ops go from
 * N × ~100us overhead to 1 × ~100us + N × ~2us (barrier cost).
 */
class ComputeGraph {
  private pending: PendingOp[] = [];
  private vk: NativeAddon | null = null;
  private _lastFlushTimeline = 0;

  attach(vk: NativeAddon): void { this.vk = vk; }

  get length(): number { return this.pending.length; }
  get lastFlushTimeline(): number { return this._lastFlushTimeline; }

  record(op: PendingOp): void {
    this.pending.push(op);
    if (this.pending.length >= MAX_PENDING_OPS) this.flush();
  }

  /**
   * Flush all pending ops as a single batch dispatch.
   * Returns the timeline value for the batch, or the last flush value if nothing pending.
   */
  flush(): number {
    if (this.pending.length === 0 || !this.vk) return this._lastFlushTimeline;
    const vk = this.vk;
    const ops = this.pending;
    this.pending = [];

    vk.batchBegin();
    for (const op of ops) {
      vk.batchDispatch(
        op.pipeline,
        [...op.inputBufs, op.outputRegion.handle],
        op.groups[0], op.groups[1], op.groups[2],
        op.push,
      );
    }
    const tv = vk.batchSubmit();
    this._lastFlushTimeline = tv;

    // Release all output regions with the batch timeline value
    for (const op of ops) {
      releaseOutputRegion(op.outputRegion, tv);
    }

    return tv;
  }
}

/** Global compute graph instance. */
const graph = new ComputeGraph();

/**
 * Create a TensorData with graph-aware lazy readback.
 * Accessing .data flushes the compute graph first, then reads from GPU.
 */
function graphLazyTensor(vk: NativeAddon, shape: Shape, region: OutputRegion): TensorData {
  let cached: Float32Array | null = null;
  const gpuInfo: GpuHandle = { handle: region.handle, byteSize: shapeSize(shape) * 4 };
  const td: TensorData = {
    shape: [...shape],
    dtype: "f32",
    get data(): Float32Array {
      if (!cached) {
        // Flush any pending ops that might write to our output
        const tv = graph.flush();
        // Wait for the batch to complete on GPU before reading
        if (tv > 0) vk.waitTimeline(tv);
        cached = vk.readBuffer(region.handle);
      }
      return cached;
    },
  };
  // Track GPU residence so subsequent ops can find this buffer
  gpuResidence.set(td, gpuInfo);
  gpuCleanup.register(td, gpuInfo);
  return td;
}

// ── HeliosBackend ───────────────────────────────────────────────────────────

export class HeliosBackend implements Backend {
  readonly name = "helios";
  private readonly rng = new SeededRng(42);
  private initialized = false;
  private _minGpuSize = DEFAULT_MIN_GPU_SIZE;
  private _f16Supported = false;

  /** Override the minimum element count for GPU dispatch (useful for benchmarking). */
  setMinGpuSize(n: number): void { this._minGpuSize = n; }

  private init(): NativeAddon {
    if (!this.initialized) {
      const info = initDevice();
      this._f16Supported = info.f16Supported;
      const vk = getNative();
      graph.attach(vk);
      autoTuneWgSize(vk);
      this.initialized = true;
      return vk;
    }
    return getNative();
  }

  /** Flush the compute graph — executes all pending GPU ops as a single batch. */
  flush(): void { graph.flush(); }

  /** Whether this device supports f16 storage buffers. */
  get f16Supported(): boolean { return this._f16Supported; }

  // ── GPU binary ops ──────────────────────────────────────────────────────

  private gpuBinaryOp(a: TensorData, b: TensorData, kernelName: string): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);
    const byteSize = size * 4;

    // Use vec4 kernel when size is aligned (4x throughput)
    const useVec4 = (size & 3) === 0;
    const pipeline = getPipeline(vk, useVec4 ? `${kernelName}_vec4` : kernelName, 3);

    // Reuse GPU buffers if inputs already on GPU (skips upload)
    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const region = acquireOutputRegion(vk, byteSize);

    // Push constants: [len, unused] — must snapshot since pushData is reused
    const effectiveSize = useVec4 ? size >> 2 : size;
    const push = new Float32Array([effectiveSize, 0]);
    const groups = Math.ceil(effectiveSize / WG_SIZE);

    // Record to compute graph — deferred execution
    graph.record({
      kind: "binary",
      kernel: kernelName,
      pipeline,
      inputBufs: [bufA, bufB],
      outputRegion: region,
      groups: [groups, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: a.shape,
    });

    return graphLazyTensor(vk, a.shape, region);
  }

  private gpuUnaryOp(a: TensorData, kernelName: string, scalar = 0): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);
    const byteSize = size * 4;

    // Use vec4 kernel when size is aligned (4x throughput)
    const useVec4 = (size & 3) === 0;
    const pipeline = getPipeline(vk, useVec4 ? `${kernelName}_vec4` : kernelName, 2);

    // Reuse GPU buffer if input already on GPU
    const bufA = ensureGpu(vk, a);
    const region = acquireOutputRegion(vk, byteSize);

    // Push constants: [len, scalar] — must snapshot
    const effectiveSize = useVec4 ? size >> 2 : size;
    const push = new Float32Array([effectiveSize, scalar]);
    const groups = Math.ceil(effectiveSize / WG_SIZE);

    // Record to compute graph — deferred execution
    graph.record({
      kind: "unary",
      kernel: kernelName,
      pipeline,
      inputBufs: [bufA],
      outputRegion: region,
      groups: [groups, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: a.shape,
    });

    return graphLazyTensor(vk, a.shape, region);
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
    if (size >= this._minGpuSize && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "add");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x + y);
  }

  sub(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "sub");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x - y);
  }

  mul(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "mul");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x * y);
  }

  div(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize && this.shapesEqual(a.shape, b.shape)) {
      return this.gpuBinaryOp(a, b, "div");
    }
    return this.cpuBinaryOp(a, b, (x, y) => x / y);
  }

  // ── Backend interface: element-wise ─────────────────────────────────────

  neg(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "neg");
    return this.cpuUnary(a, (x) => -x);
  }

  exp(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "exp");
    return this.cpuUnary(a, Math.exp);
  }

  log(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "log");
    return this.cpuUnary(a, Math.log);
  }

  sqrt(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "sqrt");
    return this.cpuUnary(a, Math.sqrt);
  }

  pow(a: TensorData, exponent: number): TensorData {
    // GPU pow kernel not yet implemented, CPU fallback
    return this.cpuUnary(a, (x) => Math.pow(x, exponent));
  }

  scale(a: TensorData, s: number): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "scale", s);
    return this.cpuUnary(a, (x) => x * s);
  }

  gelu(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "gelu");
    const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    return this.cpuUnary(a, (x) =>
      0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
    );
  }

  relu(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "relu");
    return this.cpuUnary(a, (x) => (x > 0 ? x : 0));
  }

  silu(a: TensorData): TensorData {
    if (shapeSize(a.shape) >= this._minGpuSize) return this.gpuUnaryOp(a, "silu");
    return this.cpuUnary(a, (x) => x / (1 + Math.exp(-x)));
  }

  // ── Backend interface: matmul (CPU for now, GPU matmul kernel coming) ───

  matmul(a: TensorData, b: TensorData): TensorData {
    const aNdim = a.shape.length, bNdim = b.shape.length;
    if (aNdim >= 2 && bNdim >= 2) {
      const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
      if (M * N >= this._minGpuSize) return this.gpuMatmul(a, b);
    }
    return this.cpuMatmul(a, b);
  }

  // ── Backend interface: reductions ───────────────────────────────────────

  sum(a: TensorData, axis?: number, keepdims = false): TensorData {
    // GPU sum: only for full reduction (no axis) on large tensors
    if (axis === undefined && shapeSize(a.shape) >= this._minGpuSize) {
      return this.gpuReduceSum(a, keepdims);
    }
    return this.cpuSum(a, axis, keepdims);
  }

  mean(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined && shapeSize(a.shape) >= this._minGpuSize) {
      const s = this.gpuReduceSum(a, false);
      const n = shapeSize(a.shape);
      const result = s.data[0] / n;
      return makeTensor(
        keepdims ? a.shape.map(() => 1) : [],
        a.dtype,
        dtypeArray(a.dtype).from([result]),
      );
    }
    return this.cpuMean(a, axis, keepdims);
  }

  // ── GPU reductions ─────────────────────────────────────────────────────

  private gpuReduceSum(a: TensorData, keepdims: boolean): TensorData {
    const vk = this.init();
    // Flush pending graph ops since reduction does its own multi-pass dispatches
    graph.flush();

    const totalSize = shapeSize(a.shape);
    const pipeline = getPipeline(vk, "sum_reduce", 2);

    let inputBuf = ensureGpu(vk, a);
    let remaining = totalSize;

    // Multi-pass reduction: each pass reduces by WG_SIZE
    while (remaining > 1) {
      const numGroups = Math.ceil(remaining / WG_SIZE);
      const outByteSize = numGroups * 4;
      const region = acquireOutputRegion(vk, outByteSize);

      pushData[0] = remaining; // total elements for this pass
      pushData[1] = 0;

      const tv = vk.dispatch(pipeline, [inputBuf, region.handle], numGroups, 1, 1, pushData);

      inputBuf = region.handle;
      releaseOutputRegion(region, tv);
      remaining = numGroups;
    }

    // Read the single result
    const result = vk.readBuffer(inputBuf);
    const outShape = keepdims ? a.shape.map(() => 1) : [];
    return makeTensor(outShape, a.dtype, dtypeArray(a.dtype).from([result[0]]));
  }

  // ── GPU softmax/layerNorm ───────────────────────────────────────────────

  private gpuSoftmax(a: TensorData): TensorData {
    const vk = this.init();
    const dim = a.shape[a.shape.length - 1];
    const numRows = shapeSize(a.shape) / dim;
    const byteSize = shapeSize(a.shape) * 4;

    const pipeline = getPipeline(vk, "softmax", 2);
    const bufA = ensureGpu(vk, a);
    const region = acquireOutputRegion(vk, byteSize);

    const push = new Float32Array([dim, numRows]);

    graph.record({
      kind: "softmax",
      kernel: "softmax",
      pipeline,
      inputBufs: [bufA],
      outputRegion: region,
      groups: [numRows, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: a.shape,
    });

    return graphLazyTensor(vk, a.shape, region);
  }

  private gpuLayerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    const vk = this.init();
    const dim = x.shape[x.shape.length - 1];
    const numRows = shapeSize(x.shape) / dim;
    const byteSize = shapeSize(x.shape) * 4;

    const pipeline = getPipeline(vk, "layernorm", 4);
    const bufX = ensureGpu(vk, x);
    const bufW = ensureGpu(vk, weight);
    const bufB = ensureGpu(vk, bias);
    const region = acquireOutputRegion(vk, byteSize);

    const push = new Float32Array([dim, eps]);

    graph.record({
      kind: "layernorm",
      kernel: "layernorm",
      pipeline,
      inputBufs: [bufX, bufW, bufB],
      outputRegion: region,
      groups: [numRows, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: x.shape,
    });

    return graphLazyTensor(vk, x.shape, region);
  }

  private gpuMatmul(a: TensorData, b: TensorData): TensorData {
    const vk = this.init();
    const aNdim = a.shape.length, bNdim = b.shape.length;
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;

    const TILE = 16;
    const pipeline = getPipeline(vk, "matmul", 3, 16);

    if (batchSize === 1) {
      const bufA = ensureGpu(vk, a);
      const bufB = ensureGpu(vk, b);
      const outBytes = M * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      const push = new Float32Array([M, N, K, 0]);
      const gX = Math.ceil(N / TILE);
      const gY = Math.ceil(M / TILE);

      graph.record({
        kind: "matmul",
        kernel: "matmul",
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: region,
        groups: [gX, gY, 1],
        push,
        pushSize: 16,
        shape: [...aBatch, M, N],
      });

      return graphLazyTensor(vk, [...aBatch, M, N], region);
    } else {
      return this.cpuMatmul(a, b);
    }
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
    if (shapeSize(x.shape) >= this._minGpuSize) {
      return this.gpuLayerNorm(x, weight, bias, eps);
    }
    return this.cpuLayerNorm(x, weight, bias, eps);
  }

  softmax(a: TensorData, axis?: number): TensorData {
    const ndim = a.shape.length;
    const ax = axis !== undefined ? (axis < 0 ? axis + ndim : axis) : ndim - 1;
    // GPU path: when axis is last dim and tensor is large enough
    if (ax === ndim - 1 && shapeSize(a.shape) >= this._minGpuSize) {
      return this.gpuSoftmax(a);
    }
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

  // ── Profiling ──────────────────────────────────────────────────────────

  /**
   * Profile a GPU kernel execution. Returns GPU-side execution time in microseconds.
   * Uses Vulkan timestamp queries for accurate GPU timing (not wall-clock).
   */
  profileOp(a: TensorData, kernelName: string, opts?: {
    b?: TensorData;
    scalar?: number;
    iters?: number;
  }): { gpuTimeUs: number; throughputGBps: number; elementsPerSec: number } {
    const vk = this.init();
    graph.flush(); // must flush before synchronous GPU timing
    const size = shapeSize(a.shape);
    const byteSize = size * 4;
    const iters = opts?.iters ?? 10;

    // Determine kernel config
    const useVec4 = (size & 3) === 0;
    const actualName = useVec4 ? `${kernelName}_vec4` : kernelName;
    const numBindings = opts?.b ? 3 : 2;
    const pipeline = getPipeline(vk, actualName, numBindings);

    const bufA = ensureGpu(vk, a);
    const region = acquireOutputRegion(vk, byteSize);

    const bufs = opts?.b
      ? [bufA, ensureGpu(vk, opts.b), region.handle]
      : [bufA, region.handle];

    const effectiveSize = useVec4 ? size >> 2 : size;
    pushData[0] = effectiveSize;
    pushData[1] = opts?.scalar ?? 0;
    const groups = Math.ceil(effectiveSize / WG_SIZE);

    // Warm up (1 dispatch)
    vk.gpuTime(pipeline, bufs, groups, 1, 1, pushData);

    // Timed runs
    let totalUs = 0;
    for (let i = 0; i < iters; i++) {
      totalUs += vk.gpuTime(pipeline, bufs, groups, 1, 1, pushData);
    }

    const avgUs = totalUs / iters;
    const bytesPerOp = byteSize * (opts?.b ? 3 : 2); // read inputs + write output
    const throughputGBps = bytesPerOp / (avgUs * 1e3); // GB/s
    const elementsPerSec = size / (avgUs * 1e-6);

    releaseOutputRegion(region, 0);
    return { gpuTimeUs: avgUs, throughputGBps, elementsPerSec };
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
