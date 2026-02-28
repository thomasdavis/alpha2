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
  broadcastShapes,
  broadcastIndices,
  broadcastStrides,
} from "@alpha/core";

import { getNative, initDevice, getDeviceInfo, type NativeAddon } from "./device.js";
import { getKernelSpirv } from "./kernels.js";

// ── Config ──────────────────────────────────────────────────────────────────

/**
 * Default minimum number of elements to use GPU. Below this, CPU is faster.
 * With compute graph batching (ops recorded + submitted as single command buffer),
 * per-op overhead is ~2µs. Threshold is based on when GPU compute beats CPU compute.
 * For element-wise ops on modern GPUs, crossover is ~1K-4K elements.
 */
const DEFAULT_MIN_GPU_SIZE = 4096;
const COOP_PAD_MAX_OVERHEAD = 0.20; // max tolerated element overhead from coop padding
const COOP_PAD_MIN_FLOPS = 2_000_000; // only pad large GEMMs where tensor-core win can amortize padding
const COOP_TRANSPOSED_A_MIN_FLOPS = 8_000_000; // transpose+coop path should only run when GEMM dominates transpose cost
const LARGE_TILE_THRESHOLD_DEFAULT = 65_536; // prefer tile=32 once output plane reaches this size
const MATMUL_GPU_FLOPS_THRESHOLD = 50_000; // route medium GEMMs to GPU sooner

const WG_CANDIDATES = [64, 128, 256, 512, 1024] as const;
let WG_SIZE = 256;  // default, overridden by auto-tuning
let wgAutoTuned = false;
const DISABLE_BATCH_DISPATCH_MANY = process.env.HELIOS_DISABLE_BATCH_DISPATCH_MANY === "1";
const DEBUG_COOP = process.env.HELIOS_DEBUG_COOP === "1";
const ENABLE_COOP_F16_ACCUM = process.env.HELIOS_COOP_F16_ACCUM === "1";
const ENABLE_COOP_F16IN_S2X2 = process.env.HELIOS_COOP_F16IN_S2X2 === "1";
const COOP_F16IN_S2X2_MIN_FLOPS = 20_000_000;
const MATMUL_LARGE_TILE_THRESHOLD_ENV = process.env.HELIOS_MATMUL_LARGE_TILE_THRESHOLD;
let LARGE_TILE_THRESHOLD = LARGE_TILE_THRESHOLD_DEFAULT;
if (MATMUL_LARGE_TILE_THRESHOLD_ENV) {
  const parsed = Number(MATMUL_LARGE_TILE_THRESHOLD_ENV);
  if (Number.isFinite(parsed) && parsed >= 0) {
    LARGE_TILE_THRESHOLD = Math.trunc(parsed);
  } else {
    console.warn(
      `[helios] ignoring HELIOS_MATMUL_LARGE_TILE_THRESHOLD=${MATMUL_LARGE_TILE_THRESHOLD_ENV}; expected integer >= 0`,
    );
  }
}
const WG_ENV = process.env.HELIOS_WG_SIZE;
if (WG_ENV) {
  const parsed = Number(WG_ENV);
  if (Number.isFinite(parsed) && WG_CANDIDATES.includes(parsed as any)) {
    WG_SIZE = parsed;
    wgAutoTuned = true;
  } else {
    console.warn(`[helios] ignoring HELIOS_WG_SIZE=${WG_ENV}; expected one of ${WG_CANDIDATES.join(", ")}`);
  }
}

let lastPush4A = NaN;
let lastPush4B = NaN;
let lastPush4C = NaN;
let lastPush4D = NaN;
let lastPush4Arr: Float32Array | null = null;
let lastPush2A = NaN;
let lastPush2B = NaN;
let lastPush2Arr: Float32Array | null = null;

function push2Memo(a: number, b: number): Float32Array {
  if (lastPush2Arr && a === lastPush2A && b === lastPush2B) return lastPush2Arr;
  lastPush2A = a;
  lastPush2B = b;
  lastPush2Arr = new Float32Array([a, b]);
  return lastPush2Arr;
}

function push4Memo(a: number, b: number, c: number, d = 0): Float32Array {
  if (lastPush4Arr && a === lastPush4A && b === lastPush4B && c === lastPush4C && d === lastPush4D) {
    return lastPush4Arr;
  }
  lastPush4A = a;
  lastPush4B = b;
  lastPush4C = c;
  lastPush4D = d;
  lastPush4Arr = new Float32Array([a, b, c, d]);
  return lastPush4Arr;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeTensor(shape: Shape, dtype: Dtype, data: Float32Array | Float64Array | Int32Array | Uint16Array): TensorData {
  return { shape, dtype, data };
}

function toF32(td: TensorData): Float32Array {
  if (td.data instanceof Float32Array) return td.data;
  return Float32Array.from(td.data as any);
}

/** Reinterpret Int32Array as Float32Array (preserving raw bits, no value conversion). */
function i32AsF32(data: Int32Array): Float32Array {
  return new Float32Array(data.buffer, data.byteOffset, data.length);
}

/** Convert a single f32 value to f16 bits (IEEE 754 half-precision). */
function f32ToF16Bits(val: number): number {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = val;
  const bits = new Uint32Array(buf)[0];
  const sign = (bits >> 16) & 0x8000;
  const exp = (bits >> 23) & 0xFF;
  const frac = bits & 0x7FFFFF;
  if (exp === 0xFF) return sign | 0x7C00 | (frac ? 0x200 : 0); // Inf/NaN
  const newExp = exp - 127 + 15;
  if (newExp >= 31) return sign | 0x7C00; // overflow → Inf
  if (newExp <= 0) {
    if (newExp < -10) return sign; // too small → zero
    const m = (frac | 0x800000) >> (1 - newExp);
    return sign | (m >> 13);
  }
  return sign | (newExp << 10) | (frac >> 13);
}

/** Convert f16 bits back to f32. */
function f16BitsToF32(bits: number): number {
  const sign = (bits & 0x8000) >> 15;
  const exp = (bits & 0x7C00) >> 10;
  const frac = bits & 0x3FF;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Denormalized
    let e = -14;
    let f = frac;
    while (!(f & 0x400)) { f <<= 1; e--; }
    f &= 0x3FF;
    const buf = new ArrayBuffer(4);
    new Uint32Array(buf)[0] = (sign << 31) | ((e + 127) << 23) | (f << 13);
    return new Float32Array(buf)[0];
  }
  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
  return new Float32Array(buf)[0];
}

/** Upload integer tensor to GPU preserving raw bits (for use as u32 in shaders). */
function ensureGpuRawBits(vk: NativeAddon, td: TensorData): number {
  const existing = gpuResidence.get(td);
  if (existing) return existing.handle;
  const byteSize = td.data.length * 4;
  const handle = acquireBuffer(vk, byteSize);
  // Upload raw bytes — don't convert int to float values
  if (td.data instanceof Int32Array) {
    vk.uploadBuffer(handle, i32AsF32(td.data));
  } else {
    vk.uploadBuffer(handle, toF32(td));
  }
  const info: GpuHandle = { handle, byteSize, refs: 1, released: false };
  gpuResidence.set(td, info);
  gpuCleanup.register(td, info);
  return handle;
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

function alignUp(x: number, multiple: number): number {
  return Math.ceil(x / multiple) * multiple;
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
    let anyCandidate = false;

    for (const wg of WG_CANDIDATES) {
      try {
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
        anyCandidate = true;

        if (avg < bestTime) {
          bestTime = avg;
          bestWg = wg;
        }
      } catch {
        // Skip unsupported WG sizes and continue tuning.
      }
    }

    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    vk.destroyBuffer(bufC);

    WG_SIZE = anyCandidate ? bestWg : 256;
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
let bufferPoolEntries = 0;
let bufferPoolBytes = 0;

let _totalAllocCount = 0;
let _totalAllocBytes = 0;
let _liveAllocCount = 0;

function acquireBuffer(vk: NativeAddon, byteSize: number): number {
  const pool = bufferPool.get(byteSize);
  if (pool && pool.length > 0) {
    const handle = pool.pop()!;
    bufferPoolEntries--;
    bufferPoolBytes -= byteSize;
    return handle;
  }
  _totalAllocCount++;
  _totalAllocBytes += byteSize;
  _liveAllocCount++;
  try {
    return vk.createBuffer(byteSize, 0); // device-local (staging handled in C)
  } catch (e) {
    console.error(`[helios OOM] acquireBuffer failed: requesting ${(byteSize / 1048576).toFixed(1)}MB`);
    console.error(`[helios OOM] total allocated: ${(_totalAllocBytes / 1048576).toFixed(1)}MB across ${_totalAllocCount} allocs (${_liveAllocCount} live)`);
    // Count pool sizes
    let poolBytes = 0, poolCount = 0;
    for (const [sz, bufs] of bufferPool) { poolCount += bufs.length; poolBytes += sz * bufs.length; }
    console.error(`[helios OOM] buffer pool: ${poolCount} buffers, ${(poolBytes / 1048576).toFixed(1)}MB`);
    let outPoolBytes = 0, outPoolCount = 0;
    for (const [sz, regs] of outputPool) { outPoolCount += regs.length; outPoolCount += regs.length; outPoolBytes += sz * regs.length; }
    console.error(`[helios OOM] output pool: ${outPoolCount} regions, ${(outPoolBytes / 1048576).toFixed(1)}MB`);
    throw e;
  }
}

function releaseBuffer(vk: NativeAddon, handle: number, byteSize: number): void {
  let pool = bufferPool.get(byteSize);
  if (!pool) { pool = []; bufferPool.set(byteSize, pool); }
  if (pool.length < POOL_MAX_PER_SIZE) {
    pool.push(handle);
    bufferPoolEntries++;
    bufferPoolBytes += byteSize;
  } else {
    vk.destroyBuffer(handle);
    _liveAllocCount--;
  }
}

// Push constant data — reusable typed array (8 bytes = 2 x f32: [len, scalar])
const pushData = new Float32Array(2);
const PUSH_SIZE = 8;  // bytes — all kernels use 2 x f32 push constants

// ── GPU residence tracking ──────────────────────────────────────────────────

interface GpuHandle { handle: number; byteSize: number; refs: number; released: boolean }

/** Maps TensorData → its GPU buffer. Keyed on the object identity. */
const gpuResidence = new WeakMap<object, GpuHandle>();

/**
 * Auto-release GPU buffers when TensorData is garbage collected.
 *
 * IMPORTANT: Uses graph.deferRelease() (NOT releaseBuffer()) so the buffer
 * is returned to the timeline-aware outputPool after the next graph flush.
 * This prevents buffer aliasing: FR callbacks can fire at any point during
 * normal execution (any GC event during allocation), and if we returned the
 * buffer to bufferPool directly, a pending graph operation could still be
 * referencing it. The next acquireBuffer() would grab the same handle,
 * causing two ops to share a buffer → GPU deadlock or data corruption.
 */
const gpuCleanup = new FinalizationRegistry<GpuHandle>((info) => {
  if (info.released) return; // already explicitly released
  info.refs--;
  if (info.refs <= 0) {
    info.released = true;
    try {
      graph.deferRelease({ handle: info.handle, byteSize: info.byteSize, readyValue: 0 });
    } catch { /* device may be destroyed */ }
  }
});

/** Get or create a GPU buffer for a TensorData. Returns the buffer handle. */
function ensureGpu(vk: NativeAddon, td: TensorData): number {
  const existing = gpuResidence.get(td);
  if (existing) return existing.handle;
  // Upload to a new device-local buffer
  const byteSize = td.data.length * 4;
  const handle = acquireBuffer(vk, byteSize);
  vk.uploadBuffer(handle, toF32(td));
  const info: GpuHandle = { handle, byteSize, refs: 1, released: false };
  gpuResidence.set(td, info);
  gpuCleanup.register(td, info);
  return handle;
}

/** Share GPU residence from one TensorData to another (e.g. for zero-copy reshape). */
function shareGpuResidence(src: TensorData, dst: TensorData): void {
  const gpuInfo = gpuResidence.get(src);
  if (gpuInfo) {
    gpuInfo.refs++;
    gpuResidence.set(dst, gpuInfo);
    gpuCleanup.register(dst, gpuInfo);
  }
}

/**
 * Explicitly release a TensorData's GPU buffer.
 * This is the deterministic counterpart to FinalizationRegistry-based cleanup.
 * Safe to call on non-GPU tensors (no-op) or tensors already released.
 *
 * The buffer is NOT returned to the pool immediately — it's deferred through
 * the compute graph so that pending GPU operations referencing this buffer
 * can complete first. The buffer becomes available for reuse after the next
 * graph flush, tracked by the timeline semaphore.
 */
function releaseGpuBufferFor(td: TensorData): void {
  const info = gpuResidence.get(td);
  if (!info || info.released) return;
  info.refs--;
  gpuResidence.delete(td);
  if (info.refs <= 0) {
    info.released = true;
    // Defer release through the compute graph — the buffer may be referenced
    // by pending GPU operations that haven't been submitted yet. The deferred
    // release goes to the timeline-aware outputPool after the next graph flush,
    // ensuring the buffer isn't reused until the GPU has finished with it.
    graph.deferRelease({ handle: info.handle, byteSize: info.byteSize, readyValue: 0 });
  }
}

/**
 * Invalidate the CPU-side cached data for a lazy tensor.
 * Called after in-place GPU updates (e.g. AdamW) so the next .data access
 * re-reads from the GPU buffer instead of returning stale cached values.
 */
function invalidateCache(td: TensorData): void {
  // The lazy tensor has a getter for .data that caches a Float32Array.
  // We can't directly clear that closure variable, but we can redefine .data
  // as a fresh getter that will read from GPU on next access.
  const gpuInfo = gpuResidence.get(td);
  if (!gpuInfo) return;
  const handle = gpuInfo.handle;
  const vk = getNative();
  let cached: Float32Array | null = null;
  Object.defineProperty(td, "data", {
    get(): Float32Array {
      if (!cached) {
        const tv = graph.flush();
        // Must wait for GPU to finish before reading back (batch dispatch
        // does not set per-buffer lastWriteTimeline, so readBuffer alone
        // won't wait for in-place ops like AdamW on mapped/coherent memory).
        if (tv > 0) vk.waitTimeline(tv);
        cached = vk.readBuffer(handle);
      }
      return cached;
    },
    configurable: true,
  });
}

// ── Timeline-aware output buffer pool ────────────────────────────────────────

interface OutputRegion {
  handle: number;
  byteSize: number;
  readyValue: number;  // timeline value when this region becomes available
}

const outputPool = new Map<number, OutputRegion[]>();
let outputPoolEntries = 0;
let outputPoolBytes = 0;

/**
 * Round allocation size up to coarse bins to reduce pool fragmentation.
 * Collapses many similar sizes into fewer size classes for better reuse.
 */
function roundPoolSize(bytes: number): number {
  if (bytes <= 4096) return 4096;
  if (bytes <= 1_048_576) return Math.ceil(bytes / 262144) * 262144;  // 256KB bins up to 1MB
  return Math.ceil(bytes / 4_194_304) * 4_194_304;  // 4MB bins above 1MB
}

function acquireOutputRegion(vk: NativeAddon, byteSize: number): OutputRegion {
  const rounded = roundPoolSize(byteSize);
  const completed = vk.getCompleted();
  const pool = outputPool.get(rounded);
  if (pool) {
    for (let i = 0; i < pool.length; i++) {
      if (pool[i].readyValue <= completed) {
        const region = pool.splice(i, 1)[0];
        outputPoolEntries--;
        outputPoolBytes -= rounded;
        return region;
      }
    }
  }
  return { handle: acquireBuffer(vk, rounded), byteSize: rounded, readyValue: 0 };
}

/**
 * Pending buffer destructions: handles that overflowed both outputPool and
 * bufferPool, awaiting GPU completion before they can be safely destroyed.
 * Without this, vk.destroyBuffer() on individually-allocated buffers would
 * free GPU memory while the GPU is still accessing it → undefined behavior.
 */
const pendingDestroys: { handle: number; readyValue: number }[] = [];

function releaseOutputRegion(region: OutputRegion, submitValue: number): void {
  region.readyValue = submitValue;
  let pool = outputPool.get(region.byteSize);
  if (!pool) { pool = []; outputPool.set(region.byteSize, pool); }
  if (pool.length < POOL_MAX_PER_SIZE) {
    pool.push(region);
    outputPoolEntries++;
    outputPoolBytes += region.byteSize;
  } else {
    // Pool full — defer buffer destruction until GPU has finished using it.
    // The buffer's memory may still be referenced by the just-submitted batch.
    pendingDestroys.push({ handle: region.handle, readyValue: submitValue });
  }
}

/** Destroy buffers whose GPU work has completed. Called at flush time. */
function processPendingDestroys(vk: NativeAddon): void {
  if (pendingDestroys.length === 0) return;
  const completed = vk.getCompleted();
  let writeIdx = 0;
  for (let i = 0; i < pendingDestroys.length; i++) {
    if (pendingDestroys[i].readyValue <= completed) {
      vk.destroyBuffer(pendingDestroys[i].handle);
      _liveAllocCount--;
    } else {
      pendingDestroys[writeIdx++] = pendingDestroys[i];
    }
  }
  pendingDestroys.length = writeIdx;
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

const MAX_PENDING_OPS = 2048; // auto-flush when this many ops are pending

type PendingOpKind = "binary" | "unary" | "softmax" | "layernorm" | "matmul" | "reduce_sum" | "backward" | "optimizer" | "inplace";

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
  allBufs?: number[];       // Override: use these buffers instead of inputBufs + outputRegion
  writeMask?: number;       // precomputed storage-buffer write mask for dispatch
  hasGZ?: boolean;          // whether packed dispatch stores explicit z group count
  flags?: number;           // packed dispatch flags (gY + hasGZ bit)
  packedBytes?: number;     // encoded byte size for batchDispatchMany payload
}

/**
 * The compute graph accumulates GPU operations and flushes them as a
 * single batch (one command buffer submit) when results are needed.
 * This eliminates per-op submit+wait overhead: N ops go from
 * N × ~100us overhead to 1 × ~100us + N × ~2us (barrier cost).
 */
class ComputeGraph {
  private pending: PendingOp[] = [];
  private pendingPackedBytes = 0;
  private vk: NativeAddon | null = null;
  private _lastFlushTimeline = 0;
  private deferredReleases: OutputRegion[] = [];
  totalOpsRecorded = 0;
  get deferredReleaseCount(): number { return this.deferredReleases.length; }
  opsThisStep = 0;

  attach(vk: NativeAddon): void { this.vk = vk; }

  get length(): number { return this.pending.length; }
  get lastFlushTimeline(): number { return this._lastFlushTimeline; }

  record(op: PendingOp): void {
    // Normalize per-op metadata once to avoid repeated work in flush().
    const bufCount = op.allBufs ? op.allBufs.length : (op.inputBufs.length + 1);
    if (op.writeMask === undefined) {
      op.writeMask = (op.kind === "inplace" || op.kind === "optimizer")
        ? 1
        : (1 << (bufCount - 1));
    }
    if (op.hasGZ === undefined) op.hasGZ = op.groups[2] !== 1;
    if (op.flags === undefined) op.flags = ((op.groups[1] & 0x7FFF) << 1) | (op.hasGZ ? 1 : 0);
    if (op.packedBytes === undefined) {
      op.packedBytes =
        4 + 2 + 2 + 4 +                 // pipeSlot, bufCount, flags, gX
        (op.hasGZ ? 4 : 0) +            // optional gZ
        4 +                             // writeMask
        bufCount * 4 +                  // buf handles
        op.pushSize;                    // push constants bytes
    }

    this.pending.push(op);
    this.pendingPackedBytes += op.packedBytes;
    this.totalOpsRecorded++;
    this.opsThisStep++;
    if (this.pending.length >= MAX_PENDING_OPS) this.flush();
  }

  /** Schedule an intermediate output region for release after the next flush. */
  deferRelease(region: OutputRegion): void {
    this.deferredReleases.push(region);
  }

  /**
   * Flush all pending ops as a single batch dispatch.
   * Returns the timeline value for the batch, or the last flush value if nothing pending.
   */
  flush(): number {
    // Destroy buffers from previous flushes whose GPU work has completed
    if (this.vk) processPendingDestroys(this.vk);

    if (this.pending.length === 0 || !this.vk) {
      // Even with no pending ops, release any deferred regions
      if (this.deferredReleases.length > 0) {
        for (const region of this.deferredReleases) {
          releaseOutputRegion(region, this._lastFlushTimeline);
        }
        this.deferredReleases = [];
      }
      return this._lastFlushTimeline;
    }
    const vk = this.vk;
    const ops = this.pending;
    this.pending = [];
    const packedTotalBytes = this.pendingPackedBytes;
    this.pendingPackedBytes = 0;

    vk.batchBegin();

    if (!DISABLE_BATCH_DISPATCH_MANY && typeof vk.batchDispatchMany === "function") {
      // Packed binary path: single N-API call for all dispatches.
      // Per dispatch format:
      //   int32 pipelineSlot, uint16 bufCount, uint16 flags, uint32 gX,
      //   [uint32 gZ], uint32 writeMask, int32 bufHandles[bufCount], uint8 pushData[pushSize]
      const packed = new ArrayBuffer(packedTotalBytes);
      const view = new DataView(packed);
      let offset = 0;

      for (const op of ops) {
        const bufs = op.allBufs;
        const bufCount = bufs ? bufs.length : (op.inputBufs.length + 1);
        const hasGZ = op.hasGZ!;
        const writeMask = op.writeMask!;
        const flags = op.flags!;

        view.setInt32(offset, op.pipeline, true); offset += 4;
        view.setUint16(offset, bufCount, true); offset += 2;
        view.setUint16(offset, flags, true); offset += 2;
        view.setUint32(offset, op.groups[0], true); offset += 4;
        if (hasGZ) {
          view.setUint32(offset, op.groups[2], true); offset += 4;
        }
        view.setUint32(offset, writeMask, true); offset += 4;

        // Buffer handles
        if (bufs) {
          for (let i = 0; i < bufs.length; i++) {
            view.setInt32(offset, bufs[i], true);
            offset += 4;
          }
        } else {
          for (let i = 0; i < op.inputBufs.length; i++) {
            view.setInt32(offset, op.inputBufs[i], true);
            offset += 4;
          }
          view.setInt32(offset, op.outputRegion.handle, true);
          offset += 4;
        }

        // Push constants (raw bytes from Float32Array)
        if (op.pushSize > 0) {
          const words = op.pushSize >>> 2;
          for (let i = 0; i < words; i++) {
            view.setFloat32(offset, op.push[i], true);
            offset += 4;
          }
        }
      }

      vk.batchDispatchMany(packed, ops.length);
    } else {
      // Fallback: per-dispatch N-API calls
      const wmBuf = new Uint32Array(1);
      for (const op of ops) {
        wmBuf[0] = op.writeMask!;
        const bufs = op.allBufs ?? [...op.inputBufs, op.outputRegion.handle];
        vk.batchDispatch(
          op.pipeline,
          bufs,
          op.groups[0], op.groups[1], op.groups[2],
          op.push,
          wmBuf,
        );
      }
    }

    const tv = vk.batchSubmit();
    this._lastFlushTimeline = tv;

    // Release deferred intermediate regions (e.g. from multi-pass reductions)
    for (const region of this.deferredReleases) {
      releaseOutputRegion(region, tv);
    }
    this.deferredReleases = [];

    // NOTE: We intentionally do NOT release output regions from ops here.
    // Each graphLazyTensor registers the buffer handle with the
    // FinalizationRegistry (gpuCleanup), which is the sole owner.

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
  const gpuInfo: GpuHandle = { handle: region.handle, byteSize: shapeSize(shape) * 4, refs: 1, released: false };
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

/** Like graphLazyTensor but for f16 output buffers (2 bytes per element). */
function graphLazyTensorF16(vk: NativeAddon, shape: Shape, region: OutputRegion): TensorData {
  const size = shapeSize(shape);
  const gpuInfo: GpuHandle = { handle: region.handle, byteSize: size * 2, refs: 1, released: false };
  const td: TensorData = {
    shape: [...shape],
    dtype: "f16",
    get data(): Uint16Array {
      // F16 data shouldn't normally be read back to CPU — this is a fallback
      const tv = graph.flush();
      if (tv > 0) vk.waitTimeline(tv);
      // readBuffer returns Float32Array; reinterpret as Uint16Array
      const f32 = vk.readBuffer(region.handle);
      return new Uint16Array(f32.buffer, f32.byteOffset, size);
    },
  };
  gpuResidence.set(td, gpuInfo);
  gpuCleanup.register(td, gpuInfo);
  return td;
}

// ── HeliosBackend ───────────────────────────────────────────────────────────

export interface GpuDeviceInfo {
  deviceName: string;
  vendorId: number;
  f16Supported: boolean;
  hasAsyncTransfer: boolean;
  coopMatSupported: boolean;
  coopMatM: number;
  coopMatN: number;
  coopMatK: number;
  hasPushDescriptors: boolean;
  workgroupSize: number;
  minGpuSize: number;
}

export interface CoopMatmulStats {
  totalMatmulDispatches: number;
  coopDispatches: number;
  coopDirectDispatches: number;
  coopPadded2DDispatches: number;
  coopPaddedBatchedDispatches: number;
  coopTransposedARewriteDispatches: number;
  coopHitRate: number;
}

export class HeliosBackend implements Backend {
  readonly name = "helios";
  private readonly rng = new SeededRng(42);
  private initialized = false;
  private _minGpuSize = DEFAULT_MIN_GPU_SIZE;
  private _f16Supported = false;
  private _deviceName = "";
  private _vendorId = 0;
  private _hasAsyncTransfer = false;
  private _coopMatSupported = false;
  private _coopM = 0;
  private _coopN = 0;
  private _coopK = 0;
  private _hasPushDescriptors = false;
  private _matmulDispatches = 0;
  private _coopDispatches = 0;
  private _coopDirectDispatches = 0;
  private _coopPadded2DDispatches = 0;
  private _coopPaddedBatchedDispatches = 0;
  private _coopTransposedARewriteDispatches = 0;
  private _coopF16InputCache = new WeakMap<TensorData, TensorData>();

  /** Override the minimum element count for GPU dispatch (useful for benchmarking). */
  setMinGpuSize(n: number): void { this._minGpuSize = n; }

  private init(): NativeAddon {
    if (!this.initialized) {
      const info = initDevice();
      this._f16Supported = info.f16Supported;
      this._deviceName = info.deviceName;
      this._vendorId = info.vendorId;
      this._hasAsyncTransfer = info.hasAsyncTransfer;
      this._coopMatSupported = info.coopMatSupported;
      this._coopM = info.coopMatM;
      this._coopN = info.coopMatN;
      this._coopK = info.coopMatK;
      this._hasPushDescriptors = info.hasPushDescriptors;

      // Cooperative matrix can be explicitly disabled for safety/debugging.
      const forceDisableCoop = process.env.HELIOS_DISABLE_COOP_MAT === "1";
      if (forceDisableCoop) {
        this._coopMatSupported = false;
        this._coopM = 0;
        this._coopN = 0;
        this._coopK = 0;
      }
      const vk = getNative();
      graph.attach(vk);
      autoTuneWgSize(vk);
      this.initialized = true;
      return vk;
    }
    return getNative();
  }

  private resetCoopF16InputCache(): void {
    this._coopF16InputCache = new WeakMap<TensorData, TensorData>();
  }

  /** Flush the compute graph — executes all pending GPU ops as a single batch. */
  flush(): void { graph.flush(); }

  /**
   * Flush GPU work AND wait for completion so all pending buffer releases
   * become reclaimable. Call between training steps when VRAM is tight.
   */
  syncGpu(): void {
    const vk = getNative();
    const tv = graph.flush();
    if (tv > 0) vk.waitTimeline(tv);
    processPendingDestroys(vk);
  }

  /**
   * Release all pooled GPU buffers and force-free unreachable tensor buffers.
   * Call between training steps to prevent GPU memory growth.
   */
  purgeBufferPools(): void {
    const vk = getNative();

    // Sync GPU first — output pool regions, buffer pool handles, and pending
    // destroys may all reference buffers still in use by in-flight GPU work.
    // Destroying them without waiting would cause undefined behavior.
    const tv = graph.flush();
    if (tv > 0) vk.waitTimeline(tv);

    // Drain the output pool — release all regions back to the buffer pool
    for (const [, regions] of outputPool) {
      for (const region of regions) {
        releaseBuffer(vk, region.handle, region.byteSize);
      }
    }
    outputPool.clear();
    outputPoolEntries = 0;
    outputPoolBytes = 0;

    // Drain the buffer pool — destroy all cached buffers
    for (const [, handles] of bufferPool) {
      for (const handle of handles) {
        vk.destroyBuffer(handle);
      }
    }
    bufferPool.clear();
    bufferPoolEntries = 0;
    bufferPoolBytes = 0;

    // Safe to destroy now — GPU sync above guarantees all work has completed
    processPendingDestroys(vk);
    this.resetCoopF16InputCache();
  }

  /**
   * Explicitly release a TensorData's GPU buffer, returning it to the pool.
   * Call this for intermediate tensors that are no longer needed instead of
   * relying on FinalizationRegistry, which is unreliable for timely cleanup.
   * Safe to call on non-GPU tensors (no-op) or already-released tensors.
   */
  releaseGpuTensor(td: TensorData): void {
    releaseGpuBufferFor(td);
    this._coopF16InputCache.delete(td);
  }

  /** GPU memory diagnostics: pool sizes and estimated VRAM usage. */
  gpuMemStats(): { bufferPoolEntries: number; bufferPoolBytes: number; outputPoolEntries: number; outputPoolBytes: number; deferredReleases: number; pendingDestroys: number; outputPoolSizeClasses: number; totalAllocs: number; totalAllocMB: number; liveAllocs: number } {
    return {
      bufferPoolEntries, bufferPoolBytes,
      outputPoolEntries, outputPoolBytes,
      deferredReleases: graph.deferredReleaseCount,
      pendingDestroys: pendingDestroys.length,
      outputPoolSizeClasses: outputPool.size,
      totalAllocs: _totalAllocCount,
      totalAllocMB: Math.round(_totalAllocBytes / 1024 / 1024),
      liveAllocs: _liveAllocCount,
    };
  }

  /** Detailed pool breakdown: size → count for the top N entries by bytes. */
  poolBreakdown(topN = 10): string {
    const entries: [number, number][] = [];
    for (const [size, regions] of outputPool) {
      entries.push([size, regions.length]);
    }
    entries.sort((a, b) => b[0] * b[1] - a[0] * a[1]); // sort by total bytes desc
    return entries.slice(0, topN).map(([size, count]) =>
      `${(size/1024/1024).toFixed(1)}MB×${count}`
    ).join(", ");
  }

  /** Whether this device supports f16 storage buffers. */
  get f16Supported(): boolean { return this._f16Supported; }

  /** Get GPU device info. Forces init if not already done. */
  getDeviceInfo(): GpuDeviceInfo {
    this.init();
    return {
      deviceName: this._deviceName,
      vendorId: this._vendorId,
      f16Supported: this._f16Supported,
      hasAsyncTransfer: this._hasAsyncTransfer,
      coopMatSupported: this._coopMatSupported,
      coopMatM: this._coopM,
      coopMatN: this._coopN,
      coopMatK: this._coopK,
      hasPushDescriptors: this._hasPushDescriptors,
      workgroupSize: WG_SIZE,
      minGpuSize: this._minGpuSize,
    };
  }

  getMatmulCoopStats(): CoopMatmulStats {
    const hit = this._matmulDispatches > 0 ? this._coopDispatches / this._matmulDispatches : 0;
    return {
      totalMatmulDispatches: this._matmulDispatches,
      coopDispatches: this._coopDispatches,
      coopDirectDispatches: this._coopDirectDispatches,
      coopPadded2DDispatches: this._coopPadded2DDispatches,
      coopPaddedBatchedDispatches: this._coopPaddedBatchedDispatches,
      coopTransposedARewriteDispatches: this._coopTransposedARewriteDispatches,
      coopHitRate: hit,
    };
  }

  /**
   * Run a quick GPU smoke test: dispatches a small add kernel and verifies
   * the result. Returns throughput in GB/s. Throws if GPU compute fails.
   */
  smokeTest(): { verified: boolean; throughputGBps: number } {
    this.init();
    graph.flush();

    const size = 65536;
    const a = this.full([size], 1.0);
    const b = this.full([size], 2.0);
    const c = this.add(a, b);
    graph.flush();

    // Verify result
    const data = c.data as Float32Array;
    let correct = 0;
    for (let i = 0; i < Math.min(64, data.length); i++) {
      if (Math.abs(data[i] - 3.0) < 1e-6) correct++;
    }
    const verified = correct === Math.min(64, data.length);

    // Quick throughput benchmark: 1M element add
    const benchSize = 1_048_576;
    const ba = this.full([benchSize], 1.0);
    const bb = this.full([benchSize], 2.0);
    const start = performance.now();
    for (let i = 0; i < 10; i++) {
      this.add(ba, bb);
    }
    graph.flush();
    // Force readback to include full round-trip
    const last = this.add(ba, bb);
    graph.flush();
    void (last.data as Float32Array)[0];
    const elapsed = performance.now() - start;
    const bytesPerOp = benchSize * 4 * 3; // 2 reads + 1 write
    const throughputGBps = (11 * bytesPerOp) / (elapsed * 1e6);

    return { verified, throughputGBps };
  }

  /** Get count of GPU ops dispatched this step (reset with resetStepOps). */
  get gpuOpsThisStep(): number { return graph.opsThisStep; }
  get gpuOpsTotal(): number { return graph.totalOpsRecorded; }
  resetStepOps(): void { graph.opsThisStep = 0; }

  // ── GPU binary ops ──────────────────────────────────────────────────────

  private gpuBinaryOp(a: TensorData, b: TensorData, kernelName: string, forceScalar = false): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);
    const byteSize = size * 4;

    // Use vec4 kernel when size is aligned (4x throughput)
    const useVec4 = !forceScalar && (size & 3) === 0;
    const pipeline = getPipeline(vk, useVec4 ? `${kernelName}_vec4` : kernelName, 3);

    // Reuse GPU buffers if inputs already on GPU (skips upload)
    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const region = acquireOutputRegion(vk, byteSize);

    // Push constants: [len, unused] — must snapshot since pushData is reused
    const effectiveSize = useVec4 ? size >> 2 : size;
    const push = push2Memo(effectiveSize, 0);
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
    const push = push2Memo(effectiveSize, scalar);
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

  clamp(a: TensorData, lo: number, hi: number): TensorData {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const byteSize = size * 4;
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "clamp_vec4" : "clamp";
      const CLAMP_PUSH_SIZE = 12; // 3 x f32: [len, lo, hi]
      const pipeline = getPipeline(vk, kernelName, 2, CLAMP_PUSH_SIZE);
      const bufA = ensureGpu(vk, a);
      const region = acquireOutputRegion(vk, byteSize);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const push = new Float32Array([effectiveSize, lo, hi]);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      graph.record({
        kind: "unary",
        kernel: kernelName,
        pipeline,
        inputBufs: [bufA],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: CLAMP_PUSH_SIZE,
        shape: a.shape,
      });
      return graphLazyTensor(vk, a.shape, region);
    }
    return this.cpuUnary(a, (x) => Math.max(lo, Math.min(hi, x)));
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

  // ── Backend interface: matmul ─────────────────────────────────────────

  matmul(a: TensorData, b: TensorData): TensorData {
    const aNdim = a.shape.length, bNdim = b.shape.length;
    if (aNdim >= 2 && bNdim >= 2) {
      const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
      // Use compute FLOPs (M*N*K) not output size (M*N) — matmul is compute-bound.
      // GPU wins when there's enough arithmetic to hide dispatch latency (~100K FLOPs).
      if (M * N * K >= MATMUL_GPU_FLOPS_THRESHOLD) return this.gpuMatmul(a, b);
    }
    return this.cpuMatmul(a, b);
  }

  // ── Backend interface: reductions ───────────────────────────────────────

  sum(a: TensorData, axis?: number, keepdims = false): TensorData {
    const totalSize = shapeSize(a.shape);
    if (totalSize >= this._minGpuSize) {
      // GPU full reduction (no axis)
      if (axis === undefined) return this.gpuReduceSum(a, keepdims);
      // GPU axis-specific reduction
      return this.gpuSumAxis(a, axis, keepdims);
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
    const totalSize = shapeSize(a.shape);
    const pipeline = getPipeline(vk, "sum_reduce", 2);

    let inputBuf = ensureGpu(vk, a);
    let remaining = totalSize;
    let finalRegion: OutputRegion | null = null;

    // Multi-pass reduction: each pass reduces by WG_SIZE, all recorded to graph
    while (remaining > 1) {
      const numGroups = Math.ceil(remaining / WG_SIZE);
      const outByteSize = numGroups * 4;
      const region = acquireOutputRegion(vk, outByteSize);

      const push = push2Memo(remaining, 0);

      graph.record({
        kind: "reduce_sum",
        kernel: "sum_reduce",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [numGroups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: numGroups === 1 ? (keepdims ? a.shape.map(() => 1) : []) : [numGroups],
        allBufs: [inputBuf, region.handle],
      });

      // Defer-release intermediate regions (not the final one)
      if (finalRegion) graph.deferRelease(finalRegion);

      inputBuf = region.handle;
      finalRegion = region;
      remaining = numGroups;
    }

    if (!finalRegion) return a; // single element, already reduced

    const outShape = keepdims ? a.shape.map(() => 1) : [];
    return graphLazyTensor(vk, outShape, finalRegion);
  }

  private gpuSumAxis(a: TensorData, axis: number, keepdims: boolean): TensorData {
    const vk = this.init();
    const ndim = a.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const axisSize = a.shape[ax];

    // Compute outer/inner sizes
    let outerSize = 1;
    for (let d = 0; d < ax; d++) outerSize *= a.shape[d];
    let innerSize = 1;
    for (let d = ax + 1; d < ndim; d++) innerSize *= a.shape[d];
    const totalOutput = outerSize * innerSize;

    // Output shape
    const outShape: number[] = [];
    for (let d = 0; d < ndim; d++) {
      if (d === ax) { if (keepdims) outShape.push(1); }
      else outShape.push(a.shape[d]);
    }

    const inputBuf = ensureGpu(vk, a);
    const pipeline = getPipeline(vk, "sum_axis", 2, 3 * 4);
    const region = acquireOutputRegion(vk, totalOutput * 4);
    const groups = Math.ceil(totalOutput / WG_SIZE);

    // Pack u32 push constants
    const pushF = new Float32Array(3);
    const pushU = new Uint32Array(pushF.buffer);
    pushU[0] = totalOutput;
    pushU[1] = axisSize;
    pushU[2] = innerSize;

    graph.record({
      kind: "reduce_sum",
      kernel: "sum_axis",
      pipeline,
      inputBufs: [],
      outputRegion: region,
      groups: [groups, 1, 1],
      push: pushF,
      pushSize: 3 * 4,
      shape: outShape,
      allBufs: [inputBuf, region.handle],
    });

    return graphLazyTensor(vk, outShape, region);
  }

  // ── Sum of squares (fused: square + reduce) ────────────────────────────

  sumOfSquares(data: TensorData): TensorData {
    const totalSize = shapeSize(data.shape);
    if (totalSize >= this._minGpuSize) {
      return this.gpuReduceSumOfSquares(data);
    }
    // CPU fallback: sum of element-wise squares
    const arr = data.data as Float32Array;
    let acc = 0;
    for (let i = 0; i < arr.length; i++) acc += arr[i] * arr[i];
    return makeTensor([], data.dtype, dtypeArray(data.dtype).from([acc]));
  }

  /**
   * Sum of squared norms across many tensors with a single scalar readback.
   * This keeps the accumulation on GPU instead of reading one scalar per tensor.
   */
  totalSumOfSquares(tensors: TensorData[]): TensorData {
    if (tensors.length === 0) {
      return makeTensor([], "f32", Float32Array.from([0]));
    }
    if (tensors.length === 1) return this.sumOfSquares(tensors[0]);

    const partials = new Array<TensorData>(tensors.length);
    for (let i = 0; i < tensors.length; i++) {
      partials[i] = this.sumOfSquares(tensors[i]);
    }

    // Pairwise tree reduction on GPU (forced scalar GPU add) to keep one final readback.
    // Reduce in place to avoid per-level array allocations.
    let count = partials.length;
    while (count > 1) {
      let write = 0;
      for (let i = 0; i < count; i += 2) {
        if (i + 1 < count) {
          const a = partials[i];
          const b = partials[i + 1];
          partials[write++] = this.gpuBinaryOp(a, b, "add", true);
          // Safe: release is deferred until timeline completion.
          releaseGpuBufferFor(a);
          releaseGpuBufferFor(b);
        } else {
          partials[write++] = partials[i];
        }
      }
      count = write;
    }
    return partials[0];
  }

  private gpuReduceSumOfSquares(a: TensorData): TensorData {
    const vk = this.init();
    const totalSize = shapeSize(a.shape);
    // First pass uses sum_sq_reduce (squares at input load)
    const sqPipeline = getPipeline(vk, "sum_sq_reduce", 2);
    // Subsequent passes use plain sum_reduce (just sum partial sums)
    const sumPipeline = getPipeline(vk, "sum_reduce", 2);

    let inputBuf = ensureGpu(vk, a);
    let remaining = totalSize;
    let finalRegion: OutputRegion | null = null;
    let isFirstPass = true;

    // Multi-pass reduction: each pass reduces by WG_SIZE, all recorded to graph
    while (remaining > 1) {
      const numGroups = Math.ceil(remaining / WG_SIZE);
      const outByteSize = numGroups * 4;
      const region = acquireOutputRegion(vk, outByteSize);

      const push = push2Memo(remaining, 0);
      const pipeline = isFirstPass ? sqPipeline : sumPipeline;
      const kernel = isFirstPass ? "sum_sq_reduce" : "sum_reduce";

      graph.record({
        kind: "reduce_sum",
        kernel,
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [numGroups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: numGroups === 1 ? [] : [numGroups],
        allBufs: [inputBuf, region.handle],
      });

      // Defer-release intermediate regions (not the final one)
      if (finalRegion) graph.deferRelease(finalRegion);

      inputBuf = region.handle;
      finalRegion = region;
      remaining = numGroups;
      isFirstPass = false;
    }

    if (!finalRegion) return a; // single element, already reduced

    return graphLazyTensor(vk, [], finalRegion);
  }

  // ── GPU checkFinite ─────────────────────────────────────────────────────

  /**
   * Check if a tensor contains any Inf or NaN values.
   * Returns a scalar TensorData: 0.0 = all finite, 1.0 = contains Inf/NaN.
   * Runs entirely on GPU via parallel reduction.
   */
  checkFinite(t: TensorData): TensorData {
    const size = shapeSize(t.shape);
    if (size < this._minGpuSize) {
      // CPU fallback
      const arr = t.data as Float32Array;
      for (let i = 0; i < arr.length; i++) {
        if (!isFinite(arr[i])) return makeTensor([], "f32", Float32Array.from([1.0]));
      }
      return makeTensor([], "f32", Float32Array.from([0.0]));
    }

    const vk = this.init();
    const pipeline = getPipeline(vk, "check_finite", 2);
    const bufIn = ensureGpu(vk, t);
    const region = acquireOutputRegion(vk, 4); // scalar f32

    // Zero the output buffer (so multi-workgroup writes work via store-if-nonfinite)
    vk.uploadBuffer(region.handle, new Float32Array([0.0]));

    const push = push2Memo(size, 0);
    const groups = Math.ceil(size / WG_SIZE);

    graph.record({
      kind: "reduce_sum",
      kernel: "check_finite",
      pipeline,
      inputBufs: [],
      outputRegion: region,
      groups: [groups, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: [],
      allBufs: [bufIn, region.handle],
    });

    return graphLazyTensor(vk, [], region);
  }

  // ── Dtype casting ────────────────────────────────────────────────────────

  castDtype(a: TensorData, targetDtype: Dtype): TensorData {
    if (a.dtype === targetDtype) return a;

    const size = shapeSize(a.shape);

    // f32 → f16 (GPU path)
    if (a.dtype === "f32" && targetDtype === "f16" && this._f16Supported && size >= this._minGpuSize) {
      const vk = this.init();
      const bufA = ensureGpu(vk, a);
      const pipeline = getPipeline(vk, "cast_f32_to_f16", 2);
      const outBytes = size * 2; // f16 = 2 bytes per element
      const region = acquireOutputRegion(vk, outBytes);
      const push = push2Memo(size, 0);

      graph.record({
        kind: "unary",
        kernel: "cast_f32_to_f16",
        pipeline,
        inputBufs: [bufA],
        outputRegion: region,
        groups: [Math.ceil(size / WG_SIZE), 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: a.shape,
      });

      return graphLazyTensorF16(vk, a.shape, region);
    }

    // f16 → f32 (GPU path)
    if (a.dtype === "f16" && targetDtype === "f32" && this._f16Supported && size >= this._minGpuSize) {
      const vk = this.init();
      const bufA = this.ensureGpuF16(vk, a);
      const pipeline = getPipeline(vk, "cast_f16_to_f32", 2);
      const outBytes = size * 4; // f32 = 4 bytes per element
      const region = acquireOutputRegion(vk, outBytes);
      const push = push2Memo(size, 0);

      graph.record({
        kind: "unary",
        kernel: "cast_f16_to_f32",
        pipeline,
        inputBufs: [bufA],
        outputRegion: region,
        groups: [Math.ceil(size / WG_SIZE), 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: a.shape,
      });

      return graphLazyTensor(vk, a.shape, region);
    }

    // CPU fallback: only f32↔f16 supported
    if (a.dtype === "f32" && targetDtype === "f16") {
      const f32 = a.data as Float32Array;
      const u16 = new Uint16Array(size);
      for (let i = 0; i < size; i++) u16[i] = f32ToF16Bits(f32[i]);
      return makeTensor(a.shape, "f16", u16);
    }
    if (a.dtype === "f16" && targetDtype === "f32") {
      const u16 = a.data as Uint16Array;
      const f32 = new Float32Array(size);
      for (let i = 0; i < size; i++) f32[i] = f16BitsToF32(u16[i]);
      return makeTensor(a.shape, "f32", f32);
    }

    throw new Error(`Helios: unsupported cast ${a.dtype} → ${targetDtype}`);
  }

  /** Ensure an f16 tensor's data is on GPU. F16 tensors created by GPU ops already have residence. */
  private ensureGpuF16(vk: NativeAddon, td: TensorData): number {
    const existing = gpuResidence.get(td);
    if (existing) return existing.handle;
    // F16 data from CPU — need to upload raw u16 bits
    const byteSize = td.data.length * 2;
    const handle = acquireBuffer(vk, byteSize);
    // Pack Uint16Array into Float32Array for uploadBuffer (shares underlying bytes)
    const u16 = td.data as Uint16Array;
    const paddedLen = Math.ceil(u16.length / 2);
    const f32 = new Float32Array(paddedLen);
    const f32u16 = new Uint16Array(f32.buffer);
    f32u16.set(u16);
    vk.uploadBuffer(handle, f32);
    const info: GpuHandle = { handle, byteSize, refs: 1, released: false };
    gpuResidence.set(td, info);
    gpuCleanup.register(td, info);
    return handle;
  }

  private canUseCoopMatmulDtypes(a: TensorData, b: TensorData): boolean {
    const aOk = a.dtype === "f32" || a.dtype === "f16";
    const bOk = b.dtype === "f32" || b.dtype === "f16";
    return aOk && bOk;
  }

  private getCoopInputBuffer(vk: NativeAddon, td: TensorData): number {
    if (td.dtype === "f16") return this.ensureGpuF16(vk, td);
    if (td.dtype !== "f32") throw new Error(`Helios coop matmul only supports f32/f16 inputs (got ${td.dtype})`);

    let casted = this._coopF16InputCache.get(td);
    if (!casted) {
      casted = this.castDtype(td, "f16");
      this._coopF16InputCache.set(td, casted);
    }
    return this.ensureGpuF16(vk, casted);
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

    const push = push2Memo(dim, numRows);

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

    const push = push2Memo(dim, eps);

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

  // ── GPU backward ops ──────────────────────────────────────────────────

  geluBackward(input: TensorData, gradOutput: TensorData): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      return this.gpuBinaryOp(input, gradOutput, "gelu_backward", true);
    }
    // CPU fallback
    const SQRT2PI = Math.sqrt(2 / Math.PI);
    const src = input.data as Float32Array;
    const grad = gradOutput.data as Float32Array;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      const inner = SQRT2PI * (x + 0.044715 * x * x * x);
      const tanh_val = Math.tanh(inner);
      const sech2 = 1 - tanh_val * tanh_val;
      const dInner = SQRT2PI * (1 + 3 * 0.044715 * x * x);
      out[i] = grad[i] * (0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dInner);
    }
    return makeTensor(input.shape, input.dtype, out);
  }

  reluBackward(input: TensorData, gradOutput: TensorData): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      return this.gpuBinaryOp(input, gradOutput, "relu_backward", true);
    }
    const src = input.data as Float32Array;
    const grad = gradOutput.data as Float32Array;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i] > 0 ? grad[i] : 0;
    return makeTensor(input.shape, input.dtype, out);
  }

  clampBackward(input: TensorData, gradOutput: TensorData, lo: number, hi: number): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      const vk = this.init();
      const CLAMP_BW_PUSH_SIZE = 12; // 3 x f32: [len, lo, hi]
      const pipeline = getPipeline(vk, "clamp_backward", 3, CLAMP_BW_PUSH_SIZE);
      const bufIn = ensureGpu(vk, input);
      const bufGrad = ensureGpu(vk, gradOutput);
      const region = acquireOutputRegion(vk, size * 4);
      const push = new Float32Array([size, lo, hi]);
      const groups = Math.ceil(size / WG_SIZE);
      graph.record({
        kind: "backward",
        kernel: "clamp_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: CLAMP_BW_PUSH_SIZE,
        shape: input.shape,
        allBufs: [bufIn, bufGrad, region.handle],
      });
      return graphLazyTensor(vk, input.shape, region);
    }
    // CPU fallback
    const src = input.data as Float32Array;
    const grad = gradOutput.data as Float32Array;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = (src[i] > lo && src[i] < hi) ? grad[i] : 0;
    return makeTensor(input.shape, input.dtype, out);
  }

  softCap(input: TensorData, cap: number): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const byteSize = size * 4;
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "softcap_forward_vec4" : "softcap_forward";
      const pipeline = getPipeline(vk, kernelName, 2);
      const bufA = ensureGpu(vk, input);
      const region = acquireOutputRegion(vk, byteSize);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const push = new Float32Array([effectiveSize, cap]);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      graph.record({
        kind: "unary",
        kernel: kernelName,
        pipeline,
        inputBufs: [bufA],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: input.shape,
      });
      return graphLazyTensor(vk, input.shape, region);
    }
    // CPU fallback
    return this.cpuUnary(input, (x) => {
      const scaled = Math.max(-80, Math.min(80, x / cap));
      return Math.tanh(scaled) * cap;
    });
  }

  softCapBackward(gradOutput: TensorData, input: TensorData, cap: number): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      const vk = this.init();
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "softcap_backward_vec4" : "softcap_backward";
      const pipeline = getPipeline(vk, kernelName, 3);
      const bufGrad = ensureGpu(vk, gradOutput);
      const bufInput = ensureGpu(vk, input);
      const region = acquireOutputRegion(vk, size * 4);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const push = new Float32Array([effectiveSize, cap]);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      graph.record({
        kind: "backward",
        kernel: "softcap_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: input.shape,
        allBufs: [bufGrad, bufInput, region.handle],
      });
      return graphLazyTensor(vk, input.shape, region);
    }
    // CPU fallback
    const src = input.data as Float32Array;
    const grad = gradOutput.data as Float32Array;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const scaled = Math.max(-80, Math.min(80, src[i] / cap));
      const t = Math.tanh(scaled);
      out[i] = grad[i] * (1 - t * t);
    }
    return makeTensor(input.shape, input.dtype, out);
  }

  residualDropoutAdd(residual: TensorData, projected: TensorData, mask: TensorData): TensorData {
    const size = shapeSize(residual.shape);
    if (size >= this._minGpuSize && this.shapesEqual(residual.shape, projected.shape) && this.shapesEqual(residual.shape, mask.shape)) {
      const vk = this.init();
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "residual_dropout_add_vec4" : "residual_dropout_add";
      const pipeline = getPipeline(vk, kernelName, 4);
      const bufR = ensureGpu(vk, residual);
      const bufP = ensureGpu(vk, projected);
      const bufM = ensureGpu(vk, mask);
      const region = acquireOutputRegion(vk, size * 4);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const push = new Float32Array([effectiveSize, 0]);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      graph.record({
        kind: "binary",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: residual.shape,
        allBufs: [bufR, bufP, bufM, region.handle],
      });
      return graphLazyTensor(vk, residual.shape, region);
    }
    // CPU fallback: residual + projected * mask
    const rArr = residual.data as Float32Array;
    const pArr = projected.data as Float32Array;
    const mArr = mask.data as Float32Array;
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) out[i] = rArr[i] + pArr[i] * mArr[i];
    return makeTensor(residual.shape, residual.dtype, out);
  }

  crossEntropyBackward(logits: TensorData, targets: TensorData, gradOutput: TensorData): TensorData {
    const vk = this.init();
    const [N, C] = logits.shape;
    const totalElements = N * C;
    const gradScalar = (gradOutput.data as Float32Array)[0];

    // Compute softmax on GPU (stays lazy — no flush needed)
    const probs = this.softmax(logits, -1);

    if (totalElements >= this._minGpuSize) {
      const bufProbs = ensureGpu(vk, probs);
      // Upload targets as raw i32 bytes (bitcast in shader to u32)
      const bufTargets = ensureGpuRawBits(vk, targets);

      const pipeline = getPipeline(vk, "cross_entropy_backward", 3, 3 * 4);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(3);
      const pushU = new Uint32Array(push.buffer);
      push[0] = totalElements;  // float value for bounds check (loadPushLen reads as f32)
      pushU[1] = C;             // u32 bits — kernel bitcasts f32→u32
      push[2] = gradScalar / N; // scale by upstream gradient

      graph.record({
        kind: "backward",
        kernel: "cross_entropy_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: logits.shape,
        allBufs: [bufProbs, bufTargets, region.handle],
      });

      // Release probs GPU buffer after dispatch completes (deferred through graph)
      // Without this, the buffer stays alive until GC collects the local `probs` var
      releaseGpuBufferFor(probs);

      return graphLazyTensor(vk, logits.shape, region);
    }

    // CPU fallback
    const probsArr = probs.data as Float32Array;
    const out = new Float32Array(totalElements);
    const scale = gradScalar / N;
    for (let i = 0; i < N; i++) {
      const off = i * C;
      const target = targets.data[i];
      for (let j = 0; j < C; j++) {
        out[off + j] = (probsArr[off + j] - (j === target ? 1 : 0)) * scale;
      }
    }
    return makeTensor(logits.shape, logits.dtype, out);
  }

  embeddingBackward(indices: TensorData, gradOutput: TensorData, vocabSize: number): TensorData {
    const vk = this.init();
    const nIdx = shapeSize(indices.shape);
    const dim = gradOutput.shape[gradOutput.shape.length - 1];
    const totalElements = nIdx * dim;
    const outputSize = vocabSize * dim;

    if (totalElements >= this._minGpuSize) {
      const bufIndices = ensureGpuRawBits(vk, indices);
      const bufGradOut = ensureGpu(vk, gradOutput);

      // Allocate output via output pool and zero it (avoids double-tracking
      // that would happen with zeros() + ensureGpu())
      const outByteSize = outputSize * 4;
      const region = acquireOutputRegion(vk, outByteSize);
      vk.uploadBuffer(region.handle, new Float32Array(outputSize));

      // Dispatch the scatter-add kernel
      const pipeline = getPipeline(vk, "embedding_backward", 3, 2 * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      push[0] = totalElements;  // float value for bounds check
      pushU[1] = dim;           // u32 bits — kernel bitcasts f32→u32

      graph.record({
        kind: "backward",
        kernel: "embedding_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 2 * 4,
        shape: [vocabSize, dim],
        allBufs: [bufIndices, bufGradOut, region.handle],
      });

      return graphLazyTensor(vk, [vocabSize, dim], region);
    }

    // CPU fallback
    const out = new Float32Array(outputSize);
    const gradArr = gradOutput.data as Float32Array;
    for (let i = 0; i < nIdx; i++) {
      const idx = indices.data[i] as number;
      const srcOff = i * dim;
      const dstOff = idx * dim;
      for (let d = 0; d < dim; d++) {
        out[dstOff + d] += gradArr[srcOff + d];
      }
    }
    return makeTensor([vocabSize, dim], gradOutput.dtype, out);
  }

  layerNormBackward(x: TensorData, weight: TensorData, gradOutput: TensorData, eps: number): { dx: TensorData; dw: TensorData; db: TensorData } {
    const vk = this.init();
    const dim = x.shape[x.shape.length - 1];
    const numRows = shapeSize(x.shape) / dim;
    const xSize = shapeSize(x.shape);

    // GPU path — all dispatches recorded to graph (no flush/sync)
    if (xSize >= this._minGpuSize) {
      const bufX = ensureGpu(vk, x);
      const bufW = ensureGpu(vk, weight);
      const bufG = ensureGpu(vk, gradOutput);

      const dxRegion = acquireOutputRegion(vk, xSize * 4);
      const dwPartialRegion = acquireOutputRegion(vk, xSize * 4);
      const dbPartialRegion = acquireOutputRegion(vk, xSize * 4);
      const dwRegion = acquireOutputRegion(vk, dim * 4);
      const dbRegion = acquireOutputRegion(vk, dim * 4);

      // Main backward kernel (6 bindings) — recorded to graph
      const pipeline1 = getPipeline(vk, "layernorm_backward", 6);
      const push1 = push2Memo(dim, eps);
      graph.record({
        kind: "backward",
        kernel: "layernorm_backward",
        pipeline: pipeline1,
        inputBufs: [],
        outputRegion: dxRegion,
        groups: [numRows, 1, 1],
        push: push1,
        pushSize: 2 * 4,
        shape: x.shape,
        allBufs: [bufX, bufW, bufG, dxRegion.handle, dwPartialRegion.handle, dbPartialRegion.handle],
      });

      // Column sum to reduce partials: [numRows, dim] → [dim]
      const pipeline2 = getPipeline(vk, "column_sum", 2);
      const push2 = push2Memo(dim, numRows);
      const groups = Math.ceil(dim / WG_SIZE);
      graph.record({
        kind: "backward",
        kernel: "column_sum_dw",
        pipeline: pipeline2,
        inputBufs: [],
        outputRegion: dwRegion,
        groups: [groups, 1, 1],
        push: push2,
        pushSize: 2 * 4,
        shape: weight.shape,
        allBufs: [dwPartialRegion.handle, dwRegion.handle],
      });
      graph.record({
        kind: "backward",
        kernel: "column_sum_db",
        pipeline: pipeline2,
        inputBufs: [],
        outputRegion: dbRegion,
        groups: [groups, 1, 1],
        push: push2Memo(dim, numRows),
        pushSize: 2 * 4,
        shape: weight.shape,
        allBufs: [dbPartialRegion.handle, dbRegion.handle],
      });

      // Defer-release intermediate partial buffers (freed after graph flush)
      graph.deferRelease(dwPartialRegion);
      graph.deferRelease(dbPartialRegion);

      return {
        dx: graphLazyTensor(vk, x.shape, dxRegion),
        dw: graphLazyTensor(vk, weight.shape, dwRegion),
        db: graphLazyTensor(vk, weight.shape, dbRegion),
      };
    }

    // CPU fallback
    const n = numRows;
    const xArr = x.data as Float32Array;
    const wArr = weight.data as Float32Array;
    const gArr = gradOutput.data as Float32Array;
    const dxOut = this.zeros(x.shape, x.dtype);
    const dwOut = this.zeros(weight.shape, weight.dtype);
    const dbOut = this.zeros(weight.shape, weight.dtype);
    const dxArr = dxOut.data as Float32Array;
    const dwArr = dwOut.data as Float32Array;
    const dbArr = dbOut.data as Float32Array;
    for (let i = 0; i < n; i++) {
      const off = i * dim;
      let mu = 0;
      for (let j = 0; j < dim; j++) mu += xArr[off + j];
      mu /= dim;
      let v = 0;
      for (let j = 0; j < dim; j++) { const d = xArr[off + j] - mu; v += d * d; }
      v /= dim;
      const is = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < dim; j++) {
        const xhat = (xArr[off + j] - mu) * is;
        dwArr[j] += gArr[off + j] * xhat;
        dbArr[j] += gArr[off + j];
      }
      let s1 = 0, s2 = 0;
      for (let j = 0; j < dim; j++) {
        const dy = gArr[off + j] * wArr[j];
        s1 += dy;
        s2 += dy * (xArr[off + j] - mu) * is;
      }
      for (let j = 0; j < dim; j++) {
        const xhat = (xArr[off + j] - mu) * is;
        const dy = gArr[off + j] * wArr[j];
        dxArr[off + j] = is * (dy - (s1 + xhat * s2) / dim);
      }
    }
    return { dx: dxOut, dw: dwOut, db: dbOut };
  }

  private gpuMatmul(a: TensorData, b: TensorData): TensorData {
    const vk = this.init();
    this._matmulDispatches++;
    const aNdim = a.shape.length, bNdim = b.shape.length;
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b);

    // Try cooperative matrix (tensor core) path for aligned dimensions
    if (coopInputDtypesOk && this._coopMatSupported &&
        M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopK === 0) {
      this._coopDispatches++;
      this._coopDirectDispatches++;
      return this.gpuMatmulCoop(vk, a, b, M, N, K, aBatch, batchSize, false);
    }
    const coopPaddedBatched = coopInputDtypesOk
      ? this.tryPaddedCoopMatmulBatched(vk, a, b, M, N, K, aBatch, batchSize, false)
      : null;
    if (coopPaddedBatched) {
      this._coopDispatches++;
      this._coopPaddedBatchedDispatches++;
      return coopPaddedBatched;
    }
    // For large 2D GEMMs, opportunistically pad to coop tile sizes so we can still
    // use tensor cores on non-aligned shapes (generic, non-device-specific path).
    const coopPadded = coopInputDtypesOk
      ? this.tryPaddedCoopMatmul2D(vk, a, b, M, N, K, false, batchSize)
      : null;
    if (coopPadded) {
      this._coopDispatches++;
      this._coopPadded2DDispatches++;
      return coopPadded;
    }

    // Use tile=32 for large matrices (better memory efficiency, half the inner loop)
    // Tile=16 for small matrices (better occupancy when parallelism is limited)
    // All discrete GPUs we target (A100, L4, etc.) support 1024 invocations per workgroup
    const useLargeTile = M * N >= LARGE_TILE_THRESHOLD;
    const TILE = useLargeTile ? 32 : 16;
    const suffix = useLargeTile ? "_T32" : "";

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul${suffix}`, 3, 16);
      const bufA = ensureGpu(vk, a);
      const bufB = ensureGpu(vk, b);
      const outBytes = M * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      const push = push4Memo(M, N, K, 0);
      const gX = Math.ceil(N / TILE);
      const gY = Math.ceil(M / TILE);

      graph.record({
        kind: "matmul",
        kernel: `matmul${suffix}`,
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
      // Batched matmul — dispatch all batches in one GPU submission
      const pipeline = getPipeline(vk, `matmul_batched${suffix}`, 3, 16);
      const bufA = ensureGpu(vk, a);
      const bufB = ensureGpu(vk, b);
      const outBytes = batchSize * M * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      const push = push4Memo(M, N, K, 0);
      const gX = Math.ceil(N / TILE);
      const gY = Math.ceil(M / TILE);

      graph.record({
        kind: "matmul",
        kernel: `matmul_batched${suffix}`,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: region,
        groups: [gX, gY, batchSize],
        push,
        pushSize: 16,
        shape: [...aBatch, M, N],
      });

      return graphLazyTensor(vk, [...aBatch, M, N], region);
    }
  }

  // ── Fused matmul with B transposed: C = A × B^T ─────────────────────────

  /**
   * GPU matmul where B is transposed: computes A[M,K] × B[N,K]^T = C[M,N].
   * B is stored as [N,K] but used as if transposed to [K,N].
   * Eliminates the need for a separate transpose dispatch before matmul.
   */
  matmulTransposed(a: TensorData, b: TensorData): TensorData {
    const aNdim = a.shape.length, bNdim = b.shape.length;
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1];
    const N = b.shape[bNdim - 2]; // B is [N, K], so N is dim -2
    // Use compute FLOPs threshold like regular matmul
    if (M * N * K >= MATMUL_GPU_FLOPS_THRESHOLD) return this.gpuMatmulTransposed(a, b, M, N, K);
    // CPU fallback: materialize transpose then multiply
    const bT = this.transpose(b, bNdim - 2, bNdim - 1);
    return this.cpuMatmul(a, bT);
  }

  /**
   * GPU matmul where A is transposed: computes A[M,K]^T × B[M,N] = C[K,N].
   * A is stored as [M,K] but read in transposed layout directly in the kernel.
   */
  matmulTransposedA(a: TensorData, b: TensorData): TensorData {
    const aNdim = a.shape.length, bNdim = b.shape.length;
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1];
    const bM = b.shape[bNdim - 2], N = b.shape[bNdim - 1];
    if (bM !== M) throw new Error("matmulTransposedA shape mismatch");
    // Use compute FLOPs threshold like regular matmul
    if (M * N * K >= MATMUL_GPU_FLOPS_THRESHOLD) return this.gpuMatmulTransposedA(a, b, M, N, K);
    // CPU fallback: materialize transpose then multiply
    const aT = this.transpose(a, aNdim - 2, aNdim - 1);
    return this.cpuMatmul(aT, b);
  }

  private gpuMatmulTransposed(a: TensorData, b: TensorData, M: number, N: number, K: number): TensorData {
    const vk = this.init();
    this._matmulDispatches++;
    const aNdim = a.shape.length;
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b);

    // Try cooperative matrix (tensor core) path for aligned dimensions
    if (coopInputDtypesOk && this._coopMatSupported &&
        M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopK === 0) {
      this._coopDispatches++;
      this._coopDirectDispatches++;
      return this.gpuMatmulCoop(vk, a, b, M, N, K, aBatch, batchSize, true);
    }
    const coopPaddedBatched = coopInputDtypesOk
      ? this.tryPaddedCoopMatmulBatched(vk, a, b, M, N, K, aBatch, batchSize, true)
      : null;
    if (coopPaddedBatched) {
      this._coopDispatches++;
      this._coopPaddedBatchedDispatches++;
      return coopPaddedBatched;
    }
    const coopPadded = coopInputDtypesOk
      ? this.tryPaddedCoopMatmul2D(vk, a, b, M, N, K, true, batchSize)
      : null;
    if (coopPadded) {
      this._coopDispatches++;
      this._coopPadded2DDispatches++;
      return coopPadded;
    }

    // Use tile=32 for large matrices (better memory efficiency, half the inner loop)
    // Tile=16 for small matrices (better occupancy when parallelism is limited)
    const useLargeTile = M * N >= LARGE_TILE_THRESHOLD;
    const TILE = useLargeTile ? 32 : 16;
    const suffix = useLargeTile ? "_T32" : "";

    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const gX = Math.ceil(N / TILE);
    const gY = Math.ceil(M / TILE);
    const push = push4Memo(M, N, K, 0);

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul_transposed${suffix}`, 3, 16);
      const outBytes = M * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      graph.record({
        kind: "matmul",
        kernel: `matmul_transposed${suffix}`,
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
      const pipeline = getPipeline(vk, `matmul_transposed_batched${suffix}`, 3, 16);
      const outBytes = batchSize * M * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      graph.record({
        kind: "matmul",
        kernel: `matmul_transposed_batched${suffix}`,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: region,
        groups: [gX, gY, batchSize],
        push,
        pushSize: 16,
        shape: [...aBatch, M, N],
      });

      return graphLazyTensor(vk, [...aBatch, M, N], region);
    }
  }

  private gpuMatmulTransposedA(a: TensorData, b: TensorData, M: number, N: number, K: number): TensorData {
    const vk = this.init();
    this._matmulDispatches++;
    const aNdim = a.shape.length;
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b);

    // matmul_transposed_a computes C[K,N] = A[M,K]^T × B[M,N].
    const outM = K;
    const loopK = M;

    // Direct cooperative path for aligned transposed-A GEMMs.
    if (coopInputDtypesOk && this._coopMatSupported &&
        outM % this._coopM === 0 && N % this._coopN === 0 && loopK % this._coopK === 0) {
      this._coopDispatches++;
      this._coopDirectDispatches++;
      return this.gpuMatmulCoopTransposedA(vk, a, b, outM, N, loopK, aBatch, batchSize);
    }

    // Generic tensor-core route for large transposed-A GEMMs:
    // A^T @ B == (transpose(A)) @ (transpose(B))^T.
    // This allows reuse of the cooperative matmul-transposed path.
    if (coopInputDtypesOk && this._coopMatSupported &&
        outM * N * loopK >= COOP_TRANSPOSED_A_MIN_FLOPS &&
        this.canUseCoopWithOptionalPadding(outM, N, loopK)) {
      this._coopTransposedARewriteDispatches++;
      const aT = this.transpose(a, aNdim - 2, aNdim - 1);               // [..., K, M]
      const bT = this.transpose(b, b.shape.length - 2, b.shape.length - 1); // [..., N, M]
      // gpuMatmulTransposed() maintains its own matmul dispatch accounting.
      this._matmulDispatches--;
      return this.gpuMatmulTransposed(aT, bT, outM, N, loopK);
    }

    // Use tile=32 for large matrices (better memory efficiency, half the inner loop)
    // Tile=16 for small matrices (better occupancy when parallelism is limited)
    const useLargeTile = outM * N >= LARGE_TILE_THRESHOLD;
    const TILE = useLargeTile ? 32 : 16;
    const suffix = useLargeTile ? "_T32" : "";

    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const gX = Math.ceil(N / TILE);
    const gY = Math.ceil(outM / TILE);
    const push = push4Memo(outM, N, loopK, 0);

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul_transposed_a${suffix}`, 3, 16);
      const outBytes = outM * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      graph.record({
        kind: "matmul",
        kernel: `matmul_transposed_a${suffix}`,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: region,
        groups: [gX, gY, 1],
        push,
        pushSize: 16,
        shape: [...aBatch, outM, N],
      });

      return graphLazyTensor(vk, [...aBatch, outM, N], region);
    } else {
      const pipeline = getPipeline(vk, `matmul_transposed_a_batched${suffix}`, 3, 16);
      const outBytes = batchSize * outM * N * 4;
      const region = acquireOutputRegion(vk, outBytes);

      graph.record({
        kind: "matmul",
        kernel: `matmul_transposed_a_batched${suffix}`,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: region,
        groups: [gX, gY, batchSize],
        push,
        pushSize: 16,
        shape: [...aBatch, outM, N],
      });

      return graphLazyTensor(vk, [...aBatch, outM, N], region);
    }
  }

  // ── Cooperative matrix matmul dispatch ───────────────────────────────────────

  private gpuMatmulCoop(
    vk: NativeAddon, a: TensorData, b: TensorData,
    M: number, N: number, K: number,
    aBatch: number[], batchSize: number, transposed: boolean,
  ): TensorData {
    const useS2x2 =
      ENABLE_COOP_F16IN_S2X2 &&
      (M * N * K >= COOP_F16IN_S2X2_MIN_FLOPS) &&
      (M % (this._coopM * 2) === 0) &&
      (N % (this._coopN * 2) === 0);
    const subgroupTilesX = useS2x2 ? 2 : 1;
    const subgroupTilesY = useS2x2 ? 2 : 1;
    const variant = transposed
      ? (batchSize > 1 ? "transposed_batched" : "transposed")
      : (batchSize > 1 ? "batched" : "basic");
    const kernelName =
      `matmul_coop_${variant}_${this._coopM}_${this._coopN}_${this._coopK}_f16in` +
      (ENABLE_COOP_F16_ACCUM ? "_f16acc" : "") +
      (useS2x2 ? "_s2x2" : "");
    if (DEBUG_COOP) {
      console.error(
        `[helios:coop] enter kernel=${kernelName} M=${M} N=${N} K=${K} batch=${batchSize} transposed=${transposed}`,
      );
    }
    const pipeline = getPipeline(vk, kernelName, 3, 16);
    if (DEBUG_COOP) console.error(`[helios:coop] pipeline ready kernel=${kernelName} handle=${pipeline}`);
    const bufA = this.getCoopInputBuffer(vk, a);
    const bufB = this.getCoopInputBuffer(vk, b);
    const outBytes = batchSize * M * N * 4;
    const region = acquireOutputRegion(vk, outBytes);

    const push = push4Memo(M, N, K, 0);
    const gX = Math.ceil(N / (this._coopN * subgroupTilesX));
    const gY = Math.ceil(M / (this._coopM * subgroupTilesY));

    graph.record({
      kind: "matmul",
      kernel: kernelName,
      pipeline,
      inputBufs: [bufA, bufB],
      outputRegion: region,
      groups: [gX, gY, batchSize],
      push,
      pushSize: 16,
      shape: [...aBatch, M, N],
    });
    if (DEBUG_COOP) console.error(`[helios:coop] recorded kernel=${kernelName} g=(${gX},${gY},${batchSize})`);

    return graphLazyTensor(vk, [...aBatch, M, N], region);
  }

  private gpuMatmulCoopTransposedA(
    vk: NativeAddon, a: TensorData, b: TensorData,
    outM: number, N: number, loopK: number,
    aBatch: number[], batchSize: number,
  ): TensorData {
    const useS2x2 =
      ENABLE_COOP_F16IN_S2X2 &&
      (outM * N * loopK >= COOP_F16IN_S2X2_MIN_FLOPS) &&
      (outM % (this._coopM * 2) === 0) &&
      (N % (this._coopN * 2) === 0);
    const subgroupTilesX = useS2x2 ? 2 : 1;
    const subgroupTilesY = useS2x2 ? 2 : 1;
    const variant = batchSize > 1 ? "transposed_a_batched" : "transposed_a";
    const kernelName =
      `matmul_coop_${variant}_${this._coopM}_${this._coopN}_${this._coopK}_f16in` +
      (ENABLE_COOP_F16_ACCUM ? "_f16acc" : "") +
      (useS2x2 ? "_s2x2" : "");
    if (DEBUG_COOP) {
      console.error(
        `[helios:coop] enter kernel=${kernelName} outM=${outM} N=${N} K=${loopK} batch=${batchSize} transposedA=true`,
      );
    }
    const pipeline = getPipeline(vk, kernelName, 3, 16);
    if (DEBUG_COOP) console.error(`[helios:coop] pipeline ready kernel=${kernelName} handle=${pipeline}`);
    const bufA = this.getCoopInputBuffer(vk, a);
    const bufB = this.getCoopInputBuffer(vk, b);
    const outBytes = batchSize * outM * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    const push = push4Memo(outM, N, loopK, 0);
    const gX = Math.ceil(N / (this._coopN * subgroupTilesX));
    const gY = Math.ceil(outM / (this._coopM * subgroupTilesY));

    graph.record({
      kind: "matmul",
      kernel: kernelName,
      pipeline,
      inputBufs: [bufA, bufB],
      outputRegion: region,
      groups: [gX, gY, batchSize],
      push,
      pushSize: 16,
      shape: [...aBatch, outM, N],
    });
    if (DEBUG_COOP) console.error(`[helios:coop] recorded kernel=${kernelName} g=(${gX},${gY},${batchSize})`);

    return graphLazyTensor(vk, [...aBatch, outM, N], region);
  }

  private tryPaddedCoopMatmul2D(
    vk: NativeAddon,
    a: TensorData,
    b: TensorData,
    M: number,
    N: number,
    K: number,
    transposed: boolean,
    batchSize: number,
  ): TensorData | null {
    if (!this._coopMatSupported) return null;
    if (batchSize !== 1) return null;
    if (a.shape.length !== 2 || b.shape.length !== 2) return null;
    if (a.dtype !== "f32" || b.dtype !== "f32") return null;
    if (M * N * K < COOP_PAD_MIN_FLOPS) return null;
    if (!this.canUseCoopWithOptionalPadding(M, N, K)) return null;

    const alignedM = alignUp(M, this._coopM);
    const alignedN = alignUp(N, this._coopN);
    const alignedK = alignUp(K, this._coopK);
    if (alignedM === M && alignedN === N && alignedK === K) return null;

    const baseElems = M * K + K * N + M * N;
    const paddedElems = alignedM * alignedK + alignedK * alignedN + alignedM * alignedN;
    const overhead = (paddedElems - baseElems) / baseElems;
    if (overhead > COOP_PAD_MAX_OVERHEAD) return null;

    const paddedA = this.scatterSlice(a, [alignedM, alignedK], [0, 0], [M, K]);
    const paddedB = transposed
      ? this.scatterSlice(b, [alignedN, alignedK], [0, 0], [N, K])
      : this.scatterSlice(b, [alignedK, alignedN], [0, 0], [K, N]);
    const paddedOut = this.gpuMatmulCoop(vk, paddedA, paddedB, alignedM, alignedN, alignedK, [], 1, transposed);
    return this.slice(paddedOut, [0, 0], [M, N]);
  }

  private tryPaddedCoopMatmulBatched(
    vk: NativeAddon,
    a: TensorData,
    b: TensorData,
    M: number,
    N: number,
    K: number,
    aBatch: number[],
    batchSize: number,
    transposed: boolean,
  ): TensorData | null {
    if (!this._coopMatSupported) return null;
    if (batchSize <= 1) return null;
    if (a.dtype !== "f32" || b.dtype !== "f32") return null;
    if (M * N * K < COOP_PAD_MIN_FLOPS) return null;
    if (!this.canUseCoopWithOptionalPadding(M, N, K)) return null;

    const alignedM = alignUp(M, this._coopM);
    const alignedN = alignUp(N, this._coopN);
    const alignedK = alignUp(K, this._coopK);
    if (alignedM === M && alignedN === N && alignedK === K) return null;

    const a3 = this.reshape(a, [batchSize, M, K]);
    const b3 = transposed
      ? this.reshape(b, [batchSize, N, K])
      : this.reshape(b, [batchSize, K, N]);

    const paddedA = this.scatterSlice(a3, [batchSize, alignedM, alignedK], [0, 0, 0], [batchSize, M, K]);
    const paddedB = transposed
      ? this.scatterSlice(b3, [batchSize, alignedN, alignedK], [0, 0, 0], [batchSize, N, K])
      : this.scatterSlice(b3, [batchSize, alignedK, alignedN], [0, 0, 0], [batchSize, K, N]);

    const paddedOut = this.gpuMatmulCoop(vk, paddedA, paddedB, alignedM, alignedN, alignedK, [batchSize], batchSize, transposed);
    const cropped = this.slice(paddedOut, [0, 0, 0], [batchSize, M, N]);
    return this.reshape(cropped, [...aBatch, M, N]);
  }

  private canUseCoopWithOptionalPadding(M: number, N: number, K: number): boolean {
    if (!this._coopMatSupported) return false;
    if (M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopK === 0) return true;
    const alignedM = alignUp(M, this._coopM);
    const alignedN = alignUp(N, this._coopN);
    const alignedK = alignUp(K, this._coopK);
    const baseElems = M * K + K * N + M * N;
    const paddedElems = alignedM * alignedK + alignedK * alignedN + alignedM * alignedN;
    const overhead = (paddedElems - baseElems) / baseElems;
    return overhead <= COOP_PAD_MAX_OVERHEAD;
  }

  // ── In-place add: A += B on GPU ────────────────────────────────────────────

  addInplace(a: TensorData, b: TensorData): void {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const bufA = ensureGpu(vk, a);
      const bufB = ensureGpu(vk, b);
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "add_inplace_vec4" : "add_inplace";
      const effectiveSize = useVec4 ? size >> 2 : size;
      const pipeline = getPipeline(vk, kernelName, 2);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      const push = push2Memo(effectiveSize, 0);

      graph.record({
        kind: "inplace",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: null as any, // in-place — no output region
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: a.shape as number[],
        allBufs: [bufA, bufB],
      });
      invalidateCache(a);
      this._coopF16InputCache.delete(a);
      return;
    }
    // CPU fallback
    const aArr = a.data as Float32Array;
    const bArr = b.data as Float32Array;
    for (let i = 0; i < size; i++) aArr[i] += bArr[i];
    this._coopF16InputCache.delete(a);
  }

  // ── In-place scale: A *= scalar on GPU ─────────────────────────────────────

  scaleInplace(a: TensorData, scalar: number): void {
    const size = shapeSize(a.shape);
    if (a.dtype === "f32" && size >= this._minGpuSize) {
      const vk = this.init();
      const bufA = ensureGpu(vk, a);
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "scale_inplace_vec4" : "scale_inplace";
      const effectiveSize = useVec4 ? size >> 2 : size;
      const pipeline = getPipeline(vk, kernelName, 1);
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      const push = push2Memo(effectiveSize, scalar);

      graph.record({
        kind: "inplace",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: null as any,
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: a.shape as number[],
        allBufs: [bufA],
      });

      invalidateCache(a);
      this._coopF16InputCache.delete(a);
      return;
    }
    if (a.dtype === "f32") {
      const arr = a.data as Float32Array;
      for (let i = 0; i < size; i++) arr[i] *= scalar;
      this._coopF16InputCache.delete(a);
      return;
    }
    // Generic fallback for non-f32 dtypes: compute out-of-place then copy back.
    const scaled = this.scale(a, scalar);
    const src = scaled.data as any;
    const dst = a.data as any;
    if (typeof dst.set === "function") {
      dst.set(src);
    } else {
      for (let i = 0; i < size; i++) dst[i] = src[i];
    }
    this._coopF16InputCache.delete(a);
  }

  // ── Flash Attention (fused forward + backward) ─────────────────────────

  flashAttention(Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap: number): { output: TensorData; lse: TensorData } {
    const vk = this.init();
    // Q/K/V are [BH, T, D] where BH = batch * nHeads
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 32, Bc = 32;

    const kernelName = `flash_attn_fwd_${Br}_${Bc}_${D}`;
    const pipeline = getPipeline(vk, kernelName, 5, 16);

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);

    // Output: O [BH, T, D]
    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    // LSE: [BH, T]
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = new Float32Array([T, scale, softCap, 0]);
    const gX = Math.ceil(T / Br);

    graph.record({
      kind: "matmul", // reuse matmul kind for flash attention dispatch
      kernel: kernelName,
      pipeline,
      inputBufs: [],
      outputRegion: oRegion, // not used directly — allBufs overrides
      groups: [gX, BH, 1],
      push,
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
    });

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { output, lse };
  }

  flashAttentionBackward(Q: TensorData, K: TensorData, V: TensorData,
    O: TensorData, dO: TensorData, lse: TensorData,
    T: number, scale: number, softCap: number): { dQ: TensorData; dK: TensorData; dV: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 32, Bc = 32;

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);
    const bufO = ensureGpu(vk, O);
    const bufDO = ensureGpu(vk, dO);
    const bufLSE = ensureGpu(vk, lse);

    // Step 1: Precompute D[i] = sum_d(dO[i,d] * O[i,d]) via element-wise mul + sum
    // D_precomp: [BH, T] — one scalar per query position
    const doTimesO = this.mul(dO, O); // [BH, T, D]
    const dPrecomp = this.sum(doTimesO, 2); // [BH, T] — sum over last axis (D)
    const bufDpre = ensureGpu(vk, dPrecomp);

    // Step 2: dQ kernel
    const dqKernel = `flash_attn_bwd_dq_${Br}_${Bc}_${D}`;
    const dqPipeline = getPipeline(vk, dqKernel, 7, 16);
    const dqBytes = BH * T * D * 4;
    const dqRegion = acquireOutputRegion(vk, dqBytes);
    const push = new Float32Array([T, scale, softCap, 0]);

    graph.record({
      kind: "backward",
      kernel: dqKernel,
      pipeline: dqPipeline,
      inputBufs: [],
      outputRegion: dqRegion,
      groups: [Math.ceil(T / Br), BH, 1],
      push,
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, bufDO, bufLSE, bufDpre, dqRegion.handle],
    });

    // Step 3: dKV kernel
    const dkvKernel = `flash_attn_bwd_dkv_${Br}_${Bc}_${D}`;
    const dkvPipeline = getPipeline(vk, dkvKernel, 8, 16);
    const dkBytes = BH * T * D * 4;
    const dkRegion = acquireOutputRegion(vk, dkBytes);
    const dvBytes = BH * T * D * 4;
    const dvRegion = acquireOutputRegion(vk, dvBytes);

    graph.record({
      kind: "backward",
      kernel: dkvKernel,
      pipeline: dkvPipeline,
      inputBufs: [],
      outputRegion: dkRegion,
      groups: [Math.ceil(T / Bc), BH, 1],
      push: new Float32Array([T, scale, softCap, 0]),
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, bufDO, bufLSE, bufDpre, dkRegion.handle, dvRegion.handle],
    });

    const dQ = graphLazyTensor(vk, [BH, T, D], dqRegion);
    const dK = graphLazyTensor(vk, [BH, T, D], dkRegion);
    const dV = graphLazyTensor(vk, [BH, T, D], dvRegion);
    return { dQ, dK, dV };
  }

  // ── Backend interface: nn ops ───────────────────────────────────────────

  embedding(weight: TensorData, indices: TensorData): TensorData {
    const dim = weight.shape[1];
    const nIdx = shapeSize(indices.shape);
    const totalElements = nIdx * dim;
    const outShape = [...indices.shape, dim];

    // GPU path: gather rows from weight matrix on GPU
    if (totalElements >= this._minGpuSize) {
      const vk = this.init();
      const bufWeight = ensureGpu(vk, weight);
      const bufIndices = ensureGpuRawBits(vk, indices);

      const pipeline = getPipeline(vk, "embedding_forward", 3, 2 * 4);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      push[0] = totalElements;  // float value for bounds check
      pushU[1] = dim;           // u32 bits — kernel bitcasts f32→u32

      graph.record({
        kind: "unary",
        kernel: "embedding_forward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 2 * 4,
        shape: outShape,
        allBufs: [bufWeight, bufIndices, region.handle],
      });

      return graphLazyTensor(vk, outShape, region);
    }

    // CPU fallback
    const Ctor = dtypeArray(weight.dtype);
    const out = new Ctor(totalElements);
    for (let i = 0; i < nIdx; i++) {
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

    if (N * C >= this._minGpuSize) {
      const vk = this.init();
      // GPU path: fused log-sum-exp CE kernel (one workgroup per row)
      // Replaces 5-op chain (softmax → clamp → log → pick → negate) with
      // a single kernel that computes loss[i] = log(sum(exp(x - max))) + max - x[target]
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const pipeline = getPipeline(vk, "ce_fwd_fused", 3, 2 * 4);
      const region = acquireOutputRegion(vk, N * 4);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = N;
      pushU[1] = C;

      graph.record({
        kind: "unary",
        kernel: "ce_fwd_fused",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [N, 1, 1],  // one workgroup per row
        push,
        pushSize: 2 * 4,
        shape: [N],
        allBufs: [bufLogits, bufTargets, region.handle],
      });

      // Sum per-row losses on GPU, then read single scalar
      const perRowLosses = graphLazyTensor(vk, [N], region);
      const totalLoss = this.gpuReduceSum(perRowLosses, false);
      const total = (totalLoss.data as Float32Array)[0];
      return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([total / N]));
    }

    // CPU fallback — numerically stable log-softmax
    const logProbs = this.cpuLogSoftmax(logits, 1);
    let loss = 0;
    for (let i = 0; i < N; i++) loss -= logProbs.data[i * C + targets.data[i]];
    loss /= N;
    return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([loss]));
  }

  // ── Backend interface: reshape / slice ──────────────────────────────────

  reshape(a: TensorData, shape: Shape): TensorData {
    if (shapeSize(shape) !== shapeSize(a.shape)) throw new Error(`Cannot reshape [${a.shape}] to [${shape}]`);
    // Zero-copy: share underlying data + GPU buffer (avoids forced readback)
    const td: TensorData = {
      shape: [...shape],
      dtype: a.dtype,
      get data() { return a.data; },
    };
    shareGpuResidence(a, td);
    return td;
  }

  broadcast(a: TensorData, targetShape: Shape): TensorData {
    const srcSize = shapeSize(a.shape);
    const dstSize = shapeSize(targetShape);
    if (srcSize === dstSize) return this.reshape(a, targetShape);

    // GPU broadcast: B[i] = A[i % srcSize]
    if (dstSize >= this._minGpuSize) {
      const vk = this.init();
      const inputBuf = ensureGpu(vk, a);
      const pipeline = getPipeline(vk, "broadcast", 2, 2 * 4);
      const region = acquireOutputRegion(vk, dstSize * 4);
      const groups = Math.ceil(dstSize / WG_SIZE);

      const pushF = new Float32Array(2);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = dstSize;
      pushU[1] = srcSize;

      graph.record({
        kind: "unary",
        kernel: "broadcast",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 2 * 4,
        shape: targetShape,
        allBufs: [inputBuf, region.handle],
      });

      return graphLazyTensor(vk, targetShape, region);
    }

    // CPU fallback — stride-based for correct non-trailing broadcasts
    const out = new Float32Array(dstSize);
    const src = a.data as Float32Array;
    if (srcSize === 1) { out.fill(src[0]); }
    else {
      const strides = broadcastStrides(a.shape, targetShape);
      const ndim = targetShape.length;
      for (let i = 0; i < dstSize; i++) {
        let srcIdx = 0;
        let remainder = i;
        for (let d = ndim - 1; d >= 0; d--) {
          const coord = remainder % targetShape[d];
          remainder = (remainder - coord) / targetShape[d];
          srcIdx += coord * strides[d];
        }
        out[i] = src[srcIdx];
      }
    }
    return { shape: targetShape, dtype: a.dtype, data: out };
  }

  transpose(a: TensorData, dim0: number, dim1: number): TensorData {
    const ndim = a.shape.length;
    const d0 = dim0 < 0 ? dim0 + ndim : dim0;
    const d1 = dim1 < 0 ? dim1 + ndim : dim1;
    const newShape = [...a.shape]; newShape[d0] = a.shape[d1]; newShape[d1] = a.shape[d0];
    const size = shapeSize(a.shape);

    // GPU path — use stride-based 4D transpose kernel
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const inputBuf = ensureGpu(vk, a);

      // Pad shape to 4D (prepend 1s) and adjust dim indices
      const pad = 4 - ndim;
      const shape4 = ndim < 4
        ? [...Array(pad).fill(1) as number[], ...a.shape]
        : a.shape.slice(0, 4);
      const d0_4 = d0 + pad;
      const d1_4 = d1 + pad;

      // Compute input strides for padded 4D shape
      const inStrides = shapeStrides(shape4);

      // Compute output strides: swap dims in shape, recompute strides,
      // then swap stride positions so input coords map to correct output positions
      const outShape4 = [...shape4];
      outShape4[d0_4] = shape4[d1_4];
      outShape4[d1_4] = shape4[d0_4];
      const outStrides = shapeStrides(outShape4);
      const tmpS = outStrides[d0_4]; outStrides[d0_4] = outStrides[d1_4]; outStrides[d1_4] = tmpS;

      // Pack push constants as u32 via shared ArrayBuffer
      const pushF = new Float32Array(9);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = size;
      pushU[1] = inStrides[0]; pushU[2] = inStrides[1];
      pushU[3] = inStrides[2]; pushU[4] = inStrides[3];
      pushU[5] = outStrides[0]; pushU[6] = outStrides[1];
      pushU[7] = outStrides[2]; pushU[8] = outStrides[3];

      const pipeline = getPipeline(vk, "transpose", 2, 9 * 4);
      const outRegion = acquireOutputRegion(vk, size * 4);
      const groups = Math.ceil(size / WG_SIZE);

      graph.record({
        kind: "unary",
        kernel: "transpose",
        pipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 9 * 4,
        shape: newShape,
        allBufs: [inputBuf, outRegion.handle],
      });

      return graphLazyTensor(vk, newShape, outRegion);
    }

    // CPU fallback for small tensors
    const srcStrides = shapeStrides(a.shape);
    const dstStrides = shapeStrides(newShape);
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

    // GPU path for 2D tensors above threshold
    if (ndim === 2 && outSize >= this._minGpuSize) {
      const vk = this.init();
      const inputBuf = ensureGpu(vk, a);

      const pushF = new Float32Array(5);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = outSize;
      pushU[1] = outShape[1];  // outCols
      pushU[2] = a.shape[1];   // srcCols
      pushU[3] = starts[0];    // startRow
      pushU[4] = starts[1];    // startCol

      const pipeline = getPipeline(vk, "slice_2d", 2, 5 * 4);
      const outRegion = acquireOutputRegion(vk, outSize * 4);
      const groups = Math.ceil(outSize / WG_SIZE);

      graph.record({
        kind: "unary",
        kernel: "slice_2d",
        pipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 5 * 4,
        shape: outShape,
        allBufs: [inputBuf, outRegion.handle],
      });

      return graphLazyTensor(vk, outShape, outRegion);
    }

    // GPU path for 3D tensors above threshold
    if (ndim === 3 && outSize >= this._minGpuSize) {
      const vk = this.init();
      const inputBuf = ensureGpu(vk, a);

      const pushF = new Float32Array(8);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = outSize;
      pushU[1] = outShape[1];
      pushU[2] = outShape[2];
      pushU[3] = a.shape[1];
      pushU[4] = a.shape[2];
      pushU[5] = starts[0];
      pushU[6] = starts[1];
      pushU[7] = starts[2];

      const pipeline = getPipeline(vk, "slice_3d", 2, 8 * 4);
      const outRegion = acquireOutputRegion(vk, outSize * 4);
      const groups = Math.ceil(outSize / WG_SIZE);

      graph.record({
        kind: "unary",
        kernel: "slice_3d",
        pipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 8 * 4,
        shape: outShape,
        allBufs: [inputBuf, outRegion.handle],
      });

      return graphLazyTensor(vk, outShape, outRegion);
    }

    // CPU fallback
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

  scatterSlice(grad: TensorData, origShape: Shape, starts: number[], ends: number[]): TensorData {
    const ndim = origShape.length;
    const outSize = shapeSize(origShape);

    // GPU path for 2D tensors above threshold
    if (ndim === 2 && outSize >= this._minGpuSize) {
      const vk = this.init();
      const gradBuf = ensureGpu(vk, grad);

      const sliceRows = ends[0] - starts[0];
      const sliceCols = ends[1] - starts[1];

      const pushF = new Float32Array(6);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = outSize;
      pushU[1] = origShape[1];  // totalCols
      pushU[2] = sliceCols;
      pushU[3] = starts[0];     // startRow
      pushU[4] = starts[1];     // startCol
      pushU[5] = sliceRows;

      const pipeline = getPipeline(vk, "scatter_slice_2d", 2, 6 * 4);
      const outRegion = acquireOutputRegion(vk, outSize * 4);
      const groups = Math.ceil(outSize / WG_SIZE);

      graph.record({
        kind: "unary",
        kernel: "scatter_slice_2d",
        pipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 6 * 4,
        shape: [...origShape],
        allBufs: [gradBuf, outRegion.handle],
      });

      return graphLazyTensor(vk, [...origShape], outRegion);
    }

    // GPU path for 3D tensors above threshold
    if (ndim === 3 && outSize >= this._minGpuSize) {
      const vk = this.init();
      const gradBuf = ensureGpu(vk, grad);

      const sliceD0 = ends[0] - starts[0];
      const sliceD1 = ends[1] - starts[1];
      const sliceD2 = ends[2] - starts[2];

      const pushF = new Float32Array(9);
      const pushU = new Uint32Array(pushF.buffer);
      pushU[0] = outSize;
      pushU[1] = origShape[1];
      pushU[2] = origShape[2];
      pushU[3] = sliceD0;
      pushU[4] = sliceD1;
      pushU[5] = sliceD2;
      pushU[6] = starts[0];
      pushU[7] = starts[1];
      pushU[8] = starts[2];

      const pipeline = getPipeline(vk, "scatter_slice_3d", 2, 9 * 4);
      const outRegion = acquireOutputRegion(vk, outSize * 4);
      const groups = Math.ceil(outSize / WG_SIZE);

      graph.record({
        kind: "unary",
        kernel: "scatter_slice_3d",
        pipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [groups, 1, 1],
        push: pushF,
        pushSize: 9 * 4,
        shape: [...origShape],
        allBufs: [gradBuf, outRegion.handle],
      });

      return graphLazyTensor(vk, [...origShape], outRegion);
    }

    // CPU fallback
    const Ctor = dtypeArray(grad.dtype);
    const out = new Ctor(outSize);
    const origStrides = shapeStrides(origShape);
    const gradStrides = shapeStrides(grad.shape);
    const gradSize = shapeSize(grad.shape);
    for (let i = 0; i < gradSize; i++) {
      const coords = flatToMulti(i, grad.shape);
      let outFlat = 0;
      for (let d = 0; d < ndim; d++) outFlat += (coords[d] + starts[d]) * origStrides[d];
      out[outFlat] = grad.data[i];
    }
    return makeTensor([...origShape], grad.dtype, out);
  }

  dropoutMask(shape: Shape, seed: number, counter: number, p: number): TensorData {
    const size = shapeSize(shape);
    const scaleVal = 1 / (1 - p);
    const vk = this.init();

    const pushF = new Float32Array(5);
    const pushU = new Uint32Array(pushF.buffer);
    pushU[0] = size;
    pushU[1] = (seed | 0) >>> 0;    // seed as u32
    pushU[2] = (counter | 0) >>> 0;  // counter as u32
    // p and scale as f32 bit patterns stored in u32 slots
    pushF[3] = p;
    pushF[4] = scaleVal;

    const pipeline = getPipeline(vk, "dropout_mask", 1, 5 * 4);
    const outRegion = acquireOutputRegion(vk, size * 4);
    const groups = Math.ceil(size / WG_SIZE);

    graph.record({
      kind: "unary",
      kernel: "dropout_mask",
      pipeline,
      inputBufs: [],
      outputRegion: outRegion,
      groups: [groups, 1, 1],
      push: pushF,
      pushSize: 5 * 4,
      shape: [...shape],
      allBufs: [outRegion.handle],
    });

    return graphLazyTensor(vk, [...shape], outRegion);
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
    // GPU clone: copy buffer on GPU without reading back to CPU.
    // This is critical for backward pass performance — clone() is called for
    // every gradient, and GPU→CPU→GPU ping-pong was causing 30s backward times.
    if (gpuResidence.has(a) && shapeSize(a.shape) >= this._minGpuSize) {
      return this.gpuUnaryOp(a, "scale", 1.0);
    }
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
    const totalElements = shapeSize(a.shape);
    const maskSize = shapeSize(mask.shape);

    if (totalElements >= this._minGpuSize) {
      const vk = this.init();
      const bufA = ensureGpu(vk, a);
      const bufMask = ensureGpu(vk, mask);
      const pipeline = getPipeline(vk, "masked_fill", 3, 3 * 4);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(3);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = totalElements;
      pushU[1] = maskSize;
      push[2] = value;

      graph.record({
        kind: "unary",
        kernel: "masked_fill",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: a.shape,
        allBufs: [bufA, bufMask, region.handle],
      });

      return graphLazyTensor(vk, a.shape, region);
    }

    // CPU fallback
    const Ctor = dtypeArray(a.dtype);
    const out = Ctor.from(a.data);
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
    if (this.shapesEqual(a.shape, b.shape)) {
      const size = shapeSize(a.shape);
      const Ctor = dtypeArray(a.dtype);
      const out = new Ctor(size);
      for (let i = 0; i < size; i++) out[i] = fn(a.data[i], b.data[i]);
      return makeTensor(a.shape, a.dtype, out);
    }
    // Stride-based broadcast for correct non-trailing dimension handling
    const [resultShape, stridesA, stridesB] = broadcastShapes(a.shape, b.shape);
    const size = shapeSize(resultShape);
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(size);
    for (let i = 0; i < size; i++) {
      const [ia, ib] = broadcastIndices(i, resultShape, stridesA, stridesB);
      out[i] = fn(a.data[ia], b.data[ib]);
    }
    return makeTensor(resultShape, a.dtype, out);
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
      for (let d = 0; d < ndim; d++) { if (d === ax) { inCoords.push(0); if (keepdims) oi++; } else inCoords.push(outCoords[oi++]); }
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

  // ── GPU AdamW optimizer step ─────────────────────────────────────────────

  adamwStep(
    params: TensorData, grads: TensorData, m: TensorData, v: TensorData,
    lr: number, beta1: number, beta2: number, eps: number,
    weightDecay: number, bc1: number, bc2: number, gradScale = 1.0,
  ): void {
    const vk = this.init();
    const size = shapeSize(params.shape);

    if (size < this._minGpuSize) {
      // CPU fallback for small tensors
      const pData = params.data as Float32Array;
      const gData = grads.data as Float32Array;
      const mData = m.data as Float32Array;
      const vData = v.data as Float32Array;
      for (let i = 0; i < size; i++) {
        const g = gData[i] * gradScale;
        pData[i] -= lr * weightDecay * pData[i];
        mData[i] = beta1 * mData[i] + (1 - beta1) * g;
        vData[i] = beta2 * vData[i] + (1 - beta2) * g * g;
        const mHat = mData[i] / bc1;
        const vHat = vData[i] / bc2;
        pData[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
      this._coopF16InputCache.delete(params);
      this._coopF16InputCache.delete(m);
      this._coopF16InputCache.delete(v);
      return;
    }

    // GPU path: record AdamW dispatch to graph (batched with other ops)
    const bufP = ensureGpu(vk, params);
    const bufG = ensureGpu(vk, grads);
    const bufM = ensureGpu(vk, m);
    const bufV = ensureGpu(vk, v);

    const pipeline = getPipeline(vk, "adamw_step", 4, 9 * 4);
    const push = new Float32Array([size, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale]);
    const groups = Math.ceil(size / WG_SIZE);

    // Use a dummy output region since this is an in-place op
    graph.record({
      kind: "optimizer",
      kernel: "adamw_step",
      pipeline,
      inputBufs: [],
      outputRegion: { handle: bufP, byteSize: 0, readyValue: 0 },
      groups: [groups, 1, 1],
      push,
      pushSize: 9 * 4,
      shape: params.shape,
      allBufs: [bufP, bufG, bufM, bufV],
    });

    // Invalidate CPU caches immediately — next .data access will flush graph + readback
    invalidateCache(params);
    invalidateCache(m);
    invalidateCache(v);
    this._coopF16InputCache.delete(params);
    this._coopF16InputCache.delete(m);
    this._coopF16InputCache.delete(v);
  }
}
