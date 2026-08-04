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
  type UnlikelihoodLossStats,
  shapeSize,
  shapeStrides,
  dtypeArray,
  SeededRng,
  broadcastShapes,
  broadcastIndices,
  broadcastStrides,
} from "@alpha/core";

import { getNative, initDevice, getDeviceInfo, type NativeAddon, type NativeDeviceInfo } from "./device.js";
import { getKernelSpirv } from "./kernels.js";
import { loadStaticSlotPlan, type StaticSlotPlan } from "./static-slot-plan.js";

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
// The fixed-order embedding-gradient gather is bitwise deterministic but does
// O(outputElements * tokenPositions) work. Keep it for bounded replay and
// correctness workloads; production training retains the linear-work atomic
// scatter. This is a work budget, not a model- or vendor-specific special case.
const DETERMINISTIC_EMBEDDING_BACKWARD_MAX_WORK = 50_000_000;

const WG_CANDIDATES = [64, 128, 256, 512, 1024] as const;
let WG_SIZE = 128;  // stable default on L4; can be overridden by env/auto-tuning
let wgAutoTuned = false;
const ENABLE_WG_AUTOTUNE = process.env.HELIOS_WG_AUTOTUNE === "1";
const DISABLE_BATCH_DISPATCH_MANY = process.env.HELIOS_DISABLE_BATCH_DISPATCH_MANY === "1";
// Detailed operation accounting is deliberately opt-in: it adds Map updates to
// every recorded GPU operation and is intended for bounded performance sweeps,
// not production training runs.
const PROFILE_GPU_OPS = process.env.HELIOS_PROFILE_GPU_OPS === "1";
// Timestamp the dispatches in their real batched command buffer. This path is
// synchronous and deliberately diagnostic-only: it changes scheduling enough
// that its aggregate throughput must not be reported as the production rate.
const PROFILE_GPU_TIMESTAMPS = process.env.HELIOS_PROFILE_GPU_TIMESTAMPS === "1";
// A structural signature records the exact ordered operation topology and flush
// boundaries while deliberately excluding buffer handles and tensor values.
// It is diagnostic-only: stable signatures across steps are the prerequisite
// for ahead-of-time graph compilation and command replay.
const PROFILE_GRAPH_SIGNATURE = process.env.HELIOS_PROFILE_GRAPH_SIGNATURE === "1";
const PROFILE_GRAPH_TRACE = process.env.HELIOS_PROFILE_GRAPH_TRACE === "1";
// Matched-control switch for the residual-add + RMSNorm fusion. The public
// backend hook remains available so the autograd topology is identical; only
// the physical implementation changes from one dispatch back to add+rmsnorm.
const DISABLE_RESIDUAL_ADD_RMSNORM =
  process.env.HELIOS_DISABLE_RESIDUAL_ADD_RMSNORM === "1";
// Matched-control switch for grouped-QKV unpack + head-major layout + RoPE.
// The public hooks and autograd topology remain unchanged; the disabled path
// composes the established slice/transpose/rope kernels on the same backend.
const DISABLE_QKV_HEAD_MAJOR_ROPE =
  process.env.HELIOS_DISABLE_QKV_HEAD_MAJOR_ROPE === "1";
// An offline, warmup-excluded lifetime trace can be compiled into a fixed set
// of reusable VkBuffer handles. Physical r6 testing found that a complete plan
// can preserve loss while corrupting backward gradients before structural
// validation can observe the error. Keep the research implementation
// reproducible, but require an explicit unsafe acknowledgement until the
// missing alias/dependency semantics are understood.
const STATIC_SLOT_PLAN_PATH = process.env.HELIOS_STATIC_SLOT_PLAN?.trim() ?? "";
if (STATIC_SLOT_PLAN_PATH && process.env.HELIOS_STATIC_SLOT_PLAN_UNSAFE !== "1") {
  throw new Error(
    "HELIOS_STATIC_SLOT_PLAN is experimental and failed full-step numerical parity; " +
      "set HELIOS_STATIC_SLOT_PLAN_UNSAFE=1 only for bounded research reproduction",
  );
}
const STATIC_SLOT_PLAN: StaticSlotPlan | null = STATIC_SLOT_PLAN_PATH
  ? loadStaticSlotPlan(STATIC_SLOT_PLAN_PATH)
  : null;
const STATIC_SLOT_PLAN_WARMUP_STEPS = Math.max(
  0,
  Number.parseInt(process.env.HELIOS_STATIC_SLOT_PLAN_WARMUP_STEPS ?? "1", 10) || 0,
);
// Generic FP32 GEMMs have multiple portable Vulkan implementations.  The old
// size threshold is retained as the zero-overhead default, while this opt-in
// path measures the real device/driver/shape combination once and caches the
// winner.  It deliberately does not run during full-graph timestamp profiling:
// standalone probes would contaminate that diagnostic's accounting.
const ENABLE_MATMUL_TILE_AUTOTUNE = process.env.HELIOS_MATMUL_TILE_AUTOTUNE === "1";
const LOG_MATMUL_TILE_AUTOTUNE = process.env.HELIOS_MATMUL_TILE_AUTOTUNE_LOG === "1";
const ENABLE_MATMUL_REG2X2 = process.env.HELIOS_MATMUL_REG2X2 === "1";
const ENABLE_MATMUL_REG4X2 = process.env.HELIOS_MATMUL_REG4X2 === "1";
// R4x2 is not universally better across physical layouts. Keep transposed-B
// separately selectable, and keep both transposed input remaps independently
// gated, so the measured portfolio does not assume one loader fits every
// physical tensor layout.
const ENABLE_MATMUL_REG4X2_TRANSPOSED_B =
  process.env.HELIOS_MATMUL_REG4X2_TRANSPOSED_B === "1";
type ColumnSumRowLanes = 0 | 4 | 8 | 16;
function parseColumnSumRowLanes(): ColumnSumRowLanes {
  const raw = process.env.HELIOS_COLUMN_SUM_ROW_LANES?.trim() ?? "0";
  if (raw === "" || raw === "0") return 0;
  if (raw === "1" || raw === "8") return 8;
  if (raw === "4" || raw === "16") return Number(raw) as ColumnSumRowLanes;
  console.warn(
    `[helios] ignoring invalid HELIOS_COLUMN_SUM_ROW_LANES=${JSON.stringify(raw)}; ` +
      "expected 0, 4, 8, or 16",
  );
  return 0;
}
const COLUMN_SUM_ROW_LANES = parseColumnSumRowLanes();
const ENABLE_MATMUL_TRANSPOSED_B_COALESCED =
  process.env.HELIOS_MATMUL_TRANSPOSED_B_COALESCED === "1";
const ENABLE_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32 =
  process.env.HELIOS_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32 === "1";
const ENABLE_MATMUL_TRANSPOSED_A_COALESCED =
  process.env.HELIOS_MATMUL_TRANSPOSED_A_COALESCED === "1";
const MATMUL_TILE_OVERRIDE_ENV = process.env.HELIOS_MATMUL_TILE?.trim() ?? "";
let MATMUL_TILE_OVERRIDE: 16 | 32 | null = null;
if (MATMUL_TILE_OVERRIDE_ENV) {
  const parsed = Number(MATMUL_TILE_OVERRIDE_ENV);
  if (parsed === 16 || parsed === 32) {
    MATMUL_TILE_OVERRIDE = parsed;
  } else {
    console.warn(
      `[helios] ignoring HELIOS_MATMUL_TILE=${MATMUL_TILE_OVERRIDE_ENV}; expected 16 or 32`,
    );
  }
}
const DEBUG_COOP = process.env.HELIOS_DEBUG_COOP === "1";
const ENABLE_COOP_F16_ACCUM = process.env.HELIOS_COOP_F16_ACCUM === "1";
// The cooperative shader can either consume pre-cast f16 SSBOs or load f32
// operands and narrow each reusable tile into f16 workgroup memory.  The
// pre-cast path wins isolated GEMM microbenchmarks, but a training graph pays
// for whole-tensor cast dispatches and their lifetimes.  Keep the historical
// default while exposing the fused graph-level alternative explicitly.
const COOP_PRECAST_F16_INPUT = process.env.HELIOS_COOP_PRECAST_F16_INPUT !== "0";
// Diagnostic shape gates for cooperative-matrix training parity work.  Each
// entry is either a logical MxNxK triple (for example "10240x1920x640") or a
// layout-qualified key (for example "tb:10240x1920x640").  Unqualified keys
// retain the historical all-layout behavior.  Qualified keys are required for
// causal bisection because forward, ordinary-backward, and transposed-A
// backward can share dimensions while exercising different storage layouts.
// ALLOW is an optional positive list; DENY is applied after it.  These gates
// remain generic and opt-in: no model-specific dimensions are baked into
// Helios.
function parseCoopShapeSet(name: string): ReadonlySet<string> {
  const raw = process.env[name]?.trim() ?? "";
  if (!raw) return new Set<string>();
  const parsed = new Set<string>();
  for (const entry of raw.split(",")) {
    const key = entry.trim().toLowerCase();
    if (!/^(?:(?:nn|tb|ta):)?\d+x\d+x\d+$/.test(key)) {
      console.warn(
        `[helios] ignoring invalid ${name} entry ${JSON.stringify(entry)}; ` +
          "expected MxNxK or (nn|tb|ta):MxNxK",
      );
      continue;
    }
    parsed.add(key);
  }
  return parsed;
}
const COOP_SHAPE_ALLOW = parseCoopShapeSet("HELIOS_COOP_SHAPE_ALLOW");
const COOP_SHAPE_DENY = parseCoopShapeSet("HELIOS_COOP_SHAPE_DENY");
function coopShapeKey(M: number, N: number, K: number): string {
  return `${M}x${N}x${K}`;
}
type CoopShapeLayout = "nn" | "tb" | "ta";
function coopShapeIsEnabled(layout: CoopShapeLayout, M: number, N: number, K: number): boolean {
  const key = coopShapeKey(M, N, K);
  const qualified = `${layout}:${key}`;
  const allowed = COOP_SHAPE_ALLOW.size === 0 || COOP_SHAPE_ALLOW.has(key) || COOP_SHAPE_ALLOW.has(qualified);
  const denied = COOP_SHAPE_DENY.has(key) || COOP_SHAPE_DENY.has(qualified);
  return allowed && !denied;
}
const ENABLE_COOP_F16IN_S2X2 = process.env.HELIOS_COOP_F16IN_S2X2 !== "0";
const COOP_F16IN_S2X2_MIN_FLOPS = 20_000_000;
const COOP_SUBGROUP_TILES_ENV = process.env.HELIOS_COOP_F16IN_SUBGROUP_TILES?.trim() ?? "";
let COOP_SUBGROUP_TILES_X = 1;
let COOP_SUBGROUP_TILES_Y = 1;
if (COOP_SUBGROUP_TILES_ENV) {
  const m = COOP_SUBGROUP_TILES_ENV.match(/^(\d+)x(\d+)$/);
  if (m) {
    COOP_SUBGROUP_TILES_X = Math.max(1, parseInt(m[1], 10));
    COOP_SUBGROUP_TILES_Y = Math.max(1, parseInt(m[2], 10));
  } else {
    console.warn(
      `[helios] ignoring HELIOS_COOP_F16IN_SUBGROUP_TILES=${COOP_SUBGROUP_TILES_ENV}; expected <X>x<Y>`,
    );
  }
}
// Register tiling: each subgroup computes regTilesM × regTilesN cooperative matrix tiles
const COOP_REG_TILES_ENV = process.env.HELIOS_COOP_REG_TILES?.trim() ?? "";
let COOP_REG_TILES_M = 1;
let COOP_REG_TILES_N = 1;
if (COOP_REG_TILES_ENV) {
  const m = COOP_REG_TILES_ENV.match(/^(\d+)x(\d+)$/);
  if (m) {
    COOP_REG_TILES_M = Math.max(1, parseInt(m[1], 10));
    COOP_REG_TILES_N = Math.max(1, parseInt(m[2], 10));
  } else {
    console.warn(`[helios] ignoring HELIOS_COOP_REG_TILES=${COOP_REG_TILES_ENV}; expected <M>x<N>`);
  }
}
// Double buffering: overlap global memory loads with cooperative matrix MMA
const ENABLE_COOP_DOUBLE_BUF = process.env.HELIOS_COOP_DOUBLE_BUF === "1";
// Super-tile swizzle size for L2 cache reuse (0=disabled, must match kernel codegen)
const COOP_SWIZZLE_SIZE = parseInt(process.env.HELIOS_COOP_SWIZZLE ?? "4", 10);
// Split-K: partition K-reduction across multiple WGs for better SM occupancy.
// 0 or 1 = disabled; 2-8 = fixed split count; -1 = auto (heuristic based on WG count).
const COOP_SPLIT_K = parseInt(process.env.HELIOS_COOP_SPLIT_K ?? "0", 10);
// Adaptive kMulti switch threshold (base WG count). Below this threshold, kMulti
// drops to 2 when base kMulti>=4 to improve occupancy on smaller shapes.
// Disabled by default (0): kMulti=4 is universally better even at low WG counts.
const COOP_KMULTI_ADAPT_MIN_WGS = Math.max(
  0,
  parseInt(process.env.HELIOS_COOP_KMULTI_ADAPT_MIN_WGS ?? "0", 10),
);

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

type FlashCoop2ScopeTag = "wg" | "sg";

type FlashDispatchPath = "scalar" | "coop2";

export interface FlashDispatchDebug {
  requestedOp: "flashAttention" | "flashAttentionCoop2" | "flashAttentionCoop2Probe";
  executedPath: FlashDispatchPath;
  mode: "full" | "qk" | "qk_mask" | "qk_softmax" | "pv" | "kv_only" | "kv_synth" | "per_elem_only";
  softCap: number;
  BH: number;
  T: number;
  D: number;
  kernelName: string;
  pipelineKey: string;
  pipelineHandle: number;
  pipelineCreated: boolean;
  scope?: FlashCoop2ScopeTag;
  fallbackReason?: string;
}

interface PipelineLookupResult {
  key: string;
  handle: number;
  created: boolean;
}

function parseFlashCoop2ScopeTag(): FlashCoop2ScopeTag {
  const raw = (process.env.HELIOS_FLASH_COOP2_SCOPE ?? "workgroup").trim().toLowerCase();
  if (raw === "subgroup" || raw === "sg") return "sg";
  return "wg";
}

function parseFlashFwdPreferCoop2(): boolean {
  const raw = (process.env.HELIOS_FLASH_FWD_PREFER_COOP2 ?? "1").trim().toLowerCase();
  return !(raw === "0" || raw === "false" || raw === "off" || raw === "no");
}

function parseFlashFwdCoop2Strict(): boolean {
  const raw = (process.env.HELIOS_FLASH_FWD_COOP2_STRICT ?? "0").trim().toLowerCase();
  return !(raw === "0" || raw === "false" || raw === "off" || raw === "no");
}

function parseFlashDispatchDebugEnabled(): boolean {
  const raw = (process.env.HELIOS_FLASH_DISPATCH_DEBUG ?? "0").trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "on" || raw === "yes";
}

function parseFlashCoop2PreferF16Input(): boolean {
  const raw = (process.env.HELIOS_FLASH_COOP2_F16_INPUT ?? "1").trim().toLowerCase();
  return !(raw === "0" || raw === "false" || raw === "off" || raw === "no");
}

function parseFlashCoop2LocalSize(): number {
  const raw = (process.env.HELIOS_FLASH_COOP2_LS ?? "128").trim();
  const parsed = parseInt(raw, 10);
  if (Number.isFinite(parsed) && (parsed === 32 || parsed === 64 || parsed === 128)) {
    return parsed;
  }
  console.warn(`[helios] ignoring HELIOS_FLASH_COOP2_LS=${raw}; expected one of 32,64,128`);
  return 64;
}

function parseFlashCoop2QTiles(): number {
  const raw = (process.env.HELIOS_FLASH_COOP2_QT ?? "2").trim();
  const parsed = parseInt(raw, 10);
  if (Number.isFinite(parsed) && (parsed === 1 || parsed === 2 || parsed === 4)) return parsed;
  console.warn(`[helios] ignoring HELIOS_FLASH_COOP2_QT=${raw}; expected one of 1,2,4`);
  return 1;
}

function parseFlashCoop2BlockCols(): number {
  const raw = (process.env.HELIOS_FLASH_COOP2_BC ?? "16").trim();
  const parsed = parseInt(raw, 10);
  if (Number.isFinite(parsed) && parsed >= 16 && (parsed % 16) === 0) return parsed;
  console.warn(`[helios] ignoring HELIOS_FLASH_COOP2_BC=${raw}; expected a multiple of 16 (>=16)`);
  return 16;
}

function parseFlashCoop2SkipLseWrite(): boolean {
  const raw = (process.env.HELIOS_FLASH_COOP2_SKIP_LSE_WRITE ?? "0").trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "on" || raw === "yes";
}

function parseFlashCoop2DoubleBuf(): boolean {
  const raw = (process.env.HELIOS_FLASH_COOP2_DOUBLE_BUF ?? "0").trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "on" || raw === "yes";
}

/**
 * Pick a power-of-two flash tile that divides the sequence length exactly.
 *
 * The scalar flash kernels contain workgroup barriers. Lanes may not branch
 * around those barriers, so a partially populated final workgroup is invalid
 * Vulkan synchronization (and corrupted T=16 parity on NVIDIA when Br=32).
 * Training's usual power-of-two blocks retain their preferred tile; unusual
 * lengths safely step down to the largest exact divisor.
 */
function safeFlashTile(sequenceLength: number, requested: number): number {
  const capped = Math.max(1, Math.min(Math.floor(requested), sequenceLength));
  let tile = 1;
  while ((tile << 1) <= capped) tile <<= 1;
  while (tile > 1 && sequenceLength % tile !== 0) tile >>= 1;
  return tile;
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
    // Balanced auto-tune: include both latency-sensitive and throughput-sensitive sizes.
    const testSmall = 8192;
    const testLarge = 262144;
    const byteSize = testLarge * 4;
    const bufA = vk.createBuffer(byteSize, 0);
    const bufB = vk.createBuffer(byteSize, 0);
    const bufC = vk.createBuffer(byteSize, 0);
    const pushBuf = new Float32Array(2);

    let bestTime = Infinity;
    let bestWg = 64;
    let anyCandidate = false;

    for (const wg of WG_CANDIDATES) {
      try {
        // Create a pipeline with this WG size
        const spirv = getKernelSpirv("add_vec4", wg);
        const pipe = vk.createPipeline(spirv, 3, PUSH_SIZE);

        // Small tensor latency probe.
        let totalSmall = 0;
        {
          const vecSize = testSmall >> 2;
          pushBuf[0] = vecSize;
          pushBuf[1] = 0;
          const groups = Math.ceil(vecSize / wg);
          vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf); // warmup
          for (let i = 0; i < 5; i++) {
            totalSmall += vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf);
          }
        }

        // Larger tensor throughput probe.
        let totalLarge = 0;
        {
          const vecSize = testLarge >> 2;
          pushBuf[0] = vecSize;
          pushBuf[1] = 0;
          const groups = Math.ceil(vecSize / wg);
          vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf); // warmup
          for (let i = 0; i < 5; i++) {
            totalLarge += vk.gpuTime(pipe, [bufA, bufB, bufC], groups, 1, 1, pushBuf);
          }
        }

        const avgSmall = totalSmall / 5;
        const avgLarge = totalLarge / 5;
        // Prioritize small-op latency while still considering large-op throughput.
        const score = avgSmall * 0.7 + avgLarge * 0.3;
        anyCandidate = true;

        if (score < bestTime) {
          bestTime = score;
          bestWg = wg;
        }
      } catch {
        // Skip unsupported WG sizes and continue tuning.
      }
    }

    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    vk.destroyBuffer(bufC);

    WG_SIZE = anyCandidate ? bestWg : 128;
  } catch {
    // If timestamp queries aren't supported, keep default WG_SIZE=128
    WG_SIZE = 128;
  }
}

// ── Pipeline cache ──────────────────────────────────────────────────────────

const pipelineCache = new Map<string, number>();
let waitTimelineCount = 0;
// Wall time spent inside synchronous GPU-completion calls during the current
// step. This is distinct from timestamped kernel execution: a wait can return
// immediately after useful overlap, while profiled timestamp readback blocks.
let gpuBlockingTimeMsThisStep = 0;

function makePipelineCacheKey(name: string, numBindings: number, pushSize = PUSH_SIZE, wgSize = WG_SIZE): string {
  return `${name}:${numBindings}:${pushSize}:${wgSize}`;
}

function getPipelineLookup(
  vk: NativeAddon,
  name: string,
  numBindings: number,
  pushSize = PUSH_SIZE,
  wgSize = WG_SIZE,
): PipelineLookupResult {
  const key = makePipelineCacheKey(name, numBindings, pushSize, wgSize);
  let handle = pipelineCache.get(key);
  if (handle !== undefined) return { key, handle, created: false };

  const spirv = getKernelSpirv(name, wgSize);
  handle = vk.createPipeline(spirv, numBindings, pushSize);
  pipelineCache.set(key, handle);
  return { key, handle, created: true };
}

function getPipeline(vk: NativeAddon, name: string, numBindings: number, pushSize = PUSH_SIZE, wgSize = WG_SIZE): number {
  return getPipelineLookup(vk, name, numBindings, pushSize, wgSize).handle;
}

function waitTimelineTracked(vk: NativeAddon, timelineValue: number): void {
  if (timelineValue <= 0) return;
  waitTimelineCount++;
  const started = performance.now();
  vk.waitTimeline(timelineValue);
  gpuBlockingTimeMsThisStep += performance.now() - started;
}

// ── Buffer pool (device-local) ──────────────────────────────────────────────

// Adaptive pool cap: allow more entries for small buffers (cheap), fewer for large.
// Prevents both vk.destroyBuffer thrashing (small) and VRAM hoarding (large).
function poolMaxForSize(byteSize: number): number {
  if (byteSize <= 262_144) return OUTPUT_POOL_SMALL_PER_CLASS;
  if (byteSize <= 4_194_304) return OUTPUT_POOL_MEDIUM_PER_CLASS;
  return OUTPUT_POOL_LARGE_PER_CLASS;
}
function nonNegativeEnvInt(name: string, fallback: number): number {
  const value = Number.parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}
const OUTPUT_POOL_SMALL_PER_CLASS = nonNegativeEnvInt("HELIOS_OUTPUT_POOL_SMALL_PER_CLASS", 256);
const OUTPUT_POOL_MEDIUM_PER_CLASS = nonNegativeEnvInt("HELIOS_OUTPUT_POOL_MEDIUM_PER_CLASS", 32);
const OUTPUT_POOL_LARGE_PER_CLASS = nonNegativeEnvInt("HELIOS_OUTPUT_POOL_LARGE_PER_CLASS", 8);
const bufferPool = new Map<number, number[]>();
// Diagnostic-only handle metadata for the opt-in graph/lifetime trace. Keeping
// it behind PROFILE_GRAPH_TRACE avoids per-allocation map work in normal runs.
const traceBufferSizeByHandle = new Map<number, number>();
let bufferPoolEntries = 0;
let bufferPoolBytes = 0;
const MAX_BUFFER_POOL_ENTRIES = Math.max(0, parseInt(process.env.HELIOS_MAX_BUFFER_POOL_ENTRIES ?? "512", 10));
const MAX_OUTPUT_POOL_ENTRIES = Math.max(0, parseInt(process.env.HELIOS_MAX_OUTPUT_POOL_ENTRIES ?? "512", 10));
const LIVE_ALLOC_SOFT_CAP = Math.max(0, parseInt(process.env.HELIOS_LIVE_ALLOC_SOFT_CAP ?? "8000", 10));
const LIVE_ALLOC_HARD_CAP = Math.max(LIVE_ALLOC_SOFT_CAP + 1, parseInt(process.env.HELIOS_LIVE_ALLOC_HARD_CAP ?? "10000", 10));
const EXACT_BUFFER_SIZES = process.env.HELIOS_EXACT_BUFFER_SIZES === "1";

let _totalAllocCount = 0;
let _totalAllocBytes = 0;
let _liveAllocCount = 0;

function getLiveAllocCount(): number {
  const native = getNative();
  if (native && (native as any).getGpuStats) {
    return (native as any).getGpuStats().liveAllocs;
  }
  return _liveAllocCount;
}

function getTotalAllocBytes(): number {
  const native = getNative();
  if (native && (native as any).getGpuStats) {
    return Number((native as any).getGpuStats().totalBytes);
  }
  return _totalAllocBytes;
}

function trimPoolsForAllocPressure(vk: NativeAddon, aggressive = false): void {
  // Ensure pooled buffers are not still referenced by in-flight work.
  graph.flushAndWait();
  processPendingDestroys(vk);

  const targetLive = aggressive
    ? Math.max(0, LIVE_ALLOC_SOFT_CAP - 256)
    : LIVE_ALLOC_SOFT_CAP;
  if (getLiveAllocCount() <= targetLive) return;

  for (const [size, regions] of [...outputPool.entries()]) {
    while (regions.length > 0 && getLiveAllocCount() > targetLive) {
      const region = regions.pop()!;
      vk.destroyBuffer(region.handle);
      if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(region.handle);
      if (_liveAllocCount > 0) _liveAllocCount--;
      outputPoolEntries--;
      outputPoolBytes -= size;
    }
    if (regions.length === 0) outputPool.delete(size);
    if (getLiveAllocCount() <= targetLive) break;
  }

  if (getLiveAllocCount() > targetLive) {
    for (const [size, handles] of [...bufferPool.entries()]) {
      while (handles.length > 0 && getLiveAllocCount() > targetLive) {
        const handle = handles.pop()!;
        vk.destroyBuffer(handle);
        if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(handle);
        if (_liveAllocCount > 0) _liveAllocCount--;
        bufferPoolEntries--;
        bufferPoolBytes -= size;
      }
      if (handles.length === 0) bufferPool.delete(size);
      if (getLiveAllocCount() <= targetLive) break;
    }
  }
}

function acquireBuffer(vk: NativeAddon, byteSize: number, temporary = false): number {
  // Round to 4MB bins above 1MB. NVIDIA Vulkan driver has 4× bandwidth degradation
  // for device-local buffers whose size is not 4MB-aligned (observed on L4 driver 570.x).
  const rounded = roundPoolSize(byteSize);
  const pool = bufferPool.get(rounded);
  if (pool && pool.length > 0) {
    const handle = pool.pop()!;
    bufferPoolEntries--;
    bufferPoolBytes -= rounded;
    _flowBufferPoolHits++;
    if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.set(handle, rounded);
    return handle;
  }
  _totalAllocCount++;
  _totalAllocBytes += rounded;
  const createFreshBuffer = (): number => {
    _liveAllocCount++;
    _flowNewCreates++;
    try {
      // Output/intermediate tensors are short-lived and belong in the native
      // device-local slab pool. Uploaded inputs, parameters, and optimizer
      // state remain individual allocations so they cannot pin temp slabs.
      const handle = vk.createBuffer(rounded, 0, temporary ? 1 : 0);
      if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.set(handle, rounded);
      return handle;
    } catch (err) {
      if (_liveAllocCount > 0) _liveAllocCount--;
      throw err;
    }
  };

  if (getLiveAllocCount() >= LIVE_ALLOC_HARD_CAP) {
    trimPoolsForAllocPressure(vk, true);
  }

  try {
    return createFreshBuffer();
  } catch (_firstErr) {
    // Last-chance allocator recovery: aggressively trim pooled handles and retry once.
    trimPoolsForAllocPressure(vk, true);
    try {
      return createFreshBuffer();
    } catch (e) {
    console.error(`[helios OOM] acquireBuffer failed: requesting ${(byteSize / 1048576).toFixed(1)}MB`);
    console.error(`[helios OOM] total allocated: ${(getTotalAllocBytes() / 1048576).toFixed(1)}MB across ${_totalAllocCount} allocs (${getLiveAllocCount()} live)`);
    // Count pool sizes
    let poolBytes = 0, poolCount = 0;
    for (const [sz, bufs] of bufferPool) { poolCount += bufs.length; poolBytes += sz * bufs.length; }
    console.error(`[helios OOM] buffer pool: ${poolCount} buffers, ${(poolBytes / 1048576).toFixed(1)}MB`);
    let outPoolBytes = 0, outPoolCount = 0;
      for (const [sz, regs] of outputPool) { outPoolCount += regs.length; outPoolBytes += sz * regs.length; }
    console.error(`[helios OOM] output pool: ${outPoolCount} regions, ${(outPoolBytes / 1048576).toFixed(1)}MB`);
      try {
        console.error(`[helios OOM] native allocator: ${JSON.stringify(vk.getAllocatorStats?.() ?? {})}`);
      } catch (statsError) {
        console.error(`[helios OOM] native allocator stats unavailable: ${String(statsError)}`);
      }
      throw e;
    }
  }
}

function releaseBuffer(vk: NativeAddon, handle: number, byteSize: number): void {
  const rounded = roundPoolSize(byteSize);
  let pool = bufferPool.get(rounded);
  if (!pool) { pool = []; bufferPool.set(rounded, pool); }
  if (pool.length < poolMaxForSize(rounded) && bufferPoolEntries < MAX_BUFFER_POOL_ENTRIES) {
    pool.push(handle);
    bufferPoolEntries++;
    bufferPoolBytes += rounded;
  } else {
    vk.destroyBuffer(handle);
    if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(handle);
    _liveAllocCount--;
    _flowDestroys++;
  }
}

// Push constant data — reusable typed array (8 bytes = 2 x f32: [len, scalar])
const pushData = new Float32Array(2);
const PUSH_SIZE = 8;  // bytes — all kernels use 2 x f32 push constants

// ── GPU residence tracking ──────────────────────────────────────────────────

interface GpuHandle {
  handle: number;
  byteSize: number;
  refs: number;
  released: boolean;
  staticSlotId?: number;
}

/** Maps TensorData → its GPU buffer. Keyed on the object identity. */
const gpuResidence = new WeakMap<object, GpuHandle>();

// ── Leak diagnostics ──
let _diagAllocsThisStep = 0;
let _diagReleasesThisStep = 0;
let _diagFrReleasesThisStep = 0;


// ── Buffer flow counters (reset per gpuMemStats call) ──
let _flowNewCreates = 0;        // vk.createBuffer calls (new allocs)
let _flowDestroys = 0;          // vk.destroyBuffer calls
let _flowOutputPoolHits = 0;    // acquireOutputRegion reused from pool
let _flowOutputPoolMisses = 0;  // acquireOutputRegion created new
let _flowOutputPoolReturns = 0; // releaseOutputRegion → pool
let _flowOutputPoolOverflows = 0; // releaseOutputRegion → pendingDestroys
let _flowBufferPoolHits = 0;    // acquireBuffer reused from bufferPool
let _flowEnsureGpuHits = 0;     // ensureGpu found existing
let _flowEnsureGpuUploads = 0;  // ensureGpu uploaded new

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
  _diagFrReleasesThisStep++;
  if (info.staticSlotId !== undefined) {
    // Static slots are owned by the plan runtime, not by any one TensorData.
    if (info.refs <= 0) info.released = true;
    return;
  }
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
  if (existing) { _flowEnsureGpuHits++; return existing.handle; }
  // Upload to a new device-local buffer
  const byteSize = td.data.length * 4;
  const handle = acquireBuffer(vk, byteSize);
  // Pad upload data to match buffer allocation size.
  // NVIDIA L4 driver (570.x) has ~4× bandwidth degradation when the staging
  // memcpy doesn't cover the full device buffer — even if the vkCmdCopyBuffer
  // region is padded. Padding the Float32Array ensures full page coverage.
  const rounded = roundPoolSize(byteSize);
  let uploadData: Float32Array;
  if (rounded > byteSize) {
    uploadData = new Float32Array(rounded >> 2);
    uploadData.set(toF32(td));
  } else {
    uploadData = toF32(td);
  }
  vk.uploadBuffer(handle, uploadData);
  const info: GpuHandle = { handle, byteSize: rounded, refs: 1, released: false };
  gpuResidence.set(td, info);
  gpuCleanup.register(td, info);
  _flowEnsureGpuUploads++;
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
  _diagReleasesThisStep++;
  if (info.staticSlotId !== undefined) {
    if (info.refs <= 0) info.released = true;
    return;
  }
  if (info.refs <= 0) {
    info.released = true;
    gpuResidence.delete(td);
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
  const expected = shapeSize(td.shape);
  const vk = getNative();
  let cached: Float32Array | null = null;
  Object.defineProperty(td, "data", {
    get(): Float32Array {
      if (!cached) {
        const tv = graph.flush();
        // Must wait for GPU to finish before reading back (batch dispatch
        // does not set per-buffer lastWriteTimeline, so readBuffer alone
        // won't wait for in-place ops like AdamW on mapped/coherent memory).
        waitTimelineTracked(vk, tv);
        graph.traceHostRead(handle);
        const raw = vk.readBuffer(handle);
        cached = raw.length > expected ? raw.subarray(0, expected) : raw;
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
  staticSlotId?: number;
}

const outputPool = new Map<number, OutputRegion[]>();
let outputPoolEntries = 0;
let outputPoolBytes = 0;
// Only allocated when a static plan is requested, preserving the normal hot
// path. It lets record() replace a just-acquired dynamic output with the
// preplanned handle, including multi-output kernels.
const unsubmittedOutputRegions = STATIC_SLOT_PLAN ? new Map<number, OutputRegion>() : null;

// Cached timeline completion value — avoids N-API call per-op.
// Refreshed lazily: only re-queries GPU when cache is stale (after flush).
let _completedCache = 0;
let _completedCacheDirty = true;
function getCachedCompleted(vk: NativeAddon): number {
  if (_completedCacheDirty) {
    _completedCache = vk.getCompleted();
    _completedCacheDirty = false;
  }
  return _completedCache;
}
function invalidateCompletedCache(): void { _completedCacheDirty = true; }

/**
 * Round allocation size up to coarse bins to reduce pool fragmentation.
 * Collapses many similar sizes into fewer size classes for better reuse.
 */
function roundPoolSize(bytes: number): number {
  // Exact classes avoid material tail waste for stable, large training shapes.
  // The coarse policy remains the default because it can improve reuse when a
  // workload emits many nearby dynamic sizes. Tensor byte counts are f32-aligned.
  if (EXACT_BUFFER_SIZES) return Math.max(4, bytes);
  if (bytes <= 4096) return 4096;
  if (bytes <= 1_048_576) return Math.ceil(bytes / 262144) * 262144;  // 256KB bins up to 1MB
  return Math.ceil(bytes / 4_194_304) * 4_194_304;  // 4MB bins above 1MB
}

function acquireOutputRegion(vk: NativeAddon, byteSize: number): OutputRegion {
  const rounded = roundPoolSize(byteSize);
  const completed = getCachedCompleted(vk);
  const pool = outputPool.get(rounded);
  if (pool) {
    // LIFO scan: prefer most recently released buffer for L2 cache locality.
    // The most recent buffer is most likely still in L2, improving bandwidth
    // for repeated operations on the same data size.
    for (let i = pool.length - 1; i >= 0; i--) {
      if (pool[i].readyValue <= completed) {
        const region = pool.splice(i, 1)[0];
        outputPoolEntries--;
        outputPoolBytes -= rounded;
        _flowOutputPoolHits++;
        if (unsubmittedOutputRegions) unsubmittedOutputRegions.set(region.handle, region);
        return region;
      }
    }
  }
  _flowOutputPoolMisses++;
  const region = { handle: acquireBuffer(vk, rounded, true), byteSize: rounded, readyValue: 0 };
  if (unsubmittedOutputRegions) unsubmittedOutputRegions.set(region.handle, region);
  return region;
}

/**
 * Pending buffer destructions: handles that overflowed both outputPool and
 * bufferPool, awaiting GPU completion before they can be safely destroyed.
 * Without this, vk.destroyBuffer() on individually-allocated buffers would
 * free GPU memory while the GPU is still accessing it → undefined behavior.
 */
const pendingDestroys: { handle: number; readyValue: number }[] = [];

function releaseOutputRegion(region: OutputRegion, submitValue: number): void {
  if (region.staticSlotId !== undefined) return;
  if (unsubmittedOutputRegions) unsubmittedOutputRegions.delete(region.handle);
  region.readyValue = submitValue;
  let pool = outputPool.get(region.byteSize);
  if (!pool) { pool = []; outputPool.set(region.byteSize, pool); }
  if (pool.length < poolMaxForSize(region.byteSize) && outputPoolEntries < MAX_OUTPUT_POOL_ENTRIES) {
    pool.push(region);
    outputPoolEntries++;
    outputPoolBytes += region.byteSize;
    _flowOutputPoolReturns++;
  } else {
    // Pool full — defer buffer destruction until GPU has finished using it.
    // The buffer's memory may still be referenced by the just-submitted batch.
    pendingDestroys.push({ handle: region.handle, readyValue: submitValue });
    _flowOutputPoolOverflows++;
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
      if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(pendingDestroys[i].handle);
      _liveAllocCount--;
      _flowDestroys++;
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
      if (!cached) {
        graph.traceHostRead(region.handle);
        cached = vk.readBuffer(region.handle);
      }
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

type PendingOpKind = "binary" | "unary" | "softmax" | "softmax_online" | "layernorm" | "matmul" | "reduce_sum" | "backward" | "optimizer" | "inplace";

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
  elementCount?: number;    // actual element count (for DGC: scalar BDA kernel dispatch)
}

class StaticSlotRuntime {
  private readonly handles = new Map<number, { handle: number; byteSize: number }>();
  private active = false;
  private step = 0;
  private bindingsThisStep = 0;
  private completedActiveSteps = 0;

  constructor(private readonly plan: StaticSlotPlan, private readonly warmupSteps: number) {}

  beginStep(): { activated: boolean } {
    if (this.active) {
      throw new Error("[helios static slots] prior training phase was not explicitly finished");
    }
    this.step++;
    this.bindingsThisStep = 0;
    const nextActive = this.step > this.warmupSteps;
    const activated = nextActive && !this.active;
    this.active = nextActive;
    return { activated };
  }

  finishStep(operationCount: number): void {
    if (!this.active) return;
    if (operationCount !== this.plan.operationCount) {
      throw new Error(
        `[helios static slots] training phase recorded ${operationCount}/${this.plan.operationCount} planned operations`,
      );
    }
    if (this.bindingsThisStep !== this.plan.assignmentCount) {
      throw new Error(
        `[helios static slots] training phase bound ${this.bindingsThisStep}/${this.plan.assignmentCount} planned values`,
      );
    }
    this.completedActiveSteps++;
    this.active = false;
  }

  bindOperation(vk: NativeAddon, operationIndex: number, op: PendingOp): void {
    if (!this.active) return;
    if (operationIndex >= this.plan.operationCount) {
      throw new Error(
        `[helios static slots] operation ${operationIndex} exceeds planned count ${this.plan.operationCount}`,
      );
    }
    const assignments = this.plan.assignmentsByOperation.get(operationIndex);
    if (!assignments || assignments.length === 0) return;
    const handles = op.allBufs ?? [...op.inputBufs, op.outputRegion.handle];
    for (const assignment of assignments) {
      if (assignment.producerKind !== op.kind || assignment.producerKernel !== op.kernel) {
        throw new Error(
          `[helios static slots] op ${operationIndex} expected ${assignment.producerKind}/${assignment.producerKernel}, ` +
            `got ${op.kind}/${op.kernel}`,
        );
      }
      const oldHandle = handles[assignment.producerPosition];
      const target = unsubmittedOutputRegions?.get(oldHandle);
      if (!target) {
        throw new Error(
          `[helios static slots] op ${operationIndex} binding ${assignment.producerPosition} is not a fresh output region`,
        );
      }
      if (target.byteSize !== assignment.logicalBytes) {
        throw new Error(
          `[helios static slots] op ${operationIndex} binding ${assignment.producerPosition} size ` +
            `${target.byteSize} != planned ${assignment.logicalBytes}`,
        );
      }
      const slotSpec = this.plan.slots[assignment.slotId];
      let slot = this.handles.get(assignment.slotId);
      if (!slot) {
        slot = { handle: acquireBuffer(vk, slotSpec.allocationBytes, true), byteSize: slotSpec.allocationBytes };
        this.handles.set(assignment.slotId, slot);
      }

      // The old region has not appeared in a command yet. Return a copy to the
      // ordinary pool, then mutate the caller-owned region so the TensorData
      // and every descriptor see the stable slot handle.
      unsubmittedOutputRegions!.delete(oldHandle);
      releaseOutputRegion({ ...target }, target.readyValue);
      target.handle = slot.handle;
      target.byteSize = slot.byteSize;
      target.readyValue = 0;
      target.staticSlotId = assignment.slotId;
      handles[assignment.producerPosition] = slot.handle;
      this.bindingsThisStep++;
    }
    if (op.allBufs) {
      op.allBufs = handles;
    } else {
      const outputPosition = op.inputBufs.length;
      const assignment = assignments.find((row) => row.producerPosition === outputPosition);
      if (!assignment) {
        throw new Error(`[helios static slots] op ${operationIndex} planned a non-output binding without allBufs`);
      }
      op.outputRegion.handle = handles[outputPosition];
    }
  }

  destroy(vk: NativeAddon): void {
    for (const { handle } of this.handles.values()) {
      vk.destroyBuffer(handle);
      if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(handle);
      if (_liveAllocCount > 0) _liveAllocCount--;
    }
    this.handles.clear();
  }

  stats(): Record<string, number> {
    return {
      staticSlotPlanEnabled: 1,
      staticSlotPlanActive: this.active ? 1 : 0,
      staticSlotPlanStep: this.step,
      staticSlotPlanSlotsDeclared: this.plan.slots.length,
      staticSlotPlanSlotsAllocated: this.handles.size,
      staticSlotPlanBytesDeclared: this.plan.totalSlotBytes,
      staticSlotPlanBytesAllocated: [...this.handles.values()].reduce((sum, slot) => sum + slot.byteSize, 0),
      staticSlotPlanAssignments: this.plan.assignmentCount,
      staticSlotBindingsThisStep: this.bindingsThisStep,
      staticSlotCompletedSteps: this.completedActiveSteps,
    };
  }
}

const staticSlotRuntime = STATIC_SLOT_PLAN
  ? new StaticSlotRuntime(STATIC_SLOT_PLAN, STATIC_SLOT_PLAN_WARMUP_STEPS)
  : null;

export type GraphTraceEvent =
  | {
      event: "op";
      order: number;
      kind: PendingOpKind;
      kernel: string;
      bufferCount: number;
      bufferIds: number[];
      bufferBytes: number[];
      writeMask: number;
      groups: [number, number, number];
      pushSize: number;
      shape: number[];
      elementCount: number | null;
    }
  | {
      event: "flush";
      order: number;
      operationCount: number;
      withWait: boolean;
    }
  | {
      event: "host_read";
      order: number;
      operationCount: number;
      bufferId: number;
      bufferBytes: number;
    };

export interface GpuStepStats {
  profilingEnabled: boolean;
  timingEnabled: boolean;
  operations: number;
  flushes: number;
  waitedFlushes: number;
  dgcFlushes: number;
  timestampedFlushes: number;
  batchGpuTimeUs: number;
  dispatchGpuTimeUs: number;
  /** Host wall time spent inside synchronous GPU-completion calls. */
  gpuBlockingTimeMs: number;
  operationsPerFlush: number;
  graphSignature: string | null;
  graphTrace: GraphTraceEvent[] | null;
  byKind: Array<{ name: string; count: number; gpuTimeUs: number }>;
  byKernel: Array<{ name: string; count: number; gpuTimeUs: number }>;
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
  flushesThisStep = 0;
  waitedFlushesThisStep = 0;
  dgcFlushesThisStep = 0;
  timestampedFlushesThisStep = 0;
  batchGpuTimeUsThisStep = 0;
  dispatchGpuTimeUsThisStep = 0;
  private opsByKindThisStep = new Map<PendingOpKind, number>();
  private opsByKernelThisStep = new Map<string, number>();
  private gpuTimeByKindThisStep = new Map<PendingOpKind, number>();
  private gpuTimeByKernelThisStep = new Map<string, number>();
  private graphHashA = 0x811c9dc5;
  private graphHashB = 0x9e3779b9;
  private graphTraceThisStep: GraphTraceEvent[] = [];
  private graphTraceBufferIds = new Map<number, number>();
  private graphTraceNextBufferId = 0;

  private graphTraceBufferId(handle: number): number {
    const existing = this.graphTraceBufferIds.get(handle);
    if (existing !== undefined) return existing;
    const id = this.graphTraceNextBufferId++;
    this.graphTraceBufferIds.set(handle, id);
    return id;
  }

  traceHostRead(handle: number): void {
    if (!PROFILE_GRAPH_TRACE) return;
    this.graphTraceThisStep.push({
      event: "host_read",
      order: this.graphTraceThisStep.length,
      operationCount: this.opsThisStep,
      bufferId: this.graphTraceBufferId(handle),
      bufferBytes: traceBufferSizeByHandle.get(handle) ?? 0,
    });
  }

  private mixGraphWord(value: number): void {
    if (!PROFILE_GRAPH_SIGNATURE) return;
    const word = value >>> 0;
    this.graphHashA = Math.imul(this.graphHashA ^ word, 0x01000193) >>> 0;
    this.graphHashB = Math.imul((this.graphHashB + word + 0x7f4a7c15) >>> 0, 0x85ebca6b) >>> 0;
  }

  private mixGraphText(value: string): void {
    if (!PROFILE_GRAPH_SIGNATURE) return;
    for (let i = 0; i < value.length; i++) this.mixGraphWord(value.charCodeAt(i));
    this.mixGraphWord(0xff);
  }

  private graphSignature(): string | null {
    if (!PROFILE_GRAPH_SIGNATURE) return null;
    return `${this.graphHashA.toString(16).padStart(8, "0")}${this.graphHashB.toString(16).padStart(8, "0")}`;
  }

  // DGC state: pipeline slot for the BDA binary op kernel, -1 if not set up
  private dgcBinaryPipeSlot = -1;
  // Set of regular pipeline slots that map to the DGC binary pipeline
  private dgcEligiblePipelines = new Set<number>();
  private dgcReady = false;
  // DGC packed buffer (reused across flushes)
  private dgcPackedBuf: ArrayBuffer | null = null;
  // Regular packed buffer (reused across flushes to avoid allocation)
  private packedBuf: ArrayBuffer | null = null;

  attach(vk: NativeAddon): void { this.vk = vk; }

  get length(): number { return this.pending.length; }
  get lastFlushTimeline(): number { return this._lastFlushTimeline; }

  /**
   * Set up DGC for binary ops. Called once when DGC is available.
   * Creates a BDA pipeline and configures the DGC layout.
   */
  setupDGC(vk: NativeAddon): void {
    if (this.dgcReady || !vk.dgcSetup || !vk.batchExecuteAllDGC) return;
    try {
      // Create BDA binary op pipeline (0 descriptor bindings, 32-byte push constants)
      const spirv = getKernelSpirv("add_bda", WG_SIZE);
      const pipeSlot = vk.createPipeline(spirv, 0, 32);
      this.dgcBinaryPipeSlot = pipeSlot;

      // Set up DGC with this pipeline: 32 bytes push constants, up to 4096 sequences
      const ok = vk.dgcSetup(pipeSlot, 32, 4096);
      if (!ok) return;

      this.dgcReady = true;
    } catch {
      // DGC setup failed — continue with regular dispatch
    }
  }

  /**
   * Register a regular pipeline slot as DGC-eligible (maps to the BDA binary op kernel).
   * Called when a binary op pipeline is first created.
   */
  markDGCEligible(regularPipeSlot: number): void {
    if (this.dgcReady) {
      this.dgcEligiblePipelines.add(regularPipeSlot);
    }
  }

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

    if (staticSlotRuntime && this.vk) {
      staticSlotRuntime.bindOperation(this.vk, this.opsThisStep, op);
    }

    if (PROFILE_GRAPH_SIGNATURE) {
      this.mixGraphWord(0x4f500001);
      this.mixGraphText(op.kind);
      this.mixGraphText(op.kernel);
      this.mixGraphWord(bufCount);
      this.mixGraphWord(op.writeMask);
      this.mixGraphWord(op.groups[0]);
      this.mixGraphWord(op.groups[1]);
      this.mixGraphWord(op.groups[2]);
      this.mixGraphWord(op.pushSize);
      this.mixGraphWord(op.shape.length);
      for (const dimension of op.shape) this.mixGraphWord(dimension);
      this.mixGraphWord(op.elementCount ?? 0xffffffff);
    }
    if (PROFILE_GRAPH_TRACE) {
      const bufferHandles = op.allBufs ?? [...op.inputBufs, op.outputRegion.handle];
      this.graphTraceThisStep.push({
        event: "op",
        order: this.graphTraceThisStep.length,
        kind: op.kind,
        kernel: op.kernel,
        bufferCount: bufCount,
        bufferIds: bufferHandles.map((handle) => this.graphTraceBufferId(handle)),
        bufferBytes: bufferHandles.map((handle) => traceBufferSizeByHandle.get(handle) ?? 0),
        writeMask: op.writeMask,
        groups: [...op.groups],
        pushSize: op.pushSize,
        shape: [...op.shape],
        elementCount: op.elementCount ?? null,
      });
    }

    this.pending.push(op);
    this.pendingPackedBytes += op.packedBytes;
    this.totalOpsRecorded++;
    this.opsThisStep++;
    if (PROFILE_GPU_OPS || PROFILE_GPU_TIMESTAMPS) {
      this.opsByKindThisStep.set(op.kind, (this.opsByKindThisStep.get(op.kind) ?? 0) + 1);
      this.opsByKernelThisStep.set(op.kernel, (this.opsByKernelThisStep.get(op.kernel) ?? 0) + 1);
    }
    if (this.pending.length >= MAX_PENDING_OPS) this.flush();
  }

  resetStepStats(): void {
    this.opsThisStep = 0;
    this.flushesThisStep = 0;
    this.waitedFlushesThisStep = 0;
    this.dgcFlushesThisStep = 0;
    this.timestampedFlushesThisStep = 0;
    this.batchGpuTimeUsThisStep = 0;
    this.dispatchGpuTimeUsThisStep = 0;
    gpuBlockingTimeMsThisStep = 0;
    this.opsByKindThisStep.clear();
    this.opsByKernelThisStep.clear();
    this.gpuTimeByKindThisStep.clear();
    this.gpuTimeByKernelThisStep.clear();
    this.graphHashA = 0x811c9dc5;
    this.graphHashB = 0x9e3779b9;
    this.graphTraceThisStep = [];
    this.graphTraceBufferIds.clear();
    this.graphTraceNextBufferId = 0;
  }

  getStepStats(): GpuStepStats {
    const rows = (
      counts: Map<string, number>,
      times: Map<string, number>,
    ): Array<{ name: string; count: number; gpuTimeUs: number }> =>
      [...counts.entries()]
        .map(([name, count]) => ({ name, count, gpuTimeUs: times.get(name) ?? 0 }))
        .sort((a, b) =>
          PROFILE_GPU_TIMESTAMPS
            ? b.gpuTimeUs - a.gpuTimeUs || b.count - a.count || a.name.localeCompare(b.name)
            : b.count - a.count || a.name.localeCompare(b.name),
        );
    return {
      profilingEnabled: PROFILE_GPU_OPS || PROFILE_GPU_TIMESTAMPS,
      timingEnabled: PROFILE_GPU_TIMESTAMPS,
      operations: this.opsThisStep,
      flushes: this.flushesThisStep,
      waitedFlushes: this.waitedFlushesThisStep,
      dgcFlushes: this.dgcFlushesThisStep,
      timestampedFlushes: this.timestampedFlushesThisStep,
      batchGpuTimeUs: this.batchGpuTimeUsThisStep,
      dispatchGpuTimeUs: this.dispatchGpuTimeUsThisStep,
      gpuBlockingTimeMs: gpuBlockingTimeMsThisStep,
      operationsPerFlush: this.flushesThisStep > 0 ? this.opsThisStep / this.flushesThisStep : 0,
      graphSignature: this.graphSignature(),
      graphTrace: PROFILE_GRAPH_TRACE ? this.graphTraceThisStep : null,
      byKind: rows(this.opsByKindThisStep, this.gpuTimeByKindThisStep),
      byKernel: rows(this.opsByKernelThisStep, this.gpuTimeByKernelThisStep),
    };
  }

  /** Schedule an intermediate output region for release after the next flush.
   *  When the graph has no pending ops (e.g. after sync()), release immediately
   *  to the output pool so the buffer is available for reuse right away. This
   *  prevents double-buffering in tight loops (benchmark or training) where the
   *  same-size buffer would otherwise alternate between two allocations, filling
   *  the L2 cache and preventing input data from staying cached.
   */
  deferRelease(region: OutputRegion): void {
    if (this.pending.length === 0) {
      // No pending ops — buffer was used by already-completed work.
      // Release immediately so it's available for the next acquire.
      releaseOutputRegion(region, this._lastFlushTimeline);
    } else {
      this.deferredReleases.push(region);
    }
  }

  /**
   * Flush all pending ops as a single batch dispatch.
   * Returns the timeline value for the batch, or the last flush value if nothing pending.
   */
  flush(withWait = false): number {
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
    this.flushesThisStep++;
    if (withWait) this.waitedFlushesThisStep++;
    if (PROFILE_GRAPH_SIGNATURE) {
      this.mixGraphWord(0x464c5553);
      this.mixGraphWord(ops.length);
      this.mixGraphWord(withWait ? 1 : 0);
    }
    if (PROFILE_GRAPH_TRACE) {
      this.graphTraceThisStep.push({
        event: "flush",
        order: this.graphTraceThisStep.length,
        operationCount: ops.length,
        withWait,
      });
    }

    // ── DGC fast path: when all ops are DGC-eligible binary ops ──
    // Uses device-generated commands (single GPU submit, no per-op descriptor sets).
    if (!PROFILE_GPU_TIMESTAMPS && this.dgcReady && ops.length >= 2 && typeof vk.batchExecuteAllDGC === "function") {
      // Check if ALL ops are DGC-eligible (binary ops with 3 buffers)
      let allEligible = true;
      for (let i = 0; i < ops.length; i++) {
        if (!this.dgcEligiblePipelines.has(ops[i].pipeline) ||
            ops[i].inputBufs.length !== 2 || ops[i].allBufs) {
          allEligible = false;
          break;
        }
      }

      if (allEligible) {
        this.dgcFlushesThisStep++;
        // Pack DGC format: per op [bufA(i32), bufB(i32), bufC(i32), count(u32), gX(u32), gY(u32), gZ(u32)] = 28 bytes
        // The BDA kernel is SCALAR (processes 1 element/thread), so we use the actual
        // element count and recompute dispatch groups, regardless of whether the original
        // dispatch used vec4 or vec4x2.
        const DGC_OP_SIZE = 28;
        const dgcTotal = ops.length * DGC_OP_SIZE;
        if (!this.dgcPackedBuf || this.dgcPackedBuf.byteLength < dgcTotal) {
          this.dgcPackedBuf = new ArrayBuffer(Math.max(dgcTotal, 4096));
        }
        const dv = new DataView(this.dgcPackedBuf);
        let off = 0;
        for (const op of ops) {
          const count = op.elementCount ?? (op.push[0] | 0);
          const gX = Math.ceil(count / WG_SIZE);
          dv.setInt32(off, op.inputBufs[0], true); off += 4;
          dv.setInt32(off, op.inputBufs[1], true); off += 4;
          dv.setInt32(off, op.outputRegion.handle, true); off += 4;
          dv.setUint32(off, count, true); off += 4;
          dv.setUint32(off, gX, true); off += 4;
          dv.setUint32(off, 1, true); off += 4;
          dv.setUint32(off, 1, true); off += 4;
        }

        const tv = vk.batchExecuteAllDGC(this.dgcPackedBuf, ops.length);
        this._lastFlushTimeline = tv;
        invalidateCompletedCache();

        for (const region of this.deferredReleases) {
          releaseOutputRegion(region, tv);
        }
        this.deferredReleases = [];
        return tv;
      }
    }

    // ── Regular path: pack ops into binary format for C dispatch ──
    // Fresh ArrayBuffer allocations use zeroed pages (hot in L1/L2), which is faster
    // for ops with larger packed sizes. Only reuse buffer for tiny dispatches where
    // allocation overhead dominates.
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

    // Prefer combined batchExecuteAll (single N-API call) for lower dispatch overhead.
    // Falls back to 3-call path (batchBegin + batchDispatchMany + batchSubmit).
    let tv: number;
    if (PROFILE_GPU_TIMESTAMPS) {
      if (typeof vk.batchExecuteAllProfiled !== "function") {
        throw new Error(
          "HELIOS_PROFILE_GPU_TIMESTAMPS=1 requires a native addon with batchExecuteAllProfiled()",
        );
      }
      // Timestamp readback makes this native call synchronous. Count its wall
      // time just like waitTimeline so GPU execution is not mislabeled as host
      // graph construction in the trainer's direct split.
      const profileStarted = performance.now();
      const profile = vk.batchExecuteAllProfiled(packed, ops.length);
      gpuBlockingTimeMsThisStep += performance.now() - profileStarted;
      if (profile.dispatchCount !== ops.length || profile.dispatchTimesUs.length !== ops.length) {
        throw new Error(
          `Helios profiler dispatch mismatch: recorded=${profile.dispatchCount} expected=${ops.length}`,
        );
      }
      tv = profile.timeline;
      this.timestampedFlushesThisStep++;
      this.batchGpuTimeUsThisStep += profile.batchGpuTimeUs;
      for (let i = 0; i < ops.length; i++) {
        const timeUs = profile.dispatchTimesUs[i];
        const op = ops[i];
        this.dispatchGpuTimeUsThisStep += timeUs;
        this.gpuTimeByKindThisStep.set(
          op.kind,
          (this.gpuTimeByKindThisStep.get(op.kind) ?? 0) + timeUs,
        );
        this.gpuTimeByKernelThisStep.set(
          op.kernel,
          (this.gpuTimeByKernelThisStep.get(op.kernel) ?? 0) + timeUs,
        );
      }
      // The native profiling path has already waited for timestamp readback.
    } else if (!DISABLE_BATCH_DISPATCH_MANY && typeof vk.batchExecuteAll === "function") {
      tv = vk.batchExecuteAll(packed, ops.length);
      if (withWait && tv > 0) waitTimelineTracked(vk, tv);
    } else {
      vk.batchBegin();
      vk.batchDispatchMany(packed, ops.length);
      tv = vk.batchSubmit();
      if (withWait && tv > 0) waitTimelineTracked(vk, tv);
    }
    this._lastFlushTimeline = tv;
    invalidateCompletedCache();

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

  /** Flush all pending ops and wait for GPU completion. */
  flushAndWait(): void {
    this.flush(true);
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
  // Use region.byteSize (rounded) for pool key consistency.
  // acquireOutputRegion rounds to 4MB bins; release must match that key
  // or every iteration allocates a new buffer (vkAllocateMemory overhead).
  const gpuInfo: GpuHandle = {
    handle: region.handle,
    byteSize: region.byteSize,
    refs: 1,
    released: false,
    staticSlotId: region.staticSlotId,
  };
  const td: TensorData = {
    shape,
    dtype: "f32",
    get data(): Float32Array {
      if (!cached) {
        // Flush any pending ops that might write to our output
        const tv = graph.flush();
        // Wait for the batch to complete on GPU before reading
        waitTimelineTracked(vk, tv);
        graph.traceHostRead(region.handle);
        const raw = vk.readBuffer(region.handle);
        // Buffer may be larger than shape (4MB pool rounding) — truncate
        const expected = shapeSize(shape);
        cached = raw.length > expected ? raw.subarray(0, expected) : raw;
      }
      return cached;
    },
  };
  // Track GPU residence so subsequent ops can find this buffer
  gpuResidence.set(td, gpuInfo);
  gpuCleanup.register(td, gpuInfo);
  _diagAllocsThisStep++;
  return td;
}

/** Like graphLazyTensor but for f16 output buffers (2 bytes per element). */
function graphLazyTensorF16(vk: NativeAddon, shape: Shape, region: OutputRegion): TensorData {
  const size = shapeSize(shape);
  const gpuInfo: GpuHandle = {
    handle: region.handle,
    byteSize: region.byteSize,
    refs: 1,
    released: false,
    staticSlotId: region.staticSlotId,
  };
  const td: TensorData = {
    shape: [...shape],
    dtype: "f16",
    get data(): Uint16Array {
      // F16 data shouldn't normally be read back to CPU — this is a fallback
      const tv = graph.flush();
      waitTimelineTracked(vk, tv);
      // readBuffer returns Float32Array; reinterpret as Uint16Array
      graph.traceHostRead(region.handle);
      const f32 = vk.readBuffer(region.handle);
      return new Uint16Array(f32.buffer, f32.byteOffset, size);
    },
  };
  gpuResidence.set(td, gpuInfo);
  gpuCleanup.register(td, gpuInfo);
  return td;
}

// ── HeliosBackend ───────────────────────────────────────────────────────────

export interface GpuDeviceInfo extends NativeDeviceInfo {
  workgroupSize: number;
  minGpuSize: number;
  computeSubgroupArithmeticSupported: boolean;
  nativeSubgroup32: boolean;
}

export interface CoopMatmulStats {
  totalMatmulDispatches: number;
  coopDispatches: number;
  coopDirectDispatches: number;
  coopPadded2DDispatches: number;
  coopPaddedBatchedDispatches: number;
  coopTransposedARewriteDispatches: number;
  coopHitRate: number;
  lastCoopKernel: string | null;
  lastCoopShape: {
    M: number;
    N: number;
    K: number;
    batchSize: number;
    transposedA: boolean;
    transposedB: boolean;
  } | null;
  shapeCounts: Array<{
    key: string;
    kernel: string;
    M: number;
    N: number;
    K: number;
    batchSize: number;
    transposedA: boolean;
    transposedB: boolean;
    count: number;
  }>;
}

type MatmulTile = 16 | 32;

export interface MatmulTileAutotuneDecision {
  key: string;
  kernel: string;
  shape: { M: number; N: number; K: number; batchSize: number };
  tile: MatmulTile;
  tile16GpuTimeUs: number | null;
  tile32GpuTimeUs: number | null;
  tile16SamplesUs: number[];
  tile32SamplesUs: number[];
  reason: "measured" | "override" | "capability" | "heuristic" | "probe-fallback";
}

export class HeliosBackend implements Backend {
  readonly name = "helios";
  private readonly rng = new SeededRng(42);
  private initialized = false;
  private _minGpuSize = DEFAULT_MIN_GPU_SIZE;
  private _f16Supported = false;
  private _deviceName = "";
  private _vendorId = 0;
  private _nativeDeviceInfo: NativeDeviceInfo | null = null;
  private _hasAsyncTransfer = false;
  private _coopMatSupported = false;
  private _coopMatPaused = false; // Temporarily disable coop matmul (e.g. during backward)
  private _columnSumRowLanes: ColumnSumRowLanes = COLUMN_SUM_ROW_LANES;
  private _lastColumnSumKernel: string | null = null;
  private _coopMat2Supported = false;
  private _coopM = 0;
  private _coopN = 0;
  private _coopK = 0;
  private _coopKTile = 0; // coopK * kMulti — effective K-tile width for alignment
  private _kMulti = 1;
  private _hasPushDescriptors = false;
  private _matmulDispatches = 0;
  private _coopDispatches = 0;
  private _coopDirectDispatches = 0;
  private _coopPadded2DDispatches = 0;
  private _coopPaddedBatchedDispatches = 0;
  private _coopTransposedARewriteDispatches = 0;
  private _lastCoopKernel: string | null = null;
  private _lastCoopShape: CoopMatmulStats["lastCoopShape"] = null;
  private _coopShapeCounts = new Map<string, CoopMatmulStats["shapeCounts"][number]>();
  private _coopF16InputCache = new Map<TensorData, TensorData>();
  private _coopF16InputCacheLastFlushTimeline = -1;
  private _matmulTileCache = new Map<string, MatmulTile>();
  private _matmulTileDecisions = new Map<string, MatmulTileAutotuneDecision>();
  private _matmulReg2x2WarningEmitted = false;
  private _matmulReg4x2WarningEmitted = false;
  private _matmulReg4x2K32WarningEmitted = false;
  private _flashCoop2ScopeResolved: FlashCoop2ScopeTag | null = null;
  private _lastFlashDispatchDebug: FlashDispatchDebug | null = null;
  private _flashDispatchDebugEnabled = parseFlashDispatchDebugEnabled();
  private _flashFwdPreferCoop2 = parseFlashFwdPreferCoop2();
  private _flashFwdCoop2Strict = parseFlashFwdCoop2Strict();
  private _flashFwdCoop2Ready: boolean | null = null;
  private _flashCoop2PreferF16Input = parseFlashCoop2PreferF16Input();
  private _flashCoop2LocalSize = parseFlashCoop2LocalSize();
  private _flashCoop2QTiles = parseFlashCoop2QTiles();
  private _flashCoop2BlockCols = parseFlashCoop2BlockCols();
  private _flashCoop2SkipLseWrite = parseFlashCoop2SkipLseWrite();
  private _flashCoop2DoubleBuf = parseFlashCoop2DoubleBuf();
  private _lastUnlikelihoodStats: UnlikelihoodLossStats | null = null;


  /** Override the minimum element count for GPU dispatch (useful for benchmarking). */
  setMinGpuSize(n: number): void { this._minGpuSize = n; }

  private setLastFlashDispatchDebug(debug: FlashDispatchDebug): void {
    if (!this._flashDispatchDebugEnabled) return;
    this._lastFlashDispatchDebug = debug;
  }

  getLastFlashDispatchDebug(): FlashDispatchDebug | null {
    if (!this._flashDispatchDebugEnabled) return null;
    return this._lastFlashDispatchDebug;
  }

  getLastCrossEntropyUnlikelihoodMaskedStats(): UnlikelihoodLossStats | null {
    return this._lastUnlikelihoodStats;
  }

  getWaitTimelineCount(): number {
    return waitTimelineCount;
  }

  getMatmulTileAutotuneDecisions(): MatmulTileAutotuneDecision[] {
    return [...this._matmulTileDecisions.values()].map((decision) => ({
      ...decision,
      shape: { ...decision.shape },
    }));
  }

  private init(): NativeAddon {
    if (!this.initialized) {
      const info = initDevice();
      this._nativeDeviceInfo = info;
      this._f16Supported = info.f16Supported;
      this._deviceName = info.deviceName;
      this._vendorId = info.vendorId;
      this._hasAsyncTransfer = info.hasAsyncTransfer;
      this._coopMatSupported = info.coopMatSupported;
      this._coopMat2Supported = info.coopMat2Supported;
      this._coopM = info.coopMatM;
      this._coopN = info.coopMatN;
      this._coopK = info.coopMatK;
      this._kMulti = parseInt(process.env.HELIOS_COOP_K_MULTI ?? "4", 10);
      this._coopKTile = info.coopMatK * this._kMulti;
      this._hasPushDescriptors = info.hasPushDescriptors;

      // Cooperative matrix can be explicitly disabled for safety/debugging.
      const forceDisableCoop = process.env.HELIOS_DISABLE_COOP_MAT === "1";
      if (forceDisableCoop) {
        this._coopMatSupported = false;
        this._coopMat2Supported = false;
        this._coopM = 0;
        this._coopN = 0;
        this._coopK = 0;
        this._coopKTile = 0;
      }
      const vk = getNative();
      graph.attach(vk);
      if (ENABLE_WG_AUTOTUNE) {
        autoTuneWgSize(vk);
      }

      // Set up DGC (device-generated commands) if available
      if (info.hasDGC && process.env.HELIOS_DISABLE_DGC !== "1") {
        graph.setupDGC(vk);
      }

      this.initialized = true;
      return vk;
    }
    return getNative();
  }

  private resetCoopF16InputCache(releaseCached = true): void {
    if (releaseCached && this._coopF16InputCache.size > 0) {
      for (const casted of this._coopF16InputCache.values()) {
        releaseGpuBufferFor(casted);
      }
    }
    this._coopF16InputCache.clear();
    this._coopF16InputCacheLastFlushTimeline = graph.lastFlushTimeline;
  }

  private shouldUseMatmulReg2x2(batchSize: number): boolean {
    if (!ENABLE_MATMUL_REG2X2) return false;
    const info = this._nativeDeviceInfo;
    const supported =
      batchSize === 1 &&
      (info?.maxComputeWorkGroupInvocations ?? 0) >= 256 &&
      (info?.maxComputeWorkGroupSizeX ?? 0) >= 16 &&
      (info?.maxComputeWorkGroupSizeY ?? 0) >= 16 &&
      (info?.maxComputeSharedMemorySize ?? 0) >= 4096;
    if (!supported && !this._matmulReg2x2WarningEmitted) {
      this._matmulReg2x2WarningEmitted = true;
      console.warn(
        "[helios] HELIOS_MATMUL_REG2X2=1 requested, but this dispatch/device lacks " +
        "the current non-batched 256-invocation/4KiB-shared-memory contract; using tiled GEMM",
      );
    }
    return supported;
  }

  private shouldUseMatmulReg4x2(batchSize: number): boolean {
    if (!ENABLE_MATMUL_REG4X2) return false;
    const info = this._nativeDeviceInfo;
    const supported =
      batchSize === 1 &&
      (info?.maxComputeWorkGroupInvocations ?? 0) >= 128 &&
      (info?.maxComputeWorkGroupSizeX ?? 0) >= 16 &&
      (info?.maxComputeWorkGroupSizeY ?? 0) >= 8 &&
      (info?.maxComputeSharedMemorySize ?? 0) >= 4096;
    if (!supported && !this._matmulReg4x2WarningEmitted) {
      this._matmulReg4x2WarningEmitted = true;
      console.warn(
        "[helios] HELIOS_MATMUL_REG4X2=1 requested, but this dispatch/device lacks " +
        "the current non-batched 128-invocation/4KiB-shared-memory contract; using another GEMM path",
      );
    }
    return supported;
  }

  private shouldUseMatmulReg4x2K32(): boolean {
    const supported = (this._nativeDeviceInfo?.maxComputeSharedMemorySize ?? 0) >= 8192;
    if (!supported && !this._matmulReg4x2K32WarningEmitted) {
      this._matmulReg4x2K32WarningEmitted = true;
      console.warn(
        "[helios] HELIOS_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32=1 requested, but " +
        "the device exposes less than the candidate's 8KiB shared-memory contract; using R42C K16",
      );
    }
    return supported;
  }

  /**
   * Coop f16 input casts are temporary by design. Keeping them across many flushes
   * causes GPU handle growth when V8 GC lags behind training allocation rate.
   * Evicting all cached casts when a batch has completed makes lifetime explicit.
   */
  private evictCoopF16InputCacheForCompletedBatch(): void {
    const lastFlushTimeline = graph.lastFlushTimeline;
    if (lastFlushTimeline === this._coopF16InputCacheLastFlushTimeline) return;
    this.resetCoopF16InputCache(true);
  }

  private checkFallback(reason: string): void {
    if (process.env.HELIOS_NO_FALLBACK === "1" || process.env.HELIOS_NO_FALLBACK === "true") {
      throw new Error(`[helios] fallback forbidden by HELIOS_NO_FALLBACK: ${reason}`);
    }
  }

  /** Flush the compute graph — executes all pending GPU ops as a single batch. */
  flush(): void {
    graph.flush();
    this.evictCoopF16InputCacheForCompletedBatch();
  }

  /**
   * Flush GPU work AND wait for completion so all pending buffer releases
   * become reclaimable. Call between training steps when VRAM is tight.
   */
  syncGpu(): void {
    graph.flushAndWait();
    this.evictCoopF16InputCacheForCompletedBatch();
    const vk = getNative();
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
    waitTimelineTracked(vk, tv);
    this.evictCoopF16InputCacheForCompletedBatch();

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
        if (PROFILE_GRAPH_TRACE) traceBufferSizeByHandle.delete(handle);
        if (_liveAllocCount > 0) _liveAllocCount--;
      }
    }
    bufferPool.clear();
    bufferPoolEntries = 0;
    bufferPoolBytes = 0;

    // Plan-owned slots never enter either ordinary pool.
    if (staticSlotRuntime) staticSlotRuntime.destroy(vk);

    // Safe to destroy now — GPU sync above guarantees all work has completed
    processPendingDestroys(vk);
    this.resetCoopF16InputCache(false);
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
  gpuMemStats(): Record<string, number> {
    const nativeAllocatorStats = getNative().getAllocatorStats?.() ?? {};
    const stats = {
      bufferPoolEntries, bufferPoolBytes,
      outputPoolEntries, outputPoolBytes,
      deferredReleases: graph.deferredReleaseCount,
      pendingDestroys: pendingDestroys.length,
      outputPoolSizeClasses: outputPool.size,
      outputPoolSmallPerClass: OUTPUT_POOL_SMALL_PER_CLASS,
      outputPoolMediumPerClass: OUTPUT_POOL_MEDIUM_PER_CLASS,
      outputPoolLargePerClass: OUTPUT_POOL_LARGE_PER_CLASS,
      totalAllocs: _totalAllocCount,
      totalAllocMB: Math.round(_totalAllocBytes / 1024 / 1024),
      liveAllocs: _liveAllocCount,
      diagAllocsThisStep: _diagAllocsThisStep,
      diagReleasesThisStep: _diagReleasesThisStep,
      diagFrReleasesThisStep: _diagFrReleasesThisStep,
      // Running totals (never reset) — compute deltas externally
      flowNewCreates: _flowNewCreates,
      flowDestroys: _flowDestroys,
      flowOutputPoolHits: _flowOutputPoolHits,
      flowOutputPoolMisses: _flowOutputPoolMisses,
      flowOutputPoolReturns: _flowOutputPoolReturns,
      flowOutputPoolOverflows: _flowOutputPoolOverflows,
      flowBufferPoolHits: _flowBufferPoolHits,
      flowEnsureGpuHits: _flowEnsureGpuHits,
      flowEnsureGpuUploads: _flowEnsureGpuUploads,
      ...(staticSlotRuntime ? staticSlotRuntime.stats() : {
        staticSlotPlanEnabled: 0,
        staticSlotPlanActive: 0,
      }),
      ...nativeAllocatorStats,
    };
    // Only reset per-step diag counters (not flow totals)
    _diagAllocsThisStep = 0;
    _diagReleasesThisStep = 0;
    _diagFrReleasesThisStep = 0;
    return stats;
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
    const native = this._nativeDeviceInfo!;
    const computeSubgroupArithmeticSupported =
      (native.subgroupSupportedStages & 0x00000020) !== 0 &&
      (native.subgroupSupportedOperations & 0x00000004) !== 0;
    return {
      ...native,
      deviceName: this._deviceName,
      vendorId: this._vendorId,
      f16Supported: this._f16Supported,
      hasAsyncTransfer: this._hasAsyncTransfer,
      coopMatSupported: this._coopMatSupported,
      coopMat2Supported: this._coopMat2Supported,
      coopMatM: this._coopM,
      coopMatN: this._coopN,
      coopMatK: this._coopK,
      hasPushDescriptors: this._hasPushDescriptors,
      workgroupSize: WG_SIZE,
      minGpuSize: this._minGpuSize,
      computeSubgroupArithmeticSupported,
      nativeSubgroup32: native.subgroupSize === 32,
    };
  }

  /** Temporarily pause/resume cooperative matmul (f16 tensor cores).
   *  Use during backward pass to avoid f16 precision loss on large gradients. */
  set coopMatmulPaused(v: boolean) { this._coopMatPaused = v; }
  get coopMatmulPaused(): boolean { return this._coopMatPaused; }

  /** Opt-in row-parallel RMSNorm weight-gradient reduction. */
  set columnSumRowLanes(v: boolean) { this._columnSumRowLanes = v ? 8 : 0; }
  get columnSumRowLanes(): boolean { return this._columnSumRowLanes !== 0; }
  setColumnSumRowLanes(v: ColumnSumRowLanes): void { this._columnSumRowLanes = v; }
  getColumnSumRowLanes(): ColumnSumRowLanes { return this._columnSumRowLanes; }
  get lastColumnSumKernel(): string | null { return this._lastColumnSumKernel; }

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
      lastCoopKernel: this._lastCoopKernel,
      lastCoopShape: this._lastCoopShape ? { ...this._lastCoopShape } : null,
      shapeCounts: [...this._coopShapeCounts.values()]
        .map((entry) => ({ ...entry }))
        .sort((a, b) => a.key.localeCompare(b.key)),
    };
  }

  private recordCoopShape(
    kernel: string,
    M: number,
    N: number,
    K: number,
    batchSize: number,
    transposedA: boolean,
    transposedB: boolean,
  ): void {
    const key = `${transposedA ? "ta" : transposedB ? "tb" : "nn"}:${coopShapeKey(M, N, K)}:b${batchSize}`;
    const existing = this._coopShapeCounts.get(key);
    if (existing) {
      existing.count++;
      existing.kernel = kernel;
      return;
    }
    this._coopShapeCounts.set(key, {
      key,
      kernel,
      M,
      N,
      K,
      batchSize,
      transposedA,
      transposedB,
      count: 1,
    });
  }

  /**
   * Run a quick GPU smoke test: dispatches a small add kernel and verifies
   * the result. Returns throughput in GB/s. Throws if GPU compute fails.
   */
  smokeTest(): { verified: boolean; throughputGBps: number } {
    this.init();
    graph.flush();
    const tensors: TensorData[] = [];
    try {
      const size = 65536;
      const a = this.full([size], 1.0);
      const b = this.full([size], 2.0);
      const c = this.add(a, b);
      tensors.push(a, b, c);
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
      tensors.push(ba, bb);
      const start = performance.now();
      for (let i = 0; i < 10; i++) {
        const output = this.add(ba, bb);
        tensors.push(output);
      }
      graph.flush();
      // Force readback to include full round-trip
      const last = this.add(ba, bb);
      tensors.push(last);
      graph.flush();
      void (last.data as Float32Array)[0];
      const elapsed = performance.now() - start;
      const bytesPerOp = benchSize * 4 * 3; // 2 reads + 1 write
      const throughputGBps = (11 * bytesPerOp) / (elapsed * 1e6);

      return { verified, throughputGBps };
    } finally {
      // The smoke test runs before model initialization. Explicitly retire every
      // temporary tensor and destroy its pools so this preflight cannot consume
      // the small amount of VRAM headroom needed by the exact training shape.
      // FinalizationRegistry is intentionally not relied on for timely cleanup.
      for (let i = tensors.length - 1; i >= 0; i--) {
        this.releaseGpuTensor(tensors[i]);
      }
      this.purgeBufferPools();
    }
  }

  /** Get count of GPU ops dispatched this step (reset with resetStepOps). */
  get gpuOpsThisStep(): number { return graph.opsThisStep; }
  get gpuOpsTotal(): number { return graph.totalOpsRecorded; }
  resetStepOps(): void {
    const transition = staticSlotRuntime?.beginStep();
    if (transition?.activated) {
      // The warm-up path populated the ordinary pools. Retire it before
      // reserving plan-owned slots so the first active step measures the fixed
      // plan rather than an accidental union of both allocators.
      this.purgeBufferPools();
    }
    graph.resetStepStats();
  }
  finishStepOps(): void { staticSlotRuntime?.finishStep(graph.opsThisStep); }
  getGpuStepStats(): GpuStepStats { return graph.getStepStats(); }

  // ── GPU binary ops ──────────────────────────────────────────────────────

  private gpuBinaryOp(a: TensorData, b: TensorData, kernelName: string, forceScalar = false): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);
    const byteSize = size * 4;

    // Use vec4 kernel when size is aligned (4x throughput)
    const useVec4 = !forceScalar && (size & 3) === 0;
    // Coarsened vec4x2 (2 vec4 per thread) for large tensors — improves ILP.
    // Threshold: >= 1M elements (256KB). Below this, dispatch overhead dominates
    // and extra index math hurts. Only for simple binary ops (add/sub/mul).
    const VEC4X2_THRESHOLD = 1 << 20; // 1M elements
    const hasVec4x2 = useVec4 && size >= VEC4X2_THRESHOLD &&
      (kernelName === "add" || kernelName === "sub" || kernelName === "mul");
    const actualKernel = hasVec4x2 ? `${kernelName}_vec4x2`
      : useVec4 ? `${kernelName}_vec4` : kernelName;
    const pipeline = getPipeline(vk, actualKernel, 3);

    // Mark add pipelines as DGC-eligible (DGC BDA kernel does FAdd)
    if (kernelName === "add") {
      graph.markDGCEligible(pipeline);
    }

    // Reuse GPU buffers if inputs already on GPU (skips upload)
    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const region = acquireOutputRegion(vk, byteSize);

    // Dispatch WG count padded to buffer-rounded boundary for NVIDIA bandwidth alignment.
    // Push constant uses ACTUAL size for bounds check — extra WGs early-exit, avoiding
    // 2× wasted data transfer for small tensors (e.g. 512×1024: 2MB actual → 4MB padded).
    const effectiveSize = useVec4 ? size >> 2 : size;
    const roundedBytes = roundPoolSize(byteSize);
    const paddedSize = useVec4 ? (roundedBytes >> 4) : (roundedBytes >> 2);
    // For vec4x2, dispatch half the groups (each thread handles 2 vec4)
    const coarsenDiv = hasVec4x2 ? 2 : 1;
    const paddedGroups = Math.ceil(Math.max(effectiveSize, paddedSize) / coarsenDiv / WG_SIZE);
    const push = push2Memo(effectiveSize, 0);

    // Record to compute graph — deferred execution
    graph.record({
      kind: "binary",
      kernel: actualKernel,
      pipeline,
      inputBufs: [bufA, bufB],
      outputRegion: region,
      groups: [paddedGroups, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: a.shape,
      elementCount: size,  // actual element count for DGC scalar dispatch
    });

    return graphLazyTensor(vk, a.shape, region);
  }

  private gpuUnaryOp(a: TensorData, kernelName: string, scalar = 0): TensorData {
    const vk = this.init();
    const size = shapeSize(a.shape);
    const byteSize = size * 4;

    // Use vec4 kernel when size is aligned (4x throughput)
    const useVec4 = (size & 3) === 0;
    // Coarsened vec4x2 (2 vec4 per thread) for large tensors — improves ILP.
    const VEC4X2_THRESHOLD = 4 << 20; // 4M elements — lower thresholds regress on exp()-heavy kernels
    const hasVec4x2 = useVec4 && size >= VEC4X2_THRESHOLD &&
      (kernelName === "scale" || kernelName === "gelu" || kernelName === "relu" || kernelName === "silu");
    const actualKernel = hasVec4x2 ? `${kernelName}_vec4x2`
      : useVec4 ? `${kernelName}_vec4` : kernelName;
    const pipeline = getPipeline(vk, actualKernel, 2);

    // Reuse GPU buffer if input already on GPU
    const bufA = ensureGpu(vk, a);
    const region = acquireOutputRegion(vk, byteSize);

    // Dispatch WG count padded for NVIDIA bandwidth alignment; actual size for bounds check
    const effectiveSize = useVec4 ? size >> 2 : size;
    const roundedBytes = roundPoolSize(byteSize);
    const paddedSize = useVec4 ? (roundedBytes >> 4) : (roundedBytes >> 2);
    // For vec4x2, dispatch half the groups (each thread handles 2 vec4)
    const coarsenDiv = hasVec4x2 ? 2 : 1;
    const paddedGroups = Math.ceil(Math.max(effectiveSize, paddedSize) / coarsenDiv / WG_SIZE);
    const push = push2Memo(effectiveSize, scalar);
    const groups = paddedGroups;

    // Record to compute graph — deferred execution
    graph.record({
      kind: "unary",
      // Keep the profiler tied to the pipeline that actually executes.  The
      // previous logical-op label collapsed scalar, vec4, and vec4x2 variants
      // into one row, which made operation-level tuning impossible and could
      // falsely attribute a regression to the whole scale family.
      kernel: actualKernel,
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
    // A tiny tensor produced by the pending GPU graph must stay on-device.
    // Falling through to cpuUnary would access .data, force a submit+wait, and
    // split the static training graph merely to scale a scalar loss.
    if (shapeSize(a.shape) >= this._minGpuSize || (a.dtype === "f32" && gpuResidence.has(a))) {
      return this.gpuUnaryOp(a, "scale", s);
    }
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
    this.checkFallback("sumOfSquares");
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

    // For large inputs, use grid-stride kernel: fewer WGs, each thread loops
    // over many elements. Reduces 3 passes to 2 for ~8.5M elements.
    const STRIDE_THRESHOLD = 65536; // Use stride kernel above 64K elements
    const STRIDE_WGS = 256;        // Fixed number of workgroups for stride kernel

    if (totalSize >= STRIDE_THRESHOLD) {
      const stridePipeline = getPipeline(vk, "sum_sq_reduce_stride", 2);
      const sumPipeline = getPipeline(vk, "sum_reduce", 2);
      let inputBuf = ensureGpu(vk, a);

      // Pass 1: grid-stride sum-of-squares → STRIDE_WGS partial sums
      const numGroups = Math.min(STRIDE_WGS, Math.ceil(totalSize / WG_SIZE));
      const region1 = acquireOutputRegion(vk, numGroups * 4);
      const push1 = push2Memo(totalSize, 0);

      graph.record({
        kind: "reduce_sum",
        kernel: "sum_sq_reduce_stride",
        pipeline: stridePipeline,
        inputBufs: [],
        outputRegion: region1,
        groups: [numGroups, 1, 1],
        push: push1,
        pushSize: PUSH_SIZE,
        shape: [numGroups],
        allBufs: [inputBuf, region1.handle],
      });

      // Pass 2: reduce partial sums → 1 value (single WG handles up to 256)
      if (numGroups > 1) {
        const region2 = acquireOutputRegion(vk, 4);
        const push2 = push2Memo(numGroups, 0);

        graph.record({
          kind: "reduce_sum",
          kernel: "sum_reduce",
          pipeline: sumPipeline,
          inputBufs: [],
          outputRegion: region2,
          groups: [1, 1, 1],
          push: push2,
          pushSize: PUSH_SIZE,
          shape: [],
          allBufs: [region1.handle, region2.handle],
        });

        graph.deferRelease(region1);
        return graphLazyTensor(vk, [], region2);
      }

      return graphLazyTensor(vk, [], region1);
    }

    // Small inputs: use original multi-pass approach
    const sqPipeline = getPipeline(vk, "sum_sq_reduce", 2);
    const sumPipeline = getPipeline(vk, "sum_reduce", 2);

    let inputBuf = ensureGpu(vk, a);
    let remaining = totalSize;
    let finalRegion: OutputRegion | null = null;
    let isFirstPass = true;

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

      if (finalRegion) graph.deferRelease(finalRegion);

      inputBuf = region.handle;
      finalRegion = region;
      remaining = numGroups;
      isFirstPass = false;
    }

    if (!finalRegion) return a;

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
      this.checkFallback("checkFinite");
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
    this.checkFallback("castDtype");
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
    if (this._coopMatPaused) return false;
    const aOk = a.dtype === "f32" || a.dtype === "f16";
    const bOk = b.dtype === "f32" || b.dtype === "f16";
    return aOk && bOk;
  }

  private getCoopInputBuffer(vk: NativeAddon, td: TensorData): number {
    if (td.dtype === "f16") return this.ensureGpuF16(vk, td);
    if (td.dtype !== "f32") throw new Error(`Helios coop matmul only supports f32/f16 inputs (got ${td.dtype})`);

    // NOTE: Do NOT call evictCoopF16InputCacheForCompletedBatch() here!
    // castDtype() below calls graph.record() which may trigger an auto-flush.
    // If eviction runs on the NEXT getCoopInputBuffer call (which sees the new
    // timeline), it releases the f16 buffer that was just created — while the
    // caller still holds the handle for use in the matmul dispatch. This causes
    // the buffer to be recycled and overwritten before the matmul reads it.
    // Eviction is done at safe points: flush(), syncGpu(), purgeBufferPools().
    let casted = this._coopF16InputCache.get(td);
    if (!casted) {
      casted = this.castDtype(td, "f16");
      this._coopF16InputCache.set(td, casted);
    }
    return this.ensureGpuF16(vk, casted);
  }

  /**
   * Select the storage basis consumed by one cooperative dispatch.  A mixed
   * f16/f32 pair still uses the pre-cast path because both shader bindings
   * must have the same scalar width.  When both operands are f32 and graph
   * fusion is enabled, the shader narrows only the tiles it stages in shared
   * memory and no full-size cast tensor is materialized.
   */
  private coopUsesPrecastF16Inputs(a: TensorData, b: TensorData): boolean {
    return COOP_PRECAST_F16_INPUT || a.dtype === "f16" || b.dtype === "f16";
  }

  private getCoopF32InputBuffer(vk: NativeAddon, td: TensorData): number {
    if (td.dtype !== "f32") {
      throw new Error(`Helios fused-f32 cooperative matmul requires f32 inputs (got ${td.dtype})`);
    }
    return ensureGpu(vk, td);
  }

  // ── GPU softmax/layerNorm ───────────────────────────────────────────────

  private gpuSoftmax(a: TensorData): TensorData {
    const vk = this.init();
    const dim = a.shape[a.shape.length - 1];
    const numRows = shapeSize(a.shape) / dim;
    const byteSize = shapeSize(a.shape) * 4;

    // Use vec4 kernel when dim is divisible by 4
    // HELIOS_SOFTMAX_KERNEL: "online" (2-pass), "vec4" (3-pass vec4), "" (auto)
    if (dim % 4 === 0 && dim >= 16) {
      const dimVec4 = dim / 4;
      const softmaxEnv = process.env.HELIOS_SOFTMAX_KERNEL ?? "";
      // Register-resident unrolled softmax: compile-time unrolled SSA variables
      // (not an array) stay in GPU registers. Requires dimVec4 % wgSize == 0 and
      // itersPerThread <= 8. Eliminates the 2nd global memory read entirely.
      // Previous softmax_reg used a Function-scope array which spilled to L1 scratch
      // (25% regression). This version uses individual SSA values → true registers.
      const softmaxWgBase = dimVec4 <= 128 ? 32 : Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4)))));
      const itersPerThread = Math.ceil(dimVec4 / softmaxWgBase);
      const canUnroll = !softmaxEnv && dimVec4 % softmaxWgBase === 0 && itersPerThread <= 8 && softmaxWgBase <= 32;
      const kernelName = softmaxEnv === "online" ? "softmax_online"
        : softmaxEnv === "vec4" ? "softmax_vec4"
        : softmaxEnv === "reg" ? "softmax_reg"
        : canUnroll ? `softmax_reg_u${itersPerThread}`
        : "softmax_online";
      // For small dims (attention softmax, dim≤512), use wgSize=32: single subgroup
      // eliminates shared memory barriers and allows 48 WGs/SM vs 12 at wgSize=128.
      const softmaxWg = softmaxEnv === "reg" ? 32 : softmaxWgBase;

      // Persistent CTA: limit concurrent WGs so Phase 2 re-reads hit L2 cache.
      // Activates when total data exceeds ~32MB (L2 thrashing threshold on L4).
      const totalBytes = byteSize;
      const L2_TARGET_MB = parseInt(process.env.HELIOS_SOFTMAX_L2_MB ?? "10", 10);
      const L2_TARGET = L2_TARGET_MB * 1024 * 1024;
      const usePCTA = softmaxEnv === "" && totalBytes > L2_TARGET && dim >= 4096;

      if (usePCTA) {
        const rowBytes = dim * 4;
        const envWGs = parseInt(process.env.HELIOS_SOFTMAX_PCTA_WGS ?? "0", 10);
        const numWGs = envWGs > 0 ? envWGs : Math.min(numRows, Math.max(1, Math.floor(L2_TARGET / rowBytes)));
        const PCTA_PUSH_SIZE = 12; // 3 x f32: dimVec4, numRows, numWGs
        // Cap PCTA WG size at 128: fewer threads per WG → more work per thread →
        // better register utilization and less overhead per barrier within the WG.
        const pctaWg = Math.min(128, softmaxWg);
        const pctaPipeline = getPipeline(vk, "softmax_online_pcta", 2, PCTA_PUSH_SIZE, pctaWg);
        const bufA = ensureGpu(vk, a);
        const region = acquireOutputRegion(vk, byteSize);
        const push = new Float32Array([dimVec4, numRows, numWGs]);

        graph.record({
          kind: "softmax_online" as PendingOpKind,
          kernel: "softmax_online_pcta",
          pipeline: pctaPipeline,
          inputBufs: [bufA],
          outputRegion: region,
          groups: [numWGs, 1, 1],
          push,
          pushSize: PCTA_PUSH_SIZE,
          shape: a.shape,
        });

        return graphLazyTensor(vk, a.shape, region);
      }

      const pipeline = getPipeline(vk, kernelName, 2, PUSH_SIZE, softmaxWg);
      const bufA = ensureGpu(vk, a);
      const region = acquireOutputRegion(vk, byteSize);
      const push = push2Memo(dimVec4, numRows);

      graph.record({
        kind: "softmax_online" as PendingOpKind,
        kernel: kernelName,
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

    // Fallback to 3-pass scalar kernel
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

    const useVec4 = dim % 4 === 0 && dim >= 16;
    const dimVec4 = dim / 4;
    // Register-resident unrolled: compile-time unrolled SSA registers.
    // Eliminates Phase 2 re-read of X. wgSize=64 (2 subgroups) is optimal:
    // 4 vec4/thread gives good load pipelining while cross-subgroup reduce stays cheap.
    // wgSize=256 (8 subgroups, 1 vec4/thread) has poor latency hiding despite fewer iters.
    // wgSize=32 (1 subgroup, 8 vec4/thread) loses to wgSize=64 due to lower SM occupancy.
    const lnRegWgEnv = parseInt(process.env.HELIOS_LN_REG_WG ?? "0", 10);
    const lnRegWg = lnRegWgEnv > 0 ? lnRegWgEnv : Math.min(64, Math.max(32, 1 << Math.floor(Math.log2(dimVec4))));
    const lnItersPerThread = Math.ceil(dimVec4 / lnRegWg);
    const canUnrollLn = useVec4 && dimVec4 % lnRegWg === 0 && lnItersPerThread <= 16;
    // Tune wgSize for vec4 fallback: dimVec4>>1 → 128 for dim=1024.
    const lnWg = canUnrollLn
      ? lnRegWg
      : useVec4
        ? Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4 >> 1)))))
        : WG_SIZE;
    const kernelName = canUnrollLn ? `layernorm_reg_u${lnItersPerThread}` : useVec4 ? "layernorm_vec4" : "layernorm";
    const pipeline = getPipeline(vk, kernelName, 4, PUSH_SIZE, lnWg);
    const bufX = ensureGpu(vk, x);
    const bufW = ensureGpu(vk, weight);
    const bufB = ensureGpu(vk, bias);
    const region = acquireOutputRegion(vk, byteSize);

    const push = push2Memo(dim, eps);

    graph.record({
      kind: "layernorm",
      kernel: kernelName,
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

  private gpuRmsNorm(x: TensorData, weight: TensorData, eps: number): TensorData {
    const vk = this.init();
    const dim = x.shape[x.shape.length - 1];
    const numRows = shapeSize(x.shape) / dim;
    const byteSize = shapeSize(x.shape) * 4;

    // Scalar one-workgroup-per-row kernel (correct for any dim; Stage 2 tunes).
    const pipeline = getPipeline(vk, "rmsnorm", 3, PUSH_SIZE, WG_SIZE);
    const bufX = ensureGpu(vk, x);
    const bufW = ensureGpu(vk, weight);
    const region = acquireOutputRegion(vk, byteSize);
    const push = push2Memo(dim, eps);

    graph.record({
      kind: "layernorm",
      kernel: "rmsnorm",
      pipeline,
      inputBufs: [bufX, bufW],
      outputRegion: region,
      groups: [numRows, 1, 1],
      push,
      pushSize: PUSH_SIZE,
      shape: x.shape,
    });

    return graphLazyTensor(vk, x.shape, region);
  }

  private gpuRope(x: TensorData, cos: TensorData, sin: TensorData): TensorData {
    const vk = this.init();
    const D = x.shape[x.shape.length - 1];
    const T = x.shape[x.shape.length - 2];
    const half = D >> 1;
    const rows = shapeSize(x.shape) / D;       // B*H*T
    const totalPairs = rows * half;
    const byteSize = shapeSize(x.shape) * 4;

    const pipeline = getPipeline(vk, "rope", 4, 16, WG_SIZE);
    const bufX = ensureGpu(vk, x);
    const bufCos = ensureGpu(vk, cos);
    const bufSin = ensureGpu(vk, sin);
    const region = acquireOutputRegion(vk, byteSize);
    const groups = Math.ceil(totalPairs / WG_SIZE);
    const push = new Float32Array([totalPairs, half, D, T]);

    graph.record({
      kind: "unary",
      kernel: "rope",
      pipeline,
      inputBufs: [bufX, bufCos, bufSin],
      outputRegion: region,
      groups: [groups, 1, 1],
      push,
      pushSize: 16,
      shape: x.shape,
    });

    return graphLazyTensor(vk, x.shape, region);
  }

  // ── GPU backward ops ──────────────────────────────────────────────────

  geluBackward(input: TensorData, gradOutput: TensorData): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      return this.gpuBinaryOp(input, gradOutput, "gelu_backward");
    }
    // CPU fallback
    this.checkFallback("geluBackward");
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

  siluBackward(input: TensorData, gradOutput: TensorData): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      return this.gpuBinaryOp(input, gradOutput, "silu_backward");
    }
    // CPU fallback: silu'(x) = sigma(x) * (1 + x * (1 - sigma(x)))
    this.checkFallback("siluBackward");
    const src = input.data as Float32Array;
    const grad = gradOutput.data as Float32Array;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      const sigma = 1 / (1 + Math.exp(-x));
      out[i] = grad[i] * sigma * (1 + x * (1 - sigma));
    }
    return makeTensor(input.shape, input.dtype, out);
  }

  // ── Fused SiLU-Mul (SwiGLU) ───────────────────────────────────────────────

  siluMul(a: TensorData, b: TensorData): TensorData {
    const size = shapeSize(a.shape);
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const byteSize = size * 4;
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "silu_mul_vec4" : "silu_mul";
      const pipeline = getPipeline(vk, kernelName, 3);
      const bufA = ensureGpu(vk, a);
      const bufB = ensureGpu(vk, b);
      const region = acquireOutputRegion(vk, byteSize);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      const push = push2Memo(effectiveSize, 0);
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
    // CPU fallback
    this.checkFallback("siluMul");
    const aArr = a.data as Float32Array;
    const bArr = b.data as Float32Array;
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const x = aArr[i];
      out[i] = (x / (1 + Math.exp(-x))) * bArr[i];
    }
    return makeTensor(a.shape, a.dtype, out);
  }

  siluMulBackward(aData: TensorData, bData: TensorData, gradOutput: TensorData): TensorData[] {
    const size = shapeSize(aData.shape);
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const byteSize = size * 4;
      const useVec4 = (size & 3) === 0;
      const kernelName = useVec4 ? "silu_mul_backward_vec4" : "silu_mul_backward";
      const pipeline = getPipeline(vk, kernelName, 5);
      const bufDC = ensureGpu(vk, gradOutput);
      const bufA = ensureGpu(vk, aData);
      const bufB = ensureGpu(vk, bData);
      const regionDA = acquireOutputRegion(vk, byteSize);
      const regionDB = acquireOutputRegion(vk, byteSize);
      const effectiveSize = useVec4 ? size >> 2 : size;
      const groups = Math.ceil(effectiveSize / WG_SIZE);
      const push = push2Memo(effectiveSize, 0);
      graph.record({
        kind: "backward",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: regionDA,
        groups: [groups, 1, 1],
        push,
        pushSize: PUSH_SIZE,
        shape: aData.shape,
        allBufs: [bufDC, bufA, bufB, regionDA.handle, regionDB.handle],
        writeMask: (1 << 3) | (1 << 4),
      });
      return [
        graphLazyTensor(vk, aData.shape, regionDA),
        graphLazyTensor(vk, bData.shape, regionDB),
      ];
    }
    // CPU fallback
    this.checkFallback("siluMulBackward");
    const aArr = aData.data as Float32Array;
    const bArr = bData.data as Float32Array;
    const gArr = gradOutput.data as Float32Array;
    const da = new Float32Array(size);
    const db = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const x = aArr[i];
      const sig = 1 / (1 + Math.exp(-x));
      const siluDeriv = sig * (1 + x * (1 - sig));
      da[i] = gArr[i] * bArr[i] * siluDeriv;
      db[i] = gArr[i] * x * sig;
    }
    return [
      makeTensor(aData.shape, aData.dtype, da),
      makeTensor(bData.shape, bData.dtype, db),
    ];
  }

  reluBackward(input: TensorData, gradOutput: TensorData): TensorData {
    const size = shapeSize(input.shape);
    if (size >= this._minGpuSize && this.shapesEqual(input.shape, gradOutput.shape)) {
      return this.gpuBinaryOp(input, gradOutput, "relu_backward", true);
    }
    this.checkFallback("reluBackward");
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
    this.checkFallback("clampBackward");
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
    this.checkFallback("softCap");
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
    this.checkFallback("softCapBackward");
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
    this.checkFallback("residualDropoutAdd");
    const rArr = residual.data as Float32Array;
    const pArr = projected.data as Float32Array;
    const mArr = mask.data as Float32Array;
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) out[i] = rArr[i] + pArr[i] * mArr[i];
    return makeTensor(residual.shape, residual.dtype, out);
  }

  residualAddRmsNorm(
    residual: TensorData,
    projected: TensorData,
    weight: TensorData,
    eps: number,
  ): { residual: TensorData; normalized: TensorData } {
    if (DISABLE_RESIDUAL_ADD_RMSNORM) {
      const residualOut = this.add(residual, projected);
      return {
        residual: residualOut,
        normalized: this.rmsNorm(residualOut, weight, eps),
      };
    }
    const size = shapeSize(residual.shape);
    const dim = residual.shape[residual.shape.length - 1];
    if (
      size >= this._minGpuSize
      && this.shapesEqual(residual.shape, projected.shape)
      && shapeSize(weight.shape) === dim
    ) {
      const vk = this.init();
      const bufResidual = ensureGpu(vk, residual);
      const bufProjected = ensureGpu(vk, projected);
      const bufWeight = ensureGpu(vk, weight);
      const residualRegion = acquireOutputRegion(vk, size * 4);
      const normalizedRegion = acquireOutputRegion(vk, size * 4);
      const numRows = size / dim;
      const pipeline = getPipeline(vk, "residual_add_rmsnorm", 5, PUSH_SIZE, WG_SIZE);

      graph.record({
        kind: "layernorm",
        kernel: "residual_add_rmsnorm",
        pipeline,
        inputBufs: [],
        outputRegion: residualRegion,
        groups: [numRows, 1, 1],
        push: push2Memo(dim, eps),
        pushSize: PUSH_SIZE,
        shape: residual.shape,
        allBufs: [
          bufResidual,
          bufProjected,
          bufWeight,
          residualRegion.handle,
          normalizedRegion.handle,
        ],
        writeMask: 0b11000,
      });

      return {
        residual: graphLazyTensor(vk, residual.shape, residualRegion),
        normalized: graphLazyTensor(vk, residual.shape, normalizedRegion),
      };
    }

    // CPU reference path for small tensors and correctness tests.
    this.checkFallback("residualAddRmsNorm");
    if (!this.shapesEqual(residual.shape, projected.shape)) {
      throw new Error(
        `residualAddRmsNorm: residual [${residual.shape}] and projected [${projected.shape}] must match`,
      );
    }
    if (shapeSize(weight.shape) !== dim) {
      throw new Error(`residualAddRmsNorm: expected ${dim} weights, got ${shapeSize(weight.shape)}`);
    }
    const residualIn = residual.data as Float32Array;
    const projectedIn = projected.data as Float32Array;
    const weights = weight.data as Float32Array;
    const residualOut = new Float32Array(size);
    const normalizedOut = new Float32Array(size);
    const rows = size / dim;
    for (let row = 0; row < rows; row++) {
      const offset = row * dim;
      let sumSquares = 0;
      for (let col = 0; col < dim; col++) {
        const value = residualIn[offset + col] + projectedIn[offset + col];
        residualOut[offset + col] = value;
        sumSquares += value * value;
      }
      const invRms = 1 / Math.sqrt(sumSquares / dim + eps);
      for (let col = 0; col < dim; col++) {
        normalizedOut[offset + col] = residualOut[offset + col] * invRms * weights[col];
      }
    }
    return {
      residual: makeTensor(residual.shape, residual.dtype, residualOut),
      normalized: makeTensor(residual.shape, residual.dtype, normalizedOut),
    };
  }

  crossEntropyBackward(logits: TensorData, targets: TensorData, gradOutput: TensorData): TensorData {
    const vk = this.init();
    const [N, C] = logits.shape;
    const totalElements = N * C;
    const gradScalar = (gradOutput.data as Float32Array)[0];

    // Default large-vocabulary path: normalize logits and form dlogits in one
    // online vec4 kernel. The former path materialized an N*C probability
    // tensor, then read it in a second dispatch to write another N*C tensor.
    // Alpha's [10240,12288] training shape made each of those buffers ~480 MiB.
    // Keep the legacy route selectable for numerical bisection and for shapes
    // that cannot use aligned vec4 storage.
    const useFusedOnline = process.env.HELIOS_CE_BACKWARD_KERNEL !== "legacy"
      && totalElements >= this._minGpuSize
      && C >= 16
      && (C & 3) === 0;
    if (useFusedOnline) {
      const dimVec4 = C >>> 2;
      const ceWg = Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4)))));
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const pipeline = getPipeline(vk, "ce_backward_fused_online", 3, 3 * 4, ceWg);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const push = new Float32Array([dimVec4, N, gradScalar / N]);

      graph.record({
        kind: "backward",
        kernel: "ce_backward_fused_online",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [N, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: logits.shape,
        allBufs: [bufLogits, bufTargets, region.handle],
      });
      return graphLazyTensor(vk, logits.shape, region);
    }

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
    this.checkFallback("crossEntropyBackward");
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

  crossEntropyMaskedBackward(logits: TensorData, targets: TensorData, mask: TensorData, gradOutput: TensorData): TensorData {
    const vk = this.init();
    const [N, C] = logits.shape;
    const totalElements = N * C;
    const gradScalar = (gradOutput.data as Float32Array)[0];

    // sum(mask) — mask is CPU-origin (built by the SFT data loader); floor at 1
    // so an all-zero-mask micro-batch produces exactly-zero grads (no div-by-0).
    const maskArr = mask.data as Float32Array;
    let sumMask = 0;
    for (let i = 0; i < N; i++) sumMask += maskArr[i];
    const invDenom = gradScalar / Math.max(sumMask, 1);

    const useFusedOnline = process.env.HELIOS_CE_BACKWARD_KERNEL !== "legacy"
      && totalElements >= this._minGpuSize
      && C >= 16
      && (C & 3) === 0;
    if (useFusedOnline) {
      const dimVec4 = C >>> 2;
      const ceWg = Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4)))));
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const bufMask = ensureGpu(vk, mask);
      const pipeline = getPipeline(vk, "ce_masked_backward_fused_online", 4, 3 * 4, ceWg);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const push = new Float32Array([dimVec4, N, invDenom]);

      graph.record({
        kind: "backward",
        kernel: "ce_masked_backward_fused_online",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [N, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: logits.shape,
        allBufs: [bufLogits, bufTargets, bufMask, region.handle],
      });
      return graphLazyTensor(vk, logits.shape, region);
    }

    // softmax on GPU (stays lazy — no flush)
    const probs = this.softmax(logits, -1);

    if (totalElements >= this._minGpuSize) {
      const bufProbs = ensureGpu(vk, probs);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const bufMask = ensureGpu(vk, mask);

      const pipeline = getPipeline(vk, "ce_masked_backward", 4, 3 * 4);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(3);
      const pushU = new Uint32Array(push.buffer);
      push[0] = totalElements;  // f32 bounds-check value
      pushU[1] = C;             // u32 bits
      push[2] = invDenom;       // gradScalar / max(sum(mask),1)

      graph.record({
        kind: "backward",
        kernel: "ce_masked_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: logits.shape,
        allBufs: [bufProbs, bufTargets, bufMask, region.handle],
      });

      releaseGpuBufferFor(probs);
      return graphLazyTensor(vk, logits.shape, region);
    }

    // CPU fallback
    this.checkFallback("crossEntropyMaskedBackward");
    const probsArr = probs.data as Float32Array;
    const out = new Float32Array(totalElements);
    for (let i = 0; i < N; i++) {
      const m = maskArr[i];
      if (m === 0) continue; // exactly-zero grad for masked-out rows
      const off = i * C;
      const target = targets.data[i];
      const rowScale = invDenom * m;
      for (let j = 0; j < C; j++) {
        out[off + j] = (probsArr[off + j] - (j === target ? 1 : 0)) * rowScale;
      }
    }
    return makeTensor(logits.shape, logits.dtype, out);
  }

  crossEntropyUnlikelihoodMaskedBackward(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
    gradOutput: TensorData,
    epsilon: number,
  ): TensorData {
    if (!(epsilon > 0 && epsilon <= 1)) {
      throw new Error(`crossEntropyUnlikelihoodMaskedBackward epsilon must be in (0,1], got ${epsilon}`);
    }
    const vk = this.init();
    const [N, C] = logits.shape;
    const totalElements = N * C;
    const gradScalar = (gradOutput.data as Float32Array)[0];
    const maskArr = mask.data as Float32Array;
    let sumMask = 0;
    for (let i = 0; i < N; i++) sumMask += maskArr[i];
    const invDenom = gradScalar / Math.max(sumMask, 1);

    // The probability tensor remains GPU-resident and feeds the fused
    // unlikelihood derivative; no model-sized host readback occurs.
    const probs = this.softmax(logits, -1);

    if (totalElements >= this._minGpuSize) {
      const bufProbs = ensureGpu(vk, probs);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const bufMask = ensureGpu(vk, mask);
      const pipeline = getPipeline(vk, "ul_masked_backward", 4, 4 * 4);
      const region = acquireOutputRegion(vk, totalElements * 4);
      const groups = Math.ceil(totalElements / WG_SIZE);

      const push = new Float32Array(4);
      const pushU = new Uint32Array(push.buffer);
      push[0] = totalElements;
      pushU[1] = C;
      push[2] = invDenom;
      push[3] = epsilon;

      graph.record({
        kind: "backward",
        kernel: "ul_masked_backward",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize: 4 * 4,
        shape: logits.shape,
        allBufs: [bufProbs, bufTargets, bufMask, region.handle],
      });

      releaseGpuBufferFor(probs);
      return graphLazyTensor(vk, logits.shape, region);
    }

    this.checkFallback("crossEntropyUnlikelihoodMaskedBackward");
    const probsArr = probs.data as Float32Array;
    const out = new Float32Array(totalElements);
    for (let i = 0; i < N; i++) {
      const m = maskArr[i];
      if (m === 0) continue;
      const off = i * C;
      const target = targets.data[i];
      const pBad = probsArr[off + target];
      const rowScale = pBad / Math.max(1 - pBad, epsilon) * invDenom * m;
      for (let j = 0; j < C; j++) {
        out[off + j] = ((j === target ? 1 : 0) - probsArr[off + j]) * rowScale;
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

      const useDeterministicGather = outputSize * nIdx <= DETERMINISTIC_EMBEDDING_BACKWARD_MAX_WORK;
      const kernelName = useDeterministicGather
        ? "embedding_backward_deterministic"
        : "embedding_backward";

      // The scatter path accumulates into a zeroed output. The deterministic
      // gather writes every output element once and therefore needs no fill.
      const outByteSize = outputSize * 4;
      const region = acquireOutputRegion(vk, outByteSize);
      if (!useDeterministicGather) vk.fillBuffer(region.handle, outByteSize, 0);

      const pushSize = useDeterministicGather ? 3 * 4 : 2 * 4;
      const pipeline = getPipeline(vk, kernelName, 3, pushSize);
      const dispatchElements = useDeterministicGather ? outputSize : totalElements;
      const groups = Math.ceil(dispatchElements / WG_SIZE);

      const push = new Float32Array(useDeterministicGather ? 3 : 2);
      const pushU = new Uint32Array(push.buffer);
      push[0] = dispatchElements;  // float value for bounds check
      pushU[1] = dim;           // u32 bits — kernel bitcasts f32→u32
      if (useDeterministicGather) pushU[2] = nIdx;

      graph.record({
        kind: "backward",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [groups, 1, 1],
        push,
        pushSize,
        shape: [vocabSize, dim],
        allBufs: [bufIndices, bufGradOut, region.handle],
      });

      return graphLazyTensor(vk, [vocabSize, dim], region);
    }

    // CPU fallback
    this.checkFallback("embeddingBackward");
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
      const useVec4Bwd = dim % 4 === 0 && dim >= 16;
      const bwdKernel = useVec4Bwd ? "layernorm_backward_vec4" : "layernorm_backward";
      const dimVec4Bwd = dim / 4;
      const lnBwdWg = useVec4Bwd
        ? Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4Bwd >> 1)))))
        : WG_SIZE;
      const pipeline1 = getPipeline(vk, bwdKernel, 6, PUSH_SIZE, lnBwdWg);
      const push1 = push2Memo(dim, eps);
      graph.record({
        kind: "backward",
        kernel: bwdKernel,
        pipeline: pipeline1,
        inputBufs: [],
        outputRegion: dxRegion,
        groups: [numRows, 1, 1],
        push: push1,
        pushSize: 2 * 4,
        shape: x.shape,
        allBufs: [bufX, bufW, bufG, dxRegion.handle, dwPartialRegion.handle, dbPartialRegion.handle],
        writeMask: 0b111000, // dx, dw_partial, and db_partial are all written
      });

      // Fused dual column sum: reduce both partials in a single dispatch
      // Saves one dispatch + barrier vs two separate column_sum calls
      const pipeline2 = getPipeline(vk, "column_sum_dual", 4);
      const push2 = push2Memo(dim, numRows);
      const groups = Math.ceil(dim / WG_SIZE);
      graph.record({
        kind: "backward",
        kernel: "column_sum_dual",
        pipeline: pipeline2,
        inputBufs: [],
        outputRegion: dwRegion,
        groups: [groups, 1, 1],
        push: push2,
        pushSize: 2 * 4,
        shape: weight.shape,
        allBufs: [dwPartialRegion.handle, dbPartialRegion.handle, dwRegion.handle, dbRegion.handle],
        writeMask: 0b1100, // dw and db are both written
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
    this.checkFallback("layerNormBackward");
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

  rmsNormBackward(x: TensorData, weight: TensorData, gradOutput: TensorData, eps: number): { dx: TensorData; dw: TensorData } {
    const vk = this.init();
    const dim = x.shape[x.shape.length - 1];
    const numRows = shapeSize(x.shape) / dim;
    const xSize = shapeSize(x.shape);

    // GPU path — main backward kernel (dx + dw_partial) then a column sum for dw.
    if (xSize >= this._minGpuSize) {
      const bufX = ensureGpu(vk, x);
      const bufW = ensureGpu(vk, weight);
      const bufG = ensureGpu(vk, gradOutput);

      const dxRegion = acquireOutputRegion(vk, xSize * 4);
      const dwPartialRegion = acquireOutputRegion(vk, xSize * 4);
      const dwRegion = acquireOutputRegion(vk, dim * 4);

      const pipeline1 = getPipeline(vk, "rmsnorm_backward", 5, PUSH_SIZE, WG_SIZE);
      const push1 = push2Memo(dim, eps);
      graph.record({
        kind: "backward",
        kernel: "rmsnorm_backward",
        pipeline: pipeline1,
        inputBufs: [],
        outputRegion: dxRegion,
        groups: [numRows, 1, 1],
        push: push1,
        pushSize: 2 * 4,
        shape: x.shape,
        allBufs: [bufX, bufW, bufG, dxRegion.handle, dwPartialRegion.handle],
        writeMask: 0b11000, // dx and dw_partial are both written
      });

      const columnSumKernel = this._columnSumRowLanes !== 0
        ? `column_sum_row_lanes_${this._columnSumRowLanes}`
        : "column_sum";
      this._lastColumnSumKernel = columnSumKernel;
      const pipeline2 = getPipeline(vk, columnSumKernel, 2);
      const push2 = push2Memo(dim, numRows);
      const groups = this._columnSumRowLanes !== 0
        ? Math.ceil(dim / 32)
        : Math.ceil(dim / WG_SIZE);
      graph.record({
        kind: "backward",
        kernel: columnSumKernel,
        pipeline: pipeline2,
        inputBufs: [],
        outputRegion: dwRegion,
        groups: [groups, 1, 1],
        push: push2,
        pushSize: 2 * 4,
        shape: weight.shape,
        allBufs: [dwPartialRegion.handle, dwRegion.handle],
      });

      graph.deferRelease(dwPartialRegion);

      return {
        dx: graphLazyTensor(vk, x.shape, dxRegion),
        dw: graphLazyTensor(vk, weight.shape, dwRegion),
      };
    }

    // CPU fallback
    this.checkFallback("rmsNormBackward");
    const n = numRows;
    const xArr = x.data as Float32Array;
    const wArr = weight.data as Float32Array;
    const gArr = gradOutput.data as Float32Array;
    const dxOut = this.zeros(x.shape, x.dtype);
    const dwOut = this.zeros(weight.shape, weight.dtype);
    const dxArr = dxOut.data as Float32Array;
    const dwArr = dwOut.data as Float32Array;
    for (let i = 0; i < n; i++) {
      const off = i * dim;
      let ms = 0;
      for (let j = 0; j < dim; j++) ms += xArr[off + j] * xArr[off + j];
      ms /= dim;
      const r = 1 / Math.sqrt(ms + eps);
      const r3 = r * r * r;
      let S = 0;
      for (let j = 0; j < dim; j++) S += gArr[off + j] * wArr[j] * xArr[off + j];
      for (let j = 0; j < dim; j++) {
        dwArr[j] += gArr[off + j] * xArr[off + j] * r;
        dxArr[off + j] = r * gArr[off + j] * wArr[j] - xArr[off + j] * r3 * S / dim;
      }
    }
    return { dx: dxOut, dw: dwOut };
  }

  /**
   * Select the portable FP32 tiled GEMM kernel for one exact device/driver and
   * shape.  A tile-32 shader uses 1024 local invocations, so it is only a legal
   * candidate when Vulkan reports that capability.  In autotune mode both
   * legal candidates execute against the real resident inputs and the exact
   * output region already reserved for the graph dispatch.  This avoids a
   * transient duplicate allocation for the largest vocabulary projections.
   */
  private selectMatmulTile(
    vk: NativeAddon,
    kernel: string,
    bufA: number,
    bufB: number,
    outputRegion: OutputRegion,
    M: number,
    N: number,
    K: number,
    batchSize: number,
  ): MatmulTile {
    const info = this._nativeDeviceInfo;
    const key = [
      info?.vendorId ?? 0,
      info?.deviceId ?? 0,
      info?.driverVersion ?? 0,
      kernel,
      M,
      N,
      K,
      batchSize,
    ].join(":");
    const cached = this._matmulTileCache.get(key);
    if (cached !== undefined) return cached;

    const shape = { M, N, K, batchSize };
    const tile32Supported =
      (info?.maxComputeWorkGroupInvocations ?? 0) >= 1024 &&
      (info?.maxComputeWorkGroupSizeX ?? 0) >= 1024;
    const heuristicTile: MatmulTile = tile32Supported && M * N >= LARGE_TILE_THRESHOLD ? 32 : 16;

    const remember = (
      tile: MatmulTile,
      reason: MatmulTileAutotuneDecision["reason"],
      tile16GpuTimeUs: number | null = null,
      tile32GpuTimeUs: number | null = null,
      tile16SamplesUs: number[] = [],
      tile32SamplesUs: number[] = [],
    ): MatmulTile => {
      const decision: MatmulTileAutotuneDecision = {
        key,
        kernel,
        shape,
        tile,
        tile16GpuTimeUs,
        tile32GpuTimeUs,
        tile16SamplesUs,
        tile32SamplesUs,
        reason,
      };
      this._matmulTileCache.set(key, tile);
      this._matmulTileDecisions.set(key, decision);
      if (LOG_MATMUL_TILE_AUTOTUNE) {
        console.error(`[helios matmul tile] ${JSON.stringify(decision)}`);
      }
      return tile;
    };

    if (MATMUL_TILE_OVERRIDE !== null) {
      if (MATMUL_TILE_OVERRIDE === 32 && !tile32Supported) {
        console.warn(
          `[helios] tile-32 override is unsupported on ${info?.deviceName ?? "this device"}; using tile 16`,
        );
        return remember(16, "capability");
      }
      return remember(MATMUL_TILE_OVERRIDE, "override");
    }

    if (!ENABLE_MATMUL_TILE_AUTOTUNE || PROFILE_GPU_TIMESTAMPS) {
      return remember(heuristicTile, tile32Supported ? "heuristic" : "capability");
    }

    // Inputs may be graph-produced lazy tensors.  Resolve all earlier work
    // before standalone replay, otherwise the probe can race their writers.
    graph.flushAndWait();
    const push = push4Memo(M, N, K, 0);
    const candidates: MatmulTile[] = tile32Supported ? [16, 32] : [16];
    let tile16GpuTimeUs: number | null = null;
    let tile32GpuTimeUs: number | null = null;
    const tile16SamplesUs: number[] = [];
    const tile32SamplesUs: number[] = [];
    const probeErrors: string[] = [];
    const pipelines = new Map<MatmulTile, number>();

    // Compile and prewarm each legal candidate before collecting any evidence.
    // This avoids selecting the second candidate merely because it inherited
    // higher clocks and warm caches from the first candidate's cold launch.
    for (const tile of candidates) {
      const suffix = tile === 32 ? "_T32" : "";
      try {
        const pipeline = getPipeline(vk, `${kernel}${suffix}`, 3, 16);
        pipelines.set(tile, pipeline);
        vk.gpuTime(
          pipeline,
          [bufA, bufB, outputRegion.handle],
          Math.ceil(N / tile),
          Math.ceil(M / tile),
          batchSize,
          push,
          1,
          2,
        );
      } catch (error) {
        probeErrors.push(`tile${tile} warmup: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    // Measure in both orders.  The minimum of the two counterbalanced samples
    // estimates each kernel's warm steady-state path while discounting transient
    // clock ramp and interference from whichever candidate happened to run first.
    for (const order of [candidates, [...candidates].reverse()]) {
      for (const tile of order) {
        const pipeline = pipelines.get(tile);
        if (pipeline === undefined) continue;
        try {
          const elapsed = vk.gpuTime(
            pipeline,
            [bufA, bufB, outputRegion.handle],
            Math.ceil(N / tile),
            Math.ceil(M / tile),
            batchSize,
            push,
            3,
            1,
          );
          if (!Number.isFinite(elapsed) || elapsed <= 0) {
            throw new Error(`invalid GPU time ${elapsed}`);
          }
          if (tile === 16) tile16SamplesUs.push(elapsed);
          else tile32SamplesUs.push(elapsed);
        } catch (error) {
          probeErrors.push(`tile${tile} sample: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }

    if (tile16SamplesUs.length > 0) tile16GpuTimeUs = Math.min(...tile16SamplesUs);
    if (tile32SamplesUs.length > 0) tile32GpuTimeUs = Math.min(...tile32SamplesUs);

    let selected: MatmulTile | null = null;
    if (tile16GpuTimeUs !== null) selected = 16;
    if (
      tile32GpuTimeUs !== null &&
      (tile16GpuTimeUs === null || tile32GpuTimeUs < tile16GpuTimeUs)
    ) {
      selected = 32;
    }
    if (selected === null) {
      console.warn(
        `[helios] matmul tile probes failed for ${kernel} ${M}x${N}x${K}: ${probeErrors.join("; ")}`,
      );
      return remember(
        heuristicTile,
        "probe-fallback",
        tile16GpuTimeUs,
        tile32GpuTimeUs,
        tile16SamplesUs,
        tile32SamplesUs,
      );
    }
    return remember(
      selected,
      "measured",
      tile16GpuTimeUs,
      tile32GpuTimeUs,
      tile16SamplesUs,
      tile32SamplesUs,
    );
  }

  private gpuMatmul(a: TensorData, b: TensorData): TensorData {
    const vk = this.init();
    this._matmulDispatches++;
    const aNdim = a.shape.length, bNdim = b.shape.length;
    const M = a.shape[aNdim - 2], K = a.shape[aNdim - 1], N = b.shape[bNdim - 1];
    const aBatch = a.shape.slice(0, aNdim - 2);
    let batchSize = 1;
    for (const d of aBatch) batchSize *= d;
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b) && coopShapeIsEnabled("nn", M, N, K);

    // Try cooperative matrix (tensor core) path for aligned dimensions
    if (coopInputDtypesOk && this._coopMatSupported &&
        M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopKTile === 0) {
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

    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const kernel = batchSize === 1 ? "matmul" : "matmul_batched";
    const outBytes = batchSize * M * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    const useReg4x2 = this.shouldUseMatmulReg4x2(batchSize);
    const useReg2x2 = !useReg4x2 && this.shouldUseMatmulReg2x2(batchSize);
    const TILE = useReg4x2 || useReg2x2
      ? 32
      : this.selectMatmulTile(vk, kernel, bufA, bufB, region, M, N, K, batchSize);
    const suffix = useReg4x2 ? "_R42" : useReg2x2 ? "_R2" : TILE === 32 ? "_T32" : "";

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul${suffix}`, 3, 16);

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
    this.checkFallback("matmulTransposed");
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
    this.checkFallback("matmulTransposedA");
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
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b) && coopShapeIsEnabled("tb", M, N, K);

    // Try cooperative matrix (tensor core) path for aligned dimensions
    if (coopInputDtypesOk && this._coopMatSupported &&
        M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopKTile === 0) {
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

    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const kernel = batchSize === 1 ? "matmul_transposed" : "matmul_transposed_batched";
    const outBytes = batchSize * M * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    const useReg4x2 =
      ENABLE_MATMUL_REG4X2_TRANSPOSED_B && this.shouldUseMatmulReg4x2(batchSize);
    const useReg2x2 = !useReg4x2 && this.shouldUseMatmulReg2x2(batchSize);
    const TILE = useReg4x2 || useReg2x2
      ? 32
      : this.selectMatmulTile(vk, kernel, bufA, bufB, region, M, N, K, batchSize);
    const useReg4x2K32 =
      useReg4x2 &&
      ENABLE_MATMUL_TRANSPOSED_B_COALESCED &&
      ENABLE_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32 &&
      this.shouldUseMatmulReg4x2K32();
    const suffix = useReg4x2
      ? useReg4x2K32
        ? "_R42CK32"
        : ENABLE_MATMUL_TRANSPOSED_B_COALESCED ? "_R42C" : "_R42"
      : useReg2x2
        ? ENABLE_MATMUL_TRANSPOSED_B_COALESCED ? "_R2C" : "_R2"
        : TILE === 32 ? "_T32" : "";
    const gX = Math.ceil(N / TILE);
    const gY = Math.ceil(M / TILE);
    const push = push4Memo(M, N, K, 0);

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul_transposed${suffix}`, 3, 16);

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
    // matmul_transposed_a computes C[K,N] = A[M,K]^T × B[M,N].
    const outM = K;
    const loopK = M;
    const coopInputDtypesOk = this.canUseCoopMatmulDtypes(a, b) && coopShapeIsEnabled("ta", outM, N, loopK);

    // Direct cooperative path for aligned transposed-A GEMMs.
    if (coopInputDtypesOk && this._coopMatSupported &&
        outM % this._coopM === 0 && N % this._coopN === 0 && loopK % this._coopKTile === 0) {
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

    const bufA = ensureGpu(vk, a);
    const bufB = ensureGpu(vk, b);
    const kernel = batchSize === 1
      ? "matmul_transposed_a"
      : "matmul_transposed_a_batched";
    const outBytes = batchSize * outM * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    const useReg4x2 = this.shouldUseMatmulReg4x2(batchSize);
    const useReg2x2 = !useReg4x2 && this.shouldUseMatmulReg2x2(batchSize);
    const TILE = useReg4x2 || useReg2x2
      ? 32
      : this.selectMatmulTile(
          vk,
          kernel,
          bufA,
          bufB,
          region,
          outM,
          N,
          loopK,
          batchSize,
        );
    const suffix = useReg4x2
      ? ENABLE_MATMUL_TRANSPOSED_A_COALESCED ? "_R42C" : "_R42"
      : useReg2x2 ? "_R2" : TILE === 32 ? "_T32" : "";
    const gX = Math.ceil(N / TILE);
    const gY = Math.ceil(outM / TILE);
    const push = push4Memo(outM, N, loopK, 0);

    if (batchSize === 1) {
      const pipeline = getPipeline(vk, `matmul_transposed_a${suffix}`, 3, 16);

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
    let subgroupTilesX = 1;
    let subgroupTilesY = 1;
    let regTilesM = 1;
    let regTilesN = 1;
    if (M * N * K >= COOP_F16IN_S2X2_MIN_FLOPS) {
      if (COOP_SUBGROUP_TILES_X > 1 || COOP_SUBGROUP_TILES_Y > 1) {
        if (
          (M % (this._coopM * COOP_SUBGROUP_TILES_Y) === 0) &&
          (N % (this._coopN * COOP_SUBGROUP_TILES_X) === 0)
        ) {
          subgroupTilesX = COOP_SUBGROUP_TILES_X;
          subgroupTilesY = COOP_SUBGROUP_TILES_Y;
        }
      } else if (
        ENABLE_COOP_F16IN_S2X2 &&
        (M % (this._coopM * 2) === 0) &&
        (N % (this._coopN * 2) === 0)
      ) {
        subgroupTilesX = 2;
        subgroupTilesY = 2;
      }
      // Register tiling: each subgroup computes regTilesM × regTilesN tiles
      // Default to r4x4 when s2x2 is active; env var overrides.
      // Falls back to r2x2 when r4x4 doesn't fit the shape (e.g. N=2752).
      const wantM = COOP_REG_TILES_M > 1 ? COOP_REG_TILES_M : (subgroupTilesX === 2 && subgroupTilesY === 2 ? 4 : 1);
      const wantN = COOP_REG_TILES_N > 1 ? COOP_REG_TILES_N : (subgroupTilesX === 2 && subgroupTilesY === 2 ? 4 : 1);
      if (wantM > 1 || wantN > 1) {
        const effM = this._coopM * subgroupTilesY * wantM;
        const effN = this._coopN * subgroupTilesX * wantN;
        if (M % effM === 0 && N % effN === 0) {
          regTilesM = wantM;
          regTilesN = wantN;
        } else if (wantM >= 4 && wantN >= 4) {
          // Try asymmetric register tiling before r2x2 fallback.
          // r4x2 has 33% higher arithmetic intensity than r2x2 (more compute per
          // shared memory load), which helps compute-bound shapes like N=2752.
          const effM_r4x2 = this._coopM * subgroupTilesY * wantM;
          const effN_r4x2 = this._coopN * subgroupTilesX * Math.min(wantN, 2);
          const effM_r2x4 = this._coopM * subgroupTilesY * Math.min(wantM, 2);
          const effN_r2x4 = this._coopN * subgroupTilesX * wantN;
          if (M % effM_r4x2 === 0 && N % effN_r4x2 === 0) {
            regTilesM = wantM;
            regTilesN = Math.min(wantN, 2);
          } else if (M % effM_r2x4 === 0 && N % effN_r2x4 === 0) {
            regTilesM = Math.min(wantM, 2);
            regTilesN = wantN;
          } else {
            // Fallback to r2x2 when asymmetric doesn't fit either
            const fallM = Math.min(wantM, 2), fallN = Math.min(wantN, 2);
            const effM2 = this._coopM * subgroupTilesY * fallM;
            const effN2 = this._coopN * subgroupTilesX * fallN;
            if (M % effM2 === 0 && N % effN2 === 0) {
              regTilesM = fallM;
              regTilesN = fallN;
            }
          }
        }
      }
    }
    let gX = Math.ceil(N / (this._coopN * regTilesN * subgroupTilesX));
    let gY = Math.ceil(M / (this._coopM * regTilesM * subgroupTilesY));

    // Occupancy check: if r4x4 gives too few WGs (< 128), try r4x2 first for higher
    // arithmetic intensity than r2x2, then fall back to r2x2 if needed.
    // 128 WGs = ~2.2 WGs/SM on L4 (58 SMs). With r4x4 shmem ≈ 32KB, only 1 WG/SM
    // can be active, so < 128 WGs means many SMs sit idle during the second wave.
    const COOP_OCC_FALLBACK = parseInt(process.env.HELIOS_COOP_OCC_FALLBACK ?? "64", 10);
    if (gX * gY < COOP_OCC_FALLBACK && regTilesM >= 4 && regTilesN >= 4) {
      // Try r4x2: 33% higher arithmetic intensity than r2x2, 2× WGs vs r4x4.
      // Require ≥ 116 WGs (2 WGs/SM on L4). r4x2 at 64 WGs loses vs r2x2 at 128 WGs.
      const effM_42 = this._coopM * subgroupTilesY * 4;
      const effN_42 = this._coopN * subgroupTilesX * 2;
      if (M % effM_42 === 0 && N % effN_42 === 0) {
        const gX_42 = Math.ceil(N / effN_42);
        const gY_42 = Math.ceil(M / effM_42);
        if (gX_42 * gY_42 >= 116) {
          regTilesM = 4;
          regTilesN = 2;
          gX = gX_42;
          gY = gY_42;
        } else {
          // r4x2 still too few, fall back to r2x2
          const fallM = 2, fallN = 2;
          const effM2 = this._coopM * subgroupTilesY * fallM;
          const effN2 = this._coopN * subgroupTilesX * fallN;
          if (M % effM2 === 0 && N % effN2 === 0) {
            regTilesM = fallM;
            regTilesN = fallN;
            gX = Math.ceil(N / effN2);
            gY = Math.ceil(M / effM2);
          }
        }
      } else {
        const fallM = 2, fallN = 2;
        const effM2 = this._coopM * subgroupTilesY * fallM;
        const effN2 = this._coopN * subgroupTilesX * fallN;
        if (M % effM2 === 0 && N % effN2 === 0) {
          regTilesM = fallM;
          regTilesN = fallN;
          gX = Math.ceil(N / effN2);
          gY = Math.ceil(M / effM2);
        }
      }
    }

    // Adaptive kMulti: use kMulti=2 when occupancy is limited (< 512 WGs).
    // With kMulti=4 + s2x2: shmem = 32KB/WG → 1 WG/SM (48KB L4 limit).
    // With kMulti=2: shmem = 16KB/WG → 3 WGs/SM, better latency hiding.
    // For high-WG shapes (≥512), kMulti=4 is better: fewer barrier cycles.
    const baseWGs = gX * gY;
    const localKMulti = (baseWGs < COOP_KMULTI_ADAPT_MIN_WGS && this._kMulti >= 4) ? 2 : this._kMulti;
    const kTileK = this._coopK * localKMulti;

    // Split-K: partition K-reduction across multiple WGs for better SM occupancy.
    // Only for non-batched (batchSize=1) since wgId.z is repurposed as split index.
    let splitK = 1;
    if (COOP_SPLIT_K !== 0 && batchSize === 1) {
      if (COOP_SPLIT_K === -1) {
        // Auto: target ~512+ WGs for good SM occupancy on L4 (58 SMs × ~10 WGs/SM)
        const targetWGs = 512;
        splitK = Math.max(1, Math.ceil(targetWGs / baseWGs));
        // Cap: each split must process at least 1 K-tile
        const kTiles = Math.ceil(K / kTileK);
        splitK = Math.min(splitK, kTiles);
        // Reasonable upper bound
        splitK = Math.min(splitK, 16);
      } else {
        splitK = Math.max(1, COOP_SPLIT_K);
        const kTiles = Math.ceil(K / kTileK);
        splitK = Math.min(splitK, kTiles);
      }
    }

    const useSplitK = splitK > 1;

    const subgroupSuffix =
      (subgroupTilesX === 1 && subgroupTilesY === 1) ? "" : `_s${subgroupTilesX}x${subgroupTilesY}`;
    const regTileSuffix =
      (regTilesM === 1 && regTilesN === 1) ? "" : `_r${regTilesM}x${regTilesN}`;
    const dbSuffix = ENABLE_COOP_DOUBLE_BUF ? "_db" : "";
    const kmSuffix = localKMulti > 1 ? `_km${localKMulti}` : "";

    let variant: string;
    if (useSplitK) {
      variant = transposed ? "transposed" : "basic";
    } else {
      variant = transposed
        ? (batchSize > 1 ? "transposed_batched" : "transposed")
        : (batchSize > 1 ? "batched" : "basic");
    }

    const usePrecastF16Inputs = this.coopUsesPrecastF16Inputs(a, b);
    const inputStorageSuffix = usePrecastF16Inputs ? "_f16in" : "";
    const skPrefix = useSplitK ? "splitk_" : "";
    const kernelName =
      `matmul_coop_${skPrefix}${variant}_${this._coopM}_${this._coopN}_${this._coopK}${inputStorageSuffix}` +
      (ENABLE_COOP_F16_ACCUM ? "_f16acc" : "") +
      subgroupSuffix + regTileSuffix + dbSuffix + kmSuffix;
    this.recordCoopShape(kernelName, M, N, K, batchSize, false, transposed);
    if (DEBUG_COOP) {
      console.error(
        `[helios:coop] enter kernel=${kernelName} M=${M} N=${N} K=${K} batch=${batchSize} transposed=${transposed} splitK=${splitK}`,
      );
    }
    const pipeline = getPipeline(vk, kernelName, 3, 16);
    if (DEBUG_COOP) console.error(`[helios:coop] pipeline ready kernel=${kernelName} handle=${pipeline}`);
    const bufA = usePrecastF16Inputs
      ? this.getCoopInputBuffer(vk, a)
      : this.getCoopF32InputBuffer(vk, a);
    const bufB = usePrecastF16Inputs
      ? this.getCoopInputBuffer(vk, b)
      : this.getCoopF32InputBuffer(vk, b);

    if (useSplitK) {
      // Split-K path: dispatch matmul to temp buffer, then reduce.
      // NOTE: Benchmarked on L4 (commit b87a888 + split-K) — the separate reduction
      // pass adds ~1-2ms overhead per matmul (sum_axis achieves only ~30 GB/s effective
      // bandwidth). This makes split-K 2-3× SLOWER than baseline at all tested shapes
      // (1024-4096). Only useful if an atomic-add reduction is implemented instead.
      const kChunk = alignUp(Math.ceil(K / splitK), kTileK);
      const tempBytes = splitK * M * N * 4;
      const tempRegion = acquireOutputRegion(vk, tempBytes);
      const push = new Float32Array([M, N, K, kChunk]);

      graph.record({
        kind: "matmul",
        kernel: kernelName,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: tempRegion,
        groups: [gX, gY, splitK],
        push,
        pushSize: 16,
        shape: [splitK, M, N],
      });
      this._lastCoopKernel = kernelName;
      this._lastCoopShape = {
        M, N, K, batchSize,
        transposedA: false,
        transposedB: transposed,
      };
      if (DEBUG_COOP) console.error(`[helios:coop] recorded splitK kernel g=(${gX},${gY},${splitK}) kChunk=${kChunk}`);

      // Reduction: sum across split dimension using sum_axis kernel
      // Tensor is [splitK, M*N] → sum axis=0 → [M*N]
      const totalOutput = M * N;
      const reducePipeline = getPipeline(vk, "sum_axis", 2, 3 * 4);
      const outRegion = acquireOutputRegion(vk, totalOutput * 4);
      const reduceGroups = Math.ceil(totalOutput / WG_SIZE);
      const reducePush = new Float32Array(3);
      const reducePushU = new Uint32Array(reducePush.buffer);
      reducePushU[0] = totalOutput;
      reducePushU[1] = splitK;
      reducePushU[2] = totalOutput;  // innerSize = M*N

      graph.record({
        kind: "reduce_sum",
        kernel: "sum_axis",
        pipeline: reducePipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [reduceGroups, 1, 1],
        push: reducePush,
        pushSize: 3 * 4,
        shape: [...aBatch, M, N],
        allBufs: [tempRegion.handle, outRegion.handle],
      });
      if (DEBUG_COOP) console.error(`[helios:coop] recorded splitK reduce totalOutput=${totalOutput} splitK=${splitK}`);

      // Release temp buffer after this batch completes
      graph.deferRelease(tempRegion);

      return graphLazyTensor(vk, [...aBatch, M, N], outRegion);
    }

    // Standard path (no split-K)
    const outBytes = batchSize * M * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    // Pass gridX for GROUP_M swizzle (Triton-style); 0 disables.
    // Only enable when grid is large enough — small M shapes (gY < 8) don't have
    // enough tile rows for GROUP_M cycling to improve L2 reuse, and it can hurt.
    const swizzleGridX = (COOP_SWIZZLE_SIZE >= 2 && gY >= COOP_SWIZZLE_SIZE * 2) ? gX : 0;
    const push = push4Memo(M, N, K, swizzleGridX);

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
    this._lastCoopKernel = kernelName;
    this._lastCoopShape = {
      M, N, K, batchSize,
      transposedA: false,
      transposedB: transposed,
    };
    if (DEBUG_COOP) console.error(`[helios:coop] recorded kernel=${kernelName} g=(${gX},${gY},${batchSize})`);

    return graphLazyTensor(vk, [...aBatch, M, N], region);
  }

  private gpuMatmulCoopTransposedA(
    vk: NativeAddon, a: TensorData, b: TensorData,
    outM: number, N: number, loopK: number,
    aBatch: number[], batchSize: number,
  ): TensorData {
    let subgroupTilesX = 1;
    let subgroupTilesY = 1;
    let regTilesM = 1;
    let regTilesN = 1;
    if (outM * N * loopK >= COOP_F16IN_S2X2_MIN_FLOPS) {
      if (COOP_SUBGROUP_TILES_X > 1 || COOP_SUBGROUP_TILES_Y > 1) {
        if (
          (outM % (this._coopM * COOP_SUBGROUP_TILES_Y) === 0) &&
          (N % (this._coopN * COOP_SUBGROUP_TILES_X) === 0)
        ) {
          subgroupTilesX = COOP_SUBGROUP_TILES_X;
          subgroupTilesY = COOP_SUBGROUP_TILES_Y;
        }
      } else if (
        ENABLE_COOP_F16IN_S2X2 &&
        (outM % (this._coopM * 2) === 0) &&
        (N % (this._coopN * 2) === 0)
      ) {
        subgroupTilesX = 2;
        subgroupTilesY = 2;
      }
      const wantM = COOP_REG_TILES_M > 1 ? COOP_REG_TILES_M : (subgroupTilesX === 2 && subgroupTilesY === 2 ? 4 : 1);
      const wantN = COOP_REG_TILES_N > 1 ? COOP_REG_TILES_N : (subgroupTilesX === 2 && subgroupTilesY === 2 ? 4 : 1);
      if (wantM > 1 || wantN > 1) {
        const effM = this._coopM * subgroupTilesY * wantM;
        const effN = this._coopN * subgroupTilesX * wantN;
        if (outM % effM === 0 && N % effN === 0) {
          regTilesM = wantM;
          regTilesN = wantN;
        } else if (wantM >= 4 && wantN >= 4) {
          // Try asymmetric register tiling before r2x2 fallback
          const effM_r4x2 = this._coopM * subgroupTilesY * wantM;
          const effN_r4x2 = this._coopN * subgroupTilesX * Math.min(wantN, 2);
          const effM_r2x4 = this._coopM * subgroupTilesY * Math.min(wantM, 2);
          const effN_r2x4 = this._coopN * subgroupTilesX * wantN;
          if (outM % effM_r4x2 === 0 && N % effN_r4x2 === 0) {
            regTilesM = wantM;
            regTilesN = Math.min(wantN, 2);
          } else if (outM % effM_r2x4 === 0 && N % effN_r2x4 === 0) {
            regTilesM = Math.min(wantM, 2);
            regTilesN = wantN;
          } else {
            const fallM = Math.min(wantM, 2), fallN = Math.min(wantN, 2);
            const effM2 = this._coopM * subgroupTilesY * fallM;
            const effN2 = this._coopN * subgroupTilesX * fallN;
            if (outM % effM2 === 0 && N % effN2 === 0) {
              regTilesM = fallM;
              regTilesN = fallN;
            }
          }
        }
      }
    }
    let gX = Math.ceil(N / (this._coopN * regTilesN * subgroupTilesX));
    let gY = Math.ceil(outM / (this._coopM * regTilesM * subgroupTilesY));

    // Occupancy check: if r4x4 gives too few WGs (< 128), try r4x2 then r2x2.
    // Backward path threshold: 128 WGs minimum for r4x4. Higher than forward (64)
    // because transposed-A at very low WG counts (e.g. 64) regresses.
    const BWD_OCC_THRESHOLD = parseInt(process.env.HELIOS_COOP_BWD_OCC_FALLBACK ?? "128", 10);
    if (gX * gY < BWD_OCC_THRESHOLD && regTilesM >= 4 && regTilesN >= 4) {
      const effM_42 = this._coopM * subgroupTilesY * 4;
      const effN_42 = this._coopN * subgroupTilesX * 2;
      if (outM % effM_42 === 0 && N % effN_42 === 0) {
        const gX_42 = Math.ceil(N / effN_42);
        const gY_42 = Math.ceil(outM / effM_42);
        if (gX_42 * gY_42 >= 116) {
          regTilesM = 4;
          regTilesN = 2;
          gX = gX_42;
          gY = gY_42;
        } else {
          const fallM = 2, fallN = 2;
          const effM2 = this._coopM * subgroupTilesY * fallM;
          const effN2 = this._coopN * subgroupTilesX * fallN;
          if (outM % effM2 === 0 && N % effN2 === 0) {
            regTilesM = fallM;
            regTilesN = fallN;
            gX = Math.ceil(N / effN2);
            gY = Math.ceil(outM / effM2);
          }
        }
      } else {
        const fallM = 2, fallN = 2;
        const effM2 = this._coopM * subgroupTilesY * fallM;
        const effN2 = this._coopN * subgroupTilesX * fallN;
        if (outM % effM2 === 0 && N % effN2 === 0) {
          regTilesM = fallM;
          regTilesN = fallN;
          gX = Math.ceil(N / effN2);
          gY = Math.ceil(outM / effM2);
        }
      }
    }

    // Adaptive kMulti for transposed-A: use kMulti=2 when occupancy is limited.
    // With kMulti=4 + s2x2: shmem = 32KB/WG → 1 WG/SM (48KB L4 limit).
    // With kMulti=2: shmem = 16KB/WG → 3 WGs/SM, better latency hiding.
    // For high-WG shapes (≥512), kMulti=4 is better: fewer barrier cycles
    // outweigh the occupancy penalty since many waves keep SMs busy.
    const baseWGs = gX * gY;
    const localKMulti = (baseWGs < COOP_KMULTI_ADAPT_MIN_WGS && this._kMulti >= 4) ? 2 : this._kMulti;
    const kTileK = this._coopK * localKMulti;
    let splitK = 1;
    if (COOP_SPLIT_K !== 0 && batchSize === 1) {
      if (COOP_SPLIT_K === -1) {
        const targetWGs = 512;
        splitK = Math.max(1, Math.ceil(targetWGs / baseWGs));
        const kTiles = Math.ceil(loopK / kTileK);
        splitK = Math.min(splitK, kTiles);
        splitK = Math.min(splitK, 16);
      } else {
        splitK = Math.max(1, COOP_SPLIT_K);
        const kTiles = Math.ceil(loopK / kTileK);
        splitK = Math.min(splitK, kTiles);
      }
    }
    const useSplitK = splitK > 1;

    const variant = useSplitK ? "transposed_a" : (batchSize > 1 ? "transposed_a_batched" : "transposed_a");
    const subgroupSuffix =
      (subgroupTilesX === 1 && subgroupTilesY === 1) ? "" : `_s${subgroupTilesX}x${subgroupTilesY}`;
    const regTileSuffix =
      (regTilesM === 1 && regTilesN === 1) ? "" : `_r${regTilesM}x${regTilesN}`;
    const dbSuffix = ENABLE_COOP_DOUBLE_BUF ? "_db" : "";
    const kmSuffix = localKMulti > 1 ? `_km${localKMulti}` : "";
    const skPrefix = useSplitK ? "splitk_" : "";
    const usePrecastF16Inputs = this.coopUsesPrecastF16Inputs(a, b);
    const inputStorageSuffix = usePrecastF16Inputs ? "_f16in" : "";
    const kernelName =
      `matmul_coop_${skPrefix}${variant}_${this._coopM}_${this._coopN}_${this._coopK}${inputStorageSuffix}` +
      (ENABLE_COOP_F16_ACCUM ? "_f16acc" : "") +
      subgroupSuffix + regTileSuffix + dbSuffix + kmSuffix;
    this.recordCoopShape(kernelName, outM, N, loopK, batchSize, true, false);
    if (DEBUG_COOP) {
      console.error(
        `[helios:coop] enter kernel=${kernelName} outM=${outM} N=${N} K=${loopK} batch=${batchSize} transposedA=true splitK=${splitK} kMulti=${localKMulti}`,
      );
    }
    const pipeline = getPipeline(vk, kernelName, 3, 16);
    if (DEBUG_COOP) console.error(`[helios:coop] pipeline ready kernel=${kernelName} handle=${pipeline}`);
    const bufA = usePrecastF16Inputs
      ? this.getCoopInputBuffer(vk, a)
      : this.getCoopF32InputBuffer(vk, a);
    const bufB = usePrecastF16Inputs
      ? this.getCoopInputBuffer(vk, b)
      : this.getCoopF32InputBuffer(vk, b);

    if (useSplitK) {
      const kChunk = alignUp(Math.ceil(loopK / splitK), kTileK);
      const tempBytes = splitK * outM * N * 4;
      const tempRegion = acquireOutputRegion(vk, tempBytes);
      const push = new Float32Array([outM, N, loopK, kChunk]);

      graph.record({
        kind: "matmul",
        kernel: kernelName,
        pipeline,
        inputBufs: [bufA, bufB],
        outputRegion: tempRegion,
        groups: [gX, gY, splitK],
        push,
        pushSize: 16,
        shape: [splitK, outM, N],
      });
      this._lastCoopKernel = kernelName;
      this._lastCoopShape = {
        M: outM, N, K: loopK, batchSize,
        transposedA: true,
        transposedB: false,
      };
      if (DEBUG_COOP) console.error(`[helios:coop] recorded splitK transA kernel g=(${gX},${gY},${splitK}) kChunk=${kChunk}`);

      const totalOutput = outM * N;
      const reducePipeline = getPipeline(vk, "sum_axis", 2, 3 * 4);
      const outRegion = acquireOutputRegion(vk, totalOutput * 4);
      const reduceGroups = Math.ceil(totalOutput / WG_SIZE);
      const reducePush = new Float32Array(3);
      const reducePushU = new Uint32Array(reducePush.buffer);
      reducePushU[0] = totalOutput;
      reducePushU[1] = splitK;
      reducePushU[2] = totalOutput;

      graph.record({
        kind: "reduce_sum",
        kernel: "sum_axis",
        pipeline: reducePipeline,
        inputBufs: [],
        outputRegion: outRegion,
        groups: [reduceGroups, 1, 1],
        push: reducePush,
        pushSize: 3 * 4,
        shape: [...aBatch, outM, N],
        allBufs: [tempRegion.handle, outRegion.handle],
      });

      // Release temp buffer after this batch completes
      graph.deferRelease(tempRegion);

      return graphLazyTensor(vk, [...aBatch, outM, N], outRegion);
    }

    const outBytes = batchSize * outM * N * 4;
    const region = acquireOutputRegion(vk, outBytes);
    const swizzleGridX = (COOP_SWIZZLE_SIZE >= 2 && gY >= COOP_SWIZZLE_SIZE * 2) ? gX : 0;
    const push = push4Memo(outM, N, loopK, swizzleGridX);

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
    this._lastCoopKernel = kernelName;
    this._lastCoopShape = {
      M: outM, N, K: loopK, batchSize,
      transposedA: true,
      transposedB: false,
    };
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
    const alignedK = alignUp(K, this._coopKTile);
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
    const res = this.slice(paddedOut, [0, 0], [M, N]);
    releaseGpuBufferFor(paddedA);
    releaseGpuBufferFor(paddedB);
    releaseGpuBufferFor(paddedOut);
    return res;
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
    const alignedK = alignUp(K, this._coopKTile);
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
    const res = this.reshape(cropped, [...aBatch, M, N]);
    
    releaseGpuBufferFor(paddedA);
    releaseGpuBufferFor(paddedB);
    releaseGpuBufferFor(paddedOut);
    
    return res;
  }

  private canUseCoopWithOptionalPadding(M: number, N: number, K: number): boolean {
    if (!this._coopMatSupported) return false;
    if (M % this._coopM === 0 && N % this._coopN === 0 && K % this._coopKTile === 0) return true;
    const alignedM = alignUp(M, this._coopM);
    const alignedN = alignUp(N, this._coopN);
    const alignedK = alignUp(K, this._coopKTile);
    const baseElems = M * K + K * N + M * N;
    const paddedElems = alignedM * alignedK + alignedK * alignedN + alignedM * alignedN;
    const overhead = (paddedElems - baseElems) / baseElems;
    return overhead <= COOP_PAD_MAX_OVERHEAD;
  }

  // ── In-place add: A += B on GPU ────────────────────────────────────────────

  addInplace(a: TensorData, b: TensorData): void {
    const size = shapeSize(a.shape);
    // Preserve device residency for scalar loss accumulation. The historical
    // size-only gate made the trainer's nominally GPU-side accumulation read
    // every microbatch loss back to the host before backward.
    const residentScalars = a.dtype === "f32" && b.dtype === "f32"
      && gpuResidence.has(a) && gpuResidence.has(b);
    if (size >= this._minGpuSize || residentScalars) {
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
        this.checkFallback("addInplace");
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
      this.checkFallback("scaleInplace");
      if (a.dtype === "f32") {
      const arr = a.data as Float32Array;
      for (let i = 0; i < size; i++) arr[i] *= scalar;
      this._coopF16InputCache.delete(a);
      return;
      }    // Generic fallback for non-f32 dtypes: compute out-of-place then copy back.
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

    const canUseCoop2Forward =
      this._flashFwdPreferCoop2 &&
      this._coopMat2Supported &&
      Q.dtype === "f32" &&
      K.dtype === "f32" &&
      V.dtype === "f32" &&
      D === 64 &&
      K.shape[2] === D &&
      V.shape[2] === D;
    let fallbackReason: string | undefined;
    if (canUseCoop2Forward) {
      if (this._flashFwdCoop2Strict) {
        return this.flashAttentionCoop2(Q, K, V, T, scale, softCap);
      }
      if (this._flashFwdCoop2Ready === true) {
        return this.flashAttentionCoop2(Q, K, V, T, scale, softCap);
      }
      if (this._flashFwdCoop2Ready === null) {
        try {
          const result = this.flashAttentionCoop2(Q, K, V, T, scale, softCap);
          this._flashFwdCoop2Ready = true;
          return result;
        } catch (err) {
          const fallbackMessage = err instanceof Error ? err.message : String(err);
          // Do not permanently disable coop2 on transient runtime failures.
          // Retry on subsequent calls for OOM/device-loss style errors.
          const msgLower = fallbackMessage.toLowerCase();
          const transientFailure =
            msgLower.includes("oom") ||
            msgLower.includes("out of memory") ||
            msgLower.includes("vkmallocatememory failed") ||
            msgLower.includes("device lost") ||
            msgLower.includes("temporarily unavailable");
          this._flashFwdCoop2Ready = transientFailure ? null : false;
          fallbackReason = transientFailure
            ? `coop2_error_transient:${fallbackMessage}`
            : `coop2_error:${fallbackMessage}`;
          if (DEBUG_COOP) {
            console.warn(`[helios] flashAttention coop2 fallback -> scalar: ${fallbackMessage}`);
          }
        }
      } else {
        fallbackReason = "coop2_error_cached";
      }
      if (!fallbackReason) {
        // Non-strict mode only reaches here when coop2 is disabled due cached failure.
        fallbackReason = "coop2_error_cached";
      }
    } else {
      fallbackReason = "coop2_ineligible";
    }

    // Br=32, Bc=16: halved Bc reduces shared memory (sK+sV: 16KB→8KB), doubling
    // occupancy from 3 to 6 WGs/SM on L4. V1 (runtime loop) avoids the SPIR-V code
    // bloat that V2's compile-time j-unrolling causes (40% regression from icache pressure).
    const Br = safeFlashTile(T, 32);
    const Bc = safeFlashTile(T, Math.min(16, Br));
    const scSuffix = softCap > 0 ? "_sc" : "";
    const kernelName = `flash_attn_fwd${scSuffix}_${Br}_${Bc}_${D}`;
    const pipelineLookup = getPipelineLookup(vk, kernelName, 5, 16);
    const pipeline = pipelineLookup.handle;

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);

    // Output: O [BH, T, D]
    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    // LSE: [BH, T]
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, softCap, 0);
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
      writeMask: 0b11000, // O and LSE are both written
    });

    this.setLastFlashDispatchDebug({
      requestedOp: "flashAttention",
      executedPath: "scalar",
      mode: "full",
      softCap,
      BH,
      T,
      D,
      kernelName,
      pipelineKey: pipelineLookup.key,
      pipelineHandle: pipeline,
      pipelineCreated: pipelineLookup.created,
      fallbackReason,
    });

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { output, lse };
  }

  flashAttentionCoop(Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number): { output: TensorData; lse: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 16, Bc = 16;
    const kernelName = `flash_attn_coop_fwd_${Br}_${Bc}_${D}`;
    const pipeline = getPipeline(vk, kernelName, 5, 16);

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);

    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, 0, 0);
    const gX = Math.ceil(T / Br);

    graph.record({
      kind: "matmul",
      kernel: kernelName,
      pipeline,
      inputBufs: [],
      outputRegion: oRegion,
      groups: [gX, BH, 1],
      push,
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
      writeMask: 0b11000, // O and LSE are both written
    });

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { output, lse };
  }

  private resolveFlashCoop2Pipeline(
    vk: NativeAddon,
    baseKernelName: string,
  ): {
    kernelName: string;
    pipeline: number;
    pipelineKey: string;
    pipelineCreated: boolean;
    scope: FlashCoop2ScopeTag;
    fallbackReason?: string;
  } {
    const preferredScope = this._flashCoop2ScopeResolved ?? parseFlashCoop2ScopeTag();
    const preferredName = `${baseKernelName}_${preferredScope}`;
    try {
      const lookup = getPipelineLookup(vk, preferredName, 5, 16);
      this._flashCoop2ScopeResolved = preferredScope;
      return {
        kernelName: preferredName,
        pipeline: lookup.handle,
        pipelineKey: lookup.key,
        pipelineCreated: lookup.created,
        scope: preferredScope,
      };
    } catch (err) {
      if (preferredScope === "wg") {
        const fallbackName = `${baseKernelName}_sg`;
        const lookup = getPipelineLookup(vk, fallbackName, 5, 16);
        this._flashCoop2ScopeResolved = "sg";
        return {
          kernelName: fallbackName,
          pipeline: lookup.handle,
          pipelineKey: lookup.key,
          pipelineCreated: lookup.created,
          scope: "sg",
          fallbackReason: err instanceof Error ? `scope_wg_unavailable:${err.message}` : "scope_wg_unavailable",
        };
      }
      throw err;
    }
  }

  flashAttentionCoop2(Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap = 0): { output: TensorData; lse: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 16;
    const qTilesPerWG = this._flashCoop2QTiles;
    const Bc = this._flashCoop2BlockCols;
    const localSize = this._flashCoop2LocalSize;
    const useF16Input = this._f16Supported && this._flashCoop2PreferF16Input;
    const skipLseWrite = this._flashCoop2SkipLseWrite;
    const scSuffix = softCap === 30 ? "_sc30" : (softCap > 0 ? "_sc" : "");
    const in16Suffix = useF16Input ? "_in16" : "";
    const qtSuffix = qTilesPerWG > 1 ? `_qt${qTilesPerWG}` : "";
    const noLseSuffix = skipLseWrite ? "_nolse" : "";
    const dbSuffix = this._flashCoop2DoubleBuf ? "_db" : "";
    const baseKernelName = `flash_attn_coop2_fwd${scSuffix}${in16Suffix}${noLseSuffix}_${Br}_${Bc}_${D}${qtSuffix}_ls${localSize}${dbSuffix}`;
    const { kernelName, pipeline, pipelineKey, pipelineCreated, scope, fallbackReason } =
      this.resolveFlashCoop2Pipeline(vk, baseKernelName);

    const bufQ = useF16Input ? this.getCoopInputBuffer(vk, Q) : ensureGpu(vk, Q);
    const bufK = useF16Input ? this.getCoopInputBuffer(vk, K) : ensureGpu(vk, K);
    const bufV = useF16Input ? this.getCoopInputBuffer(vk, V) : ensureGpu(vk, V);

    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, softCap, 0);
    const gX = Math.ceil(T / (Br * qTilesPerWG));

    graph.record({
      kind: "matmul",
      kernel: kernelName,
      pipeline,
      inputBufs: [],
      outputRegion: oRegion,
      groups: [gX, BH, 1],
      push,
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
      writeMask: 0b11000, // O and LSE are both written
    });

    this.setLastFlashDispatchDebug({
      requestedOp: "flashAttentionCoop2",
      executedPath: "coop2",
      mode: "full",
      softCap,
      BH,
      T,
      D,
      kernelName,
      pipelineKey,
      pipelineHandle: pipeline,
      pipelineCreated,
      scope,
      fallbackReason,
    });

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { output, lse };
  }

  flashAttentionCoop2Probe(
    mode: "qk" | "qk_mask" | "qk_softmax" | "pv" | "kv_only" | "kv_synth" | "per_elem_only",
    Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap = 0,
  ): { output: TensorData; lse: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 16, Bc = 16;
    const qTilesPerWG = this._flashCoop2QTiles;
    const localSize = this._flashCoop2LocalSize;
    const useF16Input = this._f16Supported && this._flashCoop2PreferF16Input;
    const scSuffix = softCap === 30 ? "_sc30" : (softCap > 0 ? "_sc" : "");
    const in16Suffix = useF16Input ? "_in16" : "";
    const qtSuffix = qTilesPerWG > 1 ? `_qt${qTilesPerWG}` : "";
    const baseKernelName = `flash_attn_coop2_probe_${mode}_fwd${scSuffix}${in16Suffix}_${Br}_${Bc}_${D}${qtSuffix}_ls${localSize}`;
    const { kernelName, pipeline, pipelineKey, pipelineCreated, scope, fallbackReason } =
      this.resolveFlashCoop2Pipeline(vk, baseKernelName);

    const bufQ = useF16Input ? this.getCoopInputBuffer(vk, Q) : ensureGpu(vk, Q);
    const bufK = useF16Input ? this.getCoopInputBuffer(vk, K) : ensureGpu(vk, K);
    const bufV = useF16Input ? this.getCoopInputBuffer(vk, V) : ensureGpu(vk, V);

    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, softCap, 0);
    const gX = Math.ceil(T / (Br * qTilesPerWG));

    graph.record({
      kind: "matmul",
      kernel: kernelName,
      pipeline,
      inputBufs: [],
      outputRegion: oRegion,
      groups: [gX, BH, 1],
      push,
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
      writeMask: 0b11000, // O and LSE are both written
    });

    this.setLastFlashDispatchDebug({
      requestedOp: "flashAttentionCoop2Probe",
      executedPath: "coop2",
      mode,
      softCap,
      BH,
      T,
      D,
      kernelName,
      pipelineKey,
      pipelineHandle: pipeline,
      pipelineCreated,
      scope,
      fallbackReason,
    });

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { output, lse };
  }

  /**
   * GPU-timestamp-measured flash attention dispatch (coop2 path).
   * Uses vkCmdWriteTimestamp for true GPU kernel time.
   * Optional iters/warmup for steady-state measurement.
   * Returns per-iteration elapsed time in microseconds.
   */
  flashAttentionCoop2GpuTime(
    mode: "full" | "qk" | "qk_mask" | "qk_softmax" | "pv" | "kv_only" | "kv_synth" | "per_elem_only",
    Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap = 0,
    iters = 1, warmup = 0,
  ): { gpuTimeUs: number; output: TensorData; lse: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 16, Bc = 16;
    const qTilesPerWG = this._flashCoop2QTiles;
    const localSize = this._flashCoop2LocalSize;
    const useF16Input = this._f16Supported && this._flashCoop2PreferF16Input;
    const scSuffix = softCap === 30 ? "_sc30" : (softCap > 0 ? "_sc" : "");
    const in16Suffix = useF16Input ? "_in16" : "";
    const qtSuffix = qTilesPerWG > 1 ? `_qt${qTilesPerWG}` : "";

    let baseKernelName: string;
    if (mode === "full") {
      const noLseSuffix = this._flashCoop2SkipLseWrite ? "_nolse" : "";
      const dbSuffix = this._flashCoop2DoubleBuf ? "_db" : "";
      baseKernelName = `flash_attn_coop2_fwd${scSuffix}${in16Suffix}${noLseSuffix}_${Br}_${Bc}_${D}${qtSuffix}_ls${localSize}${dbSuffix}`;
    } else {
      baseKernelName = `flash_attn_coop2_probe_${mode}_fwd${scSuffix}${in16Suffix}_${Br}_${Bc}_${D}${qtSuffix}_ls${localSize}`;
    }
    const { pipeline } = this.resolveFlashCoop2Pipeline(vk, baseKernelName);

    // Flush any pending graph work and wait for completion before gpuTime
    const tv = graph.flush();
    if (tv > 0) vk.waitTimeline(tv);

    const bufQ = useF16Input ? this.getCoopInputBuffer(vk, Q) : ensureGpu(vk, Q);
    const bufK = useF16Input ? this.getCoopInputBuffer(vk, K) : ensureGpu(vk, K);
    const bufV = useF16Input ? this.getCoopInputBuffer(vk, V) : ensureGpu(vk, V);

    // Flush any graph work recorded by getCoopInputBuffer (e.g. F32→F16 cast)
    const tv2 = graph.flush();
    if (tv2 > 0) vk.waitTimeline(tv2);

    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, softCap, 0);
    const gX = Math.ceil(T / (Br * qTilesPerWG));

    const gpuTimeUs = vk.gpuTime(
      pipeline,
      [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
      gX, BH, 1,
      push,
      iters, warmup,
    );

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { gpuTimeUs, output, lse };
  }

  /**
   * GPU-timestamp-measured flash attention dispatch (scalar path).
   * Uses vkCmdWriteTimestamp for true GPU kernel time.
   * Optional iters/warmup for steady-state measurement.
   * Returns per-iteration elapsed time in microseconds.
   */
  flashAttentionGpuTime(
    Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap = 0,
    iters = 1, warmup = 0,
  ): { gpuTimeUs: number; output: TensorData; lse: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const Br = 32, Bc = 16;
    const scSuffix = softCap > 0 ? "_sc" : "";
    const kernelName = `flash_attn_fwd${scSuffix}_${Br}_${Bc}_${D}`;
    const pipeline = getPipeline(vk, kernelName, 5, 16);

    // Flush any pending graph work and wait for completion before gpuTime
    const tv = graph.flush();
    if (tv > 0) vk.waitTimeline(tv);

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);

    const oBytes = BH * T * D * 4;
    const oRegion = acquireOutputRegion(vk, oBytes);
    const lseBytes = BH * T * 4;
    const lseRegion = acquireOutputRegion(vk, lseBytes);

    const push = push4Memo(T, scale, softCap, 0);
    const gX = Math.ceil(T / Br);

    const gpuTimeUs = vk.gpuTime(
      pipeline,
      [bufQ, bufK, bufV, oRegion.handle, lseRegion.handle],
      gX, BH, 1,
      push,
      iters, warmup,
    );

    const output = graphLazyTensor(vk, [BH, T, D], oRegion);
    const lse = graphLazyTensor(vk, [BH, T], lseRegion);
    return { gpuTimeUs, output, lse };
  }

  flashAttentionBackward(Q: TensorData, K: TensorData, V: TensorData,
    O: TensorData, dO: TensorData, lse: TensorData,
    T: number, scale: number, softCap: number): { dQ: TensorData; dK: TensorData; dV: TensorData } {
    const vk = this.init();
    const BH = Q.shape[0];
    const D = Q.shape[2];
    const requestedBr = parseInt(process.env.HELIOS_FLASH_BWD_BR ?? "32", 10);
    const requestedBrDKV = parseInt(process.env.HELIOS_FLASH_BWD_BR_DKV ?? "32", 10);
    const requestedBcDQ = parseInt(process.env.HELIOS_FLASH_BWD_BC_DQ ?? "16", 10);
    const requestedBcDKV = parseInt(process.env.HELIOS_FLASH_BWD_BC_DKV ?? "32", 10);
    const Br = safeFlashTile(T, requestedBr);
    const BrDKV = safeFlashTile(T, requestedBrDKV);
    const BcDQ = safeFlashTile(T, Math.min(requestedBcDQ, Br));
    const BcDKV = safeFlashTile(T, requestedBcDKV);
    const scSuffix = softCap > 0 ? "_sc" : "";

    const bufQ = ensureGpu(vk, Q);
    const bufK = ensureGpu(vk, K);
    const bufV = ensureGpu(vk, V);
    const bufO = ensureGpu(vk, O);
    const bufDO = ensureGpu(vk, dO);
    const bufLSE = ensureGpu(vk, lse);

    // Step 1: dQ kernel computes D_precomp inline (saves 2 dispatch calls ~60µs)
    // D_precomp: [BH, T] — Di = dot(dO[i,:], O[i,:]), computed inside dQ kernel
    const dPreBytes = BH * T * 4;
    const dPreRegion = acquireOutputRegion(vk, dPreBytes);

    // dQ kernel: bindings [Q, K, V, dO, O, LSE, Dpre_out, dQ_out]
    const dqKernel = `flash_attn_bwd_dq${scSuffix}_${Br}_${BcDQ}_${D}`;
    const dqPipeline = getPipeline(vk, dqKernel, 8, 16);
    const dqBytes = BH * T * D * 4;
    const dqRegion = acquireOutputRegion(vk, dqBytes);
    const push = push4Memo(T, scale, softCap, 0);

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
      allBufs: [bufQ, bufK, bufV, bufDO, bufO, bufLSE, dPreRegion.handle, dqRegion.handle],
      writeMask: 0b11000000, // Dpre and dQ are both written
    });

    // Step 2: dKV kernel reads D_precomp written by dQ kernel
    // Experimental ILP variant: each invocation processes several query rows
    // per loop body. Keep it opt-in until a physical device profile proves
    // that the extra registers/code size beat the selected scalar kernel.
    const dkvVariant = process.env.HELIOS_FLASH_BWD_DKV_V2 === "1" ? "_v2" : "";
    const dkvKernel = `flash_attn_bwd_dkv${dkvVariant}${scSuffix}_${BrDKV}_${BcDKV}_${D}`;
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
      groups: [Math.ceil(T / BcDKV), BH, 1],
      push: push4Memo(T, scale, softCap, 0),
      pushSize: 16,
      shape: [BH, T, D],
      allBufs: [bufQ, bufK, bufV, bufDO, bufLSE, dPreRegion.handle, dkRegion.handle, dvRegion.handle],
      writeMask: 0b11000000, // dK and dV are both written
    });

    // Release intermediate D_precomp buffer — only needed between dQ and dKV kernels.
    // Without this, one buffer leaks per layer per step (~16/step), eventually
    // hitting Vulkan's live allocation limit and corrupting training.
    graph.deferRelease(dPreRegion);

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

      const useVec4 = (dim & 3) === 0;
      const kernelName = useVec4 ? "embedding_forward_vec4" : "embedding_forward";
      // Use wgSize=64 for vec4 to maximize WG count (more memory latency hiding)
      const embWg = useVec4 ? 64 : WG_SIZE;
      const pipeline = getPipeline(vk, kernelName, 3, 2 * 4, embWg);
      const region = acquireOutputRegion(vk, totalElements * 4);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      let groups: [number, number, number];
      if (useVec4) {
        // 2D dispatch: x = vec4 columns, y = samples
        const dimVec4 = dim >> 2;
        pushU[0] = dimVec4;  // u32 bits — kernel bitcasts f32→u32
        push[1] = 0;
        groups = [Math.ceil(dimVec4 / embWg), nIdx, 1];
      } else {
        push[0] = totalElements;  // float value for bounds check
        pushU[1] = dim;           // u32 bits — kernel bitcasts f32→u32
        groups = [Math.ceil(totalElements / WG_SIZE), 1, 1];
      }

      graph.record({
        kind: "unary",
        kernel: kernelName,
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups,
        push,
        pushSize: 2 * 4,
        shape: outShape,
        allBufs: [bufWeight, bufIndices, region.handle],
      });

      return graphLazyTensor(vk, outShape, region);
    }

    // CPU fallback
    this.checkFallback("embedding");
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

  rmsNorm(x: TensorData, weight: TensorData, eps: number): TensorData {
    if (shapeSize(x.shape) >= this._minGpuSize) {
      return this.gpuRmsNorm(x, weight, eps);
    }
    // CPU fallback
    this.checkFallback("rmsNorm");
    const shape = x.shape;
    const dim = shape[shape.length - 1];
    const outer = shapeSize(shape) / dim;
    const xd = x.data as Float32Array;
    const wd = weight.data as Float32Array;
    const out = new Float32Array(xd.length);
    for (let i = 0; i < outer; i++) {
      const off = i * dim;
      let ms = 0;
      for (let j = 0; j < dim; j++) ms += xd[off + j] * xd[off + j];
      ms /= dim;
      const invRms = 1 / Math.sqrt(ms + eps);
      for (let j = 0; j < dim; j++) out[off + j] = xd[off + j] * invRms * wd[j];
    }
    return makeTensor(shape, x.dtype, out);
  }

  rope(x: TensorData, cos: TensorData, sin: TensorData): TensorData {
    if (shapeSize(x.shape) >= this._minGpuSize) {
      return this.gpuRope(x, cos, sin);
    }
    // CPU fallback (matches cpu_ref.rope)
    this.checkFallback("rope");
    const shape = x.shape;
    const D = shape[shape.length - 1];
    const T = shape[shape.length - 2];
    const half = D >> 1;
    const rows = shapeSize(shape) / D;
    const xd = x.data as Float32Array;
    const cd = cos.data as Float32Array;
    const sd = sin.data as Float32Array;
    const out = new Float32Array(xd.length);
    for (let r = 0; r < rows; r++) {
      const t = r % T;
      const xBase = r * D;
      const csBase = t * half;
      for (let i = 0; i < half; i++) {
        const c = cd[csBase + i];
        const s = sd[csBase + i];
        const a = xd[xBase + i];
        const bb = xd[xBase + i + half];
        out[xBase + i] = a * c - bb * s;
        out[xBase + i + half] = bb * c + a * s;
      }
    }
    return makeTensor(shape, x.dtype, out);
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

  crossEntropyForwardBackward(
    logits: TensorData,
    targets: TensorData,
  ): { loss: TensorData; gradLogits: TensorData } | null {
    const N = logits.shape[0];
    const C = logits.shape[1];
    const totalElements = N * C;
    const supported = process.env.HELIOS_CE_TRAINING_KERNEL !== "legacy"
      && totalElements >= this._minGpuSize
      && C >= 16
      && (C & 3) === 0;
    if (!supported) return null;

    const vk = this.init();
    const dimVec4 = C >>> 2;
    const ceWg = Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4)))));
    const bufLogits = ensureGpu(vk, logits);
    const bufTargets = ensureGpuRawBits(vk, targets);
    const lossRegion = acquireOutputRegion(vk, N * 4);
    const gradRegion = acquireOutputRegion(vk, totalElements * 4);
    const pipeline = getPipeline(vk, "ce_training_fused_online", 4, 3 * 4, ceWg);
    const push = new Float32Array([dimVec4, N, 1 / N]);

    graph.record({
      kind: "backward",
      kernel: "ce_training_fused_online",
      pipeline,
      inputBufs: [],
      outputRegion: gradRegion,
      groups: [N, 1, 1],
      push,
      pushSize: 3 * 4,
      shape: logits.shape,
      allBufs: [bufLogits, bufTargets, lossRegion.handle, gradRegion.handle],
      writeMask: 0b1100,
    });

    const perRowLosses = graphLazyTensor(vk, [N], lossRegion);
    const gradLogits = graphLazyTensor(vk, logits.shape, gradRegion);
    const totalLoss = this.gpuReduceSum(perRowLosses, false);
    // Keep the scalar lazy. Reading totalLoss.data here used to force the
    // entire forward graph to submit and wait before backward could even be
    // constructed. scale() recognizes a device-resident scalar and records a
    // tiny GPU op instead of falling through to CPU.
    const meanLoss = this.scale(totalLoss, 1 / N);
    releaseGpuBufferFor(perRowLosses);
    if (totalLoss !== perRowLosses) releaseGpuBufferFor(totalLoss);
    return {
      loss: meanLoss,
      gradLogits,
    };
  }

  crossEntropyMaskedForwardBackward(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
  ): { loss: TensorData; gradLogits: TensorData } | null {
    const N = logits.shape[0];
    const C = logits.shape[1];
    const totalElements = N * C;
    const supported = process.env.HELIOS_CE_TRAINING_KERNEL !== "legacy"
      && totalElements >= this._minGpuSize
      && C >= 16
      && (C & 3) === 0;
    if (!supported) return null;

    const maskValues = mask.data as Float32Array;
    let denominator = 0;
    for (let i = 0; i < N; i++) denominator += maskValues[i];
    denominator = Math.max(denominator, 1);

    const vk = this.init();
    const dimVec4 = C >>> 2;
    const ceWg = Math.min(WG_SIZE, Math.max(32, 1 << Math.ceil(Math.log2(Math.max(1, dimVec4)))));
    const bufLogits = ensureGpu(vk, logits);
    const bufTargets = ensureGpuRawBits(vk, targets);
    const bufMask = ensureGpu(vk, mask);
    const lossRegion = acquireOutputRegion(vk, N * 4);
    const gradRegion = acquireOutputRegion(vk, totalElements * 4);
    const pipeline = getPipeline(vk, "ce_masked_training_fused_online", 5, 3 * 4, ceWg);
    const push = new Float32Array([dimVec4, N, 1 / denominator]);

    graph.record({
      kind: "backward",
      kernel: "ce_masked_training_fused_online",
      pipeline,
      inputBufs: [],
      outputRegion: gradRegion,
      groups: [N, 1, 1],
      push,
      pushSize: 3 * 4,
      shape: logits.shape,
      allBufs: [bufLogits, bufTargets, bufMask, lossRegion.handle, gradRegion.handle],
      writeMask: 0b11000,
    });

    const perRowLosses = graphLazyTensor(vk, [N], lossRegion);
    const gradLogits = graphLazyTensor(vk, logits.shape, gradRegion);
    const totalLoss = this.gpuReduceSum(perRowLosses, false);
    const meanLoss = this.scale(totalLoss, 1 / denominator);
    releaseGpuBufferFor(perRowLosses);
    if (totalLoss !== perRowLosses) releaseGpuBufferFor(totalLoss);
    return {
      loss: meanLoss,
      gradLogits,
    };
  }

  crossEntropy(logits: TensorData, targets: TensorData): TensorData {
    const N = logits.shape[0];
    const C = logits.shape[1];

    if (N * C >= this._minGpuSize) {
      const vk = this.init();
      // GPU path: fused log-sum-exp CE kernel (one workgroup per row)
      // Vec4 online variant for C divisible by 4 (vec4 loads + single-pass max+sum)
      const useVec4CE = C % 4 === 0 && C >= 16;
      const kernelName = useVec4CE ? "ce_fwd_vec4" : "ce_fwd_fused";
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const pipeline = getPipeline(vk, kernelName, 3, 2 * 4);
      const region = acquireOutputRegion(vk, N * 4);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = N;
      pushU[1] = C;

      graph.record({
        kind: "unary",
        kernel: kernelName,
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

  crossEntropyMasked(logits: TensorData, targets: TensorData, mask: TensorData): TensorData {
    const N = logits.shape[0];
    const C = logits.shape[1];

    // sum(mask) on the host — mask is CPU-origin (SFT loader) and small; floor
    // at 1 so an all-zero mask row yields exactly 0 rather than 0/0 = NaN.
    const maskArr = mask.data as Float32Array;
    let sumMask = 0;
    for (let i = 0; i < N; i++) sumMask += maskArr[i];
    const denom = Math.max(sumMask, 1);

    if (N * C >= this._minGpuSize) {
      const vk = this.init();
      // Fused masked CE: one workgroup per row writes ce_row * mask[row]. Then
      // reduce-sum on GPU and divide by denom on the host (single scalar readback).
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const bufMask = ensureGpu(vk, mask);
      const pipeline = getPipeline(vk, "ce_fwd_masked", 4, 2 * 4);
      const region = acquireOutputRegion(vk, N * 4);

      const push = new Float32Array(2);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = N;
      pushU[1] = C;

      graph.record({
        kind: "unary",
        kernel: "ce_fwd_masked",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [N, 1, 1],
        push,
        pushSize: 2 * 4,
        shape: [N],
        allBufs: [bufLogits, bufTargets, bufMask, region.handle],
      });

      const perRowLosses = graphLazyTensor(vk, [N], region);
      const totalLoss = this.gpuReduceSum(perRowLosses, false);
      const total = (totalLoss.data as Float32Array)[0];
      return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([total / denom]));
    }

    // CPU fallback — numerically stable log-softmax, masked mean.
    const logProbs = this.cpuLogSoftmax(logits, 1);
    let loss = 0;
    for (let i = 0; i < N; i++) loss -= logProbs.data[i * C + targets.data[i]] * maskArr[i];
    loss /= denom;
    return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([loss]));
  }

  crossEntropyUnlikelihoodMasked(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
    epsilon: number,
  ): TensorData {
    if (!(epsilon > 0 && epsilon <= 1)) {
      throw new Error(`crossEntropyUnlikelihoodMasked epsilon must be in (0,1], got ${epsilon}`);
    }
    const N = logits.shape[0];
    const C = logits.shape[1];
    const maskArr = mask.data as Float32Array;
    let sumMask = 0;
    for (let i = 0; i < N; i++) sumMask += maskArr[i];
    const denom = Math.max(sumMask, 1);

    if (N * C >= this._minGpuSize) {
      const vk = this.init();
      // One workgroup per row performs a stable log-sum-exp, converts the
      // target log-probability to -log(max(1-p_bad,epsilon)), applies the mask,
      // and writes one scalar. Only that N-wide result is reduced/read back.
      const bufLogits = ensureGpu(vk, logits);
      const bufTargets = ensureGpuRawBits(vk, targets);
      const bufMask = ensureGpu(vk, mask);
      const pipeline = getPipeline(vk, "ul_fwd_masked", 4, 3 * 4);
      const region = acquireOutputRegion(vk, N * 4);

      const push = new Float32Array(3);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = N;
      pushU[1] = C;
      push[2] = epsilon;

      graph.record({
        kind: "unary",
        kernel: "ul_fwd_masked",
        pipeline,
        inputBufs: [],
        outputRegion: region,
        groups: [N, 1, 1],
        push,
        pushSize: 3 * 4,
        shape: [N],
        allBufs: [bufLogits, bufTargets, bufMask, region.handle],
      });

      const perRowLosses = graphLazyTensor(vk, [N], region);
      const totalLoss = this.gpuReduceSum(perRowLosses, false);
      const total = (totalLoss.data as Float32Array)[0];
      // Read back only the N per-row scalar losses (not the N*C logits). Since
      // loss_i = -log(1-p_bad_i)*mask_i, recover the clamped p_bad audit value.
      const perRow = perRowLosses.data as Float32Array;
      let activeRows = 0;
      let weightedProbability = 0;
      let maxBadProbability = 0;
      for (let i = 0; i < N; i++) {
        const m = maskArr[i];
        if (m === 0) continue;
        const pBad = 1 - Math.exp(-perRow[i] / m);
        activeRows++;
        weightedProbability += pBad * m;
        maxBadProbability = Math.max(maxBadProbability, pBad);
      }
      this._lastUnlikelihoodStats = {
        activeRows,
        maskMass: sumMask,
        meanBadProbability: weightedProbability / Math.max(sumMask, 1),
        maxBadProbability,
      };
      return makeTensor([], logits.dtype, dtypeArray(logits.dtype).from([total / denom]));
    }

    const logProbs = this.cpuLogSoftmax(logits, 1);
    let loss = 0;
    let activeRows = 0;
    let weightedProbability = 0;
    let maxBadProbability = 0;
    for (let i = 0; i < N; i++) {
      const m = maskArr[i];
      if (m === 0) continue;
      const pBad = Math.exp(logProbs.data[i * C + targets.data[i]]);
      loss -= Math.log(Math.max(1 - pBad, epsilon)) * m;
      activeRows++;
      weightedProbability += pBad * m;
      maxBadProbability = Math.max(maxBadProbability, pBad);
    }
    loss /= denom;
    this._lastUnlikelihoodStats = {
      activeRows,
      maskMass: sumMask,
      meanBadProbability: weightedProbability / Math.max(sumMask, 1),
      maxBadProbability,
    };
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

    // GPU path
    if (size >= this._minGpuSize) {
      const vk = this.init();
      const inputBuf = ensureGpu(vk, a);

      // Fast path: tiled 2D transpose for last-2-dim swaps (attention heads)
      const isLast2 = ndim >= 2 &&
        ((d0 === ndim - 2 && d1 === ndim - 1) || (d0 === ndim - 1 && d1 === ndim - 2));
      if (isLast2) {
        const rows = a.shape[ndim - 2];
        const cols = a.shape[ndim - 1];
        const batch = size / (rows * cols);
        const TILE = 32;

        const pushF = new Float32Array(2);
        const pushU = new Uint32Array(pushF.buffer);
        pushU[0] = rows;
        pushU[1] = cols;

        const pipeline = getPipeline(vk, "transpose_2d_tiled", 2, 2 * 4);
        const outRegion = acquireOutputRegion(vk, size * 4);

        graph.record({
          kind: "unary",
          kernel: "transpose_2d_tiled",
          pipeline,
          inputBufs: [],
          outputRegion: outRegion,
          groups: [Math.ceil(cols / TILE), Math.ceil(rows / TILE), batch],
          push: pushF,
          pushSize: 2 * 4,
          shape: newShape,
          allBufs: [inputBuf, outRegion.handle],
        });

        return graphLazyTensor(vk, newShape, outRegion);
      }

      // General 4D stride-based transpose kernel
      const pad = 4 - ndim;
      const shape4 = ndim < 4
        ? [...Array(pad).fill(1) as number[], ...a.shape]
        : a.shape.slice(0, 4);
      const d0_4 = d0 + pad;
      const d1_4 = d1 + pad;

      const inStrides = shapeStrides(shape4);
      const outShape4 = [...shape4];
      outShape4[d0_4] = shape4[d1_4];
      outShape4[d1_4] = shape4[d0_4];
      const outStrides = shapeStrides(outShape4);
      const tmpS = outStrides[d0_4]; outStrides[d0_4] = outStrides[d1_4]; outStrides[d1_4] = tmpS;

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

  /**
   * Fused 3-way column slice: split [rows, 3*D] into [rows, D] × 3.
   * Single GPU dispatch instead of 3 separate slice_2d dispatches.
   */
  sliceQkv(a: TensorData): [TensorData, TensorData, TensorData] {
    if (a.shape.length !== 2 || a.shape[1] % 3 !== 0)
      throw new Error(`sliceQkv: expected [rows, 3*D], got [${a.shape}]`);
    const rows = a.shape[0];
    const sliceCols = a.shape[1] / 3;
    const srcCols = a.shape[1];
    const outSize = rows * sliceCols;
    const outShape: Shape = [rows, sliceCols];

    const vk = this.init();
    const inputBuf = ensureGpu(vk, a);

    // Use vec4 when both sliceCols and srcCols are divisible by 4
    const useVec4 = (sliceCols & 3) === 0 && (srcCols & 3) === 0;

    const pushF = new Float32Array(3);
    const pushU = new Uint32Array(pushF.buffer);
    if (useVec4) {
      pushU[0] = outSize >>> 2;      // totalVec4 elements
      pushU[1] = sliceCols >>> 2;    // sliceVec4Cols
      pushU[2] = srcCols >>> 2;      // srcVec4Cols
    } else {
      pushU[0] = outSize;            // totalElements per slice
      pushU[1] = sliceCols;          // slice width (D)
      pushU[2] = srcCols;            // source width (3*D)
    }

    const kernelName = useVec4 ? "slice_3way_vec4" : "slice_3way";
    const pipeline = getPipeline(vk, kernelName, 4, 3 * 4);
    const r0 = acquireOutputRegion(vk, outSize * 4);
    const r1 = acquireOutputRegion(vk, outSize * 4);
    const r2 = acquireOutputRegion(vk, outSize * 4);
    const dispatchLen = useVec4 ? (outSize >>> 2) : outSize;
    const groups = Math.ceil(dispatchLen / WG_SIZE);

    graph.record({
      kind: "unary",
      kernel: kernelName,
      pipeline,
      inputBufs: [],
      outputRegion: r0,
      groups: [groups, 1, 1],
      push: pushF,
      pushSize: 3 * 4,
      shape: outShape,
      allBufs: [inputBuf, r0.handle, r1.handle, r2.handle],
      writeMask: 0b1110,  // outputs at binding 1, 2, 3
    });

    return [
      graphLazyTensor(vk, outShape, r0),
      graphLazyTensor(vk, outShape, r1),
      graphLazyTensor(vk, outShape, r2),
    ];
  }

  qkvHeadMajorRope(
    qkv: TensorData,
    cos: TensorData,
    sin: TensorData,
    batch: number,
    sequence: number,
    heads: number,
    headDim: number,
  ): [TensorData, TensorData, TensorData] {
    const modelDim = heads * headDim;
    const rows = batch * sequence;
    if (headDim <= 0 || (headDim & 1) !== 0) {
      throw new Error(`qkvHeadMajorRope: headDim must be positive and even, got ${headDim}`);
    }
    if (qkv.shape.length !== 2 || qkv.shape[0] !== rows || qkv.shape[1] !== 3 * modelDim) {
      throw new Error(
        `qkvHeadMajorRope: expected [${rows},${3 * modelDim}], got [${qkv.shape}]`,
      );
    }
    if (cos.shape[0] !== sequence || cos.shape[1] !== headDim / 2
      || sin.shape[0] !== sequence || sin.shape[1] !== headDim / 2) {
      throw new Error(
        `qkvHeadMajorRope: expected cos/sin [${sequence},${headDim / 2}], got `
          + `[${cos.shape}] and [${sin.shape}]`,
      );
    }

    if (DISABLE_QKV_HEAD_MAJOR_ROPE) {
      const [qFlat, kFlat, vFlat] = this.sliceQkv(qkv);
      const toHeadMajor = (x: TensorData): TensorData => this.reshape(
        this.transpose(this.reshape(x, [batch, sequence, heads, headDim]), 1, 2),
        [batch * heads, sequence, headDim],
      );
      return [
        this.rope(toHeadMajor(qFlat), cos, sin),
        this.rope(toHeadMajor(kFlat), cos, sin),
        toHeadMajor(vFlat),
      ];
    }

    const outputShape: Shape = [batch * heads, sequence, headDim];
    const outputSize = shapeSize(outputShape);
    if (outputSize >= this._minGpuSize) {
      const vk = this.init();
      const qkvBuffer = ensureGpu(vk, qkv);
      const cosBuffer = ensureGpu(vk, cos);
      const sinBuffer = ensureGpu(vk, sin);
      const qRegion = acquireOutputRegion(vk, outputSize * 4);
      const kRegion = acquireOutputRegion(vk, outputSize * 4);
      const vRegion = acquireOutputRegion(vk, outputSize * 4);
      const push = new Float32Array(5);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = outputSize >>> 1;
      pushU[1] = sequence;
      pushU[2] = heads;
      pushU[3] = headDim;
      pushU[4] = modelDim;
      const pipeline = getPipeline(vk, "qkv_head_major_rope", 6, 5 * 4);
      graph.record({
        kind: "unary",
        kernel: "qkv_head_major_rope",
        pipeline,
        inputBufs: [],
        outputRegion: qRegion,
        groups: [Math.ceil((outputSize >>> 1) / WG_SIZE), 1, 1],
        push,
        pushSize: 5 * 4,
        shape: outputShape,
        allBufs: [
          qkvBuffer,
          cosBuffer,
          sinBuffer,
          qRegion.handle,
          kRegion.handle,
          vRegion.handle,
        ],
        writeMask: 0b111000,
      });
      return [
        graphLazyTensor(vk, outputShape, qRegion),
        graphLazyTensor(vk, outputShape, kRegion),
        graphLazyTensor(vk, outputShape, vRegion),
      ];
    }

    this.checkFallback("qkvHeadMajorRope");
    const source = qkv.data as Float32Array;
    const cosData = cos.data as Float32Array;
    const sinData = sin.data as Float32Array;
    const qOut = new Float32Array(outputSize);
    const kOut = new Float32Array(outputSize);
    const vOut = new Float32Array(outputSize);
    const half = headDim >>> 1;
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < heads; h++) {
        for (let t = 0; t < sequence; t++) {
          const sourceBase = (b * sequence + t) * (3 * modelDim) + h * headDim;
          const outputBase = ((b * heads + h) * sequence + t) * headDim;
          const tableBase = t * half;
          for (let i = 0; i < half; i++) {
            const c = cosData[tableBase + i];
            const s = sinData[tableBase + i];
            const qA = source[sourceBase + i];
            const qB = source[sourceBase + i + half];
            const kA = source[sourceBase + modelDim + i];
            const kB = source[sourceBase + modelDim + i + half];
            qOut[outputBase + i] = qA * c - qB * s;
            qOut[outputBase + i + half] = qB * c + qA * s;
            kOut[outputBase + i] = kA * c - kB * s;
            kOut[outputBase + i + half] = kB * c + kA * s;
            vOut[outputBase + i] = source[sourceBase + 2 * modelDim + i];
            vOut[outputBase + i + half] = source[sourceBase + 2 * modelDim + i + half];
          }
        }
      }
    }
    return [
      makeTensor(outputShape, qkv.dtype, qOut),
      makeTensor(outputShape, qkv.dtype, kOut),
      makeTensor(outputShape, qkv.dtype, vOut),
    ];
  }

  qkvHeadMajorRopeBackward(
    grad: TensorData,
    cos: TensorData,
    inverseSin: TensorData,
    batch: number,
    sequence: number,
    heads: number,
    headDim: number,
    which: 0 | 1 | 2,
  ): TensorData {
    const modelDim = heads * headDim;
    const expectedGradShape: Shape = [batch * heads, sequence, headDim];
    if (!this.shapesEqual(grad.shape, expectedGradShape)) {
      throw new Error(
        `qkvHeadMajorRopeBackward: expected grad [${expectedGradShape}], got [${grad.shape}]`,
      );
    }
    if (which !== 0 && which !== 1 && which !== 2) {
      throw new Error(`qkvHeadMajorRopeBackward: invalid branch ${which}`);
    }
    const outputShape: Shape = [batch * sequence, 3 * modelDim];
    const outputSize = shapeSize(outputShape);

    if (DISABLE_QKV_HEAD_MAJOR_ROPE) {
      const unrotated = which < 2 ? this.rope(grad, cos, inverseSin) : grad;
      const tokenMajor = this.reshape(
        this.transpose(this.reshape(unrotated, [batch, heads, sequence, headDim]), 1, 2),
        [batch * sequence, modelDim],
      );
      return this.scatterSlice(
        tokenMajor,
        outputShape,
        [0, which * modelDim],
        [batch * sequence, (which + 1) * modelDim],
      );
    }

    if (outputSize >= this._minGpuSize) {
      const vk = this.init();
      const gradBuffer = ensureGpu(vk, grad);
      const cosBuffer = ensureGpu(vk, cos);
      const inverseSinBuffer = ensureGpu(vk, inverseSin);
      const outputRegion = acquireOutputRegion(vk, outputSize * 4);
      const push = new Float32Array(6);
      const pushU = new Uint32Array(push.buffer);
      pushU[0] = outputSize;
      pushU[1] = sequence;
      pushU[2] = heads;
      pushU[3] = headDim;
      pushU[4] = modelDim;
      pushU[5] = which;
      const pipeline = getPipeline(vk, "qkv_head_major_rope_backward", 4, 6 * 4);
      graph.record({
        kind: "unary",
        kernel: "qkv_head_major_rope_backward",
        pipeline,
        inputBufs: [],
        outputRegion,
        groups: [Math.ceil(outputSize / WG_SIZE), 1, 1],
        push,
        pushSize: 6 * 4,
        shape: outputShape,
        allBufs: [gradBuffer, cosBuffer, inverseSinBuffer, outputRegion.handle],
      });
      return graphLazyTensor(vk, outputShape, outputRegion);
    }

    this.checkFallback("qkvHeadMajorRopeBackward");
    const gradData = grad.data as Float32Array;
    const cosData = cos.data as Float32Array;
    const inverseSinData = inverseSin.data as Float32Array;
    const output = new Float32Array(outputSize);
    const half = headDim >>> 1;
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < heads; h++) {
        for (let t = 0; t < sequence; t++) {
          const gradBase = ((b * heads + h) * sequence + t) * headDim;
          const outputBase = (b * sequence + t) * (3 * modelDim)
            + which * modelDim + h * headDim;
          const tableBase = t * half;
          for (let i = 0; i < half; i++) {
            const gA = gradData[gradBase + i];
            const gB = gradData[gradBase + i + half];
            if (which < 2) {
              const c = cosData[tableBase + i];
              const sInv = inverseSinData[tableBase + i];
              output[outputBase + i] = gA * c - gB * sInv;
              output[outputBase + i + half] = gB * c + gA * sInv;
            } else {
              output[outputBase + i] = gA;
              output[outputBase + i + half] = gB;
            }
          }
        }
      }
    }
    return makeTensor(outputShape, grad.dtype, output);
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
    const useVec4 = (size & 3) === 0;

    const pushF = new Float32Array(5);
    const pushU = new Uint32Array(pushF.buffer);
    pushU[0] = useVec4 ? size >>> 2 : size;
    pushU[1] = (seed | 0) >>> 0;    // seed as u32
    pushU[2] = (counter | 0) >>> 0;  // counter as u32
    // p and scale as f32 bit patterns stored in u32 slots
    pushF[3] = p;
    pushF[4] = scaleVal;

    const kernelName = useVec4 ? "dropout_mask_vec4" : "dropout_mask";
    const pipeline = getPipeline(vk, kernelName, 1, 5 * 4);
    const outRegion = acquireOutputRegion(vk, size * 4);
    const effectiveSize = useVec4 ? size >>> 2 : size;
    const groups = Math.ceil(effectiveSize / WG_SIZE);

    graph.record({
      kind: "unary",
      kernel: kernelName,
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
      if (a.dtype === "f16") {
        const vk = this.init();
        const size = shapeSize(a.shape);
        const bufA = this.ensureGpuF16(vk, a);
        const pipeline = getPipeline(vk, "copy_f16", 2);
        const region = acquireOutputRegion(vk, size * 2);
        graph.record({
          kind: "unary",
          kernel: "copy_f16",
          pipeline,
          inputBufs: [bufA],
          outputRegion: region,
          groups: [Math.ceil(size / WG_SIZE), 1, 1],
          push: push2Memo(size, 0),
          pushSize: PUSH_SIZE,
          shape: a.shape,
        });
        return graphLazyTensorF16(vk, a.shape, region);
      }
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
    this.checkFallback("cpuUnary");
    const Ctor = dtypeArray(a.dtype);
    const out = new Ctor(a.data.length);
    for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]);
    return makeTensor(a.shape, a.dtype, out);
  }

  private cpuBinaryOp(a: TensorData, b: TensorData, fn: (x: number, y: number) => number): TensorData {
    this.checkFallback("cpuBinaryOp");
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
    this.checkFallback("cpuMatmul");
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
    this.checkFallback("cpuSoftmax");
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
    this.checkFallback("cpuLogSoftmax");
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
    this.checkFallback("cpuLayerNorm");
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
      writeMask: 0b1101, // params, first moment, and second moment are written
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
