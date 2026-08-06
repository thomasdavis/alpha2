/**
 * device.ts — Vulkan device management via our native addon.
 *
 * Loads helios_vk.node (compiled from native/helios_vk.c) and
 * provides a typed TypeScript interface over the raw N-API bindings.
 */

import { createRequire } from "node:module";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import type { CooperativeMatrixProperty } from "./cooperative-matrix-capabilities.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Native addon interface ──────────────────────────────────────────────────

/**
 * Capability snapshot returned by the native Vulkan bridge.
 *
 * Values describe the loaded physical device and driver, not an inferred
 * vendor profile. Keep this record additive and fingerprintable: training
 * admission and kernel selection must depend on capabilities rather than a
 * hard-coded NVIDIA/AMD allow-list.
 */
export interface NativeDeviceInfo {
  deviceName: string;
  vendorId: number;
  deviceId: number;
  deviceType: number;
  apiVersion: number;
  driverVersion: number;
  deviceLocalMemoryBytes: number;
  maxComputeSharedMemorySize: number;
  maxComputeWorkGroupInvocations: number;
  maxComputeWorkGroupSizeX: number;
  maxComputeWorkGroupSizeY: number;
  maxComputeWorkGroupSizeZ: number;
  subgroupSize: number;
  subgroupSupportedStages: number;
  subgroupSupportedOperations: number;
  subgroupQuadOperationsInAllStages: boolean;
  subgroupSizeControlSupported: boolean;
  computeFullSubgroupsSupported: boolean;
  minSubgroupSize: number;
  maxSubgroupSize: number;
  maxComputeWorkgroupSubgroups: number;
  requiredSubgroupSizeStages: number;
  timestampValidBits: number;
  timestampPeriodNs: number;
  f16Supported: boolean;
  hasAsyncTransfer: boolean;
  coopMatSupported: boolean;
  coopMat2Supported: boolean;
  coopMatM: number;
  coopMatN: number;
  coopMatK: number;
  /** Complete tuple list returned by vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR. */
  cooperativeMatrixProperties: CooperativeMatrixProperty[];
  hasPushDescriptors: boolean;
  hasBDA: boolean;
  hasDGC: boolean;
}

/**
 * X39 native host-interval breakdown. Phases are disjoint segments of the
 * native dispatch path. `ring_wait` is a GPU-completion wait rather than host
 * work and must not be treated as removable overhead.
 */
export interface NativeHostTiming {
  enabled: boolean;
  batches: number;
  dispatches: number;
  clockReads: number;
  phases: Record<string, { us: number; calls: number }>;
}

export interface NativeAddon {
  initDevice(): NativeDeviceInfo;
  createBuffer(byteLength: number, hostVisible?: number, temporary?: number): number;
  getAllocatorStats?(): {
    activeBuffers: number;
    activeBufferBytes: number;
    slabBuffers: number;
    slabBufferBytes: number;
    individualBuffers: number;
    individualBufferBytes: number;
    tempSlabCount: number;
    tempSlabCapacityBytes: number;
    tempSlabUsedBytes: number;
    tempSlabLiveBytes: number;
    tempSlabLiveRefs: number;
    tempSlabResets: number;
    tempSlabsDisabled: number;
    hostSlabCount: number;
    hostSlabCapacityBytes: number;
    trackedVkMemoryAllocations: number;
    temporaryBufferRequests: number;
    slabFallbacks: number;
    slabFreeRangeReuses: number;
    slabFreeRangeOverflows: number;
  };
  uploadBuffer(handle: number, data: Float32Array): void;
  fillBuffer(handle: number, byteSize: number, value: number): void;
  readBuffer(handle: number): Float32Array;
  destroyBuffer(handle: number): void;
  createPipeline(spirv: Uint32Array, numBindings: number, pushConstantSize?: number): number;
  dispatch(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array): number;
  batchBegin(): void;
  batchDispatch(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array, writeMask?: Uint32Array): void;
  batchDispatchMany(packed: ArrayBuffer, count: number): void;
  batchSubmit(): number;
  batchExecuteAll?(packed: ArrayBuffer, count: number): number;
  batchExecuteAllProfiled?(packed: ArrayBuffer, count: number): {
    timeline: number;
    batchGpuTimeUs: number;
    dispatchCount: number;
    dispatchTimesUs: Float64Array;
  };
  batchExecuteAllDGC?(packed: ArrayBuffer, count: number): number;
  /** X39: disjoint native host-phase totals. Only populated when HELIOS_HOST_TIMING=1. */
  getHostTiming?(): NativeHostTiming;
  resetHostTiming?(): void;
  dgcSetup?(pipelineSlot: number, pushConstantSize: number, maxSequences: number): boolean;
  /** Device address of a buffer as [lo, hi] u32 words. Requires BDA support. */
  dgcGetBufferAddress?(bufferSlot: number): [number, number];
  dgcInfo?(): { hasBDA: boolean; hasDGC: boolean; stride: number; maxSequences: number };
  waitTimeline(value: number): void;
  getCompleted(): number;
  gpuTime(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array, iters?: number, warmup?: number): number;
  waitIdle(): void;
  destroy(): void;
}

// ── Loading ─────────────────────────────────────────────────────────────────

let _native: NativeAddon | null = null;
let _deviceInfo: NativeDeviceInfo | null = null;
let _nativeAddonPath: string | null = null;

function findNativeAddon(): string {
  const envOverride = process.env.HELIOS_NATIVE_ADDON;
  if (envOverride && existsSync(envOverride)) return envOverride;

  const execDir = dirname(process.execPath);
  const cwd = process.cwd();

  // Try multiple locations: native/ dir relative to source, or dist/
  const candidates = [
    // Bun compiled binary sidecar (preferred for distribution)
    join(execDir, "helios_vk.node"),
    // Common workspace locations
    join(cwd, "packages", "helios", "native", "helios_vk.node"),
    join(cwd, "packages", "helios", "dist", "helios_vk.node"),
    join(cwd, ".bun-out", "helios_vk.node"),
    // Node/ts runtime locations
    join(__dirname, "..", "native", "helios_vk.node"),
    join(__dirname, "helios_vk.node"),
    join(__dirname, "..", "..", "native", "helios_vk.node"),
  ];

  for (const p of candidates) {
    if (existsSync(p)) return p;
  }

  throw new Error(
    "Helios: native addon not found. Run `npm run build:native` in packages/helios first.\n" +
    `Searched: ${candidates.join(", ")}`
  );
}

/** Load the native addon and initialize the Vulkan device. */
export function initDevice(): NativeDeviceInfo {
  if (_deviceInfo) return _deviceInfo;

  const addonPath = findNativeAddon();
  const require = createRequire(import.meta.url);
  _native = require(addonPath) as NativeAddon;
  _nativeAddonPath = addonPath;

  _deviceInfo = _native.initDevice();
  return _deviceInfo;
}

/** Get the native addon (must call initDevice first). */
export function getNative(): NativeAddon {
  if (!_native) {
    initDevice();
  }
  return _native!;
}

/** Get device info. */
export function getDeviceInfo(): NativeDeviceInfo {
  if (!_deviceInfo) initDevice();
  return _deviceInfo!;
}

/** Exact native addon binary loaded for the current device session. */
export function getNativeAddonPath(): string {
  if (!_nativeAddonPath) initDevice();
  return _nativeAddonPath!;
}

/** Destroy the Vulkan device and release all resources. */
export function destroyDevice(): void {
  if (_native) {
    _native.destroy();
    _native = null;
    _deviceInfo = null;
    _nativeAddonPath = null;
  }
}
