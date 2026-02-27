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

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Native addon interface ──────────────────────────────────────────────────

export interface NativeAddon {
  initDevice():   { deviceName: string; vendorId: number; f16Supported: boolean; hasAsyncTransfer: boolean; coopMatSupported: boolean; coopMatM: number; coopMatN: number; coopMatK: number; hasPushDescriptors: boolean };
  createBuffer(byteLength: number, hostVisible?: number): number;
  uploadBuffer(handle: number, data: Float32Array): void;
  readBuffer(handle: number): Float32Array;
  destroyBuffer(handle: number): void;
  createPipeline(spirv: Uint32Array, numBindings: number, pushConstantSize?: number): number;
  dispatch(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array): number;
  batchBegin(): void;
  batchDispatch(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array, writeMask?: Uint32Array): void;
  batchDispatchMany(packed: ArrayBuffer, count: number): void;
  batchSubmit(): number;
  waitTimeline(value: number): void;
  getCompleted(): number;
  gpuTime(pipeline: number, buffers: number[], gX: number, gY?: number, gZ?: number, pushConstants?: Float32Array): number;
  waitIdle(): void;
  destroy(): void;
}

// ── Loading ─────────────────────────────────────────────────────────────────

let _native: NativeAddon | null = null;
let _deviceInfo: { deviceName: string; vendorId: number; f16Supported: boolean; hasAsyncTransfer: boolean; coopMatSupported: boolean; coopMatM: number; coopMatN: number; coopMatK: number; hasPushDescriptors: boolean } | null = null;

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
export function initDevice(): { deviceName: string; vendorId: number; f16Supported: boolean; hasAsyncTransfer: boolean; coopMatSupported: boolean; coopMatM: number; coopMatN: number; coopMatK: number; hasPushDescriptors: boolean } {
  if (_deviceInfo) return _deviceInfo;

  const addonPath = findNativeAddon();
  const require = createRequire(import.meta.url);
  _native = require(addonPath) as NativeAddon;

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
export function getDeviceInfo(): { deviceName: string; vendorId: number; f16Supported: boolean; hasAsyncTransfer: boolean; coopMatSupported: boolean; coopMatM: number; coopMatN: number; coopMatK: number; hasPushDescriptors: boolean } {
  if (!_deviceInfo) initDevice();
  return _deviceInfo!;
}

/** Destroy the Vulkan device and release all resources. */
export function destroyDevice(): void {
  if (_native) {
    _native.destroy();
    _native = null;
    _deviceInfo = null;
  }
}
