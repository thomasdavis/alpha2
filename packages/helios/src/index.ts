/**
 * @alpha/helios — Hand-written GPU compute backend.
 *
 * Zero npm dependencies. TWO backends live here during the changeover:
 *
 *   HeliosBackend        the Vulkan one, which compiles SPIR-V and lets the
 *                        vendor driver turn it into machine code
 *   NativeHeliosBackend  the from-scratch one, which assembles its own sm_86
 *                        and submits through its own ioctl path -- no Vulkan,
 *                        no vendor compiler in the runtime
 *
 * The Vulkan one is still the default and is kept ONLY as a diffing oracle
 * while the native one is brought to parity. It is the sole independent
 * implementation of these kernels that exists, and deleting it before parity
 * would throw that away for nothing. It goes at P5, not before.
 *
 * Architecture:
 *   native/helios_vk.c  → Vulkan device/buffer/pipeline/dispatch (C, ~600 lines)
 *   src/spirv.ts         → SPIR-V binary assembler (TypeScript)
 *   src/kernels.ts       → Compute kernel generators (TypeScript → SPIR-V)
 *   src/device.ts        → Native addon loader + device management
 *   src/backend.ts       → HeliosBackend implementing @alpha/core Backend
 */

export { HeliosBackend, type GpuDeviceInfo } from "./backend.js";
export { NativeHeliosBackend, type NativeTensor } from "./nativeBackend.js";
export { nativeAddon, NativeBuffer, type NativeAddon as HeliosNativeAddon } from "./nativeDevice.js";
export { initDevice, destroyDevice, getDeviceInfo, getNative, getNativeAddonPath, type NativeAddon, type NativeDeviceInfo } from "./device.js";
export {
  VulkanComponentType,
  VulkanScope,
  analyzeCooperativeMatrixCapabilities,
  canonicalizeCooperativeMatrixProperties,
  type CooperativeMatrixCapabilityAnalysis,
  type CooperativeMatrixProperty,
} from "./cooperative-matrix-capabilities.js";
export { SpirVBuilder } from "./spirv.js";
export { getKernelSpirv } from "./kernels.js";
export {
  CoopF16x3BalancePlanRuntime,
  canonicalCoopF16x3Descriptor,
  compileCoopF16x3BalancePlan,
  coopF16x3GraphFingerprint,
  loadCoopF16x3BalancePlan,
  parseCoopF16x3CalibrationJsonl,
  type CoopF16x3BalancePlan,
  type CoopF16x3CalibrationRecord,
  type CoopF16x3MatmulDescriptor,
} from "./coop-balance-plan.js";

// Re-export types from core
export type { Backend, TensorData, Dtype, Shape } from "@alpha/core";

// ── Backend registry ────────────────────────────────────────────────────────

import { Registry } from "@alpha/core";
import type { Backend } from "@alpha/core";
import { HeliosBackend } from "./backend.js";

export const heliosRegistry = new Registry<Backend>("backend");
heliosRegistry.register("helios", () => new HeliosBackend());
