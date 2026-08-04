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
 * BOTH ARE PERMANENT. The Vulkan backend is not scheduled for removal (operator
 * decision, 2026-08-04) and the from-scratch plan's "delete Vulkan at P5" step
 * is withdrawn.
 *
 * It earns its place: it is the only INDEPENDENT implementation of these
 * kernels that exists, and independence is what makes a disagreement
 * informative. It is what caught a maskedFill that did not broadcast -- head 0
 * masked correctly, head 1 read past the end of the mask, and the forward loss
 * barely moved while the gradients came back with the wrong sign. Nothing in
 * the native stack could have found that alone, because a stack compared only
 * against itself agrees with its own assumptions.
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
