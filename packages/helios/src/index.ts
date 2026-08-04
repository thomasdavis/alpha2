/**
 * @alpha/helios — Hand-written GPU compute backend.
 *
 * Zero npm dependencies. Uses a from-scratch Vulkan native addon (C + N-API)
 * and TypeScript-generated SPIR-V compute shaders.
 *
 * Architecture:
 *   native/helios_vk.c  → Vulkan device/buffer/pipeline/dispatch (C, ~600 lines)
 *   src/spirv.ts         → SPIR-V binary assembler (TypeScript)
 *   src/kernels.ts       → Compute kernel generators (TypeScript → SPIR-V)
 *   src/device.ts        → Native addon loader + device management
 *   src/backend.ts       → HeliosBackend implementing @alpha/core Backend
 */

export { HeliosBackend, type GpuDeviceInfo } from "./backend.js";
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
