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

export { HeliosBackend } from "./backend.js";
export { initDevice, destroyDevice, getDeviceInfo } from "./device.js";
export { SpirVBuilder } from "./spirv.js";

// Re-export types from core
export type { Backend, TensorData, Dtype, Shape } from "@alpha/core";

// ── Backend registry ────────────────────────────────────────────────────────

import { Registry } from "@alpha/core";
import type { Backend } from "@alpha/core";
import { HeliosBackend } from "./backend.js";

export const heliosRegistry = new Registry<Backend>("backend");
heliosRegistry.register("helios", () => new HeliosBackend());
