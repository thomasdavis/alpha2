/**
 * @alpha/tensor -- Tensor backends for the alpha system.
 */

export { CpuRefBackend } from "./cpu_ref.js";

export type {
  Backend,
  TensorData,
} from "@alpha/core";

export type { Dtype, Shape } from "@alpha/core";

// ── Backend registry ──────────────────────────────────────────────────────

import { Registry } from "@alpha/core";
import type { Backend } from "@alpha/core";
import { CpuRefBackend } from "./cpu_ref.js";

export const backendRegistry = new Registry<Backend>("backend");
backendRegistry.register("cpu_ref", () => new CpuRefBackend());
