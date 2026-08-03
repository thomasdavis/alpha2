import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";

describe("cooperative matrix input-storage variants", () => {
  it("generates the fused f32-input training kernel", () => {
    expect(
      getKernelSpirv("matmul_coop_transposed_16_16_16_s2x2_r4x4_km4", 64),
    ).toBeInstanceOf(Uint32Array);
  });

  it("retains the historical pre-cast f16-input kernel", () => {
    expect(
      getKernelSpirv("matmul_coop_transposed_16_16_16_f16in_s2x2_r4x4_km4", 64),
    ).toBeInstanceOf(Uint32Array);
  });
});
