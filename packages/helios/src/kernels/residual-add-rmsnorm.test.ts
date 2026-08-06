import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";

const SPIRV_MAGIC = 0x07230203;

describe("fused residual-add RMSNorm kernel", () => {
  it.each([32, 64, 128, 256] as const)("generates valid-looking SPIR-V at workgroup size %i", (wgSize) => {
    const words = getKernelSpirv("residual_add_rmsnorm", wgSize);
    expect(words).toBeInstanceOf(Uint32Array);
    expect(words[0]).toBe(SPIRV_MAGIC);
    expect(words.length).toBeGreaterThan(5);
  });
});
