import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";

const SPIRV_MAGIC = 0x07230203;

describe("fused QKV head-major RoPE kernels", () => {
  it.each([32, 64, 128, 256] as const)("generates forward SPIR-V at workgroup size %i", (wgSize) => {
    const words = getKernelSpirv("qkv_head_major_rope", wgSize);
    expect(words).toBeInstanceOf(Uint32Array);
    expect(words[0]).toBe(SPIRV_MAGIC);
    expect(words.length).toBeGreaterThan(5);
  });

  it.each([32, 64, 128, 256] as const)("generates backward SPIR-V at workgroup size %i", (wgSize) => {
    const words = getKernelSpirv("qkv_head_major_rope_backward", wgSize);
    expect(words).toBeInstanceOf(Uint32Array);
    expect(words[0]).toBe(SPIRV_MAGIC);
    expect(words.length).toBeGreaterThan(5);
  });
});
