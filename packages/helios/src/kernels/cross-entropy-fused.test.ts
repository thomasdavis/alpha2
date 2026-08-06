import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";

const SPIRV_MAGIC = 0x07230203;

describe("fused online cross-entropy backward kernels", () => {
  it.each([
    ["ce_backward_fused_online", 3],
    ["ce_masked_backward_fused_online", 4],
    ["ce_training_fused_online", 4],
    ["ce_masked_training_fused_online", 5],
  ] as const)("generates valid-looking SPIR-V for %s (%i bindings)", (name, _bindings) => {
    const words = getKernelSpirv(name, 256);
    expect(words).toBeInstanceOf(Uint32Array);
    expect(words[0]).toBe(SPIRV_MAGIC);
    expect(words.length).toBeGreaterThan(5);
  });

  it("supports a single-subgroup workgroup for small aligned vocabularies", () => {
    const words = getKernelSpirv("ce_backward_fused_online", 32);
    expect(words[0]).toBe(SPIRV_MAGIC);
  });
});
