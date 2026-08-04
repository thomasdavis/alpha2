import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";
import { Op } from "./helpers.js";

function countInstructions(words: Uint32Array, opcode: number): number {
  let count = 0;
  for (let i = 5; i < words.length;) {
    const first = words[i];
    const wordCount = first >>> 16;
    if (wordCount === 0) throw new Error(`invalid zero-word SPIR-V instruction at ${i}`);
    if ((first & 0xffff) === opcode) count++;
    i += wordCount;
  }
  return count;
}

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

  it("generates tile-local FP16x3 precision emulation for every layout", () => {
    for (const variant of [
      "basic",
      "batched",
      "transposed",
      "transposed_batched",
      "transposed_a",
      "transposed_a_batched",
    ]) {
      const spirv = getKernelSpirv(
        `matmul_coop_${variant}_16_16_16_f16x3_s2x2_r4x4_km2`,
        64,
      );
      expect(spirv).toBeInstanceOf(Uint32Array);
      expect(spirv.length).toBeGreaterThan(100);
    }
    for (const variant of ["basic", "transposed", "transposed_a"]) {
      const spirv = getKernelSpirv(
        `matmul_coop_splitk_${variant}_16_16_16_f16x3_s2x2_r4x4_km2`,
        64,
      );
      expect(spirv).toBeInstanceOf(Uint32Array);
      expect(spirv.length).toBeGreaterThan(100);
    }
  });

  it("emits three MMA products and explicit residual subtraction", () => {
    const ordinary = getKernelSpirv("matmul_coop_basic_16_16_16_km1", 64);
    const emulated = getKernelSpirv("matmul_coop_basic_16_16_16_f16x3_km1", 64);
    const ordinaryMma = countInstructions(ordinary, Op.OpCooperativeMatrixMulAddKHR);
    const emulatedMma = countInstructions(emulated, Op.OpCooperativeMatrixMulAddKHR);
    expect(ordinaryMma).toBeGreaterThan(0);
    expect(emulatedMma).toBe(ordinaryMma * 3);
    expect(countInstructions(emulated, Op.FSub)).toBeGreaterThan(0);
  });

  it("makes the encoded kMulti suffix authoritative", () => {
    const km1 = getKernelSpirv("matmul_coop_basic_16_16_16_km1", 64);
    const km2 = getKernelSpirv("matmul_coop_basic_16_16_16_km2", 64);
    const km4 = getKernelSpirv("matmul_coop_basic_16_16_16_km4", 64);
    expect(km1.length).not.toBe(km2.length);
    expect(km2.length).not.toBe(km4.length);
  });

  it("keeps FP16x3 incompatible with pre-cast FP16 inputs", () => {
    expect(() => getKernelSpirv(
      "matmul_coop_transposed_16_16_16_f16in_f16x3_s2x2_r4x4_km2",
      64,
    )).toThrow(/requires FP32 inputs/);
    expect(() => getKernelSpirv(
      "matmul_coop_basic_16_16_16_f16acc_f16x3_km1",
      64,
    )).toThrow(/requires FP32 inputs and FP32 accumulation/);
  });
});
