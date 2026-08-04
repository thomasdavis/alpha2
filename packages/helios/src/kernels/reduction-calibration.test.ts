import { describe, expect, it } from "vitest";

import { getKernelSpirv } from "./index.js";
import { Op } from "./helpers.js";

function countInstructions(words: Uint32Array, opcode: number): number {
  let count = 0;
  for (let index = 5; index < words.length;) {
    const wordCount = words[index] >>> 16;
    if (wordCount === 0) throw new Error(`invalid zero-word SPIR-V instruction at ${index}`);
    if ((words[index] & 0xffff) === opcode) count++;
    index += wordCount;
  }
  return count;
}

describe("FP16x3 calibration max-absolute reduction", () => {
  it("adds absolute-value and non-finite propagation without changing ordinary max", () => {
    const ordinary = getKernelSpirv("max_reduce", 128);
    const calibration = getKernelSpirv("max_abs_reduce", 128);
    expect(countInstructions(ordinary, Op.IsNan)).toBe(0);
    expect(countInstructions(calibration, Op.IsNan)).toBeGreaterThan(0);
    expect(countInstructions(calibration, Op.IsInf)).toBeGreaterThan(0);
    expect(countInstructions(calibration, Op.Select)).toBeGreaterThan(0);
    expect(calibration).not.toEqual(ordinary);
  });
});
