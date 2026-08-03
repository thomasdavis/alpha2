import { describe, expect, it } from "vitest";

import {
  kernelFlashAttentionBackwardDKV,
  kernelFlashAttentionBackwardDKVV2,
} from "./attention.js";

describe("flash attention backward dKV tile contracts", () => {
  it("generates the selected square tile and corrected integer-multiple tiles", () => {
    expect(kernelFlashAttentionBackwardDKV(32, 32, 64)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDKV(32, 16, 64)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDKV(64, 32, 64)).toBeInstanceOf(Uint32Array);
  });

  it("rejects a key tile wider than the staged query tile", () => {
    expect(() => kernelFlashAttentionBackwardDKV(32, 64, 64)).toThrow(
      "must be an integer multiple",
    );
  });

  it("keeps the experimental v2 kernel on its proven square-tile contract", () => {
    expect(kernelFlashAttentionBackwardDKVV2(32, 32, 64)).toBeInstanceOf(Uint32Array);
    expect(() => kernelFlashAttentionBackwardDKVV2(64, 32, 64)).toThrow(
      "currently requires square",
    );
  });
});
