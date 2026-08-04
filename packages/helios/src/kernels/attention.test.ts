import { describe, expect, it } from "vitest";

import {
  kernelFlashAttentionForward,
  kernelFlashAttentionBackwardDQ,
  kernelFlashAttentionBackwardDKV,
  kernelFlashAttentionBackwardDKVV2,
} from "./attention.js";
import { kernelFlashAttentionCoop2Forward } from "./attention-coop2.js";
import { getKernelSpirv } from "./index.js";

describe("token-major Flash Attention kernel generation", () => {
  it("generates scalar forward and paired backward variants", () => {
    expect(kernelFlashAttentionForward(32, 16, 64, false, true)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDQ(32, 16, 64, false, true)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDKV(32, 32, 64, false, true)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDQ(32, 16, 64, false, true, true)).toBeInstanceOf(Uint32Array);
    expect(kernelFlashAttentionBackwardDKV(32, 32, 64, false, true, true)).toBeInstanceOf(Uint32Array);
    expect(() => kernelFlashAttentionBackwardDQ(32, 16, 64, false, false, true)).toThrow(
      "requires token-major",
    );
    expect(() => kernelFlashAttentionBackwardDKV(32, 32, 60, false, true, true)).toThrow(
      "divisible by 8",
    );
  });

  it("generates the cooperative-matrix forward variant", () => {
    expect(kernelFlashAttentionCoop2Forward(
      16, 16, 64, "full", "workgroup", false, null, false, false, 1, 64, false, true,
    )).toBeInstanceOf(Uint32Array);
  });

  it("resolves stable token-major names through the kernel registry", () => {
    expect(getKernelSpirv("flash_attn_fwd_32_16_64_tm", 32)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv("flash_attn_bwd_dq_32_16_64_tm", 32)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv("flash_attn_bwd_dkv_32_32_64_tm", 32)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv("flash_attn_bwd_dq_32_16_64_tm_gqkv", 32)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv("flash_attn_bwd_dkv_32_32_64_tm_gqkv", 32)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv("flash_attn_coop2_fwd_16_16_64_ls64_tm_wg", 64)).toBeInstanceOf(Uint32Array);
    expect(getKernelSpirv(
      "flash_attn_coop2_fwd_sc30_in16_nolse_16_16_64_ls64_db_tm_wg", 64,
    )).toBeInstanceOf(Uint32Array);
  });
});

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
