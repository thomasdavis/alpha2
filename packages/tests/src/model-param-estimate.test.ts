import { describe, expect, it } from "vitest";
import { estimateGPTParamCount } from "@alpha/model";
import type { ModelConfig } from "@alpha/core";

function config(overrides: Partial<ModelConfig>): ModelConfig {
  return {
    vocabSize: 12_288,
    blockSize: 1_024,
    nLayer: 16,
    nHead: 8,
    nEmbd: 512,
    dropout: 0,
    ...overrides,
  };
}

describe("estimateGPTParamCount", () => {
  it("matches the archived tied-RoPE Alpha foundation", () => {
    expect(estimateGPTParamCount(config({
      ffnActivation: "swiglu",
      ffnDim: 1_408,
      normType: "rmsnorm",
      posEnc: "rope",
      tieEmbeddings: true,
    }))).toBe(57_688_576);
  });

  it("matches the budget-shaped tied-RoPE candidate", () => {
    expect(estimateGPTParamCount(config({
      nLayer: 18,
      nHead: 10,
      nEmbd: 640,
      ffnActivation: "swiglu",
      ffnDim: 1_728,
      normType: "rmsnorm",
      posEnc: "rope",
      tieEmbeddings: true,
    }))).toBe(97_098_880);
  });

  it("counts learned positions, untied head, and LayerNorm biases", () => {
    expect(estimateGPTParamCount(config({
      nLayer: 14,
      ffnActivation: "swiglu",
      ffnDim: 1_408,
      normType: "layernorm",
      posEnc: "learned",
      tieEmbeddings: false,
    }))).toBe(58_094_592);
  });
});
