import { describe, it, expect } from "vitest";
import { SeededRng } from "@alpha/core";
import { sampleFromLogits } from "@alpha/inference";

function makeFakeSampler(vocabSize: number) {
  return {
    config: { vocabSize },
    _sampleBuf: new Float32Array(vocabSize),
  } as any;
}

describe("Inference sampling", () => {
  it("top-p with very small threshold keeps only the top token", () => {
    const sampler = makeFakeSampler(5);
    const rng = new SeededRng(42);
    const logits = new Float32Array([10, 9, 1, 0, -1]);

    for (let i = 0; i < 100; i++) {
      const tok = sampleFromLogits(sampler, logits, 1.0, 0, rng, 0.1);
      expect(tok).toBe(0);
    }
  });

  it("top-p and top-k compose correctly", () => {
    const sampler = makeFakeSampler(5);
    const rng = new SeededRng(123);
    const logits = new Float32Array([5, 4, 3, 2, 1]);

    // With top_p=0.7 on this distribution, only top-2 tokens should be eligible.
    for (let i = 0; i < 300; i++) {
      const tok = sampleFromLogits(sampler, logits, 1.0, 0, rng, 0.7);
      expect(tok === 0 || tok === 1).toBe(true);
    }

    // top-k should still be able to hard-cap candidates.
    for (let i = 0; i < 100; i++) {
      const tok = sampleFromLogits(sampler, logits, 1.0, 1, rng, 1.0);
      expect(tok).toBe(0);
    }
  });
});
