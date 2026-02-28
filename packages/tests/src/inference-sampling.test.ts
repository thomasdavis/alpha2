import { describe, it, expect } from "vitest";
import { SeededRng } from "@alpha/core";
import { sampleFromLogits, cloneSession } from "@alpha/inference";

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

  it("cloneSession deep-copies KV cache and scratch buffers", () => {
    const src: any = {
      config: { vocabSize: 4 },
      kCache: [new Float32Array([1, 2, 3])],
      vCache: [new Float32Array([4, 5, 6])],
      _x: new Float32Array([1]),
      _lnOut: new Float32Array([2]),
      _q: new Float32Array([3]),
      _k: new Float32Array([4]),
      _v: new Float32Array([5]),
      _attnScores: new Float32Array([6, 7]),
      _attnOut: new Float32Array([8]),
      _projected: new Float32Array([9]),
      _mlpHidden: new Float32Array([10]),
      _mlpOut: new Float32Array([11]),
      _logits: new Float32Array([12, 13, 14, 15]),
      _sampleBuf: new Float32Array([0, 0, 0, 0]),
      _prefillX: new Float32Array([16]),
      _prefillMaxT: 1,
      _prefillLastLn: new Float32Array([17]),
    };
    const cloned = cloneSession(src);

    expect(Array.from(cloned.kCache[0])).toEqual([1, 2, 3]);
    expect(Array.from(cloned.vCache[0])).toEqual([4, 5, 6]);

    src.kCache[0][0] = 99;
    src._x[0] = 42;
    expect(cloned.kCache[0][0]).toBe(1);
    expect(cloned._x[0]).toBe(1);
  });
});
