import { describe, it, expect } from "vitest";
import { SeededRng } from "@alpha/core";

describe("SeededRng", () => {
  it("produces deterministic sequences", () => {
    const rng1 = new SeededRng(42);
    const rng2 = new SeededRng(42);

    const seq1 = Array.from({ length: 10 }, () => rng1.next());
    const seq2 = Array.from({ length: 10 }, () => rng2.next());

    expect(seq1).toEqual(seq2);
  });

  it("produces values in [0, 1)", () => {
    const rng = new SeededRng(123);
    for (let i = 0; i < 1000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("nextGauss has roughly zero mean", () => {
    const rng = new SeededRng(42);
    let sum = 0;
    const n = 10000;
    for (let i = 0; i < n; i++) sum += rng.nextGauss();
    expect(Math.abs(sum / n)).toBeLessThan(0.1);
  });

  it("different seeds give different sequences", () => {
    const rng1 = new SeededRng(1);
    const rng2 = new SeededRng(2);
    const v1 = rng1.next();
    const v2 = rng2.next();
    expect(v1).not.toBe(v2);
  });
});
