/**
 * known-answer-native — the native backend against ALGEBRA, not against cpu_ref.
 *
 * WHY THIS EXISTS SEPARATELY from backend-ops-native: that file compares against
 * cpu_ref, which is a SECOND IMPLEMENTATION. Two implementations agreeing proves
 * they made the same assumption, and this project has been bitten by exactly
 * that twice -- X58's gradient norm was silently half its true value and X60's
 * softmax was wrong, and both survived a full parity suite because parity asks
 * "do these agree?", which goes quiet when both sides are wrong.
 *
 * So every expected value below is derived from the DEFINITION and written out
 * by hand: a sum of an arithmetic series in closed form, a softmax of a constant
 * row, a matmul whose product is known without computing it. Where a value
 * cannot be stated in closed form it is not tested here -- it is tested against
 * cpu_ref next door, and the difference between the two files is the point.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { NativeHeliosBackend } from "@alpha/helios";

let gpu: NativeHeliosBackend | null = null;
let why = "";
beforeAll(() => {
  try { gpu = new NativeHeliosBackend(0); }
  catch (e) { why = e instanceof Error ? e.message : String(e); }
});

const at = (t: { data: ArrayLike<number> }, i: number) => t.data[i];
const TOL = 1e-4;

describe("native backend, known answers from the definitions", () => {
  it("reports why it cannot run rather than skipping silently", () => {
    if (!gpu) console.warn(`native backend unavailable: ${why}`);
    expect(true).toBe(true);
  });

  it.runIf(() => gpu !== null)("sum of 1..n is n(n+1)/2, exactly", () => {
    const g = gpu!;
    /* Closed form, and every partial sum here is an integer a float represents
     * exactly, so this is an equality rather than a tolerance. n = 1000 spans
     * many blocks and leaves a short final one. */
    const n = 1000;
    const t = g.fromArray(Array.from({ length: n }, (_, i) => i + 1), [n]);
    expect(at(g.sum(t) as never, 0)).toBe((n * (n + 1)) / 2);
  });

  it.runIf(() => gpu !== null)("mean of a constant row is that constant", () => {
    const g = gpu!;
    const t = g.full([256], 3.25);
    expect(at(g.mean(t) as never, 0)).toBe(3.25);
  });

  it.runIf(() => gpu !== null)("softmax of a constant row is uniform", () => {
    const g = gpu!;
    /* Every logit equal means every probability is 1/n, whatever the constant
     * is -- which also proves the max-subtraction, since without it a large
     * constant overflows exp and gives NaN rather than a uniform row. */
    for (const c of [0, 50, -50]) {
      const p = g.softmax(g.full([64], c));
      for (let i = 0; i < 64; i++) {
        expect(Math.abs(at(p as never, i) - 1 / 64)).toBeLessThan(TOL);
      }
    }
  });

  it.runIf(() => gpu !== null)("identity times anything is anything", () => {
    const g = gpu!;
    const N = 8;
    const I = g.fromArray(
      Array.from({ length: N * N }, (_, i) => (i % (N + 1) === 0 ? 1 : 0)), [N, N]);
    const a = Array.from({ length: N * N }, (_, i) => (i % 7) - 3);
    const out = g.matmul(g.fromArray(a, [N, N]), I);
    /* Exact: multiplying by one and adding zeros is exact in floating point,
     * so any difference at all is a real one. */
    for (let i = 0; i < N * N; i++) expect(at(out as never, i)).toBe(a[i]);
  });

  it.runIf(() => gpu !== null)("rmsNorm of a constant row has unit magnitude", () => {
    const g = gpu!;
    /* For a row of a repeated c, mean(x^2) is c^2, so x/sqrt(c^2) is sign(c) --
     * independent of the constant. Any residual dependence on c would mean the
     * mean or the reciprocal square root is wrong. */
    for (const c of [2, 8, 0.25]) {
      const out = g.rmsNorm(g.full([64], c), g.ones([64]), 0);
      for (let i = 0; i < 64; i++) {
        expect(Math.abs(at(out as never, i) - 1)).toBeLessThan(1e-3);
      }
    }
  });

  it.runIf(() => gpu !== null)("cross entropy of a uniform distribution is log(C)", () => {
    const g = gpu!;
    /* Equal logits give probability 1/C for every class, so the loss is
     * -log(1/C) = log(C) regardless of which target is chosen. */
    const C = 16, rows = 8;
    const logits = g.zeros([rows, C]);
    const targets = g.fromArray(Array.from({ length: rows }, (_, i) => i % C), [rows]);
    const loss = g.crossEntropy(logits, targets);
    expect(Math.abs(at(loss as never, 0) - Math.log(C))).toBeLessThan(1e-3);
  });

  it.runIf(() => gpu !== null)("a causal mask blocks exactly the future", () => {
    const g = gpu!;
    const N = 8;
    const m = g.causalMask(N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = at(m as never, r * N + c);
        /* Zero on and below the diagonal -- INCLUDING the diagonal, which is
         * the single most common way to get this wrong and is silent. */
        if (c <= r) expect(v).toBe(0);
        else expect(v).toBe(Number.NEGATIVE_INFINITY);
      }
    }
  });

  it.runIf(() => gpu !== null)("exp and log invert each other", () => {
    const g = gpu!;
    /* Not a closed-form value, but a closed-form RELATION, and it holds only if
     * both the base conversions are right -- the hardware provides exp2 and
     * log2, so each kernel folds a log2(e) or an ln(2), and a missing one shows
     * up here as a constant factor. */
    const src = Array.from({ length: 64 }, (_, i) => (i % 9) + 1);
    const back = g.exp(g.log(g.fromArray(src, [64])));
    for (let i = 0; i < 64; i++) {
      expect(Math.abs(at(back as never, i) - src[i]) / src[i]).toBeLessThan(1e-3);
    }
  });
});
