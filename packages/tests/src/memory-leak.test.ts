import { describe, it, expect, vi } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { Variable, Tape, add, sub, mul, div, mean, sum } from "@alpha/autograd";

/**
 * Tests that backward-pass intermediate GPU tensors are properly released
 * via the `release` callback. On real GPU backends, failure to release these
 * causes OOM after 1-3 training steps (~512 leaked allocations/step).
 *
 * Uses CpuRefBackend + a spy release function to count releases without
 * needing actual GPU memory.
 */
describe("Memory leak: backward release", () => {
  const B = new CpuRefBackend();

  function makeVar(data: number[], shape: number[]): Variable {
    return new Variable(B.fromArray(data, shape), true);
  }

  /** Create a release spy that tracks released TensorData objects. */
  function makeReleaseSpy() {
    const released: unknown[] = [];
    const fn = vi.fn((td: unknown) => { released.push(td); });
    return { fn, released };
  }

  it("reduceBroadcast releases loop intermediates for multi-step reduction", () => {
    // [2,3,4] + [4] requires leading-dim reduction: sum axis 0 twice
    // That's 2 loop iterations → 1 intermediate should be released
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar(
      Array.from({ length: 24 }, (_, i) => i),
      [2, 3, 4],
    );
    const b = makeVar([1, 2, 3, 4], [4]);
    const c = add(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    // The reduceBroadcast for b's gradient reduces [2,3,4] → [4],
    // requiring sum over axis 0 twice (shape goes 2,3,4 → 3,4 → 4).
    // The first intermediate (3,4) should be released.
    // Check that release was called at least once for an intermediate.
    expect(release.mock.calls.length).toBeGreaterThan(0);

    // Verify gradient correctness
    // a grad: all 1s (no reduction needed for a)
    const ga = Array.from(a.grad!.data as Float32Array);
    for (const v of ga) expect(v).toBeCloseTo(1, 5);
    // b grad: sum over first two dims → each element gets 2*3 = 6
    const gb = Array.from(b.grad!.data as Float32Array);
    for (const v of gb) expect(v).toBeCloseTo(6, 5);
  });

  it("add backward with broadcast calls release", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3], [3, 1]);
    const b = makeVar([10, 20, 30, 40], [1, 4]);
    const c = add(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    expect(release.mock.calls.length).toBeGreaterThan(0);
    // Gradient correctness
    expect(Array.from(a.grad!.data as Float32Array)).toEqual([4, 4, 4]);
    expect(Array.from(b.grad!.data as Float32Array)).toEqual([3, 3, 3, 3]);
  });

  it("sub backward with broadcast calls release", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3], [3, 1]);
    const b = makeVar([10, 20, 30, 40], [1, 4]);
    const c = sub(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    expect(release.mock.calls.length).toBeGreaterThan(0);
    expect(Array.from(a.grad!.data as Float32Array)).toEqual([4, 4, 4]);
    expect(Array.from(b.grad!.data as Float32Array)).toEqual([-3, -3, -3, -3]);
  });

  it("mul backward with broadcast calls release", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3], [3, 1]);
    const b = makeVar([10, 20], [1, 2]);
    const c = mul(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    expect(release.mock.calls.length).toBeGreaterThan(0);
    // da[i] = sum_j b[j] = 10+20 = 30
    expect(Array.from(a.grad!.data as Float32Array)).toEqual([30, 30, 30]);
    // db[j] = sum_i a[i] = 1+2+3 = 6
    expect(Array.from(b.grad!.data as Float32Array)).toEqual([6, 6]);
  });

  it("div backward with broadcast calls release", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([12, 24], [2, 1]);
    const b = makeVar([2, 3, 4], [1, 3]);
    const c = div(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    expect(release.mock.calls.length).toBeGreaterThan(0);
    // a/b with a=[12,24], b=[2,3,4]
    // da/da[i] = sum_j 1/b[j] = 1/2 + 1/3 + 1/4 = 13/12
    const ga = Array.from(a.grad!.data as Float32Array);
    for (const v of ga) expect(v).toBeCloseTo(13 / 12, 4);
  });

  it("mean backward releases broadcastTo intermediate", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]);
    const m = mean(ctx, a, 1, true); // [3,1]
    const loss = sum(ctx, m);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    expect(release.mock.calls.length).toBeGreaterThan(0);
    // Each element of a gets grad 1/4
    const ga = Array.from(a.grad!.data as Float32Array);
    for (const v of ga) expect(v).toBeCloseTo(0.25, 5);
  });

  it("constant release count across repeated steps (no leak growth)", () => {
    const releaseCounts: number[] = [];

    for (let step = 0; step < 5; step++) {
      const tape = new Tape();
      const ctx = { tape, backend: B };

      // Same computation each step: broadcast add → mean → sum
      const a = makeVar([1, 2, 3], [3, 1]);
      const b = makeVar([10, 20, 30, 40], [1, 4]);
      const c = add(ctx, a, b);
      const m = mean(ctx, c, 1, true);
      const loss = sum(ctx, m);

      const { fn: release } = makeReleaseSpy();
      tape.backward(loss, B, release);
      releaseCounts.push(release.mock.calls.length);
      tape.clear(release);
    }

    // All steps should have the same release count — no growth
    for (let i = 1; i < releaseCounts.length; i++) {
      expect(releaseCounts[i]).toBe(releaseCounts[0]);
    }
    // Should be releasing something
    expect(releaseCounts[0]).toBeGreaterThan(0);
  });

  it("gradient correctness: broadcast add still produces correct numerical gradients", () => {
    const eps = 1e-4;

    const fn = (aVals: number[], bVals: number[]): number => {
      const tape = new Tape();
      const ctx = { tape, backend: B };
      const a = makeVar(aVals, [3, 1]);
      const b = makeVar(bVals, [1, 4]);
      const c = add(ctx, a, b);
      const loss = sum(ctx, c);
      return (loss.data.data as Float32Array)[0];
    };

    const aBase = [1, 2, 3];
    const bBase = [10, 20, 30, 40];

    // Numerical gradient for a[0]
    const aPlus = [...aBase]; aPlus[0] += eps;
    const aMinus = [...aBase]; aMinus[0] -= eps;
    const numGradA0 = (fn(aPlus, bBase) - fn(aMinus, bBase)) / (2 * eps);

    // Numerical gradient for b[1]
    const bPlus = [...bBase]; bPlus[1] += eps;
    const bMinus = [...bBase]; bMinus[1] -= eps;
    const numGradB1 = (fn(aBase, bPlus) - fn(aBase, bMinus)) / (2 * eps);

    // Analytic gradients
    const tape = new Tape();
    const ctx = { tape, backend: B };
    const a = makeVar(aBase, [3, 1]);
    const b = makeVar(bBase, [1, 4]);
    const c = add(ctx, a, b);
    const loss = sum(ctx, c);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    const analyticA0 = (a.grad!.data as Float32Array)[0];
    const analyticB1 = (b.grad!.data as Float32Array)[1];

    expect(analyticA0).toBeCloseTo(numGradA0, 0);
    expect(analyticB1).toBeCloseTo(numGradB1, 0);
  });

  it("gradient correctness: mean with axis still produces correct numerical gradients", () => {
    const eps = 1e-4;

    const fn = (vals: number[]): number => {
      const tape = new Tape();
      const ctx = { tape, backend: B };
      const a = makeVar(vals, [3, 4]);
      const m = mean(ctx, a, 1, true);
      const loss = sum(ctx, m);
      return (loss.data.data as Float32Array)[0];
    };

    const base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    // Numerical gradient for element 5 (a[1,1])
    const plus = [...base]; plus[5] += eps;
    const minus = [...base]; minus[5] -= eps;
    const numGrad5 = (fn(plus) - fn(minus)) / (2 * eps);

    // Analytic
    const tape = new Tape();
    const ctx = { tape, backend: B };
    const a = makeVar(base, [3, 4]);
    const m = mean(ctx, a, 1, true);
    const loss = sum(ctx, m);

    const { fn: release } = makeReleaseSpy();
    tape.backward(loss, B, release);

    const analytic5 = (a.grad!.data as Float32Array)[5];
    expect(analytic5).toBeCloseTo(numGrad5, 1);
  });
});
