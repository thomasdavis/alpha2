/**
 * parity-native — the from-scratch stack against the CPU reference.
 *
 * WHAT: every operation the native backend implements, run on both, compared.
 *
 * WHY AGAINST cpu_ref AND NOT AGAINST THE VULKAN BACKEND: the Vulkan one is
 * available and it would be the wrong oracle. Both are GPU implementations of
 * the same intent by the same author, and two implementations agreeing proves
 * only that they made the same assumption -- which is precisely how X58's
 * halved gradient norm and X60's softmax bug both survived a full parity suite.
 * cpu_ref is straightforward host arithmetic written against the definitions,
 * so where it and the device disagree, one of them is wrong about the maths
 * rather than about the hardware.
 *
 * WHAT A SKIP MEANS HERE: no GPU, and nothing was verified. The suite reports
 * that rather than passing quietly, because a green run that exercised nothing
 * is the failure mode this project has already hit once.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { NativeHeliosBackend } from "@alpha/helios";
import { CpuRefBackend } from "@alpha/tensor";
import type { Backend, TensorData } from "@alpha/core";

/*
 * Tolerances, all in one place so drift is one edit.
 *
 * The transcendentals get a looser bound than the exact operations because MUFU
 * is approximate by design -- roughly 22 bits for exp2 and log2. Demanding
 * more would be demanding the hardware be something it is not; demanding less
 * would stop catching a wrong function selector.
 */
const EXACT_TOL = 0;
const MUFU_REL_TOL = 1e-4;
const REDUCE_ABS_TOL = 1e-3;
const NORM_REL_TOL = 1e-3;

const cpu: Backend = new CpuRefBackend();
let gpu: NativeHeliosBackend | null = null;
let why = "";

beforeAll(() => {
  try {
    gpu = new NativeHeliosBackend(0);
  } catch (e) {
    why = e instanceof Error ? e.message : String(e);
  }
});

/** Compare element by element, reporting the first disagreement with its index
 * -- a count of mismatches says nothing about WHERE the pattern broke. */
function agree(
  a: TensorData,
  b: TensorData,
  relTol: number,
  absTol = 0,
): void {
  const x = a.data as ArrayLike<number>;
  const y = b.data as ArrayLike<number>;
  expect(x.length).toBe(y.length);
  for (let i = 0; i < x.length; i++) {
    const diff = Math.abs(x[i] - y[i]);
    const ok = diff <= absTol + relTol * Math.abs(y[i]);
    if (!ok) {
      throw new Error(
        `element ${i}: device ${x[i]} vs reference ${y[i]} (diff ${diff})`,
      );
    }
  }
}

const N = 64;
const seq = (f: (i: number) => number) => Array.from({ length: N }, (_, i) => f(i));

describe("native backend parity with cpu_ref", () => {
  it("reports why it cannot run rather than skipping silently", () => {
    if (!gpu) console.warn(`native backend unavailable: ${why}`);
    expect(true).toBe(true);
  });

  it.runIf(() => gpu !== null)("element-wise arithmetic is exact", () => {
    const g = gpu!;
    const av = seq((i) => (i % 9) - 4);
    const bv = seq((i) => (i % 5) + 1);
    for (const [name, op] of [
      ["add", (B: Backend, x: TensorData, y: TensorData) => B.add(x, y)],
      ["sub", (B: Backend, x: TensorData, y: TensorData) => B.sub(x, y)],
      ["mul", (B: Backend, x: TensorData, y: TensorData) => B.mul(x, y)],
    ] as const) {
      const want = op(cpu, cpu.fromArray(av, [N]), cpu.fromArray(bv, [N]));
      const got = op(g, g.fromArray(av, [N]), g.fromArray(bv, [N]));
      expect(() => agree(got, want, EXACT_TOL), name).not.toThrow();
    }
  });

  it.runIf(() => gpu !== null)("transcendentals agree within MUFU's precision", () => {
    const g = gpu!;
    const pos = seq((i) => i + 1);
    for (const [name, op] of [
      ["exp", (B: Backend, x: TensorData) => B.exp(x)],
      ["log", (B: Backend, x: TensorData) => B.log(x)],
      ["sqrt", (B: Backend, x: TensorData) => B.sqrt(x)],
    ] as const) {
      const src = name === "exp" ? seq((i) => (i % 7) - 3) : pos;
      const want = op(cpu, cpu.fromArray(src, [N]));
      const got = op(g, g.fromArray(src, [N]));
      expect(() => agree(got, want, MUFU_REL_TOL), name).not.toThrow();
    }
  });

  it.runIf(() => gpu !== null)("matmul agrees exactly on integer-valued inputs", () => {
    const g = gpu!;
    const M = 8, K = 8, P = 8;
    const av = Array.from({ length: M * K }, (_, i) => (i % 5) - 2);
    const bv = Array.from({ length: K * P }, (_, i) => (i % 3) + 1);
    /* Exact, and legitimately so: every term and partial sum here is a small
     * integer, well inside what a float represents exactly, so any difference
     * at all is a real one and not accumulated rounding. */
    const want = cpu.matmul(cpu.fromArray(av, [M, K]), cpu.fromArray(bv, [K, P]));
    const got = g.matmul(g.fromArray(av, [M, K]), g.fromArray(bv, [K, P]));
    agree(got, want, EXACT_TOL);
  });

  it.runIf(() => gpu !== null)("whole-tensor sum spans many blocks", () => {
    const g = gpu!;
    /* Larger than one block, and not a multiple of it: the short final block is
     * where a two-level reduction goes wrong. */
    const n = 1000;
    const v = Array.from({ length: n }, (_, i) => (i % 13) - 6);
    const want = cpu.sum(cpu.fromArray(v, [n]));
    const got = g.sum(g.fromArray(v, [n]));
    agree(got, want, 0, REDUCE_ABS_TOL);
  });

  it.runIf(() => gpu !== null)("softmax is a distribution and matches", () => {
    const g = gpu!;
    const v = seq((i) => (i % 11) - 5);
    const want = cpu.softmax(cpu.fromArray(v, [N]));
    const got = g.softmax(g.fromArray(v, [N]));
    agree(got, want, NORM_REL_TOL);
    const total = Array.from(got.data as ArrayLike<number>).reduce((a, b) => a + b, 0);
    expect(Math.abs(total - 1)).toBeLessThan(1e-4);
  });

  it.runIf(() => gpu !== null)("reshape shares memory rather than copying", () => {
    const g = gpu!;
    const t = g.fromArray(seq((i) => i), [N]);
    const r = g.reshape(t, [8, 8]);
    /* Writing through one view must be visible through the other -- that is
     * what makes reshape free, and if it ever silently started copying this is
     * the assertion that would notice. */
    (t.data as Float32Array)[3] = 999;
    expect((r.data as ArrayLike<number>)[3]).toBe(999);
  });

  it.runIf(() => gpu !== null)("an unimplemented operation throws by name", () => {
    /* Not a fallback, on purpose: an operation that quietly ran on the host
     * would be indistinguishable from one that works. */
    expect(() => (gpu as unknown as { argmax(): unknown }).argmax()).toThrow(
      /argmax/,
    );
  });
});
