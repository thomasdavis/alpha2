/**
 * backend-ops-native — EVERY Backend method against cpu_ref, at model shapes.
 *
 * WHY THIS EXISTS, stated plainly: the kernels were tested thoroughly -- 51 of
 * them, with mutation-tested checkers -- and the backend layer above them was
 * tested at eight operations and one shape. Every bug in the bridge lived in
 * that gap: embedding returning the wrong RANK, transpose ignoring its
 * dimension arguments, keepdims dropped, broadcasting refused, cat and slice
 * limited to one axis.
 *
 * Each surfaced seventy operations downstream, in the backward pass, in a node
 * that had nothing to do with it. Not one of them was hard to find once looked
 * for at its own boundary -- they were hard to find because nothing looked
 * there.
 *
 * So: every method, at the RANKS the model uses. A [8,16] proves almost
 * nothing; the model passes [1,8,16] and [1,2,8,8], and rank is precisely what
 * was wrong.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { NativeHeliosBackend } from "@alpha/helios";
import { CpuRefBackend } from "@alpha/tensor";
import type { Backend, TensorData, Shape } from "@alpha/core";

let gpu: NativeHeliosBackend | null = null;
let why = "";
beforeAll(() => {
  try { gpu = new NativeHeliosBackend(0); }
  catch (e) { why = e instanceof Error ? e.message : String(e); }
});
const cpu: Backend = new CpuRefBackend();

const EXACT = 0;
const APPROX = 1e-4;

const fill = (shape: Shape, f: (i: number) => number) => {
  const n = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: n }, (_, i) => f(i));
};

/** Run `op` on both backends and require agreement in SHAPE and in values. */
function both(name: string, tol: number, op: (B: Backend) => TensorData): void {
  const want = op(cpu);
  const got = op(gpu!);
  expect(got.shape.join(","), `${name}: shape`).toBe(want.shape.join(","));
  const x = got.data as ArrayLike<number>;
  const y = want.data as ArrayLike<number>;
  for (let i = 0; i < y.length; i++) {
    /* Two identical infinities agree, and |x - y| on them is NaN -- which the
     * comparison below would report as a disagreement. A causal mask is made of
     * them, so this is not an edge case here, it is the common one. */
    if (x[i] === y[i]) continue;
    const d = Math.abs(x[i] - y[i]);
    if (!(d <= tol + tol * Math.abs(y[i]))) {
      throw new Error(`${name}: [${i}] device ${x[i]} vs reference ${y[i]}`);
    }
  }
}

/* Model-realistic shapes. Rank matters and was what broke. */
const V = fill([1, 8, 16], (i) => ((i % 13) - 6) * 0.25);
const W = fill([16, 16], (i) => ((i % 7) - 3) * 0.5);
const QKV = fill([8, 48], (i) => ((i % 11) - 5) * 0.25);

describe("every Backend method, native vs cpu_ref, at model ranks", () => {
  it("reports why it cannot run rather than skipping silently", () => {
    if (!gpu) console.warn(`native backend unavailable: ${why}`);
    expect(true).toBe(true);
  });

  it.runIf(() => gpu !== null)("slice takes the right columns out of a QKV block", () => {
    /* The exact call the attention block makes: three [8,16] blocks out of a
     * [8,48]. Column slicing, so no trailing dimension is taken whole and the
     * copy cannot fall back to one contiguous run. */
    for (const [lo, hi] of [[0, 16], [16, 32], [32, 48]] as const) {
      both(`slice cols ${lo}:${hi}`, EXACT, (B) =>
        B.slice(B.fromArray(QKV, [8, 48]), [0, lo], [8, hi]));
    }
    /* And a row slice, where the inner dimension IS taken whole. */
    both("slice rows", EXACT, (B) =>
      B.slice(B.fromArray(QKV, [8, 48]), [2, 0], [5, 48]));
  });

  it.runIf(() => gpu !== null)("embedding keeps the indices' rank", () => {
    both("embedding", EXACT, (B) =>
      B.embedding(B.fromArray(W, [16, 16]),
                  B.fromArray(fill([1, 8], (i) => i % 16), [1, 8])));
  });

  it.runIf(() => gpu !== null)("transpose honours its dimension arguments", () => {
    both("transpose(-2,-1)", EXACT, (B) => B.transpose(B.fromArray(V, [1, 8, 16]), -2, -1));
    both("transpose(1,2)", EXACT, (B) =>
      B.transpose(B.fromArray(fill([1, 2, 8, 8], (i) => i * 0.1), [1, 2, 8, 8]), 1, 2));
  });

  it.runIf(() => gpu !== null)("reductions honour axis and keepdims", () => {
    for (const axis of [0, 1, 2, -1] as const) {
      for (const keep of [false, true]) {
        both(`sum(${axis},${keep})`, APPROX, (B) => B.sum(B.fromArray(V, [1, 8, 16]), axis, keep));
        both(`mean(${axis},${keep})`, APPROX, (B) => B.mean(B.fromArray(V, [1, 8, 16]), axis, keep));
      }
    }
  });

  it.runIf(() => gpu !== null)("binary ops broadcast", () => {
    both("add [1,8,16]+[16]", EXACT, (B) =>
      B.add(B.fromArray(V, [1, 8, 16]), B.fromArray(fill([16], (i) => i * 0.5), [16])));
    both("mul [1,8,16]*[1,8,1]", EXACT, (B) =>
      B.mul(B.fromArray(V, [1, 8, 16]), B.fromArray(fill([1, 8, 1], (i) => i + 1), [1, 8, 1])));
  });

  it.runIf(() => gpu !== null)("matmul batches on the left and on both sides", () => {
    both("matmul [1,8,16]x[16,16]", APPROX, (B) =>
      B.matmul(B.fromArray(V, [1, 8, 16]), B.fromArray(W, [16, 16])));
    both("matmul [2,4,8]x[2,8,4]", APPROX, (B) =>
      B.matmul(B.fromArray(fill([2, 4, 8], (i) => (i % 5) - 2), [2, 4, 8]),
               B.fromArray(fill([2, 8, 4], (i) => (i % 3) + 1), [2, 8, 4])));
  });

  it.runIf(() => gpu !== null)("cat joins on any axis", () => {
    both("cat axis 0", EXACT, (B) =>
      B.cat([B.fromArray(V, [1, 8, 16]), B.fromArray(V, [1, 8, 16])], 0));
    both("cat axis 1", EXACT, (B) =>
      B.cat([B.fromArray(V, [1, 8, 16]), B.fromArray(V, [1, 8, 16])], 1));
    both("cat axis 2", EXACT, (B) =>
      B.cat([B.fromArray(V, [1, 8, 16]), B.fromArray(V, [1, 8, 16])], 2));
  });

  it.runIf(() => gpu !== null)("normalisations keep rank and act per row", () => {
    both("softmax(-1)", APPROX, (B) => B.softmax(B.fromArray(V, [1, 8, 16]), -1));
    both("rmsNorm", APPROX, (B) =>
      B.rmsNorm(B.fromArray(V, [1, 8, 16]), B.fromArray(fill([16], (i) => 1 + i * 0.1), [16]), 1e-5));
    both("layerNorm", APPROX, (B) =>
      B.layerNorm(B.fromArray(V, [1, 8, 16]),
                  B.fromArray(fill([16], (i) => 1 + i * 0.1), [16]),
                  B.fromArray(fill([16], (i) => i * 0.05), [16]), 1e-5));
  });

  /*
   * The operations the first pass of this file did not cover -- and the loss
   * function was one of them, which is why a 1.2% disagreement survived a green
   * suite. Coverage that stops before the thing being measured is not coverage.
   */
  it.runIf(() => gpu !== null)("crossEntropy matches", () => {
    both("crossEntropy", APPROX, (B) =>
      B.crossEntropy(B.fromArray(fill([8, 16], (i) => ((i % 11) - 5) * 0.3), [8, 16]),
                     B.fromArray(fill([8], (i) => (3 * i + 2) % 16), [8])));
  });

  it.runIf(() => gpu !== null)("masking matches", () => {
    both("causalMask", EXACT, (B) => B.causalMask(8));
    both("maskedFill", EXACT, (B) =>
      B.maskedFill(B.fromArray(V, [1, 8, 16]),
                   B.fromArray(fill([1, 8, 16], (i) => (i % 3 === 0 ? 1 : 0)), [1, 8, 16]),
                   -42));
  });

  it.runIf(() => gpu !== null)("scalar-parameterised ops match", () => {
    both("scale", EXACT, (B) => B.scale(B.fromArray(V, [1, 8, 16]), 0.375));
    both("clamp", EXACT, (B) => B.clamp(B.fromArray(V, [1, 8, 16]), -0.5, 0.75));
    both("pow2", APPROX, (B) => B.pow(B.fromArray(V, [1, 8, 16]), 2));
    both("log", APPROX, (B) =>
      B.log(B.fromArray(fill([1, 8, 16], (i) => (i % 9) + 1), [1, 8, 16])));
  });

  it.runIf(() => gpu !== null)("shape-only ops match", () => {
    both("reshape", EXACT, (B) => B.reshape(B.fromArray(V, [1, 8, 16]), [8, 16]));
    both("clone", EXACT, (B) => B.clone(B.fromArray(V, [1, 8, 16])));
  });

  /*
   * The pool is unique to this backend and nothing above it knows it exists,
   * so its hazards are invisible to a comparison against cpu_ref -- which has
   * no pool. These probe it directly.
   *
   * The embedding backward is the reason: it asks for zeros and then
   * ACCUMULATES into the returned view by hand. If a recycled buffer arrives
   * carrying a previous tensor's values, the gradient is that garbage plus the
   * real contribution -- finite, plausible, and wrong.
   */
  it.runIf(() => gpu !== null)("zeros is zero even on a recycled buffer", () => {
    const g = gpu!;
    /* Dirty a buffer, release it, and ask for zeros of the same size: the pool
     * hands the same memory back. */
    const dirty = g.full([64], 7.5) as { data: ArrayLike<number> };
    expect(dirty.data[0]).toBe(7.5);
    const dev = dirty as unknown as { buffer: { release(hl: unknown): void } };
    (g as unknown as { hl: unknown }).hl;
    dev.buffer.release((g as unknown as { hl: unknown }).hl);

    const fresh = g.zeros([64]);
    const f = fresh.data as ArrayLike<number>;
    for (let i = 0; i < 64; i++) {
      if (f[i] !== 0) throw new Error(`zeros[${i}] = ${f[i]} on a recycled buffer`);
    }
  });

  it.runIf(() => gpu !== null)("a host write into a tensor is visible to the next kernel", () => {
    const g = gpu!;
    /* This is what the embedding backward does: allocate, write from the host,
     * then hand the tensor to an operation. If the write did not reach device
     * memory before the launch, the kernel reads stale values. */
    const t = g.zeros([64]) as { data: Float32Array };
    for (let i = 0; i < 64; i++) t.data[i] = i + 1;
    const doubled = g.add(t as never, t as never);
    const d = doubled.data as ArrayLike<number>;
    for (let i = 0; i < 64; i++) {
      if (d[i] !== 2 * (i + 1)) {
        throw new Error(`host write not visible: [${i}] = ${d[i]}, want ${2 * (i + 1)}`);
      }
    }
  });

  it.runIf(() => gpu !== null)("element-wise unaries keep rank", () => {
    const pos = fill([1, 8, 16], (i) => (i % 9) + 1);
    for (const [n, f] of [
      ["relu", (B: Backend, t: TensorData) => B.relu(t)],
      ["gelu", (B: Backend, t: TensorData) => B.gelu(t)],
      ["silu", (B: Backend, t: TensorData) => B.silu(t)],
      ["neg", (B: Backend, t: TensorData) => B.neg(t)],
      ["sqrt", (B: Backend, t: TensorData) => B.sqrt(t)],
      ["exp", (B: Backend, t: TensorData) => B.exp(t)],
    ] as const) {
      const src = n === "exp" ? V : pos;
      both(n, APPROX, (B) => f(B, B.fromArray(src, [1, 8, 16])));
    }
  });
});
