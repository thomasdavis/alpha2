/**
 * Known-answer invariants for the Helios backend.
 *
 * These do not compare against a second implementation. Every expectation below
 * is derived from algebra, so a failure is unambiguous: the kernel is wrong, not
 * merely different.
 *
 * The method earned its place. X58 found a live correctness bug -- the gradient
 * norm was about half its true value for any tensor at or above 65,536 elements,
 * because the second reduction pass summed only the first 128 of 256 partials --
 * by feeding sum-of-squares an all-ones vector, where the answer must equal the
 * element count. Nothing failed and nothing warned; the reduction just returned a
 * plausible wrong number for months.
 *
 * Two rules carried over from that:
 *
 *  1. Prefer sizes that are NOT round numbers. A bug that halves everything is
 *     indistinguishable from a scaling constant, so 65536 -> 32768 reads as
 *     "half". 70000 -> 37232 names the mechanism.
 *  2. Straddle the boundaries where this class of bug lives: workgroup sizes
 *     (64/128/256), matmul tile sizes (16/32), the grid-stride reduction
 *     threshold (65536), and the vec4 path's dim % 4 == 0 gate.
 *
 * Run on the local software lane with:
 *   ALPHA_PARITY_ALLOW_SOFTWARE_DEVICE=1 HELIOS_DISABLE_COOP_MAT=1 \
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *   npx vitest run --root packages/tests known-answer
 */
import { describe, it, expect, afterAll } from "vitest";
import { HeliosBackend, destroyDevice, getDeviceInfo, getNative } from "@alpha/helios";

const allowSoftware = process.env.ALPHA_PARITY_ALLOW_SOFTWARE_DEVICE === "1";

function deviceUsable(): boolean {
  try {
    const info = getDeviceInfo(getNative());
    // deviceType 4 == CPU (lavapipe/llvmpipe)
    return allowSoftware || info.deviceType !== 4;
  } catch {
    return false;
  }
}

const usable = deviceUsable();
const maybe = usable ? describe : describe.skip;

let backend: HeliosBackend | null = null;
const b = (): HeliosBackend => (backend ??= new HeliosBackend());

afterAll(() => {
  try { destroyDevice(); } catch { /* teardown is best effort */ }
});

const ones = (n: number) => new Float32Array(n).fill(1);
const filled = (n: number, fn: (i: number) => number) => {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = fn(i);
  return a;
};

maybe("known-answer: reductions", () => {
  // sum-of-squares of all-ones must equal the element count, exactly.
  // Sizes straddle STRIDE_THRESHOLD (65536); the non-round ones are the
  // diagnostic ones -- see X58.
  for (const n of [65535, 65536, 65537, 70000, 98304, 131072, 200000, 262144]) {
    it(`sumOfSquares(ones[${n}]) === ${n}`, () => {
      const got = Number(b().sumOfSquares(b().fromArray(ones(n) as never, [n])).data[0]);
      expect(got).toBe(n);
    });
  }

  for (const n of [65535, 70000, 200000]) {
    it(`sum(ones[${n}]) === ${n}`, () => {
      const got = Number(b().sum(b().fromArray(ones(n) as never, [n])).data[0]);
      expect(Math.abs(got - n)).toBeLessThanOrEqual(n * 1e-6);
    });
  }
});

maybe("known-answer: matmul", () => {
  // ones(m,k) @ ones(k,n) == k in every element. Exact in f32 for k << 2^24.
  for (const [m, k, n] of [[31, 33, 29], [33, 65, 47], [129, 257, 65], [17, 1024, 19]] as const) {
    it(`ones(${m},${k}) @ ones(${k},${n}) === ${k}`, () => {
      const A = b().fromArray(ones(m * k) as never, [m, k]);
      const B = b().fromArray(ones(k * n) as never, [k, n]);
      const C = b().matmul(A, B).data as Float32Array;
      let worst = 0;
      for (let i = 0; i < m * n; i++) worst = Math.max(worst, Math.abs(C[i] - k));
      expect(worst).toBeLessThanOrEqual(k * 1e-6);
    });
  }

  // A @ I == A, elementwise.
  for (const [m, k] of [[31, 33], [65, 129]] as const) {
    it(`A(${m},${k}) @ I === A`, () => {
      const A = filled(m * k, (i) => ((i * 37) % 101) / 101 - 0.5);
      const I = new Float32Array(k * k);
      for (let i = 0; i < k; i++) I[i * k + i] = 1;
      const C = b().matmul(b().fromArray(A as never, [m, k]), b().fromArray(I as never, [k, k]))
        .data as Float32Array;
      let worst = 0;
      for (let i = 0; i < m * k; i++) worst = Math.max(worst, Math.abs(C[i] - A[i]));
      expect(worst).toBeLessThanOrEqual(1e-6);
    });
  }
});

maybe("known-answer: softmax", () => {
  // Every row of a softmax sums to 1. This is not an approximation and holds for
  // any input, any width, any backend.
  //
  // KNOWN FAILURE on the local software lane: rows sum to ~4 whenever
  // dim % 4 == 0 and the tensor has >= 4096 elements total -- see X60. The vec4
  // kernel is correct there; softmax_online and softmax_reg are not. Whether it
  // reproduces on a discrete GPU is unverified, which is exactly why this test
  // exists: it is cheap, device-independent, and will answer that question the
  // first time it runs on real hardware.
  for (const [rows, cols] of [[7, 33], [4, 128], [3, 129], [2, 2048], [1, 4096], [1, 12288]] as const) {
    it(`softmax rows sum to 1 [${rows},${cols}]`, () => {
      const x = filled(rows * cols, (i) => Math.sin(i * 0.7) * 4);
      const y = b().softmax(b().fromArray(x as never, [rows, cols]), -1).data as Float32Array;
      for (let r = 0; r < rows; r++) {
        let s = 0;
        for (let c = 0; c < cols; c++) s += y[r * cols + c];
        expect(Math.abs(s - 1)).toBeLessThan(2e-5);
      }
    });
  }

  // A uniform row must produce exactly 1/n everywhere.
  for (const cols of [33, 128, 1023] as const) {
    it(`softmax uniform [1,${cols}] === 1/${cols}`, () => {
      const y = b().softmax(b().fromArray(new Float32Array(cols).fill(3.25) as never, [1, cols]), -1)
        .data as Float32Array;
      let worst = 0;
      for (let i = 0; i < cols; i++) worst = Math.max(worst, Math.abs(y[i] - 1 / cols));
      expect(worst).toBeLessThan(2e-6);
    });
  }
});

maybe("known-answer: transpose", () => {
  // T(T(x)) must be bitwise identical to x -- no arithmetic is performed.
  for (const [r, c] of [[33, 47], [129, 65], [256, 128]] as const) {
    it(`T(T(x)) === x [${r},${c}]`, () => {
      const x = filled(r * c, (i) => i * 0.5 - 3);
      const back = b().transpose(b().transpose(b().fromArray(x as never, [r, c]))).data as Float32Array;
      let mismatches = 0;
      for (let i = 0; i < r * c; i++) if (back[i] !== x[i]) mismatches++;
      expect(mismatches).toBe(0);
    });
  }
});
