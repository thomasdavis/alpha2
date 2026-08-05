/* sum and mean over EVERY axis, against the definition.
 *
 * reduceOverAxis has been changed twice with no test of its own: once to
 * release the temporaries it was leaking (a full transposed copy of its input,
 * on the path every bias and weight gradient takes), and once to stop
 * transposing at all when the reduced axis is already last. Both are the kind
 * of change that keeps a shape correct and moves the CONTENT.
 *
 * The expectation is arithmetic done here, not another backend's answer: a sum
 * over an axis is a sum, and computing it directly from the source array is a
 * definition rather than a second implementation. cpu_ref is used only to
 * generate the input, never the answer.
 *
 * Both routes are covered on purpose. reduceOverAxis picks between a transpose
 * -and-reduce and a matmul with a vector of ones on a condition involving the
 * axis length, whether it is a power of two, and whether it fits a block — so a
 * shape list that misses either arm proves half of what it looks like it does.
 *
 * Usage: node diff-reduce-axis.mjs [native|vulkan]
 */
import { NativeHeliosBackend, HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const which = process.argv[2] ?? "native";
const B = which === "vulkan" ? new HeliosBackend() : new NativeHeliosBackend(0);

/* Deterministic, and spread over several magnitudes so a lost term shows up
 * rather than hiding under a large one. */
const value = (i) => ((i * 37) % 101) / 7 - 5;

function reference(src, shape, axis, mean) {
  const k = axis < 0 ? shape.length + axis : axis;
  const inner = shape.slice(k + 1).reduce((a, b) => a * b, 1);
  const outer = shape.slice(0, k).reduce((a, b) => a * b, 1);
  const len = shape[k];
  const out = new Float64Array(outer * inner);
  for (let o = 0; o < outer; o++)
    for (let j = 0; j < inner; j++) {
      let acc = 0;
      for (let a = 0; a < len; a++) acc += src[(o * len + a) * inner + j];
      out[o * inner + j] = mean ? acc / len : acc;
    }
  return out;
}

/*
 * Shapes chosen to hit BOTH routes and the boundary between them.
 *
 * 512 is a power of two that fits a block, so axis 0 takes the transpose route;
 * 1025 is neither, so it takes the ones-matmul; 640 is the model's real width.
 * The last-axis cases are the ones the transpose elision changed.
 */
const CASES = [
  [[512, 640], 0], [[512, 640], 1],
  [[1025, 64], 0], [[1025, 64], 1],
  [[64, 640], 0], [[64, 640], 1],
  [[8, 64, 640], 0], [[8, 64, 640], 1], [[8, 64, 640], 2],
  [[4, 10, 64, 64], 1], [[4, 10, 64, 64], 3],
  [[7, 13], 0], [[7, 13], 1],
];

let bad = 0, ran = 0;
for (const [shape, axis] of CASES) {
  const n = shape.reduce((a, b) => a * b, 1);
  const src = Float64Array.from({ length: n }, (_, i) => value(i));
  const t = B.fromArray(Array.from(src, Number), shape);
  for (const mean of [false, true]) {
    const want = reference(src, shape, axis, mean);
    const got = (mean ? B.mean(t, axis) : B.sum(t, axis)).data;
    ran++;
    let worst = 0, at = -1;
    for (let i = 0; i < want.length; i++) {
      /* Relative, because a sum over 1,025 terms of magnitude ~5 lands near
       * 5,000 and f32 has ~7 digits. */
      const e = Math.abs(got[i] - want[i]) / Math.max(1, Math.abs(want[i]));
      if (e > worst) { worst = e; at = i; }
    }
    const ok = worst < 2e-5 && got.length === want.length;
    if (!ok) bad++;
    console.log(`  ${mean ? "mean" : "sum "} [${shape}] axis ${axis}` +
      `  ${ok ? "ok" : "FAIL"}  worst rel ${worst.toExponential(2)}` +
      (ok ? "" : `  at ${at}: got ${got[at]} want ${want[at]}` +
                 `  len ${got.length} want ${want.length}`));
  }
  B.releaseGpuTensor?.(t);
  B.finishStepOps?.();
}
console.log(`\nreduce over axis: ${ran - bad}/${ran} agree with the definition`);
if (bad) process.exit(1);
