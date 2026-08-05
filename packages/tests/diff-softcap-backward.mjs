/* softCap's backward, against the definition.
 *
 * d/dx [c*tanh(x/c)] = 1 - tanh^2(x/c), so the gradient is g*(1 - t^2).
 * ops.ts composes it from SIX operations when the backend has no fused form —
 * recompute softCap, scale, mul, ones, sub, mul — each a full pass over the
 * attention scores, once per layer.
 *
 * The expectation is the closed form evaluated here in double precision, not
 * the composition's output. Tolerance is relative and loose enough for the
 * hardware's exp2/rcp approximations (MUFU is ~1 ulp for exp2 and ~1 for rcp)
 * and tight enough to catch a wrong constant, which is the failure this family
 * actually has: gelu's two folded constants were once swapped and silu's
 * negated, and both produced plausible wrong numbers.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);
/* ONE case per process when asked. The GEMM probe in this directory reported a
 * 4x layout gap that was entirely an artifact of measuring cases in sequence,
 * so a failure that only appears in a multi-case run is a hypothesis about the
 * RUN until it survives being run alone. */
const CAPS = process.env.CAP ? [Number(process.env.CAP)] : [30, 4, 1.5];
const SHAPES = process.env.SHAPE
  ? [process.env.SHAPE.split(",").map(Number)]
  : [[64], [8, 10, 64, 64], [512, 640], [7, 13]];

let bad = 0, ran = 0;
for (const cap of CAPS) {
  for (const shape of SHAPES) {
    const n = shape.reduce((a, b) => a * b, 1);
    /* Spread over |x| well past the cap, where tanh saturates and the gradient
     * goes to zero — the region a wrong constant shows up in first. */
    const x = Float64Array.from({ length: n }, (_, i) => ((i * 37) % 401) / 4 - 50);
    const g = Float64Array.from({ length: n }, (_, i) => ((i * 17) % 29) / 7 - 2);
    const want = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = Math.tanh(x[i] / cap);
      want[i] = g[i] * (1 - t * t);
    }
    const tx = B.fromArray(Array.from(x, Number), shape);
    const tg = B.fromArray(Array.from(g, Number), shape);
    if (!B.softCapBackward) { console.log("  softCapBackward absent"); process.exit(0); }
    const got = B.softCapBackward(tg, tx, cap).data;
    ran++;
    let worst = 0, at = -1;
    for (let i = 0; i < n; i++) {
      const e = Math.abs(got[i] - want[i]) / Math.max(1e-3, Math.abs(want[i]));
      if (e > worst) { worst = e; at = i; }
    }
    const ok = worst < 1e-4;
    if (!ok) bad++;
    console.log(`  cap ${String(cap).padEnd(4)} [${shape}]`.padEnd(30) +
      `${ok ? "ok" : "FAIL"}  worst rel ${worst.toExponential(2)}` +
      (ok ? "" : `  at ${at}: got ${got[at]} want ${want[at]} x ${x[at]}`));
    B.releaseGpuTensor?.(tx); B.releaseGpuTensor?.(tg);
    B.finishStepOps?.();
  }
}
console.log(bad ? `\n${bad}/${ran} WRONG` : `\nsoftCap backward: ${ran}/${ran} agree with the definition`);
if (bad) process.exit(1);
