/* transpose(1,2) on [B,T,H,D] against a reference permutation, EXACTLY.
 *
 * The permute kernel used to require H, T and D all to be powers of two and
 * fell back to a host copy otherwise. A 10-head model is not a power of two,
 * so 105M never took the kernel at all; freeing it means the kernel now runs
 * on shapes it has never run on, and "the gates are green" does not cover
 * that — the gates do not build a 10-head model.
 *
 * This compares element for element. A permutation moves floats without
 * touching them, so the correct tolerance is ZERO: any difference is a wrong
 * address, not rounding. That is what makes this a better check than loss
 * agreement, which is what was reachable otherwise (cpu_ref returns NaN at
 * these shapes and the Vulkan backend runs its matmuls in f16, so neither
 * gives an exact comparator).
 *
 * Usage: node diff-permute-heads.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);

/* Head counts on both sides of the old guard, plus sequence lengths and
 * feature widths that are and are not powers of two. 10 is the one the model
 * actually uses; 3, 5 and 12 are there so a fix that happens to work for 10
 * does not pass. */
const CASES = [
  [1, 4, 10, 8], [2, 8, 10, 64], [8, 64, 10, 64],   /* the model's shape */
  [1, 4, 3, 4], [2, 16, 5, 32], [3, 8, 12, 16],
  [1, 8, 7, 5], [2, 4, 6, 3],                        /* D not a power of two */
  [1, 4, 8, 8], [2, 32, 16, 64],                     /* powers of two still work */
];

let failed = 0;
for (const [b, t, h, d] of CASES) {
  const n = b * t * h * d;
  /* Distinct per element, so a swapped pair of axes cannot coincide. */
  const src = Array.from({ length: n }, (_, i) => i + 1);
  const a = B.fromArray(src, [b, t, h, d]);
  const out = B.transpose(a, 1, 2);
  const got = Array.from(out.data);

  /* Reference: out[b][h][t][d] = in[b][t][h][d]. */
  const want = new Array(n);
  for (let bi = 0; bi < b; bi++)
    for (let ti = 0; ti < t; ti++)
      for (let hi = 0; hi < h; hi++)
        for (let di = 0; di < d; di++)
          want[((bi * h + hi) * t + ti) * d + di] = src[((bi * t + ti) * h + hi) * d + di];

  let bad = -1;
  for (let i = 0; i < n; i++) if (got[i] !== want[i]) { bad = i; break; }
  const shape = `[${b},${t},${h},${d}]`.padEnd(16);
  if (bad < 0) {
    console.log(`  ${shape} ok        ${n} elements, exact`);
  } else {
    failed++;
    console.log(`  ${shape} MISMATCH  first at ${bad}: got ${got[bad]}, want ${want[bad]}`);
  }
  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(out); B.finishStepOps?.();
}

console.log(failed === 0
  ? `\npermute: ${CASES.length}/${CASES.length} shapes exact`
  : `\npermute: ${failed}/${CASES.length} shapes WRONG`);
process.exit(failed === 0 ? 0 : 1);
