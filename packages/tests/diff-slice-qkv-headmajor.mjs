/* sliceQkvHeadMajor (RoPE-free) against a reference, EXACTLY, forward AND
 * backward.
 *
 * The fused op reads a grouped token-major qkvFlat [B*T, 3*H*D] and writes the
 * three head-major tensors [B, H, T, D] attention consumes, replacing sliceQkv
 * plus three permutes. It moves floats without touching them, so the correct
 * tolerance is ZERO — any difference is a wrong address. The backward scatters
 * each plane's head-major gradient into its disjoint columns of one qkvFlat
 * gradient; together the three planes must reproduce the input exactly.
 *
 * Shapes span head counts and feature widths that are and are not powers of two
 * (10 is the model's; 3/5/12 are there so a fix that only works for 10 fails),
 * with D a power of two throughout — the row-splitting shift requires it.
 *
 * Usage: node diff-slice-qkv-headmajor.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);

const CASES = [
  [1, 4, 10, 8], [2, 8, 10, 64], [8, 64, 10, 64],  /* the model's head count */
  [1, 4, 3, 4], [2, 16, 5, 32], [3, 8, 12, 16],
  [1, 8, 7, 16], [2, 4, 6, 8],
  [1, 4, 8, 8], [2, 32, 16, 64],
];

let failed = 0;
for (const [b, t, h, d] of CASES) {
  const hd = h * d, cols = 3 * hd, rows = b * t, n = rows * cols;
  const shape = `[${b},${t},${h},${d}]`.padEnd(16);

  /* Distinct per element so any misplacement is visible. */
  const src = Array.from({ length: n }, (_, i) => i + 1);
  const qkv = B.fromArray(src, [rows, cols]);
  const [qH, kH, vH] = B.sliceQkvHeadMajor(qkv, b, t, h, d);
  const got = [Array.from(qH.data), Array.from(kH.data), Array.from(vH.data)];

  /* Reference forward: out_plane[b][h][t][d] = qkvFlat[b*T+t][plane*H*D+h*D+d]. */
  let fbad = -1, fp = 0;
  for (let plane = 0; plane < 3 && fbad < 0; plane++)
    for (let bi = 0; bi < b && fbad < 0; bi++)
      for (let hi = 0; hi < h && fbad < 0; hi++)
        for (let ti = 0; ti < t && fbad < 0; ti++)
          for (let di = 0; di < d; di++) {
            const want = src[(bi * t + ti) * cols + plane * hd + hi * d + di];
            const idx = ((bi * h + hi) * t + ti) * d + di;
            if (got[plane][idx] !== want) { fbad = idx; fp = plane; break; }
          }

  /* Backward: scatter three distinct head-major grads into one qkvFlat grad. */
  const gsrc = [0, 1, 2].map((plane) =>
    Array.from({ length: rows * hd }, (_, i) => (plane + 1) * 1e6 + i + 1));
  const grads = gsrc.map((s) => B.fromArray(s, [b, h, t, d]));
  let into = B.zeros([rows, cols], "f32");
  for (let plane = 0; plane < 3; plane++)
    into = B.sliceQkvHeadMajorBackward(grads[plane], into, plane, b, t, h, d);
  const gotBwd = Array.from(into.data);

  /* Reference backward: qkvGrad[b*T+t][plane*H*D+h*D+d] = grad_plane[b][h][t][d]. */
  let bbad = -1;
  for (let bi = 0; bi < b && bbad < 0; bi++)
    for (let ti = 0; ti < t && bbad < 0; ti++)
      for (let plane = 0; plane < 3 && bbad < 0; plane++)
        for (let hi = 0; hi < h && bbad < 0; hi++)
          for (let di = 0; di < d; di++) {
            const want = gsrc[plane][((bi * h + hi) * t + ti) * d + di];
            const idx = (bi * t + ti) * cols + plane * hd + hi * d + di;
            if (gotBwd[idx] !== want) { bbad = idx; break; }
          }

  if (fbad < 0 && bbad < 0) {
    console.log(`  ${shape} ok        fwd ${n} + bwd ${n} exact`);
  } else {
    failed++;
    if (fbad >= 0)
      console.log(`  ${shape} FWD MISMATCH plane ${fp} at ${fbad}: got ${got[fp][fbad]}`);
    if (bbad >= 0)
      console.log(`  ${shape} BWD MISMATCH at ${bbad}: got ${gotBwd[bbad]}`);
  }
  for (const td of [qkv, qH, kH, vH, into, ...grads]) B.releaseGpuTensor?.(td);
  B.finishStepOps?.();
}

console.log(failed === 0
  ? `\nsliceQkvHeadMajor: ${CASES.length}/${CASES.length} shapes exact (fwd+bwd)`
  : `\nsliceQkvHeadMajor: ${failed}/${CASES.length} shapes WRONG`);
process.exit(failed === 0 ? 0 : 1);
