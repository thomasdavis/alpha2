/* Does the fused layerNormBackward kernel agree with cpu_ref, element by element?
 *
 * The gate is green, but a green gate is what three wrong gradients looked like
 * this session: geluBackward passed everything except a direct comparison, and
 * the composed backwards before it were wrong in ways the loss absorbed. So
 * compare THIS operation against the definition, at shapes the model actually
 * uses, and report the worst element rather than a pass/fail on a norm.
 *
 * All three outputs, because they fail independently:
 *   dx  the kernel's own arithmetic, four reductions deep
 *   dw  sum_rows(g * xhat) -- wrong if xhat is wrong, which dx might survive
 *   db  sum_rows(g) -- touches neither the kernel nor xhat, so if THIS is wrong
 *       the reduction down the row axis is, and that is a different bug
 *
 * The tolerance is loose enough for a GPU reduction's reassociation and tight
 * enough that a mis-indexed tensor cannot hide: a wrong element is wrong by
 * order-one amounts, not by 1e-5. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";

const N = new NativeHeliosBackend(0);
const C = new CpuRefBackend();
const TOL = 2e-4;

/* Deterministic and asymmetric: a tensor of equal values would pass a kernel
 * that ignored its indices, and a mean-zero one would hide a mean bug. */
const fill = (n, seed, scale = 1) =>
  Array.from({ length: n }, (_, i) => (Math.sin(i * 0.7 + seed) * 2 + 0.35) * scale);
const size = (s) => s.reduce((a, b) => a * b, 1);

function host(B, t) {
  if (B.syncGpu) B.syncGpu();
  else if (B.flushAndWait) B.flushAndWait();
  const d = t.data;
  return Array.from({ length: size(t.shape) }, (_, i) => d[i]);
}

/* cpu_ref has no layerNormBackward either, so compute the definition directly
 * rather than through a fallback that could share a bug with the thing under
 * test. This IS the specification. */
function reference(xArr, wArr, gArr, rows, width, eps) {
  const dx = new Float64Array(rows * width);
  const dw = new Float64Array(width);
  const db = new Float64Array(width);
  for (let r = 0; r < rows; r++) {
    const o = r * width;
    let mu = 0;
    for (let j = 0; j < width; j++) mu += xArr[o + j];
    mu /= width;
    let va = 0;
    for (let j = 0; j < width; j++) va += (xArr[o + j] - mu) ** 2;
    va /= width;
    const rstd = 1 / Math.sqrt(va + eps);
    const xhat = new Float64Array(width);
    const dxhat = new Float64Array(width);
    for (let j = 0; j < width; j++) {
      xhat[j] = (xArr[o + j] - mu) * rstd;
      dxhat[j] = gArr[o + j] * wArr[j];
      db[j] += gArr[o + j];
      dw[j] += gArr[o + j] * xhat[j];
    }
    let m1 = 0, m2 = 0;
    for (let j = 0; j < width; j++) { m1 += dxhat[j]; m2 += dxhat[j] * xhat[j]; }
    m1 /= width; m2 /= width;
    for (let j = 0; j < width; j++)
      dx[o + j] = rstd * (dxhat[j] - m1 - xhat[j] * m2);
  }
  return { dx, dw, db };
}

/* [B,T,C] with C=64 is the real model shape; the others check that the row
 * count and the width vary independently. A kernel keyed on width alone would
 * pass the first and fail the third. */
const CASES = [
  { shape: [1, 8, 64], eps: 1e-5 },
  { shape: [2, 32, 64], eps: 1e-5 },
  { shape: [128, 32, 64], eps: 1e-5 },
  { shape: [4, 16, 128], eps: 1e-5 },
  { shape: [3, 5, 32], eps: 1e-3 },
];

console.log(`\nfused layerNormBackward vs the definition, tolerance ${TOL}\n`);
console.log(`  dx absolute (tol ${TOL}); dw and db RELATIVE to the sum's magnitude (tol 1e-5)\n`);
console.log("shape                 eps        max|dx|      rel|dw|      rel|db|   verdict");
let bad = 0;
for (const { shape, eps } of CASES) {
  const width = shape[shape.length - 1];
  const rows = size(shape) / width;
  const xArr = fill(size(shape), 1);
  const wArr = fill(width, 2, 0.5);
  const gArr = fill(size(shape), 3, 0.1);

  const x = N.fromArray(xArr, shape);
  const w = N.fromArray(wArr, [width]);
  const g = N.fromArray(gArr, shape);

  let got;
  try {
    got = N.layerNormBackward(x, w, g, eps);
  } catch (e) {
    console.log(`${String(shape).padEnd(20)} ${String(eps).padEnd(8)} ERROR ${e.message.slice(0, 50)}`);
    bad++;
    continue;
  }
  const ref = reference(xArr, wArr, gArr, rows, width, eps);
  const worst = (a, b) => {
    let m = 0;
    for (let i = 0; i < b.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (Number.isFinite(d)) { if (d > m) m = d; }
      else m = Infinity;
    }
    return m;
  };
  /*
   * dx gets an ABSOLUTE tolerance and dw/db a RELATIVE one, because they are
   * sums of different lengths and it would be dishonest to hold them to the
   * same number.
   *
   * dx is per-row: each element comes from reductions over `width` terms, 64 of
   * them, so f32 drift stays far below 2e-4 and an absolute bound is tight.
   * dw and db sum down the ROWS -- 4,096 of them at batch 128 -- and a
   * random-walk f32 error over N terms of a sum of magnitude S grows like
   * sqrt(N)*eps*S: sqrt(4096) * 6e-8 * 143 = 5.5e-4, against the 5.13e-4
   * actually observed. Holding a 4,096-term sum to the same absolute bound as a
   * 64-term one measures the arithmetic's precision, not its correctness.
   *
   * Relative, not loosened: a mis-indexed or dropped term is wrong by order one
   * relative to the sum, which 1e-5 still catches with four orders to spare.
   */
  const rel = (got, want) => {
    let m = 0, s = 0;
    for (let i = 0; i < want.length; i++) s = Math.max(s, Math.abs(want[i]));
    m = worst(got, want);
    return s > 0 ? m / s : m;
  };
  const RTOL = 1e-5;
  const wx = worst(host(N, got.dx), ref.dx);
  const ww = rel(host(N, got.dw), ref.dw);
  const wb = rel(host(N, got.db), ref.db);
  const ok = wx <= TOL && ww <= RTOL && wb <= RTOL;
  if (!ok) bad++;
  console.log(
    `${String(shape).padEnd(20)} ${String(eps).padEnd(8)} ` +
    `${wx.toExponential(2).padStart(11)} ${ww.toExponential(2).padStart(12)} ` +
    `${wb.toExponential(2).padStart(12)}   ${ok ? "ok" : "**WRONG**"}`);
}

/* And the shape contract, which a numeric comparison cannot see: dw and db must
 * match the WEIGHT, not the input. Returning [rows,width] here would broadcast
 * silently into the optimizer and corrupt training with finite numbers. */
{
  const shape = [2, 8, 64], width = 64;
  const r = N.layerNormBackward(
    N.fromArray(fill(size(shape), 1), shape),
    N.fromArray(fill(width, 2, 0.5), [width]),
    N.fromArray(fill(size(shape), 3, 0.1), shape), 1e-5);
  const okShape =
    String(r.dx.shape) === String(shape) &&
    String(r.dw.shape) === String([width]) &&
    String(r.db.shape) === String([width]);
  if (!okShape) bad++;
  console.log(`\nshapes: dx ${r.dx.shape} dw ${r.dw.shape} db ${r.db.shape}   ${okShape ? "ok" : "**WRONG**"}`);
}

console.log(bad ? `\n${bad} case(s) WRONG` : "\nevery case agrees with the definition");
process.exit(bad ? 1 : 0);
