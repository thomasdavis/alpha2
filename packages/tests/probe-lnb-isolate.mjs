/* Two failures, and they are not the same failure. Separate them.
 *
 * diff-layernorm-backward reports:
 *   128,32,64  dx WRONG  dw WRONG  db 3.03e-4
 *   4,16,128   dx WRONG  dw WRONG  db ok
 *   3,5,32     dx ok     dw WRONG  db WRONG
 *
 * db is sum(g, 0). It never touches the kernel and never touches xhat, so the
 * third row cannot be the kernel's fault -- something is wrong with the reduction
 * down the ROW axis. And the third row's dx is right, so whatever breaks dx in the
 * first two is a different thing.
 *
 * A: sum down axis 0, against the definition, sweeping the row count. 15 rows is
 *    not a power of two and 4096 is large; if the tree assumes one it will say so.
 * B: the kernel's dx alone, sweeping width and rows independently, so "width 128
 *    is broken" and "4096 rows is broken" cannot be confused for each other. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const size = (s) => s.reduce((a, b) => a * b, 1);
const fill = (n, seed, scale = 1) =>
  Array.from({ length: n }, (_, i) => (Math.sin(i * 0.7 + seed) * 2 + 0.35) * scale);
function host(B, t) { B.syncGpu?.(); const d = t.data; return Array.from({ length: size(t.shape) }, (_, i) => d[i]); }
const worst = (a, b) => { let m = 0; for (let i = 0; i < b.length; i++) { const d = Math.abs(a[i] - b[i]); m = Number.isFinite(d) ? Math.max(m, d) : Infinity; } return m; };

console.log("\nA. sum(a, axis 0) vs the definition — is the ROW-axis reduction sound?\n");
console.log("  rows  width      max|diff|   verdict");
for (const [rows, width] of [[8,32],[15,32],[16,32],[31,64],[32,64],[64,64],[100,64],[1024,64],[1025,64],[4096,64]]) {
  const a = fill(rows * width, 1);
  const ref = new Float64Array(width);
  for (let r = 0; r < rows; r++) for (let j = 0; j < width; j++) ref[j] += a[r * width + j];
  let got;
  try { got = host(N, N.sum(N.fromArray(a, [rows, width]), 0, false)); }
  catch (e) { console.log(`  ${String(rows).padStart(4)}  ${String(width).padStart(5)}      ERROR ${e.message.slice(0,40)}`); continue; }
  /* Scale the tolerance with the row count: summing 4096 f32 values reassociated
   * differently is allowed to drift, being wrong is not. */
  const tol = 1e-6 * rows;
  const w = worst(got, ref);
  console.log(`  ${String(rows).padStart(4)}  ${String(width).padStart(5)}   ${w.toExponential(2).padStart(12)}   ${w <= tol ? "ok" : "**WRONG**"}`);
}

console.log("\nB. the kernel's dx alone — width and rows swept independently\n");
console.log("  rows  width       max|dx|   verdict");
function refDx(x, w, g, rows, width, eps) {
  const dx = new Float64Array(rows * width);
  for (let r = 0; r < rows; r++) {
    const o = r * width;
    let mu = 0; for (let j = 0; j < width; j++) mu += x[o + j]; mu /= width;
    let va = 0; for (let j = 0; j < width; j++) va += (x[o + j] - mu) ** 2; va /= width;
    const rstd = 1 / Math.sqrt(va + eps);
    const xh = [], dh = [];
    for (let j = 0; j < width; j++) { xh[j] = (x[o + j] - mu) * rstd; dh[j] = g[o + j] * w[j]; }
    let m1 = 0, m2 = 0;
    for (let j = 0; j < width; j++) { m1 += dh[j]; m2 += dh[j] * xh[j]; }
    m1 /= width; m2 /= width;
    for (let j = 0; j < width; j++) dx[o + j] = rstd * (dh[j] - m1 - xh[j] * m2);
  }
  return dx;
}
for (const [rows, width] of [[8,64],[64,64],[256,64],[1024,64],[4096,64],[8,32],[8,128],[8,256],[64,128],[15,32]]) {
  const shape = [rows, width];
  const x = fill(rows * width, 1), w = fill(width, 2, 0.5), g = fill(rows * width, 3, 0.1);
  let got;
  try { got = host(N, N.layerNormBackward(N.fromArray(x, shape), N.fromArray(w, [width]), N.fromArray(g, shape), 1e-5).dx); }
  catch (e) { console.log(`  ${String(rows).padStart(4)}  ${String(width).padStart(5)}       ERROR ${e.message.slice(0,40)}`); continue; }
  const wd = worst(got, refDx(x, w, g, rows, width, 1e-5));
  console.log(`  ${String(rows).padStart(4)}  ${String(width).padStart(5)}   ${wd.toExponential(2).padStart(11)}   ${wd <= 2e-4 ? "ok" : "**WRONG**"}`);
}
