/* The save-stats layerNorm path must be BIT-IDENTICAL to the plain path.
 *
 * Forward: layerNormStats returns the same y as layerNorm, plus [mean, rstd]
 * per row. Backward: layerNormBackward(...,stats) LOADS those instead of
 * recomputing mean+variance (two of four reductions gone). A load returns the
 * same f32 the forward stored, so dx/dw/db must match the plain path to the
 * bit — any difference is a wrong address, a bad predicate, or a stat that was
 * not actually saved. Tolerance ZERO.
 *
 * Usage: HELIOS_VIDMEM=1 node diff-layernorm-stats.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);
const CASES = [[1536, 640], [512, 640], [64, 128], [3, 640], [7, 64]];
const eps = 1e-5;

let failed = 0;
for (const [rows, width] of CASES) {
  const n = rows * width;
  // Distinct-ish deterministic data with real spread (so mean/rstd matter).
  const x = Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 3 + (i % 13) - 6);
  const w = Array.from({ length: width }, (_, i) => 0.5 + Math.cos(i * 0.3) * 0.4);
  const b = Array.from({ length: width }, (_, i) => Math.sin(i * 0.2) * 0.1);
  const g = Array.from({ length: n }, (_, i) => Math.cos(i * 1.1) * 2);

  const xt = B.fromArray(x, [rows, width]);
  const wt = B.fromArray(w, [width]);
  const bt = B.fromArray(b, [width]);
  const gt = B.fromArray(g, [rows, width]);

  const yPlain = Array.from(B.layerNorm(xt, wt, bt, eps).data);
  const bp = B.layerNormBackward(xt, wt, gt, eps);
  const dxP = Array.from(bp.dx.data), dwP = Array.from(bp.dw.data), dbP = Array.from(bp.db.data);

  const { y, stats } = B.layerNormStats(xt, wt, bt, eps);
  const yStats = Array.from(y.data);
  const bs = B.layerNormBackward(xt, wt, gt, eps, stats);
  const dxS = Array.from(bs.dx.data), dwS = Array.from(bs.dw.data), dbS = Array.from(bs.db.data);

  const firstDiff = (a, c) => { for (let i = 0; i < a.length; i++) if (a[i] !== c[i]) return i; return -1; };
  const yd = firstDiff(yPlain, yStats);
  const dxd = firstDiff(dxP, dxS), dwd = firstDiff(dwP, dwS), dbd = firstDiff(dbP, dbS);
  const label = `[${rows}x${width}]`.padEnd(14);

  if (yd < 0 && dxd < 0 && dwd < 0 && dbd < 0) {
    console.log(`  ${label} ok        y + dx + dw + db bit-identical (stats vs plain)`);
  } else {
    failed++;
    if (yd >= 0)  console.log(`  ${label} Y  DIFF at ${yd}: plain ${yPlain[yd]} stats ${yStats[yd]}`);
    if (dxd >= 0) console.log(`  ${label} DX DIFF at ${dxd}: plain ${dxP[dxd]} stats ${dxS[dxd]}`);
    if (dwd >= 0) console.log(`  ${label} DW DIFF at ${dwd}: plain ${dwP[dwd]} stats ${dwS[dwd]}`);
    if (dbd >= 0) console.log(`  ${label} DB DIFF at ${dbd}: plain ${dbP[dbd]} stats ${dbS[dbd]}`);
  }
  for (const t of [xt, wt, bt, gt]) B.releaseGpuTensor?.(t);
  B.finishStepOps?.();
}

console.log(failed === 0
  ? `\nlayerNorm save-stats: ${CASES.length}/${CASES.length} bit-identical to plain`
  : `\nlayerNorm save-stats: ${failed}/${CASES.length} DIVERGED`);
process.exit(failed === 0 ? 0 : 1);
