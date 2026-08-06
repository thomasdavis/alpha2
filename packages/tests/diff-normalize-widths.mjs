/* EVERY normalisation, at widths that are NOT a power of two.
 *
 * This test exists because a 640-wide layerNorm — the width the 105M model
 * actually runs — normalised over 512 of its 640 features, silently, for as
 * long as the kernel had existed. The reduction tree halved from elements/2
 * while the live count stayed even; at 640 it reaches 5, and the stride-2 step
 * that follows reduces slots 0..3 and orphans slot 4.
 *
 * Nothing caught it because every width this stack had ever been arbitrated at
 * was a power of two: the parity benchmark is 64 wide, the attention softmax
 * runs at T=64, and the cross-entropy chunk rounds to 1,024. The failure is not
 * subtle once you look for it — a layer norm's output row has mean 0 by
 * construction and this one came back at 0.80 — but no test asked.
 *
 * So the widths below are chosen to make the defect impossible to reintroduce,
 * not to be a round selection:
 *
 *   8, 16, 64, 512, 1024   powers of two — these always passed and must stay
 *                          exact, because the fix is only trustworthy if it
 *                          leaves the working case byte-identical
 *   20, 40, 640            elements = 2^k * 5, the family that failed. 20 is
 *                          the smallest, so a regression is readable by hand
 *   3, 5, 7                odd and tiny: the fold is the WHOLE reduction here,
 *                          with no halving loop after it at all
 *   96, 384, 1000          mixed factors, and 1000 is 8 * 125 — three odd
 *                          halvings deep, where a single-fold fix that only
 *                          handled the first one would break
 *
 * Tolerance is 2e-3 rather than an epsilon: this is a STRUCTURAL test, and the
 * failure it hunts is off by 20% of the data, not by rounding. A tight bound
 * would only make it flaky about f32 summation order.
 */
import { NativeHeliosBackend } from "../helios/dist/index.js";

const B = new NativeHeliosBackend(0);
const EPS = 1e-5;
const ROWS = 8;
const WIDTHS = [3, 5, 7, 8, 16, 20, 40, 64, 96, 384, 512, 640, 1000, 1024];

/* Deliberately not random. A ramp with a prime stride gives every column a
 * distinct value, so a dropped slot moves the mean by a specific readable
 * amount instead of hiding inside a statistical wobble. */
const ramp = (n, seed) => {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (((i * 37 + seed * 11) % 101) / 101) + 0.5;
  return a;
};

const refLayerNorm = (x, rows, w, weight, bias) => {
  const out = new Float32Array(rows * w);
  for (let r = 0; r < rows; r++) {
    let m = 0;
    for (let c = 0; c < w; c++) m += x[r * w + c];
    m /= w;
    let v = 0;
    for (let c = 0; c < w; c++) { const d = x[r * w + c] - m; v += d * d; }
    v /= w;
    const rstd = 1 / Math.sqrt(v + EPS);
    for (let c = 0; c < w; c++) out[r * w + c] = (x[r * w + c] - m) * rstd * weight[c] + bias[c];
  }
  return out;
};

const refRmsNorm = (x, rows, w, weight) => {
  const out = new Float32Array(rows * w);
  for (let r = 0; r < rows; r++) {
    let v = 0;
    for (let c = 0; c < w; c++) v += x[r * w + c] * x[r * w + c];
    const rstd = 1 / Math.sqrt(v / w + EPS);
    for (let c = 0; c < w; c++) out[r * w + c] = x[r * w + c] * rstd * weight[c];
  }
  return out;
};

const refSoftmax = (x, rows, w) => {
  const out = new Float32Array(rows * w);
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let c = 0; c < w; c++) mx = Math.max(mx, x[r * w + c]);
    let s = 0;
    for (let c = 0; c < w; c++) { const e = Math.exp(x[r * w + c] - mx); out[r * w + c] = e; s += e; }
    for (let c = 0; c < w; c++) out[r * w + c] /= s;
  }
  return out;
};

const worstDiff = (got, want) => {
  let worst = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > worst) worst = d;
  }
  return worst;
};

let failures = 0;
const report = (op, w, worst, extra) => {
  const ok = Number.isFinite(worst) && worst < 2e-3;
  if (!ok) failures++;
  console.log(`  ${op.padEnd(9)} w=${String(w).padStart(4)}  worst ${worst.toExponential(3)}` +
              `${extra ?? ""}  ${ok ? "ok" : "WRONG"}`);
};

for (const w of WIDTHS) {
  const x = ramp(ROWS * w, 1);
  const weight = ramp(w, 2);
  const bias = ramp(w, 3);
  const dx = () => B.fromArray(Array.from(x), [ROWS, w]);
  const dw = B.fromArray(Array.from(weight), [w]);
  const db = B.fromArray(Array.from(bias), [w]);

  const ln = B.layerNorm(dx(), dw, db, EPS).data;
  /* The invariant that named the bug in the first place, kept as its own
   * column: subtract the bias, divide by the weight, and a layer norm's row
   * must average to zero whatever else is true of it. */
  let rowMean = 0;
  for (let c = 0; c < w; c++) rowMean += (ln[c] - bias[c]) / weight[c];
  rowMean /= w;
  report("layerNorm", w, worstDiff(ln, refLayerNorm(x, ROWS, w, weight, bias)),
         `  row mean ${rowMean.toExponential(1).padStart(9)}`);

  report("rmsNorm", w, worstDiff(B.rmsNorm(dx(), dw, EPS).data,
                                 refRmsNorm(x, ROWS, w, weight)));
  report("softmax", w, worstDiff(B.softmax(dx(), -1).data, refSoftmax(x, ROWS, w)));

  B.finishStepOps?.();
}

console.log();
if (failures) {
  console.log(`FAIL — ${failures} case(s) disagree with the reference`);
  process.exit(1);
}
console.log(`ok — ${WIDTHS.length * 3} cases across ${WIDTHS.length} widths`);
