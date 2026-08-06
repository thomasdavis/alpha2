/* cross-entropy's backward, against the definition.
 *
 * d/dlogits [ -log softmax(x)[t] ] = (softmax(x) - onehot(t)) * gScalar / N.
 *
 * ops.ts composes it, when the backend has no fused form, by building an
 * N x C ONE-HOT ARRAY IN JAVASCRIPT — at this model's shape that is
 * 512 x 12,288 floats, 25 MB, constructed element by element on the host,
 * uploaded across PCIe, subtracted and scaled. It is the largest single
 * host-side object a step creates.
 *
 * The expectation is the closed form in double precision. cpu_ref supplies
 * nothing: a softmax is a softmax.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);
const CASES = process.env.SHAPE
  ? [process.env.SHAPE.split(",").map(Number)]
  : [[4, 8], [3, 7], [64, 1024], [512, 12288], [2, 129], [16, 640]];

let bad = 0, ran = 0;
for (const [N, C] of CASES) {
  const x = Float64Array.from({ length: N * C }, (_, i) => ((i * 31) % 197) / 11 - 9);
  const tgt = Int32Array.from({ length: N }, (_, r) => (r * 7 + 3) % C);
  const gScalar = 0.75;

  const want = new Float64Array(N * C);
  for (let r = 0; r < N; r++) {
    const row = Array.from({ length: C }, (_, c) => x[r * C + c]);
    const m = Math.max(...row);
    const e = row.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    for (let c = 0; c < C; c++)
      want[r * C + c] = (e[c] / s - (c === tgt[r] ? 1 : 0)) * gScalar / N;
  }

  const tx = B.fromArray(Array.from(x, Number), [N, C]);
  const tt = B.fromArray(Array.from(tgt, Number), [N]);
  if (!B.crossEntropyBackward) { console.log("  crossEntropyBackward absent"); process.exit(0); }
  const g = B.fromArray([gScalar], [1]);
  const got = B.crossEntropyBackward(tx, tt, g).data;
  ran++;
  let worst = 0, at = -1;
  for (let i = 0; i < N * C; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (e > worst) { worst = e; at = i; }
  }
  /*
   * Absolute, because most entries are tiny probabilities and the one at the
   * target is near -1/N; a relative bound would be dominated by the zeros.
   *
   * 2e-6 rather than the 1e-8 the exact cases reach: the kernel takes ONE
   * reciprocal of the row sum and multiplies, where the definition divides per
   * class. MUFU.RCP is accurate to about an ulp, so the probabilities carry
   * ~1e-7 relative error by construction. That is a deliberate trade -- twelve
   * thousand multiplies against twelve thousand divides on a quarter-rate pipe
   * -- and it is four orders of magnitude below the errors a wrong reduction or
   * a misplaced barrier produces.
   */
  const ok = worst < 2e-6 && got.length === want.length;
  if (!ok) bad++;
  console.log(`  [${N},${C}]`.padEnd(16) + `${ok ? "ok" : "FAIL"}  worst abs ${worst.toExponential(2)}` +
    (ok ? "" : `  at ${at} (row ${Math.floor(at / C)}, col ${at % C}, target ${tgt[Math.floor(at / C)]}): got ${got[at]} want ${want[at]}`));
  B.releaseGpuTensor?.(tx); B.releaseGpuTensor?.(tt); B.releaseGpuTensor?.(g);
  B.finishStepOps?.();
}
/*
 * THE FORWARD, at the same widths.
 *
 * Not scope creep: the backward's failures at 7, 129 and 640 were the block
 * width handed to the reduction tree, and the forward computes its maximum and
 * its sum through the SAME width from the SAME function. It was returning a
 * loss reduced over part of each row for every non-power-of-two vocabulary
 * under 1,024, and nothing noticed, because the only vocabulary this stack runs
 * is 12,288 and 12,288 rounds to 1,024. A bug found through one door should be
 * closed at every door it opens onto.
 */
let fbad = 0, fran = 0;
for (const [N, C] of CASES) {
  const x = Float64Array.from({ length: N * C }, (_, i) => ((i * 31) % 197) / 11 - 9);
  const tgt = Int32Array.from({ length: N }, (_, r) => (r * 7 + 3) % C);
  let want = 0;
  for (let r = 0; r < N; r++) {
    const row = Array.from({ length: C }, (_, c) => x[r * C + c]);
    const m = Math.max(...row);
    const s = row.reduce((a, v) => a + Math.exp(v - m), 0);
    want += -(row[tgt[r]] - m - Math.log(s));
  }
  want /= N;

  const tx = B.fromArray(Array.from(x, Number), [N, C]);
  const tt = B.fromArray(Array.from(tgt, Number), [N]);
  const got = B.crossEntropy(tx, tt).data[0];
  fran++;
  const err = Math.abs(got - want) / Math.max(1, Math.abs(want));
  const ok = err < 1e-5;
  if (!ok) fbad++;
  console.log(`  fwd [${N},${C}]`.padEnd(20) + `${ok ? "ok" : "FAIL"}  rel ${err.toExponential(2)}` +
    (ok ? "" : `  got ${got} want ${want}`));
  B.releaseGpuTensor?.(tx); B.releaseGpuTensor?.(tt);
  B.finishStepOps?.();
}

console.log(bad || fbad ? `\n${bad}/${ran} backward, ${fbad}/${fran} forward WRONG`
  : `\ncross-entropy: ${ran}/${ran} backward and ${fran}/${fran} forward agree with the definition`);
if (bad || fbad) process.exit(1);
