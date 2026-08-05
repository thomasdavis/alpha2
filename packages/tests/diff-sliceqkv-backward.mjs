/* sliceQkv's backward, against the definition.
 *
 * q, k and v are three disjoint column ranges of one projection, so the
 * gradient of the projection is their three gradients laid side by side —
 * concatenation, not addition. The backward now says so directly: each of the
 * three writes its own third of the destination the tape offers and returns it.
 *
 * That is a change to WHO WRITES WHAT and in what order, on a path where the
 * three closures now share a buffer, so it is exactly the kind of change that
 * keeps every shape right and moves the content. The expectation below is the
 * concatenation, computed here.
 *
 * Usage: HELIOS_VIDMEM=1 node diff-sliceqkv-backward.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { Variable, Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { sliceQkv, scale, add } from "/workspace/alpha2/packages/autograd/dist/index.js";

const B = new NativeHeliosBackend(0);
let bad = 0, ran = 0;

for (const [rows, D] of [[8, 4], [768, 640], [3, 64], [64, 128]]) {
  const n = rows * 3 * D;
  const src = Float64Array.from({ length: n }, (_, i) => ((i * 17) % 61) / 5 - 6);
  const a = new Variable(B.fromArray(Array.from(src, Number), [rows, 3 * D]), true);
  const tape = new Tape();
  const ctx = { backend: B, tape };
  const [q, k, v] = sliceQkv(ctx, a);

  /*
   * Distinct coefficients per branch, deliberately. With equal weights a
   * backward that wrote the WRONG third would still produce a symmetric answer
   * and pass; three different scales make each column range's value name which
   * branch produced it.
   */
  const out = add(ctx, add(ctx, scale(ctx, q, 2), scale(ctx, k, 3)), scale(ctx, v, 5));
  tape.backward(out, B, undefined, B.ones(out.data.shape, "f32"));

  const want = new Float64Array(rows * 3 * D);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < 3 * D; c++)
      want[r * 3 * D + c] = c < D ? 2 : c < 2 * D ? 3 : 5;

  const got = a.grad.data;
  ran++;
  let worst = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (e > worst) { worst = e; at = i; }
  }
  const ok = worst < 1e-6 && got.length === want.length;
  if (!ok) bad++;
  console.log(`  [${rows},${3 * D}]`.padEnd(16) + `${ok ? "ok" : "FAIL"}  worst ${worst.toExponential(2)}` +
    (ok ? "" : `  at ${at} (col ${at % (3 * D)}): got ${got[at]} want ${want[at]}`));
  B.finishStepOps?.();
}
console.log(bad ? `\n${bad}/${ran} WRONG` : `\nsliceQkv backward: ${ran}/${ran} tile the projection correctly`);
if (bad) process.exit(1);
