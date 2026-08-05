/* Which operation does Vulkan get wrong once the batch is bigger than one?
 *
 * The loss is 4.1834 at batch 1 -- agreeing with cpu_ref and with the native
 * backend -- and 4.1795 from batch 2 onward, and it never recovers. A break at
 * exactly the point a dimension appears is a shape bug, and the native backend
 * had one of the same family: the causal mask was read flat, so head 0 masked
 * correctly and every later head read past the end of the mask.
 *
 * So compare operation by operation at a BATCHED shape, against cpu_ref, and
 * report the first that disagrees. Comparing whole models would only say that
 * something is wrong, which is already known.
 *
 * Each case runs the same inputs through both backends and reports the largest
 * absolute difference. The tolerance is loose enough to ignore the reassociation
 * a GPU reduction does and tight enough that a wrong element cannot hide: a
 * mis-indexed tensor is wrong by order-one amounts, not by 1e-6. */
import { HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";

const V = new HeliosBackend();
const C = new CpuRefBackend();
const TOL = 1e-3;

/* Deterministic, and with no symmetry to hide behind: a tensor of equal values
 * would pass a transpose that did nothing. */
const fill = (n, seed) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7 + seed) * 2);
const size = (s) => s.reduce((a, b) => a * b, 1);

function mk(B, shape, seed) {
  return B.fromArray(fill(size(shape), seed), shape);
}
function host(B, t) {
  if (B.flushAndWait) B.flushAndWait();
  else if (B.syncGpu) B.syncGpu();
  const d = t.data;
  return Array.from({ length: size(t.shape) }, (_, i) => d[i]);
}

const CASES = [];
const one = (name, shapes, run) => CASES.push({ name, shapes, run });

/* Batched everywhere: [B,T,C] with B=2, and the head-major [B,H,T,T] the
 * attention block actually produces. */
one("softmax [2,4,8,8]", [[2, 4, 8, 8]], (B, [a]) => B.softmax(a, -1));
one("layerNorm [2,8,16]", [[2, 8, 16], [16], [16]], (B, [a, w, b]) => B.layerNorm(a, w, b, 1e-5));
one("rmsNorm [2,8,16]", [[2, 8, 16], [16]], (B, [a, w]) => B.rmsNorm(a, w, 1e-5));
/* Two sizes on purpose: matmul routes to the CPU below a FLOP threshold, so a
 * small case tests a different code path from the one the model uses. The large
 * shape is the real qkv projection, [B,T,C] x [C,3C]. */
one("matmul small [2,8,16]x[16,32]", [[2, 8, 16], [16, 32]], (B, [a, b]) => B.matmul(a, b));
one("matmul REAL [2,32,64]x[64,192]", [[2, 32, 64], [64, 192]], (B, [a, b]) => B.matmul(a, b));
one("matmul REAL [4,32,64]x[64,256]", [[4, 32, 64], [64, 256]], (B, [a, b]) => B.matmul(a, b));
one("matmul batched [2,4,8,16]x[2,4,16,8]", [[2, 4, 8, 16], [2, 4, 16, 8]], (B, [a, b]) => B.matmul(a, b));
one("transpose [2,4,8,16]", [[2, 4, 8, 16]], (B, [a]) => B.transpose(a));
one("sum axis -1 [2,4,8,8]", [[2, 4, 8, 8]], (B, [a]) => B.sum(a, -1, true));
one("mean axis -1 [2,8,16]", [[2, 8, 16]], (B, [a]) => B.mean(a, -1, true));
one("add broadcast [2,8,16]+[16]", [[2, 8, 16], [16]], (B, [a, b]) => B.add(a, b));
one("mul broadcast [2,4,8,8]*[8,8]", [[2, 4, 8, 8], [8, 8]], (B, [a, b]) => B.mul(a, b));
one("maskedFill [2,4,8,8] by [8,8]", [[2, 4, 8, 8], [8, 8]], (B, [a, m]) =>
  B.maskedFill(a, B.causalMask(8), -1e30));
one("causalMask 8", [], (B) => B.causalMask(8));
one("slice [2,8,16] -> [1,8,16]", [[2, 8, 16]], (B, [a]) => B.slice(a, [0, 0, 0], [1, 8, 16]));
one("cat axis 0 [2,8,16]x2", [[2, 8, 16], [2, 8, 16]], (B, [a, b]) => B.cat([a, b], 0));
one("cat axis -1 [2,8,16]x2", [[2, 8, 16], [2, 8, 16]], (B, [a, b]) => B.cat([a, b], -1));
one("gelu [2,8,16]", [[2, 8, 16]], (B, [a]) => B.gelu(a));
one("crossEntropy [2,8,16]", [[2, 8, 16], [16]], (B, [a]) => {
  const t = B.fromArray(Array.from({ length: 16 }, (_, i) => i % 16), [16]);
  return B.crossEntropy(B.reshape(a, [16, 16]), t);
});
one("embedding [16,8] by [2,4]", [[16, 8]], (B, [w]) => {
  const ids = B.fromArray([0, 3, 7, 15, 2, 9, 4, 11], [2, 4]);
  return B.embedding(w, ids);
});

console.log(`\ncpu_ref vs helios-vulkan, batched shapes, tolerance ${TOL}\n`);
console.log("operation                              max |diff|   verdict");
let firstBad = null;
for (const c of CASES) {
  let vr, cr;
  try {
    vr = host(V, c.run(V, c.shapes.map((s, i) => mk(V, s, i + 1))));
    cr = host(C, c.run(C, c.shapes.map((s, i) => mk(C, s, i + 1))));
  } catch (e) {
    console.log(`${c.name.padEnd(38)} ${"—".padStart(10)}   ERROR ${e.message.slice(0, 40)}`);
    continue;
  }
  if (vr.length !== cr.length) {
    console.log(`${c.name.padEnd(38)} ${"—".padStart(10)}   SHAPE ${cr.length} vs ${vr.length}`);
    firstBad ??= c.name;
    continue;
  }
  let worst = 0;
  for (let i = 0; i < cr.length; i++) {
    const d = Math.abs(vr[i] - cr[i]);
    if (Number.isFinite(d) && d > worst) worst = d;
    else if (!Number.isFinite(d) && Number.isFinite(cr[i])) worst = Infinity;
  }
  const bad = !(worst <= TOL);
  if (bad) firstBad ??= c.name;
  console.log(`${c.name.padEnd(38)} ${worst.toExponential(2).padStart(10)}   ${bad ? "**DIVERGES**" : "ok"}`);
}
console.log(firstBad ? `\nfirst divergence: ${firstBad}` : "\nevery operation agrees at batch > 1");
