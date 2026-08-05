/* Which FORWARD operation does video memory get wrong?
 *
 * bench-scale runs forward and backward but never an optimizer, so the
 * parameters never change and the loss depends on the FORWARD pass alone. A
 * wrong gradient cannot move it. Under HELIOS_VIDMEM the loss moves -- 4.1866
 * to 4.1915 across runs, against a stable 4.1903-4.1905 on system memory, and
 * with a mean offset by 0.0016 rather than merely noisier. So something in the
 * forward path is wrong, and it is not the backward fallbacks.
 *
 * This is the same instrument that found Vulkan's embedding reading row zero for
 * every token: run each operation at the shapes the model actually produces and
 * compare against cpu_ref element by element, rather than comparing whole models
 * and learning only that something is wrong.
 *
 * Run it twice, under both residencies:
 *   node                     packages/tests/diff-vidmem-forward.mjs
 *   HELIOS_VIDMEM=1 node     packages/tests/diff-vidmem-forward.mjs
 *
 * Shapes are the batch-32 model: [1024, 64] activations, 4 heads of 16, and the
 * [128,32,32] score planes attention actually builds. Small shapes are useless
 * here -- every earlier probe passed under VIDMEM at small sizes. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";

const V = new NativeHeliosBackend(0);
const C = new CpuRefBackend();
const TOL = 2e-3;

const size = (s) => s.reduce((a, b) => a * b, 1);
const fill = (n, seed) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7 + seed) * 0.9);
const mk = (B, shape, seed) => B.fromArray(fill(size(shape), seed), shape);
function host(B, t) {
  B.flushAndWait?.(); B.syncGpu?.();
  const d = t.data;
  return Array.from({ length: size(t.shape) }, (_, i) => d[i]);
}

const B32 = 32, T = 32, D = 64, H = 4, ROWS = B32 * T, VOCAB = 64;
const CASES = [];
const one = (name, shapes, run) => CASES.push({ name, shapes, run });

one("embedding [64,64] by [32,32]", [[VOCAB, D]], (B, [w]) => {
  const ids = B.fromArray(Array.from({ length: ROWS }, (_, i) => i % VOCAB), [B32, T]);
  return B.embedding(w, ids);
});
one("layerNorm [1024,64]", [[ROWS, D], [D], [D]], (B, [a, w, b]) => B.layerNorm(a, w, b, 1e-5));
one("matmul qkv [1024,64]x[64,192]", [[ROWS, D], [D, 3 * D]], (B, [a, b]) => B.matmul(a, b));
one("matmul mlpup [1024,64]x[64,256]", [[ROWS, D], [D, 4 * D]], (B, [a, b]) => B.matmul(a, b));
one("matmul mlpdn [1024,256]x[256,64]", [[ROWS, 4 * D], [4 * D, D]], (B, [a, b]) => B.matmul(a, b));
one("softmax [128,32,32]", [[B32 * H, T, T]], (B, [a]) => B.softmax(a, -1));
one("gelu [1024,256]", [[ROWS, 4 * D]], (B, [a]) => B.gelu(a));
one("add [1024,64]", [[ROWS, D], [ROWS, D]], (B, [a, b]) => B.add(a, b));
one("transpose [1024,64]", [[ROWS, D]], (B, [a]) => B.transpose(a));
one("maskedFill [128,32,32]", [[B32 * H, T, T]], (B, [a]) => B.maskedFill(a, B.causalMask(T), -1e30));
one("crossEntropy [1024,64]", [[ROWS, VOCAB]], (B, [a]) => {
  const t = B.fromArray(Array.from({ length: ROWS }, (_, i) => i % VOCAB), [ROWS]);
  return B.crossEntropy(a, t);
});

console.log(`\nforward ops vs cpu_ref, batch-32 shapes, residency=${process.env.HELIOS_VIDMEM ? "VIDMEM" : "sysmem"}\n`);
console.log("operation                              max|diff|   verdict");
let bad = 0;
for (const c of CASES) {
  let vr, cr;
  try {
    vr = host(V, c.run(V, c.shapes.map((s, i) => mk(V, s, i + 1))));
    cr = host(C, c.run(C, c.shapes.map((s, i) => mk(C, s, i + 1))));
  } catch (e) {
    console.log(`${c.name.padEnd(36)} ${"—".padStart(11)}   ERROR ${e.message.slice(0, 32)}`);
    bad++; continue;
  }
  let worst = 0;
  for (let i = 0; i < cr.length; i++) {
    const d = Math.abs(vr[i] - cr[i]);
    worst = Number.isFinite(d) ? Math.max(worst, d) : Infinity;
  }
  const ok = worst <= TOL;
  if (!ok) bad++;
  console.log(`${c.name.padEnd(36)} ${worst.toExponential(2).padStart(11)}   ${ok ? "ok" : "**WRONG**"}`);
  V.finishStepOps?.();
}

/* And determinism: the same op twice in one process. A race shows here even
 * when the value happens to land inside tolerance on any single run. */
{
  const a = mk(V, [ROWS, D], 1), w = mk(V, [D], 2), b2 = mk(V, [D], 3);
  const r1 = host(V, V.layerNorm(a, w, b2, 1e-5));
  const r2 = host(V, V.layerNorm(a, w, b2, 1e-5));
  let n = 0;
  for (let i = 0; i < r1.length; i++) if (r1[i] !== r2[i]) n++;
  console.log(`\nsame layerNorm twice in one process: ${n} of ${r1.length} elements differ` +
              `   ${n ? "**NONDETERMINISTIC**" : "identical"}`);
}
console.log(bad ? `\n${bad} operation(s) WRONG` : "\nevery forward operation agrees with cpu_ref");
