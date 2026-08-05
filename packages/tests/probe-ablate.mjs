/* THE MARGINAL COST OF AN OP FAMILY, by removing it and re-timing the step.
 *
 * Two instruments disagreed about addInplace by an order of magnitude and one
 * of them had to be wrong. The per-op profiler drains after every operation and
 * charged it 382 us at [24,64,640]; the same op measured in isolation runs at
 * 355 GB/s, which is 33 us. Draining serialises, so a bandwidth-bound op that
 * overlaps happily in a real step is charged its whole isolated latency — the
 * profiler's ORDERING can be right while its magnitudes are not.
 *
 * Ablation cannot make that mistake. Stub the op, run the same step, and the
 * difference in GPU time is what it actually cost, overlap and all. The answer
 * is wrong — that is the point, and it is why this file only ever reports
 * TIMING and never a loss.
 *
 * Only ops that can be stubbed without changing a SHAPE are listed. addInplace
 * returns void, so removing it is exact; anything that returns a fresh tensor
 * would need a replacement allocation and that is its own cost.
 *
 * Usage: node probe-ablate.mjs [nLayer] [seq] [batch]
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const L = Number(process.argv[2] ?? 18), SEQ = Number(process.argv[3] ?? 64);
const BATCH = Number(process.argv[4] ?? 24);
const C = { vocabSize: 12288, blockSize: SEQ, nLayer: L, nEmbd: 640, nHead: 10, dropout: 0 };
const B = new NativeHeliosBackend(0);
const P = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1); })(P, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };
const paramVars = [];
(function walkV(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.requiresGrad !== undefined && v.data) { paramVars.push(v); return; }
  for (const x of Array.isArray(v) ? v : Object.values(v)) walkV(x, d + 1); })(P, 0);

function step() {
  const n = BATCH * SEQ;
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [BATCH, SEQ]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [BATCH, SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  tape.backward(out.loss, B, rel);
  for (const v of paramVars) { if (v.grad) { rel(v.grad); v.grad = null; } }
  tape.clear(rel);
  B.finishStepOps?.();
}

/* Warm by TIME — this card idles at 210 MHz against 2,100 and cannot be
 * clock-locked in a container, so a cold run understates by up to 4.9x. */
function measure(label) {
  const t0 = Date.now();
  while (Date.now() - t0 < 4000) step();
  const s0 = B.hl.stats();
  const spin0 = s0.spinNs, enq0 = s0.enqueued, bar0 = s0.barriers ?? 0, rw0 = s0.barriersIfRW ?? 0;
  const w0 = process.hrtime.bigint();
  const N = 9;
  for (let i = 0; i < N; i++) step();
  const wall = Number(process.hrtime.bigint() - w0) / 1e6 / N;
  const s1 = B.hl.stats();
  const gpu = (s1.spinNs - spin0) / 1e6 / N;
  const enq = (s1.enqueued - enq0) / N, bar = ((s1.barriers ?? 0) - bar0) / N;
  const rw = ((s1.barriersIfRW ?? 0) - rw0) / N;
  console.log(`  ${label.padEnd(28)} GPU ${gpu.toFixed(1).padStart(6)} ms   wall ${wall.toFixed(1).padStart(6)} ms` +
              `   ${(BATCH * SEQ / (wall / 1000)).toFixed(0).padStart(6)} tok/s` +
              `   ${enq.toFixed(0).padStart(5)} launches, ${bar.toFixed(0).padStart(5)} barriers` +
              ` (${(bar / Math.max(1, enq) * 100).toFixed(0)}%), read/write-aware would be` +
              ` ${rw.toFixed(0)} (${(rw / Math.max(1, enq) * 100).toFixed(0)}%)`);
  return gpu;
}

console.log(`${L}L seq ${SEQ} batch ${BATCH} — the step, then the step without one op family\n`);
const base = measure("baseline");

const origAdd = B.addInplace.bind(B);
B.addInplace = () => {};
const noAdd = measure("addInplace stubbed");
B.addInplace = origAdd;

console.log(`\n  addInplace's real marginal cost: ${(base - noAdd).toFixed(1)} ms of GPU` +
            `  (${((base - noAdd) / base * 100).toFixed(1)}% of the step)`);
console.log(`  the per-op profiler charged it 25.1 ms; isolation says 2.5 ms.`);
