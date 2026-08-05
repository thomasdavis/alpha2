/* WHICH TENSOR is leaking, by the only thing the allocator knows: its size.
 *
 * A leak shows up as "allocation failed" some hundreds of steps after the
 * allocation that caused it, and the allocation-site census that would name it
 * costs a stack per residency. The pool already knows how many live buffers it
 * holds of each size class, and at a fixed model shape there is essentially one
 * tensor shape per class — so the class IS the identification.
 *
 * Method: run steps, snapshot the live-by-class histogram at two step
 * boundaries far enough apart to see growth, and print the difference. Anything
 * with a non-zero delta is held across a boundary it should not survive.
 *
 * Usage: node probe-leak-by-class.mjs <nLayer> <nEmbd> <nHead> <vocab> <seq> <batch> [steps]
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const C = {
  nLayer: Number(process.argv[2] ?? 4),
  nEmbd: Number(process.argv[3] ?? 640),
  nHead: Number(process.argv[4] ?? 10),
  vocabSize: Number(process.argv[5] ?? 12288),
  blockSize: Number(process.argv[6] ?? 64),
  dropout: 0,
};
const BATCH = Number(process.argv[7] ?? 8);
const STEPS = Number(process.argv[8] ?? 24);

/* class c is (4 + c%4) << (10 + floor(c/4)) bytes — quarter-octave classes from
 * 4 KiB. Mirrored from tensor.c rather than exported, because it is arithmetic
 * and a second copy of arithmetic is cheaper than a second napi entry point. */
const classBytes = (c) => (4 + (c % 4)) * Math.pow(2, 10 + Math.floor(c / 4));
const human = (b) => b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(2)} MiB`
                   : b >= 1 << 10 ? `${(b / (1 << 10)).toFixed(0)} KiB` : `${b} B`;

const B = new NativeHeliosBackend(0);
const params = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };

const N = BATCH * C.blockSize;
function step() {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: N }, (_, i) => i % C.vocabSize), [BATCH, C.blockSize]);
  const tgt = B.fromArray(Array.from({ length: N }, (_, i) => (i + 1) % C.vocabSize), [BATCH, C.blockSize]);
  const out = gptForward(C, params, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps();
  return loss;
}

/* Warm past the one-off allocations: the pool carves for the first step or two
 * and those are not a leak, they are the pool filling. */
for (let i = 0; i < 6; i++) step();
const before = B.hl.stats().liveByClass.slice();
for (let i = 0; i < STEPS; i++) step();
const after = B.hl.stats().liveByClass.slice();

console.log(`${C.nLayer}L ${C.nEmbd}d batch ${BATCH} — growth over ${STEPS} steps\n`);
console.log("  class   size        before    after     delta   per step   bytes/step");
let totalPerStep = 0;
for (let c = 0; c < after.length; c++) {
  const d = (after[c] ?? 0) - (before[c] ?? 0);
  if (d === 0) continue;
  const perStep = d / STEPS;
  const bps = perStep * classBytes(c);
  totalPerStep += bps;
  console.log(`  ${String(c).padStart(5)}   ${human(classBytes(c)).padEnd(10)}` +
              `${String(before[c]).padStart(7)} ${String(after[c]).padStart(8)}` +
              `${String(d).padStart(9)} ${perStep.toFixed(1).padStart(10)}` +
              `   ${human(bps)}`);
}
console.log(`\n  total leaked per step: ${human(totalPerStep)}`);
if (totalPerStep === 0) console.log("  (nothing grows across a step boundary)");
