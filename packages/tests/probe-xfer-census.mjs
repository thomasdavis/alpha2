/* How much of a step is host<->device round trips that nobody asked for?
 *
 * A read of `.data` on a device tensor copies the WHOLE buffer back through the
 * elementwise kernel, flushes the queue to make the copy real, and marks the
 * mirror dirty — so the next time that tensor is an operand, commit() copies
 * the whole thing forward again. One touched element costs two full-buffer
 * copies and a mid-step drain, and a per-op profile charges it to whichever op
 * was running.
 *
 * The suspicion this exists to test: addInplace on the residual stream measures
 * 382 us for a [24,64,640] tensor, which is 31 GB/s against a 448 GB/s card.
 */
import { NativeHeliosBackend, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";
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

for (let i = 0; i < 3; i++) step();            /* warm; programs compile once */
NativeBuffer.resetCensus();
step();
const c = NativeBuffer.census;
const mb = (b) => (b / 1048576).toFixed(1);
console.log(`${L}L seq ${SEQ} batch ${BATCH}, ONE warm step:`);
console.log(`  device -> host  ${String(c.mirrorCopies).padStart(5)} copies  ${mb(c.mirrorBytes).padStart(8)} MB   (each one also DRAINS the queue)`);
console.log(`  host -> device  ${String(c.commitCopies).padStart(5)} copies  ${mb(c.commitBytes).padStart(8)} MB`);
console.log(`  total round-trip traffic ${mb(c.mirrorBytes + c.commitBytes)} MB, ${c.flushes} mid-step drains`);
