/* WHAT is the pool holding, by size? Classes are powers of two from 4 KiB, so
 * at this shape each class is one identifiable tensor kind. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";
const L = 18, SEQ = 64, V = 12288, D = 640, H = 10;
const C = { vocabSize: V, blockSize: SEQ, nLayer: L, nEmbd: D, nHead: H, dropout: 0 };
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

const snap = () => (B.stats().liveByClass ?? []).slice();
function step() {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({length:SEQ},(_,i)=>i%V),[1,SEQ]);
  const tgt = B.fromArray(Array.from({length:SEQ},(_,i)=>(i+1)%V),[1,SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  tape.backward(out.loss, B, rel);
  for (const v of paramVars) { if (v.grad) { rel(v.grad); v.grad = null; } }
  tape.clear(rel); B.finishStepOps?.();
}
step(); const a = snap();
step(); const b = snap();
console.log("class   size      live after step2   growth per step   bytes grown");
for (let i = 0; i < a.length; i++) {
  const g = (b[i] ?? 0) - (a[i] ?? 0);
  if (g === 0 && (b[i] ?? 0) === 0) continue;
  const sz = 1 << (12 + i);
  console.log(`${String(i).padStart(5)}  ${(sz/1024).toFixed(0).padStart(6)} KiB  ${String(b[i]).padStart(16)}  ${String(g).padStart(16)}  ${((g*sz)/1048576).toFixed(1).padStart(9)} MB`);
}
