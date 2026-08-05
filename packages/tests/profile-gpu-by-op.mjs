/* WHERE does the GPU time go at 105M?
 *
 * The step is 355 ms of GPU and 0.54% of this card's FP32 peak, and the one
 * explanation offered — B re-read once per output row — was refuted: L2 was
 * already providing that reuse. So stop proposing structures and ask which
 * kernels hold the clock.
 *
 * Method: drain after every operation and charge the fence-spin delta to it.
 * Draining per op removes all overlap, so the total is larger than a real
 * step's; the SHARE is what this is for, and a share needs the ops separated.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

/* Usage: node profile-gpu-by-op.mjs [nLayer] [seq]
 *
 * REPRODUCIBILITY, which is the reason to prefer this over end-to-end tok/s:
 * three consecutive runs agree to 0.1 ms (375.2, 375.2, 375.2), where the
 * whole-step spin counter varies 342-390 on the same build. Draining per op
 * removes the overlap that makes a step's total sensitive to clock ramp. A cold
 * first run still reads high — discard it. */
const L = Number(process.argv[2] ?? 18), SEQ = Number(process.argv[3] ?? 64);
/* BATCH matters to the share, not just the total. At batch 1 a matmul launches
 * SEQ blocks and leaves most of the card idle, so the elementwise ops — which
 * are bandwidth-bound and already wide — look disproportionately cheap. The
 * shape that fills the card is the shape whose profile is worth acting on. */
const BATCH = Number(process.argv[4] ?? 1);
const C = { vocabSize: 12288, blockSize: SEQ, nLayer: L, nEmbd: 640, nHead: 10, dropout: 0 };
const B = new NativeHeliosBackend(0);
const P = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1); })(P, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };

const METHODS = ["add","sub","mul","div","neg","gelu","exp","log","sqrt","scale","clamp",
  "matmul","matmulTransposed","sum","mean","layerNorm","rmsNorm","softmax","softCap",
  "embedding","crossEntropy","transpose","slice","causalMask","maskedFill","cat","zeros",
  "ones","full","clone","broadcast","addInplace"];
const cost = new Map();
let tracking = false, depth = 0;
const hl = B.hl;
for (const m of METHODS) {
  const orig = B[m];
  if (typeof orig !== "function") continue;
  B[m] = function (...args) {
    if (!tracking || depth > 0) return orig.apply(this, args);
    depth++;
    const before = hl.stats().spinNs;
    let r;
    try { r = orig.apply(this, args); hl.flush(); }
    finally { depth--; }
    const us = (hl.stats().spinNs - before) / 1000;
    const e = cost.get(m) ?? { n: 0, us: 0 };
    e.n++; e.us += us; cost.set(m, e);
    return r;
  };
}

/* Warm: programs compile on first use and this card ramps its clock. */
for (let w = 0; w < 2; w++) {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>i%C.vocabSize),[BATCH,SEQ]);
  const tgt = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>(i+1)%C.vocabSize),[BATCH,SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  tape.backward(out.loss, B, rel);
  tape.clear(rel); B.finishStepOps?.();
}

tracking = true;
const t0 = Date.now();
const tape = new Tape();
const tok = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>i%C.vocabSize),[BATCH,SEQ]);
const tgt = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>(i+1)%C.vocabSize),[BATCH,SEQ]);
const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
tape.backward(out.loss, B, rel);
tracking = false;
tape.clear(rel); B.finishStepOps?.();

let total = 0, calls = 0;
for (const [, v] of cost) { total += v.us; calls += v.n; }
console.log(`${L}L seq ${SEQ} batch ${BATCH}: ${calls} operations, ${(total/1000).toFixed(1)} ms of GPU (drained per op), wall ${Date.now()-t0} ms\n`);
console.log("operation           calls    GPU ms    %     us/call");
for (const [k, v] of [...cost.entries()].sort((a,b) => b[1].us - a[1].us).slice(0, 12))
  console.log(`${k.padEnd(18)} ${String(v.n).padStart(6)}  ${(v.us/1000).toFixed(1).padStart(8)}  ${(v.us/total*100).toFixed(1).padStart(5)}  ${(v.us/v.n).toFixed(0).padStart(8)}`);
