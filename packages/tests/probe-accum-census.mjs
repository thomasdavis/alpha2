/* What accumulates, and how big it is.
 *
 * addInplace is the second-largest item on the GPU after the GEMM — 17% of a
 * step across 166 calls — and each one is a read-add-write pass over a tensor a
 * kernel had just finished writing. matmul avoids its share by computing into
 * the destination the tape offers it; nothing else does. Before building a
 * second accumulate-into-destination path, this says which one is worth it.
 *
 * Usage: HELIOS_VIDMEM=1 node probe-accum-census.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape, drainAccumCensus } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const SEQ = 64, BATCH = Number(process.env.NATIVE_BATCH ?? 12);
const C = { vocabSize: 12288, blockSize: SEQ, nLayer: 18, nEmbd: 640, nHead: 10, dropout: 0 };
const B = new NativeHeliosBackend(0);
const P = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1); })(P, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };

/* One step is enough — the tape is the same every step. */
{
  const tape = new Tape();
  const tok = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>i%C.vocabSize),[BATCH,SEQ]);
  const tgt = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>(i+1)%C.vocabSize),[BATCH,SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  tape.backward(out.loss, B, rel);
  tape.clear(rel); B.finishStepOps?.();
}

const rows = drainAccumCensus();
let total = 0, bytes = 0;
for (const [shape, n] of rows) {
  const dims = shape.slice(1, -1).split(",").map(Number);
  const elems = dims.reduce((a, b) => a * b, 1);
  total += n;
  bytes += n * elems * 4 * 3;  /* read dest, read g, write dest */
}
console.log(`\n${total} accumulations, ${(bytes / 1e6).toFixed(0)} MB of traffic\n`);
console.log("  shape".padEnd(28) + "calls   MB");
for (const [shape, n] of rows.slice(0, 16)) {
  const dims = shape.slice(1, -1).split(",").map(Number);
  const elems = dims.reduce((a, b) => a * b, 1);
  console.log(`  ${shape}`.padEnd(28) + `${String(n).padStart(4)}  ${(n * elems * 4 * 3 / 1e6).toFixed(1)}`);
}
