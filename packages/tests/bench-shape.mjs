/* Tokens per second at a SHAPE YOU NAME, because the benchmark's shape is 0.11M
 * parameters and nobody trains that.
 *
 * bench-scale.mjs is hard-wired to 2 layers, 64 embd, 4 heads, vocab 64 and 32
 * tokens — about 110,000 parameters. It is the right size for deciding whether
 * a kernel is correct and the wrong size for deciding anything about training
 * cost: at that shape a step is almost entirely fixed overhead, so tokens/sec
 * measures the host, and the arithmetic that dominates a real model is absent.
 *
 * Usage: node bench-shape.mjs <nLayer> <nEmbd> <nHead> <vocab> <seq> <batch> [backend]
 *   node bench-shape.mjs 18 640 10 12288 1024 4 native
 *
 * Same discipline as bench-scale: warm by TIME (this card idles at 210 MHz
 * against 2100 and cannot be clock-locked in a container), report the MEDIAN
 * with the spread, release intermediates and end the step, and hold a real
 * barrier before stopping the clock.
 */
import { NativeHeliosBackend, HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const nLayer = Number(process.argv[2] ?? 18);
const nEmbd = Number(process.argv[3] ?? 640);
const nHead = Number(process.argv[4] ?? 10);
const vocabSize = Number(process.argv[5] ?? 12288);
const SEQ = Number(process.argv[6] ?? 1024);
const BATCH = Number(process.argv[7] ?? 1);
const which = process.argv[8] ?? "native";
const WARMUP_MS = Number(process.env.WARMUP_MS ?? 4000);
const SAMPLES = Number(process.env.SAMPLES ?? 9);

const C = { vocabSize, blockSize: SEQ, nLayer, nEmbd, nHead, dropout: 0 };

/* Parameters, counted rather than asserted — the whole point of this file is
 * that the number is not what people assume. GPT-2 form, untied head. */
const perLayer = 4 * nEmbd * nEmbd + 4 * nEmbd     /* attention: wqkv+proj, biases */
               + 8 * nEmbd * nEmbd + 5 * nEmbd     /* mlp 4x, biases */
               + 4 * nEmbd;                        /* two layernorms */
const params = vocabSize * nEmbd + SEQ * nEmbd + nLayer * perLayer + 2 * nEmbd
             + vocabSize * nEmbd;
console.log(`${nLayer}L ${nEmbd}d ${nHead}h vocab ${vocabSize} seq ${SEQ} batch ${BATCH} — ` +
            `~${(params / 1e6).toFixed(1)}M parameters, ${BATCH * SEQ} tokens/step — ${which}\n`);

const make = { native: () => new NativeHeliosBackend(0), vulkan: () => new HeliosBackend(),
               cpu: () => new CpuRefBackend() }[which];
if (!make) { console.error(`unknown backend ${which}`); process.exit(2); }
const B = make();
const P = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(P, 0);
const rel = B.releaseGpuTensor ? (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); } : undefined;

function step() {
  const n = BATCH * SEQ;
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [BATCH, SEQ]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [BATCH, SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps?.();
  if (B.flushAndWait) B.flushAndWait();
  else if (B.syncGpu) B.syncGpu();
  return loss;
}

let loss = 0, warm = 0;
const t0 = Date.now();
try {
  while (warm < 2 || Date.now() - t0 < WARMUP_MS) { loss = step(); warm++; }
} catch (e) {
  console.log(`FAILED during warmup after ${warm} steps: ${e.message}`);
  process.exit(1);
}

const ms = [];
for (let i = 0; i < SAMPLES; i++) {
  const t = process.hrtime.bigint();
  loss = step();
  ms.push(Number(process.hrtime.bigint() - t) / 1e6);
}
ms.sort((a, b) => a - b);
const med = ms[Math.floor(ms.length / 2)];
const st = B.stats ? B.stats() : null;
console.log(`${(BATCH * SEQ / (med / 1000)).toFixed(0).padStart(8)} tok/s   median ${med.toFixed(1)} ms  ` +
            `[${ms[0].toFixed(1)}-${ms[ms.length - 1].toFixed(1)}]  loss ${loss.toFixed(4)}  (${warm} warm)` +
            (st && st.allocations !== undefined ? `  held ${(st.allocations * 4 / 1024).toFixed(2)} GB` : ""));
