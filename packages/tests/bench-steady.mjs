/* One backend, one batch, held at steady state for a fixed WALL-CLOCK window.
 *
 * bench-scale.mjs answers "how fast is this backend alone". It cannot answer
 * "how fast are these two backends while they share a device", because it
 * reports a median with no times attached, so two of them cannot be shown to
 * have overlapped. This one stamps every step with an absolute epoch time and
 * prints the lot, which is what lets bench-coresident.mjs restrict both sides
 * to the window in which they were genuinely running together.
 *
 * Usage: node bench-steady.mjs <native|vulkan|cpu> <batch> <seq> <warmupMs> <runMs>
 * Emits: progress lines on stderr, one JSON object on stdout at the end.
 */
import { NativeHeliosBackend, HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const which = process.argv[2] ?? "native";
const BATCH = Number(process.argv[3] ?? 128);
const SEQ = Number(process.argv[4] ?? 32);
const WARMUP_MS = Number(process.argv[5] ?? 3000);
const RUN_MS = Number(process.argv[6] ?? 20000);

const C = { vocabSize: 64, blockSize: SEQ, nLayer: 2, nEmbd: 64, nHead: 4, dropout: 0 };

const make = { native: () => new NativeHeliosBackend(0), vulkan: () => new HeliosBackend(),
               cpu: () => new CpuRefBackend() }[which];
if (!make) { console.error(`unknown backend ${which}`); process.exit(2); }
const B = make();
const params = initGPT(C, B, new SeededRng(7));

/* Model parameters must survive the release callback; everything else is an
 * intermediate. Same walk bench-scale.mjs uses, and for the same reason. */
const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);
const rel = (!process.env.NO_RELEASE && B.releaseGpuTensor)
  ? (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); } : undefined;

function step() {
  const n = BATCH * SEQ;
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [BATCH, SEQ]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [BATCH, SEQ]);
  const out = gptForward(C, params, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps?.();
  /* The step is not over until the GPU says so — flushAndWait first, it is the
   * only one of the three that names waiting. Without it the timer measures
   * submission, which is how Vulkan once reported 380,529 tok/s. */
  if (B.flushAndWait) B.flushAndWait();
  else if (B.syncGpu) B.syncGpu();
  else if (B.flush) B.flush();
  return loss;
}

let loss = 0;
const t0 = Date.now();
let warm = 0;
while (warm < 3 || Date.now() - t0 < WARMUP_MS) { loss = step(); warm++; }
process.stderr.write(`[${which}] warm ${warm} steps in ${Date.now() - t0} ms, loss ${loss.toFixed(4)}\n`);

/* Absolute epoch milliseconds, both ends. The pair (start, end) is what makes a
 * step attributable to a window: a step that straddles the boundary belongs to
 * neither side and is dropped rather than argued about. */
const steps = [];
const runStart = Date.now();
let lastBeat = runStart;
while (Date.now() - runStart < RUN_MS) {
  const s = Date.now();
  loss = step();
  const e = Date.now();
  steps.push([s, e]);
  if (e - lastBeat > 5000) {
    const recent = steps.slice(-10);
    const avg = recent.reduce((a, [x, y]) => a + (y - x), 0) / recent.length;
    process.stderr.write(`[${which}] t+${((e - runStart) / 1000).toFixed(0)}s  ${(BATCH * SEQ / (avg / 1000)).toFixed(0)} tok/s\n`);
    lastBeat = e;
  }
}

const st = B.stats ? B.stats() : null;
console.log(JSON.stringify({
  backend: which, batch: BATCH, seq: SEQ, tokensPerStep: BATCH * SEQ,
  loss, warmSteps: warm, steps,
  slabsGB: st && st.allocations !== undefined ? st.allocations * 4 / 1024 : null,
}));
