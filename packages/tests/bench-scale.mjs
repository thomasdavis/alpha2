/* Tokens per second as a function of BATCH, for one backend.
 *
 * The existing benchmark runs one sequence of 32 tokens a step, and at that size
 * the step is nearly all fixed cost: the same ~184 launches and the same host
 * work happen whether the batch is 1 or 64. So tokens/sec there measures
 * overhead, not throughput, and the way to raise it is to give each launch more
 * to do -- which is also how training actually runs.
 *
 * Usage: node bench-scale.mjs <native|vulkan|cpu> [batches] [seq]
 *   node bench-scale.mjs native 1,4,16,64
 *
 * Warmup is by TIME because this card idles at 210 MHz against 2100 and cannot
 * be clock-locked inside the container. Samples are a MEDIAN with the spread
 * printed, so a difference smaller than the spread is visibly not a difference.
 *
 * One backend per process, deliberately: with the native channel open the same
 * Vulkan binary measured 142 tok/s where it measures 628, so any number drawn
 * from a shared process is worthless. */
import { NativeHeliosBackend, HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const which = process.argv[2] ?? "native";
const BATCHES = (process.argv[3] ?? "1,2,4,8,16,32").split(",").map(Number);
const SEQ = Number(process.argv[4] ?? 32);
const WARMUP_MS = Number(process.env.WARMUP_MS ?? 2000);
const SAMPLES = Number(process.env.SAMPLES ?? 15);

const C = { vocabSize: 64, blockSize: SEQ, nLayer: 2, nEmbd: 64, nHead: 4, dropout: 0 };

const make = { native: () => new NativeHeliosBackend(0), vulkan: () => new HeliosBackend(),
               cpu: () => new CpuRefBackend() }[which];
if (!make) { console.error(`unknown backend ${which}`); process.exit(2); }
const B = make();
const params = initGPT(C, B, new SeededRng(7));

/*
 * A step, ending at a STEP BOUNDARY -- which is what makes memory bounded.
 *
 * Both GPU backends get the same treatment, and it is the treatment the trainer
 * already gives them: `releaseGpuTensor` for intermediates and `finishStepOps`
 * once a step. Without it the native backend takes 12 fresh 4 MiB slabs from
 * the driver every batch-16 step and never gives them back, so the run degrades
 * into thrashing rather than measuring anything.
 *
 * Model parameters must survive, so the callback refuses them.
 */
const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);

const rel = B.releaseGpuTensor ? (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); } : undefined;

function step(batch) {
  const n = batch * SEQ;
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [batch, SEQ]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [batch, SEQ]);
  const out = gptForward(C, params, B, tape, tok, tgt, true);
  /* BEFORE backward: the tape releases each entry's forward output and nulls
   * its `.data`, and the loss is an entry output like any other. Reading it
   * afterwards works only when nothing is being released. */
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps?.();
  /*
   * The step is not over until the GPU says so.
   *
   * Without this the Vulkan backend reported 380,529 tok/s at batch 128 -- 4096
   * tokens in 10.8 ms, with the step time flat from batch 16 upward and a loss
   * that disagreed with every other backend. It enqueues and returns; the timer
   * was measuring how fast work could be SUBMITTED. Reading the loss before
   * backward (which the tape's release path requires) removed the one accidental
   * synchronisation that had been hiding it.
   *
   * Both backends get the same barrier, and it is the barrier training would
   * need anyway before the optimizer reads a gradient.
   */
  if (B.syncGpu) B.syncGpu();
  else if (B.flush) B.flush();
  return loss;
}

console.log(`${which}: ${C.nLayer}L ${C.nEmbd}d ${C.nHead}h, seq ${SEQ}\n`);
console.log("batch   tokens/step   ms/step      tok/s   spread ms        loss");
for (const batch of BATCHES) {
  let loss = 0, ms = [];
  try {
    const t0 = process.hrtime.bigint();
    let warm = 0;
    while (warm < 3 || Number(process.hrtime.bigint() - t0) / 1e6 < WARMUP_MS) { loss = step(batch); warm++; }
    for (let i = 0; i < SAMPLES; i++) {
      const t = process.hrtime.bigint();
      loss = step(batch);
      ms.push(Number(process.hrtime.bigint() - t) / 1e6);
    }
  } catch (e) {
    console.log(`${String(batch).padStart(5)}   ${String(batch * SEQ).padStart(11)}   FAILED: ${e.message}`);
    break;
  }
  ms.sort((a, b) => a - b);
  const med = ms[Math.floor(ms.length / 2)];
  console.log(
    `${String(batch).padStart(5)}   ${String(batch * SEQ).padStart(11)}   ${med.toFixed(1).padStart(7)}   ` +
    `${(batch * SEQ / (med / 1000)).toFixed(0).padStart(8)}   ` +
    `[${ms[0].toFixed(1)}-${ms[ms.length - 1].toFixed(1)}]`.padStart(15) +
    `   ${loss.toFixed(4)}`);
}
