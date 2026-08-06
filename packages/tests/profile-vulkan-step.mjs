/* Does the VULKAN backend give back what it takes, one step at a time?
 *
 * Vulkan runs 105M at batch 1 in 84.9 ms against native's 407.6 -- five times
 * faster per step -- and then dies at batch 8. At batch 16 its own allocator
 * says why:
 *
 *     activeBufferBytes 8,103,542,784      on an 8 GB card
 *     tempSlabLiveRefs  1006
 *     tempSlabResets    0
 *
 * Nothing is ever handed back, so the temp pool never resets and the run walks
 * off the end of the device. That is the same shape of bug just fixed on the
 * native side, where device() allocated a copy of every non-resident operand
 * and no caller owned it; the Vulkan spelling of device() is ensureGpu, and
 * gpuMemStats already counts its uploads.
 *
 * This asks the question directly rather than by watching a benchmark die:
 * run N steps at a shape that fits, and print allocations and releases PER
 * STEP. A backend that balances shows alloc == release once warm. A backend
 * that leaks shows a constant positive difference, and its size names the
 * cost.
 *
 * Usage: node profile-vulkan-step.mjs [nLayer] [seq] [batch] [steps]
 *
 * Deliberately NOT a tok/s benchmark: the numbers here are counters, so clock
 * ramp and queue depth do not move them and the reading needs no median.
 */
import { HeliosBackend, heliosLeakCensus, heliosLeakCensusReset } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const L = Number(process.argv[2] ?? 18);
const SEQ = Number(process.argv[3] ?? 64);
const BATCH = Number(process.argv[4] ?? 1);
const STEPS = Number(process.argv[5] ?? 6);
const C = { vocabSize: 12288, blockSize: SEQ, nLayer: L, nEmbd: 640, nHead: 10, dropout: 0 };

const B = new HeliosBackend();
const P = initGPT(C, B, new SeededRng(7));

/* Parameters survive the step; everything else is an intermediate. Same walk
 * bench-shape uses, and for the same reason -- releasing a weight would be a
 * use-after-free rather than a saving. */
const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(P, 0);
const rel = B.releaseGpuTensor ? (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); } : undefined;
if (!rel) console.log("WARNING: this backend has no releaseGpuTensor — nothing can be freed\n");

const paramVars = [];
(function walkV(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.requiresGrad !== undefined && v.data) { paramVars.push(v); return; }
  for (const x of Array.isArray(v) ? v : Object.values(v)) walkV(x, d + 1);
})(P, 0);

function step() {
  const n = BATCH * SEQ;
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [BATCH, SEQ]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [BATCH, SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  for (const v of paramVars) { if (v.grad) { rel?.(v.grad); v.grad = null; } }
  tape.clear(rel);
  B.finishStepOps?.();
  if (B.flushAndWait) B.flushAndWait();
  else if (B.syncGpu) B.syncGpu();
  return loss;
}

console.log(`${L}L seq ${SEQ} batch ${BATCH} — ${BATCH * SEQ} tokens/step, ${STEPS} steps\n`);
console.log("step   allocs  releases   frRel     live   liveMB   uploads   slabLiveMB  resets");

let prevUploads = 0;
for (let i = 0; i < STEPS; i++) {
  const loss = step();
  const s = B.gpuMemStats();
  const up = (s.flowEnsureGpuUploads ?? 0) - prevUploads;
  prevUploads = s.flowEnsureGpuUploads ?? 0;
  console.log(
    `${String(i).padStart(4)} ${String(s.diagAllocsThisStep ?? 0).padStart(8)} ` +
    `${String(s.diagReleasesThisStep ?? 0).padStart(9)} ${String(s.diagFrReleasesThisStep ?? 0).padStart(7)} ` +
    `${String(s.liveAllocs ?? 0).padStart(8)} ` +
    `${((s.activeBufferBytes ?? 0) / 1048576).toFixed(0).padStart(8)} ` +
    `${String(up).padStart(9)} ` +
    `${((s.tempSlabLiveBytes ?? 0) / 1048576).toFixed(0).padStart(12)} ` +
    `${String(s.tempSlabResets ?? 0).padStart(7)}` +
    (i === STEPS - 1 ? `   loss ${loss.toFixed(4)}` : ""),
  );
}

/*
 * The last two steps are the ones that matter: the first few carve pools and
 * compile pipelines, so their counters describe startup rather than a step.
 * A steady state that still grows is a leak; a steady state that does not is
 * a backend whose batch is limited by the model, which is the honest place to
 * be.
 */
console.log("\nRead the LAST rows: growth in liveMB after warmup is the leak.");

/*
 * One WARM step in isolation, so the census describes a step rather than
 * startup. Everything recorded before this point is pool carving and pipeline
 * compilation and would drown the signal.
 */
if (process.env.HELIOS_LEAK_CENSUS === "1") {
  heliosLeakCensusReset();
  step();
  console.log("\nsurvivors of ONE warm step, by call site:\n");
  console.log(heliosLeakCensus(6));
}
