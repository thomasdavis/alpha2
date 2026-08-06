/* Does the pool recycle when something actually frees?
 *
 * tensor.h promises that "after the first step the pool should serve every
 * request without a single allocation", and `carved` exists so the claim can be
 * checked rather than assumed. In the benchmark it climbs forever -- 67,015
 * carves and 299 slabs over a few hundred steps -- because that harness passes
 * no release callback and nothing ever gives a buffer back.
 *
 * The trainer DOES pass one: trainer.ts threads releaseFn through gptForward and
 * tape.backward. So the question that decides whether this backend is fit for a
 * real run is not whether the benchmark leaks, but whether the pool recycles
 * when it is fed. This runs the same step both ways and prints `carved` for each.
 *
 * WHAT WOULD FAIL: if release frees a tensor the graph still references, the
 * generation check rejects the stale handle, helios_tensor_addr returns 0 and
 * the dispatch fails loudly -- so the loss below is checked too. A silent
 * corruption would show up as a loss that drifts from the no-release arm. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const C = { vocabSize: 64, blockSize: 32, nLayer: 2, nEmbd: 64, nHead: 4, dropout: 0 };
const TOKENS = C.blockSize;
const STEPS = 40;

const B = new NativeHeliosBackend(0);
const params = initGPT(C, B, new SeededRng(7));

/* `mode` bisects WHERE the release callback goes. The trainer passes the same
 * function to both gptForward and tape.backward, and those are two different
 * owners of overlapping tensors -- so if one of them is the problem, wiring
 * them separately says which. */
function run(mode) {
  const toModel = mode === "model" || mode === "both";
  const toTape = mode === "tape" || mode === "both";
  const useRelease = toModel || toTape;
  /* The model's own parameters must survive the step, so the callback refuses
   * anything that is one of them. Without that guard the first backward frees
   * the weights and the second step reads a recycled buffer. */
  const kept = new Set();
  (function walk(v, depth) {
    if (!v || typeof v !== "object" || depth > 6) return;
    if (v.buffer && v.shape) { kept.add(v); return; }  /* a TensorData */
    if (v.data) kept.add(v.data);                       /* a Variable */
    for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, depth + 1);
  })(params, 0);
  const release = useRelease
    ? (td) => { if (td && !kept.has(td)) B.release(td); }
    : undefined;

  const before = B.stats();
  const t0 = process.hrtime.bigint();
  let loss = 0;
  for (let i = 0; i < STEPS; i++) {
    const tape = new Tape();
    const tok = B.fromArray(Array.from({ length: TOKENS }, (_, j) => j % C.vocabSize), [1, TOKENS]);
    const tgt = B.fromArray(Array.from({ length: TOKENS }, (_, j) => (j + 1) % C.vocabSize), [1, TOKENS]);
    const out = gptForward(C, params, B, tape, tok, tgt, true, false, false, undefined, toModel ? release : undefined);
    tape.backward(out.loss, B, toTape ? release : undefined);
    loss = out.loss.data.data[0];
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const after = B.stats();
  return {
    carved: after.carved - before.carved,
    slabs: after.allocations - before.allocations,
    pooled: after.pooled,
    ms: ms / STEPS,
    loss,
  };
}

/* Warm first so no arm pays the clock ramp or the program cache. */
run("none");
const base = run("none");

console.log(`\n${STEPS} steps each\n`);
console.log("release wired to   carved/step   slabs   pooled   ms/step   loss");
const row = (n, r) => console.log(
  `${n.padEnd(18)} ${(r.carved / STEPS).toFixed(1).padStart(11)} ${String(r.slabs).padStart(7)} ` +
  `${String(r.pooled).padStart(8)} ${r.ms.toFixed(2).padStart(9)}   ${typeof r.loss === "number" ? r.loss.toFixed(4) : r.loss}`);
row("nothing", base);
for (const mode of ["tape", "model", "both"]) {
  let r;
  try { r = run(mode); }
  catch (e) { console.log(`${mode.padEnd(18)} FAILED: ${e.message}`); continue; }
  row(mode, r);
  if (r.loss !== base.loss)
    console.log(`   ^ LOSS DIFFERS from ${base.loss.toFixed(4)} — this arm frees something still live`);
}
