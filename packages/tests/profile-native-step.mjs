/* Where does a native step's time actually go?
 *
 * The benchmark says 90 tok/s against Vulkan's 651 and the commit log blames
 * "a drain per operation", but that is an inference from what the code does,
 * not a measurement of what it costs. This measures it.
 *
 * Three questions, in order of how much they would change the plan:
 *   1. How many enqueues and how many FLUSHES does one step take? Batching only
 *      helps if flushes are far fewer than enqueues.
 *   2. Which backend methods hold the wall clock, and how many calls each?
 *   3. Of a method's time, how much is the C call and how much is JavaScript?
 *
 * Method wrapping counts the OUTERMOST call only. binary() calls expand()
 * calls device(); attributing the inner time to both would double-count and
 * make the total exceed the step. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

/*
 * The SHAPE is an argument now, because the answer depends on it.
 *
 * This was hard-wired to 2 layers, 64 embd and 32 tokens — 0.11M parameters —
 * and at that size a step is nearly all fixed overhead, so the ranking it
 * produces is a ranking of per-call costs rather than of where a real step's
 * time goes. The tensor-core GEMM moved the 105M step from 70% GPU to 47%, and
 * the question "which host call holds the other 53%" cannot be asked at a shape
 * whose host work is a different mix.
 *
 * Usage: node profile-native-step.mjs [batch] [nLayer] [nEmbd] [nHead] [vocab] [seq]
 */
const BATCH = Number(process.argv[2] ?? 1);
const C = {
  nLayer: Number(process.argv[3] ?? 2),
  nEmbd: Number(process.argv[4] ?? 64),
  nHead: Number(process.argv[5] ?? 4),
  vocabSize: Number(process.argv[6] ?? 64),
  blockSize: Number(process.argv[7] ?? 32),
  dropout: 0,
};
const TOKENS = C.blockSize;

const B = new NativeHeliosBackend(0);

/* The flush, measured separately.
 *
 * Method wrapping attributes time to whichever backend call is outermost, and
 * a large share of the step is in NEITHER -- the tape and the model read
 * `.data` directly, and that getter drains the queue. Those drains are real
 * time that no method row can show. Wrapping the addon's flush catches every
 * one of them, wherever it was triggered from. */
const hl = B.hl;
const rawFlush = hl.flush.bind(hl);
let flushN = 0, flushMs = 0;
/* A flush with nothing queued costs 0.1 us and one that actually waits costs
 * hundreds. Counting them together hides which is which, so anything over this
 * is treated as a REAL DRAIN and attributed to whoever caused it. */
const DRAIN_US = 50;
let drainN = 0, drainMs = 0;
const drainBy = new Map();
let sites = null; /* set during the measured window only — stacks are dear */
let current = "(tape/model)";
hl.flush = () => {
  const t = process.hrtime.bigint();
  try { return rawFlush(); }
  finally {
    const us = Number(process.hrtime.bigint() - t) / 1000;
    flushMs += us / 1000; flushN++;
    if (us > DRAIN_US) {
      drainN++; drainMs += us / 1000;
      const e = drainBy.get(current) ?? { n: 0, ms: 0 };
      e.n++; e.ms += us / 1000;
      drainBy.set(current, e);
      /* A drain that no backend method was executing came from somebody
       * reading `.data` directly. Naming the method row is not enough to fix
       * it -- the call SITE is what has to change -- so take the stack. */
      if (current === "(tape/model)" && sites) {
        const st = (new Error().stack ?? "").split("\n").slice(2, 6)
          .map((l) => l.trim().replace(/^at /, "").replace(/.*\/packages\//, "packages/"))
          .filter((l) => !l.includes("profile-native-step"))
          .join(" <- ");
        const c = sites.get(st) ?? { n: 0, ms: 0 };
        c.n++; c.ms += us / 1000;
        sites.set(st, c);
      }
    }
  }
};

/* Wrap every public method with a counter and a timer. Depth guards the
 * outermost-only rule. */
const prof = new Map();
let depth = 0;
const METHODS = [
  "add", "sub", "mul", "div", "neg", "relu", "gelu", "silu", "exp", "log",
  "sqrt", "scale", "clamp", "pow", "matmul", "sum", "mean", "rmsNorm",
  "layerNorm", "softmax", "embedding", "crossEntropy", "transpose", "zeros",
  "ones", "full", "fromArray", "reshape", "clone", "slice", "causalMask",
  "maskedFill", "cat", "gather", "argmax", "topk", "release", "randn",
];
for (const m of METHODS) {
  const orig = B[m];
  if (typeof orig !== "function") continue;
  B[m] = function (...args) {
    if (depth > 0) return orig.apply(this, args);
    depth++;
    const outer = current;
    current = m;
    const t = process.hrtime.bigint();
    try {
      return orig.apply(this, args);
    } finally {
      const dt = Number(process.hrtime.bigint() - t) / 1e6;
      depth--;
      current = outer;
      const e = prof.get(m) ?? { n: 0, ms: 0 };
      e.n++; e.ms += dt;
      prof.set(m, e);
    }
  };
}

const params = initGPT(C, B, new SeededRng(7));

/*
 * PROFILE THE PATH THE BENCHMARK MEASURES, not a different one.
 *
 * This ran without the release callback, so nothing was ever reclaimed and the
 * step allocated FRESH slabs the whole way down: 2,101 slabs against the
 * benchmark's 609, and a fresh 4 KB allocation costs 802 us where a pooled one
 * costs 1.0. Every per-method figure was therefore dominated by an allocation
 * cost the measured configuration does not pay, and the ranking it produced was
 * a ranking of who allocated most.
 *
 * A profiler that does not run the configuration under test is not measuring
 * the thing it is being read to explain.
 */
const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);
const rel = B.releaseGpuTensor ? (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); } : undefined;

function step(params) {
  const tape = new Tape();
  const n = BATCH * TOKENS;
  const tok = B.fromArray(Array.from({ length: n }, (_, i) => i % C.vocabSize), [BATCH, TOKENS]);
  const tgt = B.fromArray(Array.from({ length: n }, (_, i) => (i + 1) % C.vocabSize), [BATCH, TOKENS]);
  const out = gptForward(C, params, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps?.();
  /* The step is not over until the GPU says so — the same barrier the
   * benchmark applies, or the tail of the step lands in the next one. */
  if (B.flushAndWait) B.flushAndWait();
  else if (B.syncGpu) B.syncGpu();
  return loss;
}
/* Warm up by TIME, for the same reason the benchmark does: this card idles at
 * 210 MHz against 2100 max and cannot be clock locked inside the container. Ten
 * steps is 0.85 s and leaves it part-ramped, which would have had me reading
 * kernel costs that were mostly clock state and attributing them to code. */
{
  const t = process.hrtime.bigint();
  let k = 0;
  while (k < 10 || Number(process.hrtime.bigint() - t) / 1e6 < 3000) { step(params); k++; }
  console.log(`warmup: ${k} steps`);
}

prof.clear();
const before = B.stats();
flushN = 0; flushMs = 0; drainN = 0; drainMs = 0; drainBy.clear();
sites = new Map();
const t0 = process.hrtime.bigint();
const N = 10;
for (let i = 0; i < N; i++) step(params);
const total = Number(process.hrtime.bigint() - t0) / 1e6;
const after = B.stats();

const enq = (after.enqueued - before.enqueued) / N;
const fls = (after.flushes - before.flushes) / N;
console.log(`\nstep ${(total / N).toFixed(1)} ms   enqueued ${enq.toFixed(1)}/step   flushes ${fls.toFixed(1)}/step   ` +
            `enq-per-flush ${(enq / Math.max(fls, 1)).toFixed(2)}`);
console.log(`pool: live ${after.live} pooled ${after.pooled} slabs ${after.allocations} carved ${after.carved} programs ${after.programs}`);
/*
 * THE SPLIT, from the fence rather than from an inference.
 *
 * A flush submits and then SPINS until the GPU signals, so a step is host work
 * plus GPU work with no overlap at all. Spin time is the GPU half exactly;
 * everything else is host. The per-method table below cannot show this, because
 * a flush forced from inside helios_enqueue is charged to whichever operation
 * happened to fill the pushbuffer.
 */
if (after.spinNs !== undefined) {
  const gpuMs = (after.spinNs - before.spinNs) / 1e6 / N;
  const stepMs = total / N;
  console.log(`SPLIT: gpu (fence spin) ${gpuMs.toFixed(1)} ms/step ${(gpuMs / stepMs * 100).toFixed(1)}%   ` +
              `host ${(stepMs - gpuMs).toFixed(1)} ms/step ${((stepMs - gpuMs) / stepMs * 100).toFixed(1)}%   ` +
              `perfect overlap would give ${(stepMs / Math.max(gpuMs, stepMs - gpuMs)).toFixed(2)}x`);
}
console.log(`flush: ${(flushN / N).toFixed(0)} calls/step, ${(flushMs / N).toFixed(2)} ms/step total`);
console.log(`  of which REAL DRAINS (>${DRAIN_US}us): ${(drainN / N).toFixed(1)}/step  ` +
            `${(drainMs / N).toFixed(2)} ms/step  ${((drainMs / N) / (total / N) * 100).toFixed(1)}% of step  ` +
            `${((drainMs / Math.max(drainN, 1)) * 1000).toFixed(0)} us each`);
const dr = [...drainBy.entries()].sort((a, b) => b[1].ms - a[1].ms);
for (const [who, e] of dr)
  console.log(`    ${who.padEnd(16)} ${(e.n / N).toFixed(1).padStart(6)} drains/step ${(e.ms / N).toFixed(2).padStart(8)} ms/step`);
const st = [...sites.entries()].sort((a, b) => b[1].ms - a[1].ms).slice(0, 6);
if (st.length) {
  console.log(`\n  who reads .data outside a backend method:`);
  for (const [where, e] of st)
    console.log(`    ${(e.n / N).toFixed(1).padStart(5)}/step ${(e.ms / N).toFixed(2).padStart(7)} ms  ${where}`);
}
console.log();

const rows = [...prof.entries()]
  .map(([m, e]) => ({ m, n: e.n / N, ms: e.ms / N }))
  .sort((a, b) => b.ms - a.ms);
console.log("method             calls/step    ms/step   %step    us/call");
let acc = 0;
for (const r of rows) {
  if (r.ms < 0.05) continue;
  acc += r.ms;
  console.log(
    `${r.m.padEnd(18)} ${r.n.toFixed(1).padStart(9)} ${r.ms.toFixed(2).padStart(10)} ` +
    `${((r.ms / (total / N)) * 100).toFixed(1).padStart(6)}% ${((r.ms / r.n) * 1000).toFixed(1).padStart(10)}`);
}
console.log(`${"accounted".padEnd(18)} ${"".padStart(9)} ${acc.toFixed(2).padStart(10)} ` +
            `${((acc / (total / N)) * 100).toFixed(1).padStart(6)}%`);
