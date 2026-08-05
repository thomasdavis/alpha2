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

/*
 * matmulTransposedA and matmulAccumulate were MISSING and they are not a minor
 * omission: dW = G^T @ A is one GEMM per weight per layer, so roughly a third
 * of the step's matrix multiplies never appeared in this profile at all. The
 * table looked complete because every line in it was real.
 */
const METHODS = [
  /* EXHAUSTIVE, and it has to stay that way. An UNTRACKED op's GPU time does
   * not vanish — the wrapper reads the spin counter around each tracked call,
   * so work enqueued in between is charged to whichever tracked op runs NEXT.
   * That has now misled this file three times: first matmulTransposedA and
   * matmulAccumulate (a third of all GEMMs), then the fused attention kernels,
   * each time inflating a GEMM row and pointing the next day's work at the
   * wrong thing. The coverage check below is what stops a fourth. */
  "add","sub","mul","div","neg","gelu","exp","log","sqrt","scale","clamp",
  "matmul","matmulTransposed","matmulTransposedA","matmulAccumulate",
  "columnSum","embeddingBackward","crossEntropyBackward","layerNormBackward",
  "sum","mean","layerNorm","rmsNorm","softmax","softCap",
  "embedding","crossEntropy","transpose","slice","causalMask","maskedFill","cat","zeros",
  "ones","full","clone","broadcast","addInplace",
  "softmaxMasked","softmaxBackward","softCapBackward","geluBackward","clampBackward",
  "mulInto","expand","permuteSwap","reduceAll","reduceAxis","relu","silu",
  "unary","binary","writeColumns","normalized","normalizedAffine",
];
const cost = new Map();
/* SHAPES=1 adds the per-shape breakdown; off by default so the headline table
 * stays readable. */
const SHAPES = process.env.SHAPES === "1";
const shapeCost = new Map();
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
    /*
     * ALSO BY SHAPE, because "matmul: 264 us/call" is an average over two
     * populations that have nothing to do with each other. A step runs ~110
     * projection matmuls at m1536 and ~110 attention matmuls that are 240
     * independent 64x64 problems, and the second kind sustains 3.4 TFLOP/s
     * against the first kind's 20. An average over both describes neither, and
     * the decision this profile exists to inform — what to fuse or restructure
     * next — depends entirely on which of them holds the clock.
     */
    if (SHAPES) {
      const dims = (t) => (t && t.shape ? t.shape.join("x") : "?");
      const key = `${m} ${dims(args[0])} . ${dims(args[1])}`;
      const se = shapeCost.get(key) ?? { n: 0, us: 0 };
      se.n++; se.us += us; shapeCost.set(key, se);
    }
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

/*
 * COVERAGE, checked rather than assumed.
 *
 * The step's own spin counter is GPU time by construction, and this profile is
 * the same GPU time attributed per op — so the two have to agree to within the
 * overhead draining adds. If the profiled total is well BELOW the step's, some
 * op is launching kernels this file does not wrap, and its time is silently
 * inside a neighbour's row rather than missing from the table. That failure has
 * cost three wrong conclusions here, and it is invisible without this line.
 */
{
  const spin0 = B.hl.stats().spinNs;
  const tape2 = new Tape();
  const tok2 = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>i%C.vocabSize),[BATCH,SEQ]);
  const tgt2 = B.fromArray(Array.from({length:BATCH*SEQ},(_,i)=>(i+1)%C.vocabSize),[BATCH,SEQ]);
  const o2 = gptForward(C, P, B, tape2, tok2, tgt2, true, false, false, undefined, rel);
  tape2.backward(o2.loss, B, rel);
  tape2.clear(rel); B.finishStepOps?.();
  const stepMs = (B.hl.stats().spinNs - spin0) / 1e6;
  const ratio = total / 1000 / stepMs;
  console.log(`coverage: ${(total/1000).toFixed(1)} ms profiled against ${stepMs.toFixed(1)} ms of ` +
              `undrained step GPU (${ratio.toFixed(2)}x). Draining removes overlap so >1 is expected; ` +
              `${ratio < 1 ? "BELOW 1 MEANS OPS ARE MISSING FROM THE TABLE." : "no ops appear to be missing."}`);
}
console.log(`${L}L seq ${SEQ} batch ${BATCH}: ${calls} operations, ${(total/1000).toFixed(1)} ms of GPU (drained per op), wall ${Date.now()-t0} ms\n`);
console.log("operation           calls    GPU ms    %     us/call");
for (const [k, v] of [...cost.entries()].sort((a,b) => b[1].us - a[1].us).slice(0, 12))
  console.log(`${k.padEnd(18)} ${String(v.n).padStart(6)}  ${(v.us/1000).toFixed(1).padStart(8)}  ${(v.us/total*100).toFixed(1).padStart(5)}  ${(v.us/v.n).toFixed(0).padStart(8)}`);

if (SHAPES) {
  console.log("\nby shape (top 20) — the operand dimensions, not the op name");
  console.log("operation / operands                                calls    GPU ms    %     us/call");
  for (const [k, v] of [...shapeCost.entries()].sort((a, b) => b[1].us - a[1].us).slice(0, 20))
    console.log(`${k.padEnd(50)} ${String(v.n).padStart(5)}  ${(v.us / 1000).toFixed(1).padStart(8)}` +
                `  ${(v.us / total * 100).toFixed(1).padStart(5)}  ${(v.us / v.n).toFixed(0).padStart(8)}`);

  /*
   * THE RATE EACH SHAPE ACTUALLY ACHIEVES, and the time it would give back at
   * the best rate this kernel reaches anywhere.
   *
   * "GEMM is 72% of the step" does not say what to do. A shape at 22 TFLOP/s is
   * finished; one at 12 has headroom worth naming. The last column is the
   * decision: milliseconds recoverable if this shape ran at PEAK, summed over
   * its calls.
   */
  const PEAK = 22.0;
  const rows = [];
  for (const [k, v] of shapeCost) {
    const m = k.match(/^(matmul\w*) (\S+) \. (\S+)$/);
    if (!m) continue;
    const a = m[2].split("x").map(Number), b = m[3].split("x").map(Number);
    if (a.some(Number.isNaN) || b.some(Number.isNaN) || a.length < 2) continue;
    /* Leading axes are batch; the last two are the matrix. The operand shapes
     * are as PASSED, so the contraction is a[-1] against whichever of b's last
     * two axes matches — transposed forms store B as [N,K]. */
    const batch = a.slice(0, -2).reduce((x, y) => x * y, 1);
    let M, N, K;
    if (m[1] === "matmulAccumulate") {
      /* ⚠️ ITS FIRST ARGUMENT IS THE DESTINATION, not an operand — the wrapper
       * logs args[0] and args[1] blindly, so for this one they are (dest, A)
       * and the contraction is nowhere in them. dest is [M,N] and A is stored
       * [K,M], which is the transposed-A form the weight gradient uses. Reading
       * it as a plain (A, B) pair put this row at 7.0 TFLOP/s and top of the
       * table; it is 16.8 and unremarkable. */
      M = a[a.length - 2]; N = a[a.length - 1]; K = b[0];
    } else if (m[1] === "matmulTransposedA") {
      /* C = A^T @ B with A stored [K,M]: args are (A, B). */
      K = a[a.length - 2]; M = a[a.length - 1]; N = b[b.length - 1];
    } else {
      M = a[a.length - 2]; K = a[a.length - 1];
      const bl = b[b.length - 1], bp = b[b.length - 2];
      N = m[1] === "matmulTransposed" ? bp : (bl === K ? bp : bl);
    }
    const gflop = 2 * batch * M * N * K / 1e9 * v.n;
    const tflops = gflop / (v.us / 1e6) / 1e3;
    if (!Number.isFinite(tflops) || tflops <= 0 || tflops > 60) continue;
    rows.push({ k, n: v.n, ms: v.us / 1000, gflop, tflops,
                recover: (v.us / 1000) * (1 - Math.min(1, tflops / PEAK)) });
  }
  rows.sort((x, y) => y.recover - x.recover);
  console.log(`\nGEMM by shape — rate achieved, and ms recoverable at ${PEAK} TFLOP/s`);
  console.log("operation / operands                              calls   GPU ms   GFLOP  TFLOP/s  recover");
  let totalRecover = 0, totalMs = 0, totalG = 0;
  for (const r of rows) {
    totalRecover += r.recover; totalMs += r.ms; totalG += r.gflop;
    console.log(`${r.k.padEnd(48)} ${String(r.n).padStart(5)} ${r.ms.toFixed(1).padStart(8)}` +
                ` ${r.gflop.toFixed(1).padStart(7)} ${r.tflops.toFixed(1).padStart(8)}` +
                ` ${r.recover.toFixed(1).padStart(8)}`);
  }
  console.log(`${"TOTAL".padEnd(48)} ${"".padStart(5)} ${totalMs.toFixed(1).padStart(8)}` +
              ` ${totalG.toFixed(1).padStart(7)} ${(totalG / (totalMs / 1000) / 1e3).toFixed(1).padStart(8)}` +
              ` ${totalRecover.toFixed(1).padStart(8)}`);

  /*
   * THE BUDGET, which is what the whole table is for.
   *
   * A target throughput fixes the step time, and the step time has to cover the
   * GEMM before it covers anything else. When the GEMM alone exceeds the budget,
   * no amount of fusion in the other half can reach the target and the only
   * remaining lever is the GEMM's own rate.
   */
  const TARGET = Number(process.env.TARGET_TOKS ?? 30000);
  const stepMs = BATCH * SEQ / TARGET * 1000;
  /* The drained profile overstates, because draining removes the overlap a real
   * step gets. Scale by the ratio the benchmark reports so the comparison is
   * against real GPU time and not against this instrument's own total. */
  const gemmMs = rows.reduce((a, r) => a + r.ms, 0);
  console.log(`\nBUDGET at ${TARGET} tok/s: the step must be ${stepMs.toFixed(1)} ms.`);
  console.log(`  GEMM alone is ${gemmMs.toFixed(1)} ms drained (${totalG.toFixed(0)} GFLOP at ` +
              `${(totalG / (gemmMs / 1000) / 1e3).toFixed(1)} TFLOP/s).`);
  console.log(`  ${TARGET} tok/s needs ${(totalG / (stepMs / 1000) / 1e3).toFixed(1)} TFLOP/s ` +
              `sustained across the WHOLE step — every elementwise op, launch and barrier included.`);

  /* The two populations, summed. This is the number the next decision turns on. */
  let big = { n: 0, us: 0 }, small = { n: 0, us: 0 };
  for (const [k, v] of shapeCost) {
    if (!/^matmul/.test(k)) continue;
    /* An attention operand carries the batch*head dimension, so it has three
     * axes where a projection operand has two. */
    const t = (k.split(" ")[1] ?? "").split("x").length >= 3 ? small : big;
    t.n += v.n; t.us += v.us;
  }
  console.log(`\n  projection GEMMs  ${String(big.n).padStart(4)} calls  ${(big.us / 1000).toFixed(1).padStart(6)} ms` +
              `  ${(big.us / total * 100).toFixed(1)}% of GPU`);
  console.log(`  attention  GEMMs  ${String(small.n).padStart(4)} calls  ${(small.us / 1000).toFixed(1).padStart(6)} ms` +
              `  ${(small.us / total * 100).toFixed(1)}% of GPU`);
}
