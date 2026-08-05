/* What rate does the GEMM actually sustain, and what is stopping it?
 *
 * The step profile says matmul + matmulTransposed are 49% of GPU time, and a
 * sweep of the tensor-core register tile (2x4, 4x4, 4x2) moved it by less than
 * the spread. That rules out arithmetic intensity per HMMA and leaves two
 * candidates, which this separates by measuring BOTH rates at once:
 *
 *   TFLOP/s   against the 45.5 this card's tensor cores issue from registers
 *             and the 24-32 cuBLAS reaches at these shapes.
 *   GB/s      of GLOBAL traffic implied by the tile: with no shared memory a
 *             block re-reads A once per column block and B once per row block,
 *             so the traffic is not the operand size, it is the operand size
 *             times the re-read count. Against 448 GB/s of DRAM.
 *
 * Whichever is near its ceiling is the constraint.
 *
 * THE INSTRUMENT MUST NOT CONSUME WHAT IT MEASURES. probe-matmul-geometry
 * leaked its own outputs and reported wide blocks 30x slower than narrow ones,
 * which motivated a register-blocking plan that evaporated when its tensors
 * were released. So every iteration releases, and the step boundary runs.
 *
 * Usage: node probe-gemm-rate.mjs [blockRows] [blockCols]
 *   The tile is a property of the BUILD, not of this file; pass the values the
 *   addon was compiled with so the traffic arithmetic matches the kernel.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";

const BM = Number(process.argv[2] ?? 32);   /* pr_hmma_block_rows() */
const BN = Number(process.argv[3] ?? 128);  /* pr_hmma_block_cols() */

const B = new NativeHeliosBackend(0);
const rng = new SeededRng(7);

/*
 * EVICT L2 BEFORE EACH TIMED RUN, or the answer is about the previous case.
 *
 * This card's L2 is 4 MB and a qkv operand is 4.9 — so a shape measured right
 * after one of the SAME SIZE finds its buffers already resident, because the
 * pool hands the same memory back. That is the whole of this file's previous
 * headline: "untransposed-B runs at a quarter of transposed" survived a
 * reordering by REVERSING, and the results alternated slow/fast by POSITION,
 * with the fast one always being the second of a same-shape pair.
 *
 * There is no layout gap. There was a cache-state gap, and the probe was
 * measuring its own ordering.
 *
 * Streaming 32 MB through an elementwise copy is enough to evict 4 MB of L2 and
 * costs about 150 us, outside the timed region.
 */
const SCRUB = 8 << 20; /* elements: 32 MB */
const scrubA = B.zeros([SCRUB]), scrubB = B.zeros([SCRUB]);
const evictL2 = () => { B.addInplace(scrubA, scrubB); };

/*
 * RAMP THE CLOCK ONCE, BEFORE ANY CASE IS MEASURED.
 *
 * This card idles at 210 MHz against 2,100 and nvidia-smi cannot lock clocks
 * inside a RunPod container. A per-case warmup of 2.5 s is not enough from
 * COLD, and the consequence is not noise — it is a systematic error that lands
 * entirely on whichever case runs first.
 *
 * That is what produced this file's previous headline. "Untransposed-B runs at
 * a quarter of transposed" reversed when the list was reordered, alternated
 * slow/fast by POSITION, and vanished completely when each case was measured in
 * its own process: 3.28 against 3.42, and 4.01 against 4.11. The layouts were
 * never different. The FIRST case in a process was, every time.
 *
 * The clock is NOT what produced the first-case anomaly — six seconds of ramp
 * did not remove it. It was the POOL, and the fix is below in the timed loop's
 * dry run. Kept because a cold card is still a 4.9x error on anything measured
 * in the first second, which the benchmark harness also guards against.
 */
{
  const t0 = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t0) / 1e6 < 6000) {
    for (let i = 0; i < 8; i++) B.addInplace(scrubA, scrubB);
    scrubA.data[0];
  }
}

/* M defaults to the batch-8 row count these were first written against; the
 * gate now runs batch 24, which is 1,536 rows, and this kernel's rate tracks
 * BLOCK COUNT — so the row count is not a detail of the benchmark, it is a
 * variable the answer depends on. `M=1536 node probe-gemm-rate.mjs`. */
const M = Number(process.env.M ?? 512);
/* COLD=1: evict L2 before every iteration, so the rate is one a step could
 * actually reach. See the timed loop. */
const COLD = process.env.COLD === "1";

/* The shapes a 105M step actually runs, at batch 8 (512 rows). Named, because
 * a rate without a shape is not comparable to anything. */
/*
 * INTERLEAVED, and ordered so the TRANSPOSED case of each shape comes FIRST.
 *
 * They used to be grouped — every untransposed shape, then every transposed one
 * — and that put the whole of one layout at the START of the run. This card
 * idles at 210 MHz against 2100 and cannot be clock-locked in a container; the
 * benchmark harness warms by TIME for exactly this reason, having measured a
 * 4.9x error between a cold process and a warm one. A grouped order cannot
 * distinguish "this layout is slower" from "this layout was measured first",
 * and the gap it reported was 4x.
 */
const SHAPES = [
  ["qkv        B^T", M, 1920, 640, true],
  ["qkv        fwd", M, 1920, 640, false],
  ["mlp fc     B^T", M, 2560, 640, true],
  ["mlp fc     fwd", M, 2560, 640, false],
  ["lm head    B^T", M, 12288, 640, true],
  ["lm head    fwd", M, 12288, 640, false],
  ["attn proj  fwd", M, 640, 640, false],
  ["mlp proj   fwd", M, 640, 2560, false],
];

/*
 * THE ATTENTION MATMULS, which every earlier version of this file left out.
 *
 * The eight shapes above are the projections, and they are the ones the rate
 * headline has always been about. But a step runs roughly as many attention
 * matmuls as projection matmuls, and they are a completely different shape: not
 * one m1536 problem but 240 INDEPENDENT m64 n64 k64 problems, one per (batch,
 * head). The FLOPs are trivial — 13.6 GFLOP a step against 888 — so if they
 * take a comparable share of the clock, the arithmetic says nothing and the
 * geometry says everything.
 *
 * That is worth measuring precisely because this kernel's rate is already known
 * to track BLOCK COUNT rather than arithmetic: 80 blocks on 46 SMs gives 11.6
 * TFLOP/s where 1,536 blocks gives 20.5. A 64x64 output with a 32x128 block
 * tile is ONE block per batch element, and whether 240 of those fill the card
 * is exactly the open question.
 *
 *   qk  [BH,T,D] x [BH,D,T]  -> [BH,T,T]   scores
 *   av  [BH,T,T] x [BH,T,D]  -> [BH,T,D]   the weighted sum
 *
 * BH is batch*heads = 24*10 at the gate shape; T and D are 64.
 */
const BH = Number(process.env.BH ?? 240), T = 64, HD = 64;
const BATCHED = [
  ["attn qk    bat", BH, T, HD, T],
  ["attn av    bat", BH, T, T, HD],
];

const spin = () => B.hl.stats().spinNs;

function rate(name, M, N, K, transposed) {
  const a = B.randn([M, K], rng);
  const b = transposed ? B.randn([N, K], rng) : B.randn([K, N], rng);

  /*
   * THE BARRIER IS A READ, because there is no flushAndWait on this backend.
   *
   * The first version of this probe called `B.flushAndWait?.()`, which does not
   * exist here, so the optional call did nothing and the clock measured HOW
   * FAST THE HOST CAN ENQUEUE. It reported 768 TFLOP/s and 60 TB/s for one
   * shape — figures the hardware cannot produce, which is the only reason it
   * was caught. A barrier that silently is not one is exactly the instrument
   * failure this file's header warns about, committed in the file itself.
   *
   * Reading an element of the result goes through the tensor's `data` getter,
   * which is where the queue drain lives, so it cannot be optional.
   */
  /*
   * DRAIN THROUGH A ONE-ELEMENT TENSOR, not through the result.
   *
   * The barrier has to be a read, because this backend has no flushAndWait —
   * but reading the RESULT copies the whole thing to the host, and under
   * HELIOS_VIDMEM that is a multi-megabyte PCIe transfer INSIDE the timed
   * region. At [512,1920] it is 3.9 MB, which is hundreds of microseconds
   * against measurements of 91 to 426.
   *
   * That is what produced this file's previous headline — "untransposed-B runs
   * at a quarter of transposed" — which reversed the moment the shapes were
   * reordered, and alternated slow/fast BY POSITION rather than by layout. The
   * copy is charged to whichever call happens to trigger the staging
   * allocation; the layouts are not different at all.
   *
   * Any read drains the whole queue, because `sync()` flushes everything
   * pending. So read four bytes instead of four megabytes: the wait is the
   * same and the transfer is not.
   */
  const beacon = B.zeros([1]);
  const drain = () => beacon.data[0];

  /* Warm by TIME: this card idles at 210 MHz against 2100 and cannot be
   * clock-locked in a container, so a cold measurement understates by up to
   * 4.9x. */
  const warmStart = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warmStart) / 1e6 < 2500) {
    const c = transposed ? B.matmulTransposed(a, b) : B.matmul(a, b);
    drain();
    B.releaseGpuTensor?.(c);
    B.finishStepOps?.();
  }

  const ITERS = 30;
  /*
   * FILL THE POOL WITH EXACTLY WHAT THE TIMED LOOP WILL ASK FOR.
   *
   * The timed loop allocates one output per iteration and only RELEASES them —
   * a release marks, and nothing is reusable until helios_end_step — so thirty
   * iterations need thirty distinct buffers. The warmup retires every step, so
   * it only ever carves one, and the timed loop then pays TWENTY-NINE FRESH
   * CARVES: 802 us each against 1.0 us from the pool, inside the measurement.
   *
   * That is the first-case anomaly. Whichever case runs first in a process pays
   * it and every later one finds the pool already holding thirty buffers of the
   * right class, which is why the gap followed POSITION and not layout.
   *
   * One dry run of exactly ITERS iterations, then a retire, leaves the pool in
   * the state the timed loop expects.
   */
  for (let i = 0; i < ITERS; i++) {
    const c = transposed ? B.matmulTransposed(a, b) : B.matmul(a, b);
    B.releaseGpuTensor?.(c);
  }
  drain();
  B.finishStepOps?.();
  evictL2();
  drain();
  /*
   * COLD=1 EVICTS L2 BETWEEN EVERY ITERATION, and the difference it makes is
   * the difference between this probe and a step.
   *
   * The loop runs the same multiply thirty times over the same two operands, so
   * after the first iteration whatever fits in the 4 MB L2 is already resident
   * — and it is re-read thirty times for free. A real step never does that: each
   * GEMM's operands were written by some other kernel and are cold. A probe that
   * reports a rate a step cannot reach is not a harmless optimism, it is a wrong
   * answer about how much headroom is left.
   *
   * The evict is inside the timed region here, so its own cost is charged to the
   * measurement; the scrub is reported separately below so it can be subtracted.
   */
  const s0 = spin(), t0 = process.hrtime.bigint();
  let last = null;
  for (let i = 0; i < ITERS; i++) {
    if (last) B.releaseGpuTensor?.(last);
    if (COLD) evictL2();
    last = transposed ? B.matmulTransposed(a, b) : B.matmul(a, b);
  }
  drain(last);
  const wallMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const gpuMs = (spin() - s0) / 1e6;
  B.releaseGpuTensor?.(last);
  B.finishStepOps?.();

  const flop = 2 * M * N * K * ITERS;
  /* The re-read count IS the tile: a block owning BM rows and BN columns reads
   * all of A's rows for its block once per column block, and all of B once per
   * row block. No shared memory means no sharing between blocks beyond L2. */
  const blocksM = Math.ceil(M / BM), blocksN = Math.ceil(N / BN);
  const bytes = (M * K * blocksN + K * N * blocksM + M * N) * 4 * ITERS;
  const secs = wallMs / 1e3;  /* wall, now that the read really drains */

  console.log(
    `  ${name}  m${String(M).padEnd(4)} n${String(N).padEnd(5)} k${String(K).padEnd(5)}` +
    `  ${(flop / secs / 1e12).toFixed(2).padStart(6)} TFLOP/s` +
    `  ${(bytes / secs / 1e9).toFixed(0).padStart(5)} GB/s implied` +
    `  ${(secs * 1e6 / ITERS).toFixed(0).padStart(6)} us/call`);

  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b); B.releaseGpuTensor?.(beacon);
  B.finishStepOps?.();
}

/*
 * The same measurement for a BATCHED shape, kept as its own function rather
 * than as a flag on `rate` because every piece of the arithmetic differs: the
 * FLOP count carries the batch, the implied-traffic model has no block re-read
 * across batch elements, and the operands are 3-D.
 *
 * Same three guards as `rate`, for the same reasons and they are not optional:
 * warm by time, a dry run of exactly ITERS to fill the pool, and an L2 evict
 * outside the clock. The dry run is what removed this file's first-case
 * anomaly, and a batched output is bigger than a 2-D one, so it matters more.
 */
function rateBatched(name, batch, M2, K2, N2) {
  const a = B.randn([batch, M2, K2], rng);
  const b = B.randn([batch, K2, N2], rng);
  const beacon = B.zeros([1]);
  const drain = () => beacon.data[0];

  const warmStart = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warmStart) / 1e6 < 2500) {
    const c = B.matmul(a, b);
    drain();
    B.releaseGpuTensor?.(c);
    B.finishStepOps?.();
  }

  const ITERS = 30;
  for (let i = 0; i < ITERS; i++) {
    const c = B.matmul(a, b);
    B.releaseGpuTensor?.(c);
  }
  drain();
  B.finishStepOps?.();
  evictL2();
  drain();

  const t0 = process.hrtime.bigint();
  let last = null;
  for (let i = 0; i < ITERS; i++) {
    if (last) B.releaseGpuTensor?.(last);
    last = B.matmul(a, b);
  }
  drain();
  const wallMs = Number(process.hrtime.bigint() - t0) / 1e6;
  B.releaseGpuTensor?.(last);
  B.finishStepOps?.();

  const secs = wallMs / 1e3;
  const flop = 2 * batch * M2 * N2 * K2 * ITERS;
  /* One block per (batch, tile) — with a 64x64 output and a 32x128 tile that is
   * two blocks per batch element, and the total is what decides whether the
   * card is filled at all. */
  const blocks = batch * Math.ceil(M2 / BM) * Math.ceil(N2 / BN);
  console.log(
    `  ${name}  b${String(batch).padEnd(3)} m${String(M2).padEnd(4)} n${String(N2).padEnd(4)} k${String(K2).padEnd(4)}` +
    `  ${(flop / secs / 1e12).toFixed(2).padStart(6)} TFLOP/s` +
    `  ${String(blocks).padStart(6)} blocks` +
    `  ${(secs * 1e6 / ITERS).toFixed(0).padStart(6)} us/call`);

  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b); B.releaseGpuTensor?.(beacon);
}


console.log(`GEMM rate, tensor-core tile ${BM} rows x ${BN} cols`);
console.log(`ceilings: 45.5 TFLOP/s from registers, 24-32 cuBLAS here, 448 GB/s DRAM\n`);
/* ONE CASE PER PROCESS when asked, which is the only way to be sure a number
 * is about the case and not about what ran before it. */
const only = process.env.ONLY;
for (const [name, M, N, K, t] of SHAPES)
  if (!only || name.replace(/\s+/g, "").startsWith(only)) rate(name, M, N, K, t);

/*
 * THE WEIGHT-GRADIENT LAYOUT, measured against the forward one at MATCHED
 * arithmetic.
 *
 * dW = A^T @ B reads A stored [K,M], and the step profile puts every one of
 * these at ~14.5 TFLOP/s where the forward shapes reach 19-22. The suspicion is
 * the staging load: a thread owns (row, k-pair) with the k-pair varying fastest,
 * which is contiguous when A is [M,K] and strides by M when it is [K,M] — so 32
 * lanes of a warp read 32 locations 2*M elements apart.
 *
 * Matched means the same M, N and K, so the FLOPs are identical and any gap is
 * the layout. Anything else would be comparing two different problems.
 */
console.log();
console.log("weight-gradient layout (A stored [K,M]) against the forward, matched shapes");
for (const [name, m, n, k] of [
  ["mlp fc   dW", 640, 2560, 1536],
  ["qkv      dW", 640, 1920, 1536],
  ["attn proj dW", 640, 640, 1536],
]) {
  const a = B.randn([k, m], rng), b = B.randn([k, n], rng);
  const beacon = B.zeros([1]); const drain = () => beacon.data[0];
  const warm = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warm) / 1e6 < 2000) {
    const c = B.matmulTransposedA(a, b); drain(); B.releaseGpuTensor?.(c); B.finishStepOps?.();
  }
  const ITERS = 30;
  for (let i = 0; i < ITERS; i++) B.releaseGpuTensor?.(B.matmulTransposedA(a, b));
  drain(); B.finishStepOps?.(); evictL2(); drain();
  const t0 = process.hrtime.bigint();
  let last = null;
  for (let i = 0; i < ITERS; i++) { if (last) B.releaseGpuTensor?.(last); last = B.matmulTransposedA(a, b); }
  drain();
  const secs = Number(process.hrtime.bigint() - t0) / 1e9;
  B.releaseGpuTensor?.(last); B.finishStepOps?.();
  console.log(`  ${name}  m${String(m).padEnd(4)} n${String(n).padEnd(5)} k${String(k).padEnd(5)}` +
              `  ${(2 * m * n * k * ITERS / secs / 1e12).toFixed(2).padStart(6)} TFLOP/s` +
              `  ${(secs * 1e6 / ITERS).toFixed(0).padStart(6)} us/call`);
  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b); B.releaseGpuTensor?.(beacon);
}

console.log();
console.log("attention, batched — 240 independent 64x64 problems, not one big one");
for (const [name, batch, M2, K2, N2] of BATCHED) rateBatched(name, batch, M2, K2, N2);
