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

/* The shapes a 105M step actually runs, at batch 8 (512 rows). Named, because
 * a rate without a shape is not comparable to anything. */
const SHAPES = [
  ["qkv        fwd", 512, 1920, 640, false],
  ["attn proj  fwd", 512, 640, 640, false],
  ["mlp fc     fwd", 512, 2560, 640, false],
  ["mlp proj   fwd", 512, 640, 2560, false],
  ["lm head    fwd", 512, 12288, 640, false],
  ["qkv        B^T", 512, 1920, 640, true],
  ["mlp fc     B^T", 512, 2560, 640, true],
  ["lm head    B^T", 512, 12288, 640, true],
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
  const drain = (c) => c.data[0];

  /* Warm by TIME: this card idles at 210 MHz against 2100 and cannot be
   * clock-locked in a container, so a cold measurement understates by up to
   * 4.9x. */
  const warmStart = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warmStart) / 1e6 < 2500) {
    const c = transposed ? B.matmulTransposed(a, b) : B.matmul(a, b);
    drain(c);
    B.releaseGpuTensor?.(c);
    B.finishStepOps?.();
  }

  const ITERS = 30;
  const s0 = spin(), t0 = process.hrtime.bigint();
  let last = null;
  for (let i = 0; i < ITERS; i++) {
    if (last) B.releaseGpuTensor?.(last);
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

  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b);
  B.finishStepOps?.();
}

console.log(`GEMM rate, tensor-core tile ${BM} rows x ${BN} cols`);
console.log(`ceilings: 45.5 TFLOP/s from registers, 24-32 cuBLAS here, 448 GB/s DRAM\n`);
for (const [name, M, N, K, t] of SHAPES) rate(name, M, N, K, t);
