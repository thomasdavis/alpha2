/*
 * Does the cp.async f16 GEMM actually beat the staged f32 GEMM? THE number.
 *
 * The staged path moves each tile through a register round trip — 12 LDG + 8
 * F2FP + 8 STS — and the SASS says that is 28 of its 42 k-step instructions and
 * the reason it sustains 15-21 TFLOP/s against cuBLAS's 24-32. The cp.async f16
 * GEMM replaces the whole chain with two 128-bit LDGSTS. This measures whether
 * that predicted ~2x on the instruction mix is a real ~2x on the clock.
 *
 * It compares the two RAW kernels at the addon level so nothing above them
 * (autograd, the composed fallback) is in the way:
 *   hl.matmulCpasync  — f16 operands, cp.async staging   (the new kernel)
 *   hl.matmulTransposed — f32 operands, load-pack-store   (the shipping kernel)
 * at the SAME logical shapes and FLOP count, so the ratio is the kernel and
 * nothing else. For timing the operand VALUES do not matter — both do the same
 * work regardless — so the buffers are left as allocated. Correctness is proven
 * separately by the C hardware test (128x64x64 vs an exact reference).
 *
 * The measurement discipline is probe-gemm-rate's, because every piece of it was
 * a bug once: warm by TIME (the card idles at 210 MHz and cannot be locked in
 * the container), evict L2 before each timed run (or you measure the previous
 * case's cache), drain through a ONE-element beacon (a full read copies the
 * result over PCIe inside the timed region), and take a median.
 *
 * Run with HELIOS_VIDMEM=1 (mandatory — without it the operands sit in host
 * memory and every number is a PCIe measurement).
 */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
if (!process.env.HELIOS_VIDMEM)
  console.log("⚠️  HELIOS_VIDMEM is not set — operands are in HOST memory; this is a PCIe measurement.\n");

const buf = (nElems) => NativeBuffer.alloc(hl, nElems);

/* L2 is 4 MB; stream 32 MB through an add to evict a previous case's tiles. */
const SCRUB = 8 << 20;
const scrubA = buf(SCRUB), scrubB = buf(SCRUB);
const evictL2 = () => hl.elementwise(hl.op.add, scrubA.handle, scrubA.handle, scrubB.handle,
                                     SCRUB, 0, 0, 0, 0, 0, 0, 0);
/* The drain is a flush: submit the queue and spin until the GPU is idle. There
 * is no flushAndWait on the low-level addon; hl.flush() is submit-and-wait. */
const drain = () => hl.flush();

/*
 * Ramp the clock ONCE before any case — a per-case warmup from cold understates
 * by up to 4.9x, which is a systematic error, not noise.
 */
function rampClock() {
  const A = buf(1 << 20), B = buf(1 << 20), C = buf(1 << 20);
  const t0 = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t0) / 1e6 < 3000) {
    hl.matmul(C.handle, A.handle, B.handle, 512, 512, 512, 1);
  }
  drain();
  A.free?.(); B.free?.(); C.free?.();
}

const ITERS = 30;
function time(fire) {
  for (let i = 0; i < 5; i++) fire();     /* compile + fill the pool */
  drain();
  const warm = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warm) / 1e6 < 1500) fire();
  drain();
  const runs = [];
  for (let s = 0; s < 5; s++) {
    evictL2(); drain();
    const t = process.hrtime.bigint();
    for (let i = 0; i < ITERS; i++) fire();
    drain();
    runs.push(Number(process.hrtime.bigint() - t) / 1e9);
  }
  runs.sort((a, b) => a - b);
  return runs[2] / ITERS; /* median seconds/call */
}

/* The model's NT forward shapes — the ones the cp.async f16 GEMM targets. */
const SHAPES = [
  ["qkv    ", 1536, 1920, 640],
  ["mlp fc ", 1536, 2560, 640],
  ["lm head", 1536, 12288, 640],
];

console.log("cp.async f16 GEMM vs staged f32 GEMM — NT forward shapes, RTX 3070\n");
console.log("shape                   staged f32        cp.async f16      speedup");
console.log("                        TFLOP/s   us       TFLOP/s   us");

rampClock();

const rows = [];
for (const [name, M, N, K] of SHAPES) {
  const flop = 2 * M * N * K;
  /* f32 operands for the staged kernel; f16-sized for the cp.async one (M*K f16
   * = M*K/2 four-byte slots). C is f32 either way. */
  const aF32 = buf(M * K), bF32 = buf(N * K), cF32 = buf(M * N);
  const aF16 = buf((M * K) >> 1), bF16 = buf((N * K) >> 1), cCp = buf(M * N);

  const secStaged = time(() =>
    hl.matmulTransposed(cF32.handle, aF32.handle, bF32.handle, M, N, K, 1));
  const secCp = time(() =>
    hl.matmulCpasync(cCp.handle, aF16.handle, bF16.handle, M, N, K, 1));

  const tfStaged = flop / secStaged / 1e12, tfCp = flop / secCp / 1e12;
  const speedup = secStaged / secCp;
  rows.push({ name, tfStaged, tfCp, speedup });
  console.log(
    `${name}  ${tfStaged.toFixed(2).padStart(8)} ${(secStaged * 1e6).toFixed(0).padStart(6)}` +
    `    ${tfCp.toFixed(2).padStart(8)} ${(secCp * 1e6).toFixed(0).padStart(6)}` +
    `    ${speedup.toFixed(2)}x`);

  for (const b of [aF32, bF32, cF32, aF16, bF16, cCp]) b.free?.();
}

const g = rows.reduce((s, r) => s + r.speedup, 0) / rows.length;
console.log(`\ngeomeanish speedup ${g.toFixed(2)}x. cuBLAS is 24-32 TFLOP/s at these shapes;`);
console.log("the staged path sits at 15-21. This says how much of that gap cp.async closes,");
console.log("and whether the ~2x the instruction count predicts is real on the clock.");
console.log("(single-buffered — the double-buffered pipeline is the next lever on top.)");
