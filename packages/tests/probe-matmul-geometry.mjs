/* Which variable makes one matmul 5.5x faster than another that does the same work?
 *
 * mlp-up [4096,64]x[64,256] and mlp-dn [4096,256]x[256,64] are both 67M fused
 * multiply-adds and measure 5.57 ms against 1.02 ms. But they differ in THREE
 * things at once -- N (which is the block width), K (which is the loop length)
 * and M (which is the block count) -- so the earlier reading cannot say which
 * one matters, and writing a register-blocked kernel on the wrong one would be
 * expensive.
 *
 * The kernel is gridX = M blocks of blockX = N threads, each looping K. So hold
 * M*N*K fixed at 67,108,864 and vary which factor carries the size:
 *
 *   N=256 K=64  M=4096    wide blocks, short loop     (mlp-up's shape)
 *   N=64  K=256 M=4096    narrow blocks, long loop    (mlp-dn's shape)
 *   N=256 K=256 M=1024    wide blocks, long loop, few blocks
 *   N=64  K=64  M=16384   narrow blocks, short loop, many blocks
 *
 * If the narrow-block rows are fast and the wide ones slow regardless of K,
 * it is the per-thread preamble and the fix is register blocking: give each
 * thread several output columns so there are fewer threads doing more each.
 * If instead it tracks K, it is loop overhead and the fix is unrolling. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const rand = (n) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 0.5);
const mk = (shape) => N.fromArray(rand(shape.reduce((a, b) => a * b, 1)), shape);

/*
 * ONE operation per sync, and a MEDIAN.
 *
 * The first version of this queued 20 launches and divided by 20, and it
 * disagreed with an earlier probe by 5x on identical shapes -- 1.02 ms against
 * 4.96 for [4096,256]x[256,64]. Queuing 20 launches measures the queue as much
 * as the kernel: the ring holds a bounded number of slots, so a deep enough
 * batch drains in the middle and serialises, and each iteration allocates a
 * fresh 4 MB output so the pool's carving state moves under the measurement
 * too. Both vary with what ran before, which is exactly how the same shape
 * reads differently in two probes.
 *
 * An instrument that disagrees with itself by 5x cannot evaluate a 2x
 * optimisation, so: enqueue one, drain, time that, and report the median with
 * the spread so an unstable reading is visible rather than averaged away.
 */
function timeOp(fn, iters = 15) {
  /*
   * WARM UP BY TIME, NOT BY ITERATION COUNT.
   *
   * This card idles at 210 MHz against 2100 and cannot be clock-locked inside
   * the container, so a warmup of five iterations of a sub-millisecond kernel
   * warms nothing -- it finishes long before the clock ramps. That is what made
   * two passes of this very probe disagree by 5x on identical shapes: the first
   * measured a cold card and the second, run seconds later, measured a boosted
   * one. The tell was that the EARLY cases moved between passes and the late
   * ones did not.
   *
   * bench-scale.mjs already warms by time for exactly this reason. Two seconds
   * per case is enough to reach and hold boost.
   */
  /*
   * RELEASE THE OUTPUT, or the probe dies before it answers.
   *
   * Every call allocates a fresh output and nothing freed it, so a two-second
   * warmup of a sub-millisecond kernel carves thousands of them and the pool
   * runs dry — this probe crashed on its THIRD case with "allocation of
   * 1048576 floats failed", which reads like a device limit and is really this
   * file's own litter. The three cases it never reached are the ones that
   * separate N from K, i.e. the entire question it exists to answer.
   */
  const drop = (r) => { if (r && r.buffer) N.releaseGpuTensor?.(r); };
  drop(fn()); N.syncGpu();
  const warmUntil = process.hrtime.bigint() + 2_000_000_000n;
  while (process.hrtime.bigint() < warmUntil) { drop(fn()); N.syncGpu(); N.finishStepOps?.(); }
  const ms = [];
  for (let i = 0; i < iters; i++) {
    const t0 = process.hrtime.bigint();
    const r = fn();
    N.syncGpu();
    ms.push(Number(process.hrtime.bigint() - t0) / 1e6);
    drop(r);
    N.finishStepOps?.();
  }
  ms.sort((a, b) => a - b);
  return { med: ms[ms.length >> 1], lo: ms[0], hi: ms[ms.length - 1] };
}

const FMA = 4096 * 256 * 64; /* 67,108,864, held constant */
const CASES = [
  { M: 4096, K: 64, Ncols: 256, note: "mlp-up shape" },
  { M: 4096, K: 256, Ncols: 64, note: "mlp-dn shape" },
  { M: 1024, K: 256, Ncols: 256, note: "wide + long" },
  { M: 16384, K: 64, Ncols: 64, note: "narrow + short" },
  { M: 2048, K: 128, Ncols: 256, note: "wide, mid" },
  { M: 8192, K: 128, Ncols: 64, note: "narrow, mid" },
  { M: 4096, K: 128, Ncols: 128, note: "square-ish" },
];

console.log("\n67.1M fused multiply-adds every row; only the geometry changes\n");
console.log("  blockX=N   loop=K   blocks=M    med ms      spread   Gflop/s   note");
for (const c of CASES) {
  const a = mk([c.M, c.K]);
  const b = mk([c.K, c.Ncols]);
  let t;
  try { t = timeOp(() => N.matmul(a, b)); }
  catch (e) { console.log(`  ${String(c.Ncols).padStart(8)} ${String(c.K).padStart(8)} ${String(c.M).padStart(9)}   ERROR ${e.message.slice(0, 28)}`); continue; }
  const gflops = (2 * FMA) / (t.med / 1000) / 1e9;
  N.releaseGpuTensor?.(a); N.releaseGpuTensor?.(b); N.finishStepOps?.();
  console.log(`  ${String(c.Ncols).padStart(8)} ${String(c.K).padStart(8)} ${String(c.M).padStart(9)} ${t.med.toFixed(2).padStart(9)} ${(t.lo.toFixed(2) + "-" + t.hi.toFixed(2)).padStart(12)} ${gflops.toFixed(0).padStart(9)}   ${c.note}`);
}
console.log("\nAn RTX 3070 does ~20,000 Gflop/s f32 at peak.");
