/* What does a launch cost when the kernel does almost nothing?
 *
 * hermes_barrier puts NVC7C0_WAIT_FOR_IDLE between every dispatch, and qmd.c is
 * explicit that this is the blunt instrument -- it drains the whole pipe rather
 * than expressing which kernel depends on which -- and that replacing it with
 * the QMD's dependent-launch fields "wants a measurement saying the drain costs
 * something". This is that measurement.
 *
 * The step runs 241 launches and the fence says 148 ms of GPU time, which is
 * 614 us a launch. Elementwise kernels on a megabyte should be tens of
 * microseconds, so either they are far slower than they look standalone or most
 * of that 614 us is not the kernel at all.
 *
 * Sweeping SIZE separates those. If a 64-element add costs the same as a
 * 262,144-element one, the cost is per-LAUNCH -- doorbell, QMD fetch, program
 * fetch, and the pipe drain -- and no amount of kernel tuning touches it. The
 * flat part of this curve is the floor the whole backend is standing on, and
 * 241 times it is the budget.
 *
 * Warmed by TIME: this card idles at 210 MHz against 2100 and an iteration
 * count warms nothing, which is how two earlier probes disagreed 5x. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const rand = (n) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 0.5);

function timeOp(fn, iters = 21) {
  fn(); N.syncGpu();
  const until = process.hrtime.bigint() + 1_500_000_000n;
  while (process.hrtime.bigint() < until) { fn(); N.syncGpu(); N.finishStepOps?.(); }
  const ms = [];
  for (let i = 0; i < iters; i++) {
    const t0 = process.hrtime.bigint();
    fn();
    N.syncGpu();
    ms.push(Number(process.hrtime.bigint() - t0) / 1e6);
    N.finishStepOps?.();
  }
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

console.log("\nONE elementwise add, swept by size, each timed with the queue drained\n");
console.log("   elements        MB      ms      GB/s");
let floor = null;
for (const n of [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304]) {
  const a = N.fromArray(rand(n), [n]);
  const b = N.fromArray(rand(n), [n]);
  let ms;
  try { ms = timeOp(() => N.add(a, b)); }
  catch (e) { console.log(`${String(n).padStart(11)}   ERROR ${e.message.slice(0, 34)}`); continue; }
  floor ??= ms;
  const gbs = (n * 12) / (ms / 1000) / 1e9;
  console.log(`${String(n).padStart(11) } ${(n * 4 / 1e6).toFixed(2).padStart(9)} ${ms.toFixed(3).padStart(7)} ${gbs.toFixed(1).padStart(9)}`);
}

console.log(`\nfloor (64 elements, so essentially pure launch): ${floor.toFixed(3)} ms`);
console.log(`a 241-launch step therefore spends at least ${(floor * 241).toFixed(0)} ms in launches alone,`);
console.log(`against a measured GPU half of 148 ms and a whole step of ~193-221 ms.`);

/* And the same launch cost, but with several queued before the drain. If N
 * launches back to back cost N times the floor, they are serialised -- which is
 * what WAIT_FOR_IDLE between every dispatch would do. If they cost meaningfully
 * less, the pipe is already overlapping them and the barrier is cheap. */
console.log("\nqueued back to back, one drain at the end -- do launches overlap?\n");
console.log("   launches      ms   ms/launch   vs floor");
{
  const a = N.fromArray(rand(4096), [4096]);
  const b = N.fromArray(rand(4096), [4096]);
  for (const k of [1, 2, 4, 8, 16, 32]) {
    const ms = timeOp(() => { for (let i = 0; i < k; i++) N.add(a, b); });
    console.log(`${String(k).padStart(11)} ${ms.toFixed(3).padStart(7)} ${(ms / k).toFixed(3).padStart(11)} ${(ms / k / floor).toFixed(2).padStart(10)}x`);
  }
}
