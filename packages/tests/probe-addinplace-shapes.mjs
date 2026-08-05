/* addInplace, at the shapes a step actually runs, measured in isolation.
 *
 * The step profile charges 382 us to `addInplace [24,64,640]` and 466 us to
 * `addInplace [2560,640]`. Those are 983,040 and 1,638,400 elements; at three
 * accesses each (two reads and a write) against this card's 448 GB/s they are
 * 26 and 44 us of work. The standing record for this op is 340-417 GB/s at four
 * other shapes, so either that record does not cover these or something about
 * them is different.
 *
 * Measured here in isolation so the answer cannot be about queue state,
 * neighbouring ops, or the drain the per-op profiler inserts. Same three guards
 * as the GEMM probe and for the same reasons: warm by time because this card
 * idles at 210 MHz, a dry run to fill the pool because a fresh carve is 802 us
 * against 1.0 from the pool, and a beacon read rather than a result read so the
 * barrier does not drag a multi-megabyte transfer into the timed region.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";

const B = new NativeHeliosBackend(0);
const rng = new SeededRng(7);
const beacon = B.zeros([1]);
const drain = () => beacon.data[0];

const SHAPES = [
  ["residual stream", [24, 64, 640]],
  ["residual, 2-D  ", [1536, 640]],
  ["mlp down weight", [2560, 640]],
  ["mlp down, other", [640, 2560]],
  ["lm head weight ", [12288, 640]],
  ["small control  ", [512, 640]],
];

/* Ramp the clock once, before anything is measured. */
{
  const a = B.zeros([8 << 20]), b = B.zeros([8 << 20]);
  const t0 = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t0) / 1e6 < 4000) { B.addInplace(a, b); drain(); }
  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b); B.finishStepOps?.();
}

console.log("addInplace — 3 accesses per element against 448 GB/s\n");
for (const [name, shape] of SHAPES) {
  const n = shape.reduce((x, y) => x * y, 1);
  const a = B.randn(shape, rng), b = B.randn(shape, rng);

  const warm = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warm) / 1e6 < 1200) { B.addInplace(a, b); drain(); }
  B.finishStepOps?.();

  const ITERS = 50;
  for (let i = 0; i < ITERS; i++) B.addInplace(a, b);
  drain();
  B.finishStepOps?.();

  const t0 = process.hrtime.bigint();
  for (let i = 0; i < ITERS; i++) B.addInplace(a, b);
  drain();
  const us = Number(process.hrtime.bigint() - t0) / 1e3 / ITERS;
  const gbs = n * 3 * 4 / (us * 1e-6) / 1e9;
  console.log(`  ${name}  ${JSON.stringify(shape).padEnd(16)} ${String(n).padStart(9)} elems` +
              `  ${us.toFixed(1).padStart(7)} us  ${gbs.toFixed(0).padStart(4)} GB/s`);
  B.releaseGpuTensor?.(a); B.releaseGpuTensor?.(b); B.finishStepOps?.();
}
