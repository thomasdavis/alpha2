/* What do the NORMALISE kernels actually cost, measured against the roofline?
 *
 * WHY THIS FILE EXISTS RATHER THAN A ROW IN THE STEP PROFILE. The drained
 * per-op profiler says layerNorm 91 us/call and layerNormBackward 189 at
 * [1536,640], which would make the pair 11% of a step. That profiler has
 * misled this stack three times by charging an untracked op's GPU time to
 * whichever tracked op ran next, and it inflates every row by the ~10 us a
 * drain costs. Neither error is small next to 91.
 *
 * So this measures the same kernels the way micro-vidmem-bandwidth measures an
 * elementwise add: allocate, fire N times, drain ONCE, divide. No drain per
 * call, nothing else in the queue, the same shape the model runs.
 *
 * WHAT MAKES IT A ROOFLINE RATHER THAN A NUMBER: every case declares the bytes
 * it MUST move — its inputs read once and its outputs written once — and the
 * table prints the implied GB/s beside a copy control measured on the same
 * card in the same process. A kernel at the control's bandwidth is finished. A
 * kernel at a fifth of it has a factor in it, and the factor is not arithmetic:
 * a layer norm is two adds and a multiply per element.
 *
 * Usage: HELIOS_VIDMEM=1 node micro-norm-bandwidth.mjs [rows] [width]
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const ROWS = Number(process.argv[2] ?? 1536); /* batch 24 x seq 64 */
const WIDTH = Number(process.argv[3] ?? 640); /* nEmbd */
const B = new NativeHeliosBackend(0);

if (!process.env.HELIOS_VIDMEM)
  console.log("⚠️  HELIOS_VIDMEM is not set — tensors are in HOST memory and " +
              "every number below is a PCIe measurement, not a kernel one.\n");

const fill = (n, f) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = f(i); return a; };
const mk = (shape, f) => B.fromArray(fill(shape.reduce((x, y) => x * y, 1), f), shape);

/* Deterministic, and small enough that a variance is not dominated by one
 * outlier — the kernels are bandwidth-bound, so the VALUES do not matter to the
 * timing, but a NaN would and a denormal might. */
const x = mk([ROWS, WIDTH], (i) => Math.sin(i * 0.017) * 0.9);
const g = mk([ROWS, WIDTH], (i) => Math.cos(i * 0.013) * 0.1);
const w = mk([WIDTH], (i) => 1 + 0.01 * Math.sin(i));
const bias = mk([WIDTH], (i) => 0.001 * i);
const scratch = mk([ROWS, WIDTH], () => 0);

const F32 = 4;
const ELEMS = ROWS * WIDTH;

/*
 * The measurement. Fire REPS times into the queue and drain once, so what is
 * timed is the kernels back to back rather than one kernel plus a fence.
 *
 * Releasing inside the loop is deliberate and is not a leak: helios_tensor_free
 * only MARKS, and retire runs at the step boundary, so without it REPS
 * iterations carve REPS fresh buffers and the first case pays ~800 us a carve
 * INSIDE the measurement. That is the exact defect that made probe-gemm-rate
 * report one layout 4.5x slower than the other for a year.
 */
const REPS = 50;
function time(label, bytes, fire) {
  /* One iteration = fire, mark the output free, retire. WITHOUT the retire the
   * warm-up exhausts the pool and the run dies at "allocation of 983040 floats
   * failed" — which is what happened the first time this file ran, and is the
   * same fact probe-gemm-rate records: a release MARKS, and nothing is reusable
   * until endStep. */
  const once = () => { const t = fire(); if (t) B.releaseGpuTensor(t); };

  for (let i = 0; i < 5; i++) once();
  B.hl.flush(); B.hl.endStep();

  /* The card idles at 210 MHz against 2100 and cannot be clock-locked in this
   * container, so warm by TIME and not by iteration count. */
  const warm = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warm) / 1e6 < 1500) {
    once(); B.hl.flush(); B.hl.endStep();
  }

  /*
   * A DRY RUN OF EXACTLY REPS ITERATIONS, then a retire — probe-gemm-rate's
   * fix, and it is not optional. A measured run enqueues REPS outputs before
   * anything retires, so it needs REPS distinct buffers; if the pool holds
   * fewer, the shortfall is CARVED at ~800 us each, 800x a pool hit, inside the
   * measurement. That defect made one GEMM layout read 4.5x slower than the
   * other for a year.
   */
  for (let i = 0; i < REPS; i++) once();
  B.hl.flush(); B.hl.endStep();

  const runs = [];
  for (let s = 0; s < 5; s++) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < REPS; i++) once();
    B.hl.flush();
    runs.push(Number(process.hrtime.bigint() - t0) / 1e9);
    B.hl.endStep(); /* AFTER the clock stops — retiring is not what is measured */
  }
  runs.sort((a, b) => a - b);
  const sec = runs[2] / REPS;
  const gbs = bytes / sec / 1e9;
  return { label, us: sec * 1e6, gbs, bytes };
}

const rows = [];
const push = (r) => { rows.push(r); return r; };

/*
 * THE CONTROL, first, and it is the whole point of the file. An elementwise add
 * over the same tensor moves 3 x ELEMS floats and is known to sit at the card's
 * ceiling (417 GB/s measured at [512,640], against a 448 GB/s spec). Whatever
 * it reads here is what "finished" means for every row below it, on this card,
 * in this process, at this clock.
 */
const control = push(time("add (control)", ELEMS * F32 * 3, () => B.add(x, g)));

/*
 * layerNorm: reads x and the two [WIDTH] vectors, writes y. The vectors are
 * 640 floats against 983,040 — noise — so the floor is 2 x ELEMS.
 */
push(time("layerNorm", ELEMS * F32 * 2, () => B.layerNorm(x, w, bias, 1e-5)));
push(time("rmsNorm", ELEMS * F32 * 2, () => B.rmsNorm(x, w, 1e-5)));

/*
 * layerNormBackward: reads x and g, writes dx — and also produces dw and db,
 * which are [WIDTH] and therefore noise. Three full-size passes is the floor.
 *
 * It returns a TRIPLE rather than a tensor, so the release has to walk it. A
 * missed release here does not fail, it carves: the run would slow down over
 * its own warm-up and report the allocator rather than the kernel.
 */
push(time("layerNormBackward", ELEMS * F32 * 3, () => {
  const r = B.layerNormBackward(x, w, g, 1e-5);
  for (const t of Array.isArray(r) ? r : Object.values(r))
    if (t && t.buffer) B.releaseGpuTensor(t);
  return null; /* already released — `once` must not release it twice */
}));

/*
 * softmax at the ATTENTION shape, which is the other consumer of the reduction
 * tree and a very different one: rows of 64 rather than 640, and 240 x 64 of
 * them. A tree over 64 is six steps where 640 is ten, so if the barriers are
 * what cost, this row should be much closer to the control than layerNorm is.
 * That comparison is the diagnosis, not the number.
 */
{
  const heads = mk([24 * 10 * 64, 64], (i) => Math.sin(i * 0.011));
  const n = 24 * 10 * 64 * 64;
  push(time("softmax [15360,64]", n * F32 * 2, () => B.softmax(heads, 1)));
}

/* A copy through the same elementwise machinery, as a second control: it moves
 * two arrays where add moves three, so it isolates whether the control's rate
 * depends on the read:write ratio the norms have. */
push(time("scale (2-array ctl)", ELEMS * F32 * 2, () => B.scale(x, 1.5)));

console.log(`normalize kernels at [${ROWS},${WIDTH}] — ${(ELEMS * F32 / 1e6).toFixed(2)} MB a pass\n`);
console.log("kernel                    us/call     GB/s   % of control   bytes/call");
for (const r of rows)
  console.log(`${r.label.padEnd(22)} ${r.us.toFixed(1).padStart(8)} ${r.gbs.toFixed(1).padStart(8)}` +
              `   ${(100 * r.gbs / control.gbs).toFixed(0).padStart(11)}%   ${(r.bytes / 1e6).toFixed(2).padStart(8)} MB`);

console.log(`\ncontrol is ${control.gbs.toFixed(0)} GB/s on a 448 GB/s card.`);
console.log("A row far below it is not doing arithmetic — a layer norm is two adds and");
console.log("a multiply per element. It is waiting, and what it waits on is the tree:");
console.log(`a ${WIDTH}-wide reduction is a fold plus ${Math.floor(Math.log2(WIDTH))} halving steps, each ending in a`);
console.log("block-wide BAR.SYNC, and layerNorm runs two of them.");
/* The other half of the diagnosis, and the reason the block width is printed:
 * one element per thread means a 640-wide row is a 640-thread block, and an SM
 * holds 1536 threads, so only TWO blocks are resident to hide those barriers
 * behind each other. */
console.log(`Block width is ${WIDTH} threads (one element each), so ${Math.floor(1536 / WIDTH)} blocks fit an SM.`);
