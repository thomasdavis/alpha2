/* What is video memory WORTH, on a kernel that never touches the host?
 *
 * The step is 78% GPU and that GPU half sustains 19.7 GB/s on an elementwise
 * add -- PCIe, not memory, because every tensor lives in system memory. Moving
 * them to VRAM was tried once and cost 60x, but for a reason that says nothing
 * about the kernels: host reads then cross the BAR1 aperture uncached, and
 * slice, permute, broadcast and the autograd fallbacks all read on the host.
 *
 * So the decision to spend days removing those host reads rests on a number
 * nobody has measured: what the SAME kernel does from VRAM with no host read
 * anywhere near it. This measures exactly that and nothing else -- allocate,
 * fire N times, drain once, divide.
 *
 * Run it twice, once with HELIOS_VIDMEM=1. The ratio is the prize.
 */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
const where = process.env.HELIOS_VIDMEM ? "VIDMEM" : "sysmem";

/* 1 Mi floats = 4 MB a tensor, three of them: two read, one written. Big
 * enough that the launch is not the measurement, small enough that three fit
 * the 256 MiB BAR1 aperture even when mapped. */
const N = Number(process.env.N ?? (1 << 20));
const BYTES = N * 4 * 3;
const a = NativeBuffer.alloc(hl, N), b = NativeBuffer.alloc(hl, N), c = NativeBuffer.alloc(hl, N);

const REPS = 64;
function bandwidth(label, fire) {
  for (let i = 0; i < 8; i++) fire();   /* compile the program, ramp the clock */
  hl.flush();
  /* Warm by TIME: this card idles at 210 MHz against 2100 and cannot be clock
   * locked inside the container, so a cold measurement is a clock measurement. */
  const warm = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - warm) / 1e6 < 2000) { fire(); hl.flush(); }

  const best = [];
  for (let s = 0; s < 5; s++) {
    const t = process.hrtime.bigint();
    for (let i = 0; i < REPS; i++) fire();
    hl.flush();
    best.push(Number(process.hrtime.bigint() - t) / 1e9);
  }
  best.sort((x, y) => x - y);
  const sec = best[Math.floor(best.length / 2)];
  const gbs = (BYTES * REPS) / sec / 1e9;
  console.log(`${label.padEnd(22)} ${gbs.toFixed(1).padStart(7)} GB/s   ` +
              `${(sec / REPS * 1e6).toFixed(1).padStart(8)} us/launch   (${where})`);
  return gbs;
}

console.log(`elementwise over ${(N * 4 / 1048576).toFixed(0)} MB tensors, ${REPS} launches a drain\n`);
bandwidth("add (2 read 1 write)", () =>
  hl.elementwise(hl.op.add, c.handle, a.handle, b.handle, N, 0, 0, 0, 0, 0, 0, 0));
bandwidth("copy (1 read 1 write)", () =>
  hl.elementwise(hl.op.copy, c.handle, a.handle, a.handle, N, 0, 0, 0, 0, 0, 0, 0));

/*
 * AND THE SAME KERNEL AT EVERY SIZE, because "19.7 GB/s" is only a bandwidth if
 * the cost actually falls with the work.
 *
 * The step fires 261 launches over tensors of about 1 MB and spends 178.6 ms on
 * the GPU -- 684 us a launch, which is what this one measured at 12 MB. If a
 * launch a thousand times smaller costs the same, the number is a FIXED COST
 * wearing a bandwidth's clothes, and every conclusion drawn from it (that the
 * GPU half is PCIe-bound, that video memory is the prize) is drawn from the
 * wrong model.
 */
console.log("\nsize sweep — same kernel, same drain discipline\n");
console.log("elements      bytes moved      us/launch      implied GB/s");
for (const n of [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20]) {
  const fire = () => hl.elementwise(hl.op.add, c.handle, a.handle, b.handle, n, 0, 0, 0, 0, 0, 0, 0);
  for (let i = 0; i < 8; i++) fire();
  hl.flush();
  const samples = [];
  for (let s = 0; s < 5; s++) {
    const t = process.hrtime.bigint();
    for (let i = 0; i < REPS; i++) fire();
    hl.flush();
    samples.push(Number(process.hrtime.bigint() - t) / 1e9 / REPS);
  }
  samples.sort((x, y) => x - y);
  const sec = samples[Math.floor(samples.length / 2)];
  console.log(`${String(n).padStart(8)}   ${(n * 12 / 1024).toFixed(0).padStart(10)} KB   ` +
              `${(sec * 1e6).toFixed(1).padStart(10)}   ${(n * 12 / sec / 1e9).toFixed(2).padStart(14)}`);
}
