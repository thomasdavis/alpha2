/* Does a drain cost more when the GPU has been left idle before it?
 *
 * The step profile says a real drain costs 976 us. The microbenchmark, doing
 * nothing but enqueue-and-flush in a tight loop, says 28.9 us. Both measure the
 * same function on the same card, so one of the two conditions is doing the
 * work -- and the obvious difference is that a real step spends ~2.5 ms in
 * JavaScript between drains, during which nothing is submitted.
 *
 * This card idles at 210 MHz and runs at 2100, and nvidia-smi cannot lock
 * clocks inside the container. So the hypothesis is that the gap, not the
 * queue, is what makes a drain dear: the GPU winds down between them and every
 * drain pays to wake it.
 *
 * It matters because it decides the next fortnight of work. If drains are dear
 * because of the gap, then removing drains pays SUPER-linearly -- each one
 * removed also removes an idle window -- and writing fused backward kernels to
 * eliminate CPU fallbacks is the right thing to do. If a drain is dear because
 * of the work it waits for, removing drains only moves that work and the effort
 * should go into the kernels themselves instead.
 *
 * The gap is a BUSY loop, not a sleep, because that is what a step does: the
 * host is computing, not blocked. A sleep would also let the CPU drop its own
 * clocks and confound the two effects. */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
const a = NativeBuffer.alloc(hl, 1024), b = NativeBuffer.alloc(hl, 1024), o = NativeBuffer.alloc(hl, 1024);
const fire = () => hl.elementwise(hl.op.add, o.handle, a.handle, b.handle, 1024, 0, 0, 0, 0, 0, 0, 0);

/* Burn host time without touching the GPU or yielding. */
function spin(ms) {
  const until = process.hrtime.bigint() + BigInt(Math.round(ms * 1e6));
  let x = 0;
  while (process.hrtime.bigint() < until) x += Math.sqrt(x + 1);
  return x;
}

/* Warm the GPU first, by time — otherwise the first condition measured is just
 * the clock ramp, which is the very effect under test. */
{
  const t = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t) / 1e6 < 3000) { fire(); hl.flush(); }
}

function measure(gapMs, kernels, samples = 60) {
  const xs = [];
  for (let i = 0; i < samples; i++) {
    if (gapMs > 0) spin(gapMs);
    for (let k = 0; k < kernels; k++) fire();
    const t = process.hrtime.bigint();
    hl.flush();
    xs.push(Number(process.hrtime.bigint() - t) / 1000); /* us */
  }
  xs.sort((x, y) => x - y);
  return xs[Math.floor(xs.length / 2)];
}

console.log("drain cost (us, median of 60) — rows are the host gap BEFORE the drain\n");
console.log("gap(ms)   1 kernel   4 kernels   8 kernels");
for (const gap of [0, 0.25, 0.5, 1, 2, 2.5, 5]) {
  const k1 = measure(gap, 1), k4 = measure(gap, 4), k8 = measure(gap, 8);
  console.log(`${gap.toFixed(2).padStart(6)} ${k1.toFixed(1).padStart(10)} ${k4.toFixed(1).padStart(11)} ${k8.toFixed(1).padStart(11)}`);
}

/* And the control the whole argument rests on: if the gap is what costs, then
 * the SAME total work with the same total gap, drained half as often, should
 * cost markedly less than half again. */
console.log("\ncontrol — 32 kernels and 8 ms of host work, split across N drains");
console.log("drains    total us");
for (const d of [16, 8, 4, 2, 1]) {
  const perDrain = 32 / d, gapEach = 8 / d;
  const xs = [];
  for (let i = 0; i < 30; i++) {
    const t = process.hrtime.bigint();
    for (let j = 0; j < d; j++) {
      spin(gapEach);
      for (let k = 0; k < perDrain; k++) fire();
      hl.flush();
    }
    xs.push(Number(process.hrtime.bigint() - t) / 1000);
  }
  xs.sort((x, y) => x - y);
  console.log(`${String(d).padStart(6)} ${xs[15].toFixed(0).padStart(11)}`);
}
