/* What each of the model's kernels actually costs on the GPU.
 *
 * Every estimate so far has been an inference. The step spends ~32 ms waiting
 * on drains; the drain-gap experiment proved drains themselves are cheap (8
 * kernels in ~51 us, and splitting the same work across 16 drains instead of 1
 * costs the same), so that 32 ms is GPU time. Dividing it by the step's 184
 * launches gives ~174 us a kernel, against ~6 us for a 1024-element add -- a
 * 30x gap I attributed to the stall-15 encoding, lowered it, and gained
 * nothing.
 *
 * So stop dividing totals and measure the kernels. Each row enqueues one shape
 * many times and drains once, so the figure is issue-to-retire cost per launch
 * with the host taken out of it.
 *
 * The shapes are the ones the 2L/64d/4h/32-token benchmark actually runs. A
 * kernel measured at a round size would answer a question nobody asked. */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
const buf = (n) => NativeBuffer.alloc(hl, n);

/* Generous, so no shape has to think about capacity. */
const A = buf(1 << 16), B = buf(1 << 16), C = buf(1 << 16), D = buf(1 << 16);
const scratch = buf(1024);

const REPS = 32;
function cost(name, fire, reps = REPS) {
  /* Warm: the program cache compiles on first use, and a kernel measured
   * including its own assembly is measuring the assembler. */
  for (let i = 0; i < 4; i++) fire();
  hl.flush();
  const xs = [];
  for (let s = 0; s < 15; s++) {
    const t = process.hrtime.bigint();
    for (let i = 0; i < reps; i++) fire();
    hl.flush();
    xs.push(Number(process.hrtime.bigint() - t) / 1000 / reps);
  }
  xs.sort((a, b) => a - b);
  return { name, us: xs[7] };
}

/* Hold the GPU busy by time first — this card idles at 210 MHz. */
{
  const t = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t) / 1e6 < 3000) {
    for (let i = 0; i < 16; i++)
      hl.elementwise(hl.op.add, C.handle, A.handle, B.handle, 2048, 0, 0, 0, 0, 0, 0, 0);
    hl.flush();
  }
}

const rows = [
  cost("elementwise add 2048", () =>
    hl.elementwise(hl.op.add, C.handle, A.handle, B.handle, 2048, 0, 0, 0, 0, 0, 0, 0)),
  cost("elementwise add 8192", () =>
    hl.elementwise(hl.op.add, C.handle, A.handle, B.handle, 8192, 0, 0, 0, 0, 0, 0, 0)),
  cost("matmul 32x64x64  (proj)", () =>
    hl.matmul(C.handle, A.handle, B.handle, 32, 64, 64, 1)),
  cost("matmul 32x192x64 (qkv)", () =>
    hl.matmul(C.handle, A.handle, B.handle, 32, 192, 64, 1)),
  cost("matmul 32x256x64 (mlp)", () =>
    hl.matmul(C.handle, A.handle, B.handle, 32, 256, 64, 1)),
  cost("matmul 32x64x256 (mlp2)", () =>
    hl.matmul(C.handle, A.handle, B.handle, 32, 64, 256, 1)),
  cost("matmul 32x32x64 batch4 (scores)", () =>
    hl.matmul(C.handle, A.handle, B.handle, 32, 32, 64, 4)),
  cost("transpose 32x64 batch4", () =>
    hl.transpose(C.handle, A.handle, 32, 64, 4)),
  cost("transpose 32x64 batch1", () =>
    hl.transpose(C.handle, A.handle, 32, 64, 1)),
  cost("layerNorm 32 rows x 64", () =>
    hl.normalize(hl.op.layerNorm, C.handle, A.handle, 64, 32, 1e-5)),
  cost("softmax 128 rows x 32", () =>
    hl.normalize(hl.op.softmax, C.handle, A.handle, 32, 128, 0)),
  cost("reduceRows 32x64", () =>
    hl.reduceRows(C.handle, A.handle, 64, 32)),
  cost("reduce whole 2048", () =>
    hl.reduce(0, C.handle, A.handle, scratch.handle, 2048)),
  cost("embedding 32 tok x 64", () =>
    hl.embedding(C.handle, A.handle, D.handle, 32, 64)),
  cost("crossEntropy 32x64", () =>
    hl.crossEntropy(C.handle, A.handle, D.handle, 32, 64)),
  cost("causalMask 32x32", () =>
    hl.causalMask(C.handle, A.handle, 32, 32)),
  cost("maskedFill 4096", () =>
    hl.maskedFill(C.handle, A.handle, B.handle, 4096, -1e30)),
];

rows.sort((a, b) => b.us - a.us);
console.log("\nper-launch GPU cost, median of 15 batches of 32\n");
console.log("kernel                              us/launch");
for (const r of rows) console.log(`${r.name.padEnd(36)} ${r.us.toFixed(1).padStart(8)}`);

/* A step's mix, priced with these numbers. If the total lands near the 32 ms
 * the profile attributes to drains, the kernels ARE the step and the host is
 * not the problem; if it lands far below, something outside these launches is. */
const mix = { "matmul 32x64x64  (proj)": 39, "transpose 32x64 batch4": 46,
              "elementwise add 2048": 38, "layerNorm 32 rows x 64": 5 };
let est = 0;
for (const r of rows) if (mix[r.name]) est += r.us * mix[r.name];
console.log(`\nthe four commonest ops at their step counts: ${(est / 1000).toFixed(1)} ms`);
