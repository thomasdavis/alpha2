/* Is the step's GPU time the kernels, or something around them?
 *
 * Two numbers refuse to reconcile. The profile attributes ~32 ms of an 85 ms
 * step to real drains, and drains were shown to be cheap in isolation. But
 * pricing the step's own launch mix with per-kernel costs measured at the
 * model's shapes comes to ~7 ms. One of those is wrong, and dividing totals has
 * already misled me once.
 *
 * So fire the step's ACTUAL mix -- same kernels, same shapes, same counts -- and
 * drain once. That is the whole GPU cost of a step with the host removed.
 *
 *   near 7 ms   the kernels are not the step; the remaining 25 ms is host or
 *               drain-pattern, and the work goes into launching less often
 *   near 32 ms  the kernels ARE the step, the per-kernel measurements were
 *               optimistic for a reason worth finding, and the work goes into
 *               the matmul
 *
 * The second experiment asks whether IDENTITY matters: the per-kernel figures
 * fired one shape repeatedly, so its code was already in the GPU's instruction
 * cache and its program already in the cache here. A step rotates through 49
 * distinct programs, each fetched from system memory over PCIe. If rotation is
 * dearer than repetition, that difference is the missing time. */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
const buf = (n) => NativeBuffer.alloc(hl, n);
const A = buf(1 << 16), B = buf(1 << 16), C = buf(1 << 16), D = buf(1 << 16);
const scratch = buf(1024);

/* The counts come from the step profile; the shapes from the model. */
const STEP = [
  [13, () => hl.matmul(C.handle, A.handle, B.handle, 32, 64, 64, 1)],
  [8,  () => hl.matmul(C.handle, A.handle, B.handle, 32, 192, 64, 1)],
  [8,  () => hl.matmul(C.handle, A.handle, B.handle, 32, 256, 64, 1)],
  [6,  () => hl.matmul(C.handle, A.handle, B.handle, 32, 64, 256, 1)],
  [4,  () => hl.matmul(C.handle, A.handle, B.handle, 32, 32, 64, 4)],
  [30, () => hl.transpose(C.handle, A.handle, 32, 64, 4)],
  [16, () => hl.transpose(C.handle, A.handle, 32, 64, 1)],
  [60, () => hl.elementwise(hl.op.add, C.handle, A.handle, B.handle, 2048, 0, 0, 0, 0, 0, 0, 0)],
  [5,  () => hl.normalize(hl.op.layerNorm, C.handle, A.handle, 64, 32, 1e-5)],
  [3,  () => hl.normalize(hl.op.softmax, C.handle, A.handle, 32, 128, 0)],
  [4,  () => hl.maskedFill(C.handle, A.handle, B.handle, 4096, -1e30)],
  [2,  () => hl.reduce(0, C.handle, A.handle, scratch.handle, 2048)],
  [2,  () => hl.embedding(C.handle, A.handle, D.handle, 32, 64)],
  [1,  () => hl.crossEntropy(C.handle, A.handle, D.handle, 32, 64)],
];
const LAUNCHES = STEP.reduce((n, [k]) => n + k, 0);

function warm(ms) {
  const t = process.hrtime.bigint();
  while (Number(process.hrtime.bigint() - t) / 1e6 < ms) {
    for (let i = 0; i < 16; i++)
      hl.elementwise(hl.op.add, C.handle, A.handle, B.handle, 2048, 0, 0, 0, 0, 0, 0, 0);
    hl.flush();
  }
}
function median(f, n = 15) {
  const xs = [];
  for (let i = 0; i < n; i++) {
    const t = process.hrtime.bigint();
    f();
    xs.push(Number(process.hrtime.bigint() - t) / 1e6);
  }
  xs.sort((a, b) => a - b);
  return xs[Math.floor(n / 2)];
}

warm(3000);

const oneDrain = median(() => {
  for (const [k, fire] of STEP) for (let i = 0; i < k; i++) fire();
  hl.flush();
});
/* The same mix drained as often as a real step drains, to see whether the
 * drain PATTERN costs anything on top of the launches. */
const many = median(() => {
  let since = 0;
  for (const [k, fire] of STEP)
    for (let i = 0; i < k; i++) { fire(); if (++since >= 4) { hl.flush(); since = 0; } }
  hl.flush();
});

console.log(`\n${LAUNCHES} launches, the step's mix\n`);
console.log(`  one drain at the end      ${oneDrain.toFixed(2)} ms   (${(oneDrain * 1000 / LAUNCHES).toFixed(1)} us/launch)`);
console.log(`  a drain every 4 launches  ${many.toFixed(2)} ms   (${(many * 1000 / LAUNCHES).toFixed(1)} us/launch)`);
console.log(`  the profile attributes ~32 ms of an 85 ms step to drains`);

/* Repetition against rotation, at matched launch counts. */
const shapes = [[32,64,64],[32,192,64],[32,256,64],[32,64,256],[32,128,64],[32,96,64],[32,160,64],[32,224,64]];
const N = 64;
const repeat = median(() => {
  for (let i = 0; i < N; i++) hl.matmul(C.handle, A.handle, B.handle, 32, 64, 64, 1);
  hl.flush();
});
const rotate = median(() => {
  for (let i = 0; i < N; i++) {
    const s = shapes[i % shapes.length];
    hl.matmul(C.handle, A.handle, B.handle, s[0], s[1], s[2], 1);
  }
  hl.flush();
});
console.log(`\n${N} matmuls, one drain\n`);
console.log(`  the same shape every time ${repeat.toFixed(2)} ms   (${(repeat * 1000 / N).toFixed(1)} us/launch)`);
console.log(`  rotating over 8 shapes    ${rotate.toFixed(2)} ms   (${(rotate * 1000 / N).toFixed(1)} us/launch)`);
