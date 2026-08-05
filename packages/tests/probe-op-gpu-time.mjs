/* Where does the GPU time in a step actually go?
 *
 * The step profiler cannot answer this. Its method rows are HOST time -- an
 * enqueue returns immediately -- and the GPU time all lands in the drains,
 * which is why a drain looked like the fallback's cost and was not. So measure
 * each operation the way the step uses it: enqueue, flush, wall-clock, at the
 * shapes the model actually produces at batch 128.
 *
 * The shapes come from a 2-layer 64-dim 4-head model, seq 32, batch 128:
 *   activations [4096, 64], qkv [64,192], mlp up [64,256], scores [512,32,32]
 *
 * TRAFFIC is reported next to the time because the matmul kernel is documented
 * as naive on purpose -- "A's row is re-read by all N threads in the block and
 * B's column by all M blocks" -- so its real global traffic is M*N*K*2*4 bytes,
 * not the size of its operands. If that is where the step goes, tiling is the
 * next kernel to write and the number here says how much it is worth. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const size = (s) => s.reduce((a, b) => a * b, 1);
const rand = (n) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 0.5);
const mk = (shape) => N.fromArray(rand(size(shape)), shape);

/* Time one operation with the queue empty before and drained after, so the
 * figure is that operation's GPU time and not somebody else's backlog. */
function timeOp(fn, iters = 20) {
  fn(); N.syncGpu();                       /* warm the program cache */
  for (let w = 0; w < 8; w++) fn();
  N.syncGpu();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  N.syncGpu();
  return Number(process.hrtime.bigint() - t0) / 1e6 / iters;
}

const B = 128, T = 32, C = 64, H = 4, HD = C / H;
const ROWS = B * T;

const x = mk([ROWS, C]);
const wQkv = mk([C, 3 * C]);
const wUp = mk([C, 4 * C]);
const wDn = mk([4 * C, C]);
const up = mk([ROWS, 4 * C]);
const w1 = mk([C]);
const scores = mk([B * H, T, T]);

const CASES = [
  { name: "matmul qkv [4096,64]x[64,192]", per: 1, traffic: ROWS * 192 * C * 8,
    run: () => N.matmul(x, wQkv) },
  { name: "matmul mlp-up [4096,64]x[64,256]", per: 1, traffic: ROWS * 256 * C * 8,
    run: () => N.matmul(x, wUp) },
  { name: "matmul mlp-dn [4096,256]x[256,64]", per: 1, traffic: ROWS * C * 256 * 8,
    run: () => N.matmul(up, wDn) },
  { name: "layerNorm [4096,64]", per: 1, traffic: ROWS * C * 8,
    run: () => N.layerNorm(x, w1, w1, 1e-5) },
  { name: "softmax [512,32,32]", per: 1, traffic: size([B * H, T, T]) * 8,
    run: () => N.softmax(scores, -1) },
  { name: "gelu [4096,256]", per: 1, traffic: ROWS * 256 * 8,
    run: () => N.gelu(up) },
  { name: "add [4096,64]", per: 1, traffic: ROWS * C * 12,
    run: () => N.add(x, x) },
  { name: "transpose [4096,64]", per: 1, traffic: ROWS * C * 8,
    run: () => N.transpose(x) },
];

console.log("\nGPU time per operation at batch-128 shapes (enqueue, flush, wall clock)\n");
console.log("operation                              ms      GB/s   traffic MB");
let total = 0;
for (const c of CASES) {
  let ms;
  try { ms = timeOp(c.run); }
  catch (e) { console.log(`${c.name.padEnd(36)}  ERROR ${e.message.slice(0, 30)}`); continue; }
  total += ms;
  const gbs = c.traffic / (ms / 1000) / 1e9;
  console.log(`${c.name.padEnd(36)} ${ms.toFixed(2).padStart(6)} ${gbs.toFixed(1).padStart(9)} ${(c.traffic / 1e6).toFixed(1).padStart(12)}`);
}
console.log(`\nsystem memory is 19.7 GB/s and video memory 111.8 -- a kernel far below 19.7`);
console.log(`is not bandwidth-starved, it is doing redundant work or is launch-bound.`);
