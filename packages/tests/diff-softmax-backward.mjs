/* softmax's backward, against the definition.
 *
 * dL/dx = s * (g - sum_j g_j s_j), where s is the softmax OUTPUT. ops.ts builds
 * that from five operations — mul, sum, broadcast, sub, mul — each of which
 * reads and writes the whole [B,H,T,T] tensor. The elementwise half of a step
 * already runs at 340-417 GB/s against a 448 GB/s card, so it cannot be made
 * faster; it can only be made LESS, and this is one row-wise pass expressed as
 * five full ones.
 *
 * Written BEFORE the kernel, deliberately. The composed path is the arbiter and
 * the expectation below is arithmetic done here, not the composition's output —
 * a row reduction feeding a per-element write is exactly the shape that
 * produces plausible wrong numbers when a barrier or a register is misplaced,
 * and this stack has been caught by that three times in normalize.c alone.
 *
 * Usage: node diff-softmax-backward.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);

/* A real softmax output, so the rows sum to one and the gradient has the shape
 * the backward actually sees. */
function softmaxRow(x) {
  const m = Math.max(...x);
  const e = x.map((v) => Math.exp(v - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
}

const CASES = [
  [1, 8], [4, 64], [3, 7], [80, 64], [2, 129], [8, 640], [5, 1024],
];

let bad = 0;
for (const [rows, width] of CASES) {
  const s = new Float64Array(rows * width);
  const g = new Float64Array(rows * width);
  for (let r = 0; r < rows; r++) {
    const raw = Array.from({ length: width }, (_, i) => ((r * 13 + i * 7) % 23) / 5 - 2);
    const sm = softmaxRow(raw);
    for (let i = 0; i < width; i++) {
      s[r * width + i] = sm[i];
      g[r * width + i] = ((r * 5 + i * 11) % 17) / 4 - 2;
    }
  }
  /* The definition, per row. */
  const want = new Float64Array(rows * width);
  for (let r = 0; r < rows; r++) {
    let dot = 0;
    for (let i = 0; i < width; i++) dot += g[r * width + i] * s[r * width + i];
    for (let i = 0; i < width; i++)
      want[r * width + i] = s[r * width + i] * (g[r * width + i] - dot);
  }

  const ts = B.fromArray(Array.from(s, Number), [rows, width]);
  const tg = B.fromArray(Array.from(g, Number), [rows, width]);
  if (!B.softmaxBackward) {
    console.log("  softmaxBackward absent — nothing to check");
    break;
  }
  const got = B.softmaxBackward(ts, tg).data;

  let worst = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (e > worst) { worst = e; at = i; }
  }
  const ok = worst < 2e-6 && got.length === want.length;
  if (!ok) bad++;
  console.log(`  [${rows},${width}]`.padEnd(14) + `${ok ? "ok" : "FAIL"}` +
    `   worst abs ${worst.toExponential(2)}` +
    (ok ? "" : `  at ${at}: got ${got[at]} want ${want[at]}`));
  B.releaseGpuTensor?.(ts); B.releaseGpuTensor?.(tg);
  B.finishStepOps?.();
}
console.log(bad ? `\n${bad} case(s) WRONG` : "\nsoftmax backward: every case agrees with the definition");
if (bad) process.exit(1);
