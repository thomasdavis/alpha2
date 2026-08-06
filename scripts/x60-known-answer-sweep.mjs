#!/usr/bin/env node
/**
 * X60 — known-answer sweep across the kernel surface.
 *
 * X58's gradient-norm bug was found by feeding a reduction an input whose answer
 * was known exactly (all-ones, where sum-of-squares must equal the count) and at
 * sizes that were not round numbers. A bug that halves everything is
 * indistinguishable from a scaling constant; one that predicts 37,232 exactly is
 * not.
 *
 * That method is cheap and it worked, so this applies it across the rest of the
 * kernel surface rather than to reductions alone. Every case below has an exact
 * expected value derived from algebra, not from a second implementation.
 *
 * Sizes deliberately straddle the boundaries where this class of bug hides:
 * workgroup sizes (64/128/256), matmul tile sizes (16/32), and the
 * grid-stride reduction threshold (65536). Round numbers are avoided where
 * possible because they hide off-by-half and off-by-tile errors behind clean
 * ratios.
 *
 * Usage:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *     node scripts/x60-known-answer-sweep.mjs
 */
import { HeliosBackend } from "../packages/helios/dist/backend.js";

const b = new HeliosBackend();
let failures = 0;
let checks = 0;

function near(got, want, tol = 1e-5) {
  const d = Math.abs(got - want);
  return d <= Math.max(tol, Math.abs(want) * tol);
}

function check(name, got, want, tol) {
  checks++;
  if (!near(got, want, tol)) {
    failures++;
    console.log(`  FAIL  ${name}: got ${got} want ${want}`);
    return false;
  }
  return true;
}

function f32(t) { return t.data; }
const arr = (n, fn) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = fn(i); return a; };

// ── matmul: ones(m,k) @ ones(k,n) == k everywhere ──────────────────────────
// Exact in f32 for k well under 2^24. Sensitive to tile-boundary handling.
function matmulOnes(m, k, n) {
  const A = b.fromArray(new Float32Array(m * k).fill(1), [m, k]);
  const B = b.fromArray(new Float32Array(k * n).fill(1), [k, n]);
  const C = f32(b.matmul(A, B));
  let worst = 0, worstAt = -1;
  for (let i = 0; i < m * n; i++) {
    const d = Math.abs(C[i] - k);
    if (d > worst) { worst = d; worstAt = i; }
  }
  check(`matmul ones ${m}x${k}x${n} (worst elem ${worstAt})`, k + worst, k, 1e-6);
}

// ── matmul: A @ I == A ─────────────────────────────────────────────────────
function matmulIdentity(m, k) {
  const A = arr(m * k, (i) => ((i * 37) % 101) / 101 - 0.5);
  const I = new Float32Array(k * k);
  for (let i = 0; i < k; i++) I[i * k + i] = 1;
  const C = f32(b.matmul(b.fromArray(A, [m, k]), b.fromArray(I, [k, k])));
  let worst = 0;
  for (let i = 0; i < m * k; i++) worst = Math.max(worst, Math.abs(C[i] - A[i]));
  check(`matmul A@I ${m}x${k}`, worst, 0, 1e-6);
}

// ── softmax: uniform input -> 1/n; any input -> rows sum to 1 ──────────────
function softmaxRows(rows, cols) {
  const x = arr(rows * cols, (i) => Math.sin(i * 0.7) * 4);
  const y = f32(b.softmax(b.fromArray(x, [rows, cols]), -1));
  let worstSum = 0, minV = Infinity;
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) { const v = y[r * cols + c]; s += v; if (v < minV) minV = v; }
    worstSum = Math.max(worstSum, Math.abs(s - 1));
  }
  check(`softmax rows sum to 1 [${rows},${cols}]`, worstSum, 0, 2e-5);
  checks++;
  if (!(minV >= 0)) { failures++; console.log(`  FAIL  softmax [${rows},${cols}]: negative probability ${minV}`); }
}

function softmaxUniform(cols) {
  const y = f32(b.softmax(b.fromArray(new Float32Array(cols).fill(3.25), [1, cols]), -1));
  let worst = 0;
  for (let i = 0; i < cols; i++) worst = Math.max(worst, Math.abs(y[i] - 1 / cols));
  check(`softmax uniform [1,${cols}] -> 1/n`, worst, 0, 2e-6);
}

// ── transpose round-trip: T(T(x)) == x, bitwise ────────────────────────────
function transposeRoundTrip(r, c) {
  const x = arr(r * c, (i) => i * 0.5 - 3);
  const t = b.transpose(b.fromArray(x, [r, c]));
  const back = f32(b.transpose(t));
  let mismatches = 0;
  for (let i = 0; i < r * c; i++) if (back[i] !== x[i]) mismatches++;
  check(`transpose T(T(x))==x [${r},${c}] mismatches`, mismatches, 0, 0);
}

// ── sum along axis: ones -> the axis length ────────────────────────────────
function sumAxisOnes(rows, cols, axis) {
  const t = b.fromArray(new Float32Array(rows * cols).fill(1), [rows, cols]);
  const s = f32(b.sum(t, axis));
  const want = axis === 0 ? rows : cols;
  let worst = 0;
  for (let i = 0; i < s.length; i++) worst = Math.max(worst, Math.abs(s[i] - want));
  check(`sum axis=${axis} ones [${rows},${cols}] -> ${want}`, worst, 0, 1e-6);
}

// ── rmsnorm: constant input c with unit weight -> sign(c) ──────────────────
// rms(x) = |c|, so x/rms = c/|c| = +/-1 exactly.
function rmsnormConst(rows, cols, c) {
  if (typeof b.rmsnorm !== "function") return;
  const x = b.fromArray(new Float32Array(rows * cols).fill(c), [rows, cols]);
  const w = b.fromArray(new Float32Array(cols).fill(1), [cols]);
  const y = f32(b.rmsnorm(x, w));
  let worst = 0;
  for (let i = 0; i < rows * cols; i++) worst = Math.max(worst, Math.abs(y[i] - Math.sign(c)));
  check(`rmsnorm const ${c} [${rows},${cols}] -> ${Math.sign(c)}`, worst, 0, 1e-4);
}

console.log("matmul (ones): C[i] must equal k");
for (const [m, k, n] of [[31, 33, 29], [32, 32, 32], [33, 65, 47], [64, 128, 64],
                          [129, 257, 65], [128, 512, 128], [17, 1024, 19]]) matmulOnes(m, k, n);

console.log("matmul (identity): A@I must equal A");
for (const [m, k] of [[31, 33], [32, 32], [65, 129], [128, 256]]) matmulIdentity(m, k);

console.log("softmax");
for (const [r, c] of [[7, 33], [4, 128], [3, 129], [2, 257], [5, 1023], [2, 4096]]) softmaxRows(r, c);
for (const c of [33, 128, 129, 257, 1023]) softmaxUniform(c);

console.log("transpose");
for (const [r, c] of [[33, 47], [32, 32], [129, 65], [256, 128]]) transposeRoundTrip(r, c);

console.log("sum along axis");
for (const [r, c] of [[33, 47], [129, 257], [512, 129]]) { sumAxisOnes(r, c, 0); sumAxisOnes(r, c, 1); }

console.log("rmsnorm");
for (const [r, c] of [[7, 33], [4, 128], [3, 257]]) { rmsnormConst(r, c, 2.5); rmsnormConst(r, c, -0.75); }

console.log(failures === 0
  ? `\nresult: ${checks} known-answer checks, all exact`
  : `\nresult: ${failures} FAILURES out of ${checks} checks`);
process.exit(failures === 0 ? 0 : 1);
