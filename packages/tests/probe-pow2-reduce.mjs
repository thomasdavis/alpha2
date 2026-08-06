/* How far does the power-of-two assumption reach?
 *
 * pr_emit_tree walks strides elements/2, /4, ... down to 1. That covers a
 * power-of-two span and nothing else: over 15 elements the strides are 7, 3, 1
 * and element 14 is never combined into anything. The sum comes back finite,
 * plausible, and short by one element.
 *
 * The model never met this because every extent in it is a power of two -- C is
 * 64 and B*T is 4096 -- so the question is not whether it is a bug but how much
 * of the surface it silently covers. Both axes, because they take different
 * routes: the last axis reduces directly, an earlier axis transposes first, and
 * above 1024 an earlier axis becomes a matmul with ones instead. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const size = (s) => s.reduce((a, b) => a * b, 1);
const fill = (n) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7 + 1) * 2 + 0.35);
function host(B, t) { B.syncGpu?.(); const d = t.data; return Array.from({ length: size(t.shape) }, (_, i) => d[i]); }
const pow2 = (n) => (n & (n - 1)) === 0;

console.log("\nsum over the LAST axis — rows x width, reduce the width\n");
console.log("  width   pow2?      max|diff|   verdict");
for (const width of [8, 15, 16, 31, 32, 48, 64, 100, 128]) {
  const rows = 4, a = fill(rows * width);
  const ref = [];
  for (let r = 0; r < rows; r++) { let s = 0; for (let j = 0; j < width; j++) s += a[r * width + j]; ref.push(s); }
  let got;
  try { got = host(N, N.sum(N.fromArray(a, [rows, width]), -1, false)); }
  catch (e) { console.log(`  ${String(width).padStart(5)}   ${String(pow2(width)).padEnd(6)}   ERROR ${e.message.slice(0, 34)}`); continue; }
  let m = 0; for (let i = 0; i < rows; i++) m = Math.max(m, Math.abs(got[i] - ref[i]));
  console.log(`  ${String(width).padStart(5)}   ${String(pow2(width)).padEnd(6)} ${m.toExponential(2).padStart(12)}   ${m <= 1e-4 ? "ok" : "**WRONG**"}`);
}

console.log("\nsum over AXIS 0 — rows x width, reduce the rows\n");
console.log("  rows   pow2?   route          max|diff|   verdict");
for (const rows of [8, 15, 16, 31, 32, 100, 1024, 1025, 1500, 4096]) {
  const width = 8, a = fill(rows * width);
  const ref = new Float64Array(width);
  for (let r = 0; r < rows; r++) for (let j = 0; j < width; j++) ref[j] += a[r * width + j];
  let got;
  try { got = host(N, N.sum(N.fromArray(a, [rows, width]), 0, false)); }
  catch (e) { console.log(`  ${String(rows).padStart(5)}   ${String(pow2(rows)).padEnd(6)}  ERROR ${e.message.slice(0, 34)}`); continue; }
  let m = 0; for (let i = 0; i < width; i++) m = Math.max(m, Math.abs(got[i] - ref[i]));
  const route = rows > 1024 ? "matmul-ones" : "transpose+tree";
  console.log(`  ${String(rows).padStart(5)}   ${String(pow2(rows)).padEnd(6)}  ${route.padEnd(14)} ${m.toExponential(2).padStart(9)}   ${m <= 1e-6 * rows ? "ok" : "**WRONG**"}`);
}
