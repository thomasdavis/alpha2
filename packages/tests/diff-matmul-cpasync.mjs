/*
 * Is the cast + cp.async f16 GEMM NUMERICALLY the staged f32 GEMM?
 *
 * The cp.async kernel is proven exact at 128x64x64 in the C harness against an
 * integer reference. This asks the other question, the one M5 depends on: at the
 * real m1536 shapes, with REAL (non-integer) operands, does
 *
 *     matmulCpasync(castF16(A), castF16(B))   ==   matmulTransposed(A, B)
 *
 * to within f16 rounding? It must, because BOTH round the operands to f16 before
 * the tensor cores — the staged path in its F2FP pack, this path in the cast —
 * and both accumulate in f32. So the two should agree to the last bit of an
 * f16-operand, f32-accumulate product, and the loss the model sees must not move
 * when the forward is switched to cp.async. If this diff is larger than f16
 * rounding, the wiring is wrong and no throughput number matters.
 *
 * The reference is matmulTransposed (the staged f32 kernel) rather than a CPU
 * GEMM, on purpose: the claim is not "cp.async matches infinite precision", it
 * is "cp.async matches THE KERNEL IT REPLACES". A CPU f64 reference would flag
 * the shared f16 rounding as an error in both.
 *
 * Run with HELIOS_VIDMEM=1.
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);
const hl = B.hl;

const rng = (seed) => { let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; };

/* The model's NT forward shapes. K is 640 (nEmbd); N is the projection width. */
const SHAPES = [
  ["qkv    ", 128, 1920, 640],
  ["mlp fc ", 128, 2560, 640],
  ["lm head", 128, 12288, 640],
];

console.log("cast+cp.async f16 GEMM vs staged f32 GEMM — numerical agreement\n");
console.log("shape       maxAbs      maxRel     verdict");

let ok = true;
for (const [name, M, N, K] of SHAPES) {
  const r = rng(12345 + N);
  const aData = new Float32Array(M * K);
  const bData = new Float32Array(N * K);
  /* Small centred values so the f16-operand product is well inside range and
   * the comparison is about rounding, not overflow. */
  for (let i = 0; i < aData.length; i++) aData[i] = (r() - 0.5) * 0.5;
  for (let i = 0; i < bData.length; i++) bData[i] = (r() - 0.5) * 0.5;

  const a = B.fromArray(aData, [M, K]);
  const b = B.fromArray(bData, [N, K]);

  /* Reference: the staged f32 kernel this path replaces. */
  const ref = B.matmulTransposed(a, b);
  const refData = ref.data.slice();

  /* Under test: cast both operands to f16 buffers, then cp.async. */
  const aF16 = B.make([(M * K) >> 1], "f32");
  const bF16 = B.make([(N * K) >> 1], "f32");
  hl.cast(1, aF16.buffer.handle, B.device(a).buffer.handle, M * K);
  hl.cast(1, bF16.buffer.handle, B.device(b).buffer.handle, N * K);
  const out = B.make([M, N], "f32");
  const rc = hl.matmulCpasync(out.buffer.handle, aF16.buffer.handle,
                              bF16.buffer.handle, M, N, K, 1);
  if (!rc) { console.log(`${name}  matmulCpasync REFUSED the shape`); ok = false; continue; }
  const cpData = out.data.slice();

  let maxAbs = 0, maxRel = 0;
  for (let i = 0; i < M * N; i++) {
    const d = Math.abs(cpData[i] - refData[i]);
    maxAbs = Math.max(maxAbs, d);
    const denom = Math.abs(refData[i]);
    if (denom > 1e-3) maxRel = Math.max(maxRel, d / denom);
  }
  /*
   * THE TOLERANCE is an f16-operand, f32-accumulate product's, not zero: the two
   * kernels round each operand to f16 (11-bit mantissa, ~5e-4 relative) and then
   * sum K=640 of them, so per-element error accumulates. A few e-3 relative is
   * the honest bound; the two do NOT have to be bit-identical because the pack
   * order (F2FP vs the cast kernel) can differ, only equal to f16-product
   * precision. Larger than this and the staging is reading the wrong bytes.
   */
  const pass = maxRel < 5e-3;
  ok = ok && pass;
  console.log(`${name}  ${maxAbs.toExponential(2)}  ${maxRel.toExponential(2)}   ${pass ? "ok" : "FAIL"}`);

  aF16.buffer.release(hl); bF16.buffer.release(hl); out.buffer.release(hl);
}

console.log(ok
  ? "\nok — cast+cp.async agrees with the staged kernel to f16 precision; the loss will not move."
  : "\nFAIL — the cp.async path does not match the kernel it replaces.");
process.exit(ok ? 0 : 1);
