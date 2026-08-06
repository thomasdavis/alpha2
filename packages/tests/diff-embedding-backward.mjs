/* The embedding gradient, against the definition.
 *
 *     dW[v][c] = sum over tokens i where indices[i] == v of g[i][c]
 *
 * The existing implementation obtains that by building a [tokens, vocab]
 * one-hot through seven full-size elementwise passes — 75 MB apiece at this
 * model's batch — and running a 24 GFLOP matmul to pull a mostly-zero table
 * back out. The census named it four of the top allocation sites, 772 MiB held.
 *
 * The scatter form needs an atomic because REPEATED INDICES COLLIDE, so the
 * cases below deliberately include them: a vocabulary of 3 against 64 tokens
 * puts about twenty writers on every row, which is the condition a
 * non-atomic scatter passes at batch 1 and fails at scale. The all-distinct
 * case is there to prove the addressing separately from the collisions.
 *
 * Usage: HELIOS_VIDMEM=1 node diff-embedding-backward.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);

const CASES = [
  { tokens: 8, vocab: 8, dim: 4, ids: (i) => i },              /* all distinct */
  { tokens: 64, vocab: 3, dim: 64, ids: (i) => i % 3 },        /* ~21 collisions a row */
  { tokens: 768, vocab: 12288, dim: 640, ids: (i) => (i * 7) % 12288 },
  { tokens: 1536, vocab: 12288, dim: 640, ids: (i) => (i * 13) % 97 }, /* dense collisions */
  { tokens: 5, vocab: 12288, dim: 640, ids: () => 4095 },      /* every token one row */
  { tokens: 33, vocab: 100, dim: 7, ids: (i) => (i * 3) % 100 },
];

let bad = 0, ran = 0;
for (const { tokens, vocab, dim, ids } of CASES) {
  const idx = Int32Array.from({ length: tokens }, (_, i) => ids(i));
  /* Small magnitudes and few distinct values, so a sum over many collisions is
   * exact in f32 and any mismatch is the kernel rather than the arithmetic. */
  const g = Float64Array.from({ length: tokens * dim }, (_, i) => ((i * 11) % 9) - 4);

  const want = new Float64Array(vocab * dim);
  for (let t = 0; t < tokens; t++)
    for (let c = 0; c < dim; c++) want[idx[t] * dim + c] += g[t * dim + c];

  const ti = B.fromArray(Array.from(idx, Number), [tokens]);
  const tg = B.fromArray(Array.from(g, Number), [tokens, dim]);
  const got = B.embeddingBackward(ti, tg, vocab).data;
  ran++;
  let worst = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (e > worst) { worst = e; at = i; }
  }
  const ok = worst < 1e-4 && got.length === want.length;
  if (!ok) bad++;
  console.log(`  t${tokens} v${vocab} d${dim}`.padEnd(24) +
    `${ok ? "ok" : "FAIL"}  worst ${worst.toExponential(2)}` +
    (ok ? "" : `  at ${at} (row ${Math.floor(at / dim)}): got ${got[at]} want ${want[at]}` +
               `  len ${got.length} want ${want.length}`));
  B.releaseGpuTensor?.(ti); B.releaseGpuTensor?.(tg);
  B.finishStepOps?.();
}
console.log(bad ? `\n${bad}/${ran} WRONG` : `\nembedding backward: ${ran}/${ran} agree with the definition`);
if (bad) process.exit(1);
