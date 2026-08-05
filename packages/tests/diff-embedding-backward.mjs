/* The native embedding gradient against a reference scatter-add.
 *
 * Backend.embeddingBackward is optional and the native backend did not have
 * it, so autograd used a host loop. The replacement computes the same thing as
 * `(onehot^T @ g)`, with the one-hot built out of arithmetic rather than a
 * scatter — so the thing most worth testing is not the matmul but whether the
 * one-hot is actually one-hot: whether `1 - clamp((idx - v)^2, 0, 1)` really
 * is 1 exactly on the diagonal and 0 everywhere else, at a vocabulary large
 * enough that the square overflows f32's 24-bit mantissa.
 *
 * REPEATED IDS ARE THE CASE THAT MATTERS. A gradient for a table is an
 * ACCUMULATION: a token appearing three times must add three rows into the
 * same slot. A one-hot matmul gets that for free where a scatter needs
 * atomics, but "for free" is a claim, so it is tested directly — half these
 * cases repeat ids deliberately, and one uses a single id for every token.
 *
 * Tolerance is relative and small rather than zero: the matmul sums in a
 * different order from the reference loop, so f32 association differs.
 *
 * Usage: node diff-embedding-backward.mjs
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const B = new NativeHeliosBackend(0);

const CASES = [
  { name: "distinct ids",     tokens: 6,   vocab: 12,    dim: 8,   ids: (i) => i },
  { name: "repeated ids",     tokens: 12,  vocab: 5,     dim: 4,   ids: (i) => i % 5 },
  { name: "all one id",       tokens: 10,  vocab: 7,     dim: 3,   ids: () => 3 },
  { name: "unused rows",      tokens: 4,   vocab: 32,    dim: 6,   ids: (i) => i * 2 },
  { name: "model vocabulary", tokens: 64,  vocab: 12288, dim: 64,  ids: (i) => (i * 191) % 12288 },
  { name: "big, repeating",   tokens: 256, vocab: 12288, dim: 32,  ids: (i) => (i * 7) % 40 },
];

let failed = 0;
for (const c of CASES) {
  const ids = Array.from({ length: c.tokens }, (_, i) => c.ids(i));
  const g = Array.from({ length: c.tokens * c.dim }, (_, i) => Math.sin(i * 0.37) * 2);

  const idsT = B.fromArray(ids, [c.tokens]);
  const gT = B.fromArray(g, [c.tokens, c.dim]);
  const outT = B.embeddingBackward(idsT, gT, c.vocab);
  const got = Array.from(outT.data);

  const want = new Float64Array(c.vocab * c.dim);
  for (let i = 0; i < c.tokens; i++)
    for (let d = 0; d < c.dim; d++) want[ids[i] * c.dim + d] += g[i * c.dim + d];

  let worst = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]) / Math.max(1, Math.abs(want[i]));
    if (e > worst) { worst = e; at = i; }
  }
  const ok = worst < 1e-5;
  if (!ok) failed++;
  console.log(`  ${c.name.padEnd(18)} ${String(c.vocab).padStart(6)} vocab  ` +
              `${ok ? "ok  " : "BAD "} worst rel ${worst.toExponential(1)}` +
              (ok ? "" : ` at ${at}: got ${got[at]}, want ${want[at]}`));

  B.releaseGpuTensor?.(idsT); B.releaseGpuTensor?.(gT);
  B.releaseGpuTensor?.(outT); B.finishStepOps?.();
}

console.log(failed === 0
  ? `\nembeddingBackward: ${CASES.length}/${CASES.length} cases match the scatter-add`
  : `\nembeddingBackward: ${failed}/${CASES.length} WRONG`);
process.exit(failed === 0 ? 0 : 1);
