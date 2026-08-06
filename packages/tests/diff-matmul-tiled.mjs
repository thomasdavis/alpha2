/* Does the matmul still compute a matrix product now that far more shapes tile?
 *
 * can_tile used to require K % N == 0, which excluded the two matmuls the model
 * spends most of its time in. Relaxing it means the cooperative stage now runs
 * for shapes it never ran for, and the last round is PREDICATED -- threads past
 * the end of A's row must sit out, because A[row][K] is the next row's first
 * element and staging it would produce a finite, plausible, wrong dot product.
 *
 * That is exactly the failure this file exists to catch, so: compare against the
 * definition, element by element, across shapes chosen to hit every branch --
 * K divisible by N (the old path), K < N (one short round), K > N but not
 * divisible (several rounds then a short one), K == N, and K = 1.
 *
 * Batched too, because the stage happens per block and the batch rides the Y
 * grid: a stage that ignored the plane offset would give plane 0 the right
 * answer and every other plane the same one. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";

const N = new NativeHeliosBackend(0);
const TOL = 2e-3; /* K up to 512 accumulated in f32 */

const rand = (n, s) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.7 + s) * 0.5 + 0.1);
function host(t) { N.syncGpu(); const d = t.data; const k = t.shape.reduce((a, b) => a * b, 1); return Array.from({ length: k }, (_, i) => d[i]); }

function reference(a, b, M, K, Ncols, batch) {
  const out = new Float64Array(batch * M * Ncols);
  for (let p = 0; p < batch; p++)
    for (let i = 0; i < M; i++)
      for (let j = 0; j < Ncols; j++) {
        let s = 0;
        for (let k = 0; k < K; k++)
          s += a[p * M * K + i * K + k] * b[(batch > 1 ? p * K * Ncols : 0) + k * Ncols + j];
        out[p * M * Ncols + i * Ncols + j] = s;
      }
  return out;
}

const CASES = [
  { M: 8, K: 8, Ncols: 8, why: "K == N" },
  { M: 64, K: 256, Ncols: 64, why: "K % N == 0, the old path" },
  { M: 64, K: 64, Ncols: 192, why: "K < N, one short round  <- qkv" },
  { M: 64, K: 64, Ncols: 256, why: "K < N, one short round  <- mlp-up" },
  { M: 32, K: 96, Ncols: 64, why: "K > N, not divisible" },
  { M: 32, K: 100, Ncols: 64, why: "K > N, ragged" },
  { M: 16, K: 1, Ncols: 32, why: "K = 1" },
  { M: 16, K: 33, Ncols: 8, why: "several rounds then a short one" },
  { M: 128, K: 64, Ncols: 192, batch: 4, why: "batched, K < N" },
  { M: 32, K: 128, Ncols: 32, batch: 2, why: "batched, K % N == 0" },
];

console.log(`\nmatmul against the definition, tolerance ${TOL}\n`);
console.log("  M     K     N   batch     max|diff|   verdict   why");
let bad = 0;
for (const c of CASES) {
  const batch = c.batch ?? 1;
  const aArr = rand(batch * c.M * c.K, 1);
  const bArr = rand((batch > 1 ? batch : 1) * c.K * c.Ncols, 2);
  const A = N.fromArray(aArr, batch > 1 ? [batch, c.M, c.K] : [c.M, c.K]);
  const B = N.fromArray(bArr, batch > 1 ? [batch, c.K, c.Ncols] : [c.K, c.Ncols]);
  let got;
  try { got = host(N.matmul(A, B)); }
  catch (e) {
    console.log(`${String(c.M).padStart(5)}${String(c.K).padStart(6)}${String(c.Ncols).padStart(6)}${String(batch).padStart(8)}   ERROR ${e.message.slice(0, 26)}`);
    bad++; continue;
  }
  const ref = reference(aArr, bArr, c.M, c.K, c.Ncols, batch);
  let m = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(got[i] - ref[i]);
    m = Number.isFinite(d) ? Math.max(m, d) : Infinity;
  }
  const ok = m <= TOL;
  if (!ok) bad++;
  console.log(`${String(c.M).padStart(5)}${String(c.K).padStart(6)}${String(c.Ncols).padStart(6)}${String(batch).padStart(8)} ${m.toExponential(2).padStart(13)}   ${ok ? "ok      " : "**WRONG**"}   ${c.why}`);
  N.finishStepOps?.();
}
console.log(bad ? `\n${bad} case(s) WRONG` : "\nevery shape agrees with the definition");
process.exit(bad ? 1 : 0);
