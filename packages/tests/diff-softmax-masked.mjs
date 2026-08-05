/* The fused scale + causal mask + softmax, against the composed chain.
 *
 * These have to agree ELEMENT BY ELEMENT and not merely "sum to one", because a
 * softmax renormalises: get the scale wrong, or the mask wrong, or apply them
 * in the wrong ORDER, and the result is still a valid distribution over every
 * row. That is the failure this file exists to catch, and it is the same one
 * that let a wrong exponent base survive in hl_normalize — the property held
 * throughout.
 *
 * ORDER is the subtle one. The composed path scales and THEN masks, so the fill
 * lands in scaled units; a kernel that masked first would pick a different
 * maximum and produce a different — still normalised — answer. So the scale is
 * deliberately far from 1 here: at scale 1 the two orders coincide and the test
 * would pass either way.
 */
import { NativeHeliosBackend } from "../helios/dist/index.js";

const B = new NativeHeliosBackend(0);

let failures = 0;
for (const [heads, T] of [[1, 4], [2, 8], [3, 16], [10, 64], [2, 128]]) {
  for (const scaleVal of [1 / Math.sqrt(64), 1, 7.5]) {
    const rows = heads * T;
    const n = rows * T;

    /* A causal mask, in the model's own convention: non-zero means FORBIDDEN. */
    const mask = new Float32Array(T * T);
    for (let r = 0; r < T; r++)
      for (let c = 0; c < T; c++) mask[r * T + c] = c > r ? 1 : 0;

    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = (((i * 31) % 97) / 97 - 0.5) * 6;

    const dx = B.fromArray(Array.from(x), [heads, T, T]);
    const dm = B.fromArray(Array.from(mask), [T, T]);

    /* cap 0 is scale+mask+softmax; cap 30 is what the model runs, because
     * softCap defaults to 30 whenever RoPE is off. The capped form folds the
     * scale into softCap's exponent constant rather than applying it as a
     * multiply, so it is a DIFFERENT arithmetic path and needs its own case. */
    for (const cap of [0, 30]) {
    const got = B.softmaxMasked(dx, dm, scaleVal, cap);
    if (!got) { console.log(`  heads=${heads} T=${T}  kernel declined the shape`); continue; }

    /* The composed chain, through the same backend, in the model's order. */
    const scaled = B.scale(dx, scaleVal);
    const capped = cap > 0 ? B.softCap(scaled, cap) : scaled;
    const masked = B.maskedFill(capped, dm, -1e9);
    const want = B.softmax(masked, -1).data;
    const g = got.data;

    let worst = 0, at = -1;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(g[i] - want[i]);
      if (d > worst) { worst = d; at = i; }
    }
    /* Also check the masked positions are EXACTLY zero, which a tolerance on
     * the whole row would not: a leaked 1e-9 there is a forbidden token with a
     * real probability. */
    let leaked = -1;
    for (let i = 0; i < n && leaked < 0; i++)
      if (mask[i % (T * T)] && g[i] !== 0) leaked = i;

    /* The capped path reassociates — the scale rides in a constant instead of
     * a multiply — so it is equal in real arithmetic and not bit-for-bit. A
     * relative 1e-5 is far tighter than any structural error could hide in. */
    const ok = worst < (cap > 0 ? 1e-5 : 1e-6) && leaked < 0;
    if (!ok) {
      failures++;
      console.log(`  heads=${heads} T=${String(T).padStart(3)} scale=${scaleVal.toFixed(3)} cap=${cap}  ` +
                  (leaked >= 0 ? `masked position ${leaked} is ${g[leaked]}, not 0`
                               : `worst ${worst.toExponential(2)} at ${at}`) + "  WRONG");
    } else {
      console.log(`  heads=${String(heads).padStart(2)} T=${String(T).padStart(3)} ` +
                  `scale=${scaleVal.toFixed(3)} cap=${String(cap).padStart(2)}  ` +
                  `worst ${worst.toExponential(2)}  ok`);
    }
    }
    B.finishStepOps?.();
  }
}

if (failures) { console.log(`\nFAIL — ${failures} case(s)`); process.exit(1); }
console.log("\nok — the fused chain matches the composed one element by element");
