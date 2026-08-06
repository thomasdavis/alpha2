/* maskedFill with the mask TILED across the value, against the materialised form.
 *
 * The kernel now wraps the mask index at the mask's own size instead of
 * requiring an `expand` to the value's size first. That removes a 3.93 MB
 * allocation and a full write per call, thirty-six times a step — and it is
 * exactly the kind of change that returns a plausible wrong answer: a wrap at
 * the wrong modulus still runs, still fills roughly the right FRACTION of
 * elements, and only differs in WHICH ones.
 *
 * So the comparison is against a mask the caller materialised by hand, element
 * by element, over shapes where a wrong wrap cannot coincide with a right one:
 * several distinct repeats, and a mask whose pattern is asymmetric so that
 * transposing or rotating it is detectable.
 */
import { NativeHeliosBackend } from "../helios/dist/index.js";

const B = new NativeHeliosBackend(0);

/* [maskRows, maskCols, repeats] — the value is [repeats*maskRows, maskCols]. */
const CASES = [
  [4, 4, 1], [4, 4, 3], [8, 8, 5],
  [64, 64, 2],                    /* the model's mask */
  [64, 64, 240],                  /* the model's mask at the model's repeat */
  [2, 2, 17], [16, 16, 9],
];

let failures = 0;
for (const [mr, mc, rep] of CASES) {
  const maskN = mr * mc;
  /* Asymmetric on purpose: a causal mask is symmetric under some wrong
   * indexings and would hide them. */
  const mask = new Float32Array(maskN);
  for (let i = 0; i < maskN; i++) mask[i] = ((i * 5 + 1) % 7) < 3 ? 1 : 0;

  const n = maskN * rep;
  const vals = new Float32Array(n);
  for (let i = 0; i < n; i++) vals[i] = (i % 251) + 1;

  const want = new Float32Array(n);
  for (let i = 0; i < n; i++) want[i] = mask[i % maskN] ? -1e9 : vals[i];

  const dv = B.fromArray(Array.from(vals), [rep * mr, mc]);
  const dm = B.fromArray(Array.from(mask), [mr, mc]);
  const got = B.maskedFill(dv, dm, -1e9).data;

  let bad = -1;
  for (let i = 0; i < n; i++) if (got[i] !== want[i]) { bad = i; break; }
  const ok = bad < 0;
  if (!ok) {
    failures++;
    console.log(`  mask ${mr}x${mc} x${rep}  FAIL at ${bad}: got ${got[bad]} want ${want[bad]}`);
  } else {
    console.log(`  mask ${String(mr).padStart(2)}x${String(mc).padStart(2)} x${String(rep).padStart(3)}` +
                `  ${String(n).padStart(8)} elements  ok`);
  }
  B.finishStepOps?.();
}

if (failures) { console.log(`\nFAIL — ${failures} of ${CASES.length}`); process.exit(1); }
console.log(`\nok — ${CASES.length} tilings match an element-by-element reference`);
