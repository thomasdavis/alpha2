/* Tensors LARGER than one physically contiguous allocation.
 *
 * gaia_alloc asks RM for contiguous pages and the kernel's MAX_ORDER stops it
 * at 4 MiB, so a single tensor past that used to fail outright — reported as
 * "allocation of 1146880 floats failed", which reads as a full card and was
 * not: the failing request was 4.59 MB on an 8 GiB card holding 4.75. That
 * capped the model's batch at 24, because [28,64,640] is the first activation
 * over the line.
 *
 * gaia_alloc_large reserves one VA range and maps several chunks at consecutive
 * addresses inside it. The GPU sees one buffer because its MMU says so, and the
 * thing this test has to prove is exactly that: a kernel indexing across a
 * CHUNK BOUNDARY reads and writes the right elements. A buffer whose chunks
 * were mapped at wrong offsets still allocates, still runs, and returns data
 * that is correct for the first 4 MiB and garbage after it — so a test that
 * only checks a few elements, or only the beginning, proves nothing.
 *
 * Sizes bracket the boundary: just under one chunk, just over, several, and
 * enough to need a short final chunk (the case where the size does not divide
 * and the last allocation is smaller than the stride).
 */
import { NativeHeliosBackend } from "../helios/dist/index.js";

const B = new NativeHeliosBackend(0);
const MiB = 1024 * 1024;
const F = 4;

const CASES = [
  ["just under a chunk", Math.floor(3.9 * MiB / F)],
  ["just over a chunk", Math.floor(4.2 * MiB / F)],
  ["the model's batch 28 activation", 28 * 64 * 640],
  ["the model's batch 48 activation", 48 * 64 * 640],
  ["three chunks exactly", 3 * MiB / F * 4],
  ["a short final chunk", Math.floor(9.3 * MiB / F)],
];

let failures = 0;
for (const [name, n] of CASES) {
  let ok = true, detail = "";
  try {
    /* A ramp, so an element read from the wrong chunk is a wrong VALUE and not
     * merely a suspicious one. Modulo keeps it exact in f32. */
    const src = new Float32Array(n);
    for (let i = 0; i < n; i++) src[i] = (i % 9973) + 1;
    const t = B.fromArray(Array.from(src), [n]);

    /* Run a real kernel over the whole thing — allocation alone would not touch
     * the far chunks, and the mapping is what is on trial. */
    const doubled = B.scale(t, 2);
    const got = doubled.data;

    let bad = -1;
    for (let i = 0; i < n; i++) {
      if (Math.abs(got[i] - src[i] * 2) > 1e-3) { bad = i; break; }
    }
    if (bad >= 0) {
      ok = false;
      detail = `element ${bad} (chunk ${Math.floor(bad * F / MiB / 4)}): ` +
               `got ${got[bad]} want ${src[bad] * 2}`;
    }
    B.finishStepOps?.();
  } catch (e) {
    ok = false;
    detail = e.message;
  }
  if (!ok) failures++;
  console.log(`  ${name.padEnd(34)} ${(n * F / MiB).toFixed(1).padStart(5)} MB  ` +
              `${ok ? "ok" : "FAIL  " + detail}`);
}

if (failures) { console.log(`\nFAIL — ${failures} of ${CASES.length}`); process.exit(1); }
console.log(`\nok — ${CASES.length} sizes, all past the 4 MiB contiguous ceiling except the control`);
