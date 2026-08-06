/* out[c] = sum over rows of in[r][c], against a host reference.
 *
 * This kernel replaced `ones[1,R] @ x[R,C]`, which was 21% of the GPU step, and
 * a reduction is exactly the shape of thing that returns a plausible wrong
 * number rather than failing. Two ways it can be wrong that a single shape
 * would not catch, and which decide the case list:
 *
 *   THE ROW WALK is a loop with a stride of 32 and a guard at both ends. Row
 *   counts of 1, 31, 32, 33 and 1,536 cover: no lane owns a row past the first;
 *   the guard fires before the loop for most lanes; an exact multiple; one
 *   past a multiple, so lane 0 alone runs an extra trip; and the model's own.
 *
 *   THE COLUMN TAIL is clamped, not predicated, so the last block computes real
 *   sums for a column that already has an owner. If the clamp leaked into the
 *   store, column cols-1 would come back multiplied. Column counts of 1, 31,
 *   32, 33, 64, 640 and 1,000 put the boundary in every position: inside one
 *   warp, exactly one warp, one past, and the model's.
 *
 * The values are a ramp rather than random so that a column read at the wrong
 * offset gives a WRONG answer rather than a statistically similar one — with
 * random data every column has about the same sum, which is the one input that
 * would hide a column-indexing error completely.
 *
 * Tolerance is relative and generous at 1e-4: summing 1,536 f32 values in a
 * different order than the host does is a real difference and not a defect. The
 * failures this hunts are structural — a dropped lane is off by a thirty-second,
 * a leaked clamp by a whole column.
 */
import { NativeHeliosBackend } from "../helios/dist/index.js";

const B = new NativeHeliosBackend(0);

const CASES = [];
for (const rows of [1, 31, 32, 33, 100, 1536])
  for (const cols of [1, 31, 32, 33, 64, 640, 1000])
    CASES.push([rows, cols]);

let failures = 0;
for (const [rows, cols] of CASES) {
  const x = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      /* Distinct per column AND per row, and bounded so the sum stays exact
       * enough in f32 to compare tightly. */
      x[r * cols + c] = ((c * 7 + 1) % 61) / 61 + ((r * 13 + 1) % 37) / 370;

  const want = new Float64Array(cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) want[c] += x[r * cols + c];

  const t = B.fromArray(Array.from(x), [rows, cols]);
  const got = B.columnSum(t, rows, cols).data;

  /* The PRODUCT form, against its own reference. Its second operand is a
   * DIFFERENT ramp: with two equal inputs the product is a square and a kernel
   * that read the same operand twice would pass. */
  const y = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      y[r * cols + c] = ((c * 3 + 2) % 47) / 47 + ((r * 5 + 1) % 29) / 290;
  const wantP = new Float64Array(cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) wantP[c] += x[r * cols + c] * y[r * cols + c];
  const ty = B.fromArray(Array.from(y), [rows, cols]);
  const gotP = B.columnSum(t, rows, cols, ty).data;
  let worstP = 0, worstPAt = -1;
  for (let c = 0; c < cols; c++) {
    const rel = Math.abs(gotP[c] - wantP[c]) / Math.max(1, Math.abs(wantP[c]));
    if (rel > worstP) { worstP = rel; worstPAt = c; }
  }
  if (!(Number.isFinite(worstP) && worstP < 1e-4)) {
    failures++;
    console.log(`  rows=${String(rows).padStart(4)} cols=${String(cols).padStart(4)} PRODUCT  ` +
                `worst rel ${worstP.toExponential(2)} at column ${worstPAt}  ` +
                `got ${gotP[worstPAt]} want ${wantP[worstPAt]}  WRONG`);
  }

  let worst = 0, worstAt = -1;
  for (let c = 0; c < cols; c++) {
    const rel = Math.abs(got[c] - want[c]) / Math.max(1, Math.abs(want[c]));
    if (rel > worst) { worst = rel; worstAt = c; }
  }
  const ok = Number.isFinite(worst) && worst < 1e-4;
  if (!ok) {
    failures++;
    console.log(`  rows=${String(rows).padStart(4)} cols=${String(cols).padStart(4)}  ` +
                `worst rel ${worst.toExponential(2)} at column ${worstAt}  ` +
                `got ${got[worstAt]} want ${want[worstAt]}  WRONG`);
  }
  B.finishStepOps?.();
}

if (failures) {
  console.log(`\nFAIL — ${failures} of ${CASES.length} shapes disagree`);
  process.exit(1);
}
console.log(`ok — ${CASES.length} shapes x 2 forms agree with the reference`);
