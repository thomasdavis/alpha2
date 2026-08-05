/* Two backends, one GPU, at the same time — and an honest way to say so.
 *
 * The isolated numbers are not the claim. "Native at 30,000 and Vulkan at
 * 10,000 simultaneously" is a claim about one device in one wall-clock window,
 * and the only way to make it is to run both, stamp every step, and then throw
 * away every step that did not happen while the other side was also running.
 *
 * The hazard this exists to measure: with the native channel open the same
 * Vulkan binary once measured 142 tok/s where it measured 628 alone. That is a
 * 4.4x collapse in code that had not changed, so any co-resident number quoted
 * from a run that did not verify overlap is worthless.
 *
 * Usage: node bench-coresident.mjs [native:BATCH] [vulkan:BATCH] [seq] [runMs]
 *   node bench-coresident.mjs native:128 vulkan:16 32 20000
 */
import { spawn } from "node:child_process";

const A = (process.argv[2] ?? "native:128").split(":");
const Bk = (process.argv[3] ?? "vulkan:16").split(":");
const SEQ = process.argv[4] ?? "32";
const RUN_MS = process.argv[5] ?? "20000";
const WARMUP_MS = process.env.WARMUP_MS ?? "4000";
const STEADY = "/workspace/alpha2/packages/tests/bench-steady.mjs";

function run(backend, batch) {
  return new Promise((resolve) => {
    const p = spawn("node", [STEADY, backend, batch, SEQ, WARMUP_MS, RUN_MS],
                    { stdio: ["ignore", "pipe", "pipe"] });
    let out = "", err = "";
    p.stdout.on("data", (d) => { out += d; });
    p.stderr.on("data", (d) => { err += d; process.stderr.write(d); });
    p.on("close", (code) => {
      if (code !== 0) { resolve({ failed: `${backend} exited ${code}`, err }); return; }
      try { resolve(JSON.parse(out.trim().split("\n").pop())); }
      catch (e) { resolve({ failed: `${backend} unparseable output`, err, out }); }
    });
  });
}

/* Median over the steps that fall ENTIRELY inside [lo,hi]. A step straddling a
 * boundary was partly alone, so it is evidence for neither side. */
function windowed(r, lo, hi) {
  const inside = r.steps.filter(([s, e]) => s >= lo && e <= hi);
  if (inside.length === 0) return null;
  const ms = inside.map(([s, e]) => e - s).sort((a, b) => a - b);
  const med = ms[Math.floor(ms.length / 2)];
  return { n: inside.length, med, lo: ms[0], hi: ms[ms.length - 1],
           tps: r.tokensPerStep / (med / 1000) };
}

function solo(r) {
  const ms = r.steps.map(([s, e]) => e - s).sort((a, b) => a - b);
  const med = ms[Math.floor(ms.length / 2)];
  return { n: ms.length, med, tps: r.tokensPerStep / (med / 1000) };
}

console.log(`co-resident: ${A[0]} batch ${A[1]} + ${Bk[0]} batch ${Bk[1]}, seq ${SEQ}, ` +
            `${WARMUP_MS}ms warmup + ${RUN_MS}ms run\n`);

const [ra, rb] = await Promise.all([run(A[0], A[1]), run(Bk[0], Bk[1])]);
for (const r of [ra, rb]) if (r.failed) { console.error(`FAILED: ${r.failed}`); process.exit(1); }

/* The overlap window is where both were measuring. Steps outside it were run
 * against an idle or a warming device and would flatter whichever side got
 * there first. */
const lo = Math.max(ra.steps[0][0], rb.steps[0][0]);
const hi = Math.min(ra.steps[ra.steps.length - 1][1], rb.steps[rb.steps.length - 1][1]);
console.log(`overlap window ${((hi - lo) / 1000).toFixed(1)} s\n`);
if (hi <= lo) { console.error("NO OVERLAP — the two runs did not share any wall-clock time"); process.exit(1); }

console.log("backend         batch    tok/s (overlap)   ms/step   steps   tok/s (whole run)   loss");
for (const r of [ra, rb]) {
  const w = windowed(r, lo, hi), s = solo(r);
  if (!w) { console.log(`${r.backend.padEnd(15)} ${String(r.batch).padStart(5)}    no steps inside the overlap window`); continue; }
  console.log(`${r.backend.padEnd(15)} ${String(r.batch).padStart(5)}   ` +
              `${w.tps.toFixed(0).padStart(14)}   ${String(w.med).padStart(7)}   ${String(w.n).padStart(5)}   ` +
              `${s.tps.toFixed(0).padStart(17)}   ${r.loss.toFixed(4)}`);
}
const wa = windowed(ra, lo, hi), wb = windowed(rb, lo, hi);
if (wa && wb) console.log(`\ncombined ${(wa.tps + wb.tps).toFixed(0)} tok/s on one device`);
