#!/usr/bin/env node
/**
 * X59 — does BDA actually eliminate the descriptor phase?
 *
 * X56/X57/X58 assert that buffer-device-address kernels skip descriptor work,
 * and X39 measured desc_update at 23.4% of host time (per-dispatch). That is a
 * claim about CALL COUNTS, which is a structural property of the code path and
 * transfers off lavapipe even though absolute microseconds do not.
 *
 * This counts them. If the claim is right, the slot path issues strictly fewer
 * desc_update calls than the tree, by exactly the number of BDA dispatches.
 */
import { HeliosBackend } from "../packages/helios/dist/backend.js";

const N = 128, SIZE = 70000;
const b = new HeliosBackend();
const tds = [];
for (let t = 0; t < N; t++) tds.push(b.fromArray(new Float32Array(SIZE).fill(1.0), [SIZE]));

b.resetNativeHostTiming?.();
const ops0 = b.gpuOpsThisStep;
const r = b.totalSumOfSquares(tds);
const val = Number(r.data[0]);
const ops = b.gpuOpsThisStep - ops0;
const ht = b.getNativeHostTiming?.() ?? null;

const slot = process.env.HELIOS_SUMSQ_SLOT_REDUCE ?? "0";
if (!ht) { console.log(`  slot=${slot} ops=${ops} (no host timing; set HELIOS_HOST_TIMING=1)`); process.exit(0); }
const calls = (n) => ht.phases?.[n]?.calls ?? 0;
const us = (n) => ht.phases?.[n]?.us ?? 0;
console.log(`  slot=${slot}  ops=${ops}  value=${val === N*SIZE ? "exact" : "WRONG"}  ` +
            `dispatches=${ht.dispatches}`);
for (const n of ["desc_alloc", "desc_update", "bind", "push_const", "cmd_dispatch", "barrier"]) {
  console.log(`      ${n.padEnd(13)} calls=${String(calls(n)).padStart(5)}  us=${us(n)}`);
}
