#!/usr/bin/env node
/**
 * X40 — rank remaining fusion opportunities from a physical operation trace.
 *
 * Why
 * ---
 * X39 established that host cost is per-dispatch dominated: the phases scaling
 * with operation count (desc_update, barrier, push_const, cmd_dispatch, decode,
 * bind) are 65.3% of host time, all with one call per dispatch. Removing an
 * operation therefore removes host cost *and* a GPU dispatch, unlike an
 * arithmetic optimisation which only touches the latter.
 *
 * So the question "which operations should be fused next" is answerable from
 * the real graph rather than from intuition. This scans a preserved physical
 * RTX 3090 trace and ranks adjacent kernel pairs by how many dispatches their
 * fusion would remove.
 *
 * Method and its limit
 * --------------------
 * The trace records execution order, kernel identity, buffer count, write mask,
 * dispatch geometry and logical shape — but NOT buffer identities. So this
 * cannot prove a dataflow dependency between adjacent operations. It ranks
 * *candidates*; each still needs its producer/consumer relationship confirmed
 * in the backend before anyone writes a fused kernel.
 *
 * Adjacency is nevertheless a strong prior here: the graph is emitted in
 * execution order, and consecutive operations inside a transformer block are
 * overwhelmingly producer -> consumer.
 *
 * Usage:
 *   node scripts/x40-fusion-opportunity-scan.mjs <gpu-graph-trace.jsonl> [--json out.json]
 */

import { readFileSync, writeFileSync } from "node:fs";

const path = process.argv[2];
if (!path) {
  console.error("usage: x40-fusion-opportunity-scan.mjs <gpu-graph-trace.jsonl> [--json out.json]");
  process.exit(2);
}
const jsonIdx = process.argv.indexOf("--json");
const jsonOut = jsonIdx > 0 ? process.argv[jsonIdx + 1] : null;

const steps = readFileSync(path, "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

// Use the last recorded step: the first is warmup and may include one-time work.
const step = steps[steps.length - 1];
const ops = step.events.filter((e) => e.event === "op");

const byKernel = new Map();
for (const o of ops) {
  const e = byKernel.get(o.kernel) ?? { count: 0, kind: o.kind, groups: 0 };
  e.count++;
  e.groups += (o.groups?.[0] ?? 1) * (o.groups?.[1] ?? 1) * (o.groups?.[2] ?? 1);
  byKernel.set(o.kernel, e);
}

// Adjacent pairs, in execution order.
const bigrams = new Map();
for (let i = 0; i + 1 < ops.length; i++) {
  const k = `${ops[i].kernel} -> ${ops[i + 1].kernel}`;
  const e = bigrams.get(k) ?? { count: 0, a: ops[i].kernel, b: ops[i + 1].kernel };
  e.count++;
  bigrams.set(k, e);
}

// Runs of the same kernel back to back — these fuse into one batched dispatch
// far more easily than heterogeneous pairs, because no new kernel is needed.
const runs = new Map();
let i = 0;
while (i < ops.length) {
  let j = i;
  while (j + 1 < ops.length && ops[j + 1].kernel === ops[i].kernel) j++;
  const len = j - i + 1;
  if (len > 1) {
    const e = runs.get(ops[i].kernel) ?? { occurrences: 0, totalOps: 0, maxRun: 0 };
    e.occurrences++;
    e.totalOps += len;
    e.maxRun = Math.max(e.maxRun, len);
    runs.set(ops[i].kernel, e);
  }
  i = j + 1;
}

const total = ops.length;
const sortedKernels = [...byKernel.entries()].sort((a, b) => b[1].count - a[1].count);
const sortedBigrams = [...bigrams.values()].sort((a, b) => b.count - a.count);
const sortedRuns = [...runs.entries()].sort((a, b) => (b[1].totalOps - b[1].occurrences) - (a[1].totalOps - a[1].occurrences));

console.log(`trace: ${path}`);
console.log(`steps recorded: ${steps.length}; analysing step ${step.step} (graph ${step.graphSignature})`);
console.log(`operations in step: ${total}\n`);

console.log("── operations by kernel (top 20) ──");
console.log("count   %total  kind         kernel");
for (const [k, v] of sortedKernels.slice(0, 20)) {
  console.log(
    `${String(v.count).padStart(5)}  ${(100 * v.count / total).toFixed(1).padStart(5)}%  ` +
    `${(v.kind ?? "").padEnd(12)} ${k}`,
  );
}

console.log("\n── adjacent kernel pairs (top 20 fusion candidates) ──");
console.log("occurs  dispatches_saved  pair");
for (const b of sortedBigrams.slice(0, 20)) {
  console.log(`${String(b.count).padStart(6)}  ${String(b.count).padStart(16)}  ${b.a} -> ${b.b}`);
}

console.log("\n── back-to-back runs of one kernel (batchable without a new kernel) ──");
console.log("kernel                              runs  ops  max_run  dispatches_saved");
for (const [k, v] of sortedRuns.slice(0, 15)) {
  console.log(
    `${k.padEnd(36)} ${String(v.occurrences).padStart(4)} ${String(v.totalOps).padStart(4)} ` +
    `${String(v.maxRun).padStart(8)} ${String(v.totalOps - v.occurrences).padStart(17)}`,
  );
}

const runSavings = [...runs.values()].reduce((s, v) => s + v.totalOps - v.occurrences, 0);
const top5Pair = sortedBigrams.slice(0, 5).reduce((s, b) => s + b.count, 0);

console.log("\n── headline ──");
console.log(`total operations                       ${total}`);
console.log(`removable by batching same-kernel runs ${runSavings} (${(100 * runSavings / total).toFixed(1)}%)`);
console.log(`removable by fusing the top 5 pairs    ${top5Pair} (${(100 * top5Pair / total).toFixed(1)}%)`);
console.log("\nThese are upper bounds on dispatch removal, not speedups. Each candidate");
console.log("requires a confirmed producer->consumer dependency and a correctness gate.");

if (jsonOut) {
  writeFileSync(jsonOut, JSON.stringify({
    trace: path,
    step: step.step,
    graphSignature: step.graphSignature,
    totalOps: total,
    byKernel: Object.fromEntries(sortedKernels.map(([k, v]) => [k, v])),
    topPairs: sortedBigrams.slice(0, 40),
    sameKernelRuns: Object.fromEntries(sortedRuns),
    removableByRunBatching: runSavings,
    removableByTop5Pairs: top5Pair,
  }, null, 2));
  console.log(`\nwrote ${jsonOut}`);
}
