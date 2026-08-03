#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

if (process.argv.length < 3 || process.argv.length > 6) {
  console.error("usage: analyze_helios_buffer_lifetimes.mjs TRACE.jsonl [--step N] [--json]");
  process.exit(2);
}

const tracePath = resolve(process.argv[2]);
let selectedStep = null;
let jsonOutput = false;
for (let index = 3; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--json") {
    jsonOutput = true;
  } else if (process.argv[index] === "--step") {
    selectedStep = Number.parseInt(process.argv[++index] ?? "", 10);
    if (!Number.isInteger(selectedStep)) throw new Error("--step requires an integer");
  } else {
    throw new Error(`Unknown argument: ${process.argv[index]}`);
  }
}

const rows = readFileSync(tracePath, "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .filter((row) => selectedStep === null || row.step === selectedStep);
if (rows.length === 0) throw new Error(`No selected traces in ${tracePath}`);

function align(value, alignment = 256) {
  return Math.ceil(value / alignment) * alignment;
}

function isWrite(writeMask, position) {
  return (writeMask & (2 ** position)) !== 0;
}

function allocateIntervals(intervals) {
  const allocated = [];
  let arenaBytes = 0;
  const ordered = [...intervals].sort((a, b) => a.start - b.start || b.bytes - a.bytes || a.valueId - b.valueId);
  for (const interval of ordered) {
    const active = allocated
      .filter((candidate) => candidate.lastUse >= interval.start)
      .sort((a, b) => a.offset - b.offset);
    let cursor = 0;
    for (const candidate of active) {
      const candidateStart = candidate.offset;
      if (cursor + interval.bytes <= candidateStart) break;
      cursor = Math.max(cursor, candidate.offset + candidate.bytes);
      cursor = align(cursor);
    }
    interval.offset = cursor;
    arenaBytes = Math.max(arenaBytes, cursor + interval.bytes);
    allocated.push(interval);
  }
  return { arenaBytes: align(arenaBytes), intervals: allocated };
}

function analyzeStep(row) {
  const operations = row.events.filter((event) => event.event === "op");
  const currentValue = new Map();
  const values = [];
  const physicalBytes = new Map();
  let nextValueId = 0;

  function newValue({ physicalId, bytes, start, producer, external, persistentMutation }) {
    const value = {
      valueId: nextValueId++,
      physicalId,
      bytes,
      start,
      lastUse: start,
      producer,
      external,
      persistentMutation,
    };
    values.push(value);
    currentValue.set(physicalId, value);
    return value;
  }

  for (let operationIndex = 0; operationIndex < operations.length; operationIndex += 1) {
    const operation = operations[operationIndex];
    if (!Array.isArray(operation.bufferIds) || !Array.isArray(operation.bufferBytes)) {
      throw new Error(`Step ${row.step} operation ${operationIndex} lacks bufferIds/bufferBytes; capture with the lifetime-trace revision`);
    }
    if (operation.bufferIds.length !== operation.bufferCount || operation.bufferBytes.length !== operation.bufferCount) {
      throw new Error(`Step ${row.step} operation ${operationIndex} buffer metadata length does not match bufferCount`);
    }
    for (let position = 0; position < operation.bufferCount; position += 1) {
      const physicalId = operation.bufferIds[position];
      const bytes = operation.bufferBytes[position];
      if (!Number.isInteger(physicalId) || !Number.isFinite(bytes) || bytes <= 0) {
        throw new Error(`Step ${row.step} operation ${operationIndex} has invalid buffer metadata at position ${position}`);
      }
      physicalBytes.set(physicalId, Math.max(physicalBytes.get(physicalId) ?? 0, bytes));
      const write = isWrite(operation.writeMask, position);
      let value = currentValue.get(physicalId);
      if (!write) {
        if (!value) value = newValue({ physicalId, bytes, start: 0, producer: null, external: true, persistentMutation: false });
        value.lastUse = Math.max(value.lastUse, operationIndex);
        continue;
      }

      const persistentMutation = operation.kind === "optimizer" || operation.kind === "inplace";
      if (value && persistentMutation) value.lastUse = Math.max(value.lastUse, operationIndex);
      newValue({
        physicalId,
        bytes,
        start: operationIndex,
        producer: { operation: operationIndex, kind: operation.kind, kernel: operation.kernel, position },
        external: false,
        persistentMutation,
      });
    }
  }

  const transient = values
    .filter((value) => !value.external && !value.persistentMutation)
    .map((value) => ({ ...value }));
  const { arenaBytes, intervals } = allocateIntervals(transient);
  let peakLiveBytes = 0;
  let peakLiveOperation = 0;
  for (let operationIndex = 0; operationIndex < operations.length; operationIndex += 1) {
    const liveBytes = transient
      .filter((value) => value.start <= operationIndex && value.lastUse >= operationIndex)
      .reduce((sum, value) => sum + value.bytes, 0);
    if (liveBytes > peakLiveBytes) {
      peakLiveBytes = liveBytes;
      peakLiveOperation = operationIndex;
    }
  }
  const physicalBytesObserved = [...physicalBytes.values()].reduce((sum, bytes) => sum + bytes, 0);
  const totalTransientBytesCreated = transient.reduce((sum, value) => sum + value.bytes, 0);
  const planRows = intervals
    .map((value) => [value.start, value.lastUse, value.bytes, value.producer.kind, value.producer.kernel, value.producer.position])
    .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  const planFingerprint = createHash("sha256").update(JSON.stringify(planRows)).digest("hex");
  const topLifetimes = [...intervals]
    .sort((a, b) => b.bytes * (b.lastUse - b.start + 1) - a.bytes * (a.lastUse - a.start + 1))
    .slice(0, 20)
    .map((value) => ({
      valueId: value.valueId,
      bytes: value.bytes,
      start: value.start,
      lastUse: value.lastUse,
      lifetimeOperations: value.lastUse - value.start + 1,
      offset: value.offset,
      producerKind: value.producer.kind,
      producerKernel: value.producer.kernel,
    }));
  return {
    step: row.step,
    operations: operations.length,
    physicalBuffersObserved: physicalBytes.size,
    physicalBytesObserved,
    logicalValues: values.length,
    externalValues: values.filter((value) => value.external).length,
    persistentMutationValues: values.filter((value) => value.persistentMutation).length,
    transientValues: transient.length,
    totalTransientBytesCreated,
    peakLiveTransientBytes: peakLiveBytes,
    peakLiveOperation,
    greedyArenaBytes: arenaBytes,
    arenaFragmentationBytes: arenaBytes - peakLiveBytes,
    temporalReuseVsCreated: arenaBytes > 0 ? totalTransientBytesCreated / arenaBytes : null,
    planFingerprint,
    topLifetimes,
  };
}

const analyses = rows.map(analyzeStep);
const uniquePlanFingerprints = [...new Set(analyses.map((row) => row.planFingerprint))];
const result = {
  schemaVersion: 1,
  tracePath,
  steps: analyses.length,
  planStable: uniquePlanFingerprints.length === 1,
  uniquePlanFingerprints,
  analyses,
  limitations: [
    "This is an offline interval plan, not a measured allocator or command-replay speedup.",
    "Writable descriptors are conservatively treated as new logical versions; in-place and optimizer writes are excluded from the transient arena.",
    "Intervals overlap at a shared operation boundary, so the plan does not assume unsafe read/write aliasing inside one kernel.",
  ],
};

if (jsonOutput) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
}

console.log("# Helios buffer-lifetime and static-arena analysis\n");
console.log(`Trace: \`${tracePath}\``);
console.log(`Steps: ${result.steps}`);
console.log(`Logical lifetime plan stable: **${result.planStable ? "yes" : "no"}**\n`);
console.log("| Step | Ops | Physical buffers | Physical GiB observed | Transient values | Transient GiB created | Peak live GiB | Greedy arena GiB | Temporal reuse | Plan fingerprint |");
console.log("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
for (const row of analyses) {
  const gib = (bytes) => (bytes / (1024 ** 3)).toFixed(3);
  console.log(`| ${row.step} | ${row.operations} | ${row.physicalBuffersObserved} | ${gib(row.physicalBytesObserved)} | ${row.transientValues} | ${gib(row.totalTransientBytesCreated)} | ${gib(row.peakLiveTransientBytes)} | ${gib(row.greedyArenaBytes)} | ${row.temporalReuseVsCreated?.toFixed(2) ?? "n/a"}x | \`${row.planFingerprint.slice(0, 16)}\` |`);
}
console.log("\nThe arena result is a planning estimate only. It must be validated by implementing the plan, checking exact outputs/gradients, and timing a bounded RTX 3090 run.");
