#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

if (process.argv.length < 3 || process.argv.length > 4) {
  console.error("usage: summarize_helios_graph_trace.mjs TRACE.jsonl [--json]");
  process.exit(2);
}

const tracePath = resolve(process.argv[2]);
const jsonOutput = process.argv[3] === "--json";
const rows = readFileSync(tracePath, "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));

if (rows.length < 2) {
  throw new Error(`Expected at least two step traces in ${tracePath}; found ${rows.length}`);
}

function same(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function firstDifference(reference, candidate) {
  const commonLength = Math.min(reference.length, candidate.length);
  let index = 0;
  while (index < commonLength && same(reference[index], candidate[index])) index += 1;
  let suffix = 0;
  while (
    suffix < commonLength - index &&
    same(reference[reference.length - 1 - suffix], candidate[candidate.length - 1 - suffix])
  ) suffix += 1;
  return {
    firstDifferenceIndex: index === commonLength && reference.length === candidate.length ? null : index,
    commonPrefixEvents: index,
    commonSuffixEvents: suffix,
    referenceEventCount: reference.length,
    candidateEventCount: candidate.length,
    referenceEvent: index < reference.length ? reference[index] : null,
    candidateEvent: index < candidate.length ? candidate[index] : null,
  };
}

const reference = rows[0];
const comparisons = rows.slice(1).map((row) => ({
  referenceStep: reference.step,
  candidateStep: row.step,
  referenceSignature: reference.graphSignature,
  candidateSignature: row.graphSignature,
  ...firstDifference(reference.events, row.events),
}));
function operationEvents(row, includeBindings) {
  return row.events
    .filter((event) => event.event === "op")
    .map(({ order: _order, bufferIds, ...operation }) => (
      includeBindings ? { ...operation, ...(bufferIds === undefined ? {} : { bufferIds }) } : operation
    ));
}

function flushSchedule(row) {
  let completedOperations = 0;
  const schedule = [];
  for (const event of row.events) {
    if (event.event === "op") {
      completedOperations += 1;
    } else if (event.event === "flush") {
      schedule.push({
        afterOperation: completedOperations,
        operationCount: event.operationCount,
        withWait: event.withWait,
      });
    }
  }
  return schedule;
}

const referenceOperations = operationEvents(reference, false);
const operationComparisons = rows.slice(1).map((row) => ({
  referenceStep: reference.step,
  candidateStep: row.step,
  ...firstDifference(referenceOperations, operationEvents(row, false)),
}));
const referenceBufferBindings = operationEvents(reference, true);
const bufferBindingComparisons = rows.slice(1).map((row) => ({
  referenceStep: reference.step,
  candidateStep: row.step,
  ...firstDifference(referenceBufferBindings, operationEvents(row, true)),
}));
const flushSchedules = rows.map((row) => ({ step: row.step, flushes: flushSchedule(row) }));
const serializedReferenceFlushes = JSON.stringify(flushSchedules[0].flushes);
const result = {
  schemaVersion: 1,
  tracePath,
  steps: rows.length,
  uniqueSignatures: [...new Set(rows.map((row) => row.graphSignature))],
  exactEventStreamStable: comparisons.every(({ firstDifferenceIndex }) => firstDifferenceIndex === null),
  operationTopologyStable: operationComparisons.every(({ firstDifferenceIndex }) => firstDifferenceIndex === null),
  bufferBindingTopologyStable: bufferBindingComparisons.every(({ firstDifferenceIndex }) => firstDifferenceIndex === null),
  flushScheduleStable: flushSchedules.every(({ flushes }) => JSON.stringify(flushes) === serializedReferenceFlushes),
  eventsPerStep: rows.map((row) => ({
    step: row.step,
    total: row.events.length,
    operations: row.events.filter((event) => event.event === "op").length,
    flushes: row.events.filter((event) => event.event === "flush").length,
  })),
  comparisons,
  operationComparisons,
  bufferBindingComparisons,
  flushSchedules,
};

if (jsonOutput) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
}

console.log("# Helios ordered graph-trace comparison\n");
console.log(`Trace: \`${tracePath}\``);
console.log(`Steps: ${result.steps}`);
console.log(`Unique structural signatures: ${result.uniqueSignatures.length}`);
console.log(`Exact event stream stable: **${result.exactEventStreamStable ? "yes" : "no"}**`);
console.log(`Operation-only topology stable: **${result.operationTopologyStable ? "yes" : "no"}**`);
console.log(`Physical buffer-binding topology stable: **${result.bufferBindingTopologyStable ? "yes" : "no"}**`);
console.log(`Flush schedule stable: **${result.flushScheduleStable ? "yes" : "no"}**\n`);
console.log("| Reference | Candidate | First difference | Common prefix | Common suffix | Reference event | Candidate event |");
console.log("|---:|---:|---:|---:|---:|---|---|");
for (const comparison of comparisons) {
  console.log(
    `| ${comparison.referenceStep} | ${comparison.candidateStep} | ${comparison.firstDifferenceIndex ?? "none"} | ` +
    `${comparison.commonPrefixEvents} | ${comparison.commonSuffixEvents} | ` +
    `\`${JSON.stringify(comparison.referenceEvent)}\` | \`${JSON.stringify(comparison.candidateEvent)}\` |`,
  );
}
console.log("\n## Flush schedules\n");
console.log("| Step | Flush after operation | Operations in submitted batch | Waited |\n|---:|---:|---:|:---:|");
for (const row of flushSchedules) {
  for (const flush of row.flushes) {
    console.log(`| ${row.step} | ${flush.afterOperation} | ${flush.operationCount} | ${flush.withWait ? "yes" : "no"} |`);
  }
}
