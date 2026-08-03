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
const result = {
  schemaVersion: 1,
  tracePath,
  steps: rows.length,
  uniqueSignatures: [...new Set(rows.map((row) => row.graphSignature))],
  exactTopologyStable: comparisons.every(({ firstDifferenceIndex }) => firstDifferenceIndex === null),
  eventsPerStep: rows.map((row) => ({
    step: row.step,
    total: row.events.length,
    operations: row.events.filter((event) => event.event === "op").length,
    flushes: row.events.filter((event) => event.event === "flush").length,
  })),
  comparisons,
};

if (jsonOutput) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
}

console.log("# Helios ordered graph-trace comparison\n");
console.log(`Trace: \`${tracePath}\``);
console.log(`Steps: ${result.steps}`);
console.log(`Unique structural signatures: ${result.uniqueSignatures.length}`);
console.log(`Exact ordered topology stable: **${result.exactTopologyStable ? "yes" : "no"}**\n`);
console.log("| Reference | Candidate | First difference | Common prefix | Common suffix | Reference event | Candidate event |");
console.log("|---:|---:|---:|---:|---:|---|---|");
for (const comparison of comparisons) {
  console.log(
    `| ${comparison.referenceStep} | ${comparison.candidateStep} | ${comparison.firstDifferenceIndex ?? "none"} | ` +
    `${comparison.commonPrefixEvents} | ${comparison.commonSuffixEvents} | ` +
    `\`${JSON.stringify(comparison.referenceEvent)}\` | \`${JSON.stringify(comparison.candidateEvent)}\` |`,
  );
}
