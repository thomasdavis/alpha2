#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

function usage() {
  console.error("Usage: node scripts/summarize_helios_slab_cap_runs.mjs OUTPUT_JSON LABEL=RUN_DIR [...]");
  process.exit(2);
}

const [outputArg, ...runArgs] = process.argv.slice(2);
if (!outputArg || runArgs.length < 2) usage();

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

const runs = runArgs.map((argument) => {
  const separator = argument.indexOf("=");
  if (separator <= 0 || separator === argument.length - 1) usage();
  const label = argument.slice(0, separator);
  const runDir = resolve(argument.slice(separator + 1));
  const consoleBytes = readFileSync(`${runDir}/console.log`);
  const consoleText = consoleBytes.toString("utf8");
  const capMatch = consoleText.match(/temporary slab pool cap=(\d+)MB(?: \(([^)]+)\))?/);
  if (!capMatch) throw new Error(`${label}: missing temporary slab cap in console.log`);
  const metricBytes = readFileSync(`${runDir}/metrics.jsonl`);
  const metrics = metricBytes.toString("utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const steady = metrics.filter((row) => row.step >= 2);
  if (steady.length === 0) throw new Error(`${label}: no steady metrics at step >= 2`);
  return {
    label,
    runDir,
    capMiB: Number(capMatch[1]),
    capMode: capMatch[2] ?? "legacy",
    steadySteps: steady.length,
    meanTokensPerSecond: mean(steady.map((row) => row.tokens_per_sec)),
    meanStepMs: mean(steady.map((row) => row.elapsed_ms)),
    meanHostBuildMs: mean(steady.map((row) => row.timing_host_build_ms)),
    meanGpuBlockingMs: mean(steady.map((row) => row.timing_gpu_blocking_ms)),
    finalSlabFallbacks: steady.at(-1).gpu_allocator_slab_fallbacks,
    finalFreeRangeReuses: steady.at(-1).gpu_allocator_free_range_reuses,
    losses: steady.map((row) => row.loss),
    gradNorms: steady.map((row) => row.gradNorm),
    metricsSha256: sha256(metricBytes),
    consoleSha256: sha256(consoleBytes),
  };
});

const referenceLosses = JSON.stringify(runs[0].losses);
const lossTrajectoryExact = runs.every((run) => JSON.stringify(run.losses) === referenceLosses);
const caps = [...new Set(runs.map((run) => run.capMiB))].sort((a, b) => a - b);
const aggregates = caps.map((capMiB) => {
  const members = runs.filter((run) => run.capMiB === capMiB);
  return {
    capMiB,
    runs: members.length,
    steadySteps: members.reduce((sum, run) => sum + run.steadySteps, 0),
    meanTokensPerSecond: mean(members.map((run) => run.meanTokensPerSecond)),
    meanStepMs: mean(members.map((run) => run.meanStepMs)),
    meanHostBuildMs: mean(members.map((run) => run.meanHostBuildMs)),
    meanGpuBlockingMs: mean(members.map((run) => run.meanGpuBlockingMs)),
  };
});

const low = aggregates[0];
const high = aggregates.at(-1);
const result = {
  schemaVersion: 1,
  createdAt: new Date().toISOString(),
  steadyStepRule: "step >= 2",
  lossTrajectoryExact,
  runs,
  aggregates,
  comparison: caps.length === 2 ? {
    baselineCapMiB: low.capMiB,
    candidateCapMiB: high.capMiB,
    tokensPerSecondRatio: high.meanTokensPerSecond / low.meanTokensPerSecond,
    tokensPerSecondGainPct: (high.meanTokensPerSecond / low.meanTokensPerSecond - 1) * 100,
    stepTimeReductionPct: (1 - high.meanStepMs / low.meanStepMs) * 100,
    hostBuildReductionPct: (1 - high.meanHostBuildMs / low.meanHostBuildMs) * 100,
  } : null,
};

writeFileSync(resolve(outputArg), `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result.comparison, null, 2));
