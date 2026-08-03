#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

function quantile(sorted, probability) {
  if (sorted.length === 1) return sorted[0];
  const index = probability * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    samples: sorted.length,
    min: sorted[0],
    p10: quantile(sorted, 0.1),
    median: quantile(sorted, 0.5),
    mean: mean(sorted),
    p90: quantile(sorted, 0.9),
    max: sorted.at(-1),
  };
}

export function summarizeAB(specs, warmupSteps = 1) {
  if (specs.length < 2) throw new Error("at least two labelled runs are required");
  const labels = [...new Set(specs.map((spec) => spec.label))];
  if (labels.length !== 2) throw new Error(`exactly two labels are required, found ${labels.length}`);

  const runs = specs.map((spec) => {
    const runDir = resolve(spec.runDir);
    const bytes = readFileSync(`${runDir}/metrics.jsonl`);
    const rows = bytes.toString("utf8").trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
    const steady = rows.filter((row) => row.step > warmupSteps);
    if (steady.length === 0) throw new Error(`${spec.label}=${runDir}: no rows after warmup`);
    return {
      label: spec.label,
      runDir,
      metricsSha256: sha256(bytes),
      totalSteps: rows.length,
      steadySteps: steady.length,
      meanTokensPerSecond: mean(steady.map((row) => row.tokens_per_sec)),
      rows,
      steady,
    };
  });

  let maxLossDifference = 0;
  let maxGradNormDifference = 0;
  const byStep = new Map();
  for (const run of runs) {
    for (const row of run.rows) {
      const reference = byStep.get(row.step);
      if (!reference) {
        byStep.set(row.step, row);
      } else {
        maxLossDifference = Math.max(maxLossDifference, Math.abs(row.loss - reference.loss));
        maxGradNormDifference = Math.max(maxGradNormDifference, Math.abs(row.gradNorm - reference.gradNorm));
      }
    }
  }

  const arms = Object.fromEntries(labels.map((label) => {
    const members = runs.filter((run) => run.label === label);
    const values = members.flatMap((run) => run.steady.map((row) => row.tokens_per_sec));
    return [label, {
      ...stats(values),
      runs: members.map((run) => ({
        runDir: run.runDir,
        metricsSha256: run.metricsSha256,
        totalSteps: run.totalSteps,
        steadySteps: run.steadySteps,
        meanTokensPerSecond: run.meanTokensPerSecond,
      })),
    }];
  }));

  const baseline = arms[labels[0]];
  const candidate = arms[labels[1]];
  return {
    schema: "alpha-helios-ab-summary-v1",
    createdAt: new Date().toISOString(),
    warmupRule: `step > ${warmupSteps}`,
    baselineLabel: labels[0],
    candidateLabel: labels[1],
    trajectoryParity: {
      maxLossDifference,
      maxGradNormDifference,
      exactAtRecordedPrecision: maxLossDifference === 0 && maxGradNormDifference === 0,
    },
    arms,
    comparison: {
      meanTokensPerSecondGainPct: 100 * (candidate.mean / baseline.mean - 1),
      medianTokensPerSecondGainPct: 100 * (candidate.median / baseline.median - 1),
      minimumTokensPerSecondGainPct: 100 * (candidate.min / baseline.min - 1),
      p10TokensPerSecondGainPct: 100 * (candidate.p10 / baseline.p10 - 1),
    },
  };
}

function format(value, digits = 2) {
  return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function markdown(summary) {
  const lines = [
    "# Helios A/B sustained-throughput summary",
    "",
    `Warmup exclusion: \`${summary.warmupRule}\``,
    `Trajectory parity: ${summary.trajectoryParity.exactAtRecordedPrecision ? "exact at recorded precision" : "DIFFERENT"}`,
    "",
    "| Arm | Samples | Min | p10 | Median | Mean | p90 | Max |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const [label, arm] of Object.entries(summary.arms)) {
    lines.push(`| ${label} | ${arm.samples} | ${format(arm.min)} | ${format(arm.p10)} | ${format(arm.median)} | ${format(arm.mean)} | ${format(arm.p90)} | ${format(arm.max)} |`);
  }
  lines.push(
    "",
    "## Candidate change",
    "",
    `- mean tokens/s: ${format(summary.comparison.meanTokensPerSecondGainPct, 4)}%`,
    `- median tokens/s: ${format(summary.comparison.medianTokensPerSecondGainPct, 4)}%`,
    `- minimum tokens/s: ${format(summary.comparison.minimumTokensPerSecondGainPct, 4)}%`,
    `- p10 tokens/s: ${format(summary.comparison.p10TokensPerSecondGainPct, 4)}%`,
    "",
    "## Runs",
    "",
  );
  for (const [label, arm] of Object.entries(summary.arms)) {
    for (const run of arm.runs) {
      lines.push(`- ${label}: \`${run.runDir}\` — ${run.steadySteps} steady samples, SHA-256 \`${run.metricsSha256}\``);
    }
  }
  return `${lines.join("\n")}\n`;
}

function usage() {
  return "Usage: node scripts/summarize_helios_ab_runs.mjs [--warmup-steps N] [--markdown FILE] OUTPUT_JSON LABEL=RUN_DIR [...]";
}

function run(argv) {
  let warmupSteps = 1;
  let markdownPath = null;
  const positional = [];
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === "--warmup-steps") warmupSteps = Number(argv[++index]);
    else if (argv[index] === "--markdown") markdownPath = resolve(argv[++index]);
    else positional.push(argv[index]);
  }
  const [outputPath, ...runArgs] = positional;
  if (!outputPath || runArgs.length < 2 || !Number.isInteger(warmupSteps) || warmupSteps < 0) throw new Error(usage());
  const specs = runArgs.map((argument) => {
    const separator = argument.indexOf("=");
    if (separator <= 0 || separator === argument.length - 1) throw new Error(usage());
    return { label: argument.slice(0, separator), runDir: argument.slice(separator + 1) };
  });
  const summary = summarizeAB(specs, warmupSteps);
  writeFileSync(resolve(outputPath), `${JSON.stringify(summary, null, 2)}\n`);
  if (markdownPath) writeFileSync(markdownPath, markdown(summary));
  process.stdout.write(`${JSON.stringify(summary.comparison, null, 2)}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    run(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
