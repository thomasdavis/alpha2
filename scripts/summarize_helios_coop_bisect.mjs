#!/usr/bin/env node

import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

function parseArgs(argv) {
  const options = new Map();
  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const name = arg.slice(2);
    const value = argv[++index];
    if (value === undefined) throw new Error(`missing value for --${name}`);
    options.set(name, value);
  }
  return options;
}

function parseFinite(raw) {
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function parseRow(root, name) {
  const rowRoot = join(root, name);
  const logPath = join(rowRoot, "console.log");
  const log = readFileSync(logPath, "utf8");
  const steps = [];
  const stepPattern = /^step (\d+)\/(\d+) \| loss=([^ |]+).*?grad_norm=([^ |]+).*?\| ([0-9.]+) tok\/s/mg;
  for (const match of log.matchAll(stepPattern)) {
    steps.push({
      step: Number(match[1]),
      totalSteps: Number(match[2]),
      loss: parseFinite(match[3]),
      lossRaw: match[3],
      gradNorm: parseFinite(match[4]),
      gradNormRaw: match[4],
      tokensPerSecond: Number(match[5]),
    });
  }
  const shapeLines = [...log.matchAll(/^(?:\s*\[coop_shapes\]\s+|coop_shapes:\s+)(.+)$/mg)];
  let coopShapes = [];
  if (shapeLines.length > 0) {
    coopShapes = JSON.parse(shapeLines.at(-1)[1]);
  }
  const exitCode = Number(readFileSync(join(rowRoot, "exit-code.txt"), "utf8").trim());
  const controlledEnvironment = readFileSync(join(rowRoot, "CONTROLLED-ENVIRONMENT.txt"), "utf8")
    .trim().split("\n").filter(Boolean);
  return { name, exitCode, controlledEnvironment, steps, coopShapes };
}

function median(values) {
  if (values.length === 0) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 1 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

const options = parseArgs(process.argv);
const root = resolve(options.get("root") ?? ".");
const jsonOutput = resolve(options.get("json") ?? join(root, "summary.json"));
const markdownOutput = resolve(options.get("markdown") ?? join(root, "SUMMARY.md"));
const rowNames = readdirSync(root)
  .filter((name) => statSync(join(root, name)).isDirectory())
  .filter((name) => {
    try {
      statSync(join(root, name, "console.log"));
      statSync(join(root, name, "exit-code.txt"));
      return true;
    } catch {
      return false;
    }
  })
  .sort();

const rows = rowNames.map((name) => parseRow(root, name));
const baseline = rows.find((row) => row.name === "baseline_fp32");
const baselineByStep = new Map((baseline?.steps ?? []).map((step) => [step.step, step]));

for (const row of rows) {
  row.comparison = {
    finiteTrajectory: row.steps.length > 0 && row.steps.every((step) => step.loss !== null && step.gradNorm !== null),
    maxAbsoluteLossDifference: null,
    maxAbsoluteGradNormDifference: null,
    medianTokensPerSecond: median(row.steps.map((step) => step.tokensPerSecond)),
  };
  const matched = row.steps
    .map((step) => ({ candidate: step, baseline: baselineByStep.get(step.step) }))
    .filter((pair) => pair.baseline);
  const lossDiffs = matched
    .filter((pair) => pair.candidate.loss !== null && pair.baseline.loss !== null)
    .map((pair) => Math.abs(pair.candidate.loss - pair.baseline.loss));
  const gradDiffs = matched
    .filter((pair) => pair.candidate.gradNorm !== null && pair.baseline.gradNorm !== null)
    .map((pair) => Math.abs(pair.candidate.gradNorm - pair.baseline.gradNorm));
  row.comparison.maxAbsoluteLossDifference = lossDiffs.length > 0 ? Math.max(...lossDiffs) : null;
  row.comparison.maxAbsoluteGradNormDifference = gradDiffs.length > 0 ? Math.max(...gradDiffs) : null;
}

const report = {
  schema: "alpha-helios-coop-shape-bisect-summary-v1",
  createdAt: new Date().toISOString(),
  root,
  rows,
};
writeFileSync(jsonOutput, `${JSON.stringify(report, null, 2)}\n`);

const lines = [
  "# Helios cooperative-matrix whole-graph shape bisection",
  "",
  `**Created:** ${report.createdAt}  `,
  `**Rows:** ${rows.length}  `,
  "",
  "This is a numerical discriminator, not a promotion report. A candidate must retain finite loss and gradients before its speed is meaningful.",
  "",
  "| Row | Exit | Steps | Finite | Median tok/s | Max |loss - FP32| | Max |grad - FP32| | Cooperative shapes |",
  "|---|---:|---:|:---:|---:|---:|---:|---:|",
];
for (const row of rows) {
  const c = row.comparison;
  lines.push(
    `| \`${row.name}\` | ${row.exitCode} | ${row.steps.length} | ${c.finiteTrajectory ? "yes" : "**no**"} | ${c.medianTokensPerSecond ?? "n/a"} | ${c.maxAbsoluteLossDifference ?? "n/a"} | ${c.maxAbsoluteGradNormDifference ?? "n/a"} | ${row.coopShapes.length} |`,
  );
}
lines.push("", `Machine-readable report: \`${basename(jsonOutput)}\``, "");
writeFileSync(markdownOutput, lines.join("\n"));
