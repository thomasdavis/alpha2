#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

function usage() {
  console.error("usage: summarize_node_cpu_profile.mjs [--format json|markdown] <cpu.cpuprofile>");
  process.exit(2);
}

const args = process.argv.slice(2);
let format = "markdown";
if (args[0] === "--format") {
  format = args[1] ?? "";
  args.splice(0, 2);
}
if (!new Set(["json", "markdown"]).has(format) || args.length !== 1) usage();

const profilePath = path.resolve(args[0]);
const bytes = await readFile(profilePath);
const profile = JSON.parse(bytes.toString("utf8"));
if (!Array.isArray(profile.nodes) || !Array.isArray(profile.samples) || !Array.isArray(profile.timeDeltas)) {
  throw new Error("invalid V8 CPU profile: nodes, samples, and timeDeltas are required");
}
if (profile.samples.length !== profile.timeDeltas.length) {
  throw new Error("invalid V8 CPU profile: samples/timeDeltas length mismatch");
}

const nodes = new Map(profile.nodes.map((node) => [node.id, node]));
const rows = new Map();
let totalSampledUs = 0;
for (let i = 0; i < profile.samples.length; i++) {
  const deltaUs = Number(profile.timeDeltas[i]);
  if (!Number.isFinite(deltaUs) || deltaUs < 0) continue;
  totalSampledUs += deltaUs;
  const node = nodes.get(profile.samples[i]);
  if (!node?.callFrame) continue;
  const frame = node.callFrame;
  const functionName = frame.functionName || "(anonymous)";
  const url = frame.url || "(native)";
  const line = Number.isInteger(frame.lineNumber) ? frame.lineNumber + 1 : null;
  const column = Number.isInteger(frame.columnNumber) ? frame.columnNumber + 1 : null;
  const key = `${functionName}\u0000${url}\u0000${line ?? ""}\u0000${column ?? ""}`;
  const row = rows.get(key) ?? { functionName, url, line, column, samples: 0, selfUs: 0 };
  row.samples++;
  row.selfUs += deltaUs;
  rows.set(key, row);
}

const entries = [...rows.values()]
  .map((row) => ({
    ...row,
    selfMs: row.selfUs / 1000,
    share: totalSampledUs > 0 ? row.selfUs / totalSampledUs : 0,
  }))
  .sort((a, b) => b.selfUs - a.selfUs || b.samples - a.samples || a.functionName.localeCompare(b.functionName));

const summary = {
  schema: "alpha-node-cpu-profile-summary-v1",
  source: {
    path: profilePath,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  },
  sampleCount: profile.samples.length,
  totalSampledUs,
  entries,
};

if (format === "json") {
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
} else {
  const lines = [
    "# Node CPU profile self-time",
    "",
    `**Samples:** ${summary.sampleCount}`,
    `**Sampled wall:** ${(totalSampledUs / 1e6).toFixed(3)} s`,
    "",
    "Self-time includes native waits and idle samples. It is not inclusive call-tree time.",
    "",
    "| Rank | Function | Location | Self ms | Share | Samples |",
    "|---:|---|---|---:|---:|---:|",
  ];
  entries.slice(0, 50).forEach((row, index) => {
    const clean = (value) => String(value).replaceAll("|", "\\|");
    const location = row.line == null ? row.url : `${row.url}:${row.line}`;
    lines.push(
      `| ${index + 1} | \`${clean(row.functionName)}\` | \`${clean(location)}\` | ` +
      `${row.selfMs.toFixed(3)} | ${(100 * row.share).toFixed(2)}% | ${row.samples} |`,
    );
  });
  process.stdout.write(`${lines.join("\n")}\n`);
}
