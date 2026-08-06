#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const GPU_OPS_MARKER = "[gpu_ops]";

function parseNamedSeries(text, source, field) {
  if (!text.trim()) return [];
  return text.split(",").map((item) => {
    const match = /^(.+):(\d+)\/([0-9]+(?:\.[0-9]+)?)us$/.exec(item.trim());
    if (!match) {
      throw new Error(`${source}: malformed ${field} entry ${JSON.stringify(item)}`);
    }
    return {
      name: match[1],
      calls: Number(match[2]),
      timeUs: Number(match[3]),
    };
  });
}

export function parseGpuOpsLine(line, source = "<input>") {
  const markerIndex = line.indexOf(GPU_OPS_MARKER);
  if (markerIndex < 0) return null;
  const payload = line.slice(markerIndex + GPU_OPS_MARKER.length).trim();
  const kindsIndex = payload.indexOf(" kinds=");
  const kernelsIndex = payload.indexOf(" kernels=");
  if (kindsIndex < 0 || kernelsIndex < 0 || kernelsIndex <= kindsIndex) {
    throw new Error(`${source}: [gpu_ops] line is missing kinds= or kernels=`);
  }

  const scalarText = payload.slice(0, kindsIndex);
  const scalars = Object.fromEntries(
    scalarText.split(/\s+/).map((entry) => {
      const separator = entry.indexOf("=");
      if (separator < 1) throw new Error(`${source}: malformed scalar ${JSON.stringify(entry)}`);
      return [entry.slice(0, separator), entry.slice(separator + 1)];
    }),
  );
  const number = (name) => {
    const value = Number(scalars[name]);
    if (!Number.isFinite(value)) throw new Error(`${source}: missing or invalid ${name}`);
    return value;
  };
  const optionalNumber = (name) => {
    if (!(name in scalars)) return null;
    const value = Number(scalars[name]);
    if (!Number.isFinite(value)) throw new Error(`${source}: invalid ${name}`);
    return value;
  };

  return {
    source,
    flushes: number("flushes"),
    waited: number("waited"),
    dgc: number("dgc"),
    operationsPerFlush: number("ops_per_flush"),
    timestamped: number("timestamped"),
    batchGpuUs: number("batch_gpu_us"),
    dispatchGpuUs: number("dispatch_gpu_us"),
    hostBuildMs: optionalNumber("host_build_ms"),
    gpuBlockingMs: optionalNumber("gpu_blocking_ms"),
    coreStepMs: optionalNumber("core_step_ms"),
    kinds: parseNamedSeries(
      payload.slice(kindsIndex + " kinds=".length, kernelsIndex),
      source,
      "kind",
    ),
    kernels: parseNamedSeries(
      payload.slice(kernelsIndex + " kernels=".length),
      source,
      "kernel",
    ),
  };
}

function averageSeries(samples, field) {
  const totals = new Map();
  for (const sample of samples) {
    for (const entry of sample[field]) {
      const total = totals.get(entry.name) ?? { name: entry.name, calls: 0, timeUs: 0 };
      total.calls += entry.calls;
      total.timeUs += entry.timeUs;
      totals.set(entry.name, total);
    }
  }
  return [...totals.values()]
    .map((entry) => ({
      name: entry.name,
      averageCalls: entry.calls / samples.length,
      averageTimeUs: entry.timeUs / samples.length,
    }))
    .sort((a, b) => b.averageTimeUs - a.averageTimeUs);
}

export function summarizeGpuOps(samples, sources = []) {
  if (samples.length === 0) throw new Error("no [gpu_ops] samples found");
  const mean = (field) => samples.reduce((sum, sample) => sum + sample[field], 0) / samples.length;
  const meanOptional = (field) => {
    const values = samples.map((sample) => sample[field]).filter((value) => value !== null);
    return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const dispatchGpuUs = mean("dispatchGpuUs");
  const withShares = (entries) => entries.map((entry) => ({
    ...entry,
    averageUsPerCall: entry.averageCalls > 0 ? entry.averageTimeUs / entry.averageCalls : null,
    dispatchShare: dispatchGpuUs > 0 ? entry.averageTimeUs / dispatchGpuUs : null,
  }));
  return {
    schema: "alpha-helios-profile-summary-v1",
    sampleCount: samples.length,
    sources,
    averages: {
      flushes: mean("flushes"),
      waited: mean("waited"),
      dgc: mean("dgc"),
      operationsPerFlush: mean("operationsPerFlush"),
      timestamped: mean("timestamped"),
      batchGpuUs: mean("batchGpuUs"),
      dispatchGpuUs,
      unaccountedBatchUs: mean("batchGpuUs") - dispatchGpuUs,
      hostBuildMs: meanOptional("hostBuildMs"),
      gpuBlockingMs: meanOptional("gpuBlockingMs"),
      coreStepMs: meanOptional("coreStepMs"),
    },
    kinds: withShares(averageSeries(samples, "kinds")),
    kernels: withShares(averageSeries(samples, "kernels")),
  };
}

function formatNumber(value, digits = 1) {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatMarkdown(summary, top) {
  const lines = [
    "# Helios GPU profile summary",
    "",
    `Samples: ${summary.sampleCount}`,
    `Average measured dispatch time: ${formatNumber(summary.averages.dispatchGpuUs / 1000)} ms`,
    `Average timestamped batch time: ${formatNumber(summary.averages.batchGpuUs / 1000)} ms`,
  ];
  if (summary.averages.hostBuildMs !== null) {
    lines.push(
      `Average host build time: ${formatNumber(summary.averages.hostBuildMs)} ms`,
      `Average GPU-blocking wall time: ${formatNumber(summary.averages.gpuBlockingMs)} ms`,
      `Average pre-metrics core step: ${formatNumber(summary.averages.coreStepMs)} ms`,
    );
  }
  lines.push(
    "",
    "## Operation kinds",
    "",
    "| Rank | Kind | Avg calls | Avg total | Mean/call | Dispatch share |",
    "|---:|---|---:|---:|---:|---:|",
  );
  summary.kinds.forEach((entry, index) => {
    lines.push(
      `| ${index + 1} | \`${entry.name}\` | ${formatNumber(entry.averageCalls)} | ` +
        `${formatNumber(entry.averageTimeUs / 1000)} ms | ` +
        `${formatNumber(entry.averageUsPerCall / 1000, 3)} ms | ` +
        `${formatNumber(entry.dispatchShare * 100)}% |`,
    );
  });
  lines.push(
    "",
    `## Top ${Math.min(top, summary.kernels.length)} kernels`,
    "",
    "| Rank | Kernel | Avg calls | Avg total | Mean/call | Dispatch share |",
    "|---:|---|---:|---:|---:|---:|",
  );
  summary.kernels.slice(0, top).forEach((entry, index) => {
    lines.push(
      `| ${index + 1} | \`${entry.name}\` | ${formatNumber(entry.averageCalls)} | ` +
        `${formatNumber(entry.averageTimeUs / 1000)} ms | ` +
        `${formatNumber(entry.averageUsPerCall / 1000, 3)} ms | ` +
        `${formatNumber(entry.dispatchShare * 100)}% |`,
    );
  });
  lines.push("", "## Sources", "");
  for (const source of summary.sources) {
    lines.push(`- \`${source.path}\` — SHA-256 \`${source.sha256}\`, ${source.samples} sample(s)`);
  }
  return `${lines.join("\n")}\n`;
}

function usage() {
  return [
    "Usage: node scripts/summarize_helios_profile.mjs [--format markdown|json] [--top N] [--skip-first N] [--output FILE] LOG...",
    "",
    "Reads every [gpu_ops] line from each log and averages the dynamic kind and kernel series.",
    "It does not invent higher-level categories; the profiler's recorded operation kinds remain authoritative.",
  ].join("\n");
}

function run(argv) {
  let format = "markdown";
  let top = 20;
  let skipFirst = 0;
  let outputPath = null;
  const paths = [];
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      process.stdout.write(`${usage()}\n`);
      return;
    }
    if (argument === "--format") {
      format = argv[++index];
      continue;
    }
    if (argument === "--top") {
      top = Number(argv[++index]);
      continue;
    }
    if (argument === "--skip-first") {
      skipFirst = Number(argv[++index]);
      continue;
    }
    if (argument === "--output") {
      outputPath = resolve(argv[++index]);
      continue;
    }
    paths.push(resolve(argument));
  }
  if (!['markdown', 'json'].includes(format)) throw new Error(`unsupported format ${format}`);
  if (!Number.isInteger(top) || top < 1) throw new Error(`invalid --top value ${top}`);
  if (!Number.isInteger(skipFirst) || skipFirst < 0) throw new Error(`invalid --skip-first value ${skipFirst}`);
  if (paths.length === 0) throw new Error(usage());

  const samples = [];
  const sources = [];
  for (const path of paths) {
    const bytes = readFileSync(path);
    const text = bytes.toString("utf8");
    let count = 0;
    const pathSamples = [];
    text.split(/\r?\n/).forEach((line, lineIndex) => {
      if (!line.includes(GPU_OPS_MARKER)) return;
      pathSamples.push(parseGpuOpsLine(line, `${path}:${lineIndex + 1}`));
      count++;
    });
    samples.push(...pathSamples.slice(skipFirst));
    sources.push({
      path,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      samples: Math.max(0, count - skipFirst),
      skipped: Math.min(count, skipFirst),
    });
  }
  const summary = summarizeGpuOps(samples, sources);
  const rendered =
    format === "json"
      ? `${JSON.stringify(summary, null, 2)}\n`
      : formatMarkdown(summary, top);
  if (outputPath) writeFileSync(outputPath, rendered);
  else process.stdout.write(rendered);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    run(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
