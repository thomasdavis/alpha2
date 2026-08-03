#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function usage() {
  console.error("usage: summarize_helios_allocator_factorial.mjs <experiment-dir> [--warmup N] [--baseline MODE] [--format json|markdown]");
  process.exit(2);
}

const args = process.argv.slice(2);
if (args.length === 0) usage();
const root = path.resolve(args.shift());
let warmup = 3;
let format = "markdown";
let baselineMode = "exact_individual";
while (args.length > 0) {
  const arg = args.shift();
  if (arg === "--warmup") warmup = Number.parseInt(args.shift() ?? "", 10);
  else if (arg === "--baseline") baselineMode = args.shift() ?? "";
  else if (arg === "--format") format = args.shift() ?? "";
  else usage();
}
if (!Number.isInteger(warmup) || warmup < 0 || baselineMode.length === 0 || !["json", "markdown"].includes(format)) usage();

function quantile(values, q) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function mean(values) {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parseFlowTotals(consoleText) {
  const totals = { new: 0, dest: 0, oHit: 0, oMiss: 0, oRet: 0, oOvf: 0, bHit: 0, gHit: 0, gUp: 0 };
  const pattern = /\u0394flow: new=(\d+) dest=(\d+) oHit=(\d+) oMiss=(\d+) oRet=(\d+) oOvf=(\d+) bHit=(\d+) gHit=(\d+) gUp=(\d+)/g;
  for (const match of consoleText.matchAll(pattern)) {
    Object.keys(totals).forEach((key, index) => { totals[key] += Number(match[index + 1]); });
  }
  return totals;
}

function parseFinalAllocatorState(consoleText) {
  const lines = consoleText.split("\n").filter((line) => line.includes("[gpu_mem]"));
  const line = lines.at(-1) ?? "";
  const allocs = line.match(/allocs: (\d+) live \((\d+) total, (\d+)MB\)/);
  const native = line.match(/vkMem: (\d+) tracked \((\d+) individual, (\d+) slab buffers, temp=(\d+) slabs\/(\d+)MB live\/(\d+)MB cap, reuse=(\d+), fallback=(\d+), freeOvf=(\d+)\)/);
  const pool = line.match(/outPool: (\d+)\/(\d+)cls \(([0-9.]+)MB\)/);
  return {
    liveAllocs: allocs ? Number(allocs[1]) : null,
    totalAllocs: allocs ? Number(allocs[2]) : null,
    totalAllocMB: allocs ? Number(allocs[3]) : null,
    trackedVkMemoryAllocations: native ? Number(native[1]) : null,
    individualBuffers: native ? Number(native[2]) : null,
    slabBuffers: native ? Number(native[3]) : null,
    tempSlabCount: native ? Number(native[4]) : null,
    tempSlabLiveMB: native ? Number(native[5]) : null,
    tempSlabCapacityMB: native ? Number(native[6]) : null,
    slabFreeRangeReuses: native ? Number(native[7]) : null,
    slabFallbacks: native ? Number(native[8]) : null,
    slabFreeRangeOverflows: native ? Number(native[9]) : null,
    outputPoolEntries: pool ? Number(pool[1]) : null,
    outputPoolSizeClasses: pool ? Number(pool[2]) : null,
    outputPoolMB: pool ? Number(pool[3]) : null,
  };
}

const modesDir = path.join(root, "modes");
const modeNames = fs.existsSync(modesDir)
  ? fs.readdirSync(modesDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort()
  : [];
const rows = [];
for (const mode of modeNames) {
  const modeDir = path.join(modesDir, mode);
  const exitCodePath = path.join(modeDir, "exit-code.txt");
  const exitCode = fs.existsSync(exitCodePath) ? Number.parseInt(fs.readFileSync(exitCodePath, "utf8").trim(), 10) : null;
  const metricsPath = path.join(modeDir, "run", "metrics.jsonl");
  const consolePath = path.join(modeDir, "console.log");
  const configPath = path.join(modeDir, "MODE.json");
  const config = fs.existsSync(configPath) ? JSON.parse(fs.readFileSync(configPath, "utf8")) : {};
  if (exitCode !== 0 || !fs.existsSync(metricsPath)) {
    rows.push({ mode, config, exitCode, status: "failed" });
    continue;
  }
  const metrics = fs.readFileSync(metricsPath, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
  const measured = metrics.filter((row) => row.step > warmup);
  const consoleText = fs.readFileSync(consolePath, "utf8");
  const tps = measured.map((row) => row.tokens_per_sec);
  const elapsed = measured.map((row) => row.elapsed_ms);
  const host = measured.map((row) => row.timing_host_build_ms).filter(Number.isFinite);
  const gpu = measured.map((row) => row.timing_gpu_blocking_ms).filter(Number.isFinite);
  const core = measured.map((row) => row.timing_core_step_ms).filter(Number.isFinite);
  rows.push({
    mode,
    config,
    exitCode,
    status: "pass",
    totalSamples: metrics.length,
    warmupExcluded: warmup,
    measuredSamples: measured.length,
    tokensPerSecond: { mean: mean(tps), median: quantile(tps, 0.5), p10: quantile(tps, 0.1), p90: quantile(tps, 0.9) },
    elapsedMs: { mean: mean(elapsed), median: quantile(elapsed, 0.5), p10: quantile(elapsed, 0.1), p90: quantile(elapsed, 0.9) },
    timing: {
      hostBuildMsMean: mean(host),
      gpuBlockingMsMean: mean(gpu),
      coreStepMsMean: mean(core),
      hostShare: mean(host) != null && mean(core) ? mean(host) / mean(core) : null,
    },
    flowTotals: parseFlowTotals(consoleText),
    finalAllocatorState: parseFinalAllocatorState(consoleText),
  });
}

const baseline = rows.find((row) => row.mode === baselineMode && row.status === "pass");
for (const row of rows) {
  row.speedupVsBaseline = row.status === "pass" && baseline
    ? row.tokensPerSecond.median / baseline.tokensPerSecond.median
    : null;
}
const result = {
  schemaVersion: 1,
  experimentRoot: root,
  warmupExcluded: warmup,
  baselineMode,
  rows: rows.sort((a, b) => {
    if (a.status !== b.status) return a.status === "pass" ? -1 : 1;
    return (b.tokensPerSecond?.median ?? -1) - (a.tokensPerSecond?.median ?? -1);
  }),
};

if (format === "json") {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exit(0);
}

const f = (value, digits = 1) => value == null ? "n/a" : Number(value).toFixed(digits);
console.log("# Helios allocator factorial\n");
console.log(`Warm-up steps excluded per mode: ${warmup}\n`);
console.log("| Mode | Median tok/s | Mean tok/s | p10–p90 tok/s | Speedup | Host ms | GPU block ms | New buffers | Destroys | Output hits/misses | Final Vk allocations |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const row of result.rows) {
  if (row.status !== "pass") {
    console.log(`| ${row.mode} | failed (${row.exitCode ?? "no exit"}) | | | | | | | | | |`);
    continue;
  }
  const a = row.finalAllocatorState;
  console.log(`| ${row.mode} | ${f(row.tokensPerSecond.median, 0)} | ${f(row.tokensPerSecond.mean, 0)} | ${f(row.tokensPerSecond.p10, 0)}–${f(row.tokensPerSecond.p90, 0)} | ${f(row.speedupVsBaseline, 3)}x | ${f(row.timing.hostBuildMsMean)} | ${f(row.timing.gpuBlockingMsMean)} | ${row.flowTotals.new} | ${row.flowTotals.dest} | ${row.flowTotals.oHit}/${row.flowTotals.oMiss} | ${a.trackedVkMemoryAllocations ?? "n/a"} (${a.individualBuffers ?? "?"} individual + ${a.slabBuffers ?? "?"} slab-backed buffers) |`);
}
console.log("\n`flush` contains synchronous GPU wait time; host and GPU-blocking means come from the direct wall-clock partition in Helios. Flow totals include setup and all measured steps because the backend counters are cumulative within each fresh process.");
