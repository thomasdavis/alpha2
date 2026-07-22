#!/usr/bin/env npx tsx
/** Prove the flagship-shape G2 soak met its throughput and boundedness contract. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import * as path from "node:path";

interface Metric {
  step: number;
  loss: number;
  gradNorm: number;
  tokens_per_sec: number;
  host_rss_mb: number;
  gpu_live_allocs?: number;
  gpu_vk_memory_allocations?: number;
  gpu_temp_slab_count?: number;
  gpu_allocator_free_range_overflows?: number;
}

interface Config {
  modelConfig: {
    vocabSize: number;
    blockSize: number;
    nLayer: number;
    nEmbd: number;
    nHead: number;
    dropout: number;
    ffnActivation: string;
    ffnDim: number;
    normType: string;
    posEnc: string;
    ropeTheta: number;
    tieEmbeddings: boolean;
  };
  trainConfig: {
    iters: number;
    batchSize: number;
    gradAccumSteps: number;
    backend: string;
  };
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    result[arg.slice(2)] = value;
  }
  return result;
}

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) throw new Error("cannot summarize an empty series");
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.floor((ordered.length - 1) * fraction)];
}

function range(values: number[]): number {
  return Math.max(...values) - Math.min(...values);
}

function slopePerThousand(values: number[]): number {
  const n = values.length;
  const meanX = (n - 1) / 2;
  const meanY = values.reduce((sum, value) => sum + value, 0) / n;
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < n; index++) {
    numerator += (index - meanX) * (values[index] - meanY);
    denominator += (index - meanX) ** 2;
  }
  return (numerator / denominator) * 1000;
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const cli = parseArgs();
  const runDir = cli.run;
  const out = cli.out;
  const sourceCommit = cli.sourceCommit;
  const costPerHour = Number(cli.costPerHour);
  const expectedSteps = Number(cli.expectedSteps ?? "5400");
  if (!runDir || !out || !sourceCommit || !Number.isFinite(costPerHour)) {
    throw new Error("required: --run, --out, --sourceCommit, and --costPerHour");
  }

  const [configText, metricsText, monitorText] = await Promise.all([
    readFile(path.join(runDir, "config.json"), "utf8"),
    readFile(path.join(runDir, "metrics.jsonl"), "utf8"),
    readFile(path.join(runDir, "system-monitor.log"), "utf8"),
  ]);
  const config = JSON.parse(configText) as Config;
  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (metrics.length !== expectedSteps) throw new Error(`metric rows ${metrics.length} != ${expectedSteps}`);
  if (config.trainConfig.iters !== expectedSteps) throw new Error(`configured iters ${config.trainConfig.iters} != ${expectedSteps}`);

  const requiredModel = {
    vocabSize: 12288,
    blockSize: 1024,
    nLayer: 16,
    nEmbd: 512,
    nHead: 8,
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 1408,
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10000,
    tieEmbeddings: true,
  };
  for (const [key, expected] of Object.entries(requiredModel)) {
    const actual = config.modelConfig[key as keyof Config["modelConfig"]];
    if (actual !== expected) throw new Error(`modelConfig.${key} ${String(actual)} != ${String(expected)}`);
  }
  if (config.trainConfig.backend !== "helios") throw new Error(`backend ${config.trainConfig.backend} != helios`);
  if (costPerHour > 0.35) throw new Error(`GPU cost $${costPerHour}/hr exceeds G2 ceiling`);

  for (let index = 0; index < metrics.length; index++) {
    const metric = metrics[index];
    if (metric.step !== index + 1) throw new Error(`expected step ${index + 1}, found ${metric.step}`);
    for (const field of [
      "loss",
      "gradNorm",
      "tokens_per_sec",
      "host_rss_mb",
    ] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`non-finite ${field} at step ${metric.step}`);
    }
  }

  const telemetry = metrics.filter((metric): metric is Metric & Required<Pick<Metric,
    "gpu_live_allocs" | "gpu_vk_memory_allocations" | "gpu_temp_slab_count" | "gpu_allocator_free_range_overflows"
  >> =>
    Number.isFinite(metric.gpu_live_allocs) &&
    Number.isFinite(metric.gpu_vk_memory_allocations) &&
    Number.isFinite(metric.gpu_temp_slab_count) &&
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  const minimumTelemetrySamples = Math.floor(expectedSteps / 50);
  if (telemetry.length < minimumTelemetrySamples) {
    throw new Error(`allocator telemetry samples ${telemetry.length} < ${minimumTelemetrySamples}`);
  }
  const telemetryGaps = telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step);
  const telemetryMaxGap = Math.max(0, ...telemetryGaps);
  if (telemetryMaxGap > 50) throw new Error(`allocator telemetry gap ${telemetryMaxGap} steps exceeds 50`);
  if (telemetry.at(-1)?.step !== expectedSteps) {
    throw new Error(`final allocator telemetry step ${String(telemetry.at(-1)?.step)} != ${expectedSteps}`);
  }

  const timestamps = monitorText.match(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/gm) ?? [];
  if (timestamps.length < 2) throw new Error("system monitor has fewer than two timestamp samples");
  const firstMonitorMs = Date.parse(timestamps[0]);
  const lastMonitorMs = Date.parse(timestamps.at(-1)!);
  const monitoredDurationSeconds = (lastMonitorMs - firstMonitorMs) / 1000;
  if (monitoredDurationSeconds < 6 * 60 * 60) {
    throw new Error(`monitored duration ${monitoredDurationSeconds}s is below literal six-hour gate`);
  }

  const steady = metrics.slice(100);
  const throughput = steady.map((metric) => metric.tokens_per_sec);
  const rss = steady.map((metric) => metric.host_rss_mb);
  const steadyTelemetry = telemetry.filter((metric) => metric.step > 100);
  const liveAllocations = steadyTelemetry.map((metric) => metric.gpu_live_allocs);
  const vkAllocations = steadyTelemetry.map((metric) => metric.gpu_vk_memory_allocations);
  const tempSlabs = steadyTelemetry.map((metric) => metric.gpu_temp_slab_count);
  const overflowMax = Math.max(...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  const checks = {
    throughput_median_at_least_3000: percentile(throughput, 0.5) >= 3000,
    throughput_p10_at_least_3000: percentile(throughput, 0.1) >= 3000,
    rss_range_at_most_128_mb: range(rss) <= 128,
    rss_slope_abs_at_most_8_mb_per_1000_steps: Math.abs(slopePerThousand(rss)) <= 8,
    live_allocation_range_at_most_512: range(liveAllocations) <= 512,
    vk_memory_allocation_range_at_most_128: range(vkAllocations) <= 128,
    temp_slab_count_range_at_most_2: range(tempSlabs) <= 2,
    allocator_free_range_overflow_zero: overflowMax === 0,
    literal_six_hours: monitoredDurationSeconds >= 6 * 60 * 60,
  };

  const checkpointPath = path.join(runDir, `checkpoint-${expectedSteps}.json`);
  const checkpointStat = await stat(checkpointPath);
  const minimumCheckpointBytes = 650 * 1024 * 1024;
  const maximumCheckpointBytes = 750 * 1024 * 1024;
  if (checkpointStat.size < minimumCheckpointBytes || checkpointStat.size > maximumCheckpointBytes) {
    throw new Error(
      `checkpoint size ${checkpointStat.size} is outside full flagship+AdamW envelope ` +
      `[${minimumCheckpointBytes}, ${maximumCheckpointBytes}]`,
    );
  }
  const pass = Object.values(checks).every(Boolean);
  const report = {
    schema: "alpha-g2-soak-analysis-v1",
    result: pass ? "PASS" : "FAIL",
    source_commit: sourceCommit,
    gpu: { model: "NVIDIA GeForce RTX 3090", cost_per_hour_usd: costPerHour },
    run: {
      dir: runDir,
      expected_steps: expectedSteps,
      metric_rows: metrics.length,
      tokens: expectedSteps * config.trainConfig.batchSize * config.trainConfig.gradAccumSteps * config.modelConfig.blockSize,
      metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
      monitor_sha256: createHash("sha256").update(monitorText).digest("hex"),
      monitor_first_utc: timestamps[0],
      monitor_last_utc: timestamps.at(-1),
      monitored_duration_seconds: monitoredDurationSeconds,
      allocator_telemetry_samples: telemetry.length,
      allocator_telemetry_max_step_gap: telemetryMaxGap,
      checkpoint: {
        path: checkpointPath,
        bytes: checkpointStat.size,
        sha256: await sha256File(checkpointPath),
      },
    },
    steady_state_after_step: 100,
    metrics: {
      tokens_per_sec: { p10: percentile(throughput, 0.1), median: percentile(throughput, 0.5) },
      host_rss_mb: { min: Math.min(...rss), max: Math.max(...rss), range: range(rss), slope_per_1000_steps: slopePerThousand(rss) },
      gpu_live_allocs: { min: Math.min(...liveAllocations), max: Math.max(...liveAllocations), range: range(liveAllocations) },
      gpu_vk_memory_allocations: { min: Math.min(...vkAllocations), max: Math.max(...vkAllocations), range: range(vkAllocations) },
      gpu_temp_slab_count: { min: Math.min(...tempSlabs), max: Math.max(...tempSlabs), range: range(tempSlabs) },
      allocator_free_range_overflow_max: overflowMax,
      loss: { first_100_mean: metrics.slice(0, 100).reduce((sum, metric) => sum + metric.loss, 0) / 100, last_100_mean: metrics.slice(-100).reduce((sum, metric) => sum + metric.loss, 0) / 100 },
    },
    checks,
  };
  await writeFile(out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
