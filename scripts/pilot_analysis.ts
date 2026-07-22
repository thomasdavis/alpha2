import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import * as path from "node:path";

export interface PilotContract {
  variant: "llama" | "gpt2";
  expected_params: number;
  expected_steps: number;
  expected_tokens: number;
  minimum_train_tokens: number;
  learning_rate: number;
  learning_rate_min: number;
  source_commit: string;
  data: { path: string; sha256: string };
  tokenizer: { path: string; sha256: string };
}

interface PilotMetric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  gpu_allocator_free_range_overflows?: number;
}

interface PilotModelConfig {
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
  tieEmbeddings: boolean;
}

export interface PilotRunSummary {
  dir: string;
  contract: PilotContract;
  total_params: number;
  metrics_sha256: string;
  rows: number;
  tokens: number;
  median_tokens_per_sec_after_warmup: number;
  final_train_loss: number;
  last_100_train_loss_mean: number;
  eval: { step: number; val_loss: number }[];
  allocator_telemetry_samples: number;
  allocator_telemetry_max_step_gap: number;
  allocator_overflow_max: number;
}

export function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

export async function summarizePilot(
  dir: string,
  expectedVariant: PilotContract["variant"],
): Promise<PilotRunSummary> {
  const [contractText, configText, metricsText] = await Promise.all([
    readFile(path.join(dir, "pilot-contract.json"), "utf8"),
    readFile(path.join(dir, "config.json"), "utf8"),
    readFile(path.join(dir, "metrics.jsonl"), "utf8"),
  ]);
  const contract = JSON.parse(contractText) as PilotContract;
  const config = JSON.parse(configText) as {
    totalParams?: number;
    dataStats?: { trainTokens?: number; valTokens?: number };
    trainConfig: { batchSize: number; gradAccumSteps: number; iters: number };
    modelConfig: PilotModelConfig;
  };
  if (contract.variant !== expectedVariant) throw new Error(`${dir}: contract variant ${contract.variant} != ${expectedVariant}`);
  const rows = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as PilotMetric);
  if (rows.length !== contract.expected_steps) throw new Error(`${dir}: metric rows ${rows.length} != ${contract.expected_steps}`);
  for (let index = 0; index < rows.length; index++) {
    const metric = rows[index];
    if (metric.step !== index + 1) throw new Error(`${dir}: expected step ${index + 1}, found ${metric.step}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`${dir}: non-finite ${field} at step ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) {
      throw new Error(`${dir}: non-finite valLoss at step ${metric.step}`);
    }
  }
  const totalParams = config.totalParams;
  if (totalParams !== contract.expected_params) {
    throw new Error(`${dir}: params ${String(totalParams)} != contract ${contract.expected_params}`);
  }
  const expectedModel: PilotModelConfig = {
    vocabSize: 12288,
    blockSize: 1024,
    nLayer: expectedVariant === "llama" ? 16 : 14,
    nEmbd: 512,
    nHead: 8,
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 1408,
    normType: expectedVariant === "llama" ? "rmsnorm" : "layernorm",
    posEnc: expectedVariant === "llama" ? "rope" : "learned",
    tieEmbeddings: expectedVariant === "llama",
  };
  for (const key of Object.keys(expectedModel) as (keyof PilotModelConfig)[]) {
    if (config.modelConfig[key] !== expectedModel[key]) {
      throw new Error(`${dir}: modelConfig.${key} ${String(config.modelConfig[key])} != ${String(expectedModel[key])}`);
    }
  }
  const tokens = config.trainConfig.iters * config.trainConfig.batchSize * config.trainConfig.gradAccumSteps * config.modelConfig.blockSize;
  if (tokens !== contract.expected_tokens) throw new Error(`${dir}: tokens ${tokens} != contract ${contract.expected_tokens}`);
  if ((config.dataStats?.trainTokens ?? 0) < contract.minimum_train_tokens) {
    throw new Error(`${dir}: only ${String(config.dataStats?.trainTokens)} train tokens; contract requires ${contract.minimum_train_tokens}`);
  }
  const evalRows = rows.filter((metric): metric is PilotMetric & { valLoss: number } => metric.valLoss !== undefined);
  if (evalRows.length < 3) throw new Error(`${dir}: only ${evalRows.length} validation points`);
  const allocatorTelemetry = rows.filter((metric): metric is PilotMetric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  const minimumAllocatorSamples = Math.floor(contract.expected_steps / 100);
  if (allocatorTelemetry.length < minimumAllocatorSamples) {
    throw new Error(`${dir}: allocator telemetry samples ${allocatorTelemetry.length} < ${minimumAllocatorSamples}`);
  }
  const allocatorTelemetryGaps = allocatorTelemetry.slice(1)
    .map((metric, index) => metric.step - allocatorTelemetry[index].step);
  const allocatorTelemetryMaxStepGap = Math.max(0, ...allocatorTelemetryGaps);
  if (allocatorTelemetryMaxStepGap > 100) {
    throw new Error(`${dir}: allocator telemetry gap ${allocatorTelemetryMaxStepGap} exceeds 100 steps`);
  }
  if (allocatorTelemetry.at(-1)?.step !== contract.expected_steps) {
    throw new Error(`${dir}: final allocator telemetry step ${String(allocatorTelemetry.at(-1)?.step)} != ${contract.expected_steps}`);
  }
  return {
    dir,
    contract,
    total_params: totalParams,
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    rows: rows.length,
    tokens,
    median_tokens_per_sec_after_warmup: median(rows.slice(100).map((metric) => metric.tokens_per_sec)),
    final_train_loss: rows.at(-1)!.loss,
    last_100_train_loss_mean: mean(rows.slice(-100).map((metric) => metric.loss)),
    eval: evalRows.map((metric) => ({ step: metric.step, val_loss: metric.valLoss })),
    allocator_telemetry_samples: allocatorTelemetry.length,
    allocator_telemetry_max_step_gap: allocatorTelemetryMaxStepGap,
    allocator_overflow_max: Math.max(0, ...allocatorTelemetry.map((metric) => metric.gpu_allocator_free_range_overflows)),
  };
}
