#!/usr/bin/env npx tsx
/** Validate and select the contracted three-way assistant-only SFT LR pilot. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";

interface SftPilotContract {
  schema: string;
  job: string;
  expected_params: number;
  expected_steps: number;
  batch_size: number;
  block_size: number;
  grad_accum_steps: number;
  padded_train_tokens: number;
  train_conversations: number;
  validation_conversations: number;
  validation_fraction: number;
  learning_rate: number;
  learning_rate_min: number;
  warmup_iters: number;
  source_commit: string;
  inputs: Record<string, { path: string; sha256: string }>;
}

interface Metric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  gpu_allocator_free_range_overflows?: number;
}

function parseArgs(): Record<string, string> {
  const values: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    values[arg.slice(2)] = value;
  }
  return values;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function summarizeRun(dir: string, expectedLearningRate: number) {
  const [contractText, configText, metricsText] = await Promise.all([
    readFile(path.join(dir, "sft-pilot-contract.json"), "utf8"),
    readFile(path.join(dir, "config.json"), "utf8"),
    readFile(path.join(dir, "metrics.jsonl"), "utf8"),
  ]);
  const contract = JSON.parse(contractText) as SftPilotContract;
  const config = JSON.parse(configText) as any;
  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  const expectedContract = {
    schema: "alpha-sft-lr-pilot-contract-v1",
    job: "lr-pilot",
    expected_params: 57_688_576,
    expected_steps: 2_000,
    batch_size: 16,
    block_size: 1_024,
    grad_accum_steps: 1,
    padded_train_tokens: 32_768_000,
    train_conversations: 485_150,
    validation_conversations: 26_278,
    validation_fraction: 0.05,
    learning_rate: expectedLearningRate,
    learning_rate_min: expectedLearningRate / 10,
    warmup_iters: 200,
  };
  for (const [key, value] of Object.entries(expectedContract)) {
    if ((contract as any)[key] !== value) throw new Error(`${dir}: contract ${key} ${(contract as any)[key]} != ${value}`);
  }
  if (!/^[0-9a-f]{40}$/.test(contract.source_commit)) throw new Error(`${dir}: invalid source commit`);
  for (const [name, input] of Object.entries(contract.inputs ?? {})) {
    if (!input?.path || !/^[0-9a-f]{64}$/.test(input.sha256)) throw new Error(`${dir}: invalid ${name} input contract`);
  }
  for (const required of ["corpus", "manifest", "length_audit", "mask_audit", "tokenizer", "base_checkpoint"]) {
    if (!contract.inputs?.[required]) throw new Error(`${dir}: missing ${required} input contract`);
  }

  const expectedModel = {
    vocabSize: 12_288,
    blockSize: 1_024,
    nLayer: 16,
    nEmbd: 512,
    nHead: 8,
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 1_408,
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10_000,
    tieEmbeddings: true,
  };
  if (!isDeepStrictEqual(config.modelConfig, expectedModel)) throw new Error(`${dir}: model config drift`);
  if (config.totalParams !== contract.expected_params) throw new Error(`${dir}: parameter count drift`);
  if (config.domain !== "alpha_llama") throw new Error(`${dir}: domain ${String(config.domain)} != alpha_llama`);
  if (config.initCheckpointPath !== contract.inputs.base_checkpoint.path) throw new Error(`${dir}: base initialization path drift`);
  const trainExpected: Record<string, unknown> = {
    iters: 2_000,
    batchSize: 16,
    lr: expectedLearningRate,
    lrMin: expectedLearningRate / 10,
    warmupIters: 200,
    beta1: 0.9,
    beta2: 0.95,
    eps: 1e-8,
    weightDecay: 0.1,
    gradClip: 1,
    evalInterval: 250,
    checkpointInterval: 1_000,
    evalIters: 5,
    seed: 42,
    backend: "helios",
    gradAccumSteps: 1,
    packed: false,
  };
  for (const [key, value] of Object.entries(trainExpected)) {
    if (config.trainConfig?.[key] !== value) throw new Error(`${dir}: trainConfig.${key} ${String(config.trainConfig?.[key])} != ${String(value)}`);
  }
  if (config.dataStats?.mode !== "sft") throw new Error(`${dir}: missing SFT data stats`);
  if (config.dataStats.trainConversations !== 485_150 || config.dataStats.valConversations !== 26_278) {
    throw new Error(`${dir}: SFT split counts drifted`);
  }
  if (!(config.dataStats.trainTokens > 0) || !(config.dataStats.valTokens > 0)) throw new Error(`${dir}: empty SFT token stats`);

  if (metrics.length !== contract.expected_steps) throw new Error(`${dir}: metric rows ${metrics.length} != ${contract.expected_steps}`);
  for (const [index, metric] of metrics.entries()) {
    if (metric.step !== index + 1) throw new Error(`${dir}: expected metric step ${index + 1}, found ${metric.step}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`${dir}: non-finite ${field} at step ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) {
      throw new Error(`${dir}: non-finite valLoss at step ${metric.step}`);
    }
  }
  const evalRows = metrics.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  const expectedEvalSteps = [250, 500, 750, 1_000, 1_250, 1_500, 1_750, 2_000];
  if (!isDeepStrictEqual(evalRows.map((metric) => metric.step), expectedEvalSteps)) {
    throw new Error(`${dir}: validation cadence drifted`);
  }
  const telemetry = metrics.filter((metric): metric is Metric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  if (telemetry.length < 20) throw new Error(`${dir}: allocator telemetry samples ${telemetry.length} < 20`);
  const maxGap = Math.max(0, ...telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step));
  if (maxGap > 100 || telemetry.at(-1)?.step !== 2_000) throw new Error(`${dir}: allocator telemetry cadence is incomplete`);
  const overflowMax = Math.max(0, ...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  if (overflowMax !== 0) throw new Error(`${dir}: allocator free-range overflow ${overflowMax}`);

  const checkpointPath = path.join(dir, "checkpoint-2000.json");
  const checkpointStat = await stat(checkpointPath);
  if (checkpointStat.size < 650 * 1024 * 1024 || checkpointStat.size > 750 * 1024 * 1024) {
    throw new Error(`${dir}: final checkpoint size ${checkpointStat.size} is outside the full-state envelope`);
  }
  return {
    dir,
    contract,
    total_params: config.totalParams,
    train_tokens: config.dataStats.trainTokens,
    validation_tokens: config.dataStats.valTokens,
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    final_checkpoint: {
      path: checkpointPath,
      bytes: checkpointStat.size,
      sha256: await sha256File(checkpointPath),
    },
    final_train_loss: metrics.at(-1)!.loss,
    last_100_train_loss_mean: mean(metrics.slice(-100).map((metric) => metric.loss)),
    median_tokens_per_sec_after_warmup: median(metrics.slice(200).map((metric) => metric.tokens_per_sec)),
    validation: evalRows.map((metric) => ({ step: metric.step, loss: metric.valLoss })),
    final_three_validation_mean: mean(evalRows.slice(-3).map((metric) => metric.valLoss)),
    allocator_telemetry_samples: telemetry.length,
    allocator_telemetry_max_gap: maxGap,
    allocator_overflow_max: overflowMax,
  };
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.lr1e4 || !cli.lr3e4 || !cli.lr1e3 || !cli.out) {
    throw new Error("required: --lr1e4, --lr3e4, --lr1e3, and --out");
  }
  const runs = await Promise.all([
    summarizeRun(cli.lr1e4, 1e-4),
    summarizeRun(cli.lr3e4, 3e-4),
    summarizeRun(cli.lr1e3, 1e-3),
  ]);
  const reference = runs[0];
  for (const run of runs.slice(1)) {
    if (run.contract.source_commit !== reference.contract.source_commit) throw new Error("SFT pilot source commits differ");
    for (const name of Object.keys(reference.contract.inputs)) {
      if (run.contract.inputs[name]?.sha256 !== reference.contract.inputs[name].sha256) {
        throw new Error(`SFT pilot ${name} hashes differ`);
      }
    }
    if (!isDeepStrictEqual(run.validation.map((point) => point.step), reference.validation.map((point) => point.step))) {
      throw new Error("SFT pilot validation steps differ");
    }
  }
  const ranking = runs.map((run) => ({
    learning_rate: run.contract.learning_rate,
    run_dir: run.dir,
    final_three_validation_mean: run.final_three_validation_mean,
    final_validation_loss: run.validation.at(-1)!.loss,
    final_train_loss: run.final_train_loss,
    median_tokens_per_sec_after_warmup: run.median_tokens_per_sec_after_warmup,
  })).sort((left, right) =>
    left.final_three_validation_mean - right.final_three_validation_mean ||
    left.final_validation_loss - right.final_validation_loss ||
    left.learning_rate - right.learning_rate);
  const report = {
    schema: "alpha-sft-lr-sweep-analysis-v1",
    result: "PASS",
    selection_rule: "lowest final-three aligned SFT validation-loss mean; final validation loss then lower LR break ties",
    selected_learning_rate: ranking[0].learning_rate,
    selected_run_dir: ranking[0].run_dir,
    source_commit: reference.contract.source_commit,
    input_sha256: Object.fromEntries(Object.entries(reference.contract.inputs).map(([name, input]) => [name, input.sha256])),
    ranking,
    runs,
  };
  await writeFile(cli.out, `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
