#!/usr/bin/env npx tsx
/** Fail-closed analysis and LR selection for the 97M foundation candidate. */

import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";

interface Metric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  gpu_allocator_free_range_overflows?: number;
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

function mean(values: number[]): number {
  if (values.length === 0) throw new Error("cannot average an empty series");
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

async function summarize(dir: string, expectedLearningRate: number): Promise<Record<string, any>> {
  const [contractText, configText, metricsText, rawHashes, zstHashes, exitCode] = await Promise.all([
    readFile(path.join(dir, "pilot-contract.json"), "utf8"),
    readFile(path.join(dir, "config.json"), "utf8"),
    readFile(path.join(dir, "metrics.jsonl"), "utf8"),
    readFile(path.join(dir, "checkpoint-raw.sha256"), "utf8"),
    readFile(path.join(dir, "checkpoint-zst.sha256"), "utf8"),
    readFile(path.join(dir, "exit-code.txt"), "utf8"),
  ]);
  if (exitCode.trim() !== "0") throw new Error(`${dir}: non-zero exit code ${exitCode.trim()}`);
  const contract = JSON.parse(contractText);
  const config = JSON.parse(configText);
  const expectedContract = {
    schema: "alpha-foundation-candidate-lr-pilot-v1",
    expected_params: 97_098_880,
    expected_steps: 384,
    batch_size: 24,
    block_size: 1_024,
    grad_accum_steps: 1,
    expected_tokens: 9_437_184,
    minimum_train_tokens: 9_437_184,
    learning_rate: expectedLearningRate,
    learning_rate_min: expectedLearningRate / 10,
    warmup_iters: 38,
    eval_interval: 96,
    checkpoint_interval: 192,
  };
  for (const [key, expected] of Object.entries(expectedContract)) {
    if (contract[key] !== expected) throw new Error(`${dir}: contract ${key} ${contract[key]} != ${expected}`);
  }
  if (!/^[0-9a-f]{40}$/.test(contract.source_commit ?? "")) throw new Error(`${dir}: invalid source commit`);
  const expectedModel = {
    vocabSize: 12_288,
    blockSize: 1_024,
    nLayer: 18,
    nEmbd: 640,
    nHead: 10,
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 1_728,
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10_000,
    tieEmbeddings: true,
  };
  if (!isDeepStrictEqual(config.modelConfig, expectedModel)) throw new Error(`${dir}: model config drifted`);
  if (config.totalParams !== 97_098_880 || config.domain !== "alpha_llama") throw new Error(`${dir}: model identity drifted`);
  const expectedTrainConfig: Record<string, unknown> = {
    iters: 384,
    batchSize: 24,
    lr: expectedLearningRate,
    lrMin: expectedLearningRate / 10,
    warmupIters: 38,
    beta1: 0.9,
    beta2: 0.95,
    eps: 1e-8,
    weightDecay: 0.1,
    gradClip: 1,
    evalInterval: 96,
    checkpointInterval: 192,
    evalIters: 4,
    seed: 42,
    backend: "helios",
    gradAccumSteps: 1,
    packed: true,
    symbio: false,
  };
  for (const [key, expected] of Object.entries(expectedTrainConfig)) {
    if (config.trainConfig?.[key] !== expected) {
      throw new Error(`${dir}: trainConfig.${key} ${String(config.trainConfig?.[key])} != ${String(expected)}`);
    }
  }
  if (config.dataStats?.mode !== "pretrain" || config.dataStats?.trainTokens < 9_437_184 || config.dataStats?.valTokens < 100_000) {
    throw new Error(`${dir}: insufficient train/validation tokens`);
  }

  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (metrics.length !== 384) throw new Error(`${dir}: metric rows ${metrics.length} != 384`);
  metrics.forEach((metric, index) => {
    if (metric.step !== index + 1) throw new Error(`${dir}: metric step ${metric.step} != ${index + 1}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`${dir}: non-finite ${field} at ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) throw new Error(`${dir}: non-finite validation loss`);
  });
  const evalRows = metrics.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  if (!isDeepStrictEqual(evalRows.map((metric) => metric.step), [96, 192, 288, 384])) {
    throw new Error(`${dir}: validation cadence drifted`);
  }
  const telemetry = metrics.filter((metric): metric is Metric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  const maxGap = Math.max(0, ...telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step));
  const overflowMax = Math.max(0, ...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  if (telemetry.length < 15 || maxGap > 25 || telemetry.at(-1)?.step !== 384 || overflowMax !== 0) {
    throw new Error(`${dir}: allocator telemetry incomplete or overflowed`);
  }
  const rawLines = rawHashes.trim().split("\n");
  const zstLines = zstHashes.trim().split("\n");
  for (const step of [192, 384]) {
    const base = `checkpoint-${step}.json`;
    if (!rawLines.some((line) => line.endsWith(`  ${base}`))) throw new Error(`${dir}: missing raw hash for ${base}`);
    if (!zstLines.some((line) => line.endsWith(`  ${base}.zst`))) throw new Error(`${dir}: missing zstd hash for ${base}`);
    if ((await stat(path.join(dir, `${base}.zst`))).size < 100_000_000) throw new Error(`${dir}: compressed checkpoint is unexpectedly small`);
  }
  const validation = evalRows.map((metric) => ({ step: metric.step, loss: metric.valLoss }));
  return {
    run_dir: dir,
    learning_rate: expectedLearningRate,
    source_commit: contract.source_commit,
    train_sha256: contract.train_data.sha256,
    validation_sha256: contract.val_data.sha256,
    tokenizer_sha256: contract.tokenizer.sha256,
    contract_sha256: createHash("sha256").update(contractText).digest("hex"),
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    rows: metrics.length,
    tokens: 9_437_184,
    final_train_loss: metrics.at(-1)!.loss,
    last_48_train_loss_mean: mean(metrics.slice(-48).map((metric) => metric.loss)),
    validation,
    final_three_validation_mean: mean(validation.slice(-3).map((point) => point.loss)),
    median_tokens_per_sec_after_warmup: median(metrics.slice(38, -1).map((metric) => metric.tokens_per_sec)),
    allocator_telemetry_samples: telemetry.length,
    allocator_telemetry_max_step_gap: maxGap,
    allocator_overflow_max: overflowMax,
  };
}

async function main(): Promise<void> {
  const cli = parseArgs();
  for (const required of ["lr1e3", "lr2e3", "lr3e3", "out"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  const runs = await Promise.all([
    summarize(cli.lr1e3, 1e-3),
    summarize(cli.lr2e3, 2e-3),
    summarize(cli.lr3e3, 3e-3),
  ]);
  const reference = runs[0];
  for (const run of runs.slice(1)) {
    for (const field of ["source_commit", "train_sha256", "validation_sha256", "tokenizer_sha256"] as const) {
      if (run[field] !== reference[field]) throw new Error(`pilot arms differ in ${field}`);
    }
  }
  const ranking = [...runs].sort((left, right) =>
    left.final_three_validation_mean - right.final_three_validation_mean ||
    left.validation.at(-1)!.loss - right.validation.at(-1)!.loss ||
    left.learning_rate - right.learning_rate);
  const report = {
    schema: "alpha-foundation-candidate-lr-sweep-v1",
    result: "PASS",
    selection_rule: "lowest mean held-out loss across the final three aligned evaluations; final held-out loss then lower LR break ties",
    selected_learning_rate: ranking[0].learning_rate,
    selected_run_dir: ranking[0].run_dir,
    source_commit: reference.source_commit,
    train_sha256: reference.train_sha256,
    validation_sha256: reference.validation_sha256,
    tokenizer_sha256: reference.tokenizer_sha256,
    ranking: ranking.map((run) => ({
      learning_rate: run.learning_rate,
      run_dir: run.run_dir,
      final_three_validation_mean: run.final_three_validation_mean,
      final_validation_loss: run.validation.at(-1)!.loss,
      final_train_loss: run.final_train_loss,
      median_tokens_per_sec_after_warmup: run.median_tokens_per_sec_after_warmup,
    })),
    runs,
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
