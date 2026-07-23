#!/usr/bin/env npx tsx
/** Fail-closed completion proof for the contracted one-epoch flagship SFT run. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";

interface InputBinding {
  path: string;
  sha256: string;
}

interface SftContract {
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
  inputs: Record<string, InputBinding & { selected_learning_rate?: number }>;
}

interface Metric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  gpu_allocator_free_range_overflows?: number;
}

const EXPECTED_MODEL = {
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
  if (values.length === 0) throw new Error("cannot summarize an empty series");
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const cli = parseArgs();
  for (const required of ["run", "out", "sourceCommit", "selectionReport", "terminalVerification"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  if (!/^[0-9a-f]{40}$/.test(cli.sourceCommit)) throw new Error("invalid --sourceCommit");

  const [contractText, configText, metricsText, selectionText, verificationText] = await Promise.all([
    readFile(path.join(cli.run, "sft-contract.json"), "utf8"),
    readFile(path.join(cli.run, "config.json"), "utf8"),
    readFile(path.join(cli.run, "metrics.jsonl"), "utf8"),
    readFile(cli.selectionReport, "utf8"),
    readFile(cli.terminalVerification, "utf8"),
  ]);
  const contract = JSON.parse(contractText) as SftContract;
  const config = JSON.parse(configText) as any;
  const selection = JSON.parse(selectionText) as any;
  const verification = JSON.parse(verificationText) as any;

  const expectedContract: Record<string, unknown> = {
    schema: "alpha-flagship-sft-contract-v1",
    job: "flagship",
    expected_params: 57_688_576,
    expected_steps: 30_322,
    batch_size: 16,
    block_size: 1_024,
    grad_accum_steps: 1,
    padded_train_tokens: 496_795_648,
    train_conversations: 485_150,
    validation_conversations: 26_278,
    validation_fraction: 0.05,
    warmup_iters: 303,
    source_commit: cli.sourceCommit,
  };
  for (const [key, expected] of Object.entries(expectedContract)) {
    if ((contract as any)[key] !== expected) throw new Error(`contract ${key} ${(contract as any)[key]} != ${expected}`);
  }
  if (![1e-4, 3e-4, 1e-3].includes(contract.learning_rate) || contract.learning_rate_min !== contract.learning_rate / 10) {
    throw new Error("SFT contract learning-rate schedule drifted");
  }
  const requiredInputs = ["corpus", "manifest", "length_audit", "mask_audit", "tokenizer", "base_checkpoint", "selection_report"];
  for (const name of requiredInputs) {
    const input = contract.inputs?.[name];
    if (!input?.path || !/^[0-9a-f]{64}$/.test(input.sha256)) throw new Error(`invalid SFT ${name} binding`);
  }

  const selectionSha = await sha256File(cli.selectionReport);
  if (selectionSha !== contract.inputs.selection_report.sha256 ||
      contract.inputs.selection_report.selected_learning_rate !== contract.learning_rate ||
      selection.schema !== "alpha-sft-lr-sweep-analysis-v1" || selection.result !== "PASS" ||
      selection.selected_learning_rate !== contract.learning_rate || selection.source_commit !== contract.source_commit) {
    throw new Error("SFT LR-selection report does not authorize this run");
  }
  const selectedRates = Array.isArray(selection.ranking)
    ? selection.ranking.map((entry: any) => entry.learning_rate).sort((left: number, right: number) => left - right)
    : [];
  if (!isDeepStrictEqual(selectedRates, [1e-4, 3e-4, 1e-3]) || selection.ranking[0]?.learning_rate !== contract.learning_rate) {
    throw new Error("SFT LR-selection report ranking drifted");
  }
  for (const name of requiredInputs.filter((name) => name !== "selection_report")) {
    if (selection.input_sha256?.[name] !== contract.inputs[name].sha256) {
      throw new Error(`SFT LR-selection ${name} hash drifted`);
    }
  }

  if (!isDeepStrictEqual(config.modelConfig, EXPECTED_MODEL)) throw new Error("SFT model config drifted");
  if (config.totalParams !== contract.expected_params || config.domain !== "alpha_llama") {
    throw new Error("SFT parameter count/domain drifted");
  }
  if (config.initCheckpointPath !== contract.inputs.base_checkpoint.path) throw new Error("SFT base initialization path drifted");
  const expectedTrainConfig: Record<string, unknown> = {
    iters: 30_322,
    batchSize: 16,
    lr: contract.learning_rate,
    lrMin: contract.learning_rate_min,
    warmupIters: 303,
    beta1: 0.9,
    beta2: 0.95,
    eps: 1e-8,
    weightDecay: 0.1,
    gradClip: 1,
    evalInterval: 500,
    checkpointInterval: 1_000,
    evalIters: 5,
    seed: 42,
    backend: "helios",
    gradAccumSteps: 1,
    packed: false,
  };
  for (const [key, expected] of Object.entries(expectedTrainConfig)) {
    if (config.trainConfig?.[key] !== expected) {
      throw new Error(`trainConfig.${key} ${String(config.trainConfig?.[key])} != ${String(expected)}`);
    }
  }
  if (config.dataStats?.mode !== "sft" || config.dataStats.trainConversations !== 485_150 ||
      config.dataStats.valConversations !== 26_278 || config.dataStats.trainTokens <= 0 || config.dataStats.valTokens <= 0) {
    throw new Error("SFT data coverage drifted");
  }

  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (metrics.length !== contract.expected_steps) throw new Error(`metric rows ${metrics.length} != ${contract.expected_steps}`);
  for (const [index, metric] of metrics.entries()) {
    if (metric.step !== index + 1) throw new Error(`expected metric step ${index + 1}, found ${metric.step}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`non-finite ${field} at step ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) {
      throw new Error(`non-finite valLoss at step ${metric.step}`);
    }
  }
  const expectedEvalSteps: number[] = [];
  for (let step = 500; step <= 30_000; step += 500) expectedEvalSteps.push(step);
  expectedEvalSteps.push(30_322);
  const evalRows = metrics.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  if (!isDeepStrictEqual(evalRows.map((metric) => metric.step), expectedEvalSteps)) {
    throw new Error("SFT validation cadence drifted");
  }
  const telemetry = metrics.filter((metric): metric is Metric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  if (telemetry.length < Math.floor(contract.expected_steps / 100)) {
    throw new Error(`SFT allocator telemetry samples ${telemetry.length} are incomplete`);
  }
  const maxTelemetryGap = Math.max(0, ...telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step));
  const overflowMax = Math.max(0, ...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  if (maxTelemetryGap > 100 || telemetry.at(-1)?.step !== contract.expected_steps || overflowMax !== 0) {
    throw new Error("SFT allocator telemetry failed completeness/overflow gate");
  }

  if (verification.schema !== "alpha-flagship-sft-input-verification-v1" || verification.result !== "PASS") {
    throw new Error("terminal SFT checkpoint verification did not pass");
  }
  for (const name of ["corpus", "manifest", "length_audit", "mask_audit", "tokenizer"] as const) {
    if (verification[name]?.sha256 !== contract.inputs[name].sha256) {
      throw new Error(`terminal verification ${name} hash drifted`);
    }
  }
  const checkpointPath = path.join(cli.run, "checkpoint-30322.json");
  const checkpointStat = await stat(checkpointPath);
  const checkpointSha = await sha256File(checkpointPath);
  const terminal = verification.base_checkpoint;
  if (terminal?.path !== checkpointPath || terminal?.sha256 !== checkpointSha || terminal?.bytes !== checkpointStat.size ||
      terminal?.step !== 30_322 || terminal?.parameter_elements !== 57_688_576 ||
      terminal?.finite_parameter_elements !== 57_688_576 || terminal?.nonzero_parameter_elements < 57_688_576 / 2 ||
      !isDeepStrictEqual(terminal?.model_config, EXPECTED_MODEL)) {
    throw new Error("terminal SFT checkpoint native audit drifted");
  }

  const report = {
    schema: "alpha-flagship-sft-analysis-v1",
    result: "PASS",
    source_commit: contract.source_commit,
    run_dir: cli.run,
    contract_sha256: createHash("sha256").update(contractText).digest("hex"),
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    selection_report_sha256: selectionSha,
    terminal_verification_sha256: createHash("sha256").update(verificationText).digest("hex"),
    learning_rate: contract.learning_rate,
    rows: metrics.length,
    padded_training_tokens: contract.padded_train_tokens,
    train_conversations: config.dataStats.trainConversations,
    validation_conversations: config.dataStats.valConversations,
    median_tokens_per_sec_after_warmup: median(metrics.slice(303).map((metric) => metric.tokens_per_sec)),
    final_train_loss: metrics.at(-1)!.loss,
    last_100_train_loss_mean: mean(metrics.slice(-100).map((metric) => metric.loss)),
    validation: evalRows.map((metric) => ({ step: metric.step, loss: metric.valLoss })),
    final_three_validation_mean: mean(evalRows.slice(-3).map((metric) => metric.valLoss)),
    allocator_telemetry_samples: telemetry.length,
    allocator_telemetry_max_step_gap: maxTelemetryGap,
    allocator_overflow_max: overflowMax,
    checkpoint: {
      path: checkpointPath,
      bytes: checkpointStat.size,
      sha256: checkpointSha,
      parameter_elements: terminal.parameter_elements,
      finite_parameter_elements: terminal.finite_parameter_elements,
      nonzero_parameter_elements: terminal.nonzero_parameter_elements,
    },
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
