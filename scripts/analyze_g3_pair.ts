#!/usr/bin/env npx tsx
/** Verify and compare the two equal-token G3 architecture pilots. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";

interface Contract {
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

interface Metric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  gpu_allocator_free_range_overflows?: number;
}

interface ModelConfig {
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

interface RunSummary {
  dir: string;
  contract: Contract;
  total_params: number;
  metrics_sha256: string;
  rows: number;
  tokens: number;
  median_tokens_per_sec_after_warmup: number;
  final_train_loss: number;
  last_100_train_loss_mean: number;
  eval: { step: number; val_loss: number }[];
  allocator_overflow_max: number;
}

function args(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = process.argv[++i];
    if (!value || value.startsWith("--")) throw new Error(`missing value for --${key}`);
    result[key] = value;
  }
  return result;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

function sha256(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

async function summarize(dir: string, expectedVariant: Contract["variant"]): Promise<RunSummary> {
  const [contractText, configText, metricsText] = await Promise.all([
    readFile(path.join(dir, "pilot-contract.json"), "utf8"),
    readFile(path.join(dir, "config.json"), "utf8"),
    readFile(path.join(dir, "metrics.jsonl"), "utf8"),
  ]);
  const contract = JSON.parse(contractText) as Contract;
  const config = JSON.parse(configText) as {
    totalParams?: number;
    dataStats?: { trainTokens?: number; valTokens?: number };
    trainConfig: { batchSize: number; gradAccumSteps: number; iters: number };
    modelConfig: ModelConfig;
  };
  if (contract.variant !== expectedVariant) throw new Error(`${dir}: contract variant ${contract.variant} != ${expectedVariant}`);
  const rows = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (rows.length !== contract.expected_steps) throw new Error(`${dir}: metric rows ${rows.length} != ${contract.expected_steps}`);
  for (let i = 0; i < rows.length; i++) {
    const metric = rows[i];
    if (metric.step !== i + 1) throw new Error(`${dir}: expected step ${i + 1}, found ${metric.step}`);
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
  const expectedModel: ModelConfig = {
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
  for (const key of Object.keys(expectedModel) as (keyof ModelConfig)[]) {
    if (config.modelConfig[key] !== expectedModel[key]) {
      throw new Error(`${dir}: modelConfig.${key} ${String(config.modelConfig[key])} != ${String(expectedModel[key])}`);
    }
  }
  const tokens = config.trainConfig.iters * config.trainConfig.batchSize * config.trainConfig.gradAccumSteps * config.modelConfig.blockSize;
  if (tokens !== contract.expected_tokens) throw new Error(`${dir}: tokens ${tokens} != contract ${contract.expected_tokens}`);
  if ((config.dataStats?.trainTokens ?? 0) < contract.minimum_train_tokens) {
    throw new Error(`${dir}: only ${String(config.dataStats?.trainTokens)} train tokens; contract requires ${contract.minimum_train_tokens}`);
  }
  const evalRows = rows.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  if (evalRows.length < 3) throw new Error(`${dir}: only ${evalRows.length} validation points`);
  return {
    dir,
    contract,
    total_params: totalParams,
    metrics_sha256: sha256(metricsText),
    rows: rows.length,
    tokens,
    median_tokens_per_sec_after_warmup: median(rows.slice(100).map((metric) => metric.tokens_per_sec)),
    final_train_loss: rows.at(-1)!.loss,
    last_100_train_loss_mean: mean(rows.slice(-100).map((metric) => metric.loss)),
    eval: evalRows.map((metric) => ({ step: metric.step, val_loss: metric.valLoss })),
    allocator_overflow_max: Math.max(0, ...rows.map((metric) => metric.gpu_allocator_free_range_overflows ?? 0)),
  };
}

async function main(): Promise<void> {
  const cli = args();
  if (!cli.llama || !cli.gpt2 || !cli.out) throw new Error("required: --llama, --gpt2, and --out");
  const [llama, gpt2] = await Promise.all([summarize(cli.llama, "llama"), summarize(cli.gpt2, "gpt2")]);
  for (const key of ["source_commit", "learning_rate", "learning_rate_min"] as const) {
    if (llama.contract[key] !== gpt2.contract[key]) throw new Error(`contract ${key} differs between pilots`);
  }
  if (llama.contract.data.sha256 !== gpt2.contract.data.sha256) throw new Error("pilot data hashes differ");
  if (llama.contract.tokenizer.sha256 !== gpt2.contract.tokenizer.sha256) throw new Error("pilot tokenizer hashes differ");
  if (llama.tokens !== gpt2.tokens) throw new Error("pilot token counts differ");
  const paramDifferenceFraction = Math.abs(llama.total_params - gpt2.total_params) / gpt2.total_params;
  if (paramDifferenceFraction > 0.01) throw new Error(`parameter difference ${(paramDifferenceFraction * 100).toFixed(3)}% exceeds 1%`);

  const gpt2Eval = new Map(gpt2.eval.map((point) => [point.step, point.val_loss]));
  const aligned = llama.eval.map((point) => ({
    step: point.step,
    llama: point.val_loss,
    gpt2: gpt2Eval.get(point.step),
  }));
  if (aligned.some((point) => point.gpt2 === undefined) || aligned.length !== gpt2.eval.length) {
    throw new Error("validation steps do not align");
  }
  const final = aligned.at(-1)! as { step: number; llama: number; gpt2: number };
  const lastThree = aligned.slice(-3) as { step: number; llama: number; gpt2: number }[];
  const llamaLastThree = mean(lastThree.map((point) => point.llama));
  const gpt2LastThree = mean(lastThree.map((point) => point.gpt2));
  const pass = final.llama <= final.gpt2 && llamaLastThree <= gpt2LastThree &&
    llama.allocator_overflow_max === 0 && gpt2.allocator_overflow_max === 0;
  const report = {
    schema: "alpha-g3-pair-analysis-v1",
    result: pass ? "PASS" : "FAIL",
    gate: "Llama final and last-three mean validation loss must be <= equal-token/equal-parameter GPT-2 control; zero allocator overflow",
    contracts_match: true,
    parameter_difference_fraction: paramDifferenceFraction,
    llama,
    gpt2,
    comparison: {
      aligned_validation: aligned,
      final_validation_delta_llama_minus_gpt2: final.llama - final.gpt2,
      last_three_mean: { llama: llamaLastThree, gpt2: gpt2LastThree, delta: llamaLastThree - gpt2LastThree },
    },
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
