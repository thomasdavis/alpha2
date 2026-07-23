#!/usr/bin/env npx tsx
/** Fail-closed completion proof for the contracted 1B-token flagship pretrain. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readFile, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";

interface FlagshipContract {
  schema: string;
  expected_params: number;
  expected_steps: number;
  batch_size: number;
  block_size: number;
  grad_accum_steps: number;
  expected_tokens: number;
  minimum_train_tokens: number;
  learning_rate: number;
  learning_rate_min: number;
  warmup_iters: number;
  source_commit: string;
  lr_selection: { path: string; sha256: string; selected_learning_rate: number };
  data_manifest: { path: string; sha256: string; shards: { path: string; sha256: string }[] };
  tokenizer: { path: string; sha256: string };
}

interface Metric {
  step: number;
  loss: number;
  valLoss?: number;
  gradNorm: number;
  tokens_per_sec: number;
  host_rss_mb: number;
  host_external_mb: number;
  gpu_allocator_free_range_overflows?: number;
}

interface CheckpointHeader {
  step?: number;
  optimizerStep?: number;
  modelConfig?: Record<string, unknown>;
  tokenizerArtifacts?: Record<string, unknown>;
  tensors?: { name?: string; shape?: number[]; elements?: number }[];
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

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) throw new Error("cannot summarize an empty series");
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor((ordered.length - 1) * fraction)];
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function auditCheckpoint(
  checkpointPath: string,
  tokenizer: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const checkpointStat = await stat(checkpointPath);
  if (checkpointStat.size < 650 * 1024 * 1024 || checkpointStat.size > 750 * 1024 * 1024) {
    throw new Error(`terminal checkpoint size ${checkpointStat.size} is outside the full-state envelope`);
  }
  const handle = await open(checkpointPath, "r");
  const prefix = Buffer.alloc(8);
  let headerLength = 0;
  let header: CheckpointHeader;
  try {
    const prefixRead = await handle.read(prefix, 0, prefix.length, 0);
    if (prefixRead.bytesRead !== 8 || prefix.subarray(0, 4).toString("ascii") !== "ALPH") {
      throw new Error("terminal checkpoint is not an Alpha binary checkpoint");
    }
    headerLength = prefix.readUInt32LE(4);
    if (headerLength < 2 || headerLength > 64 * 1024 * 1024) {
      throw new Error(`terminal checkpoint header length is invalid: ${headerLength}`);
    }
    const headerBytes = Buffer.alloc(headerLength);
    const headerRead = await handle.read(headerBytes, 0, headerLength, 8);
    if (headerRead.bytesRead !== headerLength) throw new Error("terminal checkpoint header is truncated");
    header = JSON.parse(headerBytes.toString("utf8")) as CheckpointHeader;

    if (header.step !== 61_036 || header.optimizerStep !== 61_036) {
      throw new Error(`terminal checkpoint step/optimizer ${String(header.step)}/${String(header.optimizerStep)} != 61036`);
    }
    if (!isDeepStrictEqual(header.modelConfig, EXPECTED_MODEL)) throw new Error("terminal checkpoint model config drifted");
    if (!isDeepStrictEqual(header.tokenizerArtifacts, tokenizer)) throw new Error("terminal checkpoint tokenizer drifted");
    if (!Array.isArray(header.tensors) || header.tensors.length === 0) throw new Error("terminal checkpoint has no tensors");

    let expectedBytes = 8 + headerLength;
    let parameterElements = 0;
    const parameterNames = new Set<string>();
    for (const tensor of header.tensors) {
      if (!tensor.name || !Number.isSafeInteger(tensor.elements) || Number(tensor.elements) < 0) {
        throw new Error(`terminal checkpoint has an invalid tensor record: ${String(tensor.name)}`);
      }
      if (!Array.isArray(tensor.shape) || tensor.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 0)) {
        throw new Error(`terminal checkpoint tensor ${tensor.name} has an invalid shape`);
      }
      const elements = Number(tensor.elements);
      const shapeElements = tensor.shape.reduce((product, dim) => product * dim, 1);
      if (shapeElements !== elements) throw new Error(`terminal checkpoint tensor ${tensor.name} shape drifted`);
      expectedBytes += elements * 4;
      if (tensor.name.startsWith("p.")) {
        if (parameterNames.has(tensor.name)) throw new Error(`duplicate terminal parameter ${tensor.name}`);
        parameterNames.add(tensor.name);
        parameterElements += elements;
      }
    }
    if (expectedBytes !== checkpointStat.size) {
      throw new Error(`terminal checkpoint byte count ${checkpointStat.size} != ${expectedBytes}`);
    }
    if (parameterElements !== 57_688_576) {
      throw new Error(`terminal checkpoint parameters ${parameterElements} != 57688576`);
    }

    const chunk = Buffer.allocUnsafe(8 * 1024 * 1024);
    let offset = 8 + headerLength;
    let finiteParameterElements = 0;
    let nonzeroParameterElements = 0;
    for (const tensor of header.tensors) {
      let remaining = Number(tensor.elements) * 4;
      if (!tensor.name!.startsWith("p.")) {
        offset += remaining;
        continue;
      }
      while (remaining > 0) {
        const wanted = Math.min(chunk.length, remaining);
        let filled = 0;
        while (filled < wanted) {
          const result = await handle.read(chunk, filled, wanted - filled, offset + filled);
          if (result.bytesRead === 0) throw new Error(`terminal checkpoint tensor ${tensor.name} is truncated`);
          filled += result.bytesRead;
        }
        for (let byteOffset = 0; byteOffset < wanted; byteOffset += 4) {
          const bits = chunk.readUInt32LE(byteOffset);
          if ((bits & 0x7f80_0000) === 0x7f80_0000) {
            throw new Error(`terminal checkpoint tensor ${tensor.name} contains a non-finite parameter`);
          }
          finiteParameterElements++;
          if ((bits & 0x7fff_ffff) !== 0) nonzeroParameterElements++;
        }
        offset += wanted;
        remaining -= wanted;
      }
    }
    if (finiteParameterElements !== 57_688_576 || nonzeroParameterElements < finiteParameterElements / 2) {
      throw new Error(`terminal parameter payload failed finite/nonzero audit: ${finiteParameterElements}/${nonzeroParameterElements}`);
    }
    return {
      path: checkpointPath,
      bytes: checkpointStat.size,
      sha256: await sha256File(checkpointPath),
      header_bytes: headerLength,
      tensor_count: header.tensors.length,
      parameter_tensor_count: parameterNames.size,
      parameter_elements: parameterElements,
      finite_parameter_elements: finiteParameterElements,
      nonzero_parameter_elements: nonzeroParameterElements,
    };
  } finally {
    await handle.close();
  }
}

async function main(): Promise<void> {
  const cli = parseArgs();
  for (const required of ["run", "out", "sourceCommit", "selectionReport", "dataManifest", "tokenizer"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  if (!/^[0-9a-f]{40}$/.test(cli.sourceCommit)) throw new Error("invalid --sourceCommit");

  const [contractText, configText, metricsText, selectionText, manifestText, tokenizerText] = await Promise.all([
    readFile(path.join(cli.run, "flagship-contract.json"), "utf8"),
    readFile(path.join(cli.run, "config.json"), "utf8"),
    readFile(path.join(cli.run, "metrics.jsonl"), "utf8"),
    readFile(cli.selectionReport, "utf8"),
    readFile(cli.dataManifest, "utf8"),
    readFile(cli.tokenizer, "utf8"),
  ]);
  const contract = JSON.parse(contractText) as FlagshipContract;
  const config = JSON.parse(configText) as any;
  const selection = JSON.parse(selectionText) as any;
  const manifest = JSON.parse(manifestText) as any;
  const tokenizer = JSON.parse(tokenizerText) as Record<string, unknown>;

  const expectedContract: Record<string, unknown> = {
    schema: "alpha-flagship-contract-v2",
    expected_params: 57_688_576,
    expected_steps: 61_036,
    batch_size: 16,
    block_size: 1_024,
    grad_accum_steps: 1,
    expected_tokens: 1_000_013_824,
    minimum_train_tokens: 1_000_013_824,
    warmup_iters: 610,
    source_commit: cli.sourceCommit,
  };
  for (const [key, expected] of Object.entries(expectedContract)) {
    if ((contract as any)[key] !== expected) throw new Error(`contract ${key} ${(contract as any)[key]} != ${expected}`);
  }
  if (![1e-3, 2e-3, 3e-3].includes(contract.learning_rate) || contract.learning_rate_min !== contract.learning_rate / 10) {
    throw new Error("contract learning-rate schedule drifted");
  }
  const [selectionSha, manifestSha, tokenizerSha] = await Promise.all([
    sha256File(cli.selectionReport),
    sha256File(cli.dataManifest),
    sha256File(cli.tokenizer),
  ]);
  if (contract.lr_selection.sha256 !== selectionSha || contract.lr_selection.selected_learning_rate !== contract.learning_rate) {
    throw new Error("contract LR-selection binding drifted");
  }
  if (selection.schema !== "alpha-lr-sweep-analysis-v1" || selection.result !== "PASS" ||
      selection.selected_learning_rate !== contract.learning_rate || selection.tokenizer_sha256 !== tokenizerSha) {
    throw new Error("LR-selection report does not authorize this flagship");
  }
  const selectedRates = Array.isArray(selection.ranking)
    ? selection.ranking.map((entry: any) => entry.learning_rate).sort((left: number, right: number) => left - right)
    : [];
  if (!isDeepStrictEqual(selectedRates, [1e-3, 2e-3, 3e-3]) ||
      selection.ranking[0]?.learning_rate !== contract.learning_rate) {
    throw new Error("LR-selection report ranking drifted");
  }
  if (manifest.schema !== "alpha-pretrain-shards-v1" || !Array.isArray(manifest.shards) || manifest.shards.length !== 3) {
    throw new Error("flagship data manifest shape drifted");
  }
  if (contract.data_manifest.sha256 !== manifestSha || !isDeepStrictEqual(contract.data_manifest.shards, manifest.shards)) {
    throw new Error("contract data-manifest binding drifted");
  }
  if (contract.tokenizer.sha256 !== tokenizerSha) throw new Error("contract tokenizer binding drifted");

  if (!isDeepStrictEqual(config.modelConfig, EXPECTED_MODEL)) throw new Error("flagship model config drifted");
  if (config.totalParams !== contract.expected_params || config.domain !== "alpha_llama") {
    throw new Error("flagship parameter count/domain drifted");
  }
  const expectedTrainConfig: Record<string, unknown> = {
    iters: 61_036,
    batchSize: 16,
    lr: contract.learning_rate,
    lrMin: contract.learning_rate_min,
    warmupIters: 610,
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
    packed: true,
  };
  for (const [key, expected] of Object.entries(expectedTrainConfig)) {
    if (config.trainConfig?.[key] !== expected) {
      throw new Error(`trainConfig.${key} ${String(config.trainConfig?.[key])} != ${String(expected)}`);
    }
  }
  if (config.dataStats?.mode !== "pretrain" || config.dataStats?.trainTokens < contract.minimum_train_tokens) {
    throw new Error("flagship pretrain data coverage is below contract");
  }
  if (!Array.isArray(config.dataStats?.shards) || config.dataStats.shards.length !== manifest.shards.length) {
    throw new Error("flagship configured shard count drifted");
  }
  for (let index = 0; index < manifest.shards.length; index++) {
    const configured = config.dataStats.shards[index];
    if (path.basename(configured.path) !== manifest.shards[index].path || configured.trainTokens <= 0 || configured.valTokens <= 0) {
      throw new Error(`flagship configured shard ${index} drifted`);
    }
  }

  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (metrics.length !== contract.expected_steps) throw new Error(`metric rows ${metrics.length} != ${contract.expected_steps}`);
  for (const [index, metric] of metrics.entries()) {
    if (metric.step !== index + 1) throw new Error(`expected metric step ${index + 1}, found ${metric.step}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec", "host_rss_mb", "host_external_mb"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`non-finite ${field} at step ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) {
      throw new Error(`non-finite valLoss at step ${metric.step}`);
    }
  }
  const expectedEvalSteps: number[] = [];
  for (let step = 500; step <= 61_000; step += 500) expectedEvalSteps.push(step);
  expectedEvalSteps.push(61_036);
  const evalRows = metrics.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  if (!isDeepStrictEqual(evalRows.map((metric) => metric.step), expectedEvalSteps)) {
    throw new Error("flagship validation cadence drifted");
  }
  const telemetry = metrics.filter((metric): metric is Metric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  if (telemetry.length < Math.floor(contract.expected_steps / 100)) {
    throw new Error(`allocator telemetry samples ${telemetry.length} are incomplete`);
  }
  const maxTelemetryGap = Math.max(0, ...telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step));
  const overflowMax = Math.max(0, ...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  if (maxTelemetryGap > 100 || telemetry.at(-1)?.step !== contract.expected_steps || overflowMax !== 0) {
    throw new Error("flagship allocator telemetry failed completeness/overflow gate");
  }

  const checkpoint = await auditCheckpoint(path.join(cli.run, "checkpoint-61036.json"), tokenizer);
  const steady = metrics.slice(610);
  const throughput = steady.map((metric) => metric.tokens_per_sec);
  const checks = {
    exact_metric_rows: metrics.length === 61_036,
    exact_training_tokens: 61_036 * 16 * 1_024 === contract.expected_tokens,
    throughput_p10_at_least_3000: percentile(throughput, 0.1) >= 3_000,
    throughput_median_at_least_3000: percentile(throughput, 0.5) >= 3_000,
    allocator_overflow_zero: overflowMax === 0,
    terminal_checkpoint_native_audit: true,
  };
  const result = Object.values(checks).every(Boolean) ? "PASS" : "FAIL";
  const report = {
    schema: "alpha-flagship-pretrain-analysis-v1",
    result,
    source_commit: contract.source_commit,
    run_dir: cli.run,
    contract_sha256: createHash("sha256").update(contractText).digest("hex"),
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    selection_report_sha256: selectionSha,
    data_manifest_sha256: manifestSha,
    tokenizer_sha256: tokenizerSha,
    learning_rate: contract.learning_rate,
    rows: metrics.length,
    tokens: contract.expected_tokens,
    train_tokens_available: config.dataStats.trainTokens,
    validation_tokens_available: config.dataStats.valTokens,
    median_tokens_per_sec_after_warmup: percentile(throughput, 0.5),
    p10_tokens_per_sec_after_warmup: percentile(throughput, 0.1),
    final_train_loss: metrics.at(-1)!.loss,
    last_100_train_loss_mean: mean(metrics.slice(-100).map((metric) => metric.loss)),
    validation: evalRows.map((metric) => ({ step: metric.step, loss: metric.valLoss })),
    final_three_validation_mean: mean(evalRows.slice(-3).map((metric) => metric.valLoss)),
    allocator_telemetry_samples: telemetry.length,
    allocator_telemetry_max_step_gap: maxTelemetryGap,
    allocator_overflow_max: overflowMax,
    host_rss_mb: { min: Math.min(...steady.map((metric) => metric.host_rss_mb)), max: Math.max(...steady.map((metric) => metric.host_rss_mb)) },
    host_external_mb: { min: Math.min(...steady.map((metric) => metric.host_external_mb)), max: Math.max(...steady.map((metric) => metric.host_external_mb)) },
    checkpoint,
    checks,
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (result !== "PASS") process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
