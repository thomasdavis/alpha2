#!/usr/bin/env npx tsx
/** Fail-closed completion proof for the selected 97M Alpha foundation run. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readFile, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import * as path from "node:path";

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
  if (checkpointStat.size < 1_050 * 1024 * 1024 || checkpointStat.size > 1_250 * 1024 * 1024) {
    throw new Error(`terminal checkpoint size ${checkpointStat.size} is outside the full-state envelope`);
  }
  const handle = await open(checkpointPath, "r");
  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, prefix.length, 0);
    if (prefixRead.bytesRead !== 8 || prefix.subarray(0, 4).toString("ascii") !== "ALPH") {
      throw new Error("terminal checkpoint is not an Alpha binary checkpoint");
    }
    const headerLength = prefix.readUInt32LE(4);
    if (headerLength < 2 || headerLength > 64 * 1024 * 1024) throw new Error("terminal checkpoint header length is invalid");
    const headerBytes = Buffer.alloc(headerLength);
    const headerRead = await handle.read(headerBytes, 0, headerLength, 8);
    if (headerRead.bytesRead !== headerLength) throw new Error("terminal checkpoint header is truncated");
    const header = JSON.parse(headerBytes.toString("utf8")) as CheckpointHeader;
    if (header.step !== 79_020 || header.optimizerStep !== 79_020) {
      throw new Error(`terminal checkpoint step/optimizer ${String(header.step)}/${String(header.optimizerStep)} != 79020`);
    }
    if (!isDeepStrictEqual(header.modelConfig, EXPECTED_MODEL)) throw new Error("terminal checkpoint model config drifted");
    if (!isDeepStrictEqual(header.tokenizerArtifacts, tokenizer)) throw new Error("terminal checkpoint tokenizer drifted");
    if (!Array.isArray(header.tensors) || header.tensors.length === 0) throw new Error("terminal checkpoint has no tensors");

    let expectedBytes = 8 + headerLength;
    let parameterElements = 0;
    const parameterNames = new Set<string>();
    for (const tensor of header.tensors) {
      if (!tensor.name || !Number.isSafeInteger(tensor.elements) || Number(tensor.elements) < 0 || !Array.isArray(tensor.shape)) {
        throw new Error(`invalid terminal tensor record: ${String(tensor.name)}`);
      }
      const elements = Number(tensor.elements);
      if (tensor.shape.some((dim) => !Number.isSafeInteger(dim) || dim < 0) ||
          tensor.shape.reduce((product, dim) => product * dim, 1) !== elements) {
        throw new Error(`terminal checkpoint tensor ${tensor.name} shape drifted`);
      }
      expectedBytes += elements * 4;
      if (tensor.name.startsWith("p.")) {
        if (parameterNames.has(tensor.name)) throw new Error(`duplicate terminal parameter ${tensor.name}`);
        parameterNames.add(tensor.name);
        parameterElements += elements;
      }
    }
    if (expectedBytes !== checkpointStat.size) throw new Error(`terminal checkpoint bytes ${checkpointStat.size} != ${expectedBytes}`);
    if (parameterElements !== 97_098_880) throw new Error(`terminal checkpoint parameters ${parameterElements} != 97098880`);

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
          if (result.bytesRead === 0) throw new Error(`terminal tensor ${tensor.name} is truncated`);
          filled += result.bytesRead;
        }
        for (let byteOffset = 0; byteOffset < wanted; byteOffset += 4) {
          const bits = chunk.readUInt32LE(byteOffset);
          if ((bits & 0x7f80_0000) === 0x7f80_0000) throw new Error(`terminal tensor ${tensor.name} contains non-finite parameters`);
          finiteParameterElements++;
          if ((bits & 0x7fff_ffff) !== 0) nonzeroParameterElements++;
        }
        offset += wanted;
        remaining -= wanted;
      }
    }
    if (finiteParameterElements !== 97_098_880 || nonzeroParameterElements < finiteParameterElements / 2) {
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
  for (const required of ["run", "out", "selectionReport", "dataManifest", "validation", "tokenizer"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  const [contractText, configText, metricsText, exitCode, selectionText, manifestText, tokenizerText] = await Promise.all([
    readFile(path.join(cli.run, "foundation-contract.json"), "utf8"),
    readFile(path.join(cli.run, "config.json"), "utf8"),
    readFile(path.join(cli.run, "metrics.jsonl"), "utf8"),
    readFile(path.join(cli.run, "exit-code.txt"), "utf8"),
    readFile(cli.selectionReport, "utf8"),
    readFile(cli.dataManifest, "utf8"),
    readFile(cli.tokenizer, "utf8"),
  ]);
  if (exitCode.trim() !== "0") throw new Error(`foundation exit code is not zero: ${exitCode.trim()}`);
  const contract = JSON.parse(contractText) as any;
  const config = JSON.parse(configText) as any;
  const selection = JSON.parse(selectionText) as any;
  const manifest = JSON.parse(manifestText) as any;
  const tokenizer = JSON.parse(tokenizerText) as Record<string, unknown>;
  const expectedContract: Record<string, unknown> = {
    schema: "alpha-foundation-full-contract-v1",
    expected_params: 97_098_880,
    expected_steps: 79_020,
    batch_size: 24,
    block_size: 1_024,
    grad_accum_steps: 1,
    expected_tokens: 1_941_995_520,
    minimum_train_tokens: 1_941_995_520,
    warmup_iters: 790,
    eval_interval: 500,
    checkpoint_interval: 1_000,
    eval_iters: 5,
  };
  for (const [key, expected] of Object.entries(expectedContract)) {
    if (contract[key] !== expected) throw new Error(`contract ${key} ${String(contract[key])} != ${String(expected)}`);
  }
  const expectedEngine = {
    backend: "helios",
    accelerator_api: "vulkan",
    kernel_policy: "layout-portfolio-r42c-r2-v2",
    environment: {
      HELIOS_DISABLE_COOP_MAT: "1",
      HELIOS_FLASH_FWD_PREFER_COOP2: "0",
      HELIOS_WG_SIZE: "64",
      HELIOS_MATMUL_REG4X2: "1",
      HELIOS_MATMUL_REG4X2_TRANSPOSED_B: "1",
      HELIOS_MATMUL_TRANSPOSED_B_COALESCED: "1",
      HELIOS_MATMUL_REG2X2: "1",
      HELIOS_MAX_OUTPUT_POOL_ENTRIES: "512",
    },
  };
  if (!isDeepStrictEqual(contract.engine, expectedEngine)) {
    throw new Error("foundation Helios kernel policy drifted");
  }
  if (!/^[0-9a-f]{40}$/.test(contract.source_commit ?? "") ||
      ![1e-3, 2e-3, 3e-3].includes(contract.learning_rate) ||
      contract.learning_rate_min !== contract.learning_rate / 10) {
    throw new Error("foundation source or LR schedule drifted");
  }
  const [selectionSha, manifestSha, validationSha, tokenizerSha] = await Promise.all([
    sha256File(cli.selectionReport), sha256File(cli.dataManifest), sha256File(cli.validation), sha256File(cli.tokenizer),
  ]);
  if (selection.schema !== "alpha-foundation-candidate-lr-sweep-v1" || selection.result !== "PASS" ||
      selection.selected_learning_rate !== contract.learning_rate || selection.ranking?.[0]?.learning_rate !== contract.learning_rate ||
      contract.lr_selection.sha256 !== selectionSha || contract.lr_selection.pilot_source_commit !== selection.source_commit) {
    throw new Error("foundation LR-selection binding drifted");
  }
  if (manifest.schema !== "alpha-pretrain-shards-v1" || manifest.shards?.length !== 4 ||
      contract.data_manifest.sha256 !== manifestSha || !isDeepStrictEqual(contract.data_manifest.shards, manifest.shards)) {
    throw new Error("foundation data-manifest binding drifted");
  }
  if (contract.validation.sha256 !== validationSha || contract.validation.wholly_held_out !== true ||
      contract.tokenizer.sha256 !== tokenizerSha || selection.tokenizer_sha256 !== tokenizerSha) {
    throw new Error("foundation validation/tokenizer binding drifted");
  }
  if (!isDeepStrictEqual(config.modelConfig, EXPECTED_MODEL) || config.totalParams !== 97_098_880 || config.domain !== "alpha_llama") {
    throw new Error("foundation model identity drifted");
  }
  const expectedTrainConfig: Record<string, unknown> = {
    iters: 79_020, batchSize: 24, lr: contract.learning_rate, lrMin: contract.learning_rate_min,
    warmupIters: 790, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1, gradClip: 1,
    evalInterval: 500, checkpointInterval: 1_000, evalIters: 5, seed: 42, backend: "helios",
    gradAccumSteps: 1, packed: true, symbio: false,
  };
  for (const [key, expected] of Object.entries(expectedTrainConfig)) {
    if (config.trainConfig?.[key] !== expected) throw new Error(`trainConfig.${key} ${String(config.trainConfig?.[key])} != ${String(expected)}`);
  }
  if (config.dataStats?.mode !== "pretrain" || config.dataStats.trainTokens < contract.minimum_train_tokens ||
      config.dataStats.validation?.whollyHeldOut !== true || config.dataStats.validation?.valTokens < 1_000_000 ||
      !Array.isArray(config.dataStats.shards) || config.dataStats.shards.length !== 4 ||
      config.dataStats.shards.some((shard: any) => shard.trainTokens <= 0 || shard.valTokens !== 0)) {
    throw new Error("foundation train/held-out data coverage drifted");
  }

  const metrics = metricsText.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as Metric);
  if (metrics.length !== 79_020) throw new Error(`metric rows ${metrics.length} != 79020`);
  metrics.forEach((metric, index) => {
    if (metric.step !== index + 1) throw new Error(`metric step ${metric.step} != ${index + 1}`);
    for (const field of ["loss", "gradNorm", "tokens_per_sec", "host_rss_mb", "host_external_mb"] as const) {
      if (!Number.isFinite(metric[field])) throw new Error(`non-finite ${field} at ${metric.step}`);
    }
    if (metric.valLoss !== undefined && !Number.isFinite(metric.valLoss)) throw new Error(`non-finite validation at ${metric.step}`);
  });
  const expectedEvalSteps: number[] = [];
  for (let step = 500; step <= 79_000; step += 500) expectedEvalSteps.push(step);
  expectedEvalSteps.push(79_020);
  const evalRows = metrics.filter((metric): metric is Metric & { valLoss: number } => metric.valLoss !== undefined);
  if (!isDeepStrictEqual(evalRows.map((metric) => metric.step), expectedEvalSteps)) throw new Error("foundation validation cadence drifted");
  const telemetry = metrics.filter((metric): metric is Metric & { gpu_allocator_free_range_overflows: number } =>
    Number.isFinite(metric.gpu_allocator_free_range_overflows));
  const overflowMax = Math.max(0, ...telemetry.map((metric) => metric.gpu_allocator_free_range_overflows));
  const maxTelemetryGap = Math.max(0, ...telemetry.slice(1).map((metric, index) => metric.step - telemetry[index].step));
  if (telemetry.length < 790 || maxTelemetryGap > 100 || telemetry.at(-1)?.step !== 79_020 || overflowMax !== 0) {
    throw new Error("foundation allocator telemetry is incomplete or overflowed");
  }
  const checkpoint = await auditCheckpoint(path.join(cli.run, "checkpoint-79020.json"), tokenizer);
  const steady = metrics.slice(790);
  const throughput = steady.map((metric) => metric.tokens_per_sec);
  const checks = {
    exact_metric_rows: metrics.length === 79_020,
    exact_training_tokens: 79_020 * 24 * 1_024 === contract.expected_tokens,
    heldout_loss_improved: mean(evalRows.slice(-3).map((metric) => metric.valLoss)) < mean(evalRows.slice(0, 3).map((metric) => metric.valLoss)),
    throughput_p10_at_least_3000: percentile(throughput, 0.1) >= 3_000,
    throughput_median_at_least_3000: percentile(throughput, 0.5) >= 3_000,
    allocator_overflow_zero: overflowMax === 0,
    terminal_checkpoint_native_audit: true,
  };
  const result = Object.values(checks).every(Boolean) ? "PASS" : "FAIL";
  const report = {
    schema: "alpha-foundation-full-analysis-v1", result, source_commit: contract.source_commit, run_dir: cli.run,
    contract_sha256: createHash("sha256").update(contractText).digest("hex"),
    metrics_sha256: createHash("sha256").update(metricsText).digest("hex"),
    selection_report_sha256: selectionSha, data_manifest_sha256: manifestSha, validation_sha256: validationSha,
    tokenizer_sha256: tokenizerSha, learning_rate: contract.learning_rate, rows: metrics.length,
    tokens: contract.expected_tokens, train_tokens_available: config.dataStats.trainTokens,
    validation_tokens_available: config.dataStats.validation.valTokens,
    median_tokens_per_sec_after_warmup: percentile(throughput, 0.5),
    p10_tokens_per_sec_after_warmup: percentile(throughput, 0.1),
    final_train_loss: metrics.at(-1)!.loss,
    last_100_train_loss_mean: mean(metrics.slice(-100).map((metric) => metric.loss)),
    validation: evalRows.map((metric) => ({ step: metric.step, loss: metric.valLoss })),
    final_three_validation_mean: mean(evalRows.slice(-3).map((metric) => metric.valLoss)),
    allocator_telemetry_samples: telemetry.length, allocator_telemetry_max_step_gap: maxTelemetryGap,
    allocator_overflow_max: overflowMax,
    host_rss_mb: { min: Math.min(...steady.map((metric) => metric.host_rss_mb)), max: Math.max(...steady.map((metric) => metric.host_rss_mb)) },
    host_external_mb: { min: Math.min(...steady.map((metric) => metric.host_external_mb)), max: Math.max(...steady.map((metric) => metric.host_external_mb)) },
    checkpoint, checks,
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (result !== "PASS") process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
