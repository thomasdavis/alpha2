/**
 * Training loop orchestrator.
 *
 * Pure orchestration: depends on services (Backend, Tokenizer, Optimizer, Checkpoint, Logger).
 * Inspired by microgpt.py's training loop but with proper batching and logging.
 */
import type {
  ModelConfig, TrainConfig, Backend, Tokenizer, Optimizer, Rng, TensorData, SampleConfig,
} from "@alpha/core";
import { SeededRng, shapeSize, hashConfig, runId as makeRunId } from "@alpha/core";
import { Tape, DropoutRng } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries, countParams, clearForwardCache, type GPTParams } from "@alpha/model";
import {
  CusumDashboard, PopulationDynamicsController, KuramotoFusionController, SymbioMetricsCollector, SearchOrchestrator,
  defaultSymbioConfig, ffnDimForActivation, type SymbioConfig, type TrainerStepInfo,
  serializeGraph, nameGraph, type ActivationNode,
} from "@alpha/symbiogenesis";
import {
  DataLoader, ShardedDataLoader, loadText, loadAndTokenize, loadOrCacheTokens, getSplitByte,
  SftDataLoader, loadSftExamples, splitSftExamples,
  RcrUlDataLoader, loadRcrUlExamples, loadSftConversationSha256s, type BatchSource,
} from "./data.js";
import {
  FileCheckpoint,
  buildCheckpointState,
  releaseCheckpointSnapshotBuffers,
  restoreParams,
} from "./checkpoint.js";
import { sample as runSample } from "./sample.js";
import {
  ConsensusFusionShadow,
  carryOptimizerStateAcrossSwitch,
  initGPTWithTransferredWeights,
} from "./symbio-continuity.js";
import { Effect } from "effect";

// ── GPU stats ────────────────────────────────────────────────────────────

interface GpuStats {
  utilPct: number;
  vramUsedMb: number;
  vramTotalMb: number;
}

let _gpuStatsCache: GpuStats | null = null;
let _gpuStatsLastQuery = 0;
let _gpuStatsDisabled = false;
let _gpuStatsInFlight: Promise<void> | null = null;
const GPU_STATS_INTERVAL_MS = 5000; // query nvidia-smi at most every 5s

export function gpuVendorName(vendorId: number): string {
  return vendorId === 0x10de ? "NVIDIA"
    : vendorId === 0x1002 ? "AMD"
    : vendorId === 0x8086 ? "Intel"
    : `0x${vendorId.toString(16)}`;
}

export interface HeliosTrainingDeviceCapabilities {
  deviceName: string;
  vendorId: number;
  deviceType?: number;
  subgroupSize?: number;
  subgroupSupportedStages?: number;
  subgroupSupportedOperations?: number;
  computeSubgroupArithmeticSupported?: boolean;
}

export interface HeliosTrainingDeviceAssessment {
  supported: boolean;
  mode: "portable" | "unsupported";
  reasons: string[];
}

/**
 * Decide whether the current Helios kernel set can train on a physical device.
 *
 * This intentionally contains no vendor allow-list. The first portable kernel
 * generation uses subgroup arithmetic and still has several 32-lane layout
 * assumptions, so wave64 is reported precisely rather than being allowed to
 * corrupt training. AMD RDNA wave32 devices are admitted; CDNA wave64 will move
 * to supported as the wave-size-independent kernels / HIP lowering land.
 */
export function assessHeliosTrainingDevice(
  gpu: HeliosTrainingDeviceCapabilities,
): HeliosTrainingDeviceAssessment {
  const reasons: string[] = [];
  if (gpu.deviceType !== undefined && gpu.deviceType !== 1 && gpu.deviceType !== 2) {
    reasons.push(`device type ${gpu.deviceType} is not an integrated or discrete Vulkan GPU`);
  }
  const computeArithmetic = gpu.computeSubgroupArithmeticSupported ?? (
    ((gpu.subgroupSupportedStages ?? 0) & 0x00000020) !== 0 &&
    ((gpu.subgroupSupportedOperations ?? 0) & 0x00000004) !== 0
  );
  if (!computeArithmetic) {
    reasons.push("compute-stage subgroup arithmetic is unavailable");
  }
  if ((gpu.subgroupSize ?? 0) !== 32) {
    reasons.push(
      `native subgroup size ${gpu.subgroupSize ?? 0} is not yet supported by the current 32-lane kernel layouts`,
    );
  }
  return {
    supported: reasons.length === 0,
    mode: reasons.length === 0 ? "portable" : "unsupported",
    reasons,
  };
}

async function readGpuStatsOnce(): Promise<GpuStats | null> {
  try {
    const { execFile } = await import("node:child_process");
    const out = await new Promise<string>((resolve, reject) => {
      execFile(
        "nvidia-smi",
        ["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        { timeout: 2000, encoding: "utf-8" },
        (err, stdout) => {
          if (err) {
            reject(err);
            return;
          }
          resolve((stdout ?? "").trim());
        },
      );
    });
    if (!out) {
      return null;
    }
    const line = out.split(/\r?\n/, 1)[0];
    const [util, used, total] = line.split(",").map(s => parseFloat(s.trim()));
    if (isFinite(util) && isFinite(used) && isFinite(total)) {
      return { utilPct: util, vramUsedMb: used, vramTotalMb: total };
    }
  } catch { /* caller handles disable policy */ }
  return null;
}

async function queryGpuStats(blocking = false): Promise<GpuStats | null> {
  if (_gpuStatsDisabled) return null;

  const now = performance.now();
  // Rate-limit regardless of whether previous query succeeded.
  // When nvidia-smi is unavailable, this avoids repeated failing probes.
  if (now - _gpuStatsLastQuery < GPU_STATS_INTERVAL_MS) {
    return _gpuStatsCache;
  }
  _gpuStatsLastQuery = now;

  if (blocking) {
    const stats = await readGpuStatsOnce();
    if (stats) {
      _gpuStatsCache = stats;
      return _gpuStatsCache;
    }
    _gpuStatsDisabled = true;
    return null;
  }

  if (!_gpuStatsInFlight) {
    _gpuStatsInFlight = (async () => {
      const stats = await readGpuStatsOnce();
      if (stats) {
        _gpuStatsCache = stats;
        return;
      }
      _gpuStatsDisabled = true;
    })().finally(() => {
      _gpuStatsInFlight = null;
    });
  }
  return _gpuStatsCache;
}

function parseCliSampleOutput(stdout: string): string {
  const lines = stdout.split(/\r?\n/);
  const out: string[] = [];
  let inOutput = false;
  for (const raw of lines) {
    const line = raw ?? "";
    if (!inOutput) {
      if (line.trim() === "---") inOutput = true;
      continue;
    }
    if (line.trim() === "--- stats ---") break;
    if (line.trim().length === 0) continue;
    out.push(line);
  }
  return out.join("\n").trim() || "(no output captured)";
}

async function sampleFromCheckpointCli(
  checkpointPath: string,
  prompts: string[],
  cfg: SampleConfig,
): Promise<{ prompt: string; output: string }[]> {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const { execFile } = await import("node:child_process");
  const cliEntry = path.resolve(process.cwd(), "apps/cli/dist/main.js");
  const exeBase = path.basename(process.execPath).toLowerCase();
  const isCompiledAlpha = exeBase === "alpha" || exeBase.startsWith("alpha-");

  await fs.access(checkpointPath);
  if (!isCompiledAlpha) {
    await fs.access(cliEntry);
  }

  const runSampleOnce = (prompt: string): Promise<string> =>
    new Promise((resolve, reject) => {
      const args = isCompiledAlpha
        ? [
            "sample",
            `--checkpoint=${checkpointPath}`,
            "--backend=cpu_ref",
            "--slow",
            `--steps=${cfg.steps}`,
            `--temp=${cfg.temperature}`,
            `--topk=${cfg.topk}`,
            `--topp=${cfg.topp ?? 1}`,
            `--prompt=${prompt}`,
          ]
        : [
            cliEntry,
            "sample",
            `--checkpoint=${checkpointPath}`,
            "--backend=cpu_ref",
            "--slow",
            `--steps=${cfg.steps}`,
            `--temp=${cfg.temperature}`,
            `--topk=${cfg.topk}`,
            `--topp=${cfg.topp ?? 1}`,
            `--prompt=${prompt}`,
          ];
      execFile(
        process.execPath,
        args,
        { timeout: 300_000, maxBuffer: 8 * 1024 * 1024, encoding: "utf-8" },
        (err, stdout) => {
          if (err) {
            reject(err);
            return;
          }
          resolve(parseCliSampleOutput(stdout ?? ""));
        },
      );
    });

  const samples: { prompt: string; output: string }[] = [];
  for (const prompt of prompts) {
    const output = await runSampleOnce(prompt);
    samples.push({ prompt, output });
  }
  return samples;
}

// ── Step metrics ───────────────────────────────────────────────────────────

export interface StepMetrics {
  step: number;
  loss: number;
  valLoss?: number;
  lr: number;
  gradNorm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  /** Present only in the explicit RCR-UL experiment mode. */
  positive_ce_loss?: number;
  negative_ul_loss?: number;
  ul_weight?: number;
  positive_supervised_token_mass?: number;
  negative_penalty_position_mass?: number;
  negative_examples_with_penalty?: number;
  negative_first_penalty_target_position?: number;
  negative_last_penalty_target_position?: number;
  negative_mean_bad_token_probability?: number;
  negative_max_bad_token_probability?: number;
  grad_norm_before_clip?: number;
  grad_norm_after_clip?: number;
  nan_count?: number;
  inf_count?: number;
  timing_positive_fwd_ms?: number;
  timing_positive_bwd_ms?: number;
  timing_negative_fwd_ms?: number;
  timing_negative_bwd_ms?: number;
  gpu_util_pct?: number;
  gpu_vram_used_mb?: number;
  gpu_vram_total_mb?: number;
  gpu_mem_pool_mb?: number;
  gpu_live_allocs?: number;
  gpu_vk_memory_allocations?: number;
  gpu_slab_buffers?: number;
  gpu_temp_slab_count?: number;
  gpu_temp_slab_capacity_mb?: number;
  gpu_temp_slab_live_mb?: number;
  gpu_allocator_slab_fallbacks?: number;
  gpu_allocator_free_range_reuses?: number;
  gpu_allocator_free_range_overflows?: number;
  gpu_peak_live_allocs?: number;
  gpu_peak_mem_pool_mb?: number;
  host_rss_mb?: number;
  host_heap_used_mb?: number;
  host_external_mb?: number;
  host_array_buffers_mb?: number;
  // Phase 0 instrumentation: per-step timing breakdown
  timing_fwd_ms?: number;
  timing_bwd_ms?: number;
  timing_optim_ms?: number;
  timing_data_ms?: number;
  timing_flush_ms?: number;
  timing_grad_norm_ms?: number;
  timing_grad_clip_ms?: number;
  /** Pre-metrics step wall time not spent blocked on GPU completion. */
  timing_host_build_ms?: number;
  /** Wall time spent in synchronous GPU completion/timestamp calls. */
  timing_gpu_blocking_ms?: number;
  /** Step interval used by the direct host/GPU split. */
  timing_core_step_ms?: number;
  gpu_ops_count?: number;
  // Clipping telemetry
  clip_coef?: number;
  clip_pct?: number;
  // CUSUM (symbio)
  cusum_grad?: number;
  cusum_clip?: number;
  cusum_tps?: number;
  cusum_val?: number;
  cusum_alerts?: number;
  cusum_alert_reason?: string;
  // Symbio metrics
  weight_entropy?: number;
  effective_rank?: number;
  free_energy?: number;
  population_entropy?: number;
  activation_distribution?: string;
  mi_input_repr?: number;
  mi_repr_output?: number;
  mi_compression?: number;
  fitness_score?: number;
  complexity_score?: number;
  // Adaptive batch
  adaptive_batch_size?: number;
  batch_change_reason?: string;
  // Search candidate
  symbio_candidate_id?: string;
  symbio_candidate_name?: string;
  symbio_candidate_activation?: string;
  symbio_candidate_parent_id?: string;
  symbio_candidate_parent_name?: string;
  symbio_generation?: number;
  architecture_diversity?: number;
  symbio_activation_graph?: string;
  symbio_mutation_applied?: string;
  // Per-layer gradient norms (JSON: Record<layerIndex, norm>)
  per_layer_grad_norms?: string;
}

/** Validation is cadence-based, but the terminal model must always be evaluated. */
export function shouldEvaluateStep(step: number, evalInterval: number, totalIters: number): boolean {
  return evalInterval > 0 && (step % evalInterval === 0 || step === totalIters);
}

/** Add a separately computed terminal validation loss without changing prior metric bytes. */
export function repairTerminalValidationMetric(
  metricsText: string,
  totalIters: number,
  valLoss: number,
): string {
  if (!Number.isFinite(valLoss)) throw new Error("terminal validation loss is not finite");
  if (!metricsText.endsWith("\n")) throw new Error("metrics JSONL must end with a newline");
  const lines = metricsText.slice(0, -1).split("\n");
  if (lines.length !== totalIters) {
    throw new Error(`terminal metric repair expected ${totalIters} rows, found ${lines.length}`);
  }
  const rows = lines.map((line, index) => {
    const row = JSON.parse(line) as { step?: number; valLoss?: number };
    if (row.step !== index + 1) {
      throw new Error(`terminal metric repair expected step ${index + 1}, found ${String(row.step)}`);
    }
    return row;
  });
  const last = rows.at(-1)!;
  if (last.valLoss !== undefined) throw new Error("terminal metric row already has validation loss");
  lines[lines.length - 1] = JSON.stringify({ ...last, valLoss });
  return `${lines.join("\n")}\n`;
}

export type TrainingEventLevel = "debug" | "info" | "warn" | "error";

export interface TrainingEvent {
  step?: number;
  level?: TrainingEventLevel;
  kind: string;
  message: string;
  payload?: Record<string, unknown> | null;
  timestamp?: string;
}

export interface SampleTrendStatus {
  summary: string;
  plateauLikely: boolean;
  overfittingLikely: boolean;
  windowStartStep: number;
  windowEndStep: number;
  trainLossDelta: number;
  valLossDelta: number;
  gapDelta: number;
}

interface EvalSnapshot {
  step: number;
  trainLoss: number;
  valLoss: number;
}

interface ValBucketLoader {
  name: "short" | "medium" | "long";
  loader: DataLoader;
  sampleCount: number;
  tokenCount: number;
}

interface SampleQualityStats {
  avgTokenLength: number;
  uniqueTokenRatio: number;
  repeated3GramRate: number;
  repeated4GramRate: number;
}

function clamp01(x: number, lo = 0.01, hi = 0.5): number {
  if (!Number.isFinite(x)) return lo;
  return Math.min(hi, Math.max(lo, x));
}

function fnv1a32(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function splitByDelimiterDeterministic(
  rawText: string,
  delimiter: string,
  valFraction: number,
  seed: number,
): { trainText: string; valText: string; trainCount: number; valCount: number } | null {
  if (!rawText.includes(delimiter)) return null;

  const chunks = rawText.split(delimiter);
  const trainParts: string[] = [];
  const valParts: string[] = [];
  let trainCount = 0;
  let valCount = 0;

  for (let i = 0; i < chunks.length; i++) {
    const body = chunks[i].trim();
    if (body.length === 0) continue;
    const sample = `${body}${delimiter}\n`;
    const hash = fnv1a32(`${seed}:${i}:${body.slice(0, 64)}`);
    const takeVal = (hash / 0x1_0000_0000) < valFraction;
    if (takeVal) {
      valParts.push(sample);
      valCount++;
    } else {
      trainParts.push(sample);
      trainCount++;
    }
  }

  if (trainCount === 0 || valCount === 0) return null;

  return {
    trainText: trainParts.join(""),
    valText: valParts.join(""),
    trainCount,
    valCount,
  };
}

function buildValBucketLoaders(
  valText: string,
  delimiter: string,
  tokenizer: Tokenizer,
  batchSize: number,
  blockSize: number,
  seed: number,
): ValBucketLoader[] {
  const rawSamples = valText
    .split(delimiter)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (rawSamples.length < 32) return [];

  const lengths = rawSamples.map((s) => s.length).sort((a, b) => a - b);
  const p33 = lengths[Math.floor((lengths.length - 1) * 0.33)];
  const p66 = lengths[Math.floor((lengths.length - 1) * 0.66)];

  const short: string[] = [];
  const medium: string[] = [];
  const long: string[] = [];
  for (const sample of rawSamples) {
    const wrapped = `${sample}${delimiter}\n`;
    if (sample.length <= p33) short.push(wrapped);
    else if (sample.length <= p66) medium.push(wrapped);
    else long.push(wrapped);
  }

  const buckets: Array<{ name: ValBucketLoader["name"]; samples: string[] }> = [
    { name: "short", samples: short },
    { name: "medium", samples: medium },
    { name: "long", samples: long },
  ];

  const loaders: ValBucketLoader[] = [];
  for (let i = 0; i < buckets.length; i++) {
    const bucket = buckets[i];
    if (bucket.samples.length < 8) continue;
    const text = bucket.samples.join("");
    const tokens = tokenizer.encode(text);
    if (tokens.length <= blockSize + 1) continue;
    loaders.push({
      name: bucket.name,
      loader: new DataLoader(tokens, new SeededRng(seed + 1000 + i), batchSize, blockSize),
      sampleCount: bucket.samples.length,
      tokenCount: tokens.length,
    });
  }

  return loaders;
}

function analyzeSampleTrend(history: EvalSnapshot[]): SampleTrendStatus | null {
  const WINDOW = 6;
  if (history.length < WINDOW) return null;
  const window = history.slice(-WINDOW);
  const first = window[0];
  const last = window[window.length - 1];
  const trainLossDelta = last.trainLoss - first.trainLoss;
  const valLossDelta = last.valLoss - first.valLoss;
  const firstGap = first.valLoss - first.trainLoss;
  const lastGap = last.valLoss - last.trainLoss;
  const gapDelta = lastGap - firstGap;
  const trainRel = (first.trainLoss - last.trainLoss) / Math.max(1e-6, Math.abs(first.trainLoss));
  const valRel = (first.valLoss - last.valLoss) / Math.max(1e-6, Math.abs(first.valLoss));

  // Plateau: both train and validation have moved by <1% over the recent window.
  const plateauLikely = Math.abs(trainRel) < 0.01 && Math.abs(valRel) < 0.01;
  // Overfitting: train improves while validation degrades and gap widens.
  const overfittingLikely = trainRel > 0.01 && valRel < -0.003 && gapDelta > 0.03;

  let summary = "trend=unclear";
  if (overfittingLikely) {
    summary = `overfitting_likely (train ${trainLossDelta.toFixed(4)}, val ${valLossDelta.toFixed(4)}, gap +${gapDelta.toFixed(4)})`;
  } else if (plateauLikely) {
    summary = `plateau_likely (train ${trainLossDelta.toFixed(4)}, val ${valLossDelta.toFixed(4)})`;
  } else if (trainLossDelta < 0 && valLossDelta < 0) {
    summary = `improving (train ${trainLossDelta.toFixed(4)}, val ${valLossDelta.toFixed(4)})`;
  } else if (trainLossDelta < 0 && valLossDelta >= 0) {
    summary = `possible_generalization_drift (train ${trainLossDelta.toFixed(4)}, val +${valLossDelta.toFixed(4)})`;
  } else {
    summary = `unstable_or_regressing (train ${trainLossDelta.toFixed(4)}, val ${valLossDelta.toFixed(4)})`;
  }

  return {
    summary,
    plateauLikely,
    overfittingLikely,
    windowStartStep: first.step,
    windowEndStep: last.step,
    trainLossDelta,
    valLossDelta,
    gapDelta,
  };
}

function computeRepeatedNGramRate(tokenSeqs: string[][], n: number): number {
  let total = 0;
  let repeated = 0;
  for (const tokens of tokenSeqs) {
    if (tokens.length < n) continue;
    const seen = new Map<string, number>();
    for (let i = 0; i <= tokens.length - n; i++) {
      const key = tokens.slice(i, i + n).join("\u0001");
      const next = (seen.get(key) ?? 0) + 1;
      seen.set(key, next);
      total++;
      if (next > 1) repeated++;
    }
  }
  if (total <= 0) return 0;
  return repeated / total;
}

function computeSampleQualityStats(samples: Array<{ prompt: string; output: string }>): SampleQualityStats | null {
  if (samples.length === 0) return null;
  const tokenSeqs = samples.map((s) =>
    s.output
      .toLowerCase()
      .split(/\s+/)
      .map((t) => t.trim())
      .filter((t) => t.length > 0),
  );
  const flat = tokenSeqs.flat();
  if (flat.length === 0) {
    return {
      avgTokenLength: 0,
      uniqueTokenRatio: 0,
      repeated3GramRate: 0,
      repeated4GramRate: 0,
    };
  }
  const unique = new Set(flat);
  const totalChars = flat.reduce((acc, t) => acc + t.length, 0);
  return {
    avgTokenLength: totalChars / flat.length,
    uniqueTokenRatio: unique.size / flat.length,
    repeated3GramRate: computeRepeatedNGramRate(tokenSeqs, 3),
    repeated4GramRate: computeRepeatedNGramRate(tokenSeqs, 4),
  };
}

function computeTensorNormSqCpu(data: TensorData): number {
  const arr = data.data as Float32Array;
  let sumSq = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    sumSq += v * v;
  }
  return sumSq;
}

export function validateCheckpointModelCompatibility(
  resumePath: string,
  checkpointModel: ModelConfig | undefined,
  activeModel: ModelConfig,
): void {
  if (!checkpointModel) return;
  const diffs: string[] = [];
  const check = (key: keyof ModelConfig, label = key): void => {
    const a = checkpointModel[key];
    const b = activeModel[key];
    if (a !== b) diffs.push(`${label}: checkpoint=${String(a)} current=${String(b)}`);
  };

  check("vocabSize");
  check("blockSize");
  check("nLayer");
  check("nEmbd");
  check("nHead");
  check("ffnActivation");
  check("ffnDim");
  check("dropout");
  check("softCap");
  check("normType");
  check("posEnc");
  check("ropeTheta");
  check("tieEmbeddings");

  if (diffs.length === 0) return;

  const allow = (process.env.ALPHA_ALLOW_RESUME_MISMATCH ?? "0").trim() === "1";
  const msg =
    `Checkpoint/model mismatch for ${resumePath}:\n` +
    diffs.map((d) => `  - ${d}`).join("\n") +
    `\nRefusing to resume to prevent silent divergence. ` +
    `Use ALPHA_ALLOW_RESUME_MISMATCH=1 only for intentional migration experiments.`;

  if (!allow) throw new Error(msg);
  console.warn(`[warn] ${msg}`);
}

// ── Trainer ────────────────────────────────────────────────────────────────

export interface TrainerDeps {
  backend: Backend;
  tokenizer: Tokenizer;
  optimizer: Optimizer;
  rng: Rng;
  modelConfig: ModelConfig;
  trainConfig: TrainConfig;
  dataPath: string;
  /** Independently cached pretraining shards, logically concatenated by the loader. */
  dataPaths?: readonly string[];
  valDataPath?: string;
  runDir?: string;
  resumePath?: string;
  /** Restore model weights only, while starting a fresh optimizer/schedule at step zero (e.g. SFT). */
  initCheckpointPath?: string;
  /** Override activation graph for symbio resume (when checkpoint doesn't contain it). */
  resumeActivationGraph?: ActivationNode;
  tokenizerArtifacts?: import("@alpha/core").TokenizerArtifacts;
  onStep?: (metrics: StepMetrics) => void;
  onStart?: (info: {
    runId: string; configHash: string; totalParams: number; dataPath: string;
    infra?: { gpuName: string; gpuVendor: string; gpuVramMb: number; hostname: string; cpuCount: number; ramTotalMb: number; osPlatform: string };
  }) => void | Promise<void>;
  onCheckpoint?: (info: { step: number; path: string; runId: string }) => void;
  onSamples?: (samples: { prompt: string; output: string }[], step: number, trend?: SampleTrendStatus) => void | Promise<void>;
  onEvent?: (event: TrainingEvent) => void | Promise<void>;
  samplePrompts?: string[];
  domain?: string;
  activationCheckpointing?: boolean;
  mixedPrecision?: boolean;
  /**
   * Assistant-only SFT mode. When true, the training corpus is read as one
   * rendered conversation per line and each micro-batch carries a per-position
   * lossMask (user tokens + padding weight 0) threaded into the masked-CE loss.
   */
  sft?: boolean;
  /** Explicit rollout-conditioned repetition-unlikelihood branch. The JSONL
   * cohort must be row-matched to the positive SFT corpus. A weight of zero is
   * valid and still executes forward+backward for the matched control arm. */
  rcrUl?: {
    dataPath: string;
    weight: number;
    epsilon: number;
  };
}

export async function train(deps: TrainerDeps): Promise<{ params: GPTParams; modelConfig: ModelConfig }> {
  const {
    backend, tokenizer, optimizer, rng, modelConfig, trainConfig,
    dataPath, valDataPath, resumePath, initCheckpointPath, onStep, onStart, onEvent,
  } = deps;
  if (resumePath && initCheckpointPath) throw new Error("resumePath and initCheckpointPath are mutually exclusive");
  const dataPaths = deps.dataPaths?.length ? [...deps.dataPaths] : [dataPath];
  if (deps.sft && dataPaths.length > 1) throw new Error("SFT does not support a pretraining shard manifest");
  if (deps.rcrUl) {
    if (!deps.sft) throw new Error("RCR-UL requires assistant-only SFT mode");
    if (!deps.rcrUl.dataPath) throw new Error("RCR-UL requires a frozen rollout dataPath");
    if (!Number.isFinite(deps.rcrUl.weight) || deps.rcrUl.weight < 0) {
      throw new Error(`RCR-UL weight must be finite and non-negative: ${deps.rcrUl.weight}`);
    }
    if (!(deps.rcrUl.epsilon > 0 && deps.rcrUl.epsilon <= 1e-6)) {
      throw new Error(`RCR-UL epsilon must be in (0,1e-6]: ${deps.rcrUl.epsilon}`);
    }
  }

  const dataTag = dataPath.split("/").pop()?.replace(/\.[^.]+$/, "").replace(/-/g, "_");
  // When resuming into an existing runDir, reuse the directory name as run ID
  // so that remote reporting appends to the same dashboard entry.
  const rid = deps.runDir ? deps.runDir.split("/").pop()! : makeRunId(dataTag);
  const configHash = hashConfig({
    ...modelConfig,
    ...trainConfig,
    ...(deps.rcrUl ? {
      rcrUlDataPath: deps.rcrUl.dataPath,
      rcrUlWeight: deps.rcrUl.weight,
      rcrUlEpsilon: deps.rcrUl.epsilon,
    } : {}),
  } as any);
  const emitEvent = (event: TrainingEvent): void => {
    if (!onEvent) return;
    const normalized: TrainingEvent = {
      ...event,
      level: event.level ?? "info",
      timestamp: event.timestamp ?? new Date().toISOString(),
    };
    try {
      const maybePromise = onEvent(normalized);
      if (maybePromise && typeof (maybePromise as Promise<void>).then === "function") {
        void (maybePromise as Promise<void>).then(undefined, (e: unknown) => {
          console.warn(`  [remote] event emit failed: ${(e as Error).message}`);
        });
      }
    } catch (e) {
      console.warn(`  [remote] event emit failed: ${(e as Error).message}`);
    }
  };

  // Set up run directory
  const runDir = deps.runDir ?? `runs/${rid}`;
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  await fs.mkdir(runDir, { recursive: true });
  const configPath = path.join(runDir, "config.json");
  let inheritedInitCheckpointPath: string | undefined;
  if (resumePath) {
    try {
      const existingConfig = JSON.parse(await fs.readFile(configPath, "utf8")) as {
        initCheckpointPath?: unknown;
        rcrUl?: unknown;
      };
      if (existingConfig.initCheckpointPath !== undefined) {
        if (typeof existingConfig.initCheckpointPath !== "string" || existingConfig.initCheckpointPath.length === 0) {
          throw new Error("existing run config has an invalid initCheckpointPath");
        }
        inheritedInitCheckpointPath = existingConfig.initCheckpointPath;
      }
      const existingRcrUl = existingConfig.rcrUl;
      if (existingRcrUl !== undefined || deps.rcrUl !== undefined) {
        if (
          typeof existingRcrUl !== "object" || existingRcrUl === null ||
          typeof (existingRcrUl as Record<string, unknown>).dataPath !== "string" ||
          typeof (existingRcrUl as Record<string, unknown>).weight !== "number" ||
          typeof (existingRcrUl as Record<string, unknown>).epsilon !== "number"
        ) {
          throw new Error("RCR-UL resume requires the existing run config to preserve a valid rcrUl contract");
        }
        if (!deps.rcrUl) {
          throw new Error("RCR-UL resume cannot disable the existing run's paired objective");
        }
        const prior = existingRcrUl as { dataPath: string; weight: number; epsilon: number };
        if (
          prior.dataPath !== deps.rcrUl.dataPath ||
          prior.weight !== deps.rcrUl.weight ||
          prior.epsilon !== deps.rcrUl.epsilon
        ) {
          throw new Error(
            "RCR-UL resume contract mismatch: dataPath, weight, and epsilon must exactly match the existing run",
          );
        }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    if (deps.sft && !inheritedInitCheckpointPath) {
      throw new Error("SFT resume requires the existing run config to preserve initCheckpointPath provenance");
    }
  }
  const configObj: Record<string, unknown> = { modelConfig, trainConfig, configHash, runId: rid };
  if (dataPaths.length > 1) configObj.dataPaths = dataPaths;
  if (initCheckpointPath) configObj.initCheckpointPath = initCheckpointPath;
  else if (inheritedInitCheckpointPath) configObj.initCheckpointPath = inheritedInitCheckpointPath;
  if (resumePath) configObj.resumePath = resumePath;
  if (deps.domain) configObj.domain = deps.domain;
  if (deps.rcrUl) configObj.rcrUl = { ...deps.rcrUl };
  async function writeConfig(): Promise<void> {
    const tmp = configPath + ".tmp";
    await fs.writeFile(tmp, JSON.stringify(configObj, null, 2));
    await fs.rename(tmp, configPath);
  }
  await writeConfig();

  // Buffered metrics log — flushes every 50 steps, on checkpoint, and on exit
  const metricsPath = path.join(runDir, "metrics.jsonl");
  const metricsHandle = await fs.open(metricsPath, "a");
  let metricsBuffer: string[] = [];
  async function flushMetrics(): Promise<void> {
    if (metricsBuffer.length === 0) return;
    const chunk = metricsBuffer.join("");
    metricsBuffer = [];
    await metricsHandle.write(chunk);
  }

  const failOnSmokeTestRaw = process.env.ALPHA_FAIL_ON_SMOKE_TEST?.trim().toLowerCase();
  const failOnSmokeTest =
    failOnSmokeTestRaw === "1" ||
    failOnSmokeTestRaw === "true" ||
    failOnSmokeTestRaw === "yes" ||
    failOnSmokeTestRaw === "on";
  let smokePreflight: { verified: boolean; throughputGBps: number; reason?: string } | null = null;
  if ("smokeTest" in backend) {
    try {
      const smoke = (backend as any).smokeTest();
      smokePreflight = {
        verified: !!smoke.verified,
        throughputGBps: Number.isFinite(smoke.throughputGBps) ? smoke.throughputGBps : 0,
      };
      if (!smokePreflight.verified) smokePreflight.reason = "verification mismatch";
    } catch (e: any) {
      smokePreflight = {
        verified: false,
        throughputGBps: 0,
        reason: e?.message ? String(e.message) : "exception",
      };
    }
    if (!smokePreflight.verified && failOnSmokeTest) {
      throw new Error(`GPU smoke test failed (${smokePreflight.reason || "unknown"}) and ALPHA_FAIL_ON_SMOKE_TEST=1`);
    }
  }

  // Load data — use chunked tokenization for large files to avoid V8 string limit
  const sftMode = !!deps.sft;
  const tokenizerCacheIdentity = deps.tokenizerArtifacts
    ? (await import("node:crypto")).createHash("sha256").update(JSON.stringify(deps.tokenizerArtifacts)).digest("hex")
    : undefined;
  const fileStat = await fs.stat(dataPath);
  const isLargeFile = fileStat.size > 50 * 1024 * 1024; // >50MB — chunked tokenization avoids V8 array size limits
  let trainTokens: Int32Array = new Int32Array(0);
  let valTokens: Int32Array = new Int32Array(0);
  let trainTokenShards: Int32Array[] = [];
  let valTokenShards: Int32Array[] = [];
  let sftTrain: SftDataLoader | null = null;
  let sftVal: SftDataLoader | null = null;
  let sftShuffle = false;
  let rcrUlTrain: RcrUlDataLoader | null = null;
  let valBucketLoaders: ValBucketLoader[] = [];
  const valBucketEvalEnabledRaw = (process.env.ALPHA_VAL_BUCKET_EVAL ?? "0").trim().toLowerCase();
  const valBucketEvalEnabled = (
    valBucketEvalEnabledRaw === "1" ||
    valBucketEvalEnabledRaw === "true" ||
    valBucketEvalEnabledRaw === "yes" ||
    valBucketEvalEnabledRaw === "on"
  );
  const valBucketEvalItersRaw = Number.parseInt(process.env.ALPHA_VAL_BUCKET_EVAL_ITERS ?? "2", 10);
  const valBucketEvalIters = Number.isFinite(valBucketEvalItersRaw) && valBucketEvalItersRaw > 0
    ? valBucketEvalItersRaw
    : 2;

  if (sftMode) {
    // Assistant-only SFT: one rendered conversation per line → per-example
    // lossMask. Deterministic doc-aware train/val split when no valData given.
    console.log(`SFT mode: loading conversations from ${dataPath.split("/").pop()} (assistant-only masked loss)...`);
    const sftShuffleRaw = (process.env.ALPHA_SFT_SHUFFLE ?? "0").trim().toLowerCase();
    sftShuffle = sftShuffleRaw === "1" || sftShuffleRaw === "true" || sftShuffleRaw === "yes" || sftShuffleRaw === "on";
    const sftBalanceRaw = (process.env.ALPHA_SFT_BALANCE_CONVERSATIONS ?? "0").trim().toLowerCase();
    const sftBalance = sftBalanceRaw === "1" || sftBalanceRaw === "true" || sftBalanceRaw === "yes" || sftBalanceRaw === "on";
    const sftStartTokens = Math.max(0, Number.parseInt(process.env.ALPHA_SFT_START_TOKENS ?? "0", 10) || 0);
    const parsedStartWeight = Number.parseFloat(process.env.ALPHA_SFT_START_WEIGHT ?? "1");
    const sftStartWeight = Number.isFinite(parsedStartWeight) && parsedStartWeight > 0 ? parsedStartWeight : 1;
    const parsedEndWeight = Number.parseFloat(process.env.ALPHA_SFT_END_WEIGHT ?? "1");
    const sftEndWeight = Number.isFinite(parsedEndWeight) && parsedEndWeight > 0 ? parsedEndWeight : 1;
    const commonSftOptions = {
      balanceConversations: sftBalance,
      startTokenCount: sftStartTokens,
      startTokenMultiplier: sftStartWeight,
      endTokenMultiplier: sftEndWeight,
    };
    console.log(
      `SFT policy: shuffle=${sftShuffle ? `epoch(seed=${trainConfig.seed})` : "off"} ` +
      `conversationBalance=${sftBalance} startTokens=${sftStartTokens} ` +
      `startWeight=${sftStartWeight} endWeight=${sftEndWeight}`,
    );
    if (valDataPath) {
      const trainEx = await loadSftExamples(dataPath, tokenizer);
      const valEx = await loadSftExamples(valDataPath, tokenizer);
      sftTrain = new SftDataLoader(trainEx, trainConfig.batchSize, modelConfig.blockSize, {
        ...commonSftOptions,
        shuffleSeed: sftShuffle ? trainConfig.seed : undefined,
      });
      sftVal = valEx.length > 0
        ? new SftDataLoader(valEx, trainConfig.batchSize, modelConfig.blockSize, commonSftOptions)
        : null;
    } else {
      const allEx = await loadSftExamples(dataPath, tokenizer);
      const valFraction = clamp01(Number.parseFloat(process.env.ALPHA_SFT_VAL_FRACTION ?? "0.05"), 0, 0.5);
      const { train, val } = splitSftExamples(allEx, valFraction, trainConfig.seed);
      sftTrain = new SftDataLoader(train.length > 0 ? train : allEx, trainConfig.batchSize, modelConfig.blockSize, {
        ...commonSftOptions,
        shuffleSeed: sftShuffle ? trainConfig.seed : undefined,
      });
      sftVal = val.length > 0
        ? new SftDataLoader(val, trainConfig.batchSize, modelConfig.blockSize, commonSftOptions)
        : null;
    }
    console.log(
      `SFT conversations: train=${sftTrain.conversationCount} val=${sftVal?.conversationCount ?? 0} ` +
      `(${sftTrain.length.toLocaleString()} train tokens)`,
    );
    if (deps.rcrUl) {
      const negativeExamples = await loadRcrUlExamples(deps.rcrUl.dataPath);
      const positiveConversationSha256s = await loadSftConversationSha256s(dataPath);
      if (positiveConversationSha256s.length !== sftTrain.conversationCount) {
        throw new Error(
          `RCR-UL requires an unsplit positive cohort: ${positiveConversationSha256s.length} file rows != ` +
          `${sftTrain.conversationCount} SFT training rows. Set ALPHA_SFT_VAL_FRACTION=0 or provide a separate --valData file.`,
        );
      }
      if (negativeExamples.length !== positiveConversationSha256s.length) {
        throw new Error(
          `RCR-UL/positive cohort mismatch: ${negativeExamples.length} negative rows != ` +
          `${positiveConversationSha256s.length} positive rows`,
        );
      }
      for (let index = 0; index < negativeExamples.length; index++) {
        const expected = positiveConversationSha256s[index];
        const actual = negativeExamples[index].positiveConversationSha256;
        if (actual !== expected) {
          throw new Error(
            `RCR-UL row ${index + 1} is paired with ${actual}, but positive conversation hashes to ${expected}`,
          );
        }
      }
      rcrUlTrain = new RcrUlDataLoader(
        negativeExamples,
        trainConfig.batchSize,
        modelConfig.blockSize,
        sftShuffle ? trainConfig.seed : undefined,
      );
      if (rcrUlTrain.conversationCount !== sftTrain.conversationCount) {
        throw new Error(
          `RCR-UL/positive cohort mismatch: ${rcrUlTrain.conversationCount} negative rows != ` +
          `${sftTrain.conversationCount} positive rows`,
        );
      }
      if (rcrUlTrain.penaltyPositionCount === 0) {
        throw new Error("RCR-UL cohort has zero penalty positions");
      }
      console.log(
        `RCR-UL mode: rows=${rcrUlTrain.conversationCount} ` +
        `penalty_positions=${rcrUlTrain.penaltyPositionCount} ` +
        `weight=${deps.rcrUl.weight} epsilon=${deps.rcrUl.epsilon}`,
      );
    }
  } else if (dataPaths.length > 1) {
    const validationPolicy = valDataPath ? "external held-out validation" : "independent 90/10 splits";
    console.log(`Sharded dataset: ${dataPaths.length} files — ${validationPolicy}...`);
    for (const [index, shardPath] of dataPaths.entries()) {
      const shardStat = await fs.stat(shardPath);
      console.log(
        `  shard ${index + 1}/${dataPaths.length}: ${shardPath.split("/").pop()} ` +
        `(${(shardStat.size / 1024 / 1024).toFixed(0)}MB)`,
      );
      if (valDataPath) {
        trainTokenShards.push(await loadOrCacheTokens(
          shardPath, tokenizer, undefined, tokenizerCacheIdentity,
        ));
      } else {
        const splitByte = await getSplitByte(shardPath, 0.9);
        trainTokenShards.push(await loadOrCacheTokens(
          shardPath, tokenizer, { startByte: 0, endByte: splitByte }, tokenizerCacheIdentity,
        ));
        valTokenShards.push(await loadOrCacheTokens(
          shardPath, tokenizer, { startByte: splitByte, endByte: shardStat.size }, tokenizerCacheIdentity,
        ));
      }
    }
    if (valDataPath) {
      console.log(`  validation: ${valDataPath.split("/").pop()} (wholly held out)`);
      valTokenShards.push(await loadOrCacheTokens(
        valDataPath, tokenizer, undefined, tokenizerCacheIdentity,
      ));
    }
  } else if (isLargeFile) {
    console.log(`Large dataset (${(fileStat.size / 1024 / 1024).toFixed(0)}MB) — using chunked tokenization...`);
    if (valDataPath) {
      trainTokens = await loadOrCacheTokens(dataPath, tokenizer, undefined, tokenizerCacheIdentity);
      valTokens = await loadOrCacheTokens(valDataPath, tokenizer, undefined, tokenizerCacheIdentity);
    } else {
      const splitByte = await getSplitByte(dataPath, 0.9);
      trainTokens = await loadOrCacheTokens(
        dataPath, tokenizer, { startByte: 0, endByte: splitByte }, tokenizerCacheIdentity,
      );
      valTokens = await loadOrCacheTokens(
        dataPath, tokenizer, { startByte: splitByte, endByte: fileStat.size }, tokenizerCacheIdentity,
      );
    }
  } else {
    const rawText = await loadText(dataPath);
    if (valDataPath) {
      const valRaw = await loadText(valDataPath);
      trainTokens = tokenizer.encode(rawText);
      valTokens = tokenizer.encode(valRaw);
    } else {
      const splitMode = (process.env.ALPHA_TEXT_SPLIT_MODE ?? "auto").toLowerCase();
      const splitFraction = clamp01(
        Number.parseFloat(process.env.ALPHA_TEXT_SPLIT_VAL_FRACTION ?? "0.1"),
      );
      const canUseDelimiterSplit = splitMode !== "contiguous";
      const delimiter = process.env.ALPHA_TEXT_SPLIT_DELIMITER ?? "<|end_of_text|>";
      const byDelimiter = canUseDelimiterSplit
        ? splitByDelimiterDeterministic(rawText, delimiter, splitFraction, trainConfig.seed)
        : null;

      if (byDelimiter) {
        console.log(
          `Data split: delimiter-aware (${delimiter}) train=${byDelimiter.trainCount} val=${byDelimiter.valCount} ` +
          `(~${((byDelimiter.valCount / (byDelimiter.trainCount + byDelimiter.valCount)) * 100).toFixed(1)}% val)`,
        );
        trainTokens = tokenizer.encode(byDelimiter.trainText);
        valTokens = tokenizer.encode(byDelimiter.valText);
        if (valBucketEvalEnabled) {
          valBucketLoaders = buildValBucketLoaders(
            byDelimiter.valText,
            delimiter,
            tokenizer,
            trainConfig.batchSize,
            modelConfig.blockSize,
            trainConfig.seed,
          );
          if (valBucketLoaders.length > 0) {
            const bucketInfo = valBucketLoaders
              .map((b) => `${b.name}:${b.sampleCount} samples/${b.tokenCount} tok`)
              .join(" | ");
            console.log(`Val buckets: enabled (${bucketInfo}), evalIters=${valBucketEvalIters}`);
          } else {
            console.log("Val buckets: enabled but insufficient data per bucket; skipping");
          }
        }
      } else {
        const splitIdx = Math.floor(rawText.length * (1 - splitFraction));
        console.log(`Data split: contiguous (${((splitFraction) * 100).toFixed(1)}% tail as val)`);
        trainTokens = tokenizer.encode(rawText.slice(0, splitIdx));
        valTokens = tokenizer.encode(rawText.slice(splitIdx));
      }
    }
  }

  if (!sftMode && trainTokenShards.length === 0) {
    trainTokenShards = [trainTokens];
    if (valTokens.length > 0) valTokenShards = [valTokens];
  }

  configObj.dataStats = sftMode
    ? {
        mode: "sft",
        trainTokens: sftTrain?.length ?? 0,
        valTokens: sftVal?.length ?? 0,
        trainConversations: sftTrain?.conversationCount ?? 0,
        valConversations: sftVal?.conversationCount ?? 0,
      }
    : {
        mode: "pretrain",
        trainTokens: trainTokenShards.reduce((sum, shard) => sum + shard.length, 0),
        valTokens: valTokenShards.reduce((sum, shard) => sum + shard.length, 0),
        shards: trainTokenShards.map((shard, index) => ({
          path: dataPaths[index],
          trainTokens: shard.length,
          valTokens: valDataPath ? 0 : (valTokenShards[index]?.length ?? 0),
        })),
        ...(valDataPath ? {
          validation: {
            path: valDataPath,
            valTokens: valTokenShards[0]?.length ?? 0,
            whollyHeldOut: true,
          },
        } : {}),
      };

  // Initialize model
  rng.seed(trainConfig.seed);
  let params = initGPT(modelConfig, backend, rng as any);
  let totalParams = countParams(params);
  configObj.totalParams = totalParams;
  await writeConfig();

  // Resume from checkpoint
  let startStep = 0;
  let resumedActivationGraph: ActivationNode | undefined = deps.resumeActivationGraph;
  if (initCheckpointPath) {
    const checkpoint = new FileCheckpoint();
    const state = await Effect.runPromise(checkpoint.load(initCheckpointPath));
    validateCheckpointModelCompatibility(initCheckpointPath, state.modelConfig as ModelConfig | undefined, modelConfig);
    restoreParams(params, state.params);
    // Weight initialization consumed model-shape-dependent draws. A fresh
    // fine-tune must begin from its declared seed, not that incidental offset.
    rng.seed(trainConfig.seed);
    console.log(`Initialized model weights from ${initCheckpointPath}; optimizer/schedule start fresh at step 0`);
    emitEvent({
      step: 0,
      level: "info",
      kind: "run_initialized_from_checkpoint",
      message: "initialized model weights from checkpoint with fresh optimizer",
      payload: { initCheckpointPath, sourceStep: state.step },
    });
  } else if (resumePath) {
    const checkpoint = new FileCheckpoint();
    const state = await Effect.runPromise(checkpoint.load(resumePath));
    validateCheckpointModelCompatibility(resumePath, state.modelConfig as ModelConfig | undefined, modelConfig);
    restoreParams(params, state.params);
    optimizer.loadStateDict(state.optimizerState);
    rng.setState(state.rngState);
    startStep = state.step;
    // CLI override takes precedence over checkpoint-embedded graph
    if (!resumedActivationGraph) {
      resumedActivationGraph = (state as any).symbioActivationGraph;
    }
    console.log(`Resumed from step ${startStep}${resumedActivationGraph ? ` (activation: ${nameGraph(resumedActivationGraph)})` : ""}`);
    emitEvent({
      step: startStep,
      level: "info",
      kind: "run_resumed",
      message: `resumed training from checkpoint step ${startStep}`,
      payload: {
        resumePath,
        activationGraph: resumedActivationGraph ? nameGraph(resumedActivationGraph) : null,
      },
    });
  }

  // Collect infrastructure metadata (GPU runs only)
  let infra: { gpuName: string; gpuVendor: string; gpuVramMb: number; hostname: string; cpuCount: number; ramTotalMb: number; osPlatform: string } | undefined;
  if ("getDeviceInfo" in backend) {
    try {
      const os = await import("node:os");
      const gpu = (backend as any).getDeviceInfo();
      const vendorName = gpuVendorName(gpu.vendorId);
      if (gpu.vendorId !== 0x10de) {
        // nvidia-smi is NVIDIA-only; disable probe path on other vendors.
        _gpuStatsDisabled = true;
      }
      const gpuStats = await queryGpuStats(true);
      infra = {
        gpuName: gpu.deviceName,
        gpuVendor: vendorName,
        gpuVramMb: gpuStats?.vramTotalMb ?? 0,
        hostname: os.hostname(),
        cpuCount: os.cpus().length,
        ramTotalMb: Math.round(os.totalmem() / 1024 / 1024),
        osPlatform: os.platform(),
      };
    } catch { /* infra collection is best-effort */ }
  }

  // Notify start (non-blocking): remote ingest outages should never stall training startup.
  if (onStart) {
    try {
      const maybePromise = onStart({ runId: rid, configHash, totalParams, dataPath, infra });
      if (maybePromise && typeof (maybePromise as Promise<void>).then === "function") {
        void (maybePromise as Promise<void>).then(undefined, (e: unknown) => {
          console.warn(`  [remote] run_start failed: ${(e as Error).message}`);
        });
      }
    } catch (e) {
      console.warn(`  [remote] run_start failed: ${(e as Error).message}`);
    }
  }

  // Log header
  const paramBytes = totalParams * 4;
  console.log(`── alpha training ──`);
  console.log(`run_id: ${rid}`);
  console.log(`config_hash: ${configHash}`);
  console.log(`params: ${totalParams.toLocaleString()} (${(paramBytes / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`backend: ${backend.name} | tokenizer: ${tokenizer.name} | optimizer: ${optimizer.name}`);
  const effectiveBatch = trainConfig.batchSize * trainConfig.gradAccumSteps;
  const accumStr = trainConfig.gradAccumSteps > 1 ? ` (${trainConfig.batchSize}×${trainConfig.gradAccumSteps})` : "";
  console.log(`seed: ${trainConfig.seed} | block_size: ${modelConfig.blockSize} | batch: ${effectiveBatch}${accumStr}`);
  console.log(`iters: ${trainConfig.iters} | lr: ${trainConfig.lr}${trainConfig.embGradScale < 1 ? ` | embGradScale: ${trainConfig.embGradScale}` : ""}`);
  if (deps.activationCheckpointing) console.log(`activation_checkpointing: enabled`);
  if (deps.mixedPrecision) console.log(`mixed_precision: f16 activations enabled`);

  // GPU proof: log device info and run smoke test
  if ("getDeviceInfo" in backend && "smokeTest" in backend) {
    const gpu = (backend as any).getDeviceInfo();
    const vendorName = gpuVendorName(gpu.vendorId);
    console.log(`gpu: ${gpu.deviceName} (${vendorName})`);
    console.log(`  f16: ${gpu.f16Supported} | async_transfer: ${gpu.hasAsyncTransfer} | wg_size: ${gpu.workgroupSize} | min_gpu: ${gpu.minGpuSize}`);
    console.log(
      `  vulkan_api: 0x${(gpu.apiVersion ?? 0).toString(16)} | device_id: 0x${(gpu.deviceId ?? 0).toString(16)} | ` +
      `subgroup: ${gpu.subgroupSize ?? "unknown"} [${gpu.minSubgroupSize ?? "?"},${gpu.maxSubgroupSize ?? "?"}] | ` +
      `subgroup_control: ${gpu.subgroupSizeControlSupported ?? false}`,
    );

    // Fail fast on missing capabilities, not vendor identity. This admits AMD
    // RDNA wave32 Vulkan devices while rejecting software ICDs and wave-size
    // combinations whose current kernels cannot execute correctly.
    const deviceAssessment = assessHeliosTrainingDevice(gpu);
    if (!deviceAssessment.supported) {
      throw new Error(
        `Helios GPU capability guard rejected ${gpu.deviceName} (vendor=${vendorName}): ` +
        `${deviceAssessment.reasons.join("; ")}. Check the Vulkan driver and device capability report.`
      );
    }

    if (smokePreflight) {
      if (smokePreflight.reason && smokePreflight.reason !== "verification mismatch") {
        console.log(`  smoke_test: FAIL (${smokePreflight.reason})`);
      } else {
        console.log(`  smoke_test: ${smokePreflight.verified ? "PASS" : "FAIL"} | gpu_throughput: ${smokePreflight.throughputGBps.toFixed(1)} GB/s`);
      }
      if (!smokePreflight.verified) {
        throw new Error(
          `GPU smoke test FAILED (${smokePreflight.reason || "verification mismatch"}). ` +
          `Training will not proceed on an unverified GPU.`
        );
      }
    }
  }

  console.log(``);
  emitEvent({
    step: startStep,
    level: "info",
    kind: "training_started",
    message: `training started (run_id=${rid})`,
    payload: {
      runId: rid,
      backend: backend.name,
      tokenizer: tokenizer.name,
      optimizer: optimizer.name,
      totalIters: trainConfig.iters,
      batchSize: trainConfig.batchSize,
      gradAccumSteps: trainConfig.gradAccumSteps,
      learningRate: trainConfig.lr,
      warmupIters: trainConfig.warmupIters,
      evalInterval: trainConfig.evalInterval,
      checkpointInterval: trainConfig.checkpointInterval ?? trainConfig.evalInterval,
      sampleInterval: trainConfig.sampleInterval,
      totalParams,
    },
  });

  // Create data loaders from pre-tokenized arrays (or SFT conversation loaders).
  const packed = trainConfig.packed;
  // Data order must not depend on how many random draws a model architecture
  // needs for parameter initialization. Separate streams make equal-token
  // architecture comparisons use identical train/validation windows.
  const trainDataRng = new SeededRng(trainConfig.seed ^ 0x4f1bbcdc);
  const valDataRng = new SeededRng(trainConfig.seed ^ 0x7a2d91e3);
  const trainLoader: BatchSource = sftMode
    ? sftTrain!
    : (trainTokenShards.length > 1
        ? new ShardedDataLoader(trainTokenShards, trainDataRng, trainConfig.batchSize, modelConfig.blockSize, packed)
        : new DataLoader(trainTokenShards[0], trainDataRng, trainConfig.batchSize, modelConfig.blockSize, packed));
  const valLoader: BatchSource | undefined = sftMode
    ? (sftVal ?? undefined)
    : (valTokenShards.length > 0
        ? (valTokenShards.length > 1
            ? new ShardedDataLoader(valTokenShards, valDataRng, trainConfig.batchSize, modelConfig.blockSize)
            : new DataLoader(valTokenShards[0], valDataRng, trainConfig.batchSize, modelConfig.blockSize))
        : undefined);
  const rcrUlLoader: BatchSource | undefined = rcrUlTrain ?? undefined;
  if (startStep > 0) {
    const trainBatches = startStep * trainConfig.gradAccumSteps;
    const completedEvals = trainConfig.evalInterval > 0
      ? Math.floor(startStep / trainConfig.evalInterval)
      : 0;
    const valBatches = completedEvals * trainConfig.evalIters;
    trainLoader.seekBatches(trainBatches);
    rcrUlLoader?.seekBatches(trainBatches);
    valLoader?.seekBatches(valBatches);
    console.log(
      `Data resume: train_batches=${trainBatches} val_batches=${valBatches}` +
      (rcrUlLoader ? ` rcr_ul_batches=${trainBatches}` : ""),
    );
  }
  if (packed && !sftMode) {
    console.log(`Sequence packing: ON (${trainLoader.stepsPerEpoch} steps/epoch)`);
  }

  // Symbio initialization (zero overhead when disabled)
  const symbioEnabled = !!trainConfig.symbio;
  const symbioConfig: SymbioConfig = (trainConfig.symbioConfig as unknown as SymbioConfig) ?? defaultSymbioConfig;
  const cusumDash = symbioEnabled ? new CusumDashboard(symbioConfig.cusumSensitivity, symbioConfig.cusumBaselineWindow) : null;
  const symbioCollector = symbioEnabled ? new SymbioMetricsCollector(symbioConfig) : null;
  const populationDynamics = symbioEnabled && symbioConfig.populationAdaptation
    ? new PopulationDynamicsController(symbioConfig.populationSize, symbioConfig.mutationRate, symbioConfig)
    : null;
  const kuramotoFusion = symbioEnabled && symbioConfig.fuseWeightsEachStep
    ? new KuramotoFusionController(symbioConfig)
    : null;
  const fusionShadow = symbioEnabled && symbioConfig.fuseWeightsEachStep
    ? new ConsensusFusionShadow(params)
    : null;
  if (symbioEnabled) {
    const activation = modelConfig.ffnActivation ?? "gelu";
    console.log(`symbio: enabled | activation=${activation} | cusum_sens=${symbioConfig.cusumSensitivity} | metrics_interval=${symbioConfig.metricsInterval}`);
    if (symbioConfig.populationAdaptation) {
      console.log(`symbio: population_adaptation=ON scale=[${symbioConfig.populationScaleMin},${symbioConfig.populationScaleMax}] step=${symbioConfig.populationScaleStep} cooldown=${symbioConfig.populationAdaptationCooldown}`);
    }
    if (symbioConfig.preserveWeightsAcrossCandidates) {
      console.log(`symbio: continuity=ON preserve_weights=true carry_optim=${symbioConfig.carryOptimizerStateAcrossCandidates} constant_ffn_dim=${symbioConfig.constantFfnDimAcrossCandidates}`);
    }
    if (symbioConfig.fuseWeightsEachStep) {
      console.log(`symbio: fusion=ON kuramoto(K=${symbioConfig.kuramotoCoupling},dt=${symbioConfig.kuramotoDt},damp=${symbioConfig.kuramotoDamping}) alpha=[${symbioConfig.fusionBaseStrength},${symbioConfig.fusionMaxStrength}]`);
    }
  }

  // FFN activation search orchestrator (both fixed and composed modes)
  const searchActive = symbioEnabled && (symbioConfig.searchMode === "ffn-activation-search" || symbioConfig.searchMode === "composed-activation-search");
  const searchOrchestrator = searchActive ? new SearchOrchestrator(symbioConfig, resumedActivationGraph) : null;
  let activeModelConfig = { ...modelConfig };

  if (searchOrchestrator) {
    const firstCandidate = searchOrchestrator.currentCandidate;
    if (firstCandidate) {
      // Initialize first candidate, optionally preserving overlap from the base model params.
      const ffnDim = symbioConfig.constantFfnDimAcrossCandidates
        ? (modelConfig.ffnDim ?? ffnDimForActivation(modelConfig.ffnActivation ?? "gelu", modelConfig.nEmbd, modelConfig.ffnDim))
        : ffnDimForActivation(firstCandidate.activation, modelConfig.nEmbd, modelConfig.ffnDim);
      activeModelConfig = {
        ...modelConfig,
        ffnActivation: firstCandidate.activation as any,
        ffnDim,
        activationGraph: firstCandidate.activationGraph ?? undefined,
      };
      rng.seed(trainConfig.seed);
      if (symbioConfig.preserveWeightsAcrossCandidates) {
        const prevParams = params;
        const transferred = initGPTWithTransferredWeights(activeModelConfig, backend, rng as any, prevParams);
        params = transferred.params;
        totalParams = countParams(params);
        if (symbioConfig.carryOptimizerStateAcrossCandidates) {
          const optCarry = carryOptimizerStateAcrossSwitch(optimizer, prevParams, params);
          console.log(`symbio switch(init): params exact=${transferred.stats.exactCopies} partial=${transferred.stats.partialCopies} fresh=${transferred.stats.initializedFresh} | opt exact=${optCarry.copiedBuffers} partial=${optCarry.partialBuffers} fresh=${optCarry.freshBuffers}`);
        } else {
          optimizer.loadStateDict({ step: 0, buffers: new Map() });
          console.log(`symbio switch(init): params exact=${transferred.stats.exactCopies} partial=${transferred.stats.partialCopies} fresh=${transferred.stats.initializedFresh} | opt reset`);
        }
      } else {
        params = initGPT(activeModelConfig, backend, rng as any);
        totalParams = countParams(params);
        optimizer.loadStateDict({ step: 0, buffers: new Map() });
      }
      fusionShadow?.rebind(params);
      const graphName = firstCandidate.activationGraph ? nameGraph(firstCandidate.activationGraph) : firstCandidate.activation;
      console.log(`symbio search: gen=${firstCandidate.generation} candidate=${firstCandidate.name} (${firstCandidate.id}) activation=${graphName} params=${totalParams.toLocaleString()}`);
    }
    const totalSearchSteps = symbioConfig.populationSize * symbioConfig.stepsPerCandidate * symbioConfig.generations;
    console.log(`symbio search: mode=${symbioConfig.searchMode} | base=${symbioConfig.populationSize} candidates × ${symbioConfig.stepsPerCandidate} steps × ${symbioConfig.generations} gens ≈ ${totalSearchSteps} total steps`);
  }

  const paramDataMap = new Map<string, TensorData>();
  const gradMap = new Map<string, TensorData>();
  const optimizerStepParamEntries: ((entries: ReturnType<typeof collectParamEntries>, gradScale?: number) => void) | null =
    typeof (optimizer as any).stepParamEntries === "function"
      ? (optimizer as any).stepParamEntries.bind(optimizer)
      : null;
  type ParamEntry = ReturnType<typeof collectParamEntries>[number];
  type ParamVariable = ParamEntry[1];
  // Stable parameter traversal for hot loops; refresh whenever `params` is replaced.
  let paramEntries: ReturnType<typeof collectParamEntries> = [];
  let paramNames: string[] = [];
  let paramVars: ParamVariable[] = [];
  let paramSizes: number[] = [];
  const refreshParamCaches = (): void => {
    paramEntries = collectParamEntries(params);
    const paramCount = paramEntries.length;
    paramNames = new Array(paramCount);
    paramVars = new Array(paramCount);
    paramSizes = new Array(paramCount);
    for (let i = 0; i < paramCount; i++) {
      const entry = paramEntries[i];
      const name = entry[0];
      const variable = entry[1];
      paramNames[i] = name;
      paramVars[i] = variable;
      paramSizes[i] = shapeSize(variable.data.shape);
    }
    // Map-based optimizer path only.
    if (!optimizerStepParamEntries) {
      // Parameter data references are stable between switches; rebuild only on refresh.
      paramDataMap.clear();
      for (let i = 0; i < paramCount; i++) {
        paramDataMap.set(paramNames[i], paramVars[i].data);
      }
    }
  };
  refreshParamCaches();

  // Pre-compute embedding param indices for gradient scaling.
  // Recomputed whenever refreshParamCaches is called via paramNames update.
  let embParamIndices: number[] = [];
  const recomputeEmbIndices = (): void => {
    embParamIndices = [];
    for (let i = 0; i < paramNames.length; i++) {
      if (paramNames[i] === "wte" || paramNames[i] === "wpe") {
        embParamIndices.push(i);
      }
    }
  };
  recomputeEmbIndices();

  const backendAny = backend as any;
  const setOptimizerLrFn: ((lr: number) => void) | undefined =
    typeof (optimizer as any).setLr === "function"
      ? (optimizer as any).setLr.bind(optimizer)
      : undefined;
  const resetStepOpsFn: (() => void) | undefined =
    typeof backendAny.resetStepOps === "function"
      ? backendAny.resetStepOps.bind(backendAny)
      : undefined;
  const getGpuStepStatsFn: (() => {
    profilingEnabled: boolean;
    timingEnabled: boolean;
    operations: number;
    flushes: number;
    waitedFlushes: number;
    dgcFlushes: number;
    timestampedFlushes: number;
    batchGpuTimeUs: number;
    dispatchGpuTimeUs: number;
    gpuBlockingTimeMs: number;
    operationsPerFlush: number;
    byKind: Array<{ name: string; count: number; gpuTimeUs: number }>;
    byKernel: Array<{ name: string; count: number; gpuTimeUs: number }>;
  }) | undefined =
    typeof backendAny.getGpuStepStats === "function"
      ? backendAny.getGpuStepStats.bind(backendAny)
      : undefined;
  // Explicit GPU buffer release function — deterministic cleanup instead of
  // relying on FinalizationRegistry which is unreliable for timely VRAM reclaim.
  const releaseFn: ((td: TensorData) => void) | undefined =
    typeof backendAny.releaseGpuTensor === "function"
      ? (td: TensorData) => backendAny.releaseGpuTensor(td)
      : undefined;
  const flushFn: (() => void) | undefined =
    typeof backendAny.flush === "function"
      ? backendAny.flush.bind(backendAny)
      : undefined;
  const syncGpuFn: (() => void) | undefined =
    typeof backendAny.syncGpu === "function"
      ? backendAny.syncGpu.bind(backendAny)
      : undefined;
  const purgeBufferPoolsFn: (() => void) | undefined =
    typeof backendAny.purgeBufferPools === "function"
      ? backendAny.purgeBufferPools.bind(backendAny)
      : undefined;
  const gpuMemStatsFn: (() => any) | undefined =
    typeof backendAny.gpuMemStats === "function"
      ? backendAny.gpuMemStats.bind(backendAny)
      : undefined;
  const poolBreakdownFn: ((topN?: number) => string) | undefined =
    typeof backendAny.poolBreakdown === "function"
      ? backendAny.poolBreakdown.bind(backendAny)
      : undefined;
  const sumOfSquaresFn: ((data: TensorData) => TensorData) | undefined =
    typeof backend.sumOfSquares === "function"
      ? backend.sumOfSquares.bind(backend)
      : undefined;
  const totalSumOfSquaresFn: ((tensors: TensorData[]) => TensorData) | undefined =
    typeof backend.totalSumOfSquares === "function"
      ? backend.totalSumOfSquares.bind(backend)
      : undefined;
  const checkFiniteFn: ((data: TensorData) => TensorData) | undefined =
    typeof backend.checkFinite === "function"
      ? backend.checkFinite.bind(backend)
      : undefined;
  const hasSumSq = !!sumOfSquaresFn;
  const hasTotalSumSq = !!totalSumOfSquaresFn;
  const disableTotalSumSq = ((process.env.ALPHA_DISABLE_TOTAL_SUMSQ ?? "0").trim() === "1");
  const canUseTotalSumSq = hasTotalSumSq && !disableTotalSumSq;
  let useTotalSumSq = canUseTotalSumSq;
  const disableGradNorm = ((process.env.ALPHA_DISABLE_GRAD_NORM ?? "0").trim() === "1");
  const readEnvInt = (name: string, fallback: number, min = 0): number => {
    const raw = process.env[name];
    if (!raw) return fallback;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < min) return fallback;
    return parsed;
  };
  const readEnvFloat = (name: string, fallback: number, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY): number => {
    const raw = process.env[name];
    if (!raw) return fallback;
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  // Training loop
  const startTime = performance.now();
  let spikeSkips = 0;
  let clippedSteps = 0;

  // Dynamic loss scaling for mixed precision training
  const useLossScaling = !!deps.mixedPrecision;
  const logEvery = Math.max(1, trainConfig.logEvery ?? 1);
  const shouldYieldForCallbacks = !!(onStep || deps.onCheckpoint || deps.onSamples);
  const callbackYieldEvery = readEnvInt("ALPHA_CALLBACK_YIELD_EVERY", 25, 1);
  const totalIters = trainConfig.iters;
  const traceEnabled = trainConfig.trace;
  const capturePhaseTimings = traceEnabled;
  const phaseSyncProfile = process.env.ALPHA_PROFILE_PHASE_SYNC === "1";
  // Experimental kernel-research lane. The production default keeps
  // cooperative f16 matrix paths out of backward because unscaled gradient
  // activations can exceed f16 range. Enabling this flag is a falsifiable
  // performance/trajectory experiment, not an implicit production change.
  const enableCoopBackward = process.env.HELIOS_ENABLE_COOP_BACKWARD === "1";
  // Bounded performance sweeps should not create multi-gigabyte terminal
  // checkpoints. Full model runs leave this unset and preserve normal cadence.
  const disableCheckpoints = process.env.ALPHA_DISABLE_CHECKPOINTS === "1";
  if (phaseSyncProfile && !traceEnabled) {
    throw new Error("ALPHA_PROFILE_PHASE_SYNC=1 requires --trace=true");
  }
  if (enableCoopBackward) {
    console.warn("[helios:experimental] cooperative matmul is enabled during backward");
  }
  if (disableCheckpoints) {
    console.warn("[alpha:diagnostic] checkpoint writes are disabled for this run");
  }
  const gradAccumSteps = trainConfig.gradAccumSteps;
  const gradClip = trainConfig.gradClip;
  const embGradScale = trainConfig.embGradScale;
  const spikeThreshold = trainConfig.spikeThreshold;
  const evalIters = trainConfig.evalIters;
  const evalInterval = trainConfig.evalInterval;
  const checkpointInterval = Math.max(1, trainConfig.checkpointInterval ?? evalInterval);
  const sampleInterval = trainConfig.sampleInterval;
  let latestCheckpointPath: string | null = null;
  const branchCount = rcrUlLoader ? 2 : 1;
  const tokensProcessedPerStep = trainConfig.batchSize * modelConfig.blockSize * gradAccumSteps * branchCount;
  const warmup = trainConfig.warmupIters > 0
    ? trainConfig.warmupIters
    : trainConfig.warmupIters < 0
      ? 0  // negative = explicitly disabled
      : Math.min(2000, Math.floor(totalIters / 5));
  const lrMin = (trainConfig.lrMin ?? 0) === 0
    ? trainConfig.lr / 10  // auto-calc: lr/10 (nanoGPT convention)
    : trainConfig.lrMin;
  const decayDenom = Math.max(1, totalIters - warmup);
  let lossScale = useLossScaling ? 128.0 : 1.0; // start very safe, will auto-tune up
  let scaleSuccessCount = 0;
  const SCALE_GROWTH_INTERVAL = 200; // double scale after this many consecutive good steps
  let lossScaleReductions = 0;
  let repeatedSpikeNormStreak = 0;
  let prevSpikeNorm = Number.NaN;
  let hugeSpikeWindowStartStep = -1;
  let hugeSpikeCountInWindow = 0;
  const evalHistory: EvalSnapshot[] = [];
  const dropoutRng = new DropoutRng(trainConfig.seed);
  const trainTape = new Tape();
  const gradTensors: TensorData[] = [];
  const gradNamesBuf: string[] = [];
  const perParamNormsBuf: { name: string; normSq: number }[] = [];
  // Defaults are set from the latest strict (`status=ok`) adaptive 20-run sweep.
  const ADAPTIVE_MEM_STATS_POLL_EVERY = readEnvInt("ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY", 8, 1);
  const ADAPTIVE_SYNC_MIN_INTERVAL = readEnvInt("ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL", 6, 1);
  const ADAPTIVE_SYNC_DEFERRED_THRESHOLD = readEnvInt("ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD", 28, 0);
  const ADAPTIVE_SYNC_PENDING_THRESHOLD = readEnvInt("ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD", 24, 0);
  const ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD = readEnvInt("ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD", 5200, 0);
  const ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD = readEnvInt("ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD", 6000, 0);
  const ADAPTIVE_PURGE_MIN_INTERVAL = readEnvInt(
    "ALPHA_ADAPTIVE_PURGE_MIN_INTERVAL",
    Math.max(16, ADAPTIVE_SYNC_MIN_INTERVAL),
    1,
  );
  const GPU_METRICS_SAMPLE_EVERY = readEnvInt("ALPHA_GPU_METRICS_SAMPLE_EVERY", 75, 1);
  let memStatsCache: any | null = null;
  let lastFlowStats: any | null = null;
  let lastMemStatsProbeStep = 0;
  let lastAdaptiveSyncStep = 0;
  let lastAdaptivePurgeStep = 0;
  let peakGpuLiveAllocs = 0;
  let peakGpuMemPoolMb = 0;
  const spikeLrBackoff = readEnvFloat("ALPHA_SPIKE_LR_BACKOFF", 0.5, 0.01, 1.0);
  const spikeLrMinScale = readEnvFloat("ALPHA_SPIKE_LR_MIN_SCALE", 0.1, 0.001, 1.0);
  const spikeLrRecoverySteps = Math.round(readEnvFloat("ALPHA_SPIKE_LR_RECOVERY_STEPS", 200, 10, 10_000));
  let stepsSinceLastSpike = 0;
  const forceCpuGradNormDefault = ((process.env.ALPHA_FORCE_CPU_GRAD_NORM ?? "0").trim() === "1");
  const gradNormCpuRecheck = ((process.env.ALPHA_GRAD_NORM_CPU_RECHECK ?? "1").trim() !== "0");
  const gradNormCpuSticky = ((process.env.ALPHA_GRAD_NORM_CPU_STICKY ?? "1").trim() !== "0");
  const gradNormMismatchRatio = readEnvFloat("ALPHA_GRAD_NORM_MISMATCH_RATIO", 8, 1, 10_000);
  const gradNormSuspiciousThreshold = readEnvFloat(
    "ALPHA_GRAD_NORM_SUSPICIOUS_THRESHOLD",
    Math.max(500, gradClip > 0 ? gradClip * 5000 : 2000),
    0,
  );
  let runtimeLrScale = 1.0;
  let useCpuGradNorm = forceCpuGradNormDefault;
  let gradNormMismatchStreak = 0;

  const terminalEvalOnly = (process.env.ALPHA_TERMINAL_EVAL_ONLY ?? "").trim() === "1";
  if (terminalEvalOnly) {
    const repairCommit = (process.env.ALPHA_TERMINAL_EVAL_REPAIR_COMMIT ?? "").trim();
    if (!/^[0-9a-f]{40}$/.test(repairCommit)) {
      throw new Error("ALPHA_TERMINAL_EVAL_REPAIR_COMMIT must be a full commit SHA");
    }
    if (!resumePath || startStep !== totalIters || path.basename(resumePath) !== `checkpoint-${totalIters}.json`) {
      throw new Error("terminal eval-only repair requires resuming the exact terminal checkpoint");
    }
    if (!valLoader) throw new Error("terminal eval-only repair requires a validation loader");

    if (flushFn) flushFn();
    if (typeof globalThis.gc === "function") {
      (globalThis as any).gc();
      await new Promise<void>(resolve => setImmediate(resolve));
    }
    if (flushFn) flushFn();

    const repairLoader: BatchSource = valLoader;
    let valLossSum = 0;
    for (let evalIndex = 0; evalIndex < evalIters; evalIndex++) {
      const valBatch = repairLoader.nextBatch();
      const evalTape = new Tape();
      const { loss } = gptForward(
        activeModelConfig,
        params,
        backend,
        evalTape,
        valBatch.inputs,
        valBatch.targets,
        false,
        false,
        false,
        undefined,
        undefined,
        valBatch.lossMask,
      );
      if (!loss) throw new Error(`terminal eval-only repair produced no loss at batch ${evalIndex + 1}`);
      valLossSum += (loss.data.data as Float32Array)[0];
      if (releaseFn) releaseFn(loss.data);
      evalTape.clear(releaseFn);
      if (releaseFn) {
        releaseFn(valBatch.inputs);
        releaseFn(valBatch.targets);
        if (valBatch.lossMask) releaseFn(valBatch.lossMask);
      }
      if (flushFn) flushFn();
    }
    const valLoss = valLossSum / evalIters;
    const originalMetrics = await fs.readFile(metricsPath, "utf8");
    const repairedMetrics = repairTerminalValidationMetric(originalMetrics, totalIters, valLoss);
    const { createHash } = await import("node:crypto");
    const metricsSha256Before = createHash("sha256").update(originalMetrics).digest("hex");
    const metricsSha256After = createHash("sha256").update(repairedMetrics).digest("hex");
    await metricsHandle.close();
    const metricsTmp = `${metricsPath}.terminal-eval-repair.tmp`;
    await fs.writeFile(metricsTmp, repairedMetrics, { encoding: "utf8", flag: "wx" });
    await fs.rename(metricsTmp, metricsPath);
    const evidence = {
      schema: "alpha-terminal-eval-repair-v1",
      result: "PASS",
      repaired_utc: new Date().toISOString(),
      repair_commit: repairCommit,
      checkpoint: resumePath,
      step: totalIters,
      eval_iters: evalIters,
      val_loss: valLoss,
      metrics_sha256_before: metricsSha256Before,
      metrics_sha256_after: metricsSha256After,
      training_steps_executed: 0,
    };
    await fs.writeFile(
      path.join(runDir, "terminal-eval-repair.json"),
      `${JSON.stringify(evidence, null, 2)}\n`,
      { encoding: "utf8", flag: "wx" },
    );
    console.log(`terminal eval-only repair: val_loss=${valLoss.toFixed(6)} (0 training steps)`);
    return { params, modelConfig: activeModelConfig };
  }

  for (let step = startStep; step < totalIters; step++) {
    const stepStart = performance.now();
    const stepNum = step + 1;

    // Learning rate schedule: linear warmup + cosine decay to lrMin
    let baseLr: number;
    if (step < warmup) {
      baseLr = lrMin + (trainConfig.lr - lrMin) * stepNum / warmup;
    } else {
      const decay = (step - warmup) / decayDenom;
      baseLr = lrMin + (trainConfig.lr - lrMin) * 0.5 * (1 + Math.cos(Math.PI * decay));
    }
    let lr = Math.max(lrMin * spikeLrMinScale, baseLr * runtimeLrScale);
    if (setOptimizerLrFn) setOptimizerLrFn(lr);

    // Reset per-step GPU ops counter
    if (resetStepOpsFn) resetStepOpsFn();

    // Gradient accumulation: run K forward+backward passes, accumulating
    // gradients on parameter Variables. Each micro-batch contributes 1/K
    // of the total gradient. Increases effective batch size by K without
    // increasing VRAM usage (only one micro-batch is live at a time).
    const accumSteps = gradAccumSteps;
    const stepSeedBase = trainConfig.seed + step * 1000;
    let nanDetected = false;
    let spikeDetectedThisStep = false;
    let lossNanLogged = false;
    let gradNanLogged = false;
    let gradNanCount = 0;
    let lossVal = 0;
    let positiveCeLossVal = 0;
    let negativeUlLossVal = 0;
    let positiveSupervisedTokenMass = 0;
    let negativePenaltyPositionMass = 0;
    let negativeExamplesWithPenalty = 0;
    let negativeFirstPenaltyTargetPosition = Number.POSITIVE_INFINITY;
    let negativeLastPenaltyTargetPosition = Number.NEGATIVE_INFINITY;
    let negativeBadProbabilityWeighted = 0;
    let negativeBadProbabilityMaskMass = 0;
    let negativeMaxBadProbability = 0;
    let nanCount = 0;
    let infCount = 0;
    let positiveFwdMs = 0;
    let positiveBwdMs = 0;
    let negativeFwdMs = 0;
    let negativeBwdMs = 0;
    let dataLoadMs = 0;
    let fwdMs = 0;
    let bwdMs = 0;
    let stepBatchHash = 0x811c9dc5;
    let stepBatchTokMin = Number.POSITIVE_INFINITY;
    let stepBatchTokMax = Number.NEGATIVE_INFINITY;

    // GPU-side loss accumulation: avoid per-microstep CPU readback for grad accumulation.
    // For accumSteps=1, skip the no-op GPU scale/add path and read loss directly.
    const useGpuLossAccum = accumSteps > 1;
    let lossAccum: TensorData | null = null;
    let ulLossAccum: TensorData | null = null;

    for (let microStep = 0; microStep < accumSteps; microStep++) {
      const _dl0 = capturePhaseTimings ? performance.now() : 0;
      const batch = trainLoader.nextBatch();
      const _dl1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) dataLoadMs += _dl1 - _dl0;
      if (batch.lossMask) {
        const positiveMask = batch.lossMask.data as Float32Array;
        for (let i = 0; i < positiveMask.length; i++) positiveSupervisedTokenMass += positiveMask[i];
      }
      const inputTokens = batch.inputs.data as Int32Array;
      const hashSampleCount = Math.min(64, inputTokens.length);
      for (let i = 0; i < hashSampleCount; i++) {
        const tok = inputTokens[i] | 0;
        stepBatchHash ^= tok;
        stepBatchHash = Math.imul(stepBatchHash, 0x01000193);
        if (tok < stepBatchTokMin) stepBatchTokMin = tok;
        if (tok > stepBatchTokMax) stepBatchTokMax = tok;
      }
      dropoutRng.reset(stepSeedBase + microStep, 0);
      const _positiveFwd0 = deps.rcrUl ? performance.now() : 0;
      const { loss } = gptForward(activeModelConfig, params, backend, trainTape, batch.inputs, batch.targets, true, !!deps.activationCheckpointing, !!deps.mixedPrecision, dropoutRng, releaseFn, batch.lossMask);
      if (phaseSyncProfile && syncGpuFn) syncGpuFn();
      if (deps.rcrUl) positiveFwdMs += performance.now() - _positiveFwd0;
      const _fwd1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) fwdMs += _fwd1 - _dl1;

      if (!loss) throw new Error("Loss is undefined");
      const lossDataRef = !useGpuLossAccum ? loss.data : null;

      if (useGpuLossAccum) {
        // Accumulate loss on GPU: scale by 1/accumSteps and add to accumulator.
        // No .data access here — stays lazy on GPU, no flush/wait/readback.
        const scaledLoss = backend.scale(loss.data, 1 / accumSteps);
        if (lossAccum === null) {
          lossAccum = scaledLoss;
        } else {
          if (backend.addInplace) {
            backend.addInplace(lossAccum, scaledLoss);
            if (releaseFn) releaseFn(scaledLoss);
          } else {
            const prev = lossAccum;
            lossAccum = backend.add(lossAccum, scaledLoss);
            if (releaseFn) { releaseFn(prev); releaseFn(scaledLoss); }
          }
        }
      }

      // Pause cooperative matmul during backward to avoid f16 precision loss
      // on large gradient values that can overflow f16 range (>65504).
      const backendAnyBw = backend as any;
      if (!enableCoopBackward && typeof backendAnyBw.coopMatmulPaused === "boolean") {
        backendAnyBw.coopMatmulPaused = true;
      }
      // Loss scaling: pass scaled initial gradient for backward.
      // This scales all gradients by lossScale, preventing f16 underflow.
      // Gradients are unscaled back before the optimizer step.
      const _positiveBwd0 = deps.rcrUl ? performance.now() : 0;
      if (useLossScaling && lossScale !== 1.0) {
        const scaledGrad = backend.full(loss.data.shape, lossScale, loss.data.dtype);
        trainTape.backward(loss, backend, releaseFn, scaledGrad);
      } else {
        trainTape.backward(loss, backend, releaseFn);
      }
      if (phaseSyncProfile && syncGpuFn) syncGpuFn();
      if (deps.rcrUl) positiveBwdMs += performance.now() - _positiveBwd0;
      if (!enableCoopBackward && typeof backendAnyBw.coopMatmulPaused === "boolean") {
        backendAnyBw.coopMatmulPaused = false;
      }
      if (!useGpuLossAccum) {
        positiveCeLossVal = (lossDataRef!.data as Float32Array)[0];
        if (!isFinite(positiveCeLossVal)) {
          console.warn(`  [warn] positive CE loss=NaN at step ${stepNum} — skipping`);
          if (!lossNanLogged) {
            emitEvent({
              step: stepNum,
              level: "warn",
              kind: "loss_nan",
              message: `positive CE loss became non-finite; skipping optimizer update (step ${stepNum})`,
              payload: { microStep, accumSteps, branch: "positive_ce" },
            });
            lossNanLogged = true;
          }
          nanDetected = true;
        }
      }
      const _bwd1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) bwdMs += _bwd1 - _fwd1;
      // Release tape intermediates but keep param gradients for accumulation
      trainTape.clear(releaseFn);
      // DataLoader reuses batch TensorData objects in a ring. Explicitly release
      // their GPU buffers so the next in-place refill re-uploads fresh token ids.
      if (releaseFn) {
        releaseFn(batch.inputs);
        releaseFn(batch.targets);
        if (batch.lossMask) releaseFn(batch.lossMask);
      }

      if (rcrUlLoader && deps.rcrUl) {
        // Reclaim the positive branch before constructing an equally large
        // negative graph. Both C0 and U1 take this exact execution path.
        if (typeof (backend as any).syncGpu === "function") (backend as any).syncGpu();

        const _ulDl0 = capturePhaseTimings ? performance.now() : 0;
        const ulBatch = rcrUlLoader.nextBatch();
        const _ulDl1 = capturePhaseTimings ? performance.now() : 0;
        if (capturePhaseTimings) dataLoadMs += _ulDl1 - _ulDl0;
        const ulMask = ulBatch.lossMask!.data as Float32Array;
        const ulRowWidth = activeModelConfig.blockSize;
        for (let b = 0; b < trainConfig.batchSize; b++) {
          let rowHasPenalty = false;
          for (let t = 0; t < ulRowWidth; t++) {
            const maskValue = ulMask[b * ulRowWidth + t];
            negativePenaltyPositionMass += maskValue;
            if (maskValue > 0) {
              rowHasPenalty = true;
              const targetPosition = t + 1;
              negativeFirstPenaltyTargetPosition = Math.min(negativeFirstPenaltyTargetPosition, targetPosition);
              negativeLastPenaltyTargetPosition = Math.max(negativeLastPenaltyTargetPosition, targetPosition);
            }
          }
          if (rowHasPenalty) negativeExamplesWithPenalty++;
        }

        // Bind the negative trajectory into the batch audit hash too, including
        // its mask, so a changed rollout or penalty placement is observable.
        const ulInputs = ulBatch.inputs.data as Int32Array;
        const ulHashCount = Math.min(64, ulInputs.length);
        for (let i = 0; i < ulHashCount; i++) {
          stepBatchHash ^= ulInputs[i] | 0;
          stepBatchHash = Math.imul(stepBatchHash, 0x01000193);
          stepBatchHash ^= ulMask[i] > 0 ? 1 : 0;
          stepBatchHash = Math.imul(stepBatchHash, 0x01000193);
        }

        const _negativeFwd0 = performance.now();
        const { loss: ulLoss } = gptForward(
          activeModelConfig,
          params,
          backend,
          trainTape,
          ulBatch.inputs,
          ulBatch.targets,
          true,
          !!deps.activationCheckpointing,
          !!deps.mixedPrecision,
          dropoutRng,
          releaseFn,
          ulBatch.lossMask,
          { kind: "unlikelihood", epsilon: deps.rcrUl.epsilon },
        );
        negativeFwdMs += performance.now() - _negativeFwd0;
        const ulStats = backend.getLastCrossEntropyUnlikelihoodMaskedStats?.();
        if (ulStats) {
          negativeBadProbabilityWeighted += ulStats.meanBadProbability * ulStats.maskMass;
          negativeBadProbabilityMaskMass += ulStats.maskMass;
          negativeMaxBadProbability = Math.max(negativeMaxBadProbability, ulStats.maxBadProbability);
        }
        const _ulFwd1 = capturePhaseTimings ? performance.now() : 0;
        if (capturePhaseTimings) fwdMs += _ulFwd1 - _ulDl1;
        if (!ulLoss) throw new Error("RCR-UL loss is undefined");
        const ulLossDataRef = !useGpuLossAccum ? ulLoss.data : null;

        if (useGpuLossAccum) {
          const scaledUlLoss = backend.scale(ulLoss.data, 1 / accumSteps);
          if (ulLossAccum === null) {
            ulLossAccum = scaledUlLoss;
          } else if (backend.addInplace) {
            backend.addInplace(ulLossAccum, scaledUlLoss);
            if (releaseFn) releaseFn(scaledUlLoss);
          } else {
            const previous = ulLossAccum;
            ulLossAccum = backend.add(ulLossAccum, scaledUlLoss);
            if (releaseFn) { releaseFn(previous); releaseFn(scaledUlLoss); }
          }
        }

        if (!enableCoopBackward && typeof backendAnyBw.coopMatmulPaused === "boolean") backendAnyBw.coopMatmulPaused = true;
        // Always provide an explicit upstream gradient, including exact zero
        // for C0. This is the matched-control invariant: no skipped branch.
        const branchScale = deps.rcrUl.weight * lossScale;
        const ulInitialGrad = backend.full(ulLoss.data.shape, branchScale, ulLoss.data.dtype);
        const _negativeBwd0 = performance.now();
        trainTape.backward(ulLoss, backend, releaseFn, ulInitialGrad);
        negativeBwdMs += performance.now() - _negativeBwd0;
        if (!enableCoopBackward && typeof backendAnyBw.coopMatmulPaused === "boolean") backendAnyBw.coopMatmulPaused = false;

        if (!useGpuLossAccum) {
          negativeUlLossVal = (ulLossDataRef!.data as Float32Array)[0];
          if (!isFinite(negativeUlLossVal)) {
            console.warn(`  [warn] negative UL loss=NaN at step ${stepNum} — skipping`);
            if (!lossNanLogged) {
              emitEvent({
                step: stepNum,
                level: "warn",
                kind: "loss_nan",
                message: `negative UL loss became non-finite; skipping optimizer update (step ${stepNum})`,
                payload: { microStep, accumSteps, branch: "negative_ul" },
              });
              lossNanLogged = true;
            }
            nanDetected = true;
          }
        }
        const _ulBwd1 = capturePhaseTimings ? performance.now() : 0;
        if (capturePhaseTimings) bwdMs += _ulBwd1 - _ulFwd1;
        trainTape.clear(releaseFn);
        if (releaseFn) {
          releaseFn(ulBatch.inputs);
          releaseFn(ulBatch.targets);
          releaseFn(ulBatch.lossMask!);
        }
      }
      // Sync GPU after each micro-step to fully reclaim deferred buffer releases.
      // Without this, backward-pass intermediates pile up across accumulation steps,
      // causing allocation count to grow unbounded and OOM on the driver side.
      // syncGpu() flushes, waits for completion, and processes pending destroys.
      if ((accumSteps > 1 || !!rcrUlLoader) && typeof (backend as any).syncGpu === "function") {
        (backend as any).syncGpu();
      }
    }

    // Single CPU readback of accumulated loss (triggers one flush+wait)
    if (useGpuLossAccum && lossAccum) {
      positiveCeLossVal = (lossAccum.data as Float32Array)[0];
      if (!isFinite(positiveCeLossVal)) {
        console.warn(`  [warn] accumulated positive CE loss=NaN at step ${stepNum} — skipping`);
        if (!lossNanLogged) {
          emitEvent({
            step: stepNum,
            level: "warn",
            kind: "loss_nan",
            message: `accumulated positive CE loss became non-finite; skipping optimizer update (step ${stepNum})`,
            payload: { accumSteps, branch: "positive_ce" },
          });
          lossNanLogged = true;
        }
        nanDetected = true;
      }
      if (releaseFn) releaseFn(lossAccum);
    }
    if (useGpuLossAccum && ulLossAccum) {
      negativeUlLossVal = (ulLossAccum.data as Float32Array)[0];
      if (!isFinite(negativeUlLossVal)) {
        console.warn(`  [warn] accumulated negative UL loss=NaN at step ${stepNum} — skipping`);
        if (!lossNanLogged) {
          emitEvent({
            step: stepNum,
            level: "warn",
            kind: "loss_nan",
            message: `accumulated negative UL loss became non-finite; skipping optimizer update (step ${stepNum})`,
            payload: { accumSteps, branch: "negative_ul" },
          });
          lossNanLogged = true;
        }
        nanDetected = true;
      }
      if (releaseFn) releaseFn(ulLossAccum);
    }
    lossVal = positiveCeLossVal + (deps.rcrUl ? deps.rcrUl.weight * negativeUlLossVal : 0);
    for (const value of [positiveCeLossVal, negativeUlLossVal, lossVal]) {
      if (Number.isNaN(value)) nanCount++;
      else if (!Number.isFinite(value)) infCount++;
    }

    // Keep gradients unmodified and fold scaling/clipping into optimizer update.
    // This avoids one or two full tensor passes over all parameter gradients.
    const gradScaleFactor = 1.0 / (accumSteps * lossScale);
    const gradScaleAbs = Math.abs(gradScaleFactor);
    let gradNorm = 0;
    const needsGradNorm = !disableGradNorm && (
      gradClip > 0 ||
      spikeThreshold > 0 ||
      useLossScaling ||
      traceEnabled ||
      symbioEnabled ||
      stepNum % 500 === 0
    );
    const _t3 = capturePhaseTimings ? performance.now() : 0;

    // Collect gradients and compute gradient norm via backend ops (stays on GPU).

    let perLayerGradNorms: Record<string, number> | undefined;

    if (!nanDetected && needsGradNorm) {
      const collectPerParamNorms = traceEnabled || (stepNum % 500 === 0);
      const gradScaleSq = collectPerParamNorms ? (gradScaleAbs * gradScaleAbs) : 0;
      const gradNames = collectPerParamNorms ? gradNamesBuf : null;
      if (gradNames) gradNames.length = 0;
      gradTensors.length = 0;
      for (let i = 0; i < paramVars.length; i++) {
        const grad = paramVars[i].grad;
        if (grad) {
          if (gradNames) gradNames.push(paramNames[i]);
          gradTensors.push(grad);
        }
      }
      const grads = gradTensors;

      const perParamNorms = perParamNormsBuf;
      perParamNorms.length = 0;

      const computeGradNormCpu = (): number => {
        let gradNormSq = 0;
        for (let i = 0; i < grads.length; i++) {
          const val = computeTensorNormSqCpu(grads[i]);
          gradNormSq += val;
          if (collectPerParamNorms && gradNames) perParamNorms.push({ name: gradNames[i], normSq: val * gradScaleSq });
        }
        return Math.sqrt(gradNormSq) * gradScaleAbs;
      };

      if (useCpuGradNorm && grads.length > 0) {
        // Safety mode: avoid GPU reduction kernels for grad norm entirely.
        gradNorm = computeGradNormCpu();
      } else if (useTotalSumSq && grads.length > 0) {
        // Fast path: one scalar readback for total grad norm.
        const totalSq = totalSumOfSquaresFn!(grads);
        gradNorm = Math.sqrt((totalSq.data as Float32Array)[0]) * gradScaleAbs;
        if (releaseFn) releaseFn(totalSq);
      } else {
        // Fallback: compute per-parameter scalars and sum on CPU.
        const sqNormParts: TensorData[] = [];
        const g2Intermediates: TensorData[] = [];
        for (let i = 0; i < grads.length; i++) {
          const g = grads[i];
          if (hasSumSq) {
            sqNormParts.push(sumOfSquaresFn!(g));
          } else {
            const g2 = backend.mul(g, g);
            g2Intermediates.push(g2);
            sqNormParts.push(backend.sum(g2));
          }
        }
        let gradNormSq = 0;
        for (let i = 0; i < sqNormParts.length; i++) {
          const val = (sqNormParts[i].data as Float32Array)[0];
          gradNormSq += val;
          if (collectPerParamNorms && gradNames) perParamNorms.push({ name: gradNames[i], normSq: val * gradScaleSq });
        }
        gradNorm = Math.sqrt(gradNormSq) * gradScaleAbs;
        if (releaseFn) {
          for (const g2 of g2Intermediates) releaseFn(g2);
          for (const part of sqNormParts) releaseFn(part);
        }
      }

      if (
        gradNormCpuRecheck &&
        !useCpuGradNorm &&
        grads.length > 0 &&
        Number.isFinite(gradNorm) &&
        gradNorm >= gradNormSuspiciousThreshold
      ) {
        const gradNormGpu = gradNorm;
        // Re-check suspiciously large norms on CPU to avoid reduction-kernel artifacts.
        if (collectPerParamNorms) perParamNorms.length = 0;
        const gradNormCpu = computeGradNormCpu();
        const minMag = Math.max(1e-12, Math.min(Math.abs(gradNormGpu), Math.abs(gradNormCpu)));
        const maxMag = Math.max(Math.abs(gradNormGpu), Math.abs(gradNormCpu));
        const mismatchRatio = maxMag / minMag;
        const mismatch = !Number.isFinite(mismatchRatio) || mismatchRatio >= gradNormMismatchRatio;
        gradNorm = gradNormCpu;

        if (mismatch) {
          gradNormMismatchStreak++;
          console.warn(
            `  [grad_norm] suspicious GPU/CPU mismatch at step ${stepNum}: ` +
            `gpu=${gradNormGpu.toFixed(3)} cpu=${gradNormCpu.toFixed(3)} ratio=${mismatchRatio.toFixed(2)}x`,
          );
          emitEvent({
            step: stepNum,
            level: "warn",
            kind: "grad_norm_cpu_recheck_mismatch",
            message: "grad_norm GPU/CPU mismatch detected; using CPU grad norm result",
            payload: {
              gpuGradNorm: gradNormGpu,
              cpuGradNorm: gradNormCpu,
              mismatchRatio,
              mismatchStreak: gradNormMismatchStreak,
            },
          });
          if (gradNormCpuSticky && gradNormMismatchStreak >= 2) {
            useCpuGradNorm = true;
            useTotalSumSq = false;
            console.warn("  [grad_norm] switching to CPU grad norm mode for run stability");
            emitEvent({
              step: stepNum,
              level: "warn",
              kind: "grad_norm_cpu_mode_enabled",
              message: "enabled CPU grad norm mode after repeated GPU/CPU mismatch",
              payload: {
                suspiciousThreshold: gradNormSuspiciousThreshold,
                mismatchRatioThreshold: gradNormMismatchRatio,
              },
            });
          }
        } else {
          gradNormMismatchStreak = 0;
        }
      } else if (!useCpuGradNorm) {
        gradNormMismatchStreak = 0;
      }

      // Detailed per-param diagnostics are expensive; keep them off the hot path
      // unless trace mode is enabled or it's a periodic diagnostic step.
      if (collectPerParamNorms && hasSumSq && perParamNorms.length === 0) {
        const sqNormParts: TensorData[] = [];
        for (const g of grads) sqNormParts.push(sumOfSquaresFn!(g));
        for (let i = 0; i < sqNormParts.length; i++) {
          perParamNorms.push({ name: gradNames![i], normSq: (sqNormParts[i].data as Float32Array)[0] * gradScaleSq });
        }
        if (releaseFn) for (const part of sqNormParts) releaseFn(part);

        // Aggregate per-layer gradient norms (layer.N.* → layerN norm)
        perLayerGradNorms = {};
        const layerNormSqs = new Map<number, number>();
        let embedNormSq = 0;
        let headNormSq = 0;
        for (const { name, normSq } of perParamNorms) {
          const layerMatch = name.match(/^layer\.(\d+)\./);
          if (layerMatch) {
            const idx = parseInt(layerMatch[1], 10);
            layerNormSqs.set(idx, (layerNormSqs.get(idx) ?? 0) + normSq);
          } else if (name.startsWith("head.")) {
            headNormSq += normSq;
          } else {
            embedNormSq += normSq;
          }
        }
        if (embedNormSq > 0) perLayerGradNorms["embed"] = Math.sqrt(embedNormSq);
        for (const [idx, sq] of [...layerNormSqs.entries()].sort((a, b) => a[0] - b[0])) {
          perLayerGradNorms[String(idx)] = Math.sqrt(sq);
        }
        if (headNormSq > 0) perLayerGradNorms["head"] = Math.sqrt(headNormSq);

        if (traceEnabled) {
          const sorted = [...perParamNorms]
            .map(({ name: n, normSq: sq }) => [n, Math.sqrt(sq)] as const)
            .sort((a, b) => b[1] - a[1]);
          const top5 = sorted.slice(0, 5).map(([n, v]) => `${n}=${v.toFixed(2)}`).join(", ");
          console.log(`  [trace] grad norms (top 5): ${top5}`);
        }

        if (stepNum % 500 === 0 && perParamNorms.length > 0) {
          const layerNorms = perParamNorms
            .map(({ name, normSq }) => ({ name, norm: Math.sqrt(normSq) }))
            .sort((a, b) => b.norm - a.norm);

          console.log(`  [diag] per-layer grad norms (step ${stepNum}):`);
          const top10 = layerNorms.slice(0, 10);
          for (const { name, norm } of top10) {
            console.log(`    ${name}: ${norm.toFixed(4)}`);
          }
          if (layerNorms.length > 0) {
            const topParam = layerNorms[0];
            const totalNorm = Math.sqrt(layerNorms.reduce((s, l) => s + l.norm * l.norm, 0));
            const pct = ((topParam.norm / totalNorm) * 100).toFixed(1);
            console.log(`    -> dominant: ${topParam.name} (${pct}% of total norm)`);
          }
        }
      }

      // NaN guard: if forward (loss) or backward (gradNorm) produced non-finite values,
      // reduce lossScale and skip the optimizer update.
      if (nanDetected || !isFinite(gradNorm)) {
        if (useLossScaling) {
          // Dynamic loss scaling: halve the scale and retry
          const oldScale = lossScale;
          lossScale = Math.max(1.0, lossScale / 2);
          lossScaleReductions++;
          console.warn(`  [loss_scale] NaN/overflow at step ${stepNum} — reducing scale ${oldScale} -> ${lossScale}`);
          if (!gradNanLogged) {
            emitEvent({
              step: stepNum,
              level: "warn",
              kind: "grad_overflow",
              message: `NaN or gradient overflow detected; reducing loss scale to ${lossScale}`,
              payload: { lossScale, lossScaleReductions, useLossScaling: true, nanFromForward: nanDetected },
            });
            gradNanLogged = true;
          }
          scaleSuccessCount = 0;
        } else {
          console.warn(`  [warn] grad_norm=NaN at step ${stepNum} — skipping optimizer update`);
          gradNanCount++;
          // Emit event for every NaN (rate-limited: first 5, then every 10th)
          if (gradNanCount <= 5 || gradNanCount % 10 === 0) {
            emitEvent({
              step: stepNum,
              level: "warn",
              kind: "grad_norm_nan",
              message: `grad_norm became non-finite; skipping optimizer update (step ${stepNum})`,
              payload: { nanCount: gradNanCount, totalSteps: stepNum },
            });
          }
        }
        nanDetected = true; // Ensure optimizer step is skipped
      } else if (useLossScaling) {
        // Track consecutive successes for scale growth
        scaleSuccessCount++;
        if (scaleSuccessCount >= SCALE_GROWTH_INTERVAL) {
          lossScale *= 2;
          scaleSuccessCount = 0;
        }
      }

      // Spike detection: skip optimizer step when grad_norm exceeds absolute threshold.
      // This prevents pathological batches from disrupting Adam's momentum estimates.
      // spikeThreshold is an absolute grad_norm cap (e.g. 50 means skip when > 50).
      if (!nanDetected && spikeThreshold > 0 && gradNorm > spikeThreshold) {
        spikeSkips++;
        if (useTotalSumSq) {
          // Heuristic guard: repeated near-identical giant spike norms usually
          // indicate a corrupted fast-path reduction, not meaningful gradient dynamics.
          const sameAsPrev = Number.isFinite(prevSpikeNorm) &&
            Math.abs(gradNorm - prevSpikeNorm) <= Math.max(1e-3, Math.abs(prevSpikeNorm) * 0.01);
          repeatedSpikeNormStreak = sameAsPrev ? (repeatedSpikeNormStreak + 1) : 1;
          prevSpikeNorm = gradNorm;

          // Guard 2: if we see several massive spikes in a short window, treat it
          // as fast-path corruption and fall back immediately.
          const hugeSpikeThreshold = spikeThreshold * 200;
          if (gradNorm >= hugeSpikeThreshold) {
            if (hugeSpikeWindowStartStep < 0 || (stepNum - hugeSpikeWindowStartStep) > 64) {
              hugeSpikeWindowStartStep = stepNum;
              hugeSpikeCountInWindow = 1;
            } else {
              hugeSpikeCountInWindow++;
            }
          }

          if (repeatedSpikeNormStreak >= 6 || hugeSpikeCountInWindow >= 3) {
            const streakBeforeDisable = repeatedSpikeNormStreak;
            const hugeSpikesBeforeDisable = hugeSpikeCountInWindow;
            useTotalSumSq = false;
            repeatedSpikeNormStreak = 0;
            prevSpikeNorm = Number.NaN;
            hugeSpikeWindowStartStep = -1;
            hugeSpikeCountInWindow = 0;
            console.warn(
              `  [grad_norm] suspicious giant spike pattern detected at step ${stepNum}; ` +
              `switching to per-parameter grad norm fallback`,
            );
            emitEvent({
              step: stepNum,
              level: "warn",
              kind: "grad_norm_fastpath_disabled",
              message: "disabled totalSumSq grad norm fast path after giant spike pattern",
              payload: {
                threshold: spikeThreshold,
                gradNorm,
                mode: "per_param_fallback",
                repeatedSpikeNormStreak: streakBeforeDisable,
                hugeSpikeCountInWindow: hugeSpikesBeforeDisable,
              },
            });
          }
        }
        const prevScale = runtimeLrScale;
        if (spikeLrBackoff < 1.0) {
          runtimeLrScale = Math.max(spikeLrMinScale, runtimeLrScale * spikeLrBackoff);
          lr = Math.max(lrMin * spikeLrMinScale, baseLr * runtimeLrScale);
          if (setOptimizerLrFn) setOptimizerLrFn(lr);
        }
        const batchHashHex = `0x${(stepBatchHash >>> 0).toString(16).padStart(8, "0")}`;
        const tokRange = Number.isFinite(stepBatchTokMin) && Number.isFinite(stepBatchTokMax)
          ? `${stepBatchTokMin}:${stepBatchTokMax}`
          : "n/a";
        console.warn(
          `  [spike] grad_norm=${gradNorm.toFixed(1)} > ${spikeThreshold} — skipping step ${stepNum} ` +
          `(${spikeSkips} total skips) | lr_scale ${prevScale.toFixed(4)}->${runtimeLrScale.toFixed(4)} ` +
          `| batch_hash=${batchHashHex} tok_range=${tokRange}`,
        );
        emitEvent({
          step: stepNum,
          level: "warn",
          kind: "spike_skip",
          message: `spike skip: grad_norm=${gradNorm.toFixed(1)} exceeded threshold ${spikeThreshold}`,
          payload: {
            gradNorm,
            spikeThreshold,
            spikeSkips,
            lrScalePrev: prevScale,
            lrScaleNext: runtimeLrScale,
            batchHash: batchHashHex,
            tokenRange: tokRange,
          },
        });
        nanDetected = true; // reuse the skip path
        spikeDetectedThisStep = true;
        stepsSinceLastSpike = 0;
      } else {
        repeatedSpikeNormStreak = 0;
        prevSpikeNorm = Number.NaN;
      }
    }
    if (Number.isNaN(gradNorm)) nanCount++;
    else if (!Number.isFinite(gradNorm)) infCount++;

    // LR scale recovery: after enough steps without a spike, restore lr toward 1.0.
    // Note: NaN grad steps do NOT reset the counter — only actual gradient spikes
    // (which set stepsSinceLastSpike=0 above) reset it. NaN is a separate phenomenon
    // (~8% baseline) and would prevent recovery if counted.
    if (runtimeLrScale < 1.0 && !spikeDetectedThisStep) {
      stepsSinceLastSpike++;
      if (stepsSinceLastSpike >= spikeLrRecoverySteps) {
        const prevScale = runtimeLrScale;
        runtimeLrScale = Math.min(1.0, runtimeLrScale / spikeLrBackoff);
        lr = Math.max(lrMin * spikeLrMinScale, baseLr * runtimeLrScale);
        if (setOptimizerLrFn) setOptimizerLrFn(lr);
        stepsSinceLastSpike = 0;
        console.log(
          `  [spike_recovery] lr_scale ${prevScale.toFixed(4)}->${runtimeLrScale.toFixed(4)} ` +
          `after ${spikeLrRecoverySteps} steps without spike`,
        );
      }
    }

    const _t4 = capturePhaseTimings ? performance.now() : 0;

    // Clip coefficient for optimizer-side gradient scaling.
    let clipCoef = 1.0;
    let effectiveGradScale = gradScaleFactor;
    if (!nanDetected && gradClip > 0 && gradNorm > gradClip) {
      clipCoef = gradClip / gradNorm;
      clippedSteps++;
      effectiveGradScale *= clipCoef;

      // Safety: spot-check the largest gradient tensors for Inf/NaN values.
      // In f16, Inf * scale = Inf — scaling/clipping cannot fix overflow.
      // Check 3 largest tensors by computing finite checks on GPU.
      // Batch all GPU ops first, then single flush on first .data access.
      let top1: TensorData | null = null;
      let top2: TensorData | null = null;
      let top3: TensorData | null = null;
      let top1Size = -1;
      let top2Size = -1;
      let top3Size = -1;
      const considerTopGrad = (grad: TensorData, size: number): void => {
        if (size > top1Size) {
          top3 = top2; top3Size = top2Size;
          top2 = top1; top2Size = top1Size;
          top1 = grad; top1Size = size;
          return;
        }
        if (size > top2Size) {
          top3 = top2; top3Size = top2Size;
          top2 = grad; top2Size = size;
          return;
        }
        if (size > top3Size) {
          top3 = grad; top3Size = size;
        }
      };
      for (let i = 0; i < paramEntries.length; i++) {
        const variable = paramEntries[i][1];
        if (variable.grad) {
          considerTopGrad(variable.grad, paramSizes[i]);
        }
      }
      if (checkFiniteFn) {
        const finiteChecks: TensorData[] = [];
        if (top1) finiteChecks.push(checkFiniteFn(top1));
        if (top2) finiteChecks.push(checkFiniteFn(top2));
        if (top3) finiteChecks.push(checkFiniteFn(top3));
        // Single flush on first .data access (all checks recorded above)
        for (const chk of finiteChecks) {
          if ((chk.data as Float32Array)[0] !== 0) {
            console.warn(`  [warn] Inf/NaN in gradients (norm=${gradNorm.toFixed(1)}) — skipping optimizer update`);
            if (!gradNanLogged) {
              emitEvent({
                step: stepNum,
                level: "warn",
                kind: "grad_non_finite",
                message: `non-finite gradients detected; skipping optimizer update`,
                payload: { gradNorm },
              });
              gradNanLogged = true;
            }
            nanDetected = true;
            break;
          }
        }
        if (releaseFn) for (const chk of finiteChecks) releaseFn(chk);
      } else {
        const spotResults: { s: TensorData; g2: TensorData }[] = [];
        const pushSpotResult = (g: TensorData | null): void => {
          if (!g) return;
          const g2 = backend.mul(g, g);
          spotResults.push({ s: backend.sum(g2), g2 });
        };
        pushSpotResult(top1);
        pushSpotResult(top2);
        pushSpotResult(top3);
        // Single flush on first .data access (all ops recorded above)
        for (const { s } of spotResults) {
          if (!isFinite((s.data as Float32Array)[0])) {
            console.warn(`  [warn] Inf in gradients (norm=${gradNorm.toFixed(1)}) — skipping optimizer update`);
            if (!gradNanLogged) {
              emitEvent({
                step: stepNum,
                level: "warn",
                kind: "grad_non_finite",
                message: `non-finite gradients detected via spot-check; skipping optimizer update`,
                payload: { gradNorm },
              });
              gradNanLogged = true;
            }
            nanDetected = true;
            break;
          }
        }
        // Cleanup intermediates
        if (releaseFn) {
          for (const { s, g2 } of spotResults) { releaseFn(g2); releaseFn(s); }
        }
      }
    }

    const _t4b = capturePhaseTimings ? performance.now() : 0;

    if (!nanDetected) {
      // Scale down embedding gradients to prevent them dominating the optimization.
      // Embedding grads (wte, wpe) are typically 100–600× larger than layer grads.
      if (embGradScale > 0 && embGradScale < 1 && embParamIndices.length > 0) {
        for (const idx of embParamIndices) {
          const grad = paramVars[idx].grad;
          if (grad) paramVars[idx].grad = backend.scale(grad, embGradScale);
        }
      }

      // Fast path: consume cached param entries directly.
      if (optimizerStepParamEntries) {
        optimizerStepParamEntries(paramEntries, effectiveGradScale);
      } else {
        // Map fallback path for optimizers that don't support entry stepping.
        gradMap.clear();
        for (let i = 0; i < paramVars.length; i++) {
          const grad = paramVars[i].grad;
          if (grad) gradMap.set(paramNames[i], grad);
        }
        optimizer.step(paramDataMap, gradMap, effectiveGradScale);
      }
      if (phaseSyncProfile && syncGpuFn) syncGpuFn();
    }
    const _t5 = capturePhaseTimings ? performance.now() : 0;

    // Zero gradients — explicitly release GPU buffers for param grads
    for (let i = 0; i < paramVars.length; i++) {
      const variable = paramVars[i];
      if (variable.grad && releaseFn) releaseFn(variable.grad);
      variable.grad = null;
    }
    // Note: tape intermediates already released in the accumulation loop above

    // Flush pending GPU ops (optimizer commands + deferred buffer releases).
    if (flushFn) flushFn();

    // Adaptive sync/GC policy: configurable cadence instead of every step.
    // syncEvery=0 (default) triggers sync only on pool pressure.
    // syncEvery>0 forces periodic sync every N steps.
    const syncEvery = trainConfig.syncEvery;
    const gcEvery = trainConfig.gcEvery;
    // GPU memory diagnostics:
    // - trace mode: detailed cadence for debugging
    // - non-trace mode: sparse snapshots to minimize training-loop overhead
    const shouldLogGpuMem = !!gpuMemStatsFn && (
      traceEnabled
        ? (stepNum % 5 === 0)
        : (stepNum === totalIters || stepNum % 50 === 0)
    );
    let memStatsStep: any | null = memStatsCache;
    const adaptiveMemControl = gcEvery <= 0 || syncEvery <= 0;
    // Probe every step when under memory pressure, otherwise every POLL_EVERY steps
    const underMemPressure = !!(memStatsCache && (memStatsCache.liveAllocs ?? 0) > ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD * 0.6);
    const shouldProbeAdaptiveMem = adaptiveMemControl && (
      stepNum === 1 ||
      stepNum === totalIters ||
      underMemPressure ||
      stepNum - lastMemStatsProbeStep >= ADAPTIVE_MEM_STATS_POLL_EVERY
    );
    if (gpuMemStatsFn && (shouldLogGpuMem || shouldProbeAdaptiveMem)) {
      memStatsStep = gpuMemStatsFn();
      memStatsCache = memStatsStep;
      lastMemStatsProbeStep = stepNum;
    }

    const needGc = typeof globalThis.gc === "function" && (
      gcEvery > 0 ? stepNum % gcEvery === 0 :
      // Adaptive: GC when deferred releases pile up OR live allocs are high.
      // Without the liveAllocs check, FinalizationRegistry-based cleanup never
      // triggers because GC doesn't run → GPU buffers for unreachable tensors
      // stay "live" → OOM on constrained VRAM (e.g. L4 23GB).
      !!(memStatsStep && (
        memStatsStep.deferredReleases > 50 ||
        (memStatsStep.liveAllocs ?? 0) > ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD * 0.75
      ))
    );
    if (needGc) {
      (globalThis as any).gc();
      await new Promise<void>(resolve => setImmediate(resolve));
      if (gpuMemStatsFn) {
        memStatsStep = gpuMemStatsFn();
        memStatsCache = memStatsStep;
        lastMemStatsProbeStep = stepNum;
      }
    }

    // SyncGpu: flush GC'd deferred releases, then WAIT for GPU completion.
    // This ensures output pool regions are reusable without OOM.
    const adaptiveSyncPressure = !!(
      memStatsStep &&
      (
        memStatsStep.deferredReleases > ADAPTIVE_SYNC_DEFERRED_THRESHOLD ||
        (memStatsStep.pendingDestroys ?? 0) > ADAPTIVE_SYNC_PENDING_THRESHOLD ||
        (memStatsStep.liveAllocs ?? 0) > ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD
      )
    );
    // Emergency sync: bypass min-interval when liveAllocs approaching hard cap
    const emergencySyncPressure = !!(memStatsStep && (memStatsStep.liveAllocs ?? 0) > ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD * 0.85);
    const needSync = syncEvery === 1 ? true :
      syncEvery > 0 ? stepNum % syncEvery === 0 :
      // Adaptive: rate-limit syncs under sustained pressure to avoid serializing every step.
      // But bypass rate limit when approaching OOM (emergency sync).
      emergencySyncPressure || (adaptiveSyncPressure && (stepNum - lastAdaptiveSyncStep >= ADAPTIVE_SYNC_MIN_INTERVAL || stepNum === totalIters));
    if (needSync) {
      let didSync = false;
      if (syncGpuFn) {
        syncGpuFn();
        didSync = true;
      } else if (flushFn) {
        flushFn();
        didSync = true;
      }

      // Re-sample memory stats after sync so purge decisions use current pressure.
      if (didSync && gpuMemStatsFn) {
        memStatsStep = gpuMemStatsFn();
        memStatsCache = memStatsStep;
        lastMemStatsProbeStep = stepNum;
      }

      const liveAllocsAfterSync = memStatsStep?.liveAllocs ?? 0;
      const shouldPurgePools = !!(
        purgeBufferPoolsFn &&
        liveAllocsAfterSync > ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD &&
        (stepNum - lastAdaptivePurgeStep >= ADAPTIVE_PURGE_MIN_INTERVAL || stepNum === totalIters)
      );
      if (shouldPurgePools) {
        console.warn(
          `  [gpu_mem] liveAllocs=${liveAllocsAfterSync} exceeded purge threshold ${ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD}; purging pools`,
        );
        emitEvent({
          step: stepNum,
          level: "warn",
          kind: "gpu_mem_purge",
          message: `purging GPU pools: liveAllocs=${liveAllocsAfterSync} exceeded threshold ${ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD}`,
          payload: {
            liveAllocs: liveAllocsAfterSync,
            purgeThreshold: ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD,
            deferredReleases: memStatsStep?.deferredReleases ?? null,
            pendingDestroys: memStatsStep?.pendingDestroys ?? null,
          },
        });
        purgeBufferPoolsFn();

        // Manual GC trigger to accelerate FinalizationRegistry cleanup of GPU handles.
        // Requires node --expose-gc.
        if (typeof global !== "undefined" && (global as any).gc) {
          (global as any).gc();
        }

        lastAdaptivePurgeStep = stepNum;
        if (gpuMemStatsFn) {
          memStatsStep = gpuMemStatsFn();
          memStatsCache = memStatsStep;
          lastMemStatsProbeStep = stepNum;
        }
      }

      if (syncEvery <= 0 && didSync) {
        lastAdaptiveSyncStep = stepNum;
      }
    }
    const _t6 = capturePhaseTimings ? performance.now() : 0;

    // Direct host/GPU split for the performance program's G1a gate. Helios
    // measures wall time inside actual completion waits, including blocking
    // timestamp readback. Subtracting that from the pre-metrics step interval
    // exposes the non-blocking control-plane portion: data preparation,
    // autograd traversal, operation construction, allocation, packing and
    // submission, optimizer orchestration, and memory policy. This is useful
    // work today rather than generic "overhead", and a future scheduler may
    // overlap it with device execution.
    const gpuStepStats = traceEnabled ? getGpuStepStatsFn?.() : undefined;
    const coreStepElapsedMs = capturePhaseTimings ? _t6 - stepStart : 0;
    const gpuBlockingMs = gpuStepStats?.gpuBlockingTimeMs ?? 0;
    const hostBuildMs = capturePhaseTimings
      ? Math.max(0, coreStepElapsedMs - gpuBlockingMs)
      : 0;

    if (traceEnabled) {
      const gpuOps = "gpuOpsThisStep" in backend ? ` gpu_ops=${(backend as any).gpuOpsThisStep}` : "";
      console.log(`  [trace] phase_mode=${phaseSyncProfile ? "gpu_sync" : "enqueue"} data=${dataLoadMs.toFixed(0)}ms fwd=${fwdMs.toFixed(0)}ms bwd=${bwdMs.toFixed(0)}ms gradnorm=${(_t4-_t3).toFixed(0)}ms clip=${(_t4b-_t4).toFixed(0)}ms optim=${(_t5-_t4b).toFixed(0)}ms flush=${(_t6-_t5).toFixed(0)}ms${gpuOps}`);
      if (gpuStepStats?.profilingEnabled) {
        const top = (
          rows: Array<{ name: string; count: number; gpuTimeUs: number }>,
          limit: number,
        ): string => rows.slice(0, limit).map(({ name, count, gpuTimeUs }) =>
          gpuStepStats.timingEnabled
            ? `${name}:${count}/${gpuTimeUs.toFixed(1)}us`
            : `${name}:${count}`,
        ).join(",");
        const timing = gpuStepStats.timingEnabled
          ? ` timestamped=${gpuStepStats.timestampedFlushes}` +
            ` batch_gpu_us=${gpuStepStats.batchGpuTimeUs.toFixed(1)}` +
            ` dispatch_gpu_us=${gpuStepStats.dispatchGpuTimeUs.toFixed(1)}` +
            ` host_build_ms=${hostBuildMs.toFixed(1)}` +
            ` gpu_blocking_ms=${gpuBlockingMs.toFixed(1)}` +
            ` core_step_ms=${coreStepElapsedMs.toFixed(1)}`
          : "";
        console.log(
          `  [gpu_ops] flushes=${gpuStepStats.flushes} waited=${gpuStepStats.waitedFlushes}` +
          ` dgc=${gpuStepStats.dgcFlushes} ops_per_flush=${gpuStepStats.operationsPerFlush.toFixed(1)}` +
          timing + ` kinds=${top(gpuStepStats.byKind, 12)} kernels=${top(gpuStepStats.byKernel, 16)}`,
        );
      }
    }

    if (shouldLogGpuMem) {
      const stats = memStatsStep ?? gpuMemStatsFn!();
      const breakdown = traceEnabled && poolBreakdownFn ? ` | ${poolBreakdownFn(8)}` : "";
      const allocStr = stats.liveAllocs != null ? ` | allocs: ${stats.liveAllocs} live (${stats.totalAllocs} total, ${stats.totalAllocMB}MB)` : "";
      const nativeAllocStr = stats.trackedVkMemoryAllocations != null
        ? ` | vkMem: ${stats.trackedVkMemoryAllocations} tracked (${stats.individualBuffers} individual, ${stats.slabBuffers} slab buffers, temp=${stats.tempSlabCount} slabs/${(stats.tempSlabLiveBytes/1024/1024).toFixed(0)}MB live/${(stats.tempSlabCapacityBytes/1024/1024).toFixed(0)}MB cap, reuse=${stats.slabFreeRangeReuses}, fallback=${stats.slabFallbacks}, freeOvf=${stats.slabFreeRangeOverflows})`
        : "";
      const diagStr = stats.diagAllocsThisStep != null ? ` | glt_allocs=${stats.diagAllocsThisStep} glt_rel=${stats.diagReleasesThisStep} fr=${stats.diagFrReleasesThisStep}` : "";
      let flowStr = "";
      if (stats.flowNewCreates != null) {
        const d = (k: string) => (stats as any)[k] - ((lastFlowStats as any)?.[k] ?? 0);
        flowStr = ` | Δflow: new=${d("flowNewCreates")} dest=${d("flowDestroys")} oHit=${d("flowOutputPoolHits")} oMiss=${d("flowOutputPoolMisses")} oRet=${d("flowOutputPoolReturns")} oOvf=${d("flowOutputPoolOverflows")} bHit=${d("flowBufferPoolHits")} gHit=${d("flowEnsureGpuHits")} gUp=${d("flowEnsureGpuUploads")}`;
        lastFlowStats = { ...stats };
      }
      console.log(`  [gpu_mem] bufPool: ${stats.bufferPoolEntries} (${(stats.bufferPoolBytes/1024/1024).toFixed(1)}MB) | outPool: ${stats.outputPoolEntries}/${stats.outputPoolSizeClasses ?? "?"}cls (${(stats.outputPoolBytes/1024/1024).toFixed(1)}MB) | deferred: ${stats.deferredReleases} | pending: ${stats.pendingDestroys ?? 0}${allocStr}${nativeAllocStr}${diagStr}${flowStr}${breakdown}`);
    }

    // Metrics
    const stepElapsed = performance.now() - stepStart;
    const metrics: StepMetrics = {
      step: stepNum,
      loss: lossVal,
      lr,
      gradNorm,
      elapsed_ms: stepElapsed,
      tokens_per_sec: tokensProcessedPerStep / (stepElapsed / 1000),
      ms_per_iter: stepElapsed,
      // Clipping telemetry
      clip_coef: clipCoef,
      clip_pct: (clippedSteps / stepNum) * 100,
    };
    if (deps.rcrUl) {
      metrics.positive_ce_loss = positiveCeLossVal;
      metrics.negative_ul_loss = negativeUlLossVal;
      metrics.ul_weight = deps.rcrUl.weight;
      metrics.positive_supervised_token_mass = positiveSupervisedTokenMass;
      metrics.negative_penalty_position_mass = negativePenaltyPositionMass;
      metrics.negative_examples_with_penalty = negativeExamplesWithPenalty;
      metrics.negative_first_penalty_target_position = Number.isFinite(negativeFirstPenaltyTargetPosition)
        ? negativeFirstPenaltyTargetPosition
        : undefined;
      metrics.negative_last_penalty_target_position = Number.isFinite(negativeLastPenaltyTargetPosition)
        ? negativeLastPenaltyTargetPosition
        : undefined;
      metrics.negative_mean_bad_token_probability = negativeBadProbabilityWeighted /
        Math.max(negativeBadProbabilityMaskMass, 1);
      metrics.negative_max_bad_token_probability = negativeMaxBadProbability;
      metrics.grad_norm_before_clip = gradNorm;
      metrics.grad_norm_after_clip = gradNorm * clipCoef;
      metrics.nan_count = nanCount;
      metrics.inf_count = infCount;
      metrics.timing_positive_fwd_ms = positiveFwdMs;
      metrics.timing_positive_bwd_ms = positiveBwdMs;
      metrics.timing_negative_fwd_ms = negativeFwdMs;
      metrics.timing_negative_bwd_ms = negativeBwdMs;
    }
    const hostMemory = process.memoryUsage();
    metrics.host_rss_mb = Math.round(hostMemory.rss / 1024 / 1024);
    metrics.host_heap_used_mb = Math.round(hostMemory.heapUsed / 1024 / 1024);
    metrics.host_external_mb = Math.round(hostMemory.external / 1024 / 1024);
    metrics.host_array_buffers_mb = Math.round(hostMemory.arrayBuffers / 1024 / 1024);
    if (capturePhaseTimings) {
      metrics.timing_fwd_ms = fwdMs;
      metrics.timing_bwd_ms = bwdMs;
      metrics.timing_grad_norm_ms = _t4 - _t3;
      metrics.timing_grad_clip_ms = _t4b - _t4;
      metrics.timing_optim_ms = _t5 - _t4b;
      metrics.timing_flush_ms = _t6 - _t5;
      metrics.timing_data_ms = dataLoadMs;
      metrics.timing_host_build_ms = hostBuildMs;
      metrics.timing_gpu_blocking_ms = gpuBlockingMs;
      metrics.timing_core_step_ms = coreStepElapsedMs;
    }
    if ("gpuOpsThisStep" in backend) {
      metrics.gpu_ops_count = (backend as any).gpuOpsThisStep;
    }
    if (perLayerGradNorms) {
      metrics.per_layer_grad_norms = JSON.stringify(perLayerGradNorms);
    }

    // Symbio metrics (only when symbio is enabled — zero overhead otherwise)
    if (symbioEnabled && cusumDash && symbioCollector) {
      const clipPctVal = (clippedSteps / stepNum) * 100;
      const stepInfo: TrainerStepInfo = {
        step: stepNum,
        loss: lossVal,
        gradNorm,
        clipPct: clipPctVal,
        tokensPerSec: metrics.tokens_per_sec,
        valLoss: metrics.valLoss,
      };

      // CUSUM update (every step)
      const cusumResult = cusumDash.update(stepInfo);
      metrics.cusum_grad = cusumResult.cusum_grad;
      metrics.cusum_clip = cusumResult.cusum_clip;
      metrics.cusum_tps = cusumResult.cusum_tps;
      metrics.cusum_val = cusumResult.cusum_val;
      metrics.cusum_alerts = cusumResult.cusum_alerts;
      if (cusumResult.cusum_alert_reason) {
        metrics.cusum_alert_reason = cusumResult.cusum_alert_reason;
        console.log(`  [symbio] CUSUM alert: ${cusumResult.cusum_alert_reason}`);
      }

      // Population dynamics adaptation (loss + LR + CUSUM driven)
      if (populationDynamics && searchOrchestrator) {
        const dyn = populationDynamics.update({
          step: stepNum,
          loss: lossVal,
          lr,
          cusumAlerts: cusumResult.cusum_alerts,
        });
        searchOrchestrator.setAdaptiveControls({
          populationSize: dyn.effectivePopulationSize,
          mutationRate: dyn.mutationRate,
        });
        if (dyn.changed) {
          console.log(
            `  [symbio] population dynamics → pop=${dyn.effectivePopulationSize} (scale=${dyn.populationScale.toFixed(2)}) ` +
            `mut=${dyn.mutationRate.toFixed(3)} | explore=${dyn.explorePressure.toFixed(2)} converge=${dyn.convergePressure.toFixed(2)} ` +
            `plateau=${dyn.plateauPressure.toFixed(2)} cusum=${dyn.cusumPressure.toFixed(2)} (${dyn.reason})`,
          );
        }
      }

      // Record loss for population entropy sliding window
      symbioCollector.recordLoss(lossVal);

      // Expensive metrics (every metricsInterval steps)
      if (stepNum % symbioConfig.metricsInterval === 0) {
        // Collect TensorData for weight params
        const paramTDs = new Map<string, TensorData>();
        for (const [name, v] of paramEntries) paramTDs.set(name, v.data);

        const activation = activeModelConfig.ffnActivation ?? "gelu";
        const expensiveMetrics = symbioCollector.collect(
          paramTDs, lossVal, totalParams, activation, activeModelConfig.nLayer, metrics.valLoss,
        );
        Object.assign(metrics, expensiveMetrics);
      }
    }

    // Search orchestrator: record step and switch candidates when evaluation period ends
    if (searchOrchestrator && !searchOrchestrator.isDone) {
      const candidate = searchOrchestrator.currentCandidate;
      if (candidate) {
        metrics.symbio_candidate_id = candidate.id;
        metrics.symbio_candidate_name = candidate.name;
        metrics.symbio_candidate_activation = candidate.activationGraph
          ? nameGraph(candidate.activationGraph) : candidate.activation;
        metrics.symbio_candidate_parent_id = candidate.parentId ?? undefined;
        metrics.symbio_candidate_parent_name = candidate.parentName ?? undefined;
        metrics.symbio_generation = searchOrchestrator.generation;
        metrics.architecture_diversity = searchOrchestrator.architectureDiversity;
        if (candidate.activationGraph) {
          metrics.symbio_activation_graph = serializeGraph(candidate.activationGraph);
        }
        if (candidate.mutationApplied) {
          metrics.symbio_mutation_applied = candidate.mutationApplied;
        }

        const candidateDone = searchOrchestrator.recordStep(
          lossVal,
          metrics.valLoss,
          metrics.fitness_score,
        );

        if (kuramotoFusion && fusionShadow) {
          const fusionState = kuramotoFusion.update({
            loss: lossVal,
            lr,
            cusumAlerts: metrics.cusum_alerts ?? 0,
            switchedCandidate: candidateDone,
          });
          fusionShadow.step(params, fusionState.fusionAlpha, symbioConfig.fusionShadowEma);
          if ((step + 1) % Math.max(1, symbioConfig.metricsInterval) === 0) {
            console.log(
              `  [symbio] fusion α=${fusionState.fusionAlpha.toFixed(4)} sync=${fusionState.sync.toFixed(3)} ` +
              `order=${fusionState.order.toFixed(3)} Δθ=${fusionState.phaseGap.toFixed(3)}`,
            );
          }
        }

        if (candidateDone) {
          console.log(`  [symbio search] candidate ${candidate.name} done: bestLoss=${candidate.bestLoss.toFixed(4)} bestVal=${candidate.bestValLoss === Infinity ? "N/A" : candidate.bestValLoss.toFixed(4)} fitness=${candidate.fitnessScore === -Infinity ? "N/A" : candidate.fitnessScore.toFixed(4)}`);
          const nextActivation = searchOrchestrator.advance();

          if (nextActivation !== null) {
            // Switch to next candidate: preserve weights/optimizer when possible.
            const nextCandidate = searchOrchestrator.currentCandidate;
            const ffnDim = symbioConfig.constantFfnDimAcrossCandidates
              ? (modelConfig.ffnDim ?? ffnDimForActivation(modelConfig.ffnActivation ?? "gelu", modelConfig.nEmbd, modelConfig.ffnDim))
              : ffnDimForActivation(nextActivation, modelConfig.nEmbd, modelConfig.ffnDim);
            activeModelConfig = {
              ...modelConfig,
              ffnActivation: nextActivation as any,
              ffnDim,
              activationGraph: nextCandidate?.activationGraph ?? undefined,
            };
            rng.seed(trainConfig.seed + (step + 1)); // seed only affects fresh/unmatched params
            if (symbioConfig.preserveWeightsAcrossCandidates) {
              const prevParams = params;
              const transferred = initGPTWithTransferredWeights(activeModelConfig, backend, rng as any, prevParams);
              params = transferred.params;
              totalParams = countParams(params);
              refreshParamCaches();
              recomputeEmbIndices();
              if (symbioConfig.carryOptimizerStateAcrossCandidates) {
                const optCarry = carryOptimizerStateAcrossSwitch(optimizer, prevParams, params);
                console.log(
                  `  [symbio switch] params exact=${transferred.stats.exactCopies} partial=${transferred.stats.partialCopies} fresh=${transferred.stats.initializedFresh} ` +
                  `| opt exact=${optCarry.copiedBuffers} partial=${optCarry.partialBuffers} fresh=${optCarry.freshBuffers}`,
                );
              } else {
                optimizer.loadStateDict({ step: 0, buffers: new Map() });
                console.log(
                  `  [symbio switch] params exact=${transferred.stats.exactCopies} partial=${transferred.stats.partialCopies} fresh=${transferred.stats.initializedFresh} | opt reset`,
                );
              }
            } else {
              params = initGPT(activeModelConfig, backend, rng as any);
              totalParams = countParams(params);
              refreshParamCaches();
              recomputeEmbIndices();
              optimizer.loadStateDict({ step: 0, buffers: new Map() });
            }
            clearForwardCache();
            fusionShadow?.rebind(params);
            const graphName = nextCandidate?.activationGraph ? nameGraph(nextCandidate.activationGraph) : nextActivation;
            console.log(
              `  [symbio search] → gen=${searchOrchestrator.generation} candidate=${nextCandidate?.name} (${nextCandidate?.id}) ` +
              `activation=${graphName} params=${totalParams.toLocaleString()} pop=${searchOrchestrator.effectivePopulationSize} mut=${searchOrchestrator.effectiveMutationRate.toFixed(3)}`,
            );
          } else {
            // Search complete
            const winner = searchOrchestrator.getWinner();
            console.log(`\n── symbio search complete ──`);
            if (winner) {
              const winnerName = winner.activationGraph ? nameGraph(winner.activationGraph) : winner.activation;
              console.log(`winner: ${winnerName} (${winner.id}) | bestVal=${winner.bestValLoss === Infinity ? "N/A" : winner.bestValLoss.toFixed(4)} | fitness=${winner.fitnessScore === -Infinity ? "N/A" : winner.fitnessScore.toFixed(4)}`);
            }

            // Write artifacts to run directory
            const fs = await import("node:fs/promises");
            const path = await import("node:path");
            if (symbioConfig.writeReport) {
              const report = searchOrchestrator.getReport();
              await fs.writeFile(path.join(runDir, "symbio-search-report.md"), report);
              console.log(`  report: ${runDir}/symbio-search-report.md`);
            }
            if (symbioConfig.writeCandidates) {
              const jsonl = searchOrchestrator.getCandidatesJSONL();
              await fs.writeFile(path.join(runDir, "symbio-candidates.jsonl"), jsonl);
            }
            if (symbioConfig.writeSummary) {
              const summary = searchOrchestrator.getSummary();
              await fs.writeFile(path.join(runDir, "symbio-search-summary.json"), JSON.stringify(summary, null, 2));
            }
          }
        }
      }
    }

    const shouldSampleGpuMetrics =
      !!deps.rcrUl ||
      traceEnabled ||
      metrics.step === 1 ||
      metrics.step === totalIters ||
      (metrics.step % Math.max(logEvery, GPU_METRICS_SAMPLE_EVERY) === 0);

    // GPU stats (queried at most every 5s via nvidia-smi).
    // Keep sparse sampling in non-trace mode to avoid per-step async overhead.
    let gpuStats: GpuStats | null = null;
    if (!_gpuStatsDisabled && shouldSampleGpuMetrics) {
      gpuStats = await queryGpuStats();
    }
    if (gpuStats) {
      metrics.gpu_util_pct = gpuStats.utilPct;
      metrics.gpu_vram_used_mb = gpuStats.vramUsedMb;
      metrics.gpu_vram_total_mb = gpuStats.vramTotalMb;
    }
    if (gpuMemStatsFn && shouldSampleGpuMetrics) {
      const memStats = memStatsStep ?? gpuMemStatsFn();
      metrics.gpu_mem_pool_mb = Math.round((memStats.bufferPoolBytes + memStats.outputPoolBytes) / 1024 / 1024);
      metrics.gpu_live_allocs = memStats.liveAllocs;
      metrics.gpu_vk_memory_allocations = memStats.trackedVkMemoryAllocations;
      metrics.gpu_slab_buffers = memStats.slabBuffers;
      metrics.gpu_temp_slab_count = memStats.tempSlabCount;
      metrics.gpu_temp_slab_capacity_mb = memStats.tempSlabCapacityBytes == null
        ? undefined
        : Math.round(memStats.tempSlabCapacityBytes / 1024 / 1024);
      metrics.gpu_temp_slab_live_mb = memStats.tempSlabLiveBytes == null
        ? undefined
        : Math.round(memStats.tempSlabLiveBytes / 1024 / 1024);
      metrics.gpu_allocator_slab_fallbacks = memStats.slabFallbacks;
      metrics.gpu_allocator_free_range_reuses = memStats.slabFreeRangeReuses;
      metrics.gpu_allocator_free_range_overflows = memStats.slabFreeRangeOverflows;
      peakGpuLiveAllocs = Math.max(peakGpuLiveAllocs, memStats.liveAllocs ?? 0);
      const currentPoolMb = (memStats.bufferPoolBytes + memStats.outputPoolBytes) / 1024 / 1024;
      peakGpuMemPoolMb = Math.max(peakGpuMemPoolMb, currentPoolMb);
      metrics.gpu_peak_live_allocs = peakGpuLiveAllocs;
      metrics.gpu_peak_mem_pool_mb = Math.round(peakGpuMemPoolMb);
    }

    // Eval — flush GPU and wait for completion first to maximize free VRAM
    if (valLoader && shouldEvaluateStep(stepNum, evalInterval, totalIters)) {
      if (flushFn) flushFn();
      if (typeof globalThis.gc === "function") {
        (globalThis as any).gc();
        await new Promise<void>(resolve => setImmediate(resolve));
      }
      // Second flush to process deferred releases from GC's FinalizationRegistry
      if (flushFn) flushFn();

      let valLossSum = 0;
      for (let ei = 0; ei < evalIters; ei++) {
        const valBatch = valLoader.nextBatch();
        const evalTape = new Tape();
        const { loss: vl } = gptForward(activeModelConfig, params, backend, evalTape, valBatch.inputs, valBatch.targets, false, false, false, undefined, undefined, valBatch.lossMask);
        if (vl) {
          valLossSum += (vl.data.data as Float32Array)[0];
          if (releaseFn) releaseFn(vl.data);
        }
        evalTape.clear(releaseFn);
        if (releaseFn) {
          releaseFn(valBatch.inputs);
          releaseFn(valBatch.targets);
          if (valBatch.lossMask) releaseFn(valBatch.lossMask);
        }
        // Flush between eval iters to process deferred releases
        if (flushFn) flushFn();
      }
      metrics.valLoss = valLossSum / evalIters;
      if (Number.isFinite(metrics.valLoss) && Number.isFinite(lossVal)) {
        evalHistory.push({ step: stepNum, trainLoss: lossVal, valLoss: metrics.valLoss });
        if (evalHistory.length > 64) evalHistory.shift();
      }

      if (valBucketLoaders.length > 0) {
        const bucketParts: string[] = [];
        for (const bucket of valBucketLoaders) {
          let bucketLossSum = 0;
          for (let bi = 0; bi < valBucketEvalIters; bi++) {
            const bucketBatch = bucket.loader.nextBatch();
            const bucketTape = new Tape();
            const { loss: bucketLoss } = gptForward(
              activeModelConfig,
              params,
              backend,
              bucketTape,
              bucketBatch.inputs,
              bucketBatch.targets,
            );
            if (bucketLoss) {
              bucketLossSum += (bucketLoss.data.data as Float32Array)[0];
              if (releaseFn) releaseFn(bucketLoss.data);
            }
            bucketTape.clear(releaseFn);
            if (releaseFn) {
              releaseFn(bucketBatch.inputs);
              releaseFn(bucketBatch.targets);
            }
            if (flushFn) flushFn();
          }
          const bucketLossAvg = bucketLossSum / valBucketEvalIters;
          bucketParts.push(`${bucket.name}=${bucketLossAvg.toFixed(4)}`);
        }
        if (bucketParts.length > 0) {
          console.log(`  [val_bucket] ${bucketParts.join(" | ")} (iters=${valBucketEvalIters})`);
        }
      }

      // Eval-time diagnostic summary
      console.log(
        `  [diag] eval step ${stepNum}: loss=${lossVal.toFixed(4)}, ` +
        `val_loss=${metrics.valLoss.toFixed(4)}, ` +
        `grad_norm=${needsGradNorm ? gradNorm.toFixed(2) : "n/a"}, ` +
        `clip_pct=${((clippedSteps / stepNum) * 100).toFixed(1)}%`
      );
      emitEvent({
        step: stepNum,
        level: "info",
        kind: "eval_summary",
        message: `eval step ${stepNum}: val_loss=${metrics.valLoss.toFixed(4)}`,
        payload: {
          trainLoss: lossVal,
          valLoss: metrics.valLoss,
          gradNorm: needsGradNorm ? gradNorm : null,
          clipPct: (clippedSteps / stepNum) * 100,
          evalIters,
        },
      });
    }

    // Log
    const lossStr = metrics.loss.toFixed(4);
    const valStr = metrics.valLoss !== undefined ? ` val_loss=${metrics.valLoss.toFixed(4)}` : "";
    const rcrUlStr = metrics.negative_ul_loss !== undefined
      ? ` ce=${metrics.positive_ce_loss!.toFixed(4)} ul=${metrics.negative_ul_loss.toFixed(4)} λ=${metrics.ul_weight}`
      : "";
    const toksStr = (metrics.tokens_per_sec).toFixed(0);
    const gpuStr = "gpuOpsThisStep" in backend ? ` | ${(backend as any).gpuOpsThisStep} gpu_ops` : "";
    const clipStr = clipCoef < 1.0 ? ` clip=${clipCoef.toFixed(4)}` : "";
    const gradNormStr = needsGradNorm ? gradNorm.toFixed(3) : "n/a";
    const scaleStr = useLossScaling ? ` | scale=${lossScale}` : "";
    const shouldLogStep = metrics.step === 1 ||
      metrics.step === totalIters ||
      metrics.valLoss !== undefined ||
      (metrics.step % logEvery === 0);
    if (shouldLogStep) {
      console.log(
        `step ${metrics.step}/${totalIters} | loss=${lossStr}${valStr}${rcrUlStr} ` +
        `| lr=${lr.toExponential(2)} | grad_norm=${gradNormStr}${clipStr} ` +
        `| ${metrics.ms_per_iter.toFixed(0)}ms/it | ${toksStr} tok/s${gpuStr}${scaleStr}`
      );
    }

    // Buffer metrics JSONL (flush every 50 steps and on checkpoint)
    metricsBuffer.push(JSON.stringify(metrics) + "\n");
    if (metricsBuffer.length >= 50) await flushMetrics();

    if (onStep) onStep(metrics);

    // Yield periodically when callbacks are attached so timers/network can progress
    // without paying a setImmediate cost every single step.
    if (shouldYieldForCallbacks && (stepNum % callbackYieldEvery === 0 || stepNum === totalIters)) {
      await new Promise<void>(resolve => setImmediate(resolve));
    }

    // Checkpoint cadence is independent from validation: full AdamW state is
    // hundreds of MB at flagship scale, so saving on every frequent eval can
    // exhaust an ephemeral pod volume.
    if (!disableCheckpoints && (stepNum % checkpointInterval === 0 || stepNum === totalIters)) {
      await flushMetrics(); // Ensure metrics are on disk before checkpoint
      const ckptPath = path.join(runDir, `checkpoint-${stepNum}.json`);
      // Keep the cloned optimizer buffers in a nested scope so they are unreachable
      // before the explicit collection below. A flagship AdamW snapshot is hundreds
      // of MB; leaving it for a later adaptive GC keeps that host memory live between
      // the checkpoint and the next collection boundary.
      let releasedOptimizerBuffers = 0;
      {
        const state = buildCheckpointState(params, optimizer, rng.state(), configHash, stepNum, activeModelConfig, deps.tokenizerArtifacts);
        try {
          // Save current symbio activation graph so resume can seed the search correctly
          if (searchOrchestrator?.currentCandidate?.activationGraph) {
            (state as any).symbioActivationGraph = searchOrchestrator.currentCandidate.activationGraph;
          }
          await Effect.runPromise(new FileCheckpoint().save(ckptPath, state));
        } finally {
          // AdamW.stateDict() clones both full-size moment buffers. Explicitly
          // sever those references even if save fails; block scope alone does
          // not guarantee an async function frame releases them before return.
          releasedOptimizerBuffers = releaseCheckpointSnapshotBuffers(state);
        }
      }

      const checkpointMemoryBeforeGc = process.memoryUsage();
      const checkpointGcRan = typeof globalThis.gc === "function";
      if (checkpointGcRan) {
        (globalThis as any).gc();
        await new Promise<void>(resolve => setImmediate(resolve));
      }
      const checkpointMemoryAfterGc = process.memoryUsage();
      console.log(`  checkpoint saved: ${ckptPath}`);
      console.log(
        `  [checkpoint_mem] optimizer_buffers_released=${releasedOptimizerBuffers}` +
        ` gc=${checkpointGcRan ? "yes" : "no"}` +
        ` rss_mb=${Math.round(checkpointMemoryBeforeGc.rss / 1024 / 1024)}->${Math.round(checkpointMemoryAfterGc.rss / 1024 / 1024)}` +
        ` external_mb=${Math.round(checkpointMemoryBeforeGc.external / 1024 / 1024)}->${Math.round(checkpointMemoryAfterGc.external / 1024 / 1024)}` +
        ` array_buffers_mb=${Math.round(checkpointMemoryBeforeGc.arrayBuffers / 1024 / 1024)}->${Math.round(checkpointMemoryAfterGc.arrayBuffers / 1024 / 1024)}`
      );
      emitEvent({
        step: stepNum,
        level: "info",
        kind: "checkpoint_saved",
        message: `checkpoint saved at step ${stepNum}`,
        payload: {
          path: ckptPath,
          hostGcRan: checkpointGcRan,
          hostRssBeforeGcMB: Math.round(checkpointMemoryBeforeGc.rss / 1024 / 1024),
          hostRssAfterGcMB: Math.round(checkpointMemoryAfterGc.rss / 1024 / 1024),
          hostExternalBeforeGcMB: Math.round(checkpointMemoryBeforeGc.external / 1024 / 1024),
          hostExternalAfterGcMB: Math.round(checkpointMemoryAfterGc.external / 1024 / 1024),
          hostArrayBuffersBeforeGcMB: Math.round(checkpointMemoryBeforeGc.arrayBuffers / 1024 / 1024),
          hostArrayBuffersAfterGcMB: Math.round(checkpointMemoryAfterGc.arrayBuffers / 1024 / 1024),
          optimizerBuffersReleased: releasedOptimizerBuffers,
        },
      });
      latestCheckpointPath = ckptPath;
      if (deps.onCheckpoint) deps.onCheckpoint({ step: stepNum, path: ckptPath, runId: rid });

      // Emit GPU diagnostics event at eval intervals for remote monitoring
      const diagPayload: Record<string, unknown> = {
        gradNanCount,
        gradNanPct: stepNum > 0 ? ((gradNanCount / stepNum) * 100).toFixed(2) : "0",
      };
      if (gpuMemStatsFn) {
        const ms = gpuMemStatsFn();
        diagPayload.bufferPoolMB = Math.round(ms.bufferPoolBytes / 1024 / 1024);
        diagPayload.outputPoolMB = Math.round(ms.outputPoolBytes / 1024 / 1024);
        diagPayload.liveAllocs = ms.liveAllocs;
        diagPayload.totalAllocs = ms.totalAllocs;
        diagPayload.trackedVkMemoryAllocations = ms.trackedVkMemoryAllocations;
        diagPayload.slabBuffers = ms.slabBuffers;
        diagPayload.tempSlabCount = ms.tempSlabCount;
        diagPayload.tempSlabCapacityMB = ms.tempSlabCapacityBytes == null
          ? null
          : Math.round(ms.tempSlabCapacityBytes / 1024 / 1024);
        diagPayload.tempSlabLiveMB = ms.tempSlabLiveBytes == null
          ? null
          : Math.round(ms.tempSlabLiveBytes / 1024 / 1024);
        diagPayload.slabFallbacks = ms.slabFallbacks;
        diagPayload.slabFreeRangeReuses = ms.slabFreeRangeReuses;
        diagPayload.slabFreeRangeOverflows = ms.slabFreeRangeOverflows;
      }
      if (gpuStats) {
        diagPayload.gpuUtilPct = gpuStats.utilPct;
        diagPayload.vramUsedMB = gpuStats.vramUsedMb;
        diagPayload.vramTotalMB = gpuStats.vramTotalMb;
      }
      const backendAny = backend as any;
      if (typeof backendAny.getMatmulCoopStats === "function") {
        const cs = backendAny.getMatmulCoopStats();
        diagPayload.coopHitRate = cs.coopHitRate;
        diagPayload.coopDispatches = cs.coopDispatches;
        diagPayload.totalMatmulDispatches = cs.totalMatmulDispatches;
      }
      emitEvent({
        step: stepNum,
        level: "info",
        kind: "gpu_diagnostics",
        message: `GPU diagnostics at step ${stepNum}`,
        payload: diagPayload,
      });
    }

    // Generate inference samples at sampleInterval (decoupled from checkpointing)
    if (sampleInterval > 0 && (stepNum % sampleInterval === 0 || stepNum === totalIters)) {
      if (deps.samplePrompts && deps.samplePrompts.length > 0) {
        try {
          // Flush GPU before sampling to maximize free VRAM
          if (flushFn) flushFn();
          if (typeof globalThis.gc === "function") {
            (globalThis as any).gc();
            await new Promise<void>(resolve => setImmediate(resolve));
          }
          if (flushFn) flushFn();

          const sampleCfg: SampleConfig = { steps: 15, temperature: 0.8, topk: 40 };
          // Limit to 1 prompt for checkpoint CLI sampler (slow cpu_ref autograd path)
          // and 3 for in-process GPU sampler
          const useCheckpointSampler =
            process.env.ALPHA_SAMPLE_FROM_CHECKPOINT !== "0" &&
            backend.name !== "cpu_ref" &&
            !!latestCheckpointPath;
          const prompts = deps.samplePrompts.slice(0, useCheckpointSampler ? 1 : 3);
          let samples: { prompt: string; output: string }[] = [];

          if (useCheckpointSampler && latestCheckpointPath) {
            samples = await sampleFromCheckpointCli(latestCheckpointPath, prompts, sampleCfg);
          } else {
            for (const prompt of prompts) {
              const output = runSample(
                activeModelConfig, params, backend, rng,
                (t) => tokenizer.encode(t),
                (t) => tokenizer.decode(t),
                prompt, sampleCfg, releaseFn, flushFn,
              );
              samples.push({ prompt, output });
            }
          }

          for (const sample of samples) {
            console.log(`  sample: "${sample.prompt}" → ${sample.output}`);
          }

          const trend = analyzeSampleTrend(evalHistory);
          const sampleQuality = computeSampleQualityStats(samples);
          if (trend) {
            const span = `${trend.windowStartStep}->${trend.windowEndStep}`;
            const gapSign = trend.gapDelta >= 0 ? "+" : "";
            console.log(
              `  [train_status] ${trend.summary} | window=${span} ` +
              `| train_delta=${trend.trainLossDelta.toFixed(4)} ` +
              `| val_delta=${trend.valLossDelta.toFixed(4)} ` +
              `| gap_delta=${gapSign}${trend.gapDelta.toFixed(4)}`,
            );
            if (trend.plateauLikely || trend.overfittingLikely) {
              const reason = trend.overfittingLikely ? "overfitting likely" : "loss plateau likely";
              console.warn(`  [train_status] ${reason} — consider stopping this run and investigating config/data.`);
              emitEvent({
                step: stepNum,
                level: "warn",
                kind: "train_status_warning",
                message: reason,
                payload: {
                  summary: trend.summary,
                  windowStartStep: trend.windowStartStep,
                  windowEndStep: trend.windowEndStep,
                  trainLossDelta: trend.trainLossDelta,
                  valLossDelta: trend.valLossDelta,
                  gapDelta: trend.gapDelta,
                  plateauLikely: trend.plateauLikely,
                  overfittingLikely: trend.overfittingLikely,
                },
              });
            }
          } else {
            console.log("  [train_status] insufficient eval history for plateau/overfitting analysis");
          }
          if (sampleQuality) {
            console.log(
              `  [sample_quality] uniq_ratio=${sampleQuality.uniqueTokenRatio.toFixed(3)} ` +
              `rep3=${sampleQuality.repeated3GramRate.toFixed(3)} ` +
              `rep4=${sampleQuality.repeated4GramRate.toFixed(3)} ` +
              `avg_tok_len=${sampleQuality.avgTokenLength.toFixed(2)}`,
            );
            if (sampleQuality.repeated4GramRate >= 0.20) {
              emitEvent({
                step: stepNum,
                level: "warn",
                kind: "sample_degeneracy_warning",
                message: "high repeated 4-gram rate in inference samples",
                payload: {
                  repeated4GramRate: sampleQuality.repeated4GramRate,
                  repeated3GramRate: sampleQuality.repeated3GramRate,
                  uniqueTokenRatio: sampleQuality.uniqueTokenRatio,
                },
              });
            }
          }

          emitEvent({
            step: stepNum,
            level: "info",
            kind: "samples_generated",
            message: `generated ${samples.length} inference samples`,
            payload: {
              sampleCount: samples.length,
              samples: samples.map((sample) => ({
                prompt: sample.prompt,
                outputPreview: sample.output.slice(0, 160),
              })),
              trend: trend ?? null,
              sampleQuality: sampleQuality ?? null,
            },
          });

          if (deps.onSamples) await deps.onSamples(samples, stepNum, trend ?? undefined);
        } catch (e) {
          console.warn(`  sample generation failed: ${(e as Error).message}`);
          emitEvent({
            step: stepNum,
            level: "error",
            kind: "sample_generation_failed",
            message: `sample generation failed: ${(e as Error).message}`,
          });
        }
      }
    }
  }

  const totalTime = performance.now() - startTime;
  console.log(`\n── training complete ──`);
  console.log(`total time: ${(totalTime / 1000).toFixed(1)}s`);
  emitEvent({
    step: totalIters,
    level: "info",
    kind: "training_complete",
    message: `training complete in ${(totalTime / 1000).toFixed(1)}s`,
    payload: {
      totalIters,
      durationMs: totalTime,
      spikeSkips,
      clippedSteps,
      lossScaleReductions,
    },
  });

  await flushMetrics();
  await metricsHandle.close();
  return { params, modelConfig: activeModelConfig };
}
