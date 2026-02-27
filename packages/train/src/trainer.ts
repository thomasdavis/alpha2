/**
 * Training loop orchestrator.
 *
 * Pure orchestration: depends on services (Backend, Tokenizer, Optimizer, Checkpoint, Logger).
 * Inspired by microgpt.py's training loop but with proper batching and logging.
 */
import type {
  ModelConfig, TrainConfig, Backend, Tokenizer, Optimizer, Rng, TensorData, SampleConfig,
} from "@alpha/core";
import { shapeSize, hashConfig, runId as makeRunId } from "@alpha/core";
import { Tape, DropoutRng } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries, countParams, clearForwardCache, type GPTParams } from "@alpha/model";
import {
  CusumDashboard, PopulationDynamicsController, KuramotoFusionController, SymbioMetricsCollector, SearchOrchestrator,
  defaultSymbioConfig, ffnDimForActivation, type SymbioConfig, type TrainerStepInfo,
  serializeGraph, nameGraph, type ActivationNode,
} from "@alpha/symbiogenesis";
import { DataLoader, loadText, loadAndTokenize, loadOrCacheTokens, getSplitByte } from "./data.js";
import { FileCheckpoint, buildCheckpointState, restoreParams } from "./checkpoint.js";
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
  gpu_util_pct?: number;
  gpu_vram_used_mb?: number;
  gpu_vram_total_mb?: number;
  gpu_mem_pool_mb?: number;
  // Phase 0 instrumentation: per-step timing breakdown
  timing_fwd_ms?: number;
  timing_bwd_ms?: number;
  timing_optim_ms?: number;
  timing_data_ms?: number;
  timing_flush_ms?: number;
  timing_grad_norm_ms?: number;
  timing_grad_clip_ms?: number;
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

// ── Trainer ────────────────────────────────────────────────────────────────

export interface TrainerDeps {
  backend: Backend;
  tokenizer: Tokenizer;
  optimizer: Optimizer;
  rng: Rng;
  modelConfig: ModelConfig;
  trainConfig: TrainConfig;
  dataPath: string;
  valDataPath?: string;
  runDir?: string;
  resumePath?: string;
  /** Override activation graph for symbio resume (when checkpoint doesn't contain it). */
  resumeActivationGraph?: ActivationNode;
  tokenizerArtifacts?: import("@alpha/core").TokenizerArtifacts;
  onStep?: (metrics: StepMetrics) => void;
  onStart?: (info: {
    runId: string; configHash: string; totalParams: number; dataPath: string;
    infra?: { gpuName: string; gpuVendor: string; gpuVramMb: number; hostname: string; cpuCount: number; ramTotalMb: number; osPlatform: string };
  }) => void | Promise<void>;
  onCheckpoint?: (info: { step: number; path: string; runId: string }) => void;
  onSamples?: (samples: { prompt: string; output: string }[], step: number) => void | Promise<void>;
  samplePrompts?: string[];
  domain?: string;
  activationCheckpointing?: boolean;
  mixedPrecision?: boolean;
}

export async function train(deps: TrainerDeps): Promise<{ params: GPTParams; modelConfig: ModelConfig }> {
  const {
    backend, tokenizer, optimizer, rng, modelConfig, trainConfig,
    dataPath, valDataPath, resumePath, onStep, onStart,
  } = deps;

  const dataTag = dataPath.split("/").pop()?.replace(/\.[^.]+$/, "").replace(/-/g, "_");
  const rid = makeRunId(dataTag);
  const configHash = hashConfig({ ...modelConfig, ...trainConfig } as any);

  // Set up run directory
  const runDir = deps.runDir ?? `runs/${rid}`;
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  await fs.mkdir(runDir, { recursive: true });
  const configObj: Record<string, unknown> = { modelConfig, trainConfig, configHash, runId: rid };
  if (deps.domain) configObj.domain = deps.domain;
  await fs.writeFile(
    path.join(runDir, "config.json"),
    JSON.stringify(configObj, null, 2),
  );

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

  // Load data — use chunked tokenization for large files to avoid V8 string limit
  const fileStat = await fs.stat(dataPath);
  const isLargeFile = fileStat.size > 200 * 1024 * 1024; // >200MB
  let trainTokens: Int32Array;
  let valTokens: Int32Array;

  if (isLargeFile) {
    console.log(`Large dataset (${(fileStat.size / 1024 / 1024).toFixed(0)}MB) — using chunked tokenization...`);
    if (valDataPath) {
      trainTokens = await loadOrCacheTokens(dataPath, tokenizer);
      valTokens = await loadOrCacheTokens(valDataPath, tokenizer);
    } else {
      const splitByte = await getSplitByte(dataPath, 0.9);
      trainTokens = await loadOrCacheTokens(dataPath, tokenizer, { startByte: 0, endByte: splitByte });
      valTokens = await loadOrCacheTokens(dataPath, tokenizer, { startByte: splitByte, endByte: fileStat.size });
    }
  } else {
    const rawText = await loadText(dataPath);
    if (valDataPath) {
      const valRaw = await loadText(valDataPath);
      trainTokens = tokenizer.encode(rawText);
      valTokens = tokenizer.encode(valRaw);
    } else {
      const splitIdx = Math.floor(rawText.length * 0.9);
      trainTokens = tokenizer.encode(rawText.slice(0, splitIdx));
      valTokens = tokenizer.encode(rawText.slice(splitIdx));
    }
  }

  // Initialize model
  rng.seed(trainConfig.seed);
  let params = initGPT(modelConfig, backend, rng as any);
  let totalParams = countParams(params);

  // Resume from checkpoint
  let startStep = 0;
  let resumedActivationGraph: ActivationNode | undefined = deps.resumeActivationGraph;
  if (resumePath) {
    const checkpoint = new FileCheckpoint();
    const state = await Effect.runPromise(checkpoint.load(resumePath));
    restoreParams(params, state.params);
    optimizer.loadStateDict(state.optimizerState);
    rng.setState(state.rngState);
    startStep = state.step;
    // CLI override takes precedence over checkpoint-embedded graph
    if (!resumedActivationGraph) {
      resumedActivationGraph = (state as any).symbioActivationGraph;
    }
    console.log(`Resumed from step ${startStep}${resumedActivationGraph ? ` (activation: ${nameGraph(resumedActivationGraph)})` : ""}`);
  }

  // Collect infrastructure metadata (GPU runs only)
  let infra: { gpuName: string; gpuVendor: string; gpuVramMb: number; hostname: string; cpuCount: number; ramTotalMb: number; osPlatform: string } | undefined;
  if ("getDeviceInfo" in backend) {
    try {
      const os = await import("node:os");
      const gpu = (backend as any).getDeviceInfo();
      const vendorName = gpu.vendorId === 0x10de ? "NVIDIA"
        : gpu.vendorId === 0x1002 ? "AMD"
        : gpu.vendorId === 0x8086 ? "Intel"
        : `0x${gpu.vendorId.toString(16)}`;
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

  // Notify start
  if (onStart) await onStart({ runId: rid, configHash, totalParams, dataPath, infra });

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
  console.log(`iters: ${trainConfig.iters} | lr: ${trainConfig.lr}`);
  if (deps.activationCheckpointing) console.log(`activation_checkpointing: enabled`);
  if (deps.mixedPrecision) console.log(`mixed_precision: f16 activations enabled`);

  // GPU proof: log device info and run smoke test
  if ("getDeviceInfo" in backend && "smokeTest" in backend) {
    const gpu = (backend as any).getDeviceInfo();
    const vendorName = gpu.vendorId === 0x10de ? "NVIDIA"
      : gpu.vendorId === 0x1002 ? "AMD"
      : gpu.vendorId === 0x8086 ? "Intel"
      : `0x${gpu.vendorId.toString(16)}`;
    console.log(`gpu: ${gpu.deviceName} (${vendorName})`);
    console.log(`  f16: ${gpu.f16Supported} | async_transfer: ${gpu.hasAsyncTransfer} | wg_size: ${gpu.workgroupSize} | min_gpu: ${gpu.minGpuSize}`);
    try {
      const smoke = (backend as any).smokeTest();
      console.log(`  smoke_test: ${smoke.verified ? "PASS" : "FAIL"} | gpu_throughput: ${smoke.throughputGBps.toFixed(1)} GB/s`);
    } catch (e: any) {
      console.log(`  smoke_test: FAIL (${e.message})`);
    }
  }

  console.log(``);

  // Create data loaders from pre-tokenized arrays
  const packed = trainConfig.packed;
  const trainLoader = new DataLoader(trainTokens, rng, trainConfig.batchSize, modelConfig.blockSize, packed);
  const valLoader = valTokens.length > 0
    ? new DataLoader(valTokens, rng, trainConfig.batchSize, modelConfig.blockSize)
    : undefined;
  if (packed) {
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

  const backendAny = backend as any;
  const setOptimizerLrFn: ((lr: number) => void) | undefined =
    typeof (optimizer as any).setLr === "function"
      ? (optimizer as any).setLr.bind(optimizer)
      : undefined;
  const resetStepOpsFn: (() => void) | undefined =
    typeof backendAny.resetStepOps === "function"
      ? backendAny.resetStepOps.bind(backendAny)
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

  // Training loop
  const startTime = performance.now();
  let spikeSkips = 0;
  let clippedSteps = 0;

  // Dynamic loss scaling for mixed precision training
  const useLossScaling = !!deps.mixedPrecision;
  const logEvery = Math.max(1, trainConfig.logEvery ?? 1);
  const shouldYieldEachStep = !!(onStep || deps.onCheckpoint || deps.onSamples);
  const totalIters = trainConfig.iters;
  const traceEnabled = trainConfig.trace;
  const capturePhaseTimings = traceEnabled;
  const gradAccumSteps = trainConfig.gradAccumSteps;
  const gradClip = trainConfig.gradClip;
  const spikeThreshold = trainConfig.spikeThreshold;
  const evalIters = trainConfig.evalIters;
  const evalInterval = trainConfig.evalInterval;
  const sampleInterval = trainConfig.sampleInterval;
  const tokensProcessedPerStep = trainConfig.batchSize * modelConfig.blockSize * gradAccumSteps;
  const warmup = trainConfig.warmupIters > 0
    ? trainConfig.warmupIters
    : trainConfig.warmupIters < 0
      ? 0  // negative = explicitly disabled
      : Math.min(2000, Math.floor(totalIters / 5));
  const lrMin = (trainConfig.lrMin ?? 0) === 0
    ? trainConfig.lr / 10  // auto-calc: lr/10 (nanoGPT convention)
    : trainConfig.lrMin;
  const decayDenom = Math.max(1, totalIters - warmup);
  let lossScale = useLossScaling ? 65536.0 : 1.0; // start high, will auto-tune down
  let scaleSuccessCount = 0;
  const SCALE_GROWTH_INTERVAL = 200; // double scale after this many consecutive good steps
  let lossScaleReductions = 0;
  const dropoutRng = new DropoutRng(trainConfig.seed);
  const trainTape = new Tape();
  const gradTensors: TensorData[] = [];
  const gradNamesBuf: string[] = [];
  const perParamNormsBuf: { name: string; normSq: number }[] = [];
  const ADAPTIVE_MEM_STATS_POLL_EVERY = 4;
  let memStatsCache: any | null = null;
  let lastMemStatsProbeStep = 0;

  for (let step = startStep; step < totalIters; step++) {
    const stepStart = performance.now();
    const stepNum = step + 1;

    // Learning rate schedule: linear warmup + cosine decay to lrMin
    let lr: number;
    if (step < warmup) {
      lr = lrMin + (trainConfig.lr - lrMin) * stepNum / warmup;
    } else {
      const decay = (step - warmup) / decayDenom;
      lr = lrMin + (trainConfig.lr - lrMin) * 0.5 * (1 + Math.cos(Math.PI * decay));
    }
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
    let lossVal = 0;
    let dataLoadMs = 0;
    let fwdMs = 0;
    let bwdMs = 0;

    // GPU-side loss accumulation: avoid per-microstep CPU readback for grad accumulation.
    // For accumSteps=1, skip the no-op GPU scale/add path and read loss directly.
    const useGpuLossAccum = accumSteps > 1;
    let lossAccum: TensorData | null = null;

    for (let microStep = 0; microStep < accumSteps; microStep++) {
      const _dl0 = capturePhaseTimings ? performance.now() : 0;
      const batch = trainLoader.nextBatch();
      const _dl1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) dataLoadMs += _dl1 - _dl0;
      dropoutRng.reset(stepSeedBase + microStep, 0);
      const { loss } = gptForward(activeModelConfig, params, backend, trainTape, batch.inputs, batch.targets, true, !!deps.activationCheckpointing, !!deps.mixedPrecision, dropoutRng, releaseFn);
      const _fwd1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) fwdMs += _fwd1 - _dl1;

      if (!loss) throw new Error("Loss is undefined");

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

      // Loss scaling: pass scaled initial gradient for backward.
      // This scales all gradients by lossScale, preventing f16 underflow.
      // Gradients are unscaled back before the optimizer step.
      if (useLossScaling && lossScale !== 1.0) {
        const scaledGrad = backend.full(loss.data.shape, lossScale, loss.data.dtype);
        trainTape.backward(loss, backend, releaseFn, scaledGrad);
      } else {
        trainTape.backward(loss, backend, releaseFn);
      }
      if (!useGpuLossAccum) {
        lossVal = (loss.data.data as Float32Array)[0];
        if (!isFinite(lossVal)) {
          console.warn(`  [warn] loss=NaN at step ${stepNum} — skipping`);
          nanDetected = true;
        }
      }
      const _bwd1 = capturePhaseTimings ? performance.now() : 0;
      if (capturePhaseTimings) bwdMs += _bwd1 - _fwd1;
      // Release tape intermediates but keep param gradients for accumulation
      trainTape.clear(releaseFn);
    }

    // Single CPU readback of accumulated loss (triggers one flush+wait)
    if (useGpuLossAccum && lossAccum) {
      lossVal = (lossAccum.data as Float32Array)[0];
      if (!isFinite(lossVal)) {
        console.warn(`  [warn] loss=NaN at step ${stepNum} — skipping`);
        nanDetected = true;
      }
      if (releaseFn) releaseFn(lossAccum);
    }

    // Keep gradients unmodified and fold scaling/clipping into optimizer update.
    // This avoids one or two full tensor passes over all parameter gradients.
    const gradScaleFactor = 1.0 / (accumSteps * lossScale);
    const gradScaleAbs = Math.abs(gradScaleFactor);
    let gradNorm = 0;
    const needsGradNorm =
      gradClip > 0 ||
      spikeThreshold > 0 ||
      useLossScaling ||
      traceEnabled ||
      symbioEnabled ||
      stepNum % 500 === 0;
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

      if (hasTotalSumSq && grads.length > 0) {
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

      // NaN guard on gradient norm (backward produced NaN even though loss was finite)
      if (!isFinite(gradNorm)) {
        if (useLossScaling) {
          // Dynamic loss scaling: halve the scale and retry
          lossScale /= 2;
          lossScaleReductions++;
          console.warn(`  [loss_scale] grad overflow at step ${stepNum} — reducing scale to ${lossScale}`);
          scaleSuccessCount = 0;
        } else {
          console.warn(`  [warn] grad_norm=NaN at step ${stepNum} — skipping optimizer update`);
        }
        nanDetected = true;
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
        console.warn(`  [spike] grad_norm=${gradNorm.toFixed(1)} > ${spikeThreshold} — skipping step ${stepNum} (${spikeSkips} total skips)`);
        nanDetected = true; // reuse the skip path
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
        ? (stepNum <= 20 || stepNum % 5 === 0)
        : (stepNum === 1 || stepNum === totalIters || stepNum % 25 === 0)
    );
    let memStatsStep: any | null = memStatsCache;
    const adaptiveMemControl = gcEvery <= 0 || syncEvery <= 0;
    const shouldProbeAdaptiveMem = adaptiveMemControl && (
      stepNum === 1 ||
      stepNum === totalIters ||
      stepNum - lastMemStatsProbeStep >= ADAPTIVE_MEM_STATS_POLL_EVERY
    );
    if (gpuMemStatsFn && (shouldLogGpuMem || shouldProbeAdaptiveMem)) {
      memStatsStep = gpuMemStatsFn();
      memStatsCache = memStatsStep;
      lastMemStatsProbeStep = stepNum;
    }

    const needGc = typeof globalThis.gc === "function" && (
      gcEvery > 0 ? stepNum % gcEvery === 0 :
      // Adaptive: GC when deferred releases pile up
      !!(memStatsStep && memStatsStep.deferredReleases > 50)
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
    const needSync = syncEvery === 1 ? true :
      syncEvery > 0 ? stepNum % syncEvery === 0 :
      // Adaptive: sync when pool pressure is high (many pending destroys or deferred releases)
      !!(memStatsStep && (memStatsStep.deferredReleases > 20 || (memStatsStep.pendingDestroys ?? 0) > 10));
    if (needSync) {
      if (syncGpuFn) {
        syncGpuFn();
      } else if (flushFn) {
        flushFn();
      }
    }
    const _t6 = capturePhaseTimings ? performance.now() : 0;

    if (traceEnabled) {
      const gpuOps = "gpuOpsThisStep" in backend ? ` gpu_ops=${(backend as any).gpuOpsThisStep}` : "";
      console.log(`  [trace] data=${dataLoadMs.toFixed(0)}ms fwd=${fwdMs.toFixed(0)}ms bwd=${bwdMs.toFixed(0)}ms gradnorm=${(_t4-_t3).toFixed(0)}ms clip=${(_t4b-_t4).toFixed(0)}ms optim=${(_t5-_t4b).toFixed(0)}ms flush=${(_t6-_t5).toFixed(0)}ms${gpuOps}`);
    }

    if (shouldLogGpuMem) {
      const stats = memStatsStep ?? gpuMemStatsFn!();
      const breakdown = poolBreakdownFn ? ` | ${poolBreakdownFn(8)}` : "";
      const allocStr = stats.liveAllocs != null ? ` | allocs: ${stats.liveAllocs} live (${stats.totalAllocs} total, ${stats.totalAllocMB}MB)` : "";
      console.log(`  [gpu_mem] bufPool: ${stats.bufferPoolEntries} (${(stats.bufferPoolBytes/1024/1024).toFixed(1)}MB) | outPool: ${stats.outputPoolEntries}/${stats.outputPoolSizeClasses ?? "?"}cls (${(stats.outputPoolBytes/1024/1024).toFixed(1)}MB) | deferred: ${stats.deferredReleases} | pending: ${stats.pendingDestroys ?? 0}${allocStr}${breakdown}`);
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
    if (capturePhaseTimings) {
      metrics.timing_fwd_ms = fwdMs;
      metrics.timing_bwd_ms = bwdMs;
      metrics.timing_grad_norm_ms = _t4 - _t3;
      metrics.timing_grad_clip_ms = _t4b - _t4;
      metrics.timing_optim_ms = _t5 - _t4b;
      metrics.timing_flush_ms = _t6 - _t5;
      metrics.timing_data_ms = dataLoadMs;
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

    // GPU stats (queried at most every 5s via nvidia-smi)
    let gpuStats: GpuStats | null = null;
    if (!_gpuStatsDisabled) {
      gpuStats = await queryGpuStats();
    }
    if (gpuStats) {
      metrics.gpu_util_pct = gpuStats.utilPct;
      metrics.gpu_vram_used_mb = gpuStats.vramUsedMb;
      metrics.gpu_vram_total_mb = gpuStats.vramTotalMb;
    }
      const shouldSampleGpuMemMetric = traceEnabled ||
      metrics.step === 1 ||
      metrics.step === totalIters ||
      (metrics.step % Math.max(logEvery, 5) === 0);
    if (gpuMemStatsFn && shouldSampleGpuMemMetric) {
      const memStats = memStatsStep ?? gpuMemStatsFn();
      metrics.gpu_mem_pool_mb = Math.round((memStats.bufferPoolBytes + memStats.outputPoolBytes) / 1024 / 1024);
    }

    // Eval — flush GPU and wait for completion first to maximize free VRAM
    if (valLoader && stepNum % evalInterval === 0) {
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
        const { loss: vl } = gptForward(activeModelConfig, params, backend, evalTape, valBatch.inputs, valBatch.targets);
        if (vl) {
          valLossSum += (vl.data.data as Float32Array)[0];
          if (releaseFn) releaseFn(vl.data);
        }
        evalTape.clear(releaseFn);
        // Flush between eval iters to process deferred releases
        if (flushFn) flushFn();
      }
      metrics.valLoss = valLossSum / evalIters;

      // Eval-time diagnostic summary
      console.log(
        `  [diag] eval step ${stepNum}: loss=${lossVal.toFixed(4)}, ` +
        `val_loss=${metrics.valLoss.toFixed(4)}, ` +
        `grad_norm=${needsGradNorm ? gradNorm.toFixed(2) : "n/a"}, ` +
        `clip_pct=${((clippedSteps / stepNum) * 100).toFixed(1)}%`
      );
    }

    // Log
    const lossStr = metrics.loss.toFixed(4);
    const valStr = metrics.valLoss !== undefined ? ` val_loss=${metrics.valLoss.toFixed(4)}` : "";
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
        `step ${metrics.step}/${totalIters} | loss=${lossStr}${valStr} ` +
        `| lr=${lr.toExponential(2)} | grad_norm=${gradNormStr}${clipStr} ` +
        `| ${metrics.ms_per_iter.toFixed(0)}ms/it | ${toksStr} tok/s${gpuStr}${scaleStr}`
      );
    }

    // Buffer metrics JSONL (flush every 50 steps and on checkpoint)
    metricsBuffer.push(JSON.stringify(metrics) + "\n");
    if (metricsBuffer.length >= 50) await flushMetrics();

    if (onStep) onStep(metrics);

    // Yield only when async callbacks are attached; otherwise avoid per-step event-loop overhead.
    if (shouldYieldEachStep) {
      await new Promise<void>(resolve => setImmediate(resolve));
    }

    // Checkpoint (save at every eval interval and at the end)
    if (stepNum % evalInterval === 0 || stepNum === totalIters) {
      await flushMetrics(); // Ensure metrics are on disk before checkpoint
      const ckptPath = path.join(runDir, `checkpoint-${stepNum}.json`);
      const state = buildCheckpointState(params, optimizer, rng.state(), configHash, stepNum, activeModelConfig, deps.tokenizerArtifacts);
      // Save current symbio activation graph so resume can seed the search correctly
      if (searchOrchestrator?.currentCandidate?.activationGraph) {
        (state as any).symbioActivationGraph = searchOrchestrator.currentCandidate.activationGraph;
      }
      await Effect.runPromise(new FileCheckpoint().save(ckptPath, state));
      console.log(`  checkpoint saved: ${ckptPath}`);
      if (deps.onCheckpoint) deps.onCheckpoint({ step: stepNum, path: ckptPath, runId: rid });
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

          const sampleCfg: SampleConfig = { steps: 50, temperature: 0.8, topk: 40 };
          const samples: { prompt: string; output: string }[] = [];

          for (const prompt of deps.samplePrompts.slice(0, 3)) {
            const output = runSample(
              activeModelConfig, params, backend, rng,
              (t) => tokenizer.encode(t),
              (t) => tokenizer.decode(t),
              prompt, sampleCfg, releaseFn, flushFn,
            );
            console.log(`  sample: "${prompt}" → ${output}`);
            samples.push({ prompt, output });
          }

          if (deps.onSamples) await deps.onSamples(samples, stepNum);
        } catch (e) {
          console.warn(`  sample generation failed: ${(e as Error).message}`);
        }
      }
    }
  }

  const totalTime = performance.now() - startTime;
  console.log(`\n── training complete ──`);
  console.log(`total time: ${(totalTime / 1000).toFixed(1)}s`);

  await flushMetrics();
  await metricsHandle.close();
  return { params, modelConfig: activeModelConfig };
}
