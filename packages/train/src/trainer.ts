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
import { initGPT, gptForward, collectParams, countParams, clearForwardCache, type GPTParams } from "@alpha/model";
import {
  CusumDashboard, AdaptiveBatch, SymbioMetricsCollector, SearchOrchestrator,
  defaultSymbioConfig, type SymbioConfig, type TrainerStepInfo,
} from "@alpha/symbiogenesis";
import { DataLoader, loadText, loadAndTokenize, loadOrCacheTokens, getSplitByte } from "./data.js";
import { FileCheckpoint, buildCheckpointState, restoreParams } from "./checkpoint.js";
import { sample as runSample } from "./sample.js";
import { Effect } from "effect";

// ── GPU stats ────────────────────────────────────────────────────────────

interface GpuStats {
  utilPct: number;
  vramUsedMb: number;
  vramTotalMb: number;
}

let _gpuStatsCache: GpuStats | null = null;
let _gpuStatsLastQuery = 0;
const GPU_STATS_INTERVAL_MS = 5000; // query nvidia-smi at most every 5s

async function queryGpuStats(): Promise<GpuStats | null> {
  const now = performance.now();
  if (now - _gpuStatsLastQuery < GPU_STATS_INTERVAL_MS && _gpuStatsCache) {
    return _gpuStatsCache;
  }
  _gpuStatsLastQuery = now;
  try {
    const { execSync } = await import("node:child_process");
    const out = execSync(
      "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
      { timeout: 2000, encoding: "utf-8" },
    ).trim();
    const [util, used, total] = out.split(",").map(s => parseFloat(s.trim()));
    if (isFinite(util) && isFinite(used) && isFinite(total)) {
      _gpuStatsCache = { utilPct: util, vramUsedMb: used, vramTotalMb: total };
      return _gpuStatsCache;
    }
  } catch { /* nvidia-smi not available — skip */ }
  return null;
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
  symbio_candidate_activation?: string;
  symbio_generation?: number;
  architecture_diversity?: number;
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
  if (resumePath) {
    const checkpoint = new FileCheckpoint();
    const state = await Effect.runPromise(checkpoint.load(resumePath));
    restoreParams(params, state.params);
    optimizer.loadStateDict(state.optimizerState);
    rng.setState(state.rngState);
    startStep = state.step;
    console.log(`Resumed from step ${startStep}`);
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
      const gpuStats = await queryGpuStats();
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
  const adaptiveBatch = symbioEnabled && symbioConfig.adaptiveBatch
    ? new AdaptiveBatch(trainConfig.batchSize, symbioConfig)
    : null;
  const symbioCollector = symbioEnabled ? new SymbioMetricsCollector(symbioConfig) : null;
  if (symbioEnabled) {
    const activation = modelConfig.ffnActivation ?? "gelu";
    console.log(`symbio: enabled | activation=${activation} | cusum_sens=${symbioConfig.cusumSensitivity} | metrics_interval=${symbioConfig.metricsInterval}`);
    if (adaptiveBatch) console.log(`symbio: adaptive_batch=[${symbioConfig.batchMin},${symbioConfig.batchMax}] step=${symbioConfig.batchStep}`);
  }

  // FFN activation search orchestrator (only when searchMode is active)
  const searchActive = symbioEnabled && symbioConfig.searchMode === "ffn-activation-search";
  const searchOrchestrator = searchActive ? new SearchOrchestrator(symbioConfig) : null;
  let activeModelConfig = { ...modelConfig };

  if (searchOrchestrator) {
    const firstCandidate = searchOrchestrator.currentCandidate;
    if (firstCandidate) {
      // Re-init model with first candidate's activation
      const ffnDim = firstCandidate.activation === "swiglu"
        ? Math.ceil((8 / 3) * modelConfig.nEmbd / 64) * 64
        : modelConfig.ffnDim ?? 4 * modelConfig.nEmbd;
      activeModelConfig = { ...modelConfig, ffnActivation: firstCandidate.activation as any, ffnDim };
      rng.seed(trainConfig.seed);
      params = initGPT(activeModelConfig, backend, rng as any);
      totalParams = countParams(params);
      optimizer.loadStateDict({ step: 0, buffers: new Map() });
      console.log(`symbio search: gen=${firstCandidate.generation} candidate=${firstCandidate.id} activation=${firstCandidate.activation} params=${totalParams.toLocaleString()}`);
    }
    const totalSearchSteps = symbioConfig.populationSize * symbioConfig.stepsPerCandidate * symbioConfig.generations;
    console.log(`symbio search: ${symbioConfig.populationSize} candidates × ${symbioConfig.stepsPerCandidate} steps × ${symbioConfig.generations} gens = ${totalSearchSteps} total steps`);
  }

  // Training loop
  const startTime = performance.now();
  let spikeSkips = 0;
  let clippedSteps = 0;

  // Dynamic loss scaling for mixed precision training
  const useLossScaling = !!deps.mixedPrecision;
  let lossScale = useLossScaling ? 65536.0 : 1.0; // start high, will auto-tune down
  let scaleSuccessCount = 0;
  const SCALE_GROWTH_INTERVAL = 200; // double scale after this many consecutive good steps
  let lossScaleReductions = 0;

  for (let step = startStep; step < trainConfig.iters; step++) {
    const stepStart = performance.now();

    // Learning rate schedule: linear warmup + cosine decay to lrMin
    const warmup = trainConfig.warmupIters > 0
      ? trainConfig.warmupIters
      : trainConfig.warmupIters < 0
        ? 0  // negative = explicitly disabled
        : Math.min(2000, Math.floor(trainConfig.iters / 5));
    const lrMin = (trainConfig.lrMin ?? 0) === 0
      ? trainConfig.lr / 10  // auto-calc: lr/10 (nanoGPT convention)
      : trainConfig.lrMin;
    let lr: number;
    if (step < warmup) {
      lr = lrMin + (trainConfig.lr - lrMin) * (step + 1) / warmup;
    } else {
      const decay = (step - warmup) / (trainConfig.iters - warmup);
      lr = lrMin + (trainConfig.lr - lrMin) * 0.5 * (1 + Math.cos(Math.PI * decay));
    }
    if (optimizer && "setLr" in optimizer) {
      (optimizer as any).setLr(lr);
    }

    // Reset per-step GPU ops counter
    if ("resetStepOps" in backend) (backend as any).resetStepOps();

    // Explicit GPU buffer release function — deterministic cleanup instead of
    // relying on FinalizationRegistry which is unreliable for timely VRAM reclaim.
    const releaseFn = "releaseGpuTensor" in backend
      ? (td: TensorData) => (backend as any).releaseGpuTensor(td)
      : undefined;

    // Gradient accumulation: run K forward+backward passes, accumulating
    // gradients on parameter Variables. Each micro-batch contributes 1/K
    // of the total gradient. Increases effective batch size by K without
    // increasing VRAM usage (only one micro-batch is live at a time).
    const accumSteps = trainConfig.gradAccumSteps;
    let nanDetected = false;
    let lossVal = 0;
    let dataLoadMs = 0;
    let fwdMs = 0;
    let bwdMs = 0;
    const _t0 = performance.now();

    for (let microStep = 0; microStep < accumSteps; microStep++) {
      const tape = new Tape();
      const _dl0 = performance.now();
      const batch = trainLoader.nextBatch();
      const _dl1 = performance.now();
      dataLoadMs += _dl1 - _dl0;
      const dropoutRng = new DropoutRng(trainConfig.seed + step * 1000 + microStep);
      const { loss } = gptForward(activeModelConfig, params, backend, tape, batch.inputs, batch.targets, true, !!deps.activationCheckpointing, !!deps.mixedPrecision, dropoutRng);
      const _fwd1 = performance.now();
      fwdMs += _fwd1 - _dl1;

      if (!loss) throw new Error("Loss is undefined");

      const microLoss = (loss.data.data as Float32Array)[0];
      if (!isFinite(microLoss)) {
        console.warn(`  [warn] loss=NaN at step ${step + 1} micro ${microStep} — skipping`);
        nanDetected = true;
        tape.clear(releaseFn);
        break;
      }
      lossVal += microLoss / accumSteps;

      // Loss scaling: pass scaled initial gradient for backward.
      // This scales all gradients by lossScale, preventing f16 underflow.
      // Gradients are unscaled back before the optimizer step.
      if (useLossScaling && lossScale !== 1.0) {
        const scaledGrad = backend.full(loss.data.shape, lossScale, loss.data.dtype);
        tape.backward(loss, backend, releaseFn, scaledGrad);
      } else {
        tape.backward(loss, backend, releaseFn);
      }
      const _bwd1 = performance.now();
      bwdMs += _bwd1 - _fwd1;
      // Release tape intermediates but keep param gradients for accumulation
      tape.clear(releaseFn);
    }

    // Scale accumulated gradients: combine 1/accumSteps and 1/lossScale into one op.
    // accumSteps > 1: backward summed micro-batch gradients → divide by accumSteps.
    // lossScale != 1: gradients were amplified by loss scaling → divide by lossScale.
    const gradScaleFactor = 1.0 / (accumSteps * lossScale);
    if (!nanDetected && gradScaleFactor !== 1.0) {
      const paramMap0 = collectParams(params);
      for (const [, variable] of paramMap0) {
        if (variable.grad) {
          const oldGrad = variable.grad;
          variable.grad = backend.scale(variable.grad, gradScaleFactor);
          if (releaseFn) releaseFn(oldGrad);
        }
      }
    }
    let gradNorm = 0;
    const _t3 = performance.now();

    // Collect gradients and compute gradient norm via backend ops (stays on GPU)
    const paramMap = collectParams(params);
    const paramDataMap = new Map<string, TensorData>();
    const gradMap = new Map<string, TensorData>();

    if (!nanDetected) {
      // Compute gradient norm: record all GPU ops first, then single flush + CPU sum.
      // Use fused sumOfSquares when available (1 dispatch vs 2 for mul+sum per param).
      const hasSumSq = !!backend.sumOfSquares;
      const sqNormParts: TensorData[] = [];
      const sqNormNames: string[] = [];
      const g2Intermediates: TensorData[] = [];
      for (const [name, variable] of paramMap) {
        paramDataMap.set(name, variable.data);
        if (variable.grad) {
          const g = variable.grad;
          if (hasSumSq) {
            sqNormParts.push(backend.sumOfSquares!(g));
          } else {
            const g2 = backend.mul(g, g);
            g2Intermediates.push(g2);
            sqNormParts.push(backend.sum(g2));
          }
          sqNormNames.push(name);
        }
      }
      // Phase 2: first .data access triggers single flush of ALL pending GPU work
      // Always collect per-param norms for diagnostics (minimal overhead — values already on CPU)
      let gradNormSq = 0;
      const perParamNorms: { name: string; normSq: number }[] = [];
      for (let pi = 0; pi < sqNormParts.length; pi++) {
        const val = (sqNormParts[pi].data as Float32Array)[0];
        gradNormSq += val;
        perParamNorms.push({ name: sqNormNames[pi], normSq: val });
      }
      gradNorm = Math.sqrt(gradNormSq);

      // Per-layer gradient diagnostics (when trace enabled)
      if (trainConfig.trace) {
        const sorted = [...perParamNorms]
          .map(({ name: n, normSq: sq }) => [n, Math.sqrt(sq)] as const)
          .sort((a, b) => b[1] - a[1]);
        const top5 = sorted.slice(0, 5).map(([n, v]) => `${n}=${v.toFixed(2)}`).join(", ");
        console.log(`  [trace] grad norms (top 5): ${top5}`);
      }

      // Per-layer gradient diagnostics (every 500 steps, independent of trace flag)
      if ((step + 1) % 500 === 0 && perParamNorms.length > 0) {
        const layerNorms = perParamNorms
          .map(({ name, normSq }) => ({ name, norm: Math.sqrt(normSq) }))
          .sort((a, b) => b.norm - a.norm);

        console.log(`  [diag] per-layer grad norms (step ${step + 1}):`);
        const top10 = layerNorms.slice(0, 10);
        for (const { name, norm } of top10) {
          console.log(`    ${name}: ${norm.toFixed(4)}`);
        }

        // Identify spike source — which parameter dominates the total norm
        if (layerNorms.length > 0) {
          const topParam = layerNorms[0];
          const totalNorm = Math.sqrt(layerNorms.reduce((s, l) => s + l.norm * l.norm, 0));
          const pct = ((topParam.norm / totalNorm) * 100).toFixed(1);
          console.log(`    -> dominant: ${topParam.name} (${pct}% of total norm)`);
        }
      }

      // Release grad norm intermediates
      if (releaseFn) {
        for (const g2 of g2Intermediates) releaseFn(g2);
        for (const part of sqNormParts) releaseFn(part);
      }

      // NaN guard on gradient norm (backward produced NaN even though loss was finite)
      if (!isFinite(gradNorm)) {
        if (useLossScaling) {
          // Dynamic loss scaling: halve the scale and retry
          lossScale /= 2;
          lossScaleReductions++;
          console.warn(`  [loss_scale] grad overflow at step ${step + 1} — reducing scale to ${lossScale}`);
          scaleSuccessCount = 0;
        } else {
          console.warn(`  [warn] grad_norm=NaN at step ${step + 1} — skipping optimizer update`);
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
      if (!nanDetected && trainConfig.spikeThreshold > 0 && gradNorm > trainConfig.spikeThreshold) {
        spikeSkips++;
        console.warn(`  [spike] grad_norm=${gradNorm.toFixed(1)} > ${trainConfig.spikeThreshold} — skipping step ${step + 1} (${spikeSkips} total skips)`);
        nanDetected = true; // reuse the skip path
      }
    } else {
      // Still need paramDataMap for zero-grad cleanup
      for (const [name, variable] of paramMap) {
        paramDataMap.set(name, variable.data);
      }
    }

    const _t4 = performance.now();

    // Clip gradients using backend.scale (stays on GPU)
    let clipCoef = 1.0;
    if (!nanDetected && trainConfig.gradClip > 0 && gradNorm > trainConfig.gradClip) {
      clipCoef = trainConfig.gradClip / gradNorm;
      clippedSteps++;
      for (const [, variable] of paramMap) {
        if (variable.grad) {
          const oldGrad = variable.grad;
          variable.grad = backend.scale(variable.grad, clipCoef);
          if (releaseFn) releaseFn(oldGrad);
        }
      }

      // Post-clip safety: spot-check the largest gradient tensors for Inf values.
      // In f16, Inf * clip_ratio = Inf — clipping cannot fix f16 overflow.
      // Check 3 largest tensors by computing their sum-of-squares on GPU.
      const gradEntries: { grad: TensorData; size: number }[] = [];
      for (const [, variable] of paramMap) {
        if (variable.grad) {
          gradEntries.push({ grad: variable.grad, size: shapeSize(variable.grad.shape) });
        }
      }
      gradEntries.sort((a, b) => b.size - a.size);
      for (let i = 0; i < Math.min(3, gradEntries.length); i++) {
        const g = gradEntries[i].grad;
        const g2 = backend.mul(g, g);
        const s = backend.sum(g2);
        const val = (s.data as Float32Array)[0];
        if (releaseFn) { releaseFn(g2); releaseFn(s); }
        if (!isFinite(val)) {
          console.warn(`  [warn] Inf in gradients after clipping (pre-clip norm=${gradNorm.toFixed(1)}) — skipping optimizer update`);
          nanDetected = true;
          break;
        }
      }
    }

    const _t4b = performance.now();

    if (!nanDetected) {
      for (const [name, variable] of paramMap) {
        if (variable.grad) gradMap.set(name, variable.grad);
      }

      // Optimizer step
      optimizer.step(paramDataMap, gradMap);
    }
    const _t5 = performance.now();

    // Zero gradients — explicitly release GPU buffers for param grads
    for (const [, variable] of paramMap) {
      if (variable.grad && releaseFn) releaseFn(variable.grad);
      variable.grad = null;
    }
    // Note: tape intermediates already released in the accumulation loop above

    // Flush pending GPU ops (optimizer commands + deferred buffer releases).
    if ("flush" in backend) (backend as any).flush();

    // Adaptive sync/GC policy: configurable cadence instead of every step.
    // syncEvery=1 (default) preserves original every-step behavior.
    // syncEvery=0 triggers sync only on pool pressure (deferred releases or pending destroys).
    const syncEvery = trainConfig.syncEvery;
    const gcEvery = trainConfig.gcEvery;
    const stepNum = step + 1;

    const needGc = typeof globalThis.gc === "function" && (
      gcEvery > 0 ? stepNum % gcEvery === 0 :
      // Adaptive: GC when deferred releases pile up
      ("gpuMemStats" in backend && (backend as any).gpuMemStats().deferredReleases > 50)
    );
    if (needGc) {
      (globalThis as any).gc();
      await new Promise<void>(resolve => setImmediate(resolve));
    }

    // SyncGpu: flush GC'd deferred releases, then WAIT for GPU completion.
    // This ensures output pool regions are reusable without OOM.
    const needSync = syncEvery === 1 ? true :
      syncEvery > 0 ? stepNum % syncEvery === 0 :
      // Adaptive: sync when pool pressure is high (many pending destroys or deferred releases)
      ("gpuMemStats" in backend && (() => {
        const s = (backend as any).gpuMemStats();
        return (s.deferredReleases > 20 || (s.pendingDestroys ?? 0) > 10);
      })());
    if (needSync) {
      if ("syncGpu" in backend) {
        (backend as any).syncGpu();
      } else if ("flush" in backend) {
        (backend as any).flush();
      }
    }
    const _t6 = performance.now();

    if (trainConfig.trace) {
      const gpuOps = "gpuOpsThisStep" in backend ? ` gpu_ops=${(backend as any).gpuOpsThisStep}` : "";
      console.log(`  [trace] data=${dataLoadMs.toFixed(0)}ms fwd=${fwdMs.toFixed(0)}ms bwd=${bwdMs.toFixed(0)}ms gradnorm=${(_t4-_t3).toFixed(0)}ms clip=${(_t4b-_t4).toFixed(0)}ms optim=${(_t5-_t4b).toFixed(0)}ms flush=${(_t6-_t5).toFixed(0)}ms${gpuOps}`);
    }

    // GPU memory diagnostics (every 5 steps, every step for first 20)
    if ("gpuMemStats" in backend && ((step + 1) <= 20 || (step + 1) % 5 === 0)) {
      const stats = (backend as any).gpuMemStats();
      const breakdown = "poolBreakdown" in backend ? ` | ${(backend as any).poolBreakdown(8)}` : "";
      const allocStr = stats.liveAllocs != null ? ` | allocs: ${stats.liveAllocs} live (${stats.totalAllocs} total, ${stats.totalAllocMB}MB)` : "";
      console.log(`  [gpu_mem] bufPool: ${stats.bufferPoolEntries} (${(stats.bufferPoolBytes/1024/1024).toFixed(1)}MB) | outPool: ${stats.outputPoolEntries}/${stats.outputPoolSizeClasses ?? "?"}cls (${(stats.outputPoolBytes/1024/1024).toFixed(1)}MB) | deferred: ${stats.deferredReleases} | pending: ${stats.pendingDestroys ?? 0}${allocStr}${breakdown}`);
    }

    // Metrics
    const stepElapsed = performance.now() - stepStart;
    const tokensProcessed = trainConfig.batchSize * modelConfig.blockSize * trainConfig.gradAccumSteps;
    const metrics: StepMetrics = {
      step: step + 1,
      loss: lossVal,
      lr,
      gradNorm,
      elapsed_ms: stepElapsed,
      tokens_per_sec: tokensProcessed / (stepElapsed / 1000),
      ms_per_iter: stepElapsed,
      // Per-step timing breakdown (always recorded)
      timing_fwd_ms: fwdMs,
      timing_bwd_ms: bwdMs,
      timing_grad_norm_ms: _t4 - _t3,
      timing_grad_clip_ms: _t4b - _t4,
      timing_optim_ms: _t5 - _t4b,
      timing_flush_ms: _t6 - _t5,
      timing_data_ms: dataLoadMs,
      gpu_ops_count: "gpuOpsThisStep" in backend ? (backend as any).gpuOpsThisStep : undefined,
      // Clipping telemetry
      clip_coef: clipCoef,
      clip_pct: (clippedSteps / (step + 1)) * 100,
    };

    // Symbio metrics (only when symbio is enabled — zero overhead otherwise)
    if (symbioEnabled && cusumDash && symbioCollector) {
      const clipPctVal = (clippedSteps / (step + 1)) * 100;
      const stepInfo: TrainerStepInfo = {
        step: step + 1,
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

      // Adaptive batch sizing
      if (adaptiveBatch) {
        const newBatch = adaptiveBatch.onAlert(cusumResult.cusum_alerts);
        metrics.adaptive_batch_size = newBatch;
        if (adaptiveBatch.changeReason) {
          metrics.batch_change_reason = adaptiveBatch.changeReason;
          console.log(`  [symbio] batch size → ${newBatch} (${adaptiveBatch.changeReason})`);
          adaptiveBatch.clearChangeReason();
        }
      }

      // Record loss for population entropy sliding window
      symbioCollector.recordLoss(lossVal);

      // Expensive metrics (every metricsInterval steps)
      if ((step + 1) % symbioConfig.metricsInterval === 0) {
        // Collect TensorData for weight params
        const paramTDs = new Map<string, TensorData>();
        for (const [name, v] of paramMap) paramTDs.set(name, v.data);

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
        metrics.symbio_candidate_activation = candidate.activation;
        metrics.symbio_generation = searchOrchestrator.generation;
        metrics.architecture_diversity = searchOrchestrator.architectureDiversity;

        const candidateDone = searchOrchestrator.recordStep(
          lossVal,
          metrics.valLoss,
          metrics.fitness_score,
        );

        if (candidateDone) {
          console.log(`  [symbio search] candidate ${candidate.id} done: bestLoss=${candidate.bestLoss.toFixed(4)} bestVal=${candidate.bestValLoss === Infinity ? "N/A" : candidate.bestValLoss.toFixed(4)} fitness=${candidate.fitnessScore === -Infinity ? "N/A" : candidate.fitnessScore.toFixed(4)}`);
          const nextActivation = searchOrchestrator.advance();

          if (nextActivation !== null) {
            // Switch to next candidate: re-init model with new activation
            const ffnDim = nextActivation === "swiglu"
              ? Math.ceil((8 / 3) * activeModelConfig.nEmbd / 64) * 64
              : modelConfig.ffnDim ?? 4 * activeModelConfig.nEmbd;
            activeModelConfig = { ...modelConfig, ffnActivation: nextActivation as any, ffnDim };
            rng.seed(trainConfig.seed + (step + 1)); // different seed per candidate for fresh init
            params = initGPT(activeModelConfig, backend, rng as any);
            totalParams = countParams(params);
            optimizer.loadStateDict({ step: 0, buffers: new Map() });
            clearForwardCache();
            const nextCandidate = searchOrchestrator.currentCandidate;
            console.log(`  [symbio search] → gen=${searchOrchestrator.generation} candidate=${nextCandidate?.id} activation=${nextActivation} params=${totalParams.toLocaleString()}`);
          } else {
            // Search complete
            const winner = searchOrchestrator.getWinner();
            console.log(`\n── symbio search complete ──`);
            if (winner) {
              console.log(`winner: ${winner.activation} (${winner.id}) | bestVal=${winner.bestValLoss === Infinity ? "N/A" : winner.bestValLoss.toFixed(4)} | fitness=${winner.fitnessScore === -Infinity ? "N/A" : winner.fitnessScore.toFixed(4)}`);
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
    const gpuStats = await queryGpuStats();
    if (gpuStats) {
      metrics.gpu_util_pct = gpuStats.utilPct;
      metrics.gpu_vram_used_mb = gpuStats.vramUsedMb;
      metrics.gpu_vram_total_mb = gpuStats.vramTotalMb;
    }
    if ("gpuMemStats" in backend) {
      const memStats = (backend as any).gpuMemStats();
      metrics.gpu_mem_pool_mb = Math.round((memStats.bufferPoolBytes + memStats.outputPoolBytes) / 1024 / 1024);
    }

    // Eval — flush GPU and wait for completion first to maximize free VRAM
    if (valLoader && (step + 1) % trainConfig.evalInterval === 0) {
      if ("flush" in backend) (backend as any).flush();
      if (typeof globalThis.gc === "function") {
        (globalThis as any).gc();
        await new Promise<void>(resolve => setImmediate(resolve));
      }
      // Second flush to process deferred releases from GC's FinalizationRegistry
      if ("flush" in backend) (backend as any).flush();

      let valLossSum = 0;
      for (let ei = 0; ei < trainConfig.evalIters; ei++) {
        const valBatch = valLoader.nextBatch();
        const evalTape = new Tape();
        const { loss: vl } = gptForward(activeModelConfig, params, backend, evalTape, valBatch.inputs, valBatch.targets);
        if (vl) {
          valLossSum += (vl.data.data as Float32Array)[0];
          if (releaseFn) releaseFn(vl.data);
        }
        evalTape.clear(releaseFn);
        // Flush between eval iters to process deferred releases
        if ("flush" in backend) (backend as any).flush();
      }
      metrics.valLoss = valLossSum / trainConfig.evalIters;

      // Eval-time diagnostic summary
      console.log(
        `  [diag] eval step ${step + 1}: loss=${lossVal.toFixed(4)}, ` +
        `val_loss=${metrics.valLoss.toFixed(4)}, ` +
        `grad_norm=${gradNorm.toFixed(2)}, ` +
        `clip_pct=${((clippedSteps / (step + 1)) * 100).toFixed(1)}%`
      );
    }

    // Log
    const lossStr = metrics.loss.toFixed(4);
    const valStr = metrics.valLoss !== undefined ? ` val_loss=${metrics.valLoss.toFixed(4)}` : "";
    const toksStr = (metrics.tokens_per_sec).toFixed(0);
    const gpuStr = "gpuOpsThisStep" in backend ? ` | ${(backend as any).gpuOpsThisStep} gpu_ops` : "";
    const clipStr = clipCoef < 1.0 ? ` clip=${clipCoef.toFixed(4)}` : "";
    const scaleStr = useLossScaling ? ` | scale=${lossScale}` : "";
    console.log(
      `step ${metrics.step}/${trainConfig.iters} | loss=${lossStr}${valStr} ` +
      `| lr=${lr.toExponential(2)} | grad_norm=${gradNorm.toFixed(3)}${clipStr} ` +
      `| ${metrics.ms_per_iter.toFixed(0)}ms/it | ${toksStr} tok/s${gpuStr}${scaleStr}`
    );

    // Buffer metrics JSONL (flush every 50 steps and on checkpoint)
    metricsBuffer.push(JSON.stringify(metrics) + "\n");
    if (metricsBuffer.length >= 50) await flushMetrics();

    if (onStep) onStep(metrics);

    // Yield to event loop so async callbacks (remote metric reporting, etc.) can process
    await new Promise<void>(resolve => setImmediate(resolve));

    // Checkpoint (save at every eval interval and at the end)
    if ((step + 1) % trainConfig.evalInterval === 0 || step + 1 === trainConfig.iters) {
      await flushMetrics(); // Ensure metrics are on disk before checkpoint
      const ckptPath = path.join(runDir, `checkpoint-${step + 1}.json`);
      const state = buildCheckpointState(params, optimizer, rng.state(), configHash, step + 1, activeModelConfig, deps.tokenizerArtifacts);
      await Effect.runPromise(new FileCheckpoint().save(ckptPath, state));
      console.log(`  checkpoint saved: ${ckptPath}`);
      if (deps.onCheckpoint) deps.onCheckpoint({ step: step + 1, path: ckptPath, runId: rid });
    }

    // Generate inference samples at sampleInterval (decoupled from checkpointing)
    const sampleInterval = trainConfig.sampleInterval;
    if (sampleInterval > 0 && ((step + 1) % sampleInterval === 0 || step + 1 === trainConfig.iters)) {
      if (deps.samplePrompts && deps.samplePrompts.length > 0) {
        try {
          // Flush GPU before sampling to maximize free VRAM
          if ("flush" in backend) (backend as any).flush();
          if (typeof globalThis.gc === "function") {
            (globalThis as any).gc();
            await new Promise<void>(resolve => setImmediate(resolve));
          }
          if ("flush" in backend) (backend as any).flush();

          const sampleCfg: SampleConfig = { steps: 50, temperature: 0.8, topk: 40 };
          const flushFn = "flush" in backend ? () => (backend as any).flush() : undefined;
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

          if (deps.onSamples) await deps.onSamples(samples, step + 1);
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
