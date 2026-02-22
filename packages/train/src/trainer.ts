/**
 * Training loop orchestrator.
 *
 * Pure orchestration: depends on services (Backend, Tokenizer, Optimizer, Checkpoint, Logger).
 * Inspired by microgpt.py's training loop but with proper batching and logging.
 */
import type {
  ModelConfig, TrainConfig, Backend, Tokenizer, Optimizer, Rng, TensorData,
} from "@alpha/core";
import { shapeSize, hashConfig, runId as makeRunId } from "@alpha/core";
import { Tape } from "@alpha/autograd";
import { initGPT, gptForward, collectParams, countParams, type GPTParams } from "@alpha/model";
import { DataLoader, loadText, loadAndTokenize, loadOrCacheTokens, getSplitByte } from "./data.js";
import { FileCheckpoint, buildCheckpointState, restoreParams } from "./checkpoint.js";
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
  onStart?: (info: { runId: string; configHash: string; totalParams: number; dataPath: string }) => void | Promise<void>;
  onCheckpoint?: (info: { step: number; path: string; runId: string }) => void;
  domain?: string;
}

export async function train(deps: TrainerDeps): Promise<GPTParams> {
  const {
    backend, tokenizer, optimizer, rng, modelConfig, trainConfig,
    dataPath, valDataPath, resumePath, onStep, onStart,
  } = deps;

  const rid = makeRunId();
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

  // Open metrics log
  const metricsPath = path.join(runDir, "metrics.jsonl");
  const metricsHandle = await fs.open(metricsPath, "a");

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
  const params = initGPT(modelConfig, backend, rng as any);
  const totalParams = countParams(params);

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

  // Notify start
  if (onStart) await onStart({ runId: rid, configHash, totalParams, dataPath });

  // Log header
  const paramBytes = totalParams * 4;
  console.log(`── alpha training ──`);
  console.log(`run_id: ${rid}`);
  console.log(`config_hash: ${configHash}`);
  console.log(`params: ${totalParams.toLocaleString()} (${(paramBytes / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`backend: ${backend.name} | tokenizer: ${tokenizer.name} | optimizer: ${optimizer.name}`);
  console.log(`seed: ${trainConfig.seed} | block_size: ${modelConfig.blockSize} | batch: ${trainConfig.batchSize}`);
  console.log(`iters: ${trainConfig.iters} | lr: ${trainConfig.lr}`);

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
  const trainLoader = new DataLoader(trainTokens, rng, trainConfig.batchSize, modelConfig.blockSize);
  const valLoader = valTokens.length > 0
    ? new DataLoader(valTokens, rng, trainConfig.batchSize, modelConfig.blockSize)
    : undefined;

  // Training loop
  const startTime = performance.now();

  for (let step = startStep; step < trainConfig.iters; step++) {
    const stepStart = performance.now();

    // Learning rate schedule: linear warmup + cosine decay
    const warmup = Math.min(100, trainConfig.iters / 10);
    let lr: number;
    if (step < warmup) {
      lr = trainConfig.lr * (step + 1) / warmup;
    } else {
      const decay = (step - warmup) / (trainConfig.iters - warmup);
      lr = trainConfig.lr * 0.5 * (1 + Math.cos(Math.PI * decay));
    }
    if (optimizer && "setLr" in optimizer) {
      (optimizer as any).setLr(lr);
    }

    // Reset per-step GPU ops counter
    if ("resetStepOps" in backend) (backend as any).resetStepOps();

    // Forward + backward
    const tape = new Tape();
    const _t0 = performance.now();
    const batch = trainLoader.nextBatch();
    const _t1 = performance.now();
    const { loss } = gptForward(modelConfig, params, backend, tape, batch.inputs, batch.targets);
    const _t2 = performance.now();

    if (!loss) throw new Error("Loss is undefined");

    // Explicit GPU buffer release function — deterministic cleanup instead of
    // relying on FinalizationRegistry which is unreliable for timely VRAM reclaim.
    const releaseFn = "releaseGpuTensor" in backend
      ? (td: TensorData) => (backend as any).releaseGpuTensor(td)
      : undefined;

    tape.backward(loss, backend, releaseFn);
    const _t3 = performance.now();

    // Collect gradients and compute gradient norm via backend ops (stays on GPU)
    const paramMap = collectParams(params);
    const paramDataMap = new Map<string, TensorData>();
    const gradMap = new Map<string, TensorData>();

    // Compute gradient norm using backend ops (avoids CPU readback of all grads)
    const sqNormParts: TensorData[] = [];
    const g2Intermediates: TensorData[] = [];
    for (const [name, variable] of paramMap) {
      paramDataMap.set(name, variable.data);
      if (variable.grad) {
        const g = variable.grad;
        // backend.sum(backend.mul(g, g)) computes sum of squares on GPU
        const g2 = backend.mul(g, g);
        g2Intermediates.push(g2);
        sqNormParts.push(backend.sum(g2));
      }
    }
    // Sum all scalar parts (these are tiny — just read back the scalars)
    let gradNormSq = 0;
    for (const part of sqNormParts) {
      gradNormSq += (part.data as Float32Array)[0];
    }
    const gradNorm = Math.sqrt(gradNormSq);
    // Release grad norm intermediates (g² tensors and scalar parts)
    if (releaseFn) {
      for (const g2 of g2Intermediates) releaseFn(g2);
      for (const part of sqNormParts) releaseFn(part);
    }
    const _t4 = performance.now();

    // NaN guard: skip optimizer step if gradients are NaN (prevents permanent
    // weight corruption from a single bad batch)
    const nanDetected = !isFinite(gradNorm);
    if (nanDetected) {
      console.warn(`  [warn] NaN/Inf grad_norm at step ${step + 1} — skipping optimizer update`);
    }

    // Clip gradients using backend.scale (stays on GPU)
    if (!nanDetected && trainConfig.gradClip > 0 && gradNorm > trainConfig.gradClip) {
      const clipCoef = trainConfig.gradClip / gradNorm;
      for (const [, variable] of paramMap) {
        if (variable.grad) {
          const oldGrad = variable.grad;
          variable.grad = backend.scale(variable.grad, clipCoef);
          if (releaseFn) releaseFn(oldGrad);
        }
      }
    }

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
    // Release all forward/backward intermediate GPU buffers deterministically
    tape.clear(releaseFn);

    // Flush pending GPU ops, then run GC as defense-in-depth for backward
    // closure-internal intermediates (transpose results, etc.) that can't be
    // explicitly released because they're local to the backward closure scope.
    if ("flush" in backend) (backend as any).flush();
    if (typeof globalThis.gc === "function") {
      (globalThis as any).gc();
      await new Promise<void>(resolve => setImmediate(resolve));
    }
    const _t6 = performance.now();

    if (trainConfig.trace) {
      const gpuOps = "gpuOpsThisStep" in backend ? ` gpu_ops=${(backend as any).gpuOpsThisStep}` : "";
      console.log(`  [trace] batch=${(_t1-_t0).toFixed(0)}ms fwd=${(_t2-_t1).toFixed(0)}ms bwd=${(_t3-_t2).toFixed(0)}ms gradnorm=${(_t4-_t3).toFixed(0)}ms optim=${(_t5-_t4).toFixed(0)}ms flush=${(_t6-_t5).toFixed(0)}ms${gpuOps}`);
    }

    // GPU memory diagnostics (every 10 steps)
    if ("gpuMemStats" in backend && (step + 1) % 10 === 0) {
      const stats = (backend as any).gpuMemStats();
      console.log(`  [gpu_mem] bufPool: ${stats.bufferPoolEntries} (${(stats.bufferPoolBytes/1024/1024).toFixed(1)}MB) | outPool: ${stats.outputPoolEntries} (${(stats.outputPoolBytes/1024/1024).toFixed(1)}MB) | deferred: ${stats.deferredReleases}`);
    }

    // Metrics
    const stepElapsed = performance.now() - stepStart;
    const tokensProcessed = trainConfig.batchSize * modelConfig.blockSize;
    const metrics: StepMetrics = {
      step: step + 1,
      loss: (loss.data.data as Float32Array)[0],
      lr,
      gradNorm,
      elapsed_ms: stepElapsed,
      tokens_per_sec: tokensProcessed / (stepElapsed / 1000),
      ms_per_iter: stepElapsed,
    };

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

    // Eval
    if (valLoader && (step + 1) % trainConfig.evalInterval === 0) {
      let valLossSum = 0;
      for (let ei = 0; ei < trainConfig.evalIters; ei++) {
        const valBatch = valLoader.nextBatch();
        const evalTape = new Tape();
        const { loss: vl } = gptForward(modelConfig, params, backend, evalTape, valBatch.inputs, valBatch.targets);
        if (vl) valLossSum += (vl.data.data as Float32Array)[0];
        evalTape.clear(releaseFn);
      }
      metrics.valLoss = valLossSum / trainConfig.evalIters;
    }

    // Log
    const lossStr = metrics.loss.toFixed(4);
    const valStr = metrics.valLoss !== undefined ? ` val_loss=${metrics.valLoss.toFixed(4)}` : "";
    const toksStr = (metrics.tokens_per_sec).toFixed(0);
    const gpuStr = "gpuOpsThisStep" in backend ? ` | ${(backend as any).gpuOpsThisStep} gpu_ops` : "";
    console.log(
      `step ${metrics.step}/${trainConfig.iters} | loss=${lossStr}${valStr} ` +
      `| lr=${lr.toExponential(2)} | grad_norm=${gradNorm.toFixed(3)} ` +
      `| ${metrics.ms_per_iter.toFixed(0)}ms/it | ${toksStr} tok/s${gpuStr}`
    );

    // Write metrics JSONL
    await metricsHandle.write(JSON.stringify(metrics) + "\n");

    if (onStep) onStep(metrics);

    // Checkpoint (save at every eval interval and at the end)
    if ((step + 1) % trainConfig.evalInterval === 0 || step + 1 === trainConfig.iters) {
      const ckptPath = path.join(runDir, `checkpoint-${step + 1}.json`);
      const state = buildCheckpointState(params, optimizer, rng.state(), configHash, step + 1, modelConfig, deps.tokenizerArtifacts);
      await Effect.runPromise(new FileCheckpoint().save(ckptPath, state));
      console.log(`  checkpoint saved: ${ckptPath}`);
      if (deps.onCheckpoint) deps.onCheckpoint({ step: step + 1, path: ckptPath, runId: rid });
    }
  }

  const totalTime = performance.now() - startTime;
  console.log(`\n── training complete ──`);
  console.log(`total time: ${(totalTime / 1000).toFixed(1)}s`);

  await metricsHandle.close();
  return params;
}
