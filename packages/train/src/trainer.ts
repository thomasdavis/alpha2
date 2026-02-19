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
import { DataLoader, loadText, splitText } from "./data.js";
import { FileCheckpoint, buildCheckpointState, restoreParams } from "./checkpoint.js";
import { Effect } from "effect";

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
  onStart?: (info: { runId: string; configHash: string; totalParams: number; dataPath: string }) => void;
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

  // Load data
  const rawText = await loadText(dataPath);
  let valText: string | undefined;
  if (valDataPath) {
    valText = await loadText(valDataPath);
  } else {
    const split = splitText(rawText);
    valText = split.val;
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
  if (onStart) onStart({ runId: rid, configHash, totalParams, dataPath });

  // Log header
  const paramBytes = totalParams * 4;
  console.log(`── alpha training ──`);
  console.log(`run_id: ${rid}`);
  console.log(`config_hash: ${configHash}`);
  console.log(`params: ${totalParams.toLocaleString()} (${(paramBytes / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`backend: ${backend.name} | tokenizer: ${tokenizer.name} | optimizer: ${optimizer.name}`);
  console.log(`seed: ${trainConfig.seed} | block_size: ${modelConfig.blockSize} | batch: ${trainConfig.batchSize}`);
  console.log(`iters: ${trainConfig.iters} | lr: ${trainConfig.lr}`);
  console.log(``);

  // Create data loaders
  const trainLoader = DataLoader.fromText(rawText, tokenizer, rng, trainConfig.batchSize, modelConfig.blockSize);
  const valLoader = valText
    ? DataLoader.fromText(valText, tokenizer, rng, trainConfig.batchSize, modelConfig.blockSize)
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

    // Forward + backward
    const tape = new Tape();
    const batch = trainLoader.nextBatch();
    const { loss } = gptForward(modelConfig, params, backend, tape, batch.inputs, batch.targets);

    if (!loss) throw new Error("Loss is undefined");
    tape.backward(loss, backend);

    // Collect gradients
    const paramMap = collectParams(params);
    const paramDataMap = new Map<string, TensorData>();
    const gradMap = new Map<string, TensorData>();

    let gradNorm = 0;
    for (const [name, variable] of paramMap) {
      paramDataMap.set(name, variable.data);
      if (variable.grad) {
        // Grad clipping
        const gArr = variable.grad.data as Float32Array;
        for (let i = 0; i < gArr.length; i++) {
          gradNorm += gArr[i] * gArr[i];
        }
      }
    }
    gradNorm = Math.sqrt(gradNorm);

    // Clip gradients
    if (trainConfig.gradClip > 0 && gradNorm > trainConfig.gradClip) {
      const clipCoef = trainConfig.gradClip / gradNorm;
      for (const [, variable] of paramMap) {
        if (variable.grad) {
          const gArr = variable.grad.data as Float32Array;
          for (let i = 0; i < gArr.length; i++) gArr[i] *= clipCoef;
        }
      }
    }

    for (const [name, variable] of paramMap) {
      if (variable.grad) gradMap.set(name, variable.grad);
    }

    // Optimizer step
    optimizer.step(paramDataMap, gradMap);

    // Zero gradients
    for (const [, variable] of paramMap) {
      variable.grad = null;
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

    // Eval
    if (valLoader && (step + 1) % trainConfig.evalInterval === 0) {
      let valLossSum = 0;
      for (let ei = 0; ei < trainConfig.evalIters; ei++) {
        const valBatch = valLoader.nextBatch();
        const evalTape = new Tape();
        const { loss: vl } = gptForward(modelConfig, params, backend, evalTape, valBatch.inputs, valBatch.targets);
        if (vl) valLossSum += (vl.data.data as Float32Array)[0];
      }
      metrics.valLoss = valLossSum / trainConfig.evalIters;
    }

    // Log
    const lossStr = metrics.loss.toFixed(4);
    const valStr = metrics.valLoss !== undefined ? ` val_loss=${metrics.valLoss.toFixed(4)}` : "";
    const toksStr = (metrics.tokens_per_sec).toFixed(0);
    console.log(
      `step ${metrics.step}/${trainConfig.iters} | loss=${lossStr}${valStr} ` +
      `| lr=${lr.toExponential(2)} | grad_norm=${gradNorm.toFixed(3)} ` +
      `| ${metrics.ms_per_iter.toFixed(0)}ms/it | ${toksStr} tok/s`
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
