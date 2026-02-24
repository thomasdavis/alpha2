/**
 * Command: alpha train
 */
import { parseKV, requireArg, intArg, floatArg, strArg, boolArg, loadConfig } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng, listImplementations } from "../resolve.js";
import { train as runTrain, createRemoteReporter, loadTextSample, sample as runSample } from "@alpha/train";
import type { SampleGeneration } from "@alpha/train";
import { defaultModelConfig, defaultTrainConfig, getDomain, domains } from "@alpha/core";
import type { ModelConfig, TrainConfig, TensorData } from "@alpha/core";
import { loadArtifacts } from "@alpha/tokenizers";
import { Effect } from "effect";

export async function trainCmd(args: string[]): Promise<void> {
  let kv = parseKV(args);
  kv = await loadConfig(kv);

  const dataPath = requireArg(kv, "data", "path to training text");
  const valDataPath = kv["valData"];

  // Look up domain config for defaults
  const domainId = kv["domain"];
  const domain = domainId ? getDomain(domainId) : undefined;
  if (domainId && !domain) {
    const available = Array.from(domains.keys()).join(", ");
    console.error(`Unknown domain: "${domainId}". Available: ${available}`);
    process.exit(1);
  }

  const mDefaults = domain?.modelDefaults ?? {};
  const tDefaults = domain?.trainDefaults ?? {};

  const modelConfig: ModelConfig = {
    vocabSize: intArg(kv, "vocabSize", mDefaults.vocabSize ?? defaultModelConfig.vocabSize),
    blockSize: intArg(kv, "block", mDefaults.blockSize ?? defaultModelConfig.blockSize),
    nLayer: intArg(kv, "layers", mDefaults.nLayer ?? defaultModelConfig.nLayer),
    nEmbd: intArg(kv, "dim", mDefaults.nEmbd ?? defaultModelConfig.nEmbd),
    nHead: intArg(kv, "heads", mDefaults.nHead ?? defaultModelConfig.nHead),
    dropout: floatArg(kv, "dropout", mDefaults.dropout ?? defaultModelConfig.dropout),
  };

  const trainConfig: TrainConfig = {
    iters: intArg(kv, "steps", intArg(kv, "iters", tDefaults.iters ?? defaultTrainConfig.iters)),
    batchSize: intArg(kv, "batch", tDefaults.batchSize ?? defaultTrainConfig.batchSize),
    lr: floatArg(kv, "lr", tDefaults.lr ?? defaultTrainConfig.lr),
    lrMin: floatArg(kv, "lrMin", tDefaults.lrMin ?? defaultTrainConfig.lrMin),
    warmupIters: intArg(kv, "warmupIters", tDefaults.warmupIters ?? defaultTrainConfig.warmupIters),
    beta1: floatArg(kv, "beta1", tDefaults.beta1 ?? defaultTrainConfig.beta1),
    beta2: floatArg(kv, "beta2", tDefaults.beta2 ?? defaultTrainConfig.beta2),
    eps: floatArg(kv, "eps", tDefaults.eps ?? defaultTrainConfig.eps),
    weightDecay: floatArg(kv, "weightDecay", tDefaults.weightDecay ?? defaultTrainConfig.weightDecay),
    gradClip: floatArg(kv, "gradClip", tDefaults.gradClip ?? defaultTrainConfig.gradClip),
    evalInterval: intArg(kv, "evalInterval", tDefaults.evalInterval ?? defaultTrainConfig.evalInterval),
    evalIters: intArg(kv, "evalIters", tDefaults.evalIters ?? defaultTrainConfig.evalIters),
    seed: intArg(kv, "seed", tDefaults.seed ?? defaultTrainConfig.seed),
    backend: strArg(kv, "backend", tDefaults.backend ?? defaultTrainConfig.backend),
    tokenizer: domain ? domain.tokenizer : strArg(kv, "tokenizer", defaultTrainConfig.tokenizer),
    optimizer: strArg(kv, "optim", tDefaults.optimizer ?? defaultTrainConfig.optimizer),
    logLevel: strArg(kv, "log", (tDefaults.logLevel ?? defaultTrainConfig.logLevel)) as any,
    trace: boolArg(kv, "trace", tDefaults.trace ?? defaultTrainConfig.trace),
    gradAccumSteps: intArg(kv, "accumSteps", tDefaults.gradAccumSteps ?? defaultTrainConfig.gradAccumSteps),
    sampleInterval: intArg(kv, "sampleInterval", tDefaults.sampleInterval ?? defaultTrainConfig.sampleInterval),
    spikeThreshold: floatArg(kv, "spikeThreshold", tDefaults.spikeThreshold ?? defaultTrainConfig.spikeThreshold),
  };

  console.log(`Implementations available:\n${listImplementations()}\n`);

  // Resolve implementations
  const backend = resolveBackend(trainConfig.backend);
  let tokenizer = resolveTokenizer(trainConfig.tokenizer);
  // Build no-decay set: embeddings + LayerNorm weights/biases should skip weight decay
  const noDecayNames = new Set<string>();
  noDecayNames.add("wte");
  noDecayNames.add("wpe");
  noDecayNames.add("lnF.weight");
  noDecayNames.add("lnF.bias");
  for (let i = 0; i < modelConfig.nLayer; i++) {
    noDecayNames.add(`layer.${i}.ln1.weight`);
    noDecayNames.add(`layer.${i}.ln1.bias`);
    noDecayNames.add(`layer.${i}.ln2.weight`);
    noDecayNames.add(`layer.${i}.ln2.bias`);
  }

  const optimizer = resolveOptimizer(trainConfig.optimizer, backend, {
    lr: trainConfig.lr,
    beta1: trainConfig.beta1,
    beta2: trainConfig.beta2,
    eps: trainConfig.eps,
    weightDecay: trainConfig.weightDecay,
    noDecayNames,
  });
  const rng = resolveRng(trainConfig.seed);

  // Build tokenizer from training data (sample first 100MB for large files)
  const text = await loadTextSample(dataPath, 100 * 1024 * 1024);
  const tokenizerArtifacts = await Effect.runPromise(tokenizer.build(text));

  // Override vocab size from tokenizer
  const finalModelConfig: ModelConfig = {
    ...modelConfig,
    vocabSize: tokenizer.vocabSize,
  };

  // Set up remote reporter if env vars are configured
  const remoteUrl = process.env.ALPHA_REMOTE_URL;
  const remoteSecret = process.env.ALPHA_REMOTE_SECRET;
  const discordWebhook = process.env.DISCORD_WEBHOOK_URL;
  const reporter = remoteUrl && remoteSecret
    ? createRemoteReporter({ url: remoteUrl, secret: remoteSecret, discordWebhook })
    : null;

  if (reporter) {
    console.log(`Remote reporting: ${remoteUrl}`);
  }

  const params = await runTrain({
    backend,
    tokenizer,
    optimizer,
    rng,
    modelConfig: finalModelConfig,
    trainConfig,
    dataPath,
    valDataPath,
    tokenizerArtifacts,
    runDir: kv["runDir"],
    resumePath: kv["resume"],
    domain: domainId,
    samplePrompts: (domain?.samplePrompts ?? ["The ", "Once upon a time", "He walked into"]),
    onStart: reporter
      ? (info) => reporter.registerRun({
          ...info,
          domain: domainId,
          modelConfig: finalModelConfig,
          trainConfig,
          dataPath: info.dataPath,
          infra: info.infra,
        })
      : undefined,
    onStep: reporter ? (metrics) => reporter.onStep(metrics) : undefined,
    // Skip intermediate checkpoint uploads — only upload final checkpoint after training
    onCheckpoint: undefined,
    onSamples: reporter
      ? (samples, step) => reporter.sendSamples(samples, step)
      : undefined,
    activationCheckpointing: boolArg(kv, "checkpoint", false),
    mixedPrecision: boolArg(kv, "fp16", false),
  });

  // Post-training sample generation
  const samplePrompts = domain?.samplePrompts ?? ["The ", "Once upon a time", "He walked into"];
  const extraPrompts = ["In the beginning ", "We the People of "];
  const allPrompts = [...samplePrompts, ...extraPrompts].slice(0, 5);
  const releaseFn = "releaseGpuTensor" in backend
    ? (td: TensorData) => (backend as any).releaseGpuTensor(td)
    : undefined;
  const flushFn = "flush" in backend ? () => (backend as any).flush() : undefined;

  console.log("\n── sample generations ──");
  const samples: SampleGeneration[] = [];
  for (const prompt of allPrompts) {
    // Flush GPU between samples to reclaim buffers
    if (flushFn) flushFn();
    const output = runSample(
      finalModelConfig, params, backend, rng,
      (t) => tokenizer.encode(t),
      (t) => tokenizer.decode(t),
      prompt,
      { steps: 100, temperature: 0.8, topk: 40 },
      releaseFn, flushFn,
    );
    console.log(`\n  prompt: "${prompt}"`);
    console.log(`  output: ${output}`);
    samples.push({ prompt, output });
  }

  if (reporter) {
    await reporter.sendSamples(samples, trainConfig.iters);

    // Upload final checkpoint to remote server for inference
    const runDir = kv["runDir"] ?? `runs/${reporter.runId}`;
    const finalCkptPath = `${runDir}/checkpoint-${trainConfig.iters}.json`;
    try {
      const fs = await import("node:fs/promises");
      await fs.access(finalCkptPath);
      console.log(`\nUploading final checkpoint: ${finalCkptPath}`);
      reporter.uploadCheckpoint({ step: trainConfig.iters, path: finalCkptPath, runId: reporter.runId });
    } catch {
      // Final checkpoint may not exist if training was interrupted
      console.log(`Final checkpoint not found at ${finalCkptPath}, skipping upload`);
    }

    await reporter.complete(trainConfig.iters);
  }
}
