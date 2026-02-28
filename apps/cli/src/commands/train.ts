/**
 * Command: alpha train
 */
import { parseKV, requireArg, intArg, floatArg, strArg, boolArg, loadConfig } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng, listImplementations } from "../resolve.js";
import { train as runTrain, createRemoteReporter, loadTextSample, sample as runSample } from "@alpha/train";
import type { SampleGeneration } from "@alpha/train";
import { defaultModelConfig, defaultTrainConfig, getDomain, domains } from "@alpha/core";
import type { ModelConfig, TrainConfig, TensorData } from "@alpha/core";
import { loadArtifacts, saveArtifacts } from "@alpha/tokenizers";
import { loadSymbioConfig, applySymbioModelPreset, applySymbioTrainPreset, deserializeGraph } from "@alpha/symbiogenesis";
import { Effect } from "effect";

type GpuProfile = "auto" | "none" | "l4";

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
  const gpuProfileRaw = strArg(kv, "gpuProfile", "auto").toLowerCase();
  if (gpuProfileRaw !== "auto" && gpuProfileRaw !== "none" && gpuProfileRaw !== "l4") {
    console.error(`Unknown gpuProfile: "${gpuProfileRaw}". Use auto|none|l4.`);
    process.exit(1);
  }
  const gpuProfile = gpuProfileRaw as GpuProfile;

  const mDefaults = domain?.modelDefaults ?? {};
  const tDefaults = domain?.trainDefaults ?? {};

  let modelConfig: ModelConfig = {
    vocabSize: intArg(kv, "vocabSize", mDefaults.vocabSize ?? defaultModelConfig.vocabSize),
    blockSize: intArg(kv, "block", mDefaults.blockSize ?? defaultModelConfig.blockSize),
    nLayer: intArg(kv, "layers", mDefaults.nLayer ?? defaultModelConfig.nLayer),
    nEmbd: intArg(kv, "dim", mDefaults.nEmbd ?? defaultModelConfig.nEmbd),
    nHead: intArg(kv, "heads", mDefaults.nHead ?? defaultModelConfig.nHead),
    dropout: floatArg(kv, "dropout", mDefaults.dropout ?? defaultModelConfig.dropout),
    ffnActivation: (strArg(kv, "activation", mDefaults.ffnActivation ?? defaultModelConfig.ffnActivation ?? "gelu") as ModelConfig["ffnActivation"]),
    ffnDim: kv["ffnDim"] ? intArg(kv, "ffnDim", 0) : undefined,
  };

  let trainConfig: TrainConfig = {
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
    logEvery: intArg(kv, "logEvery", tDefaults.logEvery ?? defaultTrainConfig.logEvery ?? 1),
    trace: boolArg(kv, "trace", tDefaults.trace ?? defaultTrainConfig.trace),
    gradAccumSteps: intArg(kv, "accumSteps", tDefaults.gradAccumSteps ?? defaultTrainConfig.gradAccumSteps),
    sampleInterval: intArg(kv, "sampleInterval", tDefaults.sampleInterval ?? defaultTrainConfig.sampleInterval),
    spikeThreshold: floatArg(kv, "spikeThreshold", tDefaults.spikeThreshold ?? defaultTrainConfig.spikeThreshold),
    syncEvery: intArg(kv, "syncEvery", tDefaults.syncEvery ?? defaultTrainConfig.syncEvery),
    gcEvery: intArg(kv, "gcEvery", tDefaults.gcEvery ?? defaultTrainConfig.gcEvery),
    packed: boolArg(kv, "packed", tDefaults.packed ?? defaultTrainConfig.packed),
    symbio: boolArg(kv, "symbio", tDefaults.symbio ?? defaultTrainConfig.symbio),
    symbioConfig: null,
  };

  // Apply symbio preset if enabled (before resolving implementations)
  if (trainConfig.symbio) {
    const symbioConfigFile = kv["symbio-config"];
    const symbioConfig = await loadSymbioConfig(symbioConfigFile);

    // Apply symbio model/train presets (explicit CLI flags already override above)
    const presetModel = applySymbioModelPreset(modelConfig);
    const presetTrain = applySymbioTrainPreset(trainConfig);

    // Only apply preset values that weren't explicitly overridden by CLI
    if (!kv["activation"]) modelConfig = { ...modelConfig, ffnActivation: presetModel.ffnActivation, ffnDim: presetModel.ffnDim };
    if (!kv["lr"]) trainConfig = { ...trainConfig, lr: presetTrain.lr };
    if (!kv["gradClip"]) trainConfig = { ...trainConfig, gradClip: presetTrain.gradClip };
    if (!kv["warmupIters"]) trainConfig = { ...trainConfig, warmupIters: presetTrain.warmupIters };
    if (!kv["spikeThreshold"]) trainConfig = { ...trainConfig, spikeThreshold: presetTrain.spikeThreshold };

    trainConfig = { ...trainConfig, symbioConfig: symbioConfig as unknown as Record<string, unknown> };
    console.log(`Symbiogenesis mode: ON`);
    console.log(`  activation: ${modelConfig.ffnActivation ?? "gelu"} | lr: ${trainConfig.lr} | gradClip: ${trainConfig.gradClip}`);
  }

  console.log(`Implementations available:\n${listImplementations()}\n`);

  // Resolve implementations
  const backend = resolveBackend(trainConfig.backend);
  let mixedPrecisionEnabled = boolArg(kv, "fp16", false);
  const minGpuSizeOverride = kv["minGpuSize"] !== undefined ? intArg(kv, "minGpuSize", 0) : null;
  const backendAny = backend as any;
  const setMinGpuSize: ((n: number) => void) | null =
    typeof backendAny.setMinGpuSize === "function"
      ? backendAny.setMinGpuSize.bind(backendAny)
      : null;
  const getDeviceInfo: (() => { deviceName: string; vendorId: number; f16Supported: boolean; minGpuSize: number }) | null =
    typeof backendAny.getDeviceInfo === "function"
      ? backendAny.getDeviceInfo.bind(backendAny)
      : null;
  const deviceInfo = getDeviceInfo ? getDeviceInfo() : null;
  const isL4Gpu = !!(deviceInfo && deviceInfo.vendorId === 0x10de && /\bL4\b/i.test(deviceInfo.deviceName));
  const useL4Profile = gpuProfile === "l4" || (gpuProfile === "auto" && isL4Gpu);
  if (useL4Profile) {
    if (!kv["fp16"] && !!deviceInfo?.f16Supported) mixedPrecisionEnabled = true;
    if (!kv["batch"]) trainConfig = { ...trainConfig, batchSize: Math.max(trainConfig.batchSize, 8) };
    if (!kv["packed"]) trainConfig = { ...trainConfig, packed: true };
    if (!kv["logEvery"]) trainConfig = { ...trainConfig, logEvery: Math.max(trainConfig.logEvery ?? 1, 25) };
    if (!kv["minGpuSize"] && setMinGpuSize) setMinGpuSize(2048);
    const mode = gpuProfile === "auto" ? "auto-detected" : "explicit";
    console.log(`GPU profile: l4 (${mode})`);
    console.log(`  tuned: batch=${trainConfig.batchSize} fp16=${mixedPrecisionEnabled} packed=${trainConfig.packed} logEvery=${trainConfig.logEvery} minGpuSize=${kv["minGpuSize"] ?? "2048"}`);
  }
  if (minGpuSizeOverride !== null) {
    if (minGpuSizeOverride <= 0) {
      console.error(`Invalid minGpuSize: ${minGpuSizeOverride}. Must be > 0.`);
      process.exit(1);
    }
    if (setMinGpuSize) {
      setMinGpuSize(minGpuSizeOverride);
    } else {
      console.warn(`minGpuSize ignored: backend "${backend.name}" does not expose setMinGpuSize().`);
    }
  }
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

  // Tokenizer artifacts:
  // - if --tokenizerArtifacts=<path> is passed and file exists, load it
  // - otherwise build from a 100MB sample and optionally persist to that path
  const tokenizerArtifactsPath = kv["tokenizerArtifacts"];
  let tokenizerArtifacts;
  if (tokenizerArtifactsPath) {
    const fs = await import("node:fs/promises");
    const hasArtifacts = await fs.access(tokenizerArtifactsPath).then(() => true).catch(() => false);
    if (hasArtifacts) {
      tokenizerArtifacts = await Effect.runPromise(loadArtifacts(tokenizerArtifactsPath));
      const tokWithArtifacts = tokenizer as any;
      if (typeof tokWithArtifacts.loadArtifacts === "function") {
        tokWithArtifacts.loadArtifacts(tokenizerArtifacts as any);
      } else {
        throw new Error(`Tokenizer ${tokenizer.name} does not support loading artifacts`);
      }
      console.log(`Tokenizer artifacts: loaded ${tokenizerArtifactsPath}`);
    } else {
      const text = await loadTextSample(dataPath, 100 * 1024 * 1024);
      tokenizerArtifacts = await Effect.runPromise(tokenizer.build(text));
      await Effect.runPromise(saveArtifacts(tokenizerArtifactsPath, tokenizerArtifacts));
      console.log(`Tokenizer artifacts: built and saved to ${tokenizerArtifactsPath}`);
    }
  } else {
    const text = await loadTextSample(dataPath, 100 * 1024 * 1024);
    tokenizerArtifacts = await Effect.runPromise(tokenizer.build(text));
  }

  // Override vocab size from tokenizer
  const finalModelConfig: ModelConfig = {
    ...modelConfig,
    vocabSize: tokenizer.vocabSize,
  };

  // Set up remote reporter if env vars are configured
  const enableRemote = boolArg(kv, "remote", true);
  const remoteUrl = process.env.ALPHA_REMOTE_URL;
  const remoteSecret = process.env.ALPHA_REMOTE_SECRET;
  const discordWebhook = process.env.DISCORD_WEBHOOK_URL;
  const reporter = enableRemote && remoteUrl && remoteSecret
    ? createRemoteReporter({ url: remoteUrl, secret: remoteSecret, discordWebhook })
    : null;
  const postSamples = boolArg(kv, "postSamples", true);

  if (reporter) {
    console.log(`Remote reporting: ${remoteUrl}`);
  }

  const { params, modelConfig: trainedModelConfig } = await runTrain({
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
    resumeActivationGraph: kv["symbio-resume-graph"]
      ? JSON.parse(kv["symbio-resume-graph"])
      : undefined,
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
    mixedPrecision: mixedPrecisionEnabled,
  });

  const coopStats = typeof backendAny.getMatmulCoopStats === "function"
    ? backendAny.getMatmulCoopStats()
    : null;
  if (coopStats && coopStats.totalMatmulDispatches > 0) {
    const pct = (coopStats.coopHitRate * 100).toFixed(1);
    console.log(
      `coop_matmul: ${coopStats.coopDispatches}/${coopStats.totalMatmulDispatches} (${pct}%)` +
      ` direct=${coopStats.coopDirectDispatches}` +
      ` padded2d=${coopStats.coopPadded2DDispatches}` +
      ` padded_batched=${coopStats.coopPaddedBatchedDispatches}` +
      ` transposed_a_rewrite=${coopStats.coopTransposedARewriteDispatches}`,
    );
  }

  // Post-training sample generation
  const samplePrompts = domain?.samplePrompts ?? ["The ", "Once upon a time", "He walked into"];
  const extraPrompts = ["In the beginning ", "We the People of "];
  const allPrompts = [...samplePrompts, ...extraPrompts].slice(0, 5);
  const releaseFn = "releaseGpuTensor" in backend
    ? (td: TensorData) => (backend as any).releaseGpuTensor(td)
    : undefined;
  const flushFn = "flush" in backend ? () => (backend as any).flush() : undefined;

  const samples: SampleGeneration[] = [];
  if (postSamples) {
    console.log("\n── sample generations ──");
    for (const prompt of allPrompts) {
      // Flush GPU between samples to reclaim buffers
      if (flushFn) flushFn();
      const output = runSample(
        trainedModelConfig, params, backend, rng,
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
  } else {
    console.log("\nSkipping post-training sample generation (--postSamples=false)");
  }

  if (reporter) {
    if (samples.length > 0) {
      await reporter.sendSamples(samples, trainConfig.iters);
    }

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
