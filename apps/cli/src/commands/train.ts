/**
 * Command: alpha train
 */
import { parseKV, requireArg, intArg, floatArg, strArg, boolArg, loadConfig } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng, listImplementations } from "../resolve.js";
import { train as runTrain, createRemoteReporter } from "@alpha/train";
import { defaultModelConfig, defaultTrainConfig, getDomain } from "@alpha/core";
import type { ModelConfig, TrainConfig } from "@alpha/core";
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
    console.error(`Unknown domain: "${domainId}". Available: novels, chords`);
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
    iters: intArg(kv, "iters", tDefaults.iters ?? defaultTrainConfig.iters),
    batchSize: intArg(kv, "batch", tDefaults.batchSize ?? defaultTrainConfig.batchSize),
    lr: floatArg(kv, "lr", tDefaults.lr ?? defaultTrainConfig.lr),
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
  };

  console.log(`Implementations available:\n${listImplementations()}\n`);

  // Resolve implementations
  const backend = resolveBackend(trainConfig.backend);
  let tokenizer = resolveTokenizer(trainConfig.tokenizer);
  const optimizer = resolveOptimizer(trainConfig.optimizer, backend);
  const rng = resolveRng(trainConfig.seed);

  // Build tokenizer from training data
  const fs = await import("node:fs/promises");
  const text = await fs.readFile(dataPath, "utf-8");
  const tokenizerArtifacts = await Effect.runPromise(tokenizer.build(text));

  // Override vocab size from tokenizer
  const finalModelConfig: ModelConfig = {
    ...modelConfig,
    vocabSize: tokenizer.vocabSize,
  };

  // Set up remote reporter if env vars are configured
  const remoteUrl = process.env.ALPHA_REMOTE_URL;
  const remoteSecret = process.env.ALPHA_REMOTE_SECRET;
  const reporter = remoteUrl && remoteSecret
    ? createRemoteReporter({ url: remoteUrl, secret: remoteSecret })
    : null;

  if (reporter) {
    console.log(`Remote reporting: ${remoteUrl}`);
  }

  await runTrain({
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
    onStart: reporter
      ? (info) => reporter.registerRun({
          ...info,
          domain: domainId,
          modelConfig: finalModelConfig,
          trainConfig,
        })
      : undefined,
    onStep: reporter ? (metrics) => reporter.onStep(metrics) : undefined,
  });

  if (reporter) {
    await reporter.complete(trainConfig.iters);
  }
}
