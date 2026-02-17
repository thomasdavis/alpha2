/**
 * Command: alpha train
 */
import { parseKV, requireArg, intArg, floatArg, strArg, boolArg, loadConfig } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng, listImplementations } from "../resolve.js";
import { train as runTrain } from "@alpha/train";
import { defaultModelConfig, defaultTrainConfig } from "@alpha/core";
import type { ModelConfig, TrainConfig } from "@alpha/core";
import { loadArtifacts } from "@alpha/tokenizers";
import { Effect } from "effect";

export async function trainCmd(args: string[]): Promise<void> {
  let kv = parseKV(args);
  kv = await loadConfig(kv);

  const dataPath = requireArg(kv, "data", "path to training text");
  const valDataPath = kv["valData"];

  const modelConfig: ModelConfig = {
    vocabSize: intArg(kv, "vocabSize", defaultModelConfig.vocabSize),
    blockSize: intArg(kv, "block", defaultModelConfig.blockSize),
    nLayer: intArg(kv, "layers", defaultModelConfig.nLayer),
    nEmbd: intArg(kv, "dim", defaultModelConfig.nEmbd),
    nHead: intArg(kv, "heads", defaultModelConfig.nHead),
    dropout: floatArg(kv, "dropout", defaultModelConfig.dropout),
  };

  const trainConfig: TrainConfig = {
    iters: intArg(kv, "iters", defaultTrainConfig.iters),
    batchSize: intArg(kv, "batch", defaultTrainConfig.batchSize),
    lr: floatArg(kv, "lr", defaultTrainConfig.lr),
    beta1: floatArg(kv, "beta1", defaultTrainConfig.beta1),
    beta2: floatArg(kv, "beta2", defaultTrainConfig.beta2),
    eps: floatArg(kv, "eps", defaultTrainConfig.eps),
    weightDecay: floatArg(kv, "weightDecay", defaultTrainConfig.weightDecay),
    gradClip: floatArg(kv, "gradClip", defaultTrainConfig.gradClip),
    evalInterval: intArg(kv, "evalInterval", defaultTrainConfig.evalInterval),
    evalIters: intArg(kv, "evalIters", defaultTrainConfig.evalIters),
    seed: intArg(kv, "seed", defaultTrainConfig.seed),
    backend: strArg(kv, "backend", defaultTrainConfig.backend),
    tokenizer: strArg(kv, "tokenizer", defaultTrainConfig.tokenizer),
    optimizer: strArg(kv, "optim", defaultTrainConfig.optimizer),
    logLevel: strArg(kv, "log", defaultTrainConfig.logLevel) as any,
    trace: boolArg(kv, "trace", defaultTrainConfig.trace),
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
  });
}
