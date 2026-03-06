
import { initDevice } from "../packages/helios/src/device.js";
import { HeliosBackend } from "../packages/helios/src/backend.js";
import { train } from "../packages/train/src/trainer.ts";
import { tokenizerRegistry } from "../packages/tokenizers/src/index.js";
import { SeededRng } from "../packages/core/src/index.js";
import { createOptimizerRegistry } from "../packages/train/src/optimizers.js";
import { readFileSync, existsSync } from "node:fs";

async function main() {
  await initDevice();
  const backend = new HeliosBackend();
  const rng = new SeededRng(42);
  const optimizerRegistry = createOptimizerRegistry(backend);

  const tokenizerArtifacts = JSON.parse(readFileSync("/home/ajax/alpha-repo/runs/tokenizer-artifacts-super-chat-bpe4k-v3.json", "utf8"));
  const tokenizer = tokenizerRegistry.get("bpe-chat-4k");
  tokenizer.loadArtifacts(tokenizerArtifacts);

  const modelConfig = {
    vocabSize: 4096,
    blockSize: 256,
    nLayer: 6,
    nEmbd: 192,
    nHead: 6,
    ffnActivation: "silu",
    ffnDim: 512,
    dropout: 0.1,
  };

  const trainConfig = {
    tokenizer: "bpe-chat-4k",
    lr: 8e-5,
    lrMin: 8e-6,
    warmupIters: 2000,
    beta2: 0.95,
    eps: 1e-8,
    weightDecay: 0.01,
    batchSize: 10,
    gradAccumSteps: 2,
    gradClip: 0.4,
    iters: 5, // Corrected from 'steps'
    sampleInterval: 9999,
    evalInterval: 9999,
  };

  console.log("Starting mini-train...");
  await train({
    backend,
    tokenizer,
    optimizer: optimizerRegistry.get("adamw"),
    rng,
    modelConfig,
    trainConfig,
    dataPath: "/home/ajax/alpha-repo/data/super_chat.txt",
    tokenizerArtifacts,
    mixedPrecision: true,
    onStep: (m) => {
      console.log(`Step ${m.step} | Loss: ${m.loss} | GradNorm: ${m.gradNorm}`);
    }
  });
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
