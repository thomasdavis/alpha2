/**
 * Demo script: rebuild BPE tokenizer, load checkpoint, eval + sample.
 */
import { Effect } from "effect";
import { SeededRng } from "@alpha/core";
import { CpuRefBackend } from "@alpha/tensor";
import { BpeTokenizer, loadArtifacts } from "@alpha/tokenizers";
import { initGPT } from "@alpha/model";
import { FileCheckpoint, restoreParams, evaluate, sample } from "@alpha/train";
import { DataLoader, loadText } from "@alpha/train";
import { Tape } from "@alpha/autograd";
import { gptForward } from "@alpha/model";

async function main() {
  const backend = new CpuRefBackend();
  const rng = new SeededRng(42);

  // Load checkpoint
  const ckptPath = process.argv[2] ?? "runs/20260217101225_3pjr/checkpoint-200.json";
  const dataPath = process.argv[3] ?? "data/animals.txt";

  console.log("Loading checkpoint...");
  const checkpoint = new FileCheckpoint();
  const state = await Effect.runPromise(checkpoint.load(ckptPath));
  const modelConfig = state.modelConfig;

  console.log(`Model: ${modelConfig.nLayer}L ${modelConfig.nEmbd}D ${modelConfig.nHead}H vocab=${modelConfig.vocabSize}`);

  // Rebuild BPE tokenizer from training data
  console.log("Rebuilding BPE tokenizer...");
  const tokenizer = new BpeTokenizer(1000);
  const text = await loadText(dataPath);
  await Effect.runPromise(tokenizer.build(text));
  console.log(`Tokenizer vocab: ${tokenizer.vocabSize}`);

  // Init model and restore weights
  const params = initGPT(modelConfig, backend, rng);
  restoreParams(params, state.params);
  console.log(`Restored weights from step ${state.step}\n`);

  // Eval
  console.log("── Evaluation ──");
  const loader = DataLoader.fromText(text, tokenizer, rng, 4, modelConfig.blockSize);
  let totalLoss = 0;
  const evalBatches = 20;
  for (let i = 0; i < evalBatches; i++) {
    const batch = loader.nextBatch();
    const tape = new Tape();
    const { loss } = gptForward(modelConfig, params, backend, tape, batch.inputs, batch.targets);
    if (loss) totalLoss += (loss.data.data as Float32Array)[0];
  }
  const avgLoss = totalLoss / evalBatches;
  console.log(`Loss:       ${avgLoss.toFixed(4)}`);
  console.log(`Perplexity: ${Math.exp(avgLoss).toFixed(2)}`);

  // Sampling
  console.log("\n── Sampling (temperature=0.8, topk=40) ──\n");
  const prompts = [
    "The cat ",
    "The elephant ",
    "The wolf ",
    "The bear ",
    "A ",
  ];

  for (const prompt of prompts) {
    const generated = sample(
      modelConfig, params, backend, rng,
      (t) => tokenizer.encode(t),
      (t) => tokenizer.decode(t),
      prompt,
      { steps: 120, temperature: 0.8, topk: 40 },
    );
    console.log(`prompt: "${prompt}"`);
    console.log(`output: ${generated}\n`);
  }

  // Also try low temperature for more coherent output
  console.log("── Sampling (temperature=0.3, topk=10) ──\n");
  for (const prompt of ["The dog ", "The shark "]) {
    const generated = sample(
      modelConfig, params, backend, rng,
      (t) => tokenizer.encode(t),
      (t) => tokenizer.decode(t),
      prompt,
      { steps: 120, temperature: 0.3, topk: 10 },
    );
    console.log(`prompt: "${prompt}"`);
    console.log(`output: ${generated}\n`);
  }
}

main().catch(console.error);
