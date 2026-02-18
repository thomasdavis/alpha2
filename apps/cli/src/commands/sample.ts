/**
 * Command: alpha sample
 *
 * Usage:
 *   alpha sample --checkpoint=runs/.../checkpoint-200.json --prompt="animal is"
 */
import { Effect } from "effect";
import { parseKV, requireArg, intArg, floatArg, strArg } from "../parse.js";
import { resolveBackend, resolveRng } from "../resolve.js";
import { FileCheckpoint, restoreParams, sample as runSample } from "@alpha/train";
import { initGPT } from "@alpha/model";
import { defaultSampleConfig } from "@alpha/core";
import { CharTokenizer, BpeTokenizer, WordTokenizer } from "@alpha/tokenizers";
import type { SampleConfig, Tokenizer } from "@alpha/core";

function tokenizerFromArtifacts(artifacts: { type: string; vocab: readonly string[]; merges?: readonly [number, number][] }): Tokenizer {
  if (artifacts.type === "bpe") {
    const tok = new BpeTokenizer();
    tok.loadArtifacts(artifacts as any);
    return tok;
  }
  if (artifacts.type === "word") {
    const tok = new WordTokenizer();
    tok.loadArtifacts(artifacts as any);
    return tok;
  }
  const tok = new CharTokenizer();
  tok.loadArtifacts(artifacts as any);
  return tok;
}

export async function sampleCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const checkpointPath = requireArg(kv, "checkpoint", "path to checkpoint");
  const prompt = strArg(kv, "prompt", "The ");
  const backendName = strArg(kv, "backend", "cpu_ref");

  const sampleConfig: SampleConfig = {
    steps: intArg(kv, "steps", defaultSampleConfig.steps),
    temperature: floatArg(kv, "temp", defaultSampleConfig.temperature),
    topk: intArg(kv, "topk", defaultSampleConfig.topk),
  };

  // Load checkpoint (contains model weights + tokenizer)
  const checkpoint = new FileCheckpoint();
  const state = await Effect.runPromise(checkpoint.load(checkpointPath));

  if (!state.tokenizerArtifacts) {
    console.error("Error: This checkpoint does not contain tokenizer artifacts.");
    console.error("It was saved before tokenizer embedding was added.");
    console.error("Please retrain or pass --data and --tokenizer to rebuild.");
    process.exit(1);
  }

  const backend = resolveBackend(backendName);
  const rng = resolveRng(state.rngState);

  // Restore tokenizer from checkpoint
  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);

  // Init model and restore weights
  const modelConfig = state.modelConfig;
  const params = initGPT(modelConfig, backend, rng as any);
  restoreParams(params, state.params);

  console.log(`Loaded checkpoint from step ${state.step}`);
  console.log(`Model: ${modelConfig.nLayer}L ${modelConfig.nEmbd}D ${modelConfig.nHead}H vocab=${modelConfig.vocabSize}`);
  console.log(`Tokenizer: ${state.tokenizerArtifacts.type} (${tokenizer.vocabSize} tokens)`);
  console.log(`Prompt: "${prompt}"`);
  console.log(`Sampling: steps=${sampleConfig.steps} temp=${sampleConfig.temperature} topk=${sampleConfig.topk}`);
  console.log(`---`);

  const output = runSample(
    modelConfig,
    params,
    backend,
    rng,
    (t) => tokenizer.encode(t),
    (t) => tokenizer.decode(t),
    prompt,
    sampleConfig,
  );

  console.log(output);
}
