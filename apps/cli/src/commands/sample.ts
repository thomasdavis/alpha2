/**
 * Command: alpha sample
 *
 * Usage:
 *   alpha sample --checkpoint=runs/.../checkpoint-200.json --prompt="animal is"
 *   alpha sample --checkpoint=... --prompt="Hello" --slow   (use autograd path)
 *
 * By default uses @alpha/inference for fast CPU inference (KV cache, tiled
 * matmul, zero-allocation decode loop — 10-20× faster than autograd).
 * Pass --slow to use the original autograd-based forward pass.
 */
import { Effect } from "effect";
import { parseKV, requireArg, intArg, floatArg, strArg, boolArg } from "../parse.js";
import { resolveBackend, resolveRng } from "../resolve.js";
import { FileCheckpoint, restoreParams, sample as runSample } from "@alpha/train";
import { initGPT } from "@alpha/model";
import { defaultSampleConfig, SeededRng } from "@alpha/core";
import { CharTokenizer, BpeTokenizer, WordTokenizer } from "@alpha/tokenizers";
import {
  prepareInferenceModel,
  resetCache,
  prefill,
  decodeStep,
  sampleFromLogits,
} from "@alpha/inference";
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
  const useSlow = boolArg(kv, "slow", false);

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

  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);
  const modelConfig = state.modelConfig;

  console.log(`Loaded checkpoint from step ${state.step}`);
  console.log(`Model: ${modelConfig.nLayer}L ${modelConfig.nEmbd}D ${modelConfig.nHead}H vocab=${modelConfig.vocabSize}`);
  console.log(`Tokenizer: ${state.tokenizerArtifacts.type} (${tokenizer.vocabSize} tokens)`);
  console.log(`Prompt: "${prompt}"`);
  console.log(`Sampling: steps=${sampleConfig.steps} temp=${sampleConfig.temperature} topk=${sampleConfig.topk}`);
  console.log(`Engine: ${useSlow ? "autograd (slow)" : "inference (fast)"}`);
  console.log(`---`);

  if (useSlow) {
    // Original autograd-based forward pass
    const backend = resolveBackend(backendName);
    const rng = resolveRng(state.rngState);
    const params = initGPT(modelConfig, backend, rng as any);
    restoreParams(params, state.params);

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
  } else {
    // Fast inference engine with KV cache
    const model = prepareInferenceModel(modelConfig, state.params);
    resetCache(model);

    const rng = new SeededRng(state.rngState ?? 42);

    // Encode prompt
    const promptTokens = tokenizer.encode(prompt);
    const tokens = new Int32Array(promptTokens);

    // Prefill: process all prompt tokens at once
    const t0 = performance.now();
    let logits = prefill(model, tokens);
    const prefillMs = performance.now() - t0;

    // Collect generated token IDs
    const generated: number[] = [];
    let pos = tokens.length;

    // Sample first token from prefill logits
    let nextToken = sampleFromLogits(model, logits, sampleConfig.temperature, sampleConfig.topk, rng);
    generated.push(nextToken);

    // Decode loop
    const decodeStart = performance.now();
    for (let i = 1; i < sampleConfig.steps; i++) {
      if (pos >= modelConfig.blockSize) break;
      logits = decodeStep(model, nextToken, pos);
      pos++;
      nextToken = sampleFromLogits(model, logits, sampleConfig.temperature, sampleConfig.topk, rng);
      generated.push(nextToken);

      // Stop on end-of-text token
      const tokenStr = tokenizer.decode([nextToken]);
      if (tokenStr.includes("<|end_of_text|>")) break;
    }
    const decodeMs = performance.now() - decodeStart;

    // Decode and print the full output
    const promptStr = tokenizer.decode(Array.from(tokens));
    const generatedStr = tokenizer.decode(generated);
    console.log(promptStr + generatedStr);

    // Performance stats
    const totalMs = prefillMs + decodeMs;
    const tokPerSec = generated.length / (totalMs / 1000);
    console.log(`\n--- stats ---`);
    console.log(`Prefill: ${tokens.length} tokens in ${prefillMs.toFixed(0)}ms (${(tokens.length / (prefillMs / 1000)).toFixed(1)} tok/s)`);
    console.log(`Decode: ${generated.length} tokens in ${decodeMs.toFixed(0)}ms (${(generated.length / (decodeMs / 1000)).toFixed(1)} tok/s)`);
    console.log(`Total: ${totalMs.toFixed(0)}ms (${tokPerSec.toFixed(1)} tok/s)`);
  }
}
