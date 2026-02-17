/**
 * Command: alpha eval
 *
 * Usage:
 *   alpha eval --checkpoint=runs/.../checkpoint-200.json --data=data/val.txt
 */
import { Effect } from "effect";
import { parseKV, requireArg, intArg, strArg } from "../parse.js";
import { resolveBackend, resolveRng } from "../resolve.js";
import { FileCheckpoint, evaluate, restoreParams } from "@alpha/train";
import { initGPT } from "@alpha/model";
import { CharTokenizer, BpeTokenizer } from "@alpha/tokenizers";
import type { Tokenizer } from "@alpha/core";

function tokenizerFromArtifacts(artifacts: { type: string; vocab: readonly string[]; merges?: readonly [number, number][] }): Tokenizer {
  if (artifacts.type === "bpe") {
    const tok = new BpeTokenizer();
    tok.loadArtifacts(artifacts as any);
    return tok;
  }
  const tok = new CharTokenizer();
  tok.loadArtifacts(artifacts as any);
  return tok;
}

export async function evalCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const checkpointPath = requireArg(kv, "checkpoint", "path to checkpoint");
  const dataPath = requireArg(kv, "data", "path to validation text");
  const backendName = strArg(kv, "backend", "cpu_ref");
  const batchSize = intArg(kv, "batch", 4);
  const nBatches = intArg(kv, "nBatches", 50);

  // Load checkpoint (contains model weights + tokenizer)
  const checkpoint = new FileCheckpoint();
  const state = await Effect.runPromise(checkpoint.load(checkpointPath));

  if (!state.tokenizerArtifacts) {
    console.error("Error: This checkpoint does not contain tokenizer artifacts. Please retrain.");
    process.exit(1);
  }

  const backend = resolveBackend(backendName);
  const rng = resolveRng(state.rngState);

  // Restore tokenizer from checkpoint
  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);

  // Init and restore model
  const modelConfig = state.modelConfig;
  const params = initGPT(modelConfig, backend, rng as any);
  restoreParams(params, state.params);

  console.log(`Evaluating checkpoint from step ${state.step}`);
  console.log(`Model: ${modelConfig.nLayer}L ${modelConfig.nEmbd}D ${modelConfig.nHead}H vocab=${modelConfig.vocabSize}`);
  console.log(`Tokenizer: ${state.tokenizerArtifacts.type} (${tokenizer.vocabSize} tokens)`);
  console.log();

  const result = await evaluate(
    modelConfig, params, backend, tokenizer, rng, dataPath, batchSize, nBatches,
  );

  console.log(`Loss:       ${result.loss.toFixed(4)}`);
  console.log(`Perplexity: ${result.perplexity.toFixed(2)}`);
  console.log(`Batches:    ${result.nBatches}`);
}
