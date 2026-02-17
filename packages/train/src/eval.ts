/**
 * Evaluation: compute loss/perplexity on a validation set.
 */
import type { ModelConfig, Backend, Tokenizer, Rng, TensorData } from "@alpha/core";
import { Tape } from "@alpha/autograd";
import { gptForward, type GPTParams } from "@alpha/model";
import { DataLoader, loadText } from "./data.js";

export interface EvalResult {
  loss: number;
  perplexity: number;
  nBatches: number;
}

export async function evaluate(
  config: ModelConfig,
  params: GPTParams,
  backend: Backend,
  tokenizer: Tokenizer,
  rng: Rng,
  dataPath: string,
  batchSize: number,
  nBatches = 50,
): Promise<EvalResult> {
  const text = await loadText(dataPath);
  const loader = DataLoader.fromText(text, tokenizer, rng, batchSize, config.blockSize);

  let totalLoss = 0;
  for (let i = 0; i < nBatches; i++) {
    const batch = loader.nextBatch();
    const tape = new Tape();
    const { loss } = gptForward(config, params, backend, tape, batch.inputs, batch.targets);
    if (loss) {
      totalLoss += (loss.data.data as Float32Array)[0];
    }
  }

  const avgLoss = totalLoss / nBatches;
  return {
    loss: avgLoss,
    perplexity: Math.exp(avgLoss),
    nBatches,
  };
}
