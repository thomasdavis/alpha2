/**
 * Sampling / inference from a trained GPT model.
 */
import type { ModelConfig, Backend, Rng, TensorData, SampleConfig } from "@alpha/core";
import { shapeSize } from "@alpha/core";
import type { GPTParams } from "@alpha/model";
import { gptForward } from "@alpha/model";
import { Tape } from "@alpha/autograd";

/**
 * Generate text from a trained model.
 *
 * @param releaseTensor — Optional callback to release GPU buffers for intermediate
 *   tensors after each forward step. Without this, inference creates hundreds of
 *   GPU tensors (one tape per step) that accumulate until GC — causing OOM on long
 *   generations. Pass `backend.releaseGpuTensor` for GPU backends.
 * @param flushGpu — Optional callback to flush the GPU compute graph and process
 *   deferred buffer releases. Called periodically during inference to prevent
 *   GPU buffer accumulation from non-tape tensors (causal mask, position indices).
 */
export function sample(
  config: ModelConfig,
  params: GPTParams,
  backend: Backend,
  rng: Rng,
  encode: (text: string) => Int32Array,
  decode: (tokens: ArrayLike<number>) => string,
  prompt: string,
  sampleConfig: SampleConfig,
  releaseTensor?: (td: TensorData) => void,
  flushGpu?: () => void,
): string {
  const { steps, temperature, topk } = sampleConfig;
  const topP = Number.isFinite(sampleConfig.topp) ? Math.min(1, Math.max(0, sampleConfig.topp!)) : 1;

  // Encode prompt
  const promptTokens = encode(prompt);
  const maxLen = Math.min(promptTokens.length + steps, config.blockSize);
  const tokens = new Int32Array(maxLen);
  tokens.set(promptTokens);
  let currentLen = promptTokens.length;

  for (let i = 0; i < steps && currentLen < config.blockSize; i++) {
    // Use last blockSize tokens as context
    const ctxStart = Math.max(0, currentLen - config.blockSize);
    const ctxLen = currentLen - ctxStart;
    const ctx = tokens.slice(ctxStart, ctxStart + ctxLen);

    const inputData: TensorData = {
      shape: [1, ctxLen],
      dtype: "i32",
      data: new Int32Array(ctx),
    };

    // Forward pass (tape records ops but we don't need gradients)
    const tape = new Tape();
    const { logits } = gptForward(config, params, backend, tape, inputData);

    // Get logits for last position: [1, ctxLen, vocabSize] → [vocabSize]
    const vocabSize = config.vocabSize;
    const lastLogits = new Float32Array(vocabSize);
    const logitsArr = logits.data.data as Float32Array;
    const offset = (ctxLen - 1) * vocabSize;
    if (temperature <= 0) {
      for (let v = 0; v < vocabSize; v++) lastLogits[v] = logitsArr[offset + v];
    } else {
      for (let v = 0; v < vocabSize; v++) {
        lastLogits[v] = logitsArr[offset + v] / temperature;
      }
    }

    // Release tape entries to free GPU buffers (prevents OOM on long generations)
    tape.clear(releaseTensor);

    // Also release the inputData GPU buffer — it was uploaded to GPU by ensureGpu
    // inside embedding() but is not tracked by the tape. Without this, each
    // inference step leaks one GPU buffer for the input tokens.
    if (releaseTensor) releaseTensor(inputData);

    // Flush GPU periodically to process deferred buffer releases and reclaim
    // non-tape GPU buffers (causal mask, position indices) freed by FinalizationRegistry.
    // Without this, ancillary buffers accumulate over hundreds of inference steps
    // and can hit Vulkan's maxMemoryAllocationCount (~4096).
    if (flushGpu && (i & 7) === 7) flushGpu();

    // Greedy decode when temperature <= 0.
    if (temperature <= 0) {
      let bestToken = 0;
      let bestVal = lastLogits[0];
      for (let v = 1; v < vocabSize; v++) {
        if (lastLogits[v] > bestVal) {
          bestVal = lastLogits[v];
          bestToken = v;
        }
      }
      if (currentLen < maxLen) {
        tokens[currentLen] = bestToken;
        currentLen++;
      }
      continue;
    }

    // Top-k filtering
    if (topk > 0 && topk < vocabSize) {
      const indexed = Array.from(lastLogits).map((v, i) => ({ v, i }));
      indexed.sort((a, b) => b.v - a.v);
      const threshold = indexed[topk - 1].v;
      for (let v = 0; v < vocabSize; v++) {
        if (lastLogits[v] < threshold) lastLogits[v] = -Infinity;
      }
    }

    // Top-p (nucleus) filtering.
    if (topP > 0 && topP < 1) {
      let maxVal = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        if (lastLogits[v] > maxVal) maxVal = lastLogits[v];
      }
      if (Number.isFinite(maxVal)) {
        const probs = new Float32Array(vocabSize);
        const active: number[] = [];
        let sumExp = 0;
        for (let v = 0; v < vocabSize; v++) {
          if (Number.isFinite(lastLogits[v])) {
            const p = Math.exp(lastLogits[v] - maxVal);
            probs[v] = p;
            active.push(v);
            sumExp += p;
          }
        }
        if (sumExp > 0 && active.length > 0) {
          active.sort((a, b) => probs[b] - probs[a]);
          const target = sumExp * topP;
          let keptMass = 0;
          let keepCount = 0;
          for (; keepCount < active.length; keepCount++) {
            keptMass += probs[active[keepCount]];
            if (keptMass >= target) {
              keepCount++;
              break;
            }
          }
          if (keepCount <= 0) keepCount = 1;
          if (keptMass <= 0) keptMass = probs[active[0]];

          const rTopP = rng.next() * keptMass;
          let cumTopP = 0;
          let sampled = active[keepCount - 1];
          for (let iTopP = 0; iTopP < keepCount; iTopP++) {
            const idx = active[iTopP];
            cumTopP += probs[idx];
            if (rTopP < cumTopP) {
              sampled = idx;
              break;
            }
          }

          if (currentLen < maxLen) {
            tokens[currentLen] = sampled;
            currentLen++;
          }
          continue;
        }
      }
    }

    // Softmax (full/top-k filtered path when top-p is disabled)
    let maxVal = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      if (lastLogits[v] > maxVal) maxVal = lastLogits[v];
    }
    let sumExp = 0;
    const probs = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      probs[v] = Math.exp(lastLogits[v] - maxVal);
      sumExp += probs[v];
    }
    for (let v = 0; v < vocabSize; v++) {
      probs[v] /= sumExp;
    }

    // Sample from distribution
    const r = rng.next();
    let cumsum = 0;
    let nextToken = 0;
    for (let v = 0; v < vocabSize; v++) {
      cumsum += probs[v];
      if (r < cumsum) {
        nextToken = v;
        break;
      }
    }

    if (currentLen < maxLen) {
      tokens[currentLen] = nextToken;
      currentLen++;
    }
  }

  return decode(tokens.slice(0, currentLen));
}
