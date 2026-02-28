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
  prepareInferenceWeights,
  createSession,
  cloneSession,
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

type BeamCandidate = { token: number; logProb: number };

function topTokenCandidates(
  logits: Float32Array,
  temperature: number,
  topk: number,
  topp: number,
  maxCandidates: number,
): BeamCandidate[] {
  const vocabSize = logits.length;
  if (vocabSize === 0 || maxCandidates <= 0) return [];

  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < vocabSize; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return [{ token: best, logProb: 0 }];
  }

  const topK = Number.isFinite(topk) ? Math.max(0, Math.floor(topk)) : 0;
  const topP = Number.isFinite(topp) ? Math.min(1, Math.max(0, topp)) : 1;
  const invTemp = 1 / temperature;
  const scaled = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) scaled[i] = logits[i] * invTemp;

  if (topK > 0 && topK < vocabSize) {
    const sorted = Array.from(scaled).sort((a, b) => b - a);
    const threshold = sorted[topK - 1];
    for (let i = 0; i < vocabSize; i++) {
      if (scaled[i] < threshold) scaled[i] = -Infinity;
    }
  }

  let maxVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (scaled[i] > maxVal) maxVal = scaled[i];
  }
  if (!Number.isFinite(maxVal)) return [];

  const probs = new Float32Array(vocabSize);
  const active: number[] = [];
  let sumExp = 0;
  for (let i = 0; i < vocabSize; i++) {
    const v = scaled[i];
    if (Number.isFinite(v)) {
      const p = Math.exp(v - maxVal);
      probs[i] = p;
      sumExp += p;
      active.push(i);
    }
  }
  if (active.length === 0 || sumExp <= 0) return [];

  active.sort((a, b) => probs[b] - probs[a]);
  let keepCount = active.length;
  if (topP > 0 && topP < 1) {
    const target = sumExp * topP;
    let cumulative = 0;
    keepCount = 0;
    while (keepCount < active.length) {
      cumulative += probs[active[keepCount]];
      keepCount++;
      if (cumulative >= target) break;
    }
    if (keepCount <= 0) keepCount = 1;
  }

  const limit = Math.min(maxCandidates, keepCount);
  let keptMass = 0;
  for (let i = 0; i < keepCount; i++) keptMass += probs[active[i]];
  keptMass = Math.max(keptMass, 1e-30);
  const invMass = 1 / keptMass;

  const out: BeamCandidate[] = [];
  for (let i = 0; i < limit; i++) {
    const tok = active[i];
    const p = Math.max(1e-30, probs[tok] * invMass);
    out.push({ token: tok, logProb: Math.log(p) });
  }
  return out;
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
    topp: floatArg(kv, "topp", floatArg(kv, "topP", defaultSampleConfig.topp ?? 1.0)),
  };
  const beamWidth = intArg(kv, "beam", 1);
  const beamLengthPenalty = floatArg(kv, "beamLengthPenalty", 1.0);
  if (beamWidth < 1) {
    console.error("Error: --beam must be >= 1");
    process.exit(1);
  }

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
  console.log(`Sampling: steps=${sampleConfig.steps} temp=${sampleConfig.temperature} topk=${sampleConfig.topk} topp=${sampleConfig.topp ?? 1} beam=${beamWidth}`);
  console.log(`Engine: ${useSlow ? "autograd (slow)" : "inference (fast)"}`);
  console.log(`---`);

  if (useSlow && beamWidth > 1) {
    console.error("Error: beam search is only supported in fast inference mode (omit --slow).");
    process.exit(1);
  }

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
    if (beamWidth > 1) {
      // Beam search path (deterministic, no RNG).
      const weights = prepareInferenceWeights(modelConfig, state.params);
      const promptTokens = tokenizer.encode(prompt);
      const tokens = new Int32Array(promptTokens);

      const prefillStart = performance.now();
      const rootSession = createSession(weights);
      const rootLogits = prefill(weights, rootSession, tokens);
      const prefillMs = performance.now() - prefillStart;

      type Beam = {
        session: ReturnType<typeof createSession>;
        logits: Float32Array;
        generated: number[];
        logProb: number;
        pos: number;
        finished: boolean;
      };
      const scoreBeam = (b: Beam): number => {
        const len = Math.max(1, b.generated.length);
        return b.logProb / Math.pow(len, Math.max(0, beamLengthPenalty));
      };

      let beams: Beam[] = [{
        session: rootSession,
        logits: rootLogits,
        generated: [],
        logProb: 0,
        pos: tokens.length,
        finished: false,
      }];

      const decodeStart = performance.now();
      for (let step = 0; step < sampleConfig.steps; step++) {
        const candidates: Beam[] = [];
        let expanded = false;

        for (const beam of beams) {
          if (beam.finished || beam.pos >= modelConfig.blockSize) {
            candidates.push(beam);
            continue;
          }
          const tokenCands = topTokenCandidates(
            beam.logits,
            sampleConfig.temperature,
            sampleConfig.topk,
            sampleConfig.topp ?? 1,
            beamWidth,
          );
          if (tokenCands.length === 0) {
            candidates.push({ ...beam, finished: true });
            continue;
          }
          expanded = true;
          for (const cand of tokenCands) {
            const childSession = cloneSession(beam.session);
            const nextLogits = decodeStep(weights, childSession, cand.token, beam.pos);
            const tokenStr = tokenizer.decode([cand.token]);
            const finished = tokenStr.includes("<|end_of_text|>") || (beam.pos + 1 >= modelConfig.blockSize);
            candidates.push({
              session: childSession,
              logits: nextLogits,
              generated: [...beam.generated, cand.token],
              logProb: beam.logProb + cand.logProb,
              pos: beam.pos + 1,
              finished,
            });
          }
        }

        candidates.sort((a, b) => scoreBeam(b) - scoreBeam(a));
        beams = candidates.slice(0, beamWidth);
        if (!expanded || beams.every((b) => b.finished)) break;
      }
      const decodeMs = performance.now() - decodeStart;
      beams.sort((a, b) => scoreBeam(b) - scoreBeam(a));
      const best = beams[0];
      const promptStr = tokenizer.decode(Array.from(tokens));
      const generatedStr = tokenizer.decode(best.generated);
      console.log(promptStr + generatedStr);

      const totalMs = prefillMs + decodeMs;
      const tokPerSec = best.generated.length / (totalMs / 1000);
      console.log(`\n--- stats ---`);
      console.log(`Prefill: ${tokens.length} tokens in ${prefillMs.toFixed(0)}ms (${(tokens.length / (prefillMs / 1000)).toFixed(1)} tok/s)`);
      console.log(`Decode: ${best.generated.length} tokens in ${decodeMs.toFixed(0)}ms (${(best.generated.length / (decodeMs / 1000)).toFixed(1)} tok/s)`);
      console.log(`Total: ${totalMs.toFixed(0)}ms (${tokPerSec.toFixed(1)} tok/s)`);
      console.log(`Beam: width=${beamWidth} lengthPenalty=${beamLengthPenalty.toFixed(2)} score=${scoreBeam(best).toFixed(4)}`);
      return;
    }

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
    let nextToken = sampleFromLogits(model, logits, sampleConfig.temperature, sampleConfig.topk, rng, sampleConfig.topp ?? 1);
    generated.push(nextToken);

    // Decode loop
    const decodeStart = performance.now();
    for (let i = 1; i < sampleConfig.steps; i++) {
      if (pos >= modelConfig.blockSize) break;
      logits = decodeStep(model, nextToken, pos);
      pos++;
      nextToken = sampleFromLogits(model, logits, sampleConfig.temperature, sampleConfig.topk, rng, sampleConfig.topp ?? 1);
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
