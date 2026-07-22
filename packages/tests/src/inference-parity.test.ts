/**
 * inference-parity — the fast CPU inference engine (@alpha/inference) must
 * produce the SAME logits as the autograd cpu_ref forward (@alpha/model
 * gptForward), for BOTH architectures the engine serves:
 *
 *   - Llama-form: RoPE + RMSNorm + SwiGLU + tied embeddings (no wpe/bias/lmHead).
 *     This is the [defect P3] regression: prepareInferenceWeights used to crash
 *     ("Cannot read properties of undefined") on such checkpoints because it
 *     unconditionally extracted wpe / ln*.bias / lmHead. The engine now handles
 *     the Llama-form params and computes RoPE/RMSNorm/SwiGLU/tied faithfully.
 *   - GPT-2-form: learned wpe + LayerNorm + GELU + untied (regression guard for
 *     the pre-existing path — small fresh-init scores stay well inside softCap).
 *
 * Coverage: prefill (all positions at once) AND KV-cache decode (prefill T-1 +
 * decodeStep the last token) — the decode path is where RoPE absolute positions
 * are easiest to get wrong.
 */
import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type TensorData } from "@alpha/core";
import { Tape } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries } from "@alpha/model";
import {
  prepareInferenceWeights,
  createSession,
  prefill,
  decodeStep,
} from "@alpha/inference";

const B = new CpuRefBackend();

function llamaConfig(vocabSize: number): ModelConfig {
  return {
    vocabSize,
    blockSize: 16,
    nLayer: 2,
    nEmbd: 32,
    nHead: 4, // headDim = 8 (even)
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 40,
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10000,
    tieEmbeddings: true,
  };
}

function gpt2Config(vocabSize: number): ModelConfig {
  return {
    vocabSize,
    blockSize: 16,
    nLayer: 2,
    nEmbd: 32,
    nHead: 4,
    dropout: 0,
    ffnActivation: "gelu",
    normType: "layernorm",
    posEnc: "learned",
    tieEmbeddings: false,
  };
}

/** Collect params into the checkpoint param record shape the engine consumes. */
function paramRecord(params: ReturnType<typeof initGPT>): Record<string, { shape: number[]; data: Float32Array }> {
  const rec: Record<string, { shape: number[]; data: Float32Array }> = {};
  for (const [name, v] of collectParamEntries(params)) {
    rec[name] = { shape: [...v.data.shape], data: v.data.data as Float32Array };
  }
  return rec;
}

/** Autograd cpu_ref forward → logits for position `t` of batch 0. */
function refLogitsAt(config: ModelConfig, params: ReturnType<typeof initGPT>, toks: number[], t: number): Float32Array {
  const T = toks.length;
  const tokens: TensorData = { shape: [1, T], dtype: "i32", data: new Int32Array(toks) };
  const tape = new Tape();
  const res = gptForward(config, params, B, tape, tokens, undefined, /*training*/ false);
  const all = res.logits.data.data as Float32Array;
  const V = config.vocabSize;
  return all.slice(t * V, (t + 1) * V);
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function argmax(a: Float32Array): number {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

// Two independent f32 matmul implementations (cpu_ref vs tiled/matvec) → small
// accumulation-order divergence only. 5e-3 is generous; argmax must agree.
const TOL = 5e-3;

function runParity(name: string, config: ModelConfig) {
  describe(`inference parity — ${name}`, () => {
    const params = initGPT(config, B, new SeededRng(20260722));
    const rec = paramRecord(params);
    const toks = [3, 1, 4, 1, 5, 9, 2, 6].map((x) => x % config.vocabSize);

    it("prefill last-position logits match cpu_ref", () => {
      const weights = prepareInferenceWeights(config, rec);
      const session = createSession(weights);
      const fast = prefill(weights, session, new Int32Array(toks));
      const ref = refLogitsAt(config, params, toks, toks.length - 1);
      expect(maxAbsDiff(fast, ref)).toBeLessThan(TOL);
      expect(argmax(fast)).toBe(argmax(ref));
    });

    it("KV-cache decode (prefill T-1 + decodeStep) matches cpu_ref", () => {
      const weights = prepareInferenceWeights(config, rec);
      const session = createSession(weights);
      const prefixLen = toks.length - 1;
      prefill(weights, session, new Int32Array(toks.slice(0, prefixLen)));
      // decode the final token at absolute position prefixLen
      const fast = decodeStep(weights, session, toks[prefixLen], prefixLen);
      const ref = refLogitsAt(config, params, toks, toks.length - 1);
      expect(maxAbsDiff(fast, ref)).toBeLessThan(TOL);
      expect(argmax(fast)).toBe(argmax(ref));
    });

    it("prefill matches cpu_ref at EVERY position (multi-step consistency)", () => {
      const weights = prepareInferenceWeights(config, rec);
      const session = createSession(weights);
      // Decode autoregressively from a 1-token prefill; compare each step's
      // logits to the cpu_ref forward over the growing prefix.
      prefill(weights, session, new Int32Array([toks[0]]));
      for (let t = 1; t < toks.length; t++) {
        const fast = decodeStep(weights, session, toks[t], t);
        const ref = refLogitsAt(config, params, toks.slice(0, t + 1), t);
        expect(maxAbsDiff(fast, ref), `pos ${t}`).toBeLessThan(TOL);
        expect(argmax(fast), `pos ${t} argmax`).toBe(argmax(ref));
      }
    });
  });
}

runParity("llama-form (rope/rmsnorm/swiglu/tied)", llamaConfig(64));
runParity("gpt2-form (learned/layernorm/gelu/untied)", gpt2Config(64));
