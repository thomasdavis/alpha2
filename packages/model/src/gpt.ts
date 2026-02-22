/**
 * GPT decoder-only transformer model.
 *
 * Architecture follows GPT-2 with minor tweaks (matching microgpt.py):
 * - Token embedding + positional embedding
 * - N transformer blocks: LN → causal self-attn → residual, LN → MLP (GELU) → residual
 * - Final LN → linear head
 */
import type { ModelConfig, Backend, TensorData } from "@alpha/core";
import { shapeSize, SeededRng } from "@alpha/core";
import {
  Variable, Tape,
  add, matmul, gelu, layerNorm, softmax, crossEntropy,
  reshape, transpose, embedding, scale,
} from "@alpha/autograd";

// ── Parameter initialization ───────────────────────────────────────────────

export interface GPTParams {
  /** Token embeddings [vocabSize, nEmbd] */
  wte: Variable;
  /** Position embeddings [blockSize, nEmbd] */
  wpe: Variable;
  /** Transformer layers */
  layers: LayerParams[];
  /** Final layer norm */
  lnF: { weight: Variable; bias: Variable };
  /** Language model head [vocabSize, nEmbd] */
  lmHead: Variable;
}

export interface LayerParams {
  ln1: { weight: Variable; bias: Variable };
  attn: {
    wq: Variable; wk: Variable; wv: Variable; wo: Variable;
  };
  ln2: { weight: Variable; bias: Variable };
  mlp: { fc1: Variable; fc2: Variable };
}

function initWeight(backend: Backend, rng: SeededRng, shape: number[], std: number): Variable {
  const size = shapeSize(shape);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = rng.nextGauss() * std;
  const td: TensorData = { shape, dtype: "f32", data };
  return new Variable(td, true);
}

function initOnes(backend: Backend, shape: number[]): Variable {
  return new Variable(backend.ones(shape, "f32"), true);
}

function initZeros(backend: Backend, shape: number[]): Variable {
  return new Variable(backend.zeros(shape, "f32"), true);
}

export function initGPT(config: ModelConfig, backend: Backend, rng: SeededRng): GPTParams {
  const { vocabSize, blockSize, nLayer, nEmbd, nHead } = config;
  const std = 0.02;

  const wte = initWeight(backend, rng, [vocabSize, nEmbd], std);
  const wpe = initWeight(backend, rng, [blockSize, nEmbd], std);

  const layers: LayerParams[] = [];
  for (let i = 0; i < nLayer; i++) {
    layers.push({
      ln1: { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) },
      attn: {
        wq: initWeight(backend, rng, [nEmbd, nEmbd], std),
        wk: initWeight(backend, rng, [nEmbd, nEmbd], std),
        wv: initWeight(backend, rng, [nEmbd, nEmbd], std),
        wo: initWeight(backend, rng, [nEmbd, nEmbd], std / Math.sqrt(2 * nLayer)),
      },
      ln2: { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) },
      mlp: {
        fc1: initWeight(backend, rng, [4 * nEmbd, nEmbd], std),
        fc2: initWeight(backend, rng, [nEmbd, 4 * nEmbd], std / Math.sqrt(2 * nLayer)),
      },
    });
  }

  const lnF = { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) };
  const lmHead = initWeight(backend, rng, [vocabSize, nEmbd], std);

  return { wte, wpe, layers, lnF, lmHead };
}

// ── Forward pass ───────────────────────────────────────────────────────────

export interface GPTForwardResult {
  logits: Variable;
  loss?: Variable;
}

/**
 * Forward pass through the GPT model.
 *
 * @param tokens - [B, T] token indices
 * @param targets - [B, T] target indices (optional, for loss computation)
 */
export function gptForward(
  config: ModelConfig,
  params: GPTParams,
  backend: Backend,
  tape: Tape,
  tokens: TensorData,
  targets?: TensorData,
): GPTForwardResult {
  const ctx = { tape, backend };
  const { nHead, nEmbd } = config;
  const headDim = nEmbd / nHead;
  const [B, T] = tokens.shape;

  // Token + position embeddings
  const tokEmb = embedding(ctx, params.wte, tokens); // [B, T, nEmbd]

  // Create position indices [B, T]
  const posData = new Int32Array(B * T);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      posData[b * T + t] = t;
    }
  }
  const posIndices: TensorData = { shape: [B, T], dtype: "i32", data: posData };
  const posEmb = embedding(ctx, params.wpe, posIndices); // [B, T, nEmbd]

  let x = add(ctx, tokEmb, posEmb); // [B, T, nEmbd]

  // Causal mask [T, T]
  const mask = backend.causalMask(T);

  // Transformer blocks
  for (const layer of params.layers) {
    // 1) LN → Attention → Residual
    const ln1Out = layerNorm(ctx, x, layer.ln1.weight, layer.ln1.bias, 1e-5);

    // Q, K, V projections: [B, T, nEmbd] @ [nEmbd, nEmbd]^T
    const q3d = reshape(ctx, ln1Out, [B * T, nEmbd]);
    const q = reshape(ctx, matmul(ctx, q3d, transpose(ctx, layer.attn.wq, 0, 1)), [B, T, nEmbd]);
    const k = reshape(ctx, matmul(ctx, q3d, transpose(ctx, layer.attn.wk, 0, 1)), [B, T, nEmbd]);
    const v = reshape(ctx, matmul(ctx, q3d, transpose(ctx, layer.attn.wv, 0, 1)), [B, T, nEmbd]);

    // Reshape to [B, nHead, T, headDim]
    const qH = transpose(ctx, reshape(ctx, q, [B, T, nHead, headDim]), 1, 2);
    const kH = transpose(ctx, reshape(ctx, k, [B, T, nHead, headDim]), 1, 2);
    const vH = transpose(ctx, reshape(ctx, v, [B, T, nHead, headDim]), 1, 2);

    // Attention scores: [B, nHead, T, T]
    const kT = transpose(ctx, kH, 2, 3); // [B, nHead, headDim, T]
    const scores = scale(ctx, matmul(ctx, qH, kT), 1 / Math.sqrt(headDim));

    // Apply causal mask
    const maskedScores = new Variable(
      backend.maskedFill(scores.data, mask, -Infinity),
      true,
    );
    // Record identity op for gradient flow
    tape.record({
      output: maskedScores,
      inputs: [scores],
      backward: (g, B) => {
        // Zero out gradients where mask was applied
        return [B.maskedFill(g, mask, 0)];
      },
    });

    const attnWeights = softmax(ctx, maskedScores, -1); // [B, nHead, T, T]
    const attnOut = matmul(ctx, attnWeights, vH); // [B, nHead, T, headDim]

    // Reshape back: [B, T, nEmbd]
    const attnConcat = reshape(ctx, transpose(ctx, attnOut, 1, 2), [B * T, nEmbd]);
    const projected = reshape(ctx, matmul(ctx, attnConcat, transpose(ctx, layer.attn.wo, 0, 1)), [B, T, nEmbd]);

    x = add(ctx, x, projected); // Residual

    // 2) LN → MLP → Residual
    const ln2Out = layerNorm(ctx, x, layer.ln2.weight, layer.ln2.bias, 1e-5);
    const flat = reshape(ctx, ln2Out, [B * T, nEmbd]);

    const h = gelu(ctx, matmul(ctx, flat, transpose(ctx, layer.mlp.fc1, 0, 1))); // [B*T, 4*nEmbd]
    const mlpOut = reshape(ctx, matmul(ctx, h, transpose(ctx, layer.mlp.fc2, 0, 1)), [B, T, nEmbd]);

    x = add(ctx, x, mlpOut); // Residual
  }

  // Final layer norm
  x = layerNorm(ctx, x, params.lnF.weight, params.lnF.bias, 1e-5);

  // Language model head: [B, T, nEmbd] → [B, T, vocabSize]
  const flat = reshape(ctx, x, [B * T, nEmbd]);
  const logits = reshape(ctx, matmul(ctx, flat, transpose(ctx, params.lmHead, 0, 1)), [B, T, config.vocabSize]);

  // Loss
  let loss: Variable | undefined;
  if (targets) {
    const targetsFlat: TensorData = { shape: [B * T], dtype: "i32", data: targets.data };
    const logitsVar = reshape(ctx, logits, [B * T, config.vocabSize]);
    loss = crossEntropy(ctx, logitsVar, targetsFlat);
  }

  return { logits, loss };
}

// ── Parameter collection helpers ───────────────────────────────────────────

export function collectParams(params: GPTParams): Map<string, Variable> {
  const map = new Map<string, Variable>();
  map.set("wte", params.wte);
  map.set("wpe", params.wpe);
  map.set("lmHead", params.lmHead);
  map.set("lnF.weight", params.lnF.weight);
  map.set("lnF.bias", params.lnF.bias);
  for (let i = 0; i < params.layers.length; i++) {
    const l = params.layers[i];
    map.set(`layer.${i}.ln1.weight`, l.ln1.weight);
    map.set(`layer.${i}.ln1.bias`, l.ln1.bias);
    map.set(`layer.${i}.attn.wq`, l.attn.wq);
    map.set(`layer.${i}.attn.wk`, l.attn.wk);
    map.set(`layer.${i}.attn.wv`, l.attn.wv);
    map.set(`layer.${i}.attn.wo`, l.attn.wo);
    map.set(`layer.${i}.ln2.weight`, l.ln2.weight);
    map.set(`layer.${i}.ln2.bias`, l.ln2.bias);
    map.set(`layer.${i}.mlp.fc1`, l.mlp.fc1);
    map.set(`layer.${i}.mlp.fc2`, l.mlp.fc2);
  }
  return map;
}

export function countParams(params: GPTParams): number {
  let total = 0;
  for (const [, v] of collectParams(params)) {
    total += shapeSize(v.data.shape);
  }
  return total;
}
