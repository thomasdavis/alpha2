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
  Variable, Tape, DropoutRng,
  add, matmul, matmulTransposed, matmulTransposedGelu, gelu, layerNorm, softmax, crossEntropy,
  slice, reshape, transpose, embedding, scale, softCap, dropout,
  residualDropoutAdd, flashAttention, checkpoint,
  castToF16, castToF32,
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
    /** Grouped QKV weight [3*nEmbd, nEmbd] — single GEMM instead of three. */
    wqkv: Variable; wo: Variable;
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
        wqkv: initWeight(backend, rng, [3 * nEmbd, nEmbd], std),
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

// ── Forward pass caches ────────────────────────────────────────────────────

/** Cache for position indices keyed by "B,T". */
const posIndicesCache = new Map<string, TensorData>();

/** Cache for causal masks keyed by T. */
const causalMaskCache = new Map<number, TensorData>();

/** Clear forward pass caches (call when changing model config or freeing memory). */
export function clearForwardCache(): void {
  posIndicesCache.clear();
  causalMaskCache.clear();
}

// ── Forward pass ───────────────────────────────────────────────────────────

export interface GPTForwardResult {
  logits: Variable;
  loss?: Variable;
  diagnostics?: {
    maxLogitMagnitude: number;
    meanLogitMagnitude: number;
  };
}

/** Single transformer block: LN → Attention → Residual, LN → MLP → Residual. */
function transformerBlock(
  ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng },
  x: Variable,
  layer: LayerParams,
  config: ModelConfig,
  Batch: number,
  T: number,
  mask: TensorData,
  training: boolean,
): Variable {
  const { nHead, nEmbd } = config;
  const headDim = nEmbd / nHead;

  // 1) LN → Attention → Residual
  const ln1Out = layerNorm(ctx, x, layer.ln1.weight, layer.ln1.bias, 1e-5);

  // Grouped QKV projection — single GEMM instead of three
  const q3d = reshape(ctx, ln1Out, [Batch * T, nEmbd]);
  const qkvFlat = matmulTransposed(ctx, q3d, layer.attn.wqkv); // [B*T, 3*nEmbd]
  const BT = Batch * T;
  const q = reshape(ctx, slice(ctx, qkvFlat, [0, 0], [BT, nEmbd]), [Batch, T, nEmbd]);
  const k = reshape(ctx, slice(ctx, qkvFlat, [0, nEmbd], [BT, 2 * nEmbd]), [Batch, T, nEmbd]);
  const v = reshape(ctx, slice(ctx, qkvFlat, [0, 2 * nEmbd], [BT, 3 * nEmbd]), [Batch, T, nEmbd]);

  // Attention: Flash Attention (fused) or standard path
  let attnConcat: Variable;
  if (ctx.backend.flashAttention) {
    // Flash attention path: causal masking + softcap are handled inside the kernel.
    // Attention-level dropout is skipped — residual dropouts (after attention output
    // and after MLP output) still provide regularization. This matches modern
    // architectures (LLaMA, Mistral) which don't use attention dropout.
    const qFA = reshape(ctx, q, [Batch * nHead, T, headDim]);
    const kFA = reshape(ctx, k, [Batch * nHead, T, headDim]);
    const vFA = reshape(ctx, v, [Batch * nHead, T, headDim]);
    const attnOut = flashAttention(ctx, qFA, kFA, vFA, T, 1 / Math.sqrt(headDim), 30);
    attnConcat = reshape(ctx, transpose(ctx, reshape(ctx, attnOut, [Batch, nHead, T, headDim]), 1, 2), [Batch * T, nEmbd]);
  } else {
    // Standard multi-dispatch attention (CPU fallback)
    const qH = transpose(ctx, reshape(ctx, q, [Batch, T, nHead, headDim]), 1, 2);
    const kH = transpose(ctx, reshape(ctx, k, [Batch, T, nHead, headDim]), 1, 2);
    const vH = transpose(ctx, reshape(ctx, v, [Batch, T, nHead, headDim]), 1, 2);

    const kT = transpose(ctx, kH, 2, 3);
    const rawScores = scale(ctx, matmul(ctx, qH, kT), 1 / Math.sqrt(headDim));
    const scores = softCap(ctx, rawScores, 30);

    const maskedScores = new Variable(
      ctx.backend.maskedFill(scores.data, mask, -1e9),
      true,
    );
    ctx.tape.record({
      output: maskedScores,
      inputs: [scores],
      backward: (g, B) => [B.maskedFill(g, mask, 0)],
    });

    const attnWeights = softmax(ctx, maskedScores, -1);
    const attnDrop = dropout(ctx, attnWeights, config.dropout, training);
    const attnOut = matmul(ctx, attnDrop, vH);
    attnConcat = reshape(ctx, transpose(ctx, attnOut, 1, 2), [Batch * T, nEmbd]);
  }
  const projected = reshape(ctx, matmulTransposed(ctx, attnConcat, layer.attn.wo), [Batch, T, nEmbd]);
  x = residualDropoutAdd(ctx, x, projected, config.dropout, training);

  // 2) LN → MLP → Residual
  const ln2Out = layerNorm(ctx, x, layer.ln2.weight, layer.ln2.bias, 1e-5);
  const flat = reshape(ctx, ln2Out, [Batch * T, nEmbd]);
  const h = matmulTransposedGelu(ctx, flat, layer.mlp.fc1);
  const mlpOut = reshape(ctx, matmulTransposed(ctx, h, layer.mlp.fc2), [Batch, T, nEmbd]);
  return residualDropoutAdd(ctx, x, mlpOut, config.dropout, training);
}

/**
 * Forward pass through the GPT model.
 *
 * @param tokens - [B, T] token indices
 * @param targets - [B, T] target indices (optional, for loss computation)
 * @param training - whether to apply dropout (default: false)
 * @param activationCheckpointing - recompute layer intermediates during backward to save memory
 * @param mixedPrecision - store inter-layer activations as f16 to halve VRAM usage
 */
export function gptForward(
  config: ModelConfig,
  params: GPTParams,
  backend: Backend,
  tape: Tape,
  tokens: TensorData,
  targets?: TensorData,
  training = false,
  activationCheckpointing = false,
  mixedPrecision = false,
  dropoutRng?: DropoutRng,
): GPTForwardResult {
  const ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng } = { tape, backend, dropoutRng };
  const { nEmbd } = config;
  const [B, T] = tokens.shape;

  // Token + position embeddings
  const tokEmb = embedding(ctx, params.wte, tokens); // [B, T, nEmbd]

  // Position indices [B, T] — cached per (B, T) since they're constant
  const posKey = `${B},${T}`;
  let posIndices = posIndicesCache.get(posKey);
  if (!posIndices) {
    const posData = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        posData[b * T + t] = t;
      }
    }
    posIndices = { shape: [B, T], dtype: "i32", data: posData };
    posIndicesCache.set(posKey, posIndices);
  }
  const posEmb = embedding(ctx, params.wpe, posIndices); // [B, T, nEmbd]

  let x = add(ctx, tokEmb, posEmb); // [B, T, nEmbd]

  // Causal mask [T, T] — cached per T since it's constant
  let mask = causalMaskCache.get(T);
  if (!mask) {
    mask = backend.causalMask(T);
    causalMaskCache.set(T, mask);
  }

  // Transformer blocks
  for (const layer of params.layers) {
    // Mixed precision: cast inter-layer activations to f16 for VRAM savings
    if (mixedPrecision && training) x = castToF16(ctx, x);

    if (activationCheckpointing && training) {
      // Save dropout RNG counter so recomputation during backward produces identical masks
      const savedCounter = dropoutRng?.saveCounter();
      x = checkpoint(ctx, (innerCtx, inp) => {
        // Restore dropout RNG counter for deterministic replay
        if (dropoutRng && savedCounter !== undefined) dropoutRng.restoreCounter(savedCounter);
        const innerCtxWithRng = { ...innerCtx, dropoutRng };
        // Cast f16 input back to f32 for compute within the block
        const f32Inp = mixedPrecision ? castToF32(innerCtxWithRng, inp) : inp;
        return transformerBlock(innerCtxWithRng, f32Inp, layer, config, B, T, mask, training);
      }, x);
    } else {
      // Cast f16 input back to f32 for compute within the block
      if (mixedPrecision && training) x = castToF32(ctx, x);
      x = transformerBlock(ctx, x, layer, config, B, T, mask, training);
    }
  }

  // Final layer norm
  x = layerNorm(ctx, x, params.lnF.weight, params.lnF.bias, 1e-5);

  // Language model head: [B, T, nEmbd] → [B, T, vocabSize]
  const flat = reshape(ctx, x, [B * T, nEmbd]);
  const logits = reshape(ctx, matmulTransposed(ctx, flat, params.lmHead), [B, T, config.vocabSize]);

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
    map.set(`layer.${i}.attn.wqkv`, l.attn.wqkv);
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
