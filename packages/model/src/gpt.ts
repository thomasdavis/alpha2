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
  add, mul, matmul, matmulTransposed, matmulTransposedGelu, gelu, silu, relu, layerNorm, softmax, crossEntropy,
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
  mlp: {
    /** Standard MLP: fc1 [ffnDim, nEmbd], fc2 [nEmbd, ffnDim] */
    fc1: Variable; fc2: Variable;
    /** SwiGLU: gate [ffnDim, nEmbd], up [ffnDim, nEmbd], proj [nEmbd, ffnDim] */
    fc_gate?: Variable; fc_up?: Variable; fc_proj?: Variable;
    /** Universal Approximator: learnable gating — f(x) = silu(x)*gate + x*skip */
    act_gate?: Variable; act_skip?: Variable;
    /** KAN Spline: learnable basis coefficients — f(x) = c0*silu(x) + c1*relu(x) + c2*gelu(x) + c3*x + c4*x^2 */
    kan_c0?: Variable; kan_c1?: Variable; kan_c2?: Variable; kan_c3?: Variable; kan_c4?: Variable;
  };
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

function initFull(backend: Backend, shape: number[], value: number): Variable {
  return new Variable(backend.full(shape, value, "f32"), true);
}

export function initGPT(config: ModelConfig, backend: Backend, rng: SeededRng): GPTParams {
  const { vocabSize, blockSize, nLayer, nEmbd, nHead } = config;
  const activation = config.ffnActivation ?? "gelu";
  const std = 0.02;

  const wte = initWeight(backend, rng, [vocabSize, nEmbd], std);
  const wpe = initWeight(backend, rng, [blockSize, nEmbd], std);

  // FFN hidden dim: SwiGLU uses (8/3)*nEmbd rounded to multiple of 64 for parameter parity,
  // standard activations use 4*nEmbd. Config override takes precedence.
  const ffnDim = config.ffnDim ?? (activation === "swiglu"
    ? Math.ceil((8 / 3) * nEmbd / 64) * 64
    : 4 * nEmbd);

  const layers: LayerParams[] = [];
  for (let i = 0; i < nLayer; i++) {
    let mlp: LayerParams["mlp"];
    if (activation === "swiglu") {
      mlp = {
        // SwiGLU: 3 weight matrices (gate, up, proj)
        fc_gate: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc_up: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc_proj: initWeight(backend, rng, [nEmbd, ffnDim], std / Math.sqrt(2 * nLayer)),
        // Satisfy interface — alias to gate/proj so collectParams doesn't double-count
        get fc1() { return this.fc_gate!; },
        get fc2() { return this.fc_proj!; },
      };
    } else if (activation === "universal") {
      // Universal Approximator: standard fc1/fc2 + learnable gating params
      // f(x) = silu(x) * act_gate + x * act_skip
      // Init: gate=1, skip=0 → starts as SiLU. Can learn any blend.
      mlp = {
        fc1: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc2: initWeight(backend, rng, [nEmbd, ffnDim], std / Math.sqrt(2 * nLayer)),
        act_gate: initFull(backend, [1, ffnDim], 1.0),
        act_skip: initFull(backend, [1, ffnDim], 0.0),
      };
    } else if (activation === "kan_spline") {
      // KAN Spline: standard fc1/fc2 + learnable basis function coefficients
      // f(x) = c0*silu(x) + c1*relu(x) + c2*gelu(x) + c3*x + c4*x^2
      // Init: c0=0.5, c1=0, c2=0.5, c3=0, c4=0 → starts as (silu+gelu)/2
      mlp = {
        fc1: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc2: initWeight(backend, rng, [nEmbd, ffnDim], std / Math.sqrt(2 * nLayer)),
        kan_c0: initFull(backend, [1, ffnDim], 0.5),  // silu weight
        kan_c1: initFull(backend, [1, ffnDim], 0.0),  // relu weight
        kan_c2: initFull(backend, [1, ffnDim], 0.5),  // gelu weight
        kan_c3: initFull(backend, [1, ffnDim], 0.0),  // identity weight
        kan_c4: initFull(backend, [1, ffnDim], 0.0),  // quadratic weight
      };
    } else if (activation === "composed") {
      // Composed activation: just standard fc1/fc2, the graph handles activation logic
      mlp = {
        fc1: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc2: initWeight(backend, rng, [nEmbd, ffnDim], std / Math.sqrt(2 * nLayer)),
      };
    } else {
      mlp = {
        fc1: initWeight(backend, rng, [ffnDim, nEmbd], std),
        fc2: initWeight(backend, rng, [nEmbd, ffnDim], std / Math.sqrt(2 * nLayer)),
      };
    }

    layers.push({
      ln1: { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) },
      attn: {
        wqkv: initWeight(backend, rng, [3 * nEmbd, nEmbd], std),
        wo: initWeight(backend, rng, [nEmbd, nEmbd], std / Math.sqrt(2 * nLayer)),
      },
      ln2: { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) },
      mlp,
    });
  }

  const lnF = { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) };
  const lmHead = initWeight(backend, rng, [vocabSize, nEmbd], std);

  return { wte, wpe, layers, lnF, lmHead };
}

// ── Composed activation graph evaluator ─────────────────────────────────────

/**
 * Graph node types for composed activations (mirrors ActivationNode from symbiogenesis).
 * Defined locally to avoid cross-package dependency — the graph is passed as unknown
 * from ModelConfig.activationGraph and cast here.
 */
type GraphNode =
  | { type: "basis"; op: string }
  | { type: "scale"; child: GraphNode; factor: number }
  | { type: "add"; left: GraphNode; right: GraphNode }
  | { type: "mul"; left: GraphNode; right: GraphNode };

/**
 * Recursively evaluate an activation expression tree using autograd ops.
 * Backprop works automatically through any graph structure.
 */
function evalActivationGraph(
  ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng },
  x: Variable,
  node: GraphNode,
): Variable {
  switch (node.type) {
    case "basis":
      switch (node.op) {
        case "silu": return silu(ctx, x);
        case "relu": return relu(ctx, x);
        case "gelu": return gelu(ctx, x);
        case "identity": return x;
        case "square": return mul(ctx, x, x);
        default: return gelu(ctx, x); // fallback
      }
    case "scale":
      return scale(ctx, evalActivationGraph(ctx, x, node.child), node.factor);
    case "add":
      return add(ctx, evalActivationGraph(ctx, x, node.left), evalActivationGraph(ctx, x, node.right));
    case "mul":
      return mul(ctx, evalActivationGraph(ctx, x, node.left), evalActivationGraph(ctx, x, node.right));
  }
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
  const activation = config.ffnActivation ?? "gelu";

  let mlpH: Variable;
  if (activation === "composed" && config.activationGraph) {
    // Composed activation: evaluate the expression tree using autograd ops.
    // Graph is structurally mutated by symbiogenesis — backprop works through any tree.
    const h = matmulTransposed(ctx, flat, layer.mlp.fc1);
    const h_act = evalActivationGraph(ctx, h, config.activationGraph as GraphNode);
    mlpH = matmulTransposed(ctx, h_act, layer.mlp.fc2);
  } else if (activation === "swiglu") {
    // SwiGLU: h = (silu(x @ W_gate) ⊙ (x @ W_up)) @ W_proj
    const gate = silu(ctx, matmulTransposed(ctx, flat, layer.mlp.fc_gate!));
    const up = matmulTransposed(ctx, flat, layer.mlp.fc_up!);
    mlpH = matmulTransposed(ctx, mul(ctx, gate, up), layer.mlp.fc_proj!);
  } else if (activation === "universal") {
    // Universal Approximator: f(x) = silu(x) * gate + x * skip
    // Learnable per-channel gating — can represent any blend of SiLU and identity.
    // At gate=1,skip=0 → SiLU. At gate=0,skip=1 → linear. Gradients flow to gate/skip params.
    const h = matmulTransposed(ctx, flat, layer.mlp.fc1);
    const h_silu = silu(ctx, h);
    const gated = mul(ctx, h_silu, layer.mlp.act_gate!);   // [B*T, ffnDim] * [1, ffnDim] broadcast
    const skipped = mul(ctx, h, layer.mlp.act_skip!);       // residual path
    const h_act = add(ctx, gated, skipped);
    mlpH = matmulTransposed(ctx, h_act, layer.mlp.fc2);
  } else if (activation === "kan_spline") {
    // KAN Spline: f(x) = c0*silu(x) + c1*relu(x) + c2*gelu(x) + c3*x + c4*x²
    // 5-basis universal approximator inspired by Kolmogorov-Arnold representation.
    // Each coefficient is [1, ffnDim] — per-channel learnable blend of activation bases.
    const h = matmulTransposed(ctx, flat, layer.mlp.fc1);
    const h_silu = mul(ctx, silu(ctx, h), layer.mlp.kan_c0!);
    const h_relu = mul(ctx, relu(ctx, h), layer.mlp.kan_c1!);
    const h_gelu = mul(ctx, gelu(ctx, h), layer.mlp.kan_c2!);
    const h_id = mul(ctx, h, layer.mlp.kan_c3!);
    const h_sq = mul(ctx, mul(ctx, h, h), layer.mlp.kan_c4!);  // x² basis
    const h_act = add(ctx, add(ctx, add(ctx, add(ctx, h_silu, h_relu), h_gelu), h_id), h_sq);
    mlpH = matmulTransposed(ctx, h_act, layer.mlp.fc2);
  } else if (activation === "silu") {
    mlpH = matmulTransposed(ctx, silu(ctx, matmulTransposed(ctx, flat, layer.mlp.fc1)), layer.mlp.fc2);
  } else if (activation === "relu") {
    mlpH = matmulTransposed(ctx, relu(ctx, matmulTransposed(ctx, flat, layer.mlp.fc1)), layer.mlp.fc2);
  } else {
    // GELU — preserve fused matmulTransposedGelu fast path for zero regression
    mlpH = matmulTransposed(ctx, matmulTransposedGelu(ctx, flat, layer.mlp.fc1), layer.mlp.fc2);
  }

  const mlpOut = reshape(ctx, mlpH, [Batch, T, nEmbd]);
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
  release?: (td: TensorData) => void,
): GPTForwardResult {
  const ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng; release?: (td: TensorData) => void } = { tape, backend, dropoutRng, release };
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

export type ParamEntry = readonly [string, Variable];

export function collectParamEntries(params: GPTParams): ParamEntry[] {
  const entries: ParamEntry[] = [];
  entries.push(["wte", params.wte]);
  entries.push(["wpe", params.wpe]);
  entries.push(["lmHead", params.lmHead]);
  entries.push(["lnF.weight", params.lnF.weight]);
  entries.push(["lnF.bias", params.lnF.bias]);
  for (let i = 0; i < params.layers.length; i++) {
    const l = params.layers[i];
    entries.push([`layer.${i}.ln1.weight`, l.ln1.weight]);
    entries.push([`layer.${i}.ln1.bias`, l.ln1.bias]);
    entries.push([`layer.${i}.attn.wqkv`, l.attn.wqkv]);
    entries.push([`layer.${i}.attn.wo`, l.attn.wo]);
    entries.push([`layer.${i}.ln2.weight`, l.ln2.weight]);
    entries.push([`layer.${i}.ln2.bias`, l.ln2.bias]);
    if (l.mlp.fc_gate) {
      // SwiGLU: 3 separate weight matrices
      entries.push([`layer.${i}.mlp.fc_gate`, l.mlp.fc_gate]);
      entries.push([`layer.${i}.mlp.fc_up`, l.mlp.fc_up!]);
      entries.push([`layer.${i}.mlp.fc_proj`, l.mlp.fc_proj!]);
    } else {
      entries.push([`layer.${i}.mlp.fc1`, l.mlp.fc1]);
      entries.push([`layer.${i}.mlp.fc2`, l.mlp.fc2]);
    }
    // Universal Approximator learnable params
    if (l.mlp.act_gate) entries.push([`layer.${i}.mlp.act_gate`, l.mlp.act_gate]);
    if (l.mlp.act_skip) entries.push([`layer.${i}.mlp.act_skip`, l.mlp.act_skip]);
    // KAN Spline basis coefficients
    if (l.mlp.kan_c0) entries.push([`layer.${i}.mlp.kan_c0`, l.mlp.kan_c0]);
    if (l.mlp.kan_c1) entries.push([`layer.${i}.mlp.kan_c1`, l.mlp.kan_c1]);
    if (l.mlp.kan_c2) entries.push([`layer.${i}.mlp.kan_c2`, l.mlp.kan_c2]);
    if (l.mlp.kan_c3) entries.push([`layer.${i}.mlp.kan_c3`, l.mlp.kan_c3]);
    if (l.mlp.kan_c4) entries.push([`layer.${i}.mlp.kan_c4`, l.mlp.kan_c4]);
  }
  return entries;
}

export function collectParams(params: GPTParams): Map<string, Variable> {
  return new Map(collectParamEntries(params));
}

export function countParams(params: GPTParams): number {
  let total = 0;
  for (const [, v] of collectParamEntries(params)) {
    total += shapeSize(v.data.shape);
  }
  return total;
}
