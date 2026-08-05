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
  add, mul, matmul, matmulTransposed, matmulTransposedGelu, gelu, silu, siluMulMatmulTransposedRecompute, relu, layerNorm, rmsNorm, rope, softmax, crossEntropy, crossEntropyMasked,
  crossEntropyUnlikelihoodMasked,
  sliceQkv, qkvHeadMajorRope, reshape, transpose, embedding, scale, softCap, dropout,
  residualDropoutAdd, residualDropoutAddRmsNorm, flashAttention, qkvFlashAttention,
  qkvFlashAttentionTokenMajor, checkpoint,
  castToF16, castToF32,
} from "@alpha/autograd";

// ── Config-derived architecture switches (all default to GPT-2-style) ────────
type NormLayer = { weight: Variable; bias?: Variable };

/** Apply the config's normalization: LayerNorm (weight+bias) or RMSNorm (weight only). */
function applyNorm(
  ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng },
  x: Variable,
  norm: NormLayer,
  config: ModelConfig,
  eps: number,
): Variable {
  if ((config.normType ?? "layernorm") === "rmsnorm") {
    return rmsNorm(ctx, x, norm.weight, eps);
  }
  return layerNorm(ctx, x, norm.weight, norm.bias!, eps);
}

// ── Parameter initialization ───────────────────────────────────────────────

export interface GPTParams {
  /** Token embeddings [vocabSize, nEmbd] */
  wte: Variable;
  /** Position embeddings [blockSize, nEmbd] — absent when posEnc==="rope". */
  wpe?: Variable;
  /** Transformer layers */
  layers: LayerParams[];
  /** Final norm (bias absent when normType==="rmsnorm"). */
  lnF: { weight: Variable; bias?: Variable };
  /** Language model head [vocabSize, nEmbd]. When tieEmbeddings is set this IS
   *  the `wte` Variable (same object) so grads accumulate into one param. */
  lmHead: Variable;
}

/** Explicit loss selection for specialized training branches. Ordinary model
 * training defaults to cross-entropy; RCR-UL must opt into unlikelihood and
 * provide a mask identifying only frozen negative-token positions. */
export type GPTLossObjective =
  | { kind: "cross_entropy" }
  | { kind: "unlikelihood"; epsilon: number };

export interface LayerParams {
  ln1: { weight: Variable; bias?: Variable };
  attn: {
    /** Grouped QKV weight [3*nEmbd, nEmbd] — single GEMM instead of three. */
    wqkv: Variable; wo: Variable;
  };
  ln2: { weight: Variable; bias?: Variable };
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

/*
 * THE WEIGHT GOES ON THE DEVICE, and until now it did not.
 *
 * This function took `backend` and never used it: it built a Float32Array and
 * wrapped it in a bare TensorData, so every large parameter -- every attention
 * and MLP matrix, the embedding, the head -- stayed host-only, while
 * initOnes/initZeros/initFull beside it went through the backend correctly. At
 * 105M that is 11 host tensors of 25.6M elements against 10 device tensors of
 * almost nothing, and it is invisible because the result is CORRECT: an
 * operand that is not resident gets copied in, so the model trains and simply
 * pays for it.
 *
 * What it pays: every matmul re-uploads its weight EVERY STEP, through
 * `for (i) dst[i] = src[i]` on the host. Profiling a 105M step put 42% of wall
 * clock outside the GPU and the largest chains were all this copy, reached via
 * device() from matmul and transpose. It also produced the pool churn that
 * looked like a leak -- 108 weight-sized buffers a step with nobody to own
 * them.
 *
 * The initialisation stays on the host because rng.nextGauss is a host RNG and
 * the values must be reproducible from the seed; only the RESIDENCY changes.
 */
function initWeight(backend: Backend, rng: SeededRng, shape: number[], std: number): Variable {
  const size = shapeSize(shape);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) data[i] = rng.nextGauss() * std;
  return new Variable(backend.fromArray(data, shape, "f32"), true);
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

/**
 * Return the exact number of unique trainable scalar parameters that initGPT
 * creates for a configuration. Keep planning code on this architecture-aware
 * path: tied embeddings, RoPE, and RMSNorm materially change the total.
 */
export function estimateGPTParamCount(config: ModelConfig): number {
  const { vocabSize, blockSize, nLayer, nEmbd } = config;
  const activation = config.ffnActivation ?? "gelu";
  const normType = config.normType ?? "layernorm";
  const posEnc = config.posEnc ?? "learned";
  const tieEmbeddings = config.tieEmbeddings ?? false;
  const ffnDim = config.ffnDim ?? (activation === "swiglu"
    ? Math.ceil((8 / 3) * nEmbd / 64) * 64
    : 4 * nEmbd);

  const normParams = normType === "rmsnorm" ? nEmbd : 2 * nEmbd;
  const embeddingParams = vocabSize * nEmbd;
  let total = embeddingParams;
  if (posEnc !== "rope") total += blockSize * nEmbd;
  total += normParams; // final norm
  if (!tieEmbeddings) total += embeddingParams;

  let mlpParams = activation === "swiglu"
    ? 3 * nEmbd * ffnDim
    : 2 * nEmbd * ffnDim;
  if (activation === "universal") mlpParams += 2 * ffnDim;
  else if (activation === "kan_spline") mlpParams += 5 * ffnDim;

  const attentionParams = 4 * nEmbd * nEmbd;
  const perLayerNormParams = 2 * normParams;
  total += nLayer * (attentionParams + mlpParams + perLayerNormParams);
  return total;
}

export function initGPT(config: ModelConfig, backend: Backend, rng: SeededRng): GPTParams {
  const { vocabSize, blockSize, nLayer, nEmbd, nHead } = config;
  const activation = config.ffnActivation ?? "gelu";
  const normType = config.normType ?? "layernorm";
  const posEnc = config.posEnc ?? "learned";
  const tie = config.tieEmbeddings ?? false;
  const useBias = normType !== "rmsnorm"; // RMSNorm has weight only, no bias
  const std = 0.02;

  // A norm layer: weight (ones) always; bias (zeros) only for LayerNorm.
  const initNorm = (): NormLayer =>
    useBias
      ? { weight: initOnes(backend, [nEmbd]), bias: initZeros(backend, [nEmbd]) }
      : { weight: initOnes(backend, [nEmbd]) };

  const wte = initWeight(backend, rng, [vocabSize, nEmbd], std);
  // Learned position embeddings only when posEnc==="learned"; RoPE has no wpe.
  const wpe = posEnc === "rope" ? undefined : initWeight(backend, rng, [blockSize, nEmbd], std);

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
      ln1: initNorm(),
      attn: {
        wqkv: initWeight(backend, rng, [3 * nEmbd, nEmbd], std),
        wo: initWeight(backend, rng, [nEmbd, nEmbd], std / Math.sqrt(2 * nLayer)),
      },
      ln2: initNorm(),
      mlp,
    });
  }

  const lnF = initNorm();
  // Tied embeddings: the LM head IS the token-embedding table (same Variable),
  // so its gradient accumulates into `wte` and it is stored/counted once.
  const lmHead = tie ? wte : initWeight(backend, rng, [vocabSize, nEmbd], std);

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

type TransformerBlockResult = {
  output: Variable;
  normalizedForNext?: Variable;
};

/** Single transformer block: LN → Attention → Residual, LN → MLP → Residual.
 *  A caller may provide the first normalized view from a preceding fused
 *  residual boundary and request the next normalized view as a side output. */
function transformerBlock(
  ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng },
  x: Variable,
  layer: LayerParams,
  config: ModelConfig,
  Batch: number,
  T: number,
  mask: TensorData,
  training: boolean,
  normalizedInput?: Variable,
  nextNorm?: NormLayer,
): TransformerBlockResult {
  const { nHead, nEmbd } = config;
  const headDim = nEmbd / nHead;
  const ropeOn = (config.posEnc ?? "learned") === "rope";
  const ropeTheta = config.ropeTheta ?? 10000;
  // softCap defaults to 30 for GPT-2-style, but OFF for RoPE (no Llama equivalent)
  // unless the config explicitly sets it.
  const softCapVal = config.softCap ?? (ropeOn ? undefined : 30.0);
  const useSoftCap = softCapVal !== undefined && softCapVal > 0;

  // 1) Norm → Attention → Residual
  const ln1Out = normalizedInput ?? applyNorm(ctx, x, layer.ln1, config, 1e-5);

  // Grouped QKV projection — single GEMM instead of three
  const q3d = reshape(ctx, ln1Out, [Batch * T, nEmbd]);
  const qkvFlat = matmulTransposed(ctx, q3d, layer.attn.wqkv); // [B*T, 3*nEmbd]

  // Attention: Flash Attention (fused) or standard path
  let attnConcat: Variable;
  const tokenMajorQkvFlash = ropeOn
    && !!ctx.backend.flashAttentionTokenMajor
    && !!ctx.backend.flashAttentionBackwardTokenMajor
    && !!ctx.backend.qkvHeadMajorRope
    && !!ctx.backend.qkvHeadMajorRopeBackwardCombined;
  const combinedQkvFlash = ropeOn
    && !!ctx.backend.flashAttention
    && !!ctx.backend.flashAttentionBackward
    && !!ctx.backend.qkvHeadMajorRope
    && !!ctx.backend.qkvHeadMajorRopeBackwardCombined;
  const fusedQkvLayout = ropeOn
    && !!ctx.backend.flashAttention
    && !!ctx.backend.qkvHeadMajorRope
    && !!ctx.backend.qkvHeadMajorRopeBackward;
  if (tokenMajorQkvFlash) {
    // The Flash kernel writes [B*T,H*D] in the output projection's native
    // token-major order and consumes O/dO in the same layout during backward.
    // This removes both whole-tensor output transposes without changing Q/K/V.
    attnConcat = qkvFlashAttentionTokenMajor(
      ctx,
      qkvFlat,
      Batch,
      T,
      nHead,
      headDim,
      ropeTheta,
      1 / Math.sqrt(headDim),
      useSoftCap ? softCapVal! : 0,
    );
  } else if (combinedQkvFlash) {
    // Record the layout/RoPE boundary and flash attention as one autograd
    // operation. Its backward consumes dQ/dK/dV together and writes the
    // complete grouped QKV gradient once, avoiding three zero-padded branch
    // tensors plus two full-width tape additions.
    const attnOut = qkvFlashAttention(
      ctx,
      qkvFlat,
      Batch,
      T,
      nHead,
      headDim,
      ropeTheta,
      1 / Math.sqrt(headDim),
      useSoftCap ? softCapVal! : 0,
    );
    attnConcat = reshape(ctx, transpose(ctx, reshape(ctx, attnOut, [Batch, nHead, T, headDim]), 1, 2), [Batch * T, nEmbd]);
  } else if (fusedQkvLayout) {
    // The QKV projection is token-major [B*T,3*H*D], whereas flash attention
    // consumes head-major [B*H,T,D]. Cross that boundary once and rotate Q/K
    // in the same dispatch instead of materialising slice, transpose, and RoPE
    // intermediates. The backend retains a matched compositional control.
    const [qFA, kFA, vFA] = qkvHeadMajorRope(
      ctx,
      qkvFlat,
      Batch,
      T,
      nHead,
      headDim,
      ropeTheta,
    );
    const attnOut = flashAttention(ctx, qFA, kFA, vFA, T, 1 / Math.sqrt(headDim), useSoftCap ? softCapVal! : 0);
    attnConcat = reshape(ctx, transpose(ctx, reshape(ctx, attnOut, [Batch, nHead, T, headDim]), 1, 2), [Batch * T, nEmbd]);
  } else {
    const [qFlat, kFlat, vFlat] = sliceQkv(ctx, qkvFlat); // fused 3-way slice
    const q = reshape(ctx, qFlat, [Batch, T, nEmbd]);
    const k = reshape(ctx, kFlat, [Batch, T, nEmbd]);
    const v = reshape(ctx, vFlat, [Batch, T, nEmbd]);

    if (ctx.backend.flashAttention) {
      // Flash attention path: causal masking + softcap are handled inside the kernel.
      // q/k/v are [B, T, nEmbd] with memory layout [B][T][nHead][headDim]. The flash
      // kernel indexes contiguous [B*nHead, T, headDim] as head-major — a contiguous
      // [T, headDim] block per (batch, head). A PLAIN reshape to [B*nHead, T, headDim]
      // would reinterpret the [B][T][nHead][headDim] buffer without moving data, so
      // for nHead>1 the (batch,head) rows and time positions are scrambled (and RoPE
      // would then rotate by wrong positions). Reshape→transpose(1,2)→reshape lays
      // the data out head-major, matching the standard path exactly. [defect P1]
      let qFA = reshape(ctx, transpose(ctx, reshape(ctx, q, [Batch, T, nHead, headDim]), 1, 2), [Batch * nHead, T, headDim]);
      let kFA = reshape(ctx, transpose(ctx, reshape(ctx, k, [Batch, T, nHead, headDim]), 1, 2), [Batch * nHead, T, headDim]);
      const vFA = reshape(ctx, transpose(ctx, reshape(ctx, v, [Batch, T, nHead, headDim]), 1, 2), [Batch * nHead, T, headDim]);
      // RoPE rotates q/k (per-head, per-position) before attention; flash kernel unchanged.
      if (ropeOn) {
        qFA = rope(ctx, qFA, headDim, 0, ropeTheta);
        kFA = rope(ctx, kFA, headDim, 0, ropeTheta);
      }
      const attnOut = flashAttention(ctx, qFA, kFA, vFA, T, 1 / Math.sqrt(headDim), useSoftCap ? softCapVal! : 0);
      attnConcat = reshape(ctx, transpose(ctx, reshape(ctx, attnOut, [Batch, nHead, T, headDim]), 1, 2), [Batch * T, nEmbd]);
    } else {
      // Standard multi-dispatch attention (CPU fallback)
      let qH = transpose(ctx, reshape(ctx, q, [Batch, T, nHead, headDim]), 1, 2);
      let kH = transpose(ctx, reshape(ctx, k, [Batch, T, nHead, headDim]), 1, 2);
      const vH = transpose(ctx, reshape(ctx, v, [Batch, T, nHead, headDim]), 1, 2);

      // RoPE on q/k: rotate the [B*nHead, T, headDim] head-major view (position = T
      // axis), then reshape back to [B, nHead, T, headDim] for attention.
      if (ropeOn) {
        qH = reshape(ctx, rope(ctx, reshape(ctx, qH, [Batch * nHead, T, headDim]), headDim, 0, ropeTheta), [Batch, nHead, T, headDim]);
        kH = reshape(ctx, rope(ctx, reshape(ctx, kH, [Batch * nHead, T, headDim]), headDim, 0, ropeTheta), [Batch, nHead, T, headDim]);
      }

      // Q @ K^T through the FUSED entry point, not a materialised transpose.
      //
      // m16n8k16 is `row.col` — it reads A row-major and B column-major, so a
      // transposed B is the orientation the instruction natively wants and the
      // tensor-core path stages tiles through shared memory anyway. (The fused
      // form was refuted twice for the older SCALAR kernel, where a transposed
      // B is uncoalesced; that does not carry over and should not be re-tested
      // a third time.)
      //
      // What it removes is a transpose of the whole [B,H,T,D] tensor — 3.93 MB
      // read and written per layer, plus its counterpart in the backward. The
      // fused backward is also better shaped: dL/dA = G @ B and dL/dB = G^T @ A
      // both go straight to a GEMM, where the transpose route recorded a
      // separate permute to differentiate through.
      const invSqrt = 1 / Math.sqrt(headDim);
      const qk = matmulTransposed(ctx, qH, kH);

      // Scale, causal mask and softmax are ONE kernel when the backend has it.
      //
      // Composed they are three passes over the same [B,H,T,T] tensor — 23.6 MB
      // of traffic per layer to apply one multiply and one predicated move per
      // element. The fused form returns null when the shape does not suit it
      // (T must be a power of two and a row must fit one block), and softCap
      // takes the composed path because it belongs between the scale and the
      // mask.
      // softmaxBackward is part of the gate, not just softmaxMasked: the fused
      // forward's gradient goes back through the softmax, and a backend with
      // one and not the other would fuse and then fail in the backward pass,
      // which is a long way from where the decision was made.
      // The soft cap rides INSIDE the fused kernel, folded into softCap's
      // exponent constant, so the composed chain's four passes become one.
      // softCap defaults to 30 whenever RoPE is off, which is this model — an
      // earlier version of this gated the fusion OFF when it was set and
      // therefore never ran at all.
      const smBackward = ctx.backend.softmaxBackward?.bind(ctx.backend);
      const capBackward = ctx.backend.softCapBackward?.bind(ctx.backend);
      const capVal = useSoftCap ? softCapVal! : 0;
      const fused = !smBackward || (useSoftCap && !capBackward)
        ? null
        : ctx.backend.softmaxMasked?.(qk.data, mask, invSqrt, capVal);
      let attnWeights: Variable;
      if (fused && smBackward) {
        /* Bound to a local so the closure below keeps the narrowing — an
         * optional method read inside a callback is `possibly undefined` again
         * however the caller guarded it. */
        const softmaxBack = smBackward;
        const out = new Variable(fused, true);
        ctx.tape.record({
          output: out,
          inputs: [qk],
          // The chain in reverse: through the softmax, then the mask, then the
          // scale. Still three operations, because only the FORWARD is fused —
          // the composed backward is what it always was, so this cannot be a
          // regression on that side.
          // The chain in reverse. Only the FORWARD is fused, so this is the
          // same composed backward as before minus one pass: softCap's
          // gradient takes the RAW scores, which the tape holds, because the
          // scale lives in a constant rather than in a materialised
          // intermediate. Without the cap there is nothing to differentiate
          // through and the scale is a plain multiply.
          //
          // EVERY TEMPORARY HERE IS RELEASED, and the third argument is how.
          // The composed ops in ops.ts are handed this callback and use it; a
          // hand-written backward that ignores it leaks one full [B,H,T,T]
          // tensor per operation per layer per step — 141 MB a step at this
          // shape, which exhausted the card during warmup at step 16 and
          // surfaced as "allocation of 983040 floats failed", a size rather
          // than a cause.
          backward: (g, B, release) => {
            const dSoft = softmaxBack(out.data, g);
            const dCap = B.maskedFill(dSoft, mask, 0);
            release?.(dSoft);
            if (!useSoftCap) {
              const dx = B.scale(dCap, invSqrt);
              release?.(dCap);
              return [dx];
            }
            // d/dx[cap*tanh(s*x/cap)] = s*(1 - tanh^2(s*x/cap)); the kernel's
            // own scale slot absorbs the leading s, so this takes the RAW
            // scores and no scaled intermediate has to exist.
            const dx = capBackward!(dCap, qk.data, capVal, invSqrt);
            release?.(dCap);
            return [dx];
          },
        });
        attnWeights = out;
      } else {
        const rawScores = scale(ctx, qk, invSqrt);
        const scores = useSoftCap ? softCap(ctx, rawScores, softCapVal!) : rawScores;

        const maskedScores = new Variable(
          ctx.backend.maskedFill(scores.data, mask, -1e9),
          true,
        );
        ctx.tape.record({
          output: maskedScores,
          inputs: [scores],
          backward: (g, B) => [B.maskedFill(g, mask, 0)],
        });

        attnWeights = softmax(ctx, maskedScores, -1);
      }
      const attnDrop = dropout(ctx, attnWeights, config.dropout, training);
      const attnOut = matmul(ctx, attnDrop, vH);
      attnConcat = reshape(ctx, transpose(ctx, attnOut, 1, 2), [Batch * T, nEmbd]);
    }
  }
  const projected = reshape(ctx, matmulTransposed(ctx, attnConcat, layer.attn.wo), [Batch, T, nEmbd]);
  let ln2Out: Variable;
  if ((config.normType ?? "layernorm") === "rmsnorm") {
    const fused = residualDropoutAddRmsNorm(
      ctx,
      x,
      projected,
      layer.ln2.weight,
      1e-5,
      config.dropout,
      training,
    );
    x = fused.residual;
    ln2Out = fused.normalized;
  } else {
    x = residualDropoutAdd(ctx, x, projected, config.dropout, training);
    ln2Out = applyNorm(ctx, x, layer.ln2, config, 1e-5);
  }

  // 2) Norm → MLP → Residual
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
    const gatePre = matmulTransposed(ctx, flat, layer.mlp.fc_gate!);
    const up = matmulTransposed(ctx, flat, layer.mlp.fc_up!);
    mlpH = siluMulMatmulTransposedRecompute(ctx, gatePre, up, layer.mlp.fc_proj!);
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
  if (nextNorm && (config.normType ?? "layernorm") === "rmsnorm") {
    const fused = residualDropoutAddRmsNorm(
      ctx,
      x,
      mlpOut,
      nextNorm.weight,
      1e-5,
      config.dropout,
      training,
    );
    return { output: fused.residual, normalizedForNext: fused.normalized };
  }
  return { output: residualDropoutAdd(ctx, x, mlpOut, config.dropout, training) };
}

/**
 * Forward pass through the GPT model.
 *
 * @param tokens - [B, T] token indices
 * @param targets - [B, T] target indices (optional, for loss computation)
 * @param training - whether to apply dropout (default: false)
 * @param activationCheckpointing - recompute layer intermediates during backward to save memory
 * @param mixedPrecision - store inter-layer activations as f16 to halve VRAM usage
 * @param lossMask - optional [B, T] f32 per-position loss weights (assistant-only
 *   SFT). When present, the loss is the masked mean crossEntropyMasked instead of
 *   the plain crossEntropy — no behavior change when absent (pretraining path).
 * @param lossObjective - explicit specialized loss mode. The default is ordinary
 *   cross-entropy. Unlikelihood requires `lossMask` and treats targets as tokens
 *   whose probability should be reduced at mask-positive positions.
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
  lossMask?: TensorData,
  lossObjective: GPTLossObjective = { kind: "cross_entropy" },
): GPTForwardResult {
  const ctx: { tape: Tape; backend: Backend; dropoutRng?: DropoutRng; release?: (td: TensorData) => void } = { tape, backend, dropoutRng, release };
  const { nEmbd } = config;
  const [B, T] = tokens.shape;

  // Token embeddings
  const tokEmb = embedding(ctx, params.wte, tokens); // [B, T, nEmbd]

  let x: Variable;
  if (params.wpe) {
    // Learned absolute position embeddings (GPT-2 style).
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
    x = add(ctx, tokEmb, posEmb); // [B, T, nEmbd]
  } else {
    // RoPE: no positional embedding added here; rotation is applied to q/k
    // inside each attention block.
    x = tokEmb;
  }

  // Causal mask [T, T] — cached per T since it's constant
  let mask = causalMaskCache.get(T);
  if (!mask) {
    mask = backend.causalMask(T);
    causalMaskCache.set(T, mask);
  }

  // The non-checkpointed FP32 RMSNorm path can carry the normalized side
  // output of each MLP residual directly into the next block (and the final
  // head), exposing the second set of exact residual+RMSNorm fusion boundaries.
  // Activation checkpointing currently has a single-output contract, while
  // inter-layer mixed precision intentionally rounds x through f16 before the
  // next norm; either condition therefore keeps the established path.
  const carryNormalized = (config.normType ?? "layernorm") === "rmsnorm"
    && !!backend.residualAddRmsNorm
    && !(activationCheckpointing && training)
    && !(mixedPrecision && training);
  let normalizedForNext: Variable | undefined;

  // Transformer blocks
  for (let layerIndex = 0; layerIndex < params.layers.length; layerIndex++) {
    const layer = params.layers[layerIndex];
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
        return transformerBlock(innerCtxWithRng, f32Inp, layer, config, B, T, mask, training).output;
      }, x);
    } else {
      // Cast f16 input back to f32 for compute within the block
      if (mixedPrecision && training) x = castToF32(ctx, x);
      const nextNorm = carryNormalized
        ? (layerIndex + 1 < params.layers.length ? params.layers[layerIndex + 1].ln1 : params.lnF)
        : undefined;
      const block = transformerBlock(
        ctx,
        x,
        layer,
        config,
        B,
        T,
        mask,
        training,
        normalizedForNext,
        nextNorm,
      );
      x = block.output;
      normalizedForNext = block.normalizedForNext;
    }
  }

  // Final norm (LayerNorm or RMSNorm per config)
  x = normalizedForNext ?? applyNorm(ctx, x, params.lnF, config, 1e-5);

  // Language model head: [B, T, nEmbd] → [B, T, vocabSize]
  const flat = reshape(ctx, x, [B * T, nEmbd]);
  const logits = reshape(ctx, matmulTransposed(ctx, flat, params.lmHead), [B, T, config.vocabSize]);

  // Loss
  let loss: Variable | undefined;
  if (targets) {
    const targetsFlat: TensorData = { shape: [B * T], dtype: "i32", data: targets.data };
    const logitsVar = reshape(ctx, logits, [B * T, config.vocabSize]);
    if (lossObjective.kind === "unlikelihood") {
      if (!lossMask) throw new Error("unlikelihood loss requires a per-position lossMask");
      const maskFlat: TensorData = { shape: [B * T], dtype: "f32", data: lossMask.data };
      loss = crossEntropyUnlikelihoodMasked(ctx, logitsVar, targetsFlat, maskFlat, lossObjective.epsilon);
    } else if (lossMask) {
      // Assistant-only SFT: masked mean over positions (padding + user tokens
      // carry weight 0, so they contribute neither loss nor gradient).
      const maskFlat: TensorData = { shape: [B * T], dtype: "f32", data: lossMask.data };
      loss = crossEntropyMasked(ctx, logitsVar, targetsFlat, maskFlat, training);
    } else {
      loss = crossEntropy(ctx, logitsVar, targetsFlat, training);
    }
  }

  return { logits, loss };
}

// ── Parameter collection helpers ───────────────────────────────────────────

export type ParamEntry = readonly [string, Variable];

export function collectParamEntries(params: GPTParams): ParamEntry[] {
  const entries: ParamEntry[] = [];
  entries.push(["wte", params.wte]);
  // wpe absent under RoPE.
  if (params.wpe) entries.push(["wpe", params.wpe]);
  // Tied embeddings: lmHead IS wte (same Variable) → do NOT list it twice.
  if (params.lmHead !== params.wte) entries.push(["lmHead", params.lmHead]);
  entries.push(["lnF.weight", params.lnF.weight]);
  if (params.lnF.bias) entries.push(["lnF.bias", params.lnF.bias]);
  for (let i = 0; i < params.layers.length; i++) {
    const l = params.layers[i];
    entries.push([`layer.${i}.ln1.weight`, l.ln1.weight]);
    if (l.ln1.bias) entries.push([`layer.${i}.ln1.bias`, l.ln1.bias]);
    entries.push([`layer.${i}.attn.wqkv`, l.attn.wqkv]);
    entries.push([`layer.${i}.attn.wo`, l.attn.wo]);
    entries.push([`layer.${i}.ln2.weight`, l.ln2.weight]);
    if (l.ln2.bias) entries.push([`layer.${i}.ln2.bias`, l.ln2.bias]);
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
