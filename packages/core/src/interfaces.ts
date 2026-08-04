/**
 * Subsystem interfaces (ports). Every subsystem implements one of these.
 */
import { Context, Effect } from "effect";
import type { TokenizerError, BackendError, OptimizerError, CheckpointError } from "./errors.js";
import type { Dtype, Shape, ModelConfig, TensorArray } from "./types.js";

// ── Tokenizer ──────────────────────────────────────────────────────────────
export interface TokenizerArtifacts {
  readonly type: string;
  readonly vocabSize: number;
  readonly vocab: readonly string[];
  readonly merges?: readonly [number, number][];
  readonly specialTokens?: readonly string[];
  /**
   * Byte-level BPE marker (schema v2). When true, `vocab` entries are
   * GPT-2 surrogate-mapped byte strings and `decode()` maps them back through
   * the byte↔unicode table for lossless UTF-8 reconstruction.
   */
  readonly byteVocab?: boolean;
}

export interface Tokenizer {
  readonly name: string;
  build(input: string): Effect.Effect<TokenizerArtifacts, TokenizerError>;
  encode(text: string): Int32Array;
  decode(tokens: ArrayLike<number>): string;
  readonly vocabSize: number;
}

export class TokenizerService extends Context.Tag("TokenizerService")<
  TokenizerService,
  Tokenizer
>() {}

// ── Tensor (lightweight handle) ────────────────────────────────────────────
export interface TensorData {
  readonly shape: Shape;
  readonly dtype: Dtype;
  readonly data: TensorArray;
}

/** Tiny audit summary retained by the masked-unlikelihood primitive. It is
 * derived from the N per-row loss outputs, never by reading the N*C model-logit
 * tensor back to the host. */
export interface UnlikelihoodLossStats {
  readonly activeRows: number;
  readonly maskMass: number;
  readonly meanBadProbability: number;
  readonly maxBadProbability: number;
}

// ── Backend ────────────────────────────────────────────────────────────────
export interface Backend {
  readonly name: string;

  // creation
  zeros(shape: Shape, dtype?: Dtype): TensorData;
  ones(shape: Shape, dtype?: Dtype): TensorData;
  full(shape: Shape, value: number, dtype?: Dtype): TensorData;
  randn(shape: Shape, dtype?: Dtype): TensorData;
  fromArray(data: number[], shape: Shape, dtype?: Dtype): TensorData;

  // math
  add(a: TensorData, b: TensorData): TensorData;
  sub(a: TensorData, b: TensorData): TensorData;
  mul(a: TensorData, b: TensorData): TensorData;
  div(a: TensorData, b: TensorData): TensorData;
  matmul(a: TensorData, b: TensorData): TensorData;
  sum(a: TensorData, axis?: number, keepdims?: boolean): TensorData;
  mean(a: TensorData, axis?: number, keepdims?: boolean): TensorData;

  // element-wise
  neg(a: TensorData): TensorData;
  exp(a: TensorData): TensorData;
  log(a: TensorData): TensorData;
  sqrt(a: TensorData): TensorData;
  pow(a: TensorData, exp: number): TensorData;
  scale(a: TensorData, s: number): TensorData;
  clamp(a: TensorData, lo: number, hi: number): TensorData;

  // nn
  embedding(weight: TensorData, indices: TensorData): TensorData;
  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData;
  /** RMS normalization over the last dim: (x / sqrt(mean(x^2) + eps)) * weight.
   *  Mirrors layerNorm's signature style but has no mean-subtraction and no bias
   *  (the Llama normalization variant). */
  rmsNorm(x: TensorData, weight: TensorData, eps: number): TensorData;
  gelu(a: TensorData): TensorData;
  relu(a: TensorData): TensorData;
  silu(a: TensorData): TensorData;
  softmax(a: TensorData, axis?: number): TensorData;
  logSoftmax(a: TensorData, axis?: number): TensorData;
  crossEntropy(logits: TensorData, targets: TensorData): TensorData;
  /**
   * Optional training-only classifier fusion. Returns the ordinary mean
   * cross-entropy and its unscaled derivative with respect to logits in one
   * backend pass. Autograd applies any later upstream scalar. A backend may
   * return null for an unsupported shape; callers then use the ordinary
   * forward/backward hooks. Evaluation must use crossEntropy() so it does not
   * calculate or retain an unused N*C gradient.
   */
  crossEntropyForwardBackward?(
    logits: TensorData,
    targets: TensorData,
  ): { loss: TensorData; gradLogits: TensorData } | null;
  /** Masked (assistant-only SFT) cross-entropy. Optional — backends that don't
   *  implement it are driven through the cpu_ref path in the autograd op.
   *  logits [N,C], targets [N] class idx, mask [N] f32 per-row weights.
   *  Returns scalar = sum_i(ce_i * mask_i) / max(sum_i mask_i, 1). An all-zero
   *  mask yields exactly 0 (denominator floored at 1, no NaN). */
  crossEntropyMasked?(logits: TensorData, targets: TensorData, mask: TensorData): TensorData;
  /** Training-only masked classifier fusion, analogous to
   *  crossEntropyForwardBackward. gradLogits already includes mask[row] and
   *  division by max(sum(mask),1), but not a later upstream scalar. */
  crossEntropyMaskedForwardBackward?(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
  ): { loss: TensorData; gradLogits: TensorData } | null;
  /** Masked token-unlikelihood loss for rollout-conditioned negative targets.
   *  logits [N,C], targets [N] undesired class idx, mask [N] f32 per-row
   *  weights. Returns sum_i(-log(max(1-p(target_i), epsilon)) * mask_i) /
   *  max(sum_i mask_i, 1). An all-zero mask yields exactly 0. */
  crossEntropyUnlikelihoodMasked?(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
    epsilon: number,
  ): TensorData;
  /** Statistics from the most recent masked-unlikelihood forward call. */
  getLastCrossEntropyUnlikelihoodMaskedStats?(): UnlikelihoodLossStats | null;

  // reshape / slice
  reshape(a: TensorData, shape: Shape): TensorData;
  transpose(a: TensorData, dim0: number, dim1: number): TensorData;
  slice(a: TensorData, starts: number[], ends: number[]): TensorData;
  cat(tensors: TensorData[], axis: number): TensorData;

  // utility
  argmax(a: TensorData, axis?: number): TensorData;
  topk(a: TensorData, k: number, axis?: number): { values: TensorData; indices: TensorData };
  gather(a: TensorData, axis: number, indices: TensorData): TensorData;
  clone(a: TensorData): TensorData;

  // comparison
  equal(a: TensorData, b: TensorData): boolean;
  allClose(a: TensorData, b: TensorData, atol?: number, rtol?: number): boolean;

  // mask
  causalMask(size: number): TensorData;
  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData;

  // backward (GPU-optimized, optional)
  geluBackward?(input: TensorData, gradOutput: TensorData): TensorData;
  reluBackward?(input: TensorData, gradOutput: TensorData): TensorData;
  siluBackward?(input: TensorData, gradOutput: TensorData): TensorData;
  siluMul?(a: TensorData, b: TensorData): TensorData;
  siluMulBackward?(a: TensorData, b: TensorData, gradOutput: TensorData): TensorData[];
  clampBackward?(input: TensorData, gradOutput: TensorData, lo: number, hi: number): TensorData;
  layerNormBackward?(x: TensorData, weight: TensorData, gradOutput: TensorData, eps: number): { dx: TensorData; dw: TensorData; db: TensorData };
  /** Optional fused RMSNorm backward hook (mirrors layerNormBackward but with no
   *  bias gradient). Returns dx (input grad) and dw (weight grad). When absent,
   *  the autograd op falls back to a CPU loop like layerNorm's. */
  rmsNormBackward?(x: TensorData, weight: TensorData, gradOutput: TensorData, eps: number): { dx: TensorData; dw: TensorData };
  crossEntropyBackward?(logits: TensorData, targets: TensorData, gradOutput: TensorData): TensorData;
  /** Optional fused masked-CE backward hook (mirrors crossEntropyBackward but
   *  weights each row's grad by mask[row] and divides by max(sum(mask),1) rather
   *  than N). dLogits[i,c] = (softmax(logits)[i,c] - 1{c==target_i}) * mask_i *
   *  gradOutput / max(sum(mask),1). Rows with mask 0 get exactly-zero grad. */
  crossEntropyMaskedBackward?(logits: TensorData, targets: TensorData, mask: TensorData, gradOutput: TensorData): TensorData;
  /** Optional fused masked-unlikelihood backward hook. For a masked row i,
   *  dLogits[i,c] = p_bad/max(1-p_bad,epsilon) *
   *  (1{c==target_i}-p_c) * mask_i * gradOutput / max(sum(mask),1).
   *  Rows with mask 0 get exactly-zero grad. */
  crossEntropyUnlikelihoodMaskedBackward?(
    logits: TensorData,
    targets: TensorData,
    mask: TensorData,
    gradOutput: TensorData,
    epsilon: number,
  ): TensorData;
  embeddingBackward?(indices: TensorData, gradOutput: TensorData, vocabSize: number): TensorData;
  softCap?(input: TensorData, cap: number): TensorData;
  softCapBackward?(gradOutput: TensorData, input: TensorData, cap: number): TensorData;

  // fused ops (GPU-optimized, optional)
  residualDropoutAdd?(residual: TensorData, projected: TensorData, mask: TensorData): TensorData;
  matmulTransposed?(a: TensorData, b: TensorData): TensorData;
  matmulTransposedA?(a: TensorData, b: TensorData): TensorData;
  addInplace?(a: TensorData, b: TensorData): void;
  scaleInplace?(a: TensorData, scalar: number): void;

  // fused 3-way column slice: [rows, 3*D] → 3×[rows, D] in single dispatch
  sliceQkv?(a: TensorData): [TensorData, TensorData, TensorData];

  // scatter-slice backward (GPU-optimized, optional)
  // Writes grad into a zeroed output at the 2D slice position [starts, ends) within origShape.
  scatterSlice?(grad: TensorData, origShape: Shape, starts: number[], ends: number[]): TensorData;

  // GPU dropout mask generation (GPU-optimized, optional)
  // Generates deterministic mask using splitmix32 hash (same as DropoutRng on CPU).
  dropoutMask?(shape: Shape, seed: number, counter: number, p: number): TensorData;

  // flash attention (GPU-optimized, optional)
  // Q,K,V: [B*H, T, D], returns { output: [B*H, T, D], lse: [B*H, T] }
  flashAttention?(Q: TensorData, K: TensorData, V: TensorData,
    T: number, scale: number, softCap: number): { output: TensorData; lse: TensorData };
  // dO: [B*H, T, D], O: forward output, lse: from forward
  // returns { dQ, dK, dV } each [B*H, T, D]
  flashAttentionBackward?(Q: TensorData, K: TensorData, V: TensorData,
    O: TensorData, dO: TensorData, lse: TensorData,
    T: number, scale: number, softCap: number): { dQ: TensorData; dK: TensorData; dV: TensorData };

  // rotary position embedding (optional) — applies HF-Llama rotate_half rotation.
  // x: [B*H, T, D] (head-major); cos/sin: [T, D/2] precomputed per position.
  // Rotates the pair (x[..,i], x[..,i+D/2]) by the angle whose cos/sin are given.
  // Backward is the same op with sin negated (rotation is orthogonal).
  rope?(x: TensorData, cos: TensorData, sin: TensorData): TensorData;

  // broadcast (GPU-optimized, optional) — avoids CPU readback for tiling operations
  broadcast?(a: TensorData, targetShape: Shape): TensorData;

  // dtype casting (GPU-optimized, optional)
  castDtype?(a: TensorData, dtype: Dtype): TensorData;

  // reduction (GPU-optimized, optional)
  sumOfSquares?(data: TensorData): TensorData;
  totalSumOfSquares?(tensors: TensorData[]): TensorData;
  checkFinite?(data: TensorData): TensorData;

  // optimizer (GPU-optimized, optional)
  adamwStep?(params: TensorData, grads: TensorData, m: TensorData, v: TensorData,
    lr: number, beta1: number, beta2: number, eps: number, weightDecay: number, bc1: number, bc2: number, gradScale?: number): void;
}

export class BackendService extends Context.Tag("BackendService")<
  BackendService,
  Backend
>() {}

// ── Optimizer ──────────────────────────────────────────────────────────────
export interface OptimizerState {
  readonly step: number;
  readonly buffers: Map<string, TensorData>;
}

export interface Optimizer {
  readonly name: string;
  step(params: Map<string, TensorData>, grads: Map<string, TensorData>, gradScale?: number): void;
  stateDict(): OptimizerState;
  loadStateDict(state: OptimizerState): void;
}

export class OptimizerService extends Context.Tag("OptimizerService")<
  OptimizerService,
  Optimizer
>() {}

// ── Checkpoint ─────────────────────────────────────────────────────────────
export interface CheckpointState {
  readonly modelConfig: ModelConfig;
  readonly params: Record<string, { shape: number[]; data: number[] }>;
  readonly optimizerState: OptimizerState;
  readonly tokenizerArtifacts?: TokenizerArtifacts;
  readonly rngState: number;
  readonly configHash: string;
  readonly step: number;
}

export interface Checkpoint {
  save(path: string, state: CheckpointState): Effect.Effect<void, CheckpointError>;
  load(path: string): Effect.Effect<CheckpointState, CheckpointError>;
}

export class CheckpointService extends Context.Tag("CheckpointService")<
  CheckpointService,
  Checkpoint
>() {}

// ── RNG ────────────────────────────────────────────────────────────────────
export interface Rng {
  next(): number;
  nextGauss(): number;
  state(): number;
  setState(s: number): void;
  seed(s: number): void;
}

export class RngService extends Context.Tag("RngService")<
  RngService,
  Rng
>() {}
