/**
 * Subsystem interfaces (ports). Every subsystem implements one of these.
 */
import { Context, Effect } from "effect";
import type { TokenizerError, BackendError, OptimizerError, CheckpointError } from "./errors.js";
import type { Dtype, Shape, ModelConfig } from "./types.js";

// ── Tokenizer ──────────────────────────────────────────────────────────────
export interface TokenizerArtifacts {
  readonly type: string;
  readonly vocabSize: number;
  readonly vocab: readonly string[];
  readonly merges?: readonly [number, number][];
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
  readonly data: Float32Array | Float64Array | Int32Array;
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

  // nn
  embedding(weight: TensorData, indices: TensorData): TensorData;
  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData;
  gelu(a: TensorData): TensorData;
  relu(a: TensorData): TensorData;
  silu(a: TensorData): TensorData;
  softmax(a: TensorData, axis?: number): TensorData;
  logSoftmax(a: TensorData, axis?: number): TensorData;
  crossEntropy(logits: TensorData, targets: TensorData): TensorData;

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
  layerNormBackward?(x: TensorData, weight: TensorData, gradOutput: TensorData, eps: number): { dx: TensorData; dw: TensorData; db: TensorData };

  // broadcast (GPU-optimized, optional) — avoids CPU readback for tiling operations
  broadcast?(a: TensorData, targetShape: Shape): TensorData;

  // optimizer (GPU-optimized, optional)
  adamwStep?(params: TensorData, grads: TensorData, m: TensorData, v: TensorData,
    lr: number, beta1: number, beta2: number, eps: number, weightDecay: number, bc1: number, bc2: number): void;
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
  step(params: Map<string, TensorData>, grads: Map<string, TensorData>): void;
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
