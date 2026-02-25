/**
 * Core types for the alpha system.
 */

// ── Dtype ──────────────────────────────────────────────────────────────────
export type Dtype = "f32" | "f64" | "i32" | "f16";

/** Array types used for standard f32/f64/i32 tensor data. */
export type NumericArray = Float32Array | Float64Array | Int32Array;
/** All array types including f16 (Uint16Array for raw f16 bits). */
export type TensorArray = NumericArray | Uint16Array;

export function dtypeBytes(d: Dtype): number {
  switch (d) {
    case "f16": return 2;
    case "f32": return 4;
    case "f64": return 8;
    case "i32": return 4;
  }
}

export function dtypeArray(d: Dtype) {
  switch (d) {
    case "f16": return Uint16Array;  // f16 stored as raw u16 bits on CPU side
    case "f32": return Float32Array;
    case "f64": return Float64Array;
    case "i32": return Int32Array;
  }
}

// ── Shape helpers ──────────────────────────────────────────────────────────
export type Shape = readonly number[];

export function shapeSize(shape: Shape): number {
  let s = 1;
  for (const d of shape) s *= d;
  return s;
}

export function shapeStrides(shape: Shape): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// ── Model config ───────────────────────────────────────────────────────────
export interface ModelConfig {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly nLayer: number;
  readonly nEmbd: number;
  readonly nHead: number;
  readonly dropout: number;
}

export const defaultModelConfig: ModelConfig = {
  vocabSize: 256,
  blockSize: 256,
  nLayer: 6,
  nEmbd: 256,
  nHead: 8,
  dropout: 0.0,
};

// ── Training config ────────────────────────────────────────────────────────
export interface TrainConfig {
  readonly iters: number;
  readonly batchSize: number;
  readonly lr: number;
  readonly lrMin: number;
  readonly warmupIters: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly gradClip: number;
  readonly evalInterval: number;
  readonly evalIters: number;
  readonly seed: number;
  readonly backend: string;
  readonly tokenizer: string;
  readonly optimizer: string;
  readonly logLevel: "debug" | "info" | "warn" | "error";
  readonly trace: boolean;
  readonly gradAccumSteps: number;
  readonly sampleInterval: number;
  readonly spikeThreshold: number;
  /** GPU sync cadence: sync every N steps (0 = adaptive based on pool pressure) */
  readonly syncEvery: number;
  /** GC cadence: run gc() every N steps (0 = disabled, requires --expose-gc) */
  readonly gcEvery: number;
}

export const defaultTrainConfig: TrainConfig = {
  iters: 1000,
  batchSize: 64,
  lr: 3e-4,
  lrMin: 0,
  warmupIters: 0,
  beta1: 0.9,
  beta2: 0.95,
  eps: 1e-8,
  weightDecay: 0.1,
  gradClip: 1.0,
  evalInterval: 100,
  evalIters: 10,
  seed: 42,
  backend: "cpu_ref",
  tokenizer: "bpe",
  optimizer: "adamw",
  logLevel: "info",
  trace: false,
  gradAccumSteps: 1,
  sampleInterval: 100,
  spikeThreshold: 0,
  syncEvery: 1,
  gcEvery: 0,
};

// ── Sampling config ────────────────────────────────────────────────────────
export interface SampleConfig {
  readonly steps: number;
  readonly temperature: number;
  readonly topk: number;
}

export const defaultSampleConfig: SampleConfig = {
  steps: 200,
  temperature: 0.8,
  topk: 40,
};
