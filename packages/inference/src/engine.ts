/**
 * Optimized pure-TS inference engine for Alpha models (GPT-2-form AND Llama-form).
 *
 * Bypasses autograd/tape machinery for 10-20× faster CPU inference via:
 * - KV cache (avoid recomputing prior tokens)
 * - Tiled matmul (cache-friendly memory access)
 * - Zero allocation in the decode loop (pre-allocated buffers)
 * - Last-token-only LM head (skip vocabSize computation for all but final position)
 * - Fused layernorm, attention scoring, in-place GELU
 *
 * Architecture-aware (mirrors packages/model gptForward / cpu_ref exactly):
 * - Positional encoding: learned wpe (GPT-2) OR RoPE on q/k (Llama, posEnc="rope").
 * - Normalization: LayerNorm (weight+bias) OR RMSNorm (weight only, normType="rmsnorm").
 * - FFN: GELU 4× (fc1/fc2) OR SwiGLU (fc_gate/fc_up/fc_proj, ffnActivation="swiglu").
 * - LM head: separate lmHead OR tied to wte (tieEmbeddings=true → no lmHead param).
 * - softCap on attention scores: applied for GPT-2 (30), OFF for RoPE unless set.
 *
 * Architecture: weights (immutable, shared) are separate from sessions (mutable,
 * per-request) so concurrent requests cannot corrupt each other's KV cache.
 */
import type { ModelConfig } from "@alpha/core";
import type { SeededRng } from "@alpha/core";

// ── Types ──────────────────────────────────────────────────────────────────

/** MLP weights — GELU (fc1/fc2) or SwiGLU (gate/up/proj). ffnDim = hidden width. */
type MlpWeights =
  | { kind: "gelu"; fc1: Float32Array; fc2: Float32Array; ffnDim: number }
  | { kind: "swiglu"; fcGate: Float32Array; fcUp: Float32Array; fcProj: Float32Array; ffnDim: number };

interface InferenceLayer {
  norm1W: Float32Array;          // [nEmbd]
  norm1B: Float32Array | null;   // [nEmbd] — null under RMSNorm
  wq: Float32Array;              // [nEmbd, nEmbd]
  wk: Float32Array;              // [nEmbd, nEmbd]
  wv: Float32Array;              // [nEmbd, nEmbd]
  wo: Float32Array;              // [nEmbd, nEmbd]
  norm2W: Float32Array;          // [nEmbd]
  norm2B: Float32Array | null;   // [nEmbd] — null under RMSNorm
  mlp: MlpWeights;
}

/** Immutable model weights — safe to share across concurrent requests. */
export interface InferenceWeights {
  config: ModelConfig;
  wte: Float32Array;             // [vocabSize, nEmbd]
  wpe: Float32Array | null;      // [blockSize, nEmbd] — null under RoPE
  layers: InferenceLayer[];
  lnFW: Float32Array;            // [nEmbd]
  lnFB: Float32Array | null;     // [nEmbd] — null under RMSNorm
  lmHead: Float32Array;          // [vocabSize, nEmbd] — === wte when tied

  // ── Derived architecture flags (mirrors ModelConfig defaults) ──
  useRope: boolean;              // posEnc === "rope"
  useRms: boolean;               // normType === "rmsnorm"
  ropeTheta: number;             // RoPE base frequency (default 10000)
  softCapVal: number;            // attention-score clamp magnitude; 0 = off
  ffnHidden: number;             // MLP hidden width (scratch sizing)
  ropeCos: Float32Array | null;  // [blockSize * headDim/2] precomputed cos
  ropeSin: Float32Array | null;  // [blockSize * headDim/2] precomputed sin
}

/** Mutable per-request session — KV cache + pre-allocated decode buffers. */
export interface InferenceSession {
  config: ModelConfig;

  // KV cache per layer — flat [nHead * blockSize * headDim] for K and V
  kCache: Float32Array[];
  vCache: Float32Array[];

  // Pre-allocated decode buffers (single-token forward pass)
  _x: Float32Array;           // [nEmbd]
  _lnOut: Float32Array;       // [nEmbd]
  _q: Float32Array;           // [nEmbd]
  _k: Float32Array;           // [nEmbd]
  _v: Float32Array;           // [nEmbd]
  _attnScores: Float32Array;  // [blockSize]
  _attnOut: Float32Array;     // [nEmbd]
  _projected: Float32Array;   // [nEmbd]
  _mlpHidden: Float32Array;   // [ffnHidden]
  _mlpUp?: Float32Array;      // [ffnHidden] — SwiGLU only (the "up" projection)
  _mlpOut: Float32Array;      // [nEmbd]
  _logits: Float32Array;      // [vocabSize]
  _sampleBuf: Float32Array;   // [vocabSize] — scratch for sampling

  // Prefill scratch buffers (reused across prefill calls on same session)
  _prefillX?: Float32Array;
  _prefillLn?: Float32Array;
  _prefillQ?: Float32Array;
  _prefillK?: Float32Array;
  _prefillV?: Float32Array;
  _prefillAttn?: Float32Array;
  _prefillScores?: Float32Array;
  _prefillProj?: Float32Array;
  _prefillMlpH?: Float32Array;
  _prefillMlpUp?: Float32Array; // SwiGLU only
  _prefillLastLn?: Float32Array;
  _prefillMaxT?: number;      // max T these buffers were allocated for
}

/** Optional ordered post-block capture for lens/runtime consumers. */
export interface InferenceCapture {
  readonly requestedSites: ReadonlySet<string>;
  readonly sites: Map<string, Float32Array>;
}

function postBlockSiteId(index: number): string {
  return `block.${index.toString().padStart(3, "0")}.post`;
}

/** @deprecated Use InferenceWeights + InferenceSession instead. */
export type InferenceModel = InferenceWeights & InferenceSession;

// ── Math primitives ────────────────────────────────────────────────────────

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const NORM_EPS = 1e-5;

function layerNorm(
  out: Float32Array, outOff: number,
  x: Float32Array, xOff: number,
  w: Float32Array, b: Float32Array,
  N: number,
): void {
  let mean = 0;
  for (let i = 0; i < N; i++) mean += x[xOff + i];
  mean /= N;

  let variance = 0;
  for (let i = 0; i < N; i++) {
    const d = x[xOff + i] - mean;
    variance += d * d;
  }
  variance /= N;

  const invStd = 1 / Math.sqrt(variance + NORM_EPS);
  for (let i = 0; i < N; i++) {
    out[outOff + i] = (x[xOff + i] - mean) * invStd * w[i] + b[i];
  }
}

/** RMSNorm: out = x / sqrt(mean(x^2)+eps) * w. No mean-centering, no bias.
 *  Matches packages/tensor/src/cpu_ref.ts rmsNorm exactly. */
function rmsNorm(
  out: Float32Array, outOff: number,
  x: Float32Array, xOff: number,
  w: Float32Array,
  N: number,
): void {
  let ms = 0;
  for (let i = 0; i < N; i++) {
    const xi = x[xOff + i];
    ms += xi * xi;
  }
  ms /= N;
  const invRms = 1 / Math.sqrt(ms + NORM_EPS);
  for (let i = 0; i < N; i++) {
    out[outOff + i] = x[xOff + i] * invRms * w[i];
  }
}

/** Dispatch to the configured normalization. `b` is ignored (may be null) under RMSNorm. */
function applyNorm(
  out: Float32Array, outOff: number,
  x: Float32Array, xOff: number,
  w: Float32Array, b: Float32Array | null,
  N: number, useRms: boolean,
): void {
  if (useRms) rmsNorm(out, outOff, x, xOff, w, N);
  else layerNorm(out, outOff, x, xOff, w, b as Float32Array, N);
}

/**
 * Apply RoPE (HF-Llama rotate_half convention) in place to a [nEmbd] = [nHead][headDim]
 * vector at absolute position `pos`. Splits each head dim in HALF (first/second):
 *   out[i]      = a*cos - b*sin
 *   out[i+half] = b*cos + a*sin,   a=v[i], b=v[i+half], (cos,sin)=table[pos,i]
 * Mirrors packages/tensor/src/cpu_ref.ts rope + packages/autograd ropeTables.
 */
function ropeInPlace(
  v: Float32Array, off: number,
  nHead: number, headDim: number, pos: number,
  cosTable: Float32Array, sinTable: Float32Array,
): void {
  const half = headDim >> 1;
  const csBase = pos * half;
  for (let h = 0; h < nHead; h++) {
    const base = off + h * headDim;
    for (let i = 0; i < half; i++) {
      const c = cosTable[csBase + i];
      const s = sinTable[csBase + i];
      const a = v[base + i];
      const bb = v[base + i + half];
      v[base + i] = a * c - bb * s;
      v[base + i + half] = bb * c + a * s;
    }
  }
}

function matvecMul(
  out: Float32Array, oOff: number,
  x: Float32Array, xOff: number,
  W: Float32Array, wOff: number,
  N: number, K: number,
): void {
  for (let j = 0; j < N; j++) {
    let sum = 0;
    const wRow = wOff + j * K;
    for (let k = 0; k < K; k++) {
      sum += x[xOff + k] * W[wRow + k];
    }
    out[oOff + j] = sum;
  }
}

function tiledMatmul(
  out: Float32Array, oOff: number,
  A: Float32Array, aOff: number,
  B_T: Float32Array, bOff: number,
  M: number, N: number, K: number,
): void {
  const TILE = 32;
  const total = M * N;
  for (let i = 0; i < total; i++) out[oOff + i] = 0;

  for (let mi = 0; mi < M; mi += TILE) {
    const mEnd = Math.min(mi + TILE, M);
    for (let ni = 0; ni < N; ni += TILE) {
      const nEnd = Math.min(ni + TILE, N);
      for (let ki = 0; ki < K; ki += TILE) {
        const kEnd = Math.min(ki + TILE, K);
        for (let m = mi; m < mEnd; m++) {
          const aRowOff = aOff + m * K;
          const oRowOff = oOff + m * N;
          for (let n = ni; n < nEnd; n++) {
            let sum = out[oRowOff + n];
            const bRowOff = bOff + n * K;
            for (let k = ki; k < kEnd; k++) {
              sum += A[aRowOff + k] * B_T[bRowOff + k];
            }
            out[oRowOff + n] = sum;
          }
        }
      }
    }
  }
}

function geluInPlace(x: Float32Array, off: number, N: number): void {
  for (let i = 0; i < N; i++) {
    const xi = x[off + i];
    const t = SQRT_2_OVER_PI * (xi + 0.044715 * xi * xi * xi);
    x[off + i] = 0.5 * xi * (1 + Math.tanh(t));
  }
}

function siluInPlace(x: Float32Array, off: number, N: number): void {
  for (let i = 0; i < N; i++) {
    const xi = x[off + i];
    x[off + i] = xi / (1 + Math.exp(-xi));
  }
}

function writeCachePos(
  cache: Float32Array,
  vec: Float32Array, vecOff: number,
  pos: number,
  nHead: number, blockSize: number, headDim: number,
): void {
  for (let h = 0; h < nHead; h++) {
    const cacheOff = h * blockSize * headDim + pos * headDim;
    const vOff = vecOff + h * headDim;
    for (let d = 0; d < headDim; d++) {
      cache[cacheOff + d] = vec[vOff + d];
    }
  }
}

// ── Model preparation ──────────────────────────────────────────────────────

type Param = { shape: number[]; data: Float32Array | number[] };

function extractF32(param: Param): Float32Array {
  if (param.data instanceof Float32Array) return param.data;
  return new Float32Array(param.data);
}

/** Extract a param that may be absent (RMSNorm biases, tied lmHead, etc.). */
function extractF32OrNull(param: Param | undefined): Float32Array | null {
  if (!param) return null;
  return extractF32(param);
}

/** Prepare immutable weights from checkpoint params. */
export function prepareInferenceWeights(
  config: ModelConfig,
  params: Record<string, Param>,
): InferenceWeights {
  const { nLayer, nEmbd, nHead, blockSize } = config;
  const headDim = nEmbd / nHead;

  const useRope = (config.posEnc ?? "learned") === "rope";
  const useRms = (config.normType ?? "layernorm") === "rmsnorm";
  const tied = config.tieEmbeddings === true;
  const ropeTheta = config.ropeTheta ?? 10000;
  // softCap: explicit config wins; else GPT-2 default 30, but OFF under RoPE.
  const softCapVal = config.softCap ?? (useRope ? 0 : 30);
  const activation = config.ffnActivation ?? "gelu";
  const isSwiglu = activation === "swiglu";

  let ffnHidden = 0;
  const layers: InferenceLayer[] = [];
  for (let i = 0; i < nLayer; i++) {
    // Support both grouped wqkv (new) and separate wq/wk/wv (old) checkpoints
    let wq: Float32Array, wk: Float32Array, wv: Float32Array;
    const wqkvParam = params[`layer.${i}.attn.wqkv`];
    if (wqkvParam) {
      const f32 = extractF32(wqkvParam);
      const size = nEmbd * nEmbd;
      wq = f32.slice(0, size);
      wk = f32.slice(size, 2 * size);
      wv = f32.slice(2 * size, 3 * size);
    } else {
      wq = extractF32(params[`layer.${i}.attn.wq`]);
      wk = extractF32(params[`layer.${i}.attn.wk`]);
      wv = extractF32(params[`layer.${i}.attn.wv`]);
    }

    // MLP — SwiGLU (fc_gate/fc_up/fc_proj) or GELU (fc1/fc2). Prefer explicit
    // config, but fall back to whichever weights exist (robust to older configs).
    let mlp: MlpWeights;
    const gateParam = params[`layer.${i}.mlp.fc_gate`];
    if (isSwiglu || gateParam) {
      const fcGate = extractF32(params[`layer.${i}.mlp.fc_gate`]);
      const fcUp = extractF32(params[`layer.${i}.mlp.fc_up`]);
      const fcProj = extractF32(params[`layer.${i}.mlp.fc_proj`]);
      const ffnDim = fcGate.length / nEmbd;
      mlp = { kind: "swiglu", fcGate, fcUp, fcProj, ffnDim };
      ffnHidden = Math.max(ffnHidden, ffnDim);
    } else {
      const fc1 = extractF32(params[`layer.${i}.mlp.fc1`]);
      const fc2 = extractF32(params[`layer.${i}.mlp.fc2`]);
      const ffnDim = fc1.length / nEmbd;
      mlp = { kind: "gelu", fc1, fc2, ffnDim };
      ffnHidden = Math.max(ffnHidden, ffnDim);
    }

    layers.push({
      norm1W: extractF32(params[`layer.${i}.ln1.weight`]),
      norm1B: extractF32OrNull(params[`layer.${i}.ln1.bias`]),
      wq, wk, wv,
      wo: extractF32(params[`layer.${i}.attn.wo`]),
      norm2W: extractF32(params[`layer.${i}.ln2.weight`]),
      norm2B: extractF32OrNull(params[`layer.${i}.ln2.bias`]),
      mlp,
    });
  }

  const wte = extractF32(params["wte"]);
  // Tied embeddings: the checkpoint omits lmHead; the head IS wte.
  const lmHead = tied ? wte : extractF32(params["lmHead"]);

  // Precompute RoPE cos/sin tables [blockSize, headDim/2] once (mirrors ropeTables).
  let ropeCos: Float32Array | null = null;
  let ropeSin: Float32Array | null = null;
  if (useRope) {
    const half = headDim >> 1;
    ropeCos = new Float32Array(blockSize * half);
    ropeSin = new Float32Array(blockSize * half);
    for (let i = 0; i < half; i++) {
      const invFreq = Math.pow(ropeTheta, (-2 * i) / headDim);
      for (let pos = 0; pos < blockSize; pos++) {
        const angle = pos * invFreq;
        ropeCos[pos * half + i] = Math.cos(angle);
        ropeSin[pos * half + i] = Math.sin(angle);
      }
    }
  }

  return {
    config,
    wte,
    wpe: useRope ? null : extractF32OrNull(params["wpe"]),
    layers,
    lnFW: extractF32(params["lnF.weight"]),
    lnFB: extractF32OrNull(params["lnF.bias"]),
    lmHead,
    useRope,
    useRms,
    ropeTheta,
    softCapVal,
    ffnHidden,
    ropeCos,
    ropeSin,
  };
}

/** Create a mutable session with fresh KV cache and decode buffers. */
export function createSession(weights: InferenceWeights): InferenceSession {
  const { nLayer, nEmbd, nHead, blockSize, vocabSize } = weights.config;
  const headDim = nEmbd / nHead;
  const ffnHidden = weights.ffnHidden || 4 * nEmbd;
  const swiglu = weights.layers.some((l) => l.mlp.kind === "swiglu");

  const session: InferenceSession = {
    config: weights.config,
    kCache: Array.from({ length: nLayer }, () => new Float32Array(nHead * blockSize * headDim)),
    vCache: Array.from({ length: nLayer }, () => new Float32Array(nHead * blockSize * headDim)),
    _x: new Float32Array(nEmbd),
    _lnOut: new Float32Array(nEmbd),
    _q: new Float32Array(nEmbd),
    _k: new Float32Array(nEmbd),
    _v: new Float32Array(nEmbd),
    _attnScores: new Float32Array(blockSize),
    _attnOut: new Float32Array(nEmbd),
    _projected: new Float32Array(nEmbd),
    _mlpHidden: new Float32Array(ffnHidden),
    _mlpOut: new Float32Array(nEmbd),
    _logits: new Float32Array(vocabSize),
    _sampleBuf: new Float32Array(vocabSize),
    _prefillLastLn: new Float32Array(nEmbd),
  };
  if (swiglu) session._mlpUp = new Float32Array(ffnHidden);
  return session;
}

/** @deprecated Use prepareInferenceWeights + createSession instead. */
export function prepareInferenceModel(
  config: ModelConfig,
  params: Record<string, Param>,
): InferenceModel {
  const weights = prepareInferenceWeights(config, params);
  const session = createSession(weights);
  return { ...weights, ...session };
}

export function resetCache(session: InferenceSession): void {
  for (let i = 0; i < session.kCache.length; i++) {
    session.kCache[i].fill(0);
    session.vCache[i].fill(0);
  }
}

/** Deep clone an inference session (including KV cache and scratch buffers). */
export function cloneSession(session: InferenceSession): InferenceSession {
  const copy: InferenceSession = {
    config: session.config,
    kCache: session.kCache.map((k) => new Float32Array(k)),
    vCache: session.vCache.map((v) => new Float32Array(v)),
    _x: new Float32Array(session._x),
    _lnOut: new Float32Array(session._lnOut),
    _q: new Float32Array(session._q),
    _k: new Float32Array(session._k),
    _v: new Float32Array(session._v),
    _attnScores: new Float32Array(session._attnScores),
    _attnOut: new Float32Array(session._attnOut),
    _projected: new Float32Array(session._projected),
    _mlpHidden: new Float32Array(session._mlpHidden),
    _mlpOut: new Float32Array(session._mlpOut),
    _logits: new Float32Array(session._logits),
    _sampleBuf: new Float32Array(session._sampleBuf),
    _prefillLastLn: session._prefillLastLn ? new Float32Array(session._prefillLastLn) : undefined,
  };
  if (session._mlpUp) copy._mlpUp = new Float32Array(session._mlpUp);
  if (session._prefillX) copy._prefillX = new Float32Array(session._prefillX);
  if (session._prefillLn) copy._prefillLn = new Float32Array(session._prefillLn);
  if (session._prefillQ) copy._prefillQ = new Float32Array(session._prefillQ);
  if (session._prefillK) copy._prefillK = new Float32Array(session._prefillK);
  if (session._prefillV) copy._prefillV = new Float32Array(session._prefillV);
  if (session._prefillAttn) copy._prefillAttn = new Float32Array(session._prefillAttn);
  if (session._prefillScores) copy._prefillScores = new Float32Array(session._prefillScores);
  if (session._prefillProj) copy._prefillProj = new Float32Array(session._prefillProj);
  if (session._prefillMlpH) copy._prefillMlpH = new Float32Array(session._prefillMlpH);
  if (session._prefillMlpUp) copy._prefillMlpUp = new Float32Array(session._prefillMlpUp);
  if (session._prefillMaxT !== undefined) copy._prefillMaxT = session._prefillMaxT;
  return copy;
}

export function countModelParams(
  params: Record<string, Param>,
): number {
  let total = 0;
  for (const key of Object.keys(params)) {
    const p = params[key];
    let size = 1;
    for (const d of p.shape) size *= d;
    total += size;
  }
  return total;
}

// ── MLP forward (single-row and batched) ───────────────────────────────────

/** Single-row MLP forward: lnOut[nEmbd] → out[nEmbd]. Uses session scratch. */
function mlpForwardRow(
  out: Float32Array, outOff: number,
  lnOut: Float32Array, lnOff: number,
  mlp: MlpWeights, nEmbd: number,
  hiddenBuf: Float32Array, upBuf: Float32Array | undefined,
): void {
  if (mlp.kind === "swiglu") {
    const ffnDim = mlp.ffnDim;
    matvecMul(hiddenBuf, 0, lnOut, lnOff, mlp.fcGate, 0, ffnDim, nEmbd);
    matvecMul(upBuf!, 0, lnOut, lnOff, mlp.fcUp, 0, ffnDim, nEmbd);
    siluInPlace(hiddenBuf, 0, ffnDim);
    for (let j = 0; j < ffnDim; j++) hiddenBuf[j] *= upBuf![j];
    matvecMul(out, outOff, hiddenBuf, 0, mlp.fcProj, 0, nEmbd, ffnDim);
  } else {
    const ffnDim = mlp.ffnDim;
    matvecMul(hiddenBuf, 0, lnOut, lnOff, mlp.fc1, 0, ffnDim, nEmbd);
    geluInPlace(hiddenBuf, 0, ffnDim);
    matvecMul(out, outOff, hiddenBuf, 0, mlp.fc2, 0, nEmbd, ffnDim);
  }
}

// ── Prefill (batch process all prompt tokens) ──────────────────────────────

function ensurePrefillBuffers(session: InferenceSession, weights: InferenceWeights, T: number): void {
  const { nEmbd } = session.config;
  const ffnHidden = weights.ffnHidden || 4 * nEmbd;
  const swiglu = weights.layers.some((l) => l.mlp.kind === "swiglu");
  if (session._prefillMaxT && session._prefillMaxT >= T) return;
  session._prefillX = new Float32Array(T * nEmbd);
  session._prefillLn = new Float32Array(T * nEmbd);
  session._prefillQ = new Float32Array(T * nEmbd);
  session._prefillK = new Float32Array(T * nEmbd);
  session._prefillV = new Float32Array(T * nEmbd);
  session._prefillAttn = new Float32Array(T * nEmbd);
  session._prefillScores = new Float32Array(T * T);
  session._prefillProj = new Float32Array(T * nEmbd);
  session._prefillMlpH = new Float32Array(T * ffnHidden);
  if (swiglu) session._prefillMlpUp = new Float32Array(T * ffnHidden);
  session._prefillMaxT = T;
}

/**
 * Process all prompt tokens at once, populating KV cache for each layer.
 * Returns logits for the last position only (shape [vocabSize]).
 *
 * Accepts either (weights, session, tokens) or legacy (model, tokens).
 */
export function prefill(
  weightsOrModel: InferenceWeights | InferenceModel,
  sessionOrTokens: InferenceSession | Int32Array,
  maybeTokens?: Int32Array,
  capture?: InferenceCapture,
): Float32Array {
  let weights: InferenceWeights;
  let session: InferenceSession;
  let tokens: Int32Array;

  if (maybeTokens !== undefined) {
    weights = weightsOrModel as InferenceWeights;
    session = sessionOrTokens as InferenceSession;
    tokens = maybeTokens;
  } else {
    // Legacy: model has both weights and session fields
    const model = weightsOrModel as InferenceModel;
    weights = model;
    session = model;
    tokens = sessionOrTokens as Int32Array;
  }

  const { wte, wpe, layers, lnFW, lnFB, lmHead, useRope, useRms, softCapVal, ropeCos, ropeSin } = weights;
  const { nEmbd, nHead, vocabSize, blockSize } = weights.config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
  const capOn = softCapVal > 0;
  const T = tokens.length;

  if (T <= 0) throw new RangeError("prefill requires at least 1 token");
  if (T > blockSize) throw new RangeError(`prefill token count (${T}) exceeds block size (${blockSize})`);

  const { kCache, vCache } = session;

  // Reuse prefill scratch buffers from session
  ensurePrefillBuffers(session, weights, T);
  const x = session._prefillX!;
  const lnBuf = session._prefillLn!;
  const Q = session._prefillQ!;
  const K = session._prefillK!;
  const V = session._prefillV!;
  const attnOut = session._prefillAttn!;
  const scores = session._prefillScores!;
  const proj = session._prefillProj!;
  const mlpH = session._prefillMlpH!;
  const mlpUp = session._prefillMlpUp;

  // Token (+ optional learned position) embeddings
  for (let t = 0; t < T; t++) {
    const xOff = t * nEmbd;
    const wteOff = tokens[t] * nEmbd;
    if (wpe) {
      const wpeOff = t * nEmbd;
      for (let i = 0; i < nEmbd; i++) x[xOff + i] = wte[wteOff + i] + wpe[wpeOff + i];
    } else {
      for (let i = 0; i < nEmbd; i++) x[xOff + i] = wte[wteOff + i];
    }
  }

  // Transformer blocks
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];

    // ── Norm 1 ──
    for (let t = 0; t < T; t++) {
      applyNorm(lnBuf, t * nEmbd, x, t * nEmbd, layer.norm1W, layer.norm1B, nEmbd, useRms);
    }

    // ── Q, K, V projections: [T, nEmbd] @ W[nEmbd, nEmbd] ──
    tiledMatmul(Q, 0, lnBuf, 0, layer.wq, 0, T, nEmbd, nEmbd);
    tiledMatmul(K, 0, lnBuf, 0, layer.wk, 0, T, nEmbd, nEmbd);
    tiledMatmul(V, 0, lnBuf, 0, layer.wv, 0, T, nEmbd, nEmbd);

    // ── RoPE rotate q/k per position (before caching K) ──
    if (useRope) {
      for (let t = 0; t < T; t++) {
        ropeInPlace(Q, t * nEmbd, nHead, headDim, t, ropeCos!, ropeSin!);
        ropeInPlace(K, t * nEmbd, nHead, headDim, t, ropeCos!, ropeSin!);
      }
    }

    // ── Store K, V in cache for all positions ──
    for (let t = 0; t < T; t++) {
      writeCachePos(kCache[l], K, t * nEmbd, t, nHead, blockSize, headDim);
      writeCachePos(vCache[l], V, t * nEmbd, t, nHead, blockSize, headDim);
    }

    // ── Multi-head causal attention ──
    for (let h = 0; h < nHead; h++) {
      const kHeadOff = h * blockSize * headDim;
      const vHeadOff = h * blockSize * headDim;

      for (let t1 = 0; t1 < T; t1++) {
        const qOff = t1 * nEmbd + h * headDim;
        let maxScore = -Infinity;

        for (let t2 = 0; t2 <= t1; t2++) {
          let score = 0;
          const kOff = kHeadOff + t2 * headDim;
          for (let d = 0; d < headDim; d++) {
            score += Q[qOff + d] * kCache[l][kOff + d];
          }
          score *= scaleVal;
          if (capOn) {
            if (score > softCapVal) score = softCapVal;
            else if (score < -softCapVal) score = -softCapVal;
          }
          scores[t1 * T + t2] = score;
          if (score > maxScore) maxScore = score;
        }

        let sumExp = 0;
        for (let t2 = 0; t2 <= t1; t2++) {
          scores[t1 * T + t2] = Math.exp(scores[t1 * T + t2] - maxScore);
          sumExp += scores[t1 * T + t2];
        }
        const invSum = 1 / sumExp;
        for (let t2 = 0; t2 <= t1; t2++) {
          scores[t1 * T + t2] *= invSum;
        }

        const outOff = t1 * nEmbd + h * headDim;
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            sum += scores[t1 * T + t2] * vCache[l][vHeadOff + t2 * headDim + d];
          }
          attnOut[outOff + d] = sum;
        }
      }
    }

    // ── Output projection + residual ──
    tiledMatmul(proj, 0, attnOut, 0, layer.wo, 0, T, nEmbd, nEmbd);
    for (let i = 0; i < T * nEmbd; i++) x[i] += proj[i];

    // ── Norm 2 ──
    for (let t = 0; t < T; t++) {
      applyNorm(lnBuf, t * nEmbd, x, t * nEmbd, layer.norm2W, layer.norm2B, nEmbd, useRms);
    }

    // ── MLP + residual (GELU or SwiGLU), batched over all T positions ──
    const mlp = layer.mlp;
    if (mlp.kind === "swiglu") {
      const ffnDim = mlp.ffnDim;
      // gate = lnBuf @ fcGate^T ; up = lnBuf @ fcUp^T ; h = silu(gate) ⊙ up
      tiledMatmul(mlpH, 0, lnBuf, 0, mlp.fcGate, 0, T, ffnDim, nEmbd);
      tiledMatmul(mlpUp!, 0, lnBuf, 0, mlp.fcUp, 0, T, ffnDim, nEmbd);
      siluInPlace(mlpH, 0, T * ffnDim);
      for (let i = 0; i < T * ffnDim; i++) mlpH[i] *= mlpUp![i];
      tiledMatmul(proj, 0, mlpH, 0, mlp.fcProj, 0, T, nEmbd, ffnDim);
    } else {
      const ffnDim = mlp.ffnDim;
      tiledMatmul(mlpH, 0, lnBuf, 0, mlp.fc1, 0, T, ffnDim, nEmbd);
      geluInPlace(mlpH, 0, T * ffnDim);
      tiledMatmul(proj, 0, mlpH, 0, mlp.fc2, 0, T, nEmbd, ffnDim);
    }
    for (let i = 0; i < T * nEmbd; i++) x[i] += proj[i];

    const siteId = postBlockSiteId(l);
    if (capture?.requestedSites.has(siteId)) {
      capture.sites.set(siteId, new Float32Array(x.subarray(0, T * nEmbd)));
    }
  }

  // Final layer norm — only for last position
  const lastOff = (T - 1) * nEmbd;
  const lastLn = session._prefillLastLn ?? new Float32Array(nEmbd);
  applyNorm(lastLn, 0, x, lastOff, lnFW, lnFB, nEmbd, useRms);

  // LM head — only for last position
  const logits = session._logits;
  matvecMul(logits, 0, lastLn, 0, lmHead, 0, vocabSize, nEmbd);

  return logits;
}

// ── Decode step (single token with KV cache) ──────────────────────────────

/**
 * Forward pass for a single token at the given position, using cached K/V.
 * Returns logits (shape [vocabSize]).
 *
 * Accepts either (weights, session, token, pos) or legacy (model, token, pos).
 */
export function decodeStep(
  weightsOrModel: InferenceWeights | InferenceModel,
  sessionOrToken: InferenceSession | number,
  tokenOrPos: number,
  maybePos?: number,
  capture?: InferenceCapture,
): Float32Array {
  let weights: InferenceWeights;
  let session: InferenceSession;
  let token: number;
  let pos: number;

  if (maybePos !== undefined) {
    weights = weightsOrModel as InferenceWeights;
    session = sessionOrToken as InferenceSession;
    token = tokenOrPos;
    pos = maybePos;
  } else {
    const model = weightsOrModel as InferenceModel;
    weights = model;
    session = model;
    token = sessionOrToken as number;
    pos = tokenOrPos;
  }

  const { wte, wpe, layers, lnFW, lnFB, lmHead, useRope, useRms, softCapVal, ropeCos, ropeSin } = weights;
  const { nEmbd, nHead, vocabSize, blockSize } = weights.config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
  const capOn = softCapVal > 0;
  const seqLen = pos + 1;

  if (pos < 0 || pos >= blockSize) throw new RangeError(`decodeStep pos (${pos}) out of range [0, ${blockSize})`);

  const { kCache, vCache } = session;
  const x = session._x;
  const lnOut = session._lnOut;
  const q = session._q;
  const k = session._k;
  const v = session._v;
  const attnScores = session._attnScores;
  const attnOut = session._attnOut;
  const projected = session._projected;
  const mlpHidden = session._mlpHidden;
  const mlpUp = session._mlpUp;
  const mlpOut = session._mlpOut;
  const logits = session._logits;

  // Token (+ optional learned position) embedding
  const wteOff = token * nEmbd;
  if (wpe) {
    const wpeOff = pos * nEmbd;
    for (let i = 0; i < nEmbd; i++) x[i] = wte[wteOff + i] + wpe[wpeOff + i];
  } else {
    for (let i = 0; i < nEmbd; i++) x[i] = wte[wteOff + i];
  }

  // Transformer blocks
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];

    applyNorm(lnOut, 0, x, 0, layer.norm1W, layer.norm1B, nEmbd, useRms);

    matvecMul(q, 0, lnOut, 0, layer.wq, 0, nEmbd, nEmbd);
    matvecMul(k, 0, lnOut, 0, layer.wk, 0, nEmbd, nEmbd);
    matvecMul(v, 0, lnOut, 0, layer.wv, 0, nEmbd, nEmbd);

    // RoPE rotate q/k at the current absolute position before caching K.
    if (useRope) {
      ropeInPlace(q, 0, nHead, headDim, pos, ropeCos!, ropeSin!);
      ropeInPlace(k, 0, nHead, headDim, pos, ropeCos!, ropeSin!);
    }

    writeCachePos(kCache[l], k, 0, pos, nHead, blockSize, headDim);
    writeCachePos(vCache[l], v, 0, pos, nHead, blockSize, headDim);

    for (let h = 0; h < nHead; h++) {
      const qOff = h * headDim;
      const kHeadOff = h * blockSize * headDim;
      const vHeadOff = h * blockSize * headDim;

      let maxScore = -Infinity;
      for (let t = 0; t < seqLen; t++) {
        let score = 0;
        const kOff = kHeadOff + t * headDim;
        for (let d = 0; d < headDim; d++) {
          score += q[qOff + d] * kCache[l][kOff + d];
        }
        score *= scaleVal;
        if (capOn) {
          if (score > softCapVal) score = softCapVal;
          else if (score < -softCapVal) score = -softCapVal;
        }
        attnScores[t] = score;
        if (score > maxScore) maxScore = score;
      }

      let sumExp = 0;
      for (let t = 0; t < seqLen; t++) {
        attnScores[t] = Math.exp(attnScores[t] - maxScore);
        sumExp += attnScores[t];
      }
      const invSum = 1 / sumExp;
      for (let t = 0; t < seqLen; t++) {
        attnScores[t] *= invSum;
      }

      const outOff = h * headDim;
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let t = 0; t < seqLen; t++) {
          sum += attnScores[t] * vCache[l][vHeadOff + t * headDim + d];
        }
        attnOut[outOff + d] = sum;
      }
    }

    matvecMul(projected, 0, attnOut, 0, layer.wo, 0, nEmbd, nEmbd);
    for (let i = 0; i < nEmbd; i++) x[i] += projected[i];

    applyNorm(lnOut, 0, x, 0, layer.norm2W, layer.norm2B, nEmbd, useRms);

    mlpForwardRow(mlpOut, 0, lnOut, 0, layer.mlp, nEmbd, mlpHidden, mlpUp);

    for (let i = 0; i < nEmbd; i++) x[i] += mlpOut[i];

    const siteId = postBlockSiteId(l);
    if (capture?.requestedSites.has(siteId)) capture.sites.set(siteId, new Float32Array(x));
  }

  applyNorm(lnOut, 0, x, 0, lnFW, lnFB, nEmbd, useRms);
  matvecMul(logits, 0, lnOut, 0, lmHead, 0, vocabSize, nEmbd);

  return logits;
}

// ── Sampling ───────────────────────────────────────────────────────────────

/**
 * Sample a token from logits with temperature scaling, top-k filtering, and
 * optional top-p (nucleus) filtering.
 *
 * Accepts either (session, logits, ...) or legacy (model, logits, ...).
 * If temperature <= 0, returns argmax (greedy decoding).
 */
export function sampleFromLogits(
  sessionOrModel: InferenceSession | InferenceModel,
  logits: Float32Array,
  temperature: number,
  topk: number,
  rng: SeededRng,
  topp = 1.0,
): number {
  const vocabSize = sessionOrModel.config.vocabSize;
  const scaled = sessionOrModel._sampleBuf;
  const topKVal = Number(topk);
  const topPVal = Number(topp);
  const topK = Number.isFinite(topKVal) ? Math.max(0, Math.floor(topKVal)) : 0;
  const topP = Number.isFinite(topPVal) ? Math.min(1, Math.max(0, topPVal)) : 1;

  // Greedy decoding
  if (temperature <= 0) {
    let bestIdx = 0;
    let bestVal = logits[0];
    for (let i = 1; i < vocabSize; i++) {
      if (logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
    }
    return bestIdx;
  }

  // Temperature scaling
  const invTemp = 1 / temperature;
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = logits[i] * invTemp;
  }

  // Top-k filtering via partial selection (O(V) average instead of O(V log V))
  if (topK > 0 && topK < vocabSize) {
    // Find k-th largest using quickselect on a scratch copy
    // We need the threshold value. Do nth_element-style partitioning.
    const threshold = quickselectThreshold(scaled, vocabSize, topK);
    for (let i = 0; i < vocabSize; i++) {
      if (scaled[i] < threshold) scaled[i] = -Infinity;
    }
  }

  // Top-p (nucleus) filtering. Keep the minimum-probability prefix whose
  // cumulative mass >= topP, then sample from that set.
  if (topP > 0 && topP < 1) {
    let maxVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      if (scaled[i] > maxVal) maxVal = scaled[i];
    }
    if (!Number.isFinite(maxVal)) return 0;

    const active: number[] = [];
    let sumExp = 0;
    for (let i = 0; i < vocabSize; i++) {
      const v = scaled[i];
      if (Number.isFinite(v)) {
        const p = Math.exp(v - maxVal);
        scaled[i] = p;
        sumExp += p;
        active.push(i);
      } else {
        scaled[i] = 0;
      }
    }
    if (active.length === 0 || sumExp <= 0) {
      // Fallback to argmax if all candidates were filtered out numerically.
      let bestIdx = 0;
      let bestVal = logits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
      }
      return bestIdx;
    }

    active.sort((a, b) => scaled[b] - scaled[a]);
    const targetMass = sumExp * topP;
    let keptMass = 0;
    let keepCount = 0;
    for (; keepCount < active.length; keepCount++) {
      keptMass += scaled[active[keepCount]];
      if (keptMass >= targetMass) {
        keepCount++;
        break;
      }
    }
    if (keepCount <= 0) keepCount = 1;
    if (keptMass <= 0) keptMass = scaled[active[0]];

    const r = rng.next() * keptMass;
    let cumsum = 0;
    for (let i = 0; i < keepCount; i++) {
      const idx = active[i];
      cumsum += scaled[idx];
      if (r < cumsum) return idx;
    }
    return active[Math.max(0, keepCount - 1)];
  }

  // Softmax
  let maxVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (scaled[i] > maxVal) maxVal = scaled[i];
  }
  let sumExp = 0;
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = Math.exp(scaled[i] - maxVal);
    sumExp += scaled[i];
  }

  // Multinomial sample
  const r = rng.next() * sumExp;
  let cumsum = 0;
  for (let i = 0; i < vocabSize; i++) {
    cumsum += scaled[i];
    if (r < cumsum) return i;
  }
  return vocabSize - 1;
}

/** Find the k-th largest value in arr[0..n) without full sort. O(V) average. */
function quickselectThreshold(arr: Float32Array, n: number, k: number): number {
  // For small k, use a simple max-heap approach: track the k largest values
  if (k <= 64) {
    // Min-heap of size k — keep the k largest values
    const heap = new Float32Array(k);
    heap.fill(-Infinity);
    for (let i = 0; i < n; i++) {
      if (arr[i] > heap[0]) {
        heap[0] = arr[i];
        // Sift down
        let idx = 0;
        while (true) {
          const left = 2 * idx + 1;
          const right = 2 * idx + 2;
          let smallest = idx;
          if (left < k && heap[left] < heap[smallest]) smallest = left;
          if (right < k && heap[right] < heap[smallest]) smallest = right;
          if (smallest === idx) break;
          const tmp = heap[idx]; heap[idx] = heap[smallest]; heap[smallest] = tmp;
          idx = smallest;
        }
      }
    }
    return heap[0]; // min of top-k = the threshold
  }
  // For larger k, fall back to sort (rare with typical topk=40)
  const copy = new Float32Array(n);
  copy.set(arr.subarray(0, n));
  copy.sort();
  return copy[n - k];
}

// ── Session pool ───────────────────────────────────────────────────────────

/** Simple pool of inference sessions to avoid repeated allocation. */
export class SessionPool {
  private pool: InferenceSession[] = [];
  private weights: InferenceWeights;

  constructor(weights: InferenceWeights) {
    this.weights = weights;
  }

  acquire(): InferenceSession {
    const session = this.pool.pop();
    if (session) {
      resetCache(session);
      return session;
    }
    return createSession(this.weights);
  }

  release(session: InferenceSession): void {
    this.pool.push(session);
  }
}
