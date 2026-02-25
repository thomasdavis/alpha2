/**
 * Optimized pure-TS inference engine for Alpha GPT models.
 *
 * Bypasses autograd/tape machinery for 10-20× faster CPU inference via:
 * - KV cache (avoid recomputing prior tokens)
 * - Tiled matmul (cache-friendly memory access)
 * - Zero allocation in the decode loop (pre-allocated buffers)
 * - Last-token-only LM head (skip vocabSize computation for all but final position)
 * - Fused layernorm, attention scoring, in-place GELU
 *
 * Architecture: weights (immutable, shared) are separate from sessions (mutable,
 * per-request) so concurrent requests cannot corrupt each other's KV cache.
 */
import type { ModelConfig } from "@alpha/core";
import type { SeededRng } from "@alpha/core";

// ── Types ──────────────────────────────────────────────────────────────────

interface InferenceLayer {
  ln1W: Float32Array;  // [nEmbd]
  ln1B: Float32Array;  // [nEmbd]
  wq: Float32Array;    // [nEmbd, nEmbd]
  wk: Float32Array;    // [nEmbd, nEmbd]
  wv: Float32Array;    // [nEmbd, nEmbd]
  wo: Float32Array;    // [nEmbd, nEmbd]
  ln2W: Float32Array;  // [nEmbd]
  ln2B: Float32Array;  // [nEmbd]
  fc1: Float32Array;   // [4*nEmbd, nEmbd]
  fc2: Float32Array;   // [nEmbd, 4*nEmbd]
}

/** Immutable model weights — safe to share across concurrent requests. */
export interface InferenceWeights {
  config: ModelConfig;
  wte: Float32Array;     // [vocabSize, nEmbd]
  wpe: Float32Array;     // [blockSize, nEmbd]
  layers: InferenceLayer[];
  lnFW: Float32Array;    // [nEmbd]
  lnFB: Float32Array;    // [nEmbd]
  lmHead: Float32Array;  // [vocabSize, nEmbd]
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
  _mlpHidden: Float32Array;   // [4*nEmbd]
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
  _prefillLastLn?: Float32Array;
  _prefillMaxT?: number;      // max T these buffers were allocated for
}

/** @deprecated Use InferenceWeights + InferenceSession instead. */
export type InferenceModel = InferenceWeights & InferenceSession;

// ── Math primitives ────────────────────────────────────────────────────────

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

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

  const invStd = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < N; i++) {
    out[outOff + i] = (x[xOff + i] - mean) * invStd * w[i] + b[i];
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

function extractF32(param: { data: Float32Array | number[] }): Float32Array {
  if (param.data instanceof Float32Array) return param.data;
  return new Float32Array(param.data);
}

/** Prepare immutable weights from checkpoint params. */
export function prepareInferenceWeights(
  config: ModelConfig,
  params: Record<string, { shape: number[]; data: Float32Array | number[] }>,
): InferenceWeights {
  const { nLayer } = config;

  const layers: InferenceLayer[] = [];
  for (let i = 0; i < nLayer; i++) {
    layers.push({
      ln1W: extractF32(params[`layer.${i}.ln1.weight`]),
      ln1B: extractF32(params[`layer.${i}.ln1.bias`]),
      wq: extractF32(params[`layer.${i}.attn.wq`]),
      wk: extractF32(params[`layer.${i}.attn.wk`]),
      wv: extractF32(params[`layer.${i}.attn.wv`]),
      wo: extractF32(params[`layer.${i}.attn.wo`]),
      ln2W: extractF32(params[`layer.${i}.ln2.weight`]),
      ln2B: extractF32(params[`layer.${i}.ln2.bias`]),
      fc1: extractF32(params[`layer.${i}.mlp.fc1`]),
      fc2: extractF32(params[`layer.${i}.mlp.fc2`]),
    });
  }

  return {
    config,
    wte: extractF32(params["wte"]),
    wpe: extractF32(params["wpe"]),
    layers,
    lnFW: extractF32(params["lnF.weight"]),
    lnFB: extractF32(params["lnF.bias"]),
    lmHead: extractF32(params["lmHead"]),
  };
}

/** Create a mutable session with fresh KV cache and decode buffers. */
export function createSession(weights: InferenceWeights): InferenceSession {
  const { nLayer, nEmbd, nHead, blockSize, vocabSize } = weights.config;
  const headDim = nEmbd / nHead;

  return {
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
    _mlpHidden: new Float32Array(4 * nEmbd),
    _mlpOut: new Float32Array(nEmbd),
    _logits: new Float32Array(vocabSize),
    _sampleBuf: new Float32Array(vocabSize),
    _prefillLastLn: new Float32Array(nEmbd),
  };
}

/** @deprecated Use prepareInferenceWeights + createSession instead. */
export function prepareInferenceModel(
  config: ModelConfig,
  params: Record<string, { shape: number[]; data: Float32Array | number[] }>,
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

export function countModelParams(
  params: Record<string, { shape: number[]; data: Float32Array | number[] }>,
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

// ── Prefill (batch process all prompt tokens) ──────────────────────────────

function ensurePrefillBuffers(session: InferenceSession, T: number): void {
  const { nEmbd } = session.config;
  if (session._prefillMaxT && session._prefillMaxT >= T) return;
  session._prefillX = new Float32Array(T * nEmbd);
  session._prefillLn = new Float32Array(T * nEmbd);
  session._prefillQ = new Float32Array(T * nEmbd);
  session._prefillK = new Float32Array(T * nEmbd);
  session._prefillV = new Float32Array(T * nEmbd);
  session._prefillAttn = new Float32Array(T * nEmbd);
  session._prefillScores = new Float32Array(T * T);
  session._prefillProj = new Float32Array(T * nEmbd);
  session._prefillMlpH = new Float32Array(T * 4 * nEmbd);
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

  const { wte, wpe, layers, lnFW, lnFB, lmHead } = weights;
  const { nEmbd, nHead, vocabSize, blockSize } = weights.config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
  const T = tokens.length;

  if (T <= 0) throw new RangeError("prefill requires at least 1 token");
  if (T > blockSize) throw new RangeError(`prefill token count (${T}) exceeds block size (${blockSize})`);

  const { kCache, vCache } = session;

  // Reuse prefill scratch buffers from session
  ensurePrefillBuffers(session, T);
  const x = session._prefillX!;
  const lnBuf = session._prefillLn!;
  const Q = session._prefillQ!;
  const K = session._prefillK!;
  const V = session._prefillV!;
  const attnOut = session._prefillAttn!;
  const scores = session._prefillScores!;
  const proj = session._prefillProj!;
  const mlpH = session._prefillMlpH!;

  // Token + position embeddings
  for (let t = 0; t < T; t++) {
    const xOff = t * nEmbd;
    const wteOff = tokens[t] * nEmbd;
    const wpeOff = t * nEmbd;
    for (let i = 0; i < nEmbd; i++) {
      x[xOff + i] = wte[wteOff + i] + wpe[wpeOff + i];
    }
  }

  // Transformer blocks
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];

    // ── LN1 ──
    for (let t = 0; t < T; t++) {
      layerNorm(lnBuf, t * nEmbd, x, t * nEmbd, layer.ln1W, layer.ln1B, nEmbd);
    }

    // ── Q, K, V projections: [T, nEmbd] @ W[nEmbd, nEmbd] ──
    tiledMatmul(Q, 0, lnBuf, 0, layer.wq, 0, T, nEmbd, nEmbd);
    tiledMatmul(K, 0, lnBuf, 0, layer.wk, 0, T, nEmbd, nEmbd);
    tiledMatmul(V, 0, lnBuf, 0, layer.wv, 0, T, nEmbd, nEmbd);

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
          if (score > 30) score = 30;
          else if (score < -30) score = -30;
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

    // ── LN2 ──
    for (let t = 0; t < T; t++) {
      layerNorm(lnBuf, t * nEmbd, x, t * nEmbd, layer.ln2W, layer.ln2B, nEmbd);
    }

    // ── MLP: fc1 → GELU → fc2 + residual ──
    tiledMatmul(mlpH, 0, lnBuf, 0, layer.fc1, 0, T, 4 * nEmbd, nEmbd);
    geluInPlace(mlpH, 0, T * 4 * nEmbd);
    tiledMatmul(proj, 0, mlpH, 0, layer.fc2, 0, T, nEmbd, 4 * nEmbd);
    for (let i = 0; i < T * nEmbd; i++) x[i] += proj[i];
  }

  // Final layer norm — only for last position
  const lastOff = (T - 1) * nEmbd;
  const lastLn = session._prefillLastLn ?? new Float32Array(nEmbd);
  layerNorm(lastLn, 0, x, lastOff, lnFW, lnFB, nEmbd);

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

  const { wte, wpe, layers, lnFW, lnFB, lmHead } = weights;
  const { nEmbd, nHead, vocabSize, blockSize } = weights.config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
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
  const mlpOut = session._mlpOut;
  const logits = session._logits;

  // Token + position embedding
  const wteOff = token * nEmbd;
  const wpeOff = pos * nEmbd;
  for (let i = 0; i < nEmbd; i++) {
    x[i] = wte[wteOff + i] + wpe[wpeOff + i];
  }

  // Transformer blocks
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];

    layerNorm(lnOut, 0, x, 0, layer.ln1W, layer.ln1B, nEmbd);

    matvecMul(q, 0, lnOut, 0, layer.wq, 0, nEmbd, nEmbd);
    matvecMul(k, 0, lnOut, 0, layer.wk, 0, nEmbd, nEmbd);
    matvecMul(v, 0, lnOut, 0, layer.wv, 0, nEmbd, nEmbd);

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
        if (score > 30) score = 30;
        else if (score < -30) score = -30;
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

    layerNorm(lnOut, 0, x, 0, layer.ln2W, layer.ln2B, nEmbd);

    matvecMul(mlpHidden, 0, lnOut, 0, layer.fc1, 0, 4 * nEmbd, nEmbd);
    geluInPlace(mlpHidden, 0, 4 * nEmbd);
    matvecMul(mlpOut, 0, mlpHidden, 0, layer.fc2, 0, nEmbd, 4 * nEmbd);

    for (let i = 0; i < nEmbd; i++) x[i] += mlpOut[i];
  }

  layerNorm(lnOut, 0, x, 0, lnFW, lnFB, nEmbd);
  matvecMul(logits, 0, lnOut, 0, lmHead, 0, vocabSize, nEmbd);

  return logits;
}

// ── Sampling ───────────────────────────────────────────────────────────────

/**
 * Sample a token from logits with temperature scaling and top-k filtering.
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
): number {
  const vocabSize = sessionOrModel.config.vocabSize;
  const scaled = sessionOrModel._sampleBuf;

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
  if (topk > 0 && topk < vocabSize) {
    // Find k-th largest using quickselect on a scratch copy
    const buf = sessionOrModel._sampleBuf; // reuse same buffer — we already have values there
    // We need the threshold value. Do nth_element-style partitioning.
    const threshold = quickselectThreshold(scaled, vocabSize, topk);
    for (let i = 0; i < vocabSize; i++) {
      if (scaled[i] < threshold) scaled[i] = -Infinity;
    }
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
