/**
 * Optimized pure-TS inference engine for Alpha GPT models.
 *
 * Bypasses autograd/tape machinery for 10-20× faster CPU inference via:
 * - KV cache (avoid recomputing prior tokens)
 * - Tiled matmul (cache-friendly memory access)
 * - Zero allocation in the decode loop (pre-allocated buffers)
 * - Last-token-only LM head (skip vocabSize computation for all but final position)
 * - Fused layernorm, attention scoring, in-place GELU
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

export interface InferenceModel {
  config: ModelConfig;
  wte: Float32Array;     // [vocabSize, nEmbd]
  wpe: Float32Array;     // [blockSize, nEmbd]
  layers: InferenceLayer[];
  lnFW: Float32Array;    // [nEmbd]
  lnFB: Float32Array;    // [nEmbd]
  lmHead: Float32Array;  // [vocabSize, nEmbd]

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
}

// ── Math primitives ────────────────────────────────────────────────────────

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/**
 * Layer normalization over a single vector of length N.
 * out[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i]
 */
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

/**
 * Matrix-vector multiply: out[j] = sum_k x[k] * W[j*K + k], j in 0..N
 * W is [N, K] row-major — each row j is dot-producted with x.
 */
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

/**
 * Tiled matmul: C[M,N] = A[M,K] @ B_T[N,K]^T
 * B_T is stored as [N, K] (pre-transposed), so C[i,j] = sum_k A[i,k] * B_T[j,k].
 * 32×32 tiles keep sub-blocks in L1 cache (~12KB per tile triple).
 */
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

/**
 * In-place GELU (tanh approximation, matching @alpha/autograd):
 * x[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/π) * (x[i] + 0.044715 * x[i]³)))
 */
function geluInPlace(x: Float32Array, off: number, N: number): void {
  for (let i = 0; i < N; i++) {
    const xi = x[off + i];
    const t = SQRT_2_OVER_PI * (xi + 0.044715 * xi * xi * xi);
    x[off + i] = 0.5 * xi * (1 + Math.tanh(t));
  }
}

/**
 * Write a [nEmbd]-sized vector into the KV cache at a given position.
 * Cache layout: [nHead, blockSize, headDim].
 * The vector is [nEmbd] = [nHead * headDim], interleaved by head.
 */
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

export function prepareInferenceModel(
  config: ModelConfig,
  params: Record<string, { shape: number[]; data: Float32Array | number[] }>,
): InferenceModel {
  const { nLayer, nEmbd, nHead, blockSize, vocabSize } = config;
  const headDim = nEmbd / nHead;

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

    // KV cache: per layer, flat [nHead, blockSize, headDim]
    kCache: Array.from({ length: nLayer }, () => new Float32Array(nHead * blockSize * headDim)),
    vCache: Array.from({ length: nLayer }, () => new Float32Array(nHead * blockSize * headDim)),

    // Pre-allocated decode buffers
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
  };
}

export function resetCache(model: InferenceModel): void {
  for (let i = 0; i < model.kCache.length; i++) {
    model.kCache[i].fill(0);
    model.vCache[i].fill(0);
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

/**
 * Process all prompt tokens at once, populating KV cache for each layer.
 * Returns logits for the last position only (shape [vocabSize]).
 */
export function prefill(model: InferenceModel, tokens: Int32Array): Float32Array {
  const { config, wte, wpe, layers, lnFW, lnFB, lmHead, kCache, vCache } = model;
  const { nEmbd, nHead, vocabSize, blockSize } = config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
  const T = tokens.length;

  // Allocate prefill buffers (one-time per request, ~3MB for T=256)
  const x = new Float32Array(T * nEmbd);
  const lnBuf = new Float32Array(T * nEmbd);
  const Q = new Float32Array(T * nEmbd);
  const K = new Float32Array(T * nEmbd);
  const V = new Float32Array(T * nEmbd);
  const attnOut = new Float32Array(T * nEmbd);
  const scores = new Float32Array(T * T);
  const proj = new Float32Array(T * nEmbd);
  const mlpH = new Float32Array(T * 4 * nEmbd);

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

        // Compute attention scores (causal: t2 <= t1 only)
        for (let t2 = 0; t2 <= t1; t2++) {
          let score = 0;
          const kOff = kHeadOff + t2 * headDim;
          for (let d = 0; d < headDim; d++) {
            score += Q[qOff + d] * kCache[l][kOff + d];
          }
          score *= scaleVal;
          // Logit capping (PaLM/Gemma technique, matching training code)
          if (score > 30) score = 30;
          else if (score < -30) score = -30;
          scores[t1 * T + t2] = score;
          if (score > maxScore) maxScore = score;
        }

        // Softmax over valid positions
        let sumExp = 0;
        for (let t2 = 0; t2 <= t1; t2++) {
          scores[t1 * T + t2] = Math.exp(scores[t1 * T + t2] - maxScore);
          sumExp += scores[t1 * T + t2];
        }
        const invSum = 1 / sumExp;
        for (let t2 = 0; t2 <= t1; t2++) {
          scores[t1 * T + t2] *= invSum;
        }

        // Weighted sum of V
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
  const lastLn = new Float32Array(nEmbd);
  layerNorm(lastLn, 0, x, lastOff, lnFW, lnFB, nEmbd);

  // LM head — only for last position: [1, nEmbd] @ [vocabSize, nEmbd]^T
  const logits = model._logits;
  matvecMul(logits, 0, lastLn, 0, lmHead, 0, vocabSize, nEmbd);

  return logits;
}

// ── Decode step (single token with KV cache) ──────────────────────────────

/**
 * Forward pass for a single token at the given position, using cached K/V
 * from all prior positions. Returns logits (shape [vocabSize]).
 *
 * The returned Float32Array is a pre-allocated buffer — use it before the
 * next decodeStep call.
 */
export function decodeStep(
  model: InferenceModel,
  token: number,
  pos: number,
): Float32Array {
  const { config, wte, wpe, layers, lnFW, lnFB, lmHead, kCache, vCache } = model;
  const { nEmbd, nHead, vocabSize, blockSize } = config;
  const headDim = nEmbd / nHead;
  const scaleVal = 1 / Math.sqrt(headDim);
  const seqLen = pos + 1; // number of cached positions including current

  // Pre-allocated buffers (zero allocation in hot loop)
  const x = model._x;
  const lnOut = model._lnOut;
  const q = model._q;
  const k = model._k;
  const v = model._v;
  const attnScores = model._attnScores;
  const attnOut = model._attnOut;
  const projected = model._projected;
  const mlpHidden = model._mlpHidden;
  const mlpOut = model._mlpOut;
  const logits = model._logits;

  // Token + position embedding
  const wteOff = token * nEmbd;
  const wpeOff = pos * nEmbd;
  for (let i = 0; i < nEmbd; i++) {
    x[i] = wte[wteOff + i] + wpe[wpeOff + i];
  }

  // Transformer blocks
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];

    // ── LN1 → Attention → Residual ──
    layerNorm(lnOut, 0, x, 0, layer.ln1W, layer.ln1B, nEmbd);

    // Q, K, V projections: [nEmbd] → [nEmbd]
    matvecMul(q, 0, lnOut, 0, layer.wq, 0, nEmbd, nEmbd);
    matvecMul(k, 0, lnOut, 0, layer.wk, 0, nEmbd, nEmbd);
    matvecMul(v, 0, lnOut, 0, layer.wv, 0, nEmbd, nEmbd);

    // Write K, V to cache at current position
    writeCachePos(kCache[l], k, 0, pos, nHead, blockSize, headDim);
    writeCachePos(vCache[l], v, 0, pos, nHead, blockSize, headDim);

    // Multi-head attention over all cached positions
    for (let h = 0; h < nHead; h++) {
      const qOff = h * headDim;
      const kHeadOff = h * blockSize * headDim;
      const vHeadOff = h * blockSize * headDim;

      // Compute attention scores: q_h · k_ht / sqrt(headDim)
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

      // Softmax
      let sumExp = 0;
      for (let t = 0; t < seqLen; t++) {
        attnScores[t] = Math.exp(attnScores[t] - maxScore);
        sumExp += attnScores[t];
      }
      const invSum = 1 / sumExp;
      for (let t = 0; t < seqLen; t++) {
        attnScores[t] *= invSum;
      }

      // Weighted sum of V: attnOut_h[d] = sum_t scores[t] * V_h[t, d]
      const outOff = h * headDim;
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let t = 0; t < seqLen; t++) {
          sum += attnScores[t] * vCache[l][vHeadOff + t * headDim + d];
        }
        attnOut[outOff + d] = sum;
      }
    }

    // Output projection + residual
    matvecMul(projected, 0, attnOut, 0, layer.wo, 0, nEmbd, nEmbd);
    for (let i = 0; i < nEmbd; i++) x[i] += projected[i];

    // ── LN2 → MLP → Residual ──
    layerNorm(lnOut, 0, x, 0, layer.ln2W, layer.ln2B, nEmbd);

    matvecMul(mlpHidden, 0, lnOut, 0, layer.fc1, 0, 4 * nEmbd, nEmbd);
    geluInPlace(mlpHidden, 0, 4 * nEmbd);
    matvecMul(mlpOut, 0, mlpHidden, 0, layer.fc2, 0, nEmbd, 4 * nEmbd);

    for (let i = 0; i < nEmbd; i++) x[i] += mlpOut[i];
  }

  // Final layer norm
  layerNorm(lnOut, 0, x, 0, lnFW, lnFB, nEmbd);

  // LM head: [1, nEmbd] @ [vocabSize, nEmbd]^T → [vocabSize]
  matvecMul(logits, 0, lnOut, 0, lmHead, 0, vocabSize, nEmbd);

  return logits;
}

// ── Sampling ───────────────────────────────────────────────────────────────

/**
 * Sample a token from logits with temperature scaling and top-k filtering.
 * Uses the model's pre-allocated sample buffer to avoid allocations.
 */
export function sampleFromLogits(
  model: InferenceModel,
  logits: Float32Array,
  temperature: number,
  topk: number,
  rng: SeededRng,
): number {
  const vocabSize = model.config.vocabSize;
  const scaled = model._sampleBuf;

  // Temperature scaling
  const invTemp = 1 / temperature;
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = logits[i] * invTemp;
  }

  // Top-k filtering
  if (topk > 0 && topk < vocabSize) {
    // Find the k-th largest value via sorted copy
    const sorted = new Float32Array(vocabSize);
    sorted.set(scaled);
    sorted.sort();
    const threshold = sorted[vocabSize - topk];
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
