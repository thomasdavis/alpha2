/**
 * Differentiable operations: each wraps a backend op + records backward on the tape.
 *
 * Every function takes a Tape + Backend + Variable inputs, returns Variable output.
 * The backward closure captures what it needs to compute input gradients.
 */
import type { TensorData, Backend, Shape } from "@alpha/core";
import { shapeSize } from "@alpha/core";
import { Variable, type Tape } from "./tape.js";

type Ctx = { tape: Tape; backend: Backend };

// helper: create output variable and record on tape
function record(
  ctx: Ctx,
  data: TensorData,
  inputs: Variable[],
  backward: (outGrad: TensorData, b: Backend, release?: (td: TensorData) => void, needsGrad?: boolean[]) => TensorData[]
): Variable {
  const out = new Variable(data, true);
  ctx.tape.record({ output: out, inputs, backward });
  return out;
}

// ── Arithmetic ─────────────────────────────────────────────────────────────

export function add(ctx: Ctx, a: Variable, b: Variable): Variable {
  return record(ctx, ctx.backend.add(a.data, b.data), [a, b], (g, B) => {
    let ga = g;
    let gb = g;
    // Handle broadcasting: sum over broadcasted dims
    ga = reduceBroadcast(B, g, a.data.shape);
    gb = reduceBroadcast(B, g, b.data.shape);
    return [ga, gb];
  });
}

export function sub(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aShape = a.data.shape, bShape = b.data.shape;
  return record(ctx, ctx.backend.sub(a.data, b.data), [a, b], (g, B, release) => {
    const negG = B.neg(g);
    const gb = reduceBroadcast(B, negG, bShape);
    if (release && gb !== negG) release(negG);
    return [reduceBroadcast(B, g, aShape), gb];
  });
}

export function mul(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.mul(aData, bData), [a, b], (g, B, release) => {
    const gTimesB = B.mul(g, bData);
    const gTimesA = B.mul(g, aData);
    const ga = reduceBroadcast(B, gTimesB, aData.shape);
    const gb = reduceBroadcast(B, gTimesA, bData.shape);
    if (release) {
      if (ga !== gTimesB) release(gTimesB);
      if (gb !== gTimesA) release(gTimesA);
    }
    return [ga, gb];
  });
}

export function div(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.div(aData, bData), [a, b], (g, B, release) => {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    const ga = B.div(g, bData);
    const gTimesA = B.mul(g, aData);
    const bSq = B.mul(bData, bData);
    const ratio = B.div(gTimesA, bSq);
    const gb = B.neg(ratio);
    if (release) { release(gTimesA); release(bSq); release(ratio); }
    return [reduceBroadcast(B, ga, aData.shape), reduceBroadcast(B, gb, bData.shape)];
  });
}

export function scale(ctx: Ctx, a: Variable, s: number): Variable {
  return record(ctx, ctx.backend.scale(a.data, s), [a], (g, B) => {
    return [B.scale(g, s)];
  });
}

export function neg(ctx: Ctx, a: Variable): Variable {
  return record(ctx, ctx.backend.neg(a.data), [a], (g, B) => [B.neg(g)]);
}

// ── Matmul ─────────────────────────────────────────────────────────────────

export function matmul(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.matmul(aData, bData), [a, b], (g, B, release, needsGrad) => {
    // For 2D: dL/dA = G @ B^T, dL/dB = A^T @ G
    const ndimA = aData.shape.length;
    const ndimB = bData.shape.length;
    let ga: TensorData | null = null;
    let gb: TensorData | null = null;
    if (!needsGrad || needsGrad[0]) {
      const tB = B.transpose(bData, ndimB - 2, ndimB - 1);
      ga = B.matmul(g, tB);
      if (release) release(tB);
    }
    if (!needsGrad || needsGrad[1]) {
      const tA = B.transpose(aData, ndimA - 2, ndimA - 1);
      gb = B.matmul(tA, g);
      if (release) release(tA);
    }
    return [ga!, gb!];
  });
}

/**
 * Fused matmul with B transposed: computes A @ B^T.
 * B is stored as [N, K] but used as [K, N].
 * Eliminates separate transpose dispatch on the forward path.
 * Falls back to transpose + matmul if backend doesn't support it.
 */
export function matmulTransposed(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  const B = ctx.backend;
  // Use fused kernel if available, otherwise fall back to transpose + matmul
  const out = B.matmulTransposed
    ? B.matmulTransposed(aData, bData)
    : B.matmul(aData, B.transpose(bData, bData.shape.length - 2, bData.shape.length - 1));
  return record(ctx, out, [a, b], (g, B, release, needsGrad) => {
    // C = A @ B^T where B is [N, K]
    let ga: TensorData | null = null;
    let gb: TensorData | null = null;
    if (!needsGrad || needsGrad[0]) {
      // dL/dA = G @ B (G is [..., M, N], B is [..., N, K] → result [..., M, K])
      ga = B.matmul(g, bData);
    }
    if (!needsGrad || needsGrad[1]) {
      // dL/dB = G^T @ A (G^T is [..., N, M], A is [..., M, K] → result [..., N, K])
      const ndimG = g.shape.length;
      const tG = B.transpose(g, ndimG - 2, ndimG - 1);
      gb = B.matmul(tG, aData);
      if (release) release(tG);
    }
    return [ga!, gb!];
  });
}

// ── Reductions ─────────────────────────────────────────────────────────────

export function sum(ctx: Ctx, a: Variable, axis?: number, keepdims?: boolean): Variable {
  const aShape = a.data.shape;
  return record(ctx, ctx.backend.sum(a.data, axis, keepdims), [a], (g, B) => {
    // Broadcast gradient back to input shape
    return [broadcastTo(B, g, aShape)];
  });
}

export function mean(ctx: Ctx, a: Variable, axis?: number, keepdims?: boolean): Variable {
  const aShape = a.data.shape;
  const n = axis !== undefined ? aShape[axis < 0 ? aShape.length + axis : axis] : shapeSize(aShape);
  return record(ctx, ctx.backend.mean(a.data, axis, keepdims), [a], (g, B) => {
    return [B.scale(broadcastTo(B, g, aShape), 1 / n)];
  });
}

// ── Element-wise ───────────────────────────────────────────────────────────

export function exp(ctx: Ctx, a: Variable): Variable {
  const out = ctx.backend.exp(a.data);
  return record(ctx, out, [a], (g, B) => [B.mul(g, out)]);
}

export function log(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.log(aData), [a], (g, B) => {
    return [B.div(g, aData)];
  });
}

export function sqrt(ctx: Ctx, a: Variable): Variable {
  const out = ctx.backend.sqrt(a.data);
  return record(ctx, out, [a], (g, B, release) => {
    const denom = B.scale(out, 2);
    const result = B.div(g, denom);
    if (release) release(denom);
    return [result];
  });
}

export function relu(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.relu(aData), [a], (g, B) => {
    if (B.reluBackward) return [B.reluBackward(aData, g)];
    const src = aData.data as Float32Array;
    const maskArr = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) maskArr[i] = src[i] > 0 ? 1 : 0;
    const mask: TensorData = { shape: [...aData.shape], dtype: aData.dtype, data: maskArr };
    return [B.mul(g, mask)];
  });
}

export function clamp(ctx: Ctx, a: Variable, lo: number, hi: number): Variable {
  const aData = a.data;
  const clamped = ctx.backend.clamp(aData, lo, hi);
  return record(ctx, clamped, [a], (g, B) => {
    // Gradient passes through where lo < x < hi, zero where clamped.
    // Use clampBackward if available (GPU-optimized single dispatch).
    if (B.clampBackward) return [B.clampBackward(aData, g, lo, hi)];
    // CPU fallback
    const src = aData.data as Float32Array;
    const gArr = g.data as Float32Array;
    const grad = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) grad[i] = (src[i] > lo && src[i] < hi) ? gArr[i] : 0;
    return [{ shape: [...g.shape], dtype: g.dtype, data: grad } as TensorData];
  });
}

export function softCap(ctx: Ctx, a: Variable, cap: number): Variable {
  // tanh(x/cap) * cap — smooth logit capping (PaLM/Gemma technique)
  // Use native kernel if backend supports it (single dispatch vs 7 composed ops)
  if (ctx.backend.softCap) {
    const aData = a.data;
    const out = ctx.backend.softCap(aData, cap);
    return record(ctx, out, [a], (g, B, release) => {
      if (B.softCapBackward) return [B.softCapBackward(g, aData, cap)];
      // CPU fallback for backward
      const t = B.softCap!(aData, cap);
      const tanhVals = B.scale(t, 1 / cap);
      const tanhSq = B.mul(tanhVals, tanhVals);
      const ones = B.ones(tanhSq.shape, tanhSq.dtype);
      const deriv = B.sub(ones, tanhSq);
      const result = B.mul(g, deriv);
      if (release) { release(t); release(tanhVals); release(tanhSq); release(ones); release(deriv); }
      return [result];
    });
  }

  // Composed fallback (for backends without native softCap):
  // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
  // Clamp exp input to [-80, 80] to prevent float32 overflow (exp(88) ≈ max f32).
  // This only affects |x| > cap*40 = 1200 where tanh gradient is already ~0.
  const xScaled = scale(ctx, a, 2 / cap); // 2x/cap
  const xSafe = clamp(ctx, xScaled, -80, 80); // prevent exp overflow
  const e2x = exp(ctx, xSafe); // exp(2x/cap)
  const onesVar = new Variable(ctx.backend.ones(e2x.data.shape, e2x.data.dtype), false);
  const numer = sub(ctx, e2x, onesVar); // exp(2x/cap) - 1
  const denom = add(ctx, e2x, onesVar); // exp(2x/cap) + 1
  const tanhVal = div(ctx, numer, denom); // tanh(x/cap)
  return scale(ctx, tanhVal, cap); // tanh(x/cap) * cap
}

export function gelu(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.gelu(aData), [a], (g, B, release) => {
    if (B.geluBackward) return [B.geluBackward(aData, g)];
    const SQRT2PI = Math.sqrt(2 / Math.PI);
    const src = aData.data as Float32Array;
    const geluGrad = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      const inner = SQRT2PI * (x + 0.044715 * x * x * x);
      const tanh_val = Math.tanh(inner);
      const sech2 = 1 - tanh_val * tanh_val;
      const dInner = SQRT2PI * (1 + 3 * 0.044715 * x * x);
      geluGrad[i] = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dInner;
    }
    const out: TensorData = { shape: [...aData.shape], dtype: aData.dtype, data: geluGrad };
    return [B.mul(g, out)];
  });
}

// ── NN ops ─────────────────────────────────────────────────────────────────

export function embedding(ctx: Ctx, weight: Variable, indices: TensorData): Variable {
  const wData = weight.data;
  return record(ctx, ctx.backend.embedding(wData, indices), [weight], (g, B) => {
    if (B.embeddingBackward) return [B.embeddingBackward(indices, g, wData.shape[0])];
    // CPU fallback: scatter gradients back to weight rows
    const [vocabSize, dim] = wData.shape;
    const grad = B.zeros([vocabSize, dim], wData.dtype);
    const nIdx = shapeSize(indices.shape);
    for (let i = 0; i < nIdx; i++) {
      const idx = indices.data[i];
      for (let d = 0; d < dim; d++) {
        (grad.data as Float32Array)[idx * dim + d] += (g.data as Float32Array)[i * dim + d];
      }
    }
    return [grad];
  });
}

export function layerNorm(
  ctx: Ctx,
  x: Variable,
  weight: Variable,
  bias: Variable,
  eps: number,
): Variable {
  const xData = x.data;
  const wData = weight.data;
  return record(ctx, ctx.backend.layerNorm(xData, wData, bias.data, eps), [x, weight, bias], (g, B) => {
    if (B.layerNormBackward) {
      const { dx, dw, db } = B.layerNormBackward(xData, wData, g, eps);
      return [dx, dw, db];
    }
    // CPU fallback
    const shape = xData.shape;
    const dim = shape[shape.length - 1];
    const n = shapeSize(shape) / dim;
    const xArr = xData.data as Float32Array;
    const wArr = wData.data as Float32Array;
    const gArr = g.data as Float32Array;

    const dx = B.zeros(shape, xData.dtype);
    const dw = B.zeros(wData.shape, wData.dtype);
    const db = B.zeros(wData.shape, wData.dtype);
    const dxArr = dx.data as Float32Array;
    const dwArr = dw.data as Float32Array;
    const dbArr = db.data as Float32Array;

    for (let i = 0; i < n; i++) {
      const off = i * dim;
      let mu = 0;
      for (let j = 0; j < dim; j++) mu += xArr[off + j];
      mu /= dim;
      let variance = 0;
      for (let j = 0; j < dim; j++) {
        const d = xArr[off + j] - mu;
        variance += d * d;
      }
      variance /= dim;
      const invStd = 1 / Math.sqrt(variance + eps);

      for (let j = 0; j < dim; j++) {
        const xhat = (xArr[off + j] - mu) * invStd;
        dwArr[j] += gArr[off + j] * xhat;
        dbArr[j] += gArr[off + j];
      }

      let sum1 = 0, sum2 = 0;
      for (let j = 0; j < dim; j++) {
        const dy = gArr[off + j] * wArr[j];
        sum1 += dy;
        sum2 += dy * (xArr[off + j] - mu) * invStd;
      }
      for (let j = 0; j < dim; j++) {
        const xhat = (xArr[off + j] - mu) * invStd;
        const dy = gArr[off + j] * wArr[j];
        dxArr[off + j] = invStd * (dy - (sum1 + xhat * sum2) / dim);
      }
    }
    return [dx, dw, db];
  });
}

export function dropout(ctx: Ctx, a: Variable, p: number, training: boolean): Variable {
  if (!training || p === 0) return a;
  const aData = a.data;
  const size = shapeSize(aData.shape);
  const maskArr = new Float32Array(size);
  const scaleVal = 1 / (1 - p);
  // Generate dropout mask: 1/(1-p) where kept, 0 where dropped
  for (let i = 0; i < size; i++) maskArr[i] = Math.random() > p ? scaleVal : 0;
  const cpuMask: TensorData = { shape: [...aData.shape], dtype: aData.dtype, data: maskArr };
  // Clone mask into backend storage so it's GPU-resident when using GPU backend
  const mask = ctx.backend.clone(cpuMask);
  const out = ctx.backend.mul(aData, mask);
  return record(ctx, out, [a], (g, B) => {
    return [B.mul(g, mask)];
  });
}

/**
 * Fused residual + dropout + add: output = residual + dropout(projected, p, training)
 * Single GPU dispatch replaces mul(projected, mask) + add(residual, dropResult).
 * Backward: grad_residual = upstream_grad, grad_projected = upstream_grad * mask.
 */
export function residualDropoutAdd(
  ctx: Ctx,
  residual: Variable,
  projected: Variable,
  p: number,
  training: boolean,
): Variable {
  // No dropout: just add
  if (!training || p === 0) return add(ctx, residual, projected);

  const projData = projected.data;
  const resData = residual.data;
  const size = shapeSize(projData.shape);
  const scaleVal = 1 / (1 - p);

  // Generate dropout mask on CPU (same RNG as standalone dropout)
  const maskArr = new Float32Array(size);
  for (let i = 0; i < size; i++) maskArr[i] = Math.random() > p ? scaleVal : 0;
  const cpuMask: TensorData = { shape: [...projData.shape], dtype: projData.dtype, data: maskArr };
  const mask = ctx.backend.clone(cpuMask);

  // Use fused kernel if backend supports it
  if (ctx.backend.residualDropoutAdd) {
    const out = ctx.backend.residualDropoutAdd(resData, projData, mask);
    return record(ctx, out, [residual, projected], (g, B, release) => {
      // grad_residual = upstream_grad (pass-through via broadcast reduction)
      const ga = reduceBroadcast(B, g, resData.shape);
      // grad_projected = upstream_grad * mask
      const gb = B.mul(g, mask);
      return [ga, gb];
    });
  }

  // Fallback: separate ops
  const dropOut = ctx.backend.mul(projData, mask);
  const out = ctx.backend.add(resData, dropOut);
  return record(ctx, out, [residual, projected], (g, B, release) => {
    const ga = reduceBroadcast(B, g, resData.shape);
    const gb = B.mul(g, mask);
    if (release) release(dropOut);
    return [ga, gb];
  });
}

export function softmax(ctx: Ctx, a: Variable, axis?: number): Variable {
  const out = ctx.backend.softmax(a.data, axis);
  return record(ctx, out, [a], (g, B, release) => {
    // dsoftmax: s * (g - sum(g * s))
    const sg = B.mul(out, g);
    const sumSg = B.sum(sg, axis ?? -1, true);
    const expanded = broadcastTo(B, sumSg, out.shape);
    const diff = B.sub(g, expanded);
    const result = B.mul(out, diff);
    if (release) { release(sg); release(sumSg); if (expanded !== sumSg) release(expanded); release(diff); }
    return [result];
  });
}

export function crossEntropy(ctx: Ctx, logits: Variable, targets: TensorData): Variable {
  const logitsData = logits.data;
  return record(ctx, ctx.backend.crossEntropy(logitsData, targets), [logits], (g, B, release) => {
    if (B.crossEntropyBackward) return [B.crossEntropyBackward(logitsData, targets, g)];
    // CPU fallback: (softmax(logits) - one_hot(targets)) * gScalar / N
    const probs = B.softmax(logitsData, -1);
    const [N, C] = logitsData.shape;
    const gScalar = (g.data as Float32Array)[0];
    const oneHotArr = new Float32Array(N * C);
    for (let i = 0; i < N; i++) oneHotArr[targets.data[i] + i * C] = 1.0;
    const oneHot: TensorData = { shape: [N, C], dtype: logitsData.dtype, data: oneHotArr };
    const diff = B.sub(probs, oneHot);
    const result = B.scale(diff, gScalar / N);
    if (release) { release(probs); release(diff); }
    return [result];
  });
}

// ── Flash Attention ────────────────────────────────────────────────────────

/**
 * Fused multi-head attention with Flash Attention algorithm.
 * Q, K, V are [B*H, T, D] (already reshaped to per-head layout).
 * Returns [B*H, T, D] attention output.
 *
 * Replaces: matmul(Q, K^T) → scale → softCap → maskedFill → softmax → dropout → matmul(@V)
 * with a single fused GPU dispatch (forward) and two dispatches (backward).
 */
export function flashAttention(
  ctx: Ctx, q: Variable, k: Variable, v: Variable,
  T: number, scale: number, softCap: number,
): Variable {
  const B = ctx.backend;
  if (!B.flashAttention) throw new Error("flashAttention requires GPU backend");

  const { output, lse } = B.flashAttention(q.data, k.data, v.data, T, scale, softCap);

  return record(ctx, output, [q, k, v], (g, B2, release, needsGrad) => {
    if (!B2.flashAttentionBackward) throw new Error("flashAttentionBackward requires GPU backend");

    const { dQ, dK, dV } = B2.flashAttentionBackward(
      q.data, k.data, v.data, output, g, lse, T, scale, softCap,
    );

    return [
      (!needsGrad || needsGrad[0]) ? dQ : null as any,
      (!needsGrad || needsGrad[1]) ? dK : null as any,
      (!needsGrad || needsGrad[2]) ? dV : null as any,
    ];
  });
}

// ── Reshape / view ops ─────────────────────────────────────────────────────

export function reshape(ctx: Ctx, a: Variable, shape: Shape): Variable {
  const origShape = a.data.shape;
  return record(ctx, ctx.backend.reshape(a.data, shape), [a], (g, B) => {
    return [B.reshape(g, origShape)];
  });
}

export function transpose(ctx: Ctx, a: Variable, dim0: number, dim1: number): Variable {
  return record(ctx, ctx.backend.transpose(a.data, dim0, dim1), [a], (g, B, _release) => {
    return [B.transpose(g, dim0, dim1)];
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────

/** Reduce grad to match target shape (undo broadcasting). */
function reduceBroadcast(B: Backend, grad: TensorData, targetShape: Shape): TensorData {
  if (arraysEqual(grad.shape, targetShape)) return grad;
  // Scalar target
  if (targetShape.length === 0 || (targetShape.length === 1 && targetShape[0] === 1 && grad.shape.length > 1)) {
    return B.sum(grad);
  }
  let result = grad;
  // Sum over leading dims that were broadcast
  while (result.shape.length > targetShape.length) {
    result = B.sum(result, 0);
  }
  // Sum over dims that are 1 in target
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && result.shape[i] !== 1) {
      result = B.sum(result, i, true);
    }
  }
  return result;
}

/** Broadcast a (possibly reduced) tensor to a target shape. */
function broadcastTo(B: Backend, t: TensorData, targetShape: Shape): TensorData {
  if (arraysEqual(t.shape, targetShape)) return t;
  // Use GPU broadcast if available (avoids CPU readback + O(N) copy)
  if (B.broadcast) return B.broadcast(t, targetShape);
  // CPU fallback
  const size = shapeSize(targetShape);
  const srcSize = shapeSize(t.shape);
  const out = new Float32Array(size);
  const src = t.data as Float32Array;
  if (srcSize === 1) {
    out.fill(src[0]);
  } else {
    for (let i = 0; i < size; i++) {
      out[i] = src[i % srcSize];
    }
  }
  return { shape: targetShape, dtype: t.dtype, data: out };
}

function arraysEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
