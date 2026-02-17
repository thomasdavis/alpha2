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
  backward: (outGrad: TensorData, b: Backend) => TensorData[]
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
  return record(ctx, ctx.backend.sub(a.data, b.data), [a, b], (g, B) => {
    return [reduceBroadcast(B, g, a.data.shape), reduceBroadcast(B, B.neg(g), b.data.shape)];
  });
}

export function mul(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.mul(aData, bData), [a, b], (g, B) => {
    return [
      reduceBroadcast(B, B.mul(g, bData), aData.shape),
      reduceBroadcast(B, B.mul(g, aData), bData.shape),
    ];
  });
}

export function div(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.div(aData, bData), [a, b], (g, B) => {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    const ga = B.div(g, bData);
    const gb = B.neg(B.div(B.mul(g, aData), B.mul(bData, bData)));
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
  return record(ctx, ctx.backend.matmul(aData, bData), [a, b], (g, B) => {
    // For 2D: dL/dA = G @ B^T, dL/dB = A^T @ G
    const ndimA = aData.shape.length;
    const ndimB = bData.shape.length;
    const ga = B.matmul(g, B.transpose(bData, ndimB - 2, ndimB - 1));
    const gb = B.matmul(B.transpose(aData, ndimA - 2, ndimA - 1), g);
    return [ga, gb];
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
  return record(ctx, out, [a], (g, B) => {
    return [B.div(g, B.scale(out, 2))];
  });
}

export function relu(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.relu(aData), [a], (g, B) => {
    // mask: 1 where input > 0, 0 otherwise
    const mask = B.fromArray(
      Array.from(aData.data).map((v) => (v > 0 ? 1 : 0)),
      [...aData.shape],
      aData.dtype,
    );
    return [B.mul(g, mask)];
  });
}

export function gelu(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.gelu(aData), [a], (g, B) => {
    // Approximate GELU gradient
    const SQRT2PI = Math.sqrt(2 / Math.PI);
    const out = B.fromArray(
      Array.from(aData.data).map((x) => {
        const inner = SQRT2PI * (x + 0.044715 * x * x * x);
        const tanh_val = Math.tanh(inner);
        const sech2 = 1 - tanh_val * tanh_val;
        const dInner = SQRT2PI * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dInner;
      }),
      [...aData.shape],
      aData.dtype,
    );
    return [B.mul(g, out)];
  });
}

// ── NN ops ─────────────────────────────────────────────────────────────────

export function embedding(ctx: Ctx, weight: Variable, indices: TensorData): Variable {
  const wData = weight.data;
  return record(ctx, ctx.backend.embedding(wData, indices), [weight], (g, B) => {
    // Scatter gradients back to weight rows
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
    // Simplified layernorm backward
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

      // Accumulate dw, db
      for (let j = 0; j < dim; j++) {
        const xhat = (xArr[off + j] - mu) * invStd;
        dwArr[j] += gArr[off + j] * xhat;
        dbArr[j] += gArr[off + j];
      }

      // dx
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

export function softmax(ctx: Ctx, a: Variable, axis?: number): Variable {
  const out = ctx.backend.softmax(a.data, axis);
  return record(ctx, out, [a], (g, B) => {
    // dsoftmax: s * (g - sum(g * s))
    const sg = B.mul(out, g);
    const sumSg = B.sum(sg, axis ?? -1, true);
    const expanded = broadcastTo(B, sumSg, out.shape);
    return [B.mul(out, B.sub(g, expanded))];
  });
}

export function crossEntropy(ctx: Ctx, logits: Variable, targets: TensorData): Variable {
  const logitsData = logits.data;
  return record(ctx, ctx.backend.crossEntropy(logitsData, targets), [logits], (g, B) => {
    // Gradient: softmax(logits) - one_hot(targets)
    const probs = B.softmax(logitsData, -1);
    const [N, C] = logitsData.shape;
    const gradArr = new Float32Array(N * C);
    const probArr = probs.data as Float32Array;
    const gScalar = (g.data as Float32Array)[0];
    for (let i = 0; i < N; i++) {
      const t = targets.data[i];
      for (let c = 0; c < C; c++) {
        gradArr[i * C + c] = (probArr[i * C + c] - (c === t ? 1 : 0)) * gScalar / N;
      }
    }
    return [B.fromArray(Array.from(gradArr), [N, C], logitsData.dtype)];
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
  return record(ctx, ctx.backend.transpose(a.data, dim0, dim1), [a], (g, B) => {
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
  // For now, simple: create full-size tensor and fill
  const size = shapeSize(targetShape);
  const srcSize = shapeSize(t.shape);
  const out = new Float32Array(size);
  const src = t.data as Float32Array;
  if (srcSize === 1) {
    out.fill(src[0]);
  } else {
    // Repeat pattern
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
