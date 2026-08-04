/**
 * Differentiable operations: each wraps a backend op + records backward on the tape.
 *
 * Every function takes a Tape + Backend + Variable inputs, returns Variable output.
 * The backward closure captures what it needs to compute input gradients.
 */
import type { TensorData, Backend, Shape } from "@alpha/core";
import { shapeSize, broadcastStrides } from "@alpha/core";
import { Variable, type Tape } from "./tape.js";

type Ctx = { tape: Tape; backend: Backend; dropoutRng?: DropoutRng; release?: (td: TensorData) => void };

// ── Deterministic dropout RNG ─────────────────────────────────────────────

/**
 * Counter-based deterministic RNG for dropout masks.
 *
 * When activation checkpointing recomputes a block during backward,
 * dropout must produce the same mask as the forward pass. This RNG uses
 * a simple counter so the mask sequence is reproducible: save the counter
 * before a block, restore it before recomputation.
 *
 * Algorithm: splitmix64-style mixing of (seed + counter).
 */
export class DropoutRng {
  private seed: number;
  private counter: number;

  constructor(seed: number) {
    this.seed = seed | 0;
    this.counter = 0;
  }

  /** Reset RNG stream (used to reuse one DropoutRng instance in hot loops). */
  reset(seed: number, counter = 0): void {
    this.seed = seed | 0;
    this.counter = counter | 0;
  }

  /** Generate a dropout mask: values are 1/(1-p) where kept, 0 where dropped. */
  nextMask(size: number, p: number): Float32Array {
    const mask = new Float32Array(size);
    const scaleVal = 1 / (1 - p);
    for (let i = 0; i < size; i++) {
      // Simple hash: combine seed, counter, and element index
      const x = this.hash(this.counter, i);
      mask[i] = x > p ? scaleVal : 0;
    }
    this.counter++;
    return mask;
  }

  /**
   * Return seed + counter for GPU mask generation, then advance counter.
   * The GPU kernel reproduces the same hash, producing an identical mask.
   */
  nextMaskParams(): { seed: number; counter: number } {
    const result = { seed: this.seed, counter: this.counter };
    this.counter++;
    return result;
  }

  /** Save counter position for later restore (activation checkpointing). */
  saveCounter(): number {
    return this.counter;
  }

  /** Restore counter to a previously saved position. */
  restoreCounter(n: number): void {
    this.counter = n;
  }

  /** Hash function: maps (counter, index) → uniform [0, 1). */
  private hash(counter: number, index: number): number {
    // splitmix32-style: mix seed + counter + index
    let h = (this.seed + counter * 2654435761 + index * 2246822519) | 0;
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
    h = h ^ (h >>> 16);
    return (h >>> 0) / 4294967296; // [0, 1)
  }
}

// helper: create output variable and record on tape
function record(
  ctx: Ctx,
  data: TensorData,
  inputs: Variable[],
  backward: (outGrad: TensorData, b: Backend, release?: (td: TensorData) => void, needsGrad?: boolean[]) => TensorData[],
  cleanup?: (release?: (td: TensorData) => void) => void,
): Variable {
  const out = new Variable(data, true);
  ctx.tape.record({ output: out, inputs, backward, cleanup });
  return out;
}

// ── Arithmetic ─────────────────────────────────────────────────────────────

export function add(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aShape = a.data.shape, bShape = b.data.shape;
  return record(ctx, ctx.backend.add(a.data, b.data), [a, b], (g, B, release) => {
    const ga = reduceBroadcast(B, g, aShape, release);
    const gb = reduceBroadcast(B, g, bShape, release);
    return [ga, gb];
  });
}

export function sub(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aShape = a.data.shape, bShape = b.data.shape;
  return record(ctx, ctx.backend.sub(a.data, b.data), [a, b], (g, B, release) => {
    const negG = B.neg(g);
    const gb = reduceBroadcast(B, negG, bShape, release);
    if (release && gb !== negG) release(negG);
    return [reduceBroadcast(B, g, aShape, release), gb];
  });
}

export function mul(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  return record(ctx, ctx.backend.mul(aData, bData), [a, b], (g, B, release) => {
    const gTimesB = B.mul(g, bData);
    const gTimesA = B.mul(g, aData);
    const ga = reduceBroadcast(B, gTimesB, aData.shape, release);
    const gb = reduceBroadcast(B, gTimesA, bData.shape, release);
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
    const gaR = reduceBroadcast(B, ga, aData.shape, release);
    const gbR = reduceBroadcast(B, gb, bData.shape, release);
    if (release) {
      if (gaR !== ga) release(ga);
      if (gbR !== gb) release(gb);
    }
    return [gaR, gbR];
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
      if (B.matmulTransposedA) {
        gb = B.matmulTransposedA(aData, g);
      } else {
        const tA = B.transpose(aData, ndimA - 2, ndimA - 1);
        gb = B.matmul(tA, g);
        if (release) release(tA);
      }
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
  const tBFallback = B.matmulTransposed
    ? null
    : B.transpose(bData, bData.shape.length - 2, bData.shape.length - 1);
  let tBLive: TensorData | null = tBFallback;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (!tBLive) return;
    if (release) release(tBLive);
    tBLive = null;
  };
  // Use fused kernel if available, otherwise fall back to transpose + matmul
  const out = B.matmulTransposed
    ? B.matmulTransposed(aData, bData)
    : B.matmul(aData, tBFallback!);
  return record(ctx, out, [a, b], (g, B, release, needsGrad) => {
    // C = A @ B^T where B is [N, K]
    let ga: TensorData | null = null;
    let gb: TensorData | null = null;
    if (!needsGrad || needsGrad[0]) {
      // dL/dA = G @ B (G is [..., M, N], B is [..., N, K] → result [..., M, K])
      ga = B.matmul(g, bData);
    }
    if (!needsGrad || needsGrad[1]) {
      // dL/dB = G^T @ A (result [..., N, K])
      if (B.matmulTransposedA) {
        gb = B.matmulTransposedA(g, aData);
      } else {
        const ndimG = g.shape.length;
        const tG = B.transpose(g, ndimG - 2, ndimG - 1);
        gb = B.matmul(tG, aData);
        if (release) release(tG);
      }
    }
    return [ga!, gb!];
  }, cleanup);
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
  return record(ctx, ctx.backend.mean(a.data, axis, keepdims), [a], (g, B, release) => {
    const expanded = broadcastTo(B, g, aShape);
    const result = B.scale(expanded, 1 / n);
    if (release && expanded !== g) release(expanded);
    return [result];
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

export function silu(ctx: Ctx, a: Variable): Variable {
  const aData = a.data;
  return record(ctx, ctx.backend.silu(aData), [a], (g, B) => {
    if (B.siluBackward) return [B.siluBackward(aData, g)];
    // CPU fallback: silu(x) = x * sigmoid(x), silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    const src = aData.data as Float32Array;
    const gArr = g.data as Float32Array;
    const grad = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      const sig = 1 / (1 + Math.exp(-x));
      grad[i] = gArr[i] * (sig * (1 + x * (1 - sig)));
    }
    return [{ shape: [...aData.shape], dtype: aData.dtype, data: grad } as TensorData];
  });
}

/** Compute the two input gradients for silu(a) * b without retaining its output. */
function siluMulBackwardData(
  B: Backend,
  aData: TensorData,
  bData: TensorData,
  g: TensorData,
  release?: (td: TensorData) => void,
  reusableSiluA?: TensorData,
): [TensorData, TensorData] {
  if (B.siluMulBackward) {
    const grads = B.siluMulBackward(aData, bData, g);
    return [grads[0], grads[1]];
  }

  // Portable fallback: da = (g*b)*silu'(a), db = g*silu(a).
  const siluA = reusableSiluA ?? B.silu(aData);
  const gTimesB = B.mul(g, bData);
  const da = B.siluBackward ? B.siluBackward(aData, gTimesB) : (() => {
    const src = aData.data as Float32Array;
    const gradOut = gTimesB.data as Float32Array;
    const grad = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      const sig = 1 / (1 + Math.exp(-x));
      grad[i] = gradOut[i] * (sig * (1 + x * (1 - sig)));
    }
    return { shape: [...aData.shape], dtype: aData.dtype, data: grad } as TensorData;
  })();
  const db = B.mul(g, siluA);
  if (release) {
    release(gTimesB);
    if (!reusableSiluA) release(siluA);
  }
  return [da, db];
}

/**
 * Fused SiLU-Mul: output = silu(a) * b — single dispatch for SwiGLU.
 * Backward: da = dout * b * silu'(a), db = dout * silu(a)
 */
export function siluMul(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  const B = ctx.backend;
  const siluFallback = B.siluMul ? null : B.silu(aData);
  let siluFallbackLive: TensorData | null = siluFallback;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (!siluFallbackLive) return;
    if (release) release(siluFallbackLive);
    siluFallbackLive = null;
  };
  const out = B.siluMul ? B.siluMul(aData, bData) : B.mul(siluFallback!, bData);
  return record(ctx, out, [a, b], (g, B2, release) => {
    const grads = siluMulBackwardData(
      B2, aData, bData, g, release, siluFallbackLive ?? undefined,
    );
    if (siluFallbackLive && release) release(siluFallbackLive);
    siluFallbackLive = null;
    return grads;
  }, cleanup);
}

/**
 * Selectively rematerialized SwiGLU projection:
 *
 *   output = (silu(gate) * up) @ weight^T
 *
 * The product activation is consumed by the projection and released during
 * the forward graph construction. Backward recomputes only that elementwise
 * product; gate, up, and weight remain ordinary tape inputs. This avoids
 * retaining one [tokens, ffnDim] activation per transformer layer without the
 * much larger GEMM cost of whole-block activation checkpointing.
 *
 * Backends without explicit release keep the forward activation for backward,
 * which preserves portable CPU semantics. GPU backends pass ctx.release and
 * therefore exercise the rematerialized path.
 */
export function siluMulMatmulTransposedRecompute(
  ctx: Ctx,
  gate: Variable,
  up: Variable,
  weight: Variable,
): Variable {
  const gateData = gate.data;
  const upData = up.data;
  const weightData = weight.data;
  const B = ctx.backend;

  const siluFallback = B.siluMul ? null : B.silu(gateData);
  const activation = B.siluMul
    ? B.siluMul(gateData, upData)
    : B.mul(siluFallback!, upData);

  const transposedWeight = B.matmulTransposed
    ? null
    : B.transpose(weightData, weightData.shape.length - 2, weightData.shape.length - 1);
  const out = B.matmulTransposed
    ? B.matmulTransposed(activation, weightData)
    : B.matmul(activation, transposedWeight!);

  let forwardAuxiliaries: TensorData[] = [];
  if (siluFallback) forwardAuxiliaries.push(siluFallback);
  if (transposedWeight) forwardAuxiliaries.push(transposedWeight);
  if (ctx.release) {
    for (const auxiliary of forwardAuxiliaries) ctx.release(auxiliary);
    forwardAuxiliaries = [];
  }

  // When explicit ownership is available the graph has already recorded every
  // forward consumer, so release can safely defer reuse until those consumers
  // complete. Without explicit ownership (CPU/reference backends), keep the
  // activation as a portable fallback instead of pretending it was freed.
  let forwardActivation: TensorData | null = activation;
  if (ctx.release) {
    ctx.release(activation);
    forwardActivation = null;
  }

  const cleanup = (release?: (td: TensorData) => void): void => {
    if (forwardActivation && release) release(forwardActivation);
    forwardActivation = null;
    if (release) {
      for (const auxiliary of forwardAuxiliaries) release(auxiliary);
    }
    forwardAuxiliaries = [];
  };

  return record(ctx, out, [gate, up, weight], (g, B2, release, needsGrad) => {
    const needGate = !needsGrad || needsGrad[0];
    const needUp = !needsGrad || needsGrad[1];
    const needWeight = !needsGrad || needsGrad[2];
    // Only dWeight consumes the SwiGLU product. Gate/up gradients require the
    // projection adjoint, but not the forward product itself.
    const needActivation = needWeight;

    let recomputedActivation: TensorData | null = null;
    let recomputeSiluFallback: TensorData | null = null;
    if (needActivation) {
      if (forwardActivation) {
        recomputedActivation = forwardActivation;
        forwardActivation = null;
      } else if (B2.siluMul) {
        recomputedActivation = B2.siluMul(gateData, upData);
      } else {
        recomputeSiluFallback = B2.silu(gateData);
        recomputedActivation = B2.mul(recomputeSiluFallback, upData);
      }
    }

    let dGate: TensorData | null = null;
    let dUp: TensorData | null = null;
    if (needGate || needUp) {
      const dActivation = B2.matmul(g, weightData);
      const [allDGate, allDUp] = siluMulBackwardData(
        B2, gateData, upData, dActivation, release,
      );
      dGate = needGate ? allDGate : null;
      dUp = needUp ? allDUp : null;
      if (release) {
        release(dActivation);
        if (!needGate) release(allDGate);
        if (!needUp) release(allDUp);
      }
    }

    let dWeight: TensorData | null = null;
    if (needWeight) {
      if (B2.matmulTransposedA) {
        dWeight = B2.matmulTransposedA(g, recomputedActivation!);
      } else {
        const tG = B2.transpose(g, g.shape.length - 2, g.shape.length - 1);
        dWeight = B2.matmul(tG, recomputedActivation!);
        if (release) release(tG);
      }
    }

    if (release) {
      if (recomputedActivation) release(recomputedActivation);
      if (recomputeSiluFallback) release(recomputeSiluFallback);
    }
    return [dGate!, dUp!, dWeight!];
  }, cleanup);
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

/**
 * Fused matmulTransposed + GELU: computes gelu(A @ B^T) in one tape entry.
 * Eliminates an intermediate Variable and tape entry vs separate ops.
 * The pre-GELU matmul output is captured in the backward closure for gelu gradient.
 */
export function matmulTransposedGelu(ctx: Ctx, a: Variable, b: Variable): Variable {
  const aData = a.data, bData = b.data;
  const B = ctx.backend;
  const tBFallback = B.matmulTransposed
    ? null
    : B.transpose(bData, bData.shape.length - 2, bData.shape.length - 1);
  let tBLive: TensorData | null = tBFallback;
  const mmOut = B.matmulTransposed
    ? B.matmulTransposed(aData, bData)
    : B.matmul(aData, tBFallback!);
  let mmOutLive: TensorData | null = mmOut;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (mmOutLive && release) release(mmOutLive);
    mmOutLive = null;
    if (tBLive && release) release(tBLive);
    tBLive = null;
  };
  const geluOut = B.gelu(mmOut);
  return record(ctx, geluOut, [a, b], (g, B2, release, needsGrad) => {
    // Chain rule: d(gelu(mmOut))/d(inputs) = gelu'(mmOut) * d(mmOut)/d(inputs)
    const mmRef = mmOutLive ?? mmOut;
    const dMM = B2.geluBackward
      ? B2.geluBackward(mmRef, g)
      : (() => {
          // CPU fallback for gelu backward
          const SQRT2PI = Math.sqrt(2 / Math.PI);
          const src = mmRef.data as Float32Array;
          const grad = g.data as Float32Array;
          const out = new Float32Array(src.length);
          for (let i = 0; i < src.length; i++) {
            const x = src[i];
            const inner = SQRT2PI * (x + 0.044715 * x * x * x);
            const tanh_val = Math.tanh(inner);
            const sech2 = 1 - tanh_val * tanh_val;
            const dInner = SQRT2PI * (1 + 3 * 0.044715 * x * x);
            out[i] = grad[i] * (0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dInner);
          }
          return { shape: [...mmRef.shape], dtype: mmRef.dtype, data: out } as TensorData;
        })();
    if (mmOutLive && release) release(mmOutLive);
    mmOutLive = null;
    if (tBLive && release) release(tBLive);
    tBLive = null;
    let ga: TensorData | null = null;
    let gb: TensorData | null = null;
    if (!needsGrad || needsGrad[0]) {
      ga = B2.matmul(dMM, bData);
    }
    if (!needsGrad || needsGrad[1]) {
      if (B2.matmulTransposedA) {
        gb = B2.matmulTransposedA(dMM, aData);
      } else {
        const ndim = dMM.shape.length;
        const tG = B2.transpose(dMM, ndim - 2, ndim - 1);
        gb = B2.matmul(tG, aData);
        if (release) release(tG);
      }
    }
    if (release) release(dMM);
    return [ga!, gb!];
  }, cleanup);
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

/**
 * RMS normalization over the last dim (Llama variant): weight-only, no bias,
 * no mean-subtraction. Mirrors layerNorm's autograd structure — uses the
 * backend's fused rmsNormBackward when present, otherwise a CPU-loop fallback.
 */
function rmsNormBackwardData(
  B: Backend,
  xData: TensorData,
  wData: TensorData,
  g: TensorData,
  eps: number,
): [TensorData, TensorData] {
  if (B.rmsNormBackward) {
    const { dx, dw } = B.rmsNormBackward(xData, wData, g, eps);
    return [dx, dw];
  }
  // CPU fallback.
  //   out_j = x_j * r * w_j,   r = 1/sqrt(mean(x^2)+eps)
  //   dw_j += g_j * x_j * r
  //   S    = Σ_j g_j * w_j * x_j
  //   dx_j = r*g_j*w_j - x_j * r^3 * S / dim
  const shape = xData.shape;
  const dim = shape[shape.length - 1];
  const n = shapeSize(shape) / dim;
  const xArr = xData.data as Float32Array;
  const wArr = wData.data as Float32Array;
  const gArr = g.data as Float32Array;

  const dx = B.zeros(shape, xData.dtype);
  const dw = B.zeros(wData.shape, wData.dtype);
  const dxArr = dx.data as Float32Array;
  const dwArr = dw.data as Float32Array;

  for (let i = 0; i < n; i++) {
    const off = i * dim;
    let ms = 0;
    for (let j = 0; j < dim; j++) ms += xArr[off + j] * xArr[off + j];
    ms /= dim;
    const r = 1 / Math.sqrt(ms + eps);
    const r3 = r * r * r;
    let S = 0;
    for (let j = 0; j < dim; j++) S += gArr[off + j] * wArr[j] * xArr[off + j];
    for (let j = 0; j < dim; j++) {
      dwArr[j] += gArr[off + j] * xArr[off + j] * r;
      dxArr[off + j] = r * gArr[off + j] * wArr[j] - xArr[off + j] * r3 * S / dim;
    }
  }
  return [dx, dw];
}

export function rmsNorm(
  ctx: Ctx,
  x: Variable,
  weight: Variable,
  eps: number,
): Variable {
  const xData = x.data;
  const wData = weight.data;
  return record(
    ctx,
    ctx.backend.rmsNorm(xData, wData, eps),
    [x, weight],
    (g, B) => rmsNormBackwardData(B, xData, wData, g, eps),
  );
}

// ── Rotary position embedding (RoPE) ─────────────────────────────────────────

/** Cache of precomputed {cos, sin, negSin} keyed by (T, D, theta, posOffset).
 *  Objects are held strongly here so Helios keeps their GPU uploads resident. */
const ropeTablesCache = new Map<string, { cos: TensorData; sin: TensorData; negSin: TensorData }>();

/**
 * Build (or fetch cached) cos/sin tables of shape [T, D/2] for RoPE.
 *
 * Uses the HF-Llama frequency convention EXACTLY:
 *   inv_freq_i = theta^(-2i/D)      for i in [0, D/2)
 *   angle(t,i) = (t + posOffset) * inv_freq_i
 * cos/sin are stored ONLY for the first half [T, D/2]; the rope op reuses each
 * value for the paired element (HF duplicates the freqs across the two halves).
 */
function ropeTables(T: number, D: number, theta: number, posOffset: number) {
  const key = `${T}:${D}:${theta}:${posOffset}`;
  const cached = ropeTablesCache.get(key);
  if (cached) return cached;
  const half = D >> 1;
  const cos = new Float32Array(T * half);
  const sin = new Float32Array(T * half);
  const negSin = new Float32Array(T * half);
  for (let i = 0; i < half; i++) {
    const invFreq = Math.pow(theta, (-2 * i) / D);
    for (let t = 0; t < T; t++) {
      const angle = (t + posOffset) * invFreq;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[t * half + i] = c;
      sin[t * half + i] = s;
      negSin[t * half + i] = -s;
    }
  }
  const tables = {
    cos: { shape: [T, half], dtype: "f32", data: cos } as TensorData,
    sin: { shape: [T, half], dtype: "f32", data: sin } as TensorData,
    negSin: { shape: [T, half], dtype: "f32", data: negSin } as TensorData,
  };
  ropeTablesCache.set(key, tables);
  return tables;
}

/** Apply the rope rotation to x:[B*H,T,D] with the given cos/sin:[T,D/2].
 *  Uses the backend's fused rope kernel when present, else a CPU loop. */
function ropeApply(B: Backend, x: TensorData, cos: TensorData, sin: TensorData): TensorData {
  if (B.rope) return B.rope(x, cos, sin);
  // CPU fallback (defensive — cpu_ref/Helios both implement B.rope).
  const shape = x.shape;
  const D = shape[shape.length - 1];
  const T = shape[shape.length - 2];
  const half = D >> 1;
  const rows = shapeSize(shape) / D;
  const xArr = x.data as Float32Array;
  const cArr = cos.data as Float32Array;
  const sArr = sin.data as Float32Array;
  const out = new Float32Array(xArr.length);
  for (let r = 0; r < rows; r++) {
    const t = r % T;
    const xBase = r * D;
    const csBase = t * half;
    for (let i = 0; i < half; i++) {
      const c = cArr[csBase + i];
      const s = sArr[csBase + i];
      const a = xArr[xBase + i];
      const bb = xArr[xBase + i + half];
      out[xBase + i] = a * c - bb * s;
      out[xBase + i + half] = bb * c + a * s;
    }
  }
  return { shape: [...shape], dtype: x.dtype, data: out };
}

/**
 * Rotary position embedding applied to q or k of shape [B*H, T, D].
 *
 * Forward rotates each even/odd... NO — the HF-Llama `rotate_half` convention is
 * used EXACTLY (this matters for safetensors export compatibility): the head dim
 * is split in HALF (first half / second half), NOT even/odd interleave. For pair
 * (a=x[i], b=x[i+D/2]) at position t with angle θ_{t,i}:
 *     out[i]     = a*cos - b*sin
 *     out[i+D/2] = b*cos + a*sin
 * i.e. out = x*cos + rotate_half(x)*sin, rotate_half(x) = [-x[D/2:], x[:D/2]].
 *
 * Backward is rotation by -angle (the rotation is orthogonal): dX = ropeApply
 * with sin negated. cos/sin are precomputed on CPU per (T, D, theta, posOffset).
 *
 * @param x         [B*H, T, D] head-major q or k tensor.
 * @param headDim   D — the per-head dimension (must be even).
 * @param posOffset absolute position of x[..,0,..] (0 for training; >0 for KV cache).
 * @param theta     RoPE base frequency (default 10000).
 */
export function rope(ctx: Ctx, x: Variable, headDim: number, posOffset: number, theta: number): Variable {
  const shape = x.data.shape;
  const D = headDim;
  const T = shape[shape.length - 2];
  const { cos, sin, negSin } = ropeTables(T, D, theta, posOffset);
  const out = ropeApply(ctx.backend, x.data, cos, sin);
  return record(ctx, out, [x], (g, B) => {
    // Rotate the incoming gradient by -angle (orthogonal transpose).
    return [ropeApply(B, g, cos, negSin)];
  });
}

/**
 * Convert grouped token-major QKV [B*T,3*H*D] directly to the head-major
 * [B*H,T,D] representation consumed by flash attention, applying HF-Llama
 * RoPE to Q and K along the way. Backends without the paired forward/backward
 * hooks retain the exact established slice → transpose → rope graph.
 */
export function qkvHeadMajorRope(
  ctx: Ctx,
  qkv: Variable,
  batch: number,
  sequence: number,
  heads: number,
  headDim: number,
  theta: number,
): [Variable, Variable, Variable] {
  const modelDim = heads * headDim;
  const expectedShape = [batch * sequence, 3 * modelDim];
  if (qkv.data.shape.length !== 2
    || qkv.data.shape[0] !== expectedShape[0]
    || qkv.data.shape[1] !== expectedShape[1]) {
    throw new Error(`qkvHeadMajorRope: expected [${expectedShape}], got [${qkv.data.shape}]`);
  }
  if (headDim <= 0 || (headDim & 1) !== 0) {
    throw new Error(`qkvHeadMajorRope: headDim must be positive and even, got ${headDim}`);
  }

  const B = ctx.backend;
  if (!B.qkvHeadMajorRope || !B.qkvHeadMajorRopeBackward) {
    const [qFlat, kFlat, vFlat] = sliceQkv(ctx, qkv);
    const toHeadMajor = (value: Variable): Variable => reshape(
      ctx,
      transpose(ctx, reshape(ctx, value, [batch, sequence, heads, headDim]), 1, 2),
      [batch * heads, sequence, headDim],
    );
    return [
      rope(ctx, toHeadMajor(qFlat), headDim, 0, theta),
      rope(ctx, toHeadMajor(kFlat), headDim, 0, theta),
      toHeadMajor(vFlat),
    ];
  }

  const { cos, sin, negSin } = ropeTables(sequence, headDim, theta, 0);
  const [qData, kData, vData] = B.qkvHeadMajorRope(
    qkv.data,
    cos,
    sin,
    batch,
    sequence,
    heads,
    headDim,
  );
  const branch = (data: TensorData, which: 0 | 1 | 2): Variable => record(
    ctx,
    data,
    [qkv],
    (grad, backwardBackend) => {
      if (!backwardBackend.qkvHeadMajorRopeBackward) {
        throw new Error("qkvHeadMajorRope backward hook disappeared after forward");
      }
      return [backwardBackend.qkvHeadMajorRopeBackward(
        grad,
        cos,
        negSin,
        batch,
        sequence,
        heads,
        headDim,
        which,
      )];
    },
  );
  return [branch(qData, 0), branch(kData, 1), branch(vData, 2)];
}

export function dropout(ctx: Ctx, a: Variable, p: number, training: boolean): Variable {
  if (!training || p === 0) return a;
  const aData = a.data;
  const size = shapeSize(aData.shape);

  let mask: TensorData;
  if (ctx.backend.dropoutMask && ctx.dropoutRng) {
    // GPU-native mask generation — no CPU→GPU transfer
    const params = ctx.dropoutRng.nextMaskParams();
    mask = ctx.backend.dropoutMask(aData.shape, params.seed, params.counter, p);
  } else {
    // CPU mask generation
    const maskArr = ctx.dropoutRng
      ? ctx.dropoutRng.nextMask(size, p)
      : (() => {
          const m = new Float32Array(size);
          const scaleVal = 1 / (1 - p);
          for (let i = 0; i < size; i++) m[i] = Math.random() > p ? scaleVal : 0;
          return m;
        })();
    const cpuMask: TensorData = { shape: [...aData.shape], dtype: aData.dtype, data: maskArr };
    mask = ctx.backend.clone(cpuMask);
  }

  let maskLive: TensorData | null = mask;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (!maskLive) return;
    if (release) release(maskLive);
    maskLive = null;
  };

  const out = ctx.backend.mul(aData, mask);
  return record(ctx, out, [a], (g, B) => {
    const m = maskLive ?? mask;
    return [B.mul(g, m)];
  }, cleanup);
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

  let mask: TensorData;
  if (ctx.backend.dropoutMask && ctx.dropoutRng) {
    // GPU-native mask generation — no CPU→GPU transfer
    const params = ctx.dropoutRng.nextMaskParams();
    mask = ctx.backend.dropoutMask(projData.shape, params.seed, params.counter, p);
  } else {
    // CPU mask generation
    const maskArr = ctx.dropoutRng
      ? ctx.dropoutRng.nextMask(size, p)
      : (() => {
          const m = new Float32Array(size);
          const scaleVal = 1 / (1 - p);
          for (let i = 0; i < size; i++) m[i] = Math.random() > p ? scaleVal : 0;
          return m;
        })();
    const cpuMask: TensorData = { shape: [...projData.shape], dtype: projData.dtype, data: maskArr };
    mask = ctx.backend.clone(cpuMask);
  }

  let maskLive: TensorData | null = mask;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (!maskLive) return;
    if (release) release(maskLive);
    maskLive = null;
  };

  // Use fused kernel if backend supports it
  if (ctx.backend.residualDropoutAdd) {
    const out = ctx.backend.residualDropoutAdd(resData, projData, mask);
    return record(ctx, out, [residual, projected], (g, B, release) => {
      const m = maskLive ?? mask;
      // grad_residual = upstream_grad (pass-through via broadcast reduction)
      const ga = reduceBroadcast(B, g, resData.shape, release);
      // grad_projected = upstream_grad * mask
      const gb = B.mul(g, m);
      return [ga, gb];
    }, cleanup);
  }

  // Fallback: separate ops
  const dropOut = ctx.backend.mul(projData, mask);
  let dropOutLive: TensorData | null = dropOut;
  const fallbackCleanup = (release?: (td: TensorData) => void): void => {
    if (dropOutLive && release) release(dropOutLive);
    dropOutLive = null;
    cleanup(release);
  };
  const out = ctx.backend.add(resData, dropOut);
  return record(ctx, out, [residual, projected], (g, B, release) => {
    const m = maskLive ?? mask;
    const ga = reduceBroadcast(B, g, resData.shape, release);
    const gb = B.mul(g, m);
    if (dropOutLive && release) release(dropOutLive);
    dropOutLive = null;
    return [ga, gb];
  }, fallbackCleanup);
}

/**
 * Residual addition plus the immediately following RMSNorm. When dropout is
 * inactive and the backend exposes the fused two-output primitive, this
 * records the same autograd graph as add() followed by rmsNorm() while issuing
 * one backend operation. With active dropout (or an unsupported backend), the
 * ordinary compositional path remains authoritative.
 */
export function residualDropoutAddRmsNorm(
  ctx: Ctx,
  residual: Variable,
  projected: Variable,
  weight: Variable,
  eps: number,
  p: number,
  training: boolean,
): { residual: Variable; normalized: Variable } {
  if ((training && p !== 0) || !ctx.backend.residualAddRmsNorm) {
    const residualOut = residualDropoutAdd(ctx, residual, projected, p, training);
    return {
      residual: residualOut,
      normalized: rmsNorm(ctx, residualOut, weight, eps),
    };
  }

  const residualData = residual.data;
  const projectedData = projected.data;
  const weightData = weight.data;
  const fused = ctx.backend.residualAddRmsNorm(residualData, projectedData, weightData, eps);

  // Record the residual addition first. Its output remains a real graph node:
  // one downstream path carries the residual stream and the other passes
  // through RMSNorm. Reverse traversal therefore accumulates both gradients
  // before sending them to the original residual and projection.
  const residualOut = record(
    ctx,
    fused.residual,
    [residual, projected],
    (g, B, release) => [
      reduceBroadcast(B, g, residualData.shape, release),
      reduceBroadcast(B, g, projectedData.shape, release),
    ],
  );
  const normalized = record(
    ctx,
    fused.normalized,
    [residualOut, weight],
    (g, B) => rmsNormBackwardData(B, fused.residual, weightData, g, eps),
  );
  return { residual: residualOut, normalized };
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

export function crossEntropy(
  ctx: Ctx,
  logits: Variable,
  targets: TensorData,
  training = false,
): Variable {
  const logitsData = logits.data;
  const fused = training ? ctx.backend.crossEntropyForwardBackward?.(logitsData, targets) ?? null : null;
  let cachedGrad: TensorData | null = fused?.gradLogits ?? null;
  let targetsLive: TensorData | null = targets;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (release) {
      if (targetsLive) release(targetsLive);
      if (cachedGrad) release(cachedGrad);
    }
    targetsLive = null;
    cachedGrad = null;
  };
  return record(ctx, fused?.loss ?? ctx.backend.crossEntropy(logitsData, targets), [logits], (g, B, release) => {
    if (cachedGrad) {
      const rawGrad = cachedGrad;
      cachedGrad = null; // ownership transfers to the returned gradient path
      const upstream = (g.data as Float32Array)[0];
      if (upstream === 1) return [rawGrad];
      const scaled = B.scale(rawGrad, upstream);
      if (release) release(rawGrad);
      return [scaled];
    }
    const tgt = targetsLive ?? targets;
    if (B.crossEntropyBackward) return [B.crossEntropyBackward(logitsData, tgt, g)];
    // CPU fallback: (softmax(logits) - one_hot(targets)) * gScalar / N
    const probs = B.softmax(logitsData, -1);
    const [N, C] = logitsData.shape;
    const gScalar = (g.data as Float32Array)[0];
    const oneHotArr = new Float32Array(N * C);
    for (let i = 0; i < N; i++) oneHotArr[tgt.data[i] + i * C] = 1.0;
    const oneHot: TensorData = { shape: [N, C], dtype: logitsData.dtype, data: oneHotArr };
    const diff = B.sub(probs, oneHot);
    const result = B.scale(diff, gScalar / N);
    if (release) {
      release(oneHot);
      release(probs);
      release(diff);
    }
    return [result];
  }, cleanup);
}

/**
 * Masked (assistant-only SFT) cross-entropy.
 *
 * Forward:  loss = sum_i(ce_i * mask_i) / max(sum_i mask_i, 1)
 * Backward: dLogits[i,c] = (softmax(logits)[i,c] - 1{c==t_i}) * mask_i * g
 *                          / max(sum_i mask_i, 1)
 *
 * `mask` is a fixed (non-differentiable) [N] f32 tensor; only `logits` gets a
 * gradient (targets/mask are data, like `targets` in `crossEntropy`). Rows with
 * mask 0 contribute nothing to the loss AND get an exactly-zero gradient — the
 * property the SFT loss-masking test asserts.
 */
export function crossEntropyMasked(
  ctx: Ctx,
  logits: Variable,
  targets: TensorData,
  mask: TensorData,
  training = false,
): Variable {
  const logitsData = logits.data;
  const B = ctx.backend;
  if (!B.crossEntropyMasked) throw new Error("crossEntropyMasked requires a backend with crossEntropyMasked");
  const fused = training ? B.crossEntropyMaskedForwardBackward?.(logitsData, targets, mask) ?? null : null;
  let cachedGrad: TensorData | null = fused?.gradLogits ?? null;
  let targetsLive: TensorData | null = targets;
  let maskLive: TensorData | null = mask;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (release) {
      if (targetsLive) release(targetsLive);
      if (maskLive) release(maskLive);
      if (cachedGrad) release(cachedGrad);
    }
    targetsLive = null;
    maskLive = null;
    cachedGrad = null;
  };
  return record(ctx, fused?.loss ?? B.crossEntropyMasked(logitsData, targets, mask), [logits], (g, Bk, release) => {
    if (cachedGrad) {
      const rawGrad = cachedGrad;
      cachedGrad = null;
      const upstream = (g.data as Float32Array)[0];
      if (upstream === 1) return [rawGrad];
      const scaled = Bk.scale(rawGrad, upstream);
      if (release) release(rawGrad);
      return [scaled];
    }
    const tgt = targetsLive ?? targets;
    const msk = maskLive ?? mask;
    if (Bk.crossEntropyMaskedBackward) return [Bk.crossEntropyMaskedBackward(logitsData, tgt, msk, g)];
    // CPU fallback: (softmax(logits) - one_hot(targets)) * mask[row] * g / max(sum(mask),1)
    const probs = Bk.softmax(logitsData, -1);
    const probsArr = probs.data as Float32Array;
    const maskArr = msk.data as Float32Array;
    const tgtArr = tgt.data;
    const [N, C] = logitsData.shape;
    const gScalar = (g.data as Float32Array)[0];
    let sumMask = 0;
    for (let i = 0; i < N; i++) sumMask += maskArr[i];
    const scaleVal = gScalar / Math.max(sumMask, 1);
    const out = new Float32Array(N * C);
    for (let i = 0; i < N; i++) {
      const m = maskArr[i];
      if (m === 0) continue; // leave this row's grad exactly zero
      const off = i * C;
      const t = tgtArr[i];
      for (let c = 0; c < C; c++) {
        out[off + c] = (probsArr[off + c] - (c === t ? 1 : 0)) * m * scaleVal;
      }
    }
    const result: TensorData = { shape: [...logitsData.shape], dtype: logitsData.dtype, data: out };
    if (release) release(probs);
    return [result];
  }, cleanup);
}

/**
 * Masked token-unlikelihood for model-generated negative trajectories.
 *
 * Forward:  loss = sum_i(-log(max(1-p(t_i),epsilon)) * mask_i)
 *                  / max(sum_i mask_i, 1)
 * Backward: dLogits[i,c] = p_bad/max(1-p_bad,epsilon)
 *                           * (1{c==t_i} - p_c) * mask_i * g
 *                           / max(sum_i mask_i, 1)
 *
 * The epsilon clamp is used as a stable denominator in backward, matching the
 * declared RCR-UL experiment contract. Targets and mask are fixed data; only
 * logits receive gradients. Mask-zero rows remain bit-exactly zero.
 */
export function crossEntropyUnlikelihoodMasked(
  ctx: Ctx,
  logits: Variable,
  targets: TensorData,
  mask: TensorData,
  epsilon = 1e-6,
): Variable {
  if (!(epsilon > 0 && epsilon <= 1)) {
    throw new Error(`crossEntropyUnlikelihoodMasked epsilon must be in (0,1], got ${epsilon}`);
  }
  const logitsData = logits.data;
  const B = ctx.backend;
  if (!B.crossEntropyUnlikelihoodMasked) {
    throw new Error("crossEntropyUnlikelihoodMasked requires a backend with crossEntropyUnlikelihoodMasked");
  }
  let targetsLive: TensorData | null = targets;
  let maskLive: TensorData | null = mask;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (release) {
      if (targetsLive) release(targetsLive);
      if (maskLive) release(maskLive);
    }
    targetsLive = null;
    maskLive = null;
  };
  return record(
    ctx,
    B.crossEntropyUnlikelihoodMasked(logitsData, targets, mask, epsilon),
    [logits],
    (g, Bk, release) => {
      const tgt = targetsLive ?? targets;
      const msk = maskLive ?? mask;
      if (Bk.crossEntropyUnlikelihoodMaskedBackward) {
        return [Bk.crossEntropyUnlikelihoodMaskedBackward(logitsData, tgt, msk, g, epsilon)];
      }

      const probs = Bk.softmax(logitsData, -1);
      const probsArr = probs.data as Float32Array;
      const maskArr = msk.data as Float32Array;
      const tgtArr = tgt.data;
      const [N, C] = logitsData.shape;
      const gScalar = (g.data as Float32Array)[0];
      let sumMask = 0;
      for (let i = 0; i < N; i++) sumMask += maskArr[i];
      const normalizedUpstream = gScalar / Math.max(sumMask, 1);
      const out = new Float32Array(N * C);
      for (let i = 0; i < N; i++) {
        const m = maskArr[i];
        if (m === 0) continue;
        const off = i * C;
        const t = tgtArr[i];
        const pBad = probsArr[off + t];
        const ratio = pBad / Math.max(1 - pBad, epsilon);
        const rowScale = ratio * m * normalizedUpstream;
        for (let c = 0; c < C; c++) {
          out[off + c] = ((c === t ? 1 : 0) - probsArr[off + c]) * rowScale;
        }
      }
      const result: TensorData = { shape: [...logitsData.shape], dtype: logitsData.dtype, data: out };
      if (release) release(probs);
      return [result];
    },
    cleanup,
  );
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
  let lseForBackward: TensorData | null = lse;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (!lseForBackward) return;
    if (release) release(lseForBackward);
    lseForBackward = null;
  };

  return record(ctx, output, [q, k, v], (g, B2, _release, needsGrad) => {
    if (!B2.flashAttentionBackward) throw new Error("flashAttentionBackward requires GPU backend");
    if (!lseForBackward) throw new Error("flashAttention backward missing LSE buffer");

    const { dQ, dK, dV } = B2.flashAttentionBackward(
      q.data, k.data, v.data, output, g, lseForBackward, T, scale, softCap,
    );

    return [
      (!needsGrad || needsGrad[0]) ? dQ : null as any,
      (!needsGrad || needsGrad[1]) ? dK : null as any,
      (!needsGrad || needsGrad[2]) ? dV : null as any,
    ];
  }, cleanup);
}

/**
 * One-tape-entry form of grouped-QKV layout/RoPE followed by flash attention.
 * Forward still uses the backend's two exact physical operations. Backward can
 * consume dQ, dK, and dV together and write the complete grouped QKV gradient
 * once, avoiding three zero-padded branch tensors and their accumulation.
 */
export function qkvFlashAttention(
  ctx: Ctx,
  qkv: Variable,
  batch: number,
  sequence: number,
  heads: number,
  headDim: number,
  theta: number,
  scaleValue: number,
  softCapValue: number,
): Variable {
  const B = ctx.backend;
  if (!B.qkvHeadMajorRope
    || !B.qkvHeadMajorRopeBackwardCombined
    || !B.flashAttention
    || !B.flashAttentionBackward) {
    const [q, k, v] = qkvHeadMajorRope(
      ctx, qkv, batch, sequence, heads, headDim, theta,
    );
    return flashAttention(ctx, q, k, v, sequence, scaleValue, softCapValue);
  }

  const { cos, sin, negSin } = ropeTables(sequence, headDim, theta, 0);
  const [qData, kData, vData] = B.qkvHeadMajorRope(
    qkv.data, cos, sin, batch, sequence, heads, headDim,
  );
  const { output, lse } = B.flashAttention(
    qData, kData, vData, sequence, scaleValue, softCapValue,
  );
  let qForBackward: TensorData | null = qData;
  let kForBackward: TensorData | null = kData;
  let vForBackward: TensorData | null = vData;
  let lseForBackward: TensorData | null = lse;
  const cleanup = (release?: (td: TensorData) => void): void => {
    if (release) {
      if (qForBackward) release(qForBackward);
      if (kForBackward) release(kForBackward);
      if (vForBackward) release(vForBackward);
      if (lseForBackward) release(lseForBackward);
    }
    qForBackward = null;
    kForBackward = null;
    vForBackward = null;
    lseForBackward = null;
  };

  return record(ctx, output, [qkv], (grad, backwardBackend, release) => {
    if (!backwardBackend.flashAttentionBackward
      || !backwardBackend.qkvHeadMajorRopeBackwardCombined) {
      throw new Error("qkvFlashAttention backward hooks disappeared after forward");
    }
    if (!qForBackward || !kForBackward || !vForBackward || !lseForBackward) {
      throw new Error("qkvFlashAttention backward state was released before use");
    }
    const { dQ, dK, dV } = backwardBackend.flashAttentionBackward(
      qForBackward,
      kForBackward,
      vForBackward,
      output,
      grad,
      lseForBackward,
      sequence,
      scaleValue,
      softCapValue,
    );
    const grouped = backwardBackend.qkvHeadMajorRopeBackwardCombined(
      dQ,
      dK,
      dV,
      cos,
      negSin,
      batch,
      sequence,
      heads,
      headDim,
    );
    if (release) {
      release(dQ);
      release(dK);
      release(dV);
    }
    return [grouped];
  }, cleanup);
}

// ── Slice ──────────────────────────────────────────────────────────────────

/** Slice a tensor: out = a[starts:ends] along each dimension. */
export function slice(ctx: Ctx, a: Variable, starts: number[], ends: number[]): Variable {
  const origShape = [...a.data.shape];
  return record(ctx, ctx.backend.slice(a.data, starts, ends), [a], (g, B, release) => {
    // Fast path: use GPU scatterSlice if backend supports it
    if (B.scatterSlice) {
      return [B.scatterSlice(g, origShape, starts, ends)];
    }

    // Fallback: pad gradient with zeros using cat to reconstruct original shape.
    const ndim = origShape.length;
    let padded: TensorData = g;
    for (let d = ndim - 1; d >= 0; d--) {
      if (starts[d] === 0 && ends[d] === origShape[d]) continue;
      const chunks: TensorData[] = [];
      if (starts[d] > 0) {
        const zShape = [...padded.shape];
        zShape[d] = starts[d];
        chunks.push(B.zeros(zShape, padded.dtype));
      }
      chunks.push(padded);
      if (ends[d] < origShape[d]) {
        const zShape = [...padded.shape];
        zShape[d] = origShape[d] - ends[d];
        chunks.push(B.zeros(zShape, padded.dtype));
      }
      const old = padded;
      padded = B.cat(chunks, d);
      if (release) {
        for (const c of chunks) { if (c !== g && c !== old) release(c); }
        if (old !== g) release(old);
      }
    }
    return [padded];
  });
}

/**
 * Fused 3-way column slice: split [rows, 3*D] into 3 × [rows, D].
 * Single GPU dispatch instead of 3 separate slice ops.
 * Backward: 3 separate scatterSlice operations (same dispatch count as unfused).
 */
export function sliceQkv(ctx: Ctx, a: Variable): [Variable, Variable, Variable] {
  const B = ctx.backend;
  const data = a.data;
  const [rows, cols3] = data.shape;
  const D = cols3 / 3;

  if (B.sliceQkv) {
    const [qData, kData, vData] = B.sliceQkv(data);
    const origShape = [...data.shape];
    const q = record(ctx, qData, [a], (g, B2, release) => {
      if (B2.scatterSlice) return [B2.scatterSlice(g, origShape, [0, 0], [rows, D])];
      const z0 = B2.zeros([rows, D], g.dtype);
      const z1 = B2.zeros([rows, D], g.dtype);
      const res = B2.cat([g, z0, z1], 1);
      if (release) { release(z0); release(z1); }
      return [res];
    });
    const k = record(ctx, kData, [a], (g, B2, release) => {
      if (B2.scatterSlice) return [B2.scatterSlice(g, origShape, [0, D], [rows, 2 * D])];
      const z0 = B2.zeros([rows, D], g.dtype);
      const z1 = B2.zeros([rows, D], g.dtype);
      const res = B2.cat([z0, g, z1], 1);
      if (release) { release(z0); release(z1); }
      return [res];
    });
    const v = record(ctx, vData, [a], (g, B2, release) => {
      if (B2.scatterSlice) return [B2.scatterSlice(g, origShape, [0, 2 * D], [rows, 3 * D])];
      const z0 = B2.zeros([rows, D], g.dtype);
      const z1 = B2.zeros([rows, D], g.dtype);
      const res = B2.cat([z0, z1, g], 1);
      if (release) { release(z0); release(z1); }
      return [res];
    });
    return [q, k, v];
  }

  // Fallback to 3 separate slices
  return [
    slice(ctx, a, [0, 0], [rows, D]),
    slice(ctx, a, [0, D], [rows, 2 * D]),
    slice(ctx, a, [0, 2 * D], [rows, 3 * D]),
  ];
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
function reduceBroadcast(B: Backend, grad: TensorData, targetShape: Shape, release?: (td: TensorData) => void): TensorData {
  if (arraysEqual(grad.shape, targetShape)) return grad;
  // Scalar target
  if (targetShape.length === 0 || (targetShape.length === 1 && targetShape[0] === 1 && grad.shape.length > 1)) {
    return B.sum(grad);
  }
  let result = grad;
  // Sum over leading dims that were broadcast
  while (result.shape.length > targetShape.length) {
    const prev = result;
    result = B.sum(result, 0);
    if (release && prev !== grad) release(prev);
  }
  // Sum over dims that are 1 in target
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && result.shape[i] !== 1) {
      const prev = result;
      result = B.sum(result, i, true);
      if (release && prev !== grad) release(prev);
    }
  }
  return result;
}

/** Broadcast a (possibly reduced) tensor to a target shape. */
function broadcastTo(B: Backend, t: TensorData, targetShape: Shape): TensorData {
  if (arraysEqual(t.shape, targetShape)) return t;
  // Use GPU broadcast if available (avoids CPU readback + O(N) copy)
  if (B.broadcast) return B.broadcast(t, targetShape);
  // CPU fallback — stride-based for correct non-trailing broadcasts
  const size = shapeSize(targetShape);
  const srcSize = shapeSize(t.shape);
  const out = new Float32Array(size);
  const src = t.data as Float32Array;
  if (srcSize === 1) {
    out.fill(src[0]);
  } else {
    const strides = broadcastStrides(t.shape, targetShape);
    const ndim = targetShape.length;
    for (let i = 0; i < size; i++) {
      let srcIdx = 0;
      let remainder = i;
      for (let d = ndim - 1; d >= 0; d--) {
        const coord = remainder % targetShape[d];
        remainder = (remainder - coord) / targetShape[d];
        srcIdx += coord * strides[d];
      }
      out[i] = src[srcIdx];
    }
  }
  return { shape: targetShape, dtype: t.dtype, data: out };
}

function arraysEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ── Mixed precision ──────────────────────────────────────────────────────

/**
 * Cast activation to f16 for storage, reducing VRAM by 50%.
 * Forward: f32 → f16. Backward: gradient cast f16 → f32 (or passed through as-is if already f32).
 * No-op if backend doesn't support castDtype.
 */
export function castToF16(ctx: Ctx, x: Variable): Variable {
  const B = ctx.backend;
  if (!B.castDtype) return x; // no-op on backends without f16 support
  const f16Data = B.castDtype(x.data, "f16");
  return record(ctx, f16Data, [x], (g, backend) => {
    // Gradient is f32 (backward always computes in f32)
    // If it's somehow f16, cast back to f32
    if (g.dtype === "f16" && backend.castDtype) {
      return [backend.castDtype(g, "f32")];
    }
    return [g];
  });
}

/**
 * Cast activation from f16 back to f32 for computation.
 * Forward: f16 → f32. Backward: gradient stays f32 (no cast needed).
 */
export function castToF32(ctx: Ctx, x: Variable): Variable {
  const B = ctx.backend;
  if (x.data.dtype === "f32") return x; // already f32
  if (!B.castDtype) return x;
  const f32Data = B.castDtype(x.data, "f32");
  return record(ctx, f32Data, [x], (g, backend) => {
    // Backward: cast gradient to f16 to match input dtype
    if (backend.castDtype) {
      return [backend.castDtype(g, "f16")];
    }
    return [g];
  });
}
