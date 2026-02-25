/**
 * Broadcast helpers — shared between CPU ref backend, autograd ops, and GPU fallbacks.
 *
 * These implement NumPy-style broadcasting: shapes are right-aligned, dimensions
 * of size 1 are stretched to match the other operand.
 */
import type { Shape } from "./types.js";

/**
 * Broadcast two shapes and return [resultShape, stridesA, stridesB].
 * Supports: same shape, scalar broadcast, and general N-dimensional broadcast.
 */
export function broadcastShapes(sa: Shape, sb: Shape): [number[], number[], number[]] {
  const ndim = Math.max(sa.length, sb.length);
  const result: number[] = new Array(ndim);
  const padA = ndim - sa.length;
  const padB = ndim - sb.length;

  for (let i = 0; i < ndim; i++) {
    const da = i < padA ? 1 : sa[i - padA];
    const db = i < padB ? 1 : sb[i - padB];
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes [${sa}] and [${sb}]`);
    }
    result[i] = Math.max(da, db);
  }

  // Build strides: if a dimension is 1 (and needs broadcasting), stride = 0.
  const stridesA = new Array(ndim);
  const stridesB = new Array(ndim);
  let strA = 1;
  let strB = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    const da = i < padA ? 1 : sa[i - padA];
    const db = i < padB ? 1 : sb[i - padB];
    stridesA[i] = da === 1 && result[i] !== 1 ? 0 : strA;
    stridesB[i] = db === 1 && result[i] !== 1 ? 0 : strB;
    strA *= da;
    strB *= db;
  }

  return [result, stridesA, stridesB];
}

/**
 * Convert a flat index in the result to flat indices in a and b using broadcast strides.
 */
export function broadcastIndices(
  flatIdx: number,
  resultShape: Shape,
  stridesA: number[],
  stridesB: number[],
): [number, number] {
  const ndim = resultShape.length;
  let idxA = 0;
  let idxB = 0;
  let remainder = flatIdx;
  for (let d = ndim - 1; d >= 0; d--) {
    const coord = remainder % resultShape[d];
    remainder = (remainder - coord) / resultShape[d];
    idxA += coord * stridesA[d];
    idxB += coord * stridesB[d];
  }
  return [idxA, idxB];
}

/**
 * Compute broadcast strides for expanding a source shape into a target shape.
 * Returns stride array where dimensions of size 1 in src have stride 0.
 * Assumes target is a valid broadcast of src (caller must verify).
 */
export function broadcastStrides(srcShape: Shape, targetShape: Shape): number[] {
  const ndim = targetShape.length;
  const pad = ndim - srcShape.length;
  const strides = new Array(ndim);
  let str = 1;
  for (let i = ndim - 1; i >= 0; i--) {
    const srcDim = i < pad ? 1 : srcShape[i - pad];
    strides[i] = srcDim === 1 && targetShape[i] !== 1 ? 0 : str;
    str *= srcDim;
  }
  return strides;
}
