/**
 * nativeBackend.ts — the Backend interface on top of our own GPU stack.
 *
 * WHAT: each operation allocates its result from the pool, launches the kernel
 * that computes it, and returns a TensorData whose `data` IS the device
 * memory -- not a copy of it.
 *
 * WHY IT DOES NOT INHERIT FROM THE VULKAN BACKEND: that one is 8,000 lines of
 * pipeline and descriptor management, none of which applies. Sharing a base
 * class would mean carrying its lifecycle into a stack that has a different
 * one, and it would blur which path a given operation actually took -- the
 * exact ambiguity that let a test suite pass while silently running on the CPU.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not fall back. An operation without
 * a kernel throws by name. That is louder than degrading to the host and it is
 * the only way the coverage gap stays visible: a backend that quietly computes
 * the right answer somewhere else is indistinguishable from one that works.
 *
 * WHAT IS NOT IMPLEMENTED YET is listed in `unsupported()` below rather than
 * scattered through the file, so the remaining surface can be read in one
 * place.
 */
import type {
  Backend,
  Dtype,
  Shape,
  TensorData,
} from "@alpha/core";
import { shapeSize } from "@alpha/core";
import { NativeBuffer, nativeAddon, type NativeAddon } from "./nativeDevice.js";

/**
 * A tensor that lives on the device.
 *
 * `data` is the zero-copy view, so reading it is reading device memory and
 * writing it is writing device memory. The buffer is carried alongside so the
 * next operation can pass the handle down without a lookup table.
 */
export interface NativeTensor extends TensorData {
  readonly buffer: NativeBuffer;
}

function isNative(t: TensorData): t is NativeTensor {
  return (t as NativeTensor).buffer !== undefined;
}

export class NativeHeliosBackend implements Backend {
  readonly name = "helios-native";
  private readonly hl: NativeAddon;

  /**
   * Scratch for two-level reductions.
   *
   * One buffer, reused, because reductions are serialised on the channel
   * anyway -- two cannot be in flight, so two scratch buffers would be idle
   * memory. Sized for the largest partial count a reduction can produce.
   */
  private readonly scratch: NativeBuffer;

  constructor(deviceIndex = 0) {
    this.hl = nativeAddon(deviceIndex);
    this.scratch = NativeBuffer.alloc(this.hl, 1024);
  }

  /** What this backend cannot do yet, named rather than silently worked around. */
  private unsupported(op: string): never {
    throw new Error(
      `helios-native: ${op} has no kernel yet. Implemented: add sub mul div ` +
        `neg relu gelu silu exp log sqrt scale clamp matmul sum mean rmsNorm ` +
        `layerNorm softmax embedding crossEntropy transpose zeros ones full ` +
        `fromArray reshape clone slice causalMask maskedFill equal allClose.`,
    );
  }

  private make(shape: Shape, dtype: Dtype = "f32"): NativeTensor {
    const n = shapeSize(shape);
    const buffer = NativeBuffer.alloc(this.hl, n);
    return { shape, dtype, data: buffer.floats.subarray(0, n), buffer };
  }

  /** Upload a host tensor, or pass a device one straight through. */
  private device(t: TensorData): NativeTensor {
    if (isNative(t)) return t;
    const out = this.make(t.shape, "f32");
    const src = t.data as ArrayLike<number>;
    for (let i = 0; i < src.length; i++) out.buffer.floats[i] = src[i];
    return out;
  }

  private check(ok: boolean, op: string): void {
    if (!ok) throw new Error(`helios-native: ${op} failed on the device`);
  }

  // ── creation ─────────────────────────────────────────────────────────────

  zeros(shape: Shape, dtype: Dtype = "f32"): TensorData {
    const t = this.make(shape, dtype);
    t.buffer.floats.fill(0, 0, shapeSize(shape));
    return t;
  }

  ones(shape: Shape, dtype: Dtype = "f32"): TensorData {
    return this.full(shape, 1, dtype);
  }

  full(shape: Shape, value: number, dtype: Dtype = "f32"): TensorData {
    const t = this.make(shape, dtype);
    t.buffer.floats.fill(value, 0, shapeSize(shape));
    return t;
  }

  randn(shape: Shape, dtype: Dtype = "f32"): TensorData {
    /* Host-side, because a normal deviate needs a Box-Muller pair and there is
     * no kernel for it. Initialisation happens once, so this costs nothing per
     * step -- unlike the operations above, which are on the hot path. */
    const t = this.make(shape, dtype);
    const n = shapeSize(shape);
    for (let i = 0; i < n; i++) {
      const u = Math.random() || Number.EPSILON;
      t.buffer.floats[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
    }
    return t;
  }

  fromArray(data: number[], shape: Shape, dtype: Dtype = "f32"): TensorData {
    const t = this.make(shape, dtype);
    for (let i = 0; i < data.length; i++) t.buffer.floats[i] = data[i];
    return t;
  }

  // ── element-wise ─────────────────────────────────────────────────────────

  private binary(name: string, opId: number, a: TensorData, b: TensorData): TensorData {
    const da = this.device(a), db = this.device(b);
    const n = shapeSize(a.shape);
    const out = this.make(a.shape, "f32");
    this.check(
      this.hl.elementwise(opId, out.buffer.handle, da.buffer.handle,
                          db.buffer.handle, n, 0, 0, 0, 0, 0, 0, 0),
      name,
    );
    return out;
  }

  private unary(name: string, opId: number, a: TensorData, ...scalars: number[]): TensorData {
    const da = this.device(a);
    const n = shapeSize(a.shape);
    const out = this.make(a.shape, "f32");
    const s = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < scalars.length; i++) s[i] = scalars[i];
    this.check(
      this.hl.elementwise(opId, out.buffer.handle, da.buffer.handle,
                          da.buffer.handle, n, s[0], s[1], s[2], s[3], s[4],
                          s[5], scalars.length),
      name,
    );
    return out;
  }

  add(a: TensorData, b: TensorData): TensorData { return this.binary("add", this.hl.op.add, a, b); }
  sub(a: TensorData, b: TensorData): TensorData { return this.binary("sub", this.hl.op.sub, a, b); }
  mul(a: TensorData, b: TensorData): TensorData { return this.binary("mul", this.hl.op.mul, a, b); }
  div(a: TensorData, b: TensorData): TensorData { return this.binary("div", this.hl.op.div, a, b); }

  neg(a: TensorData): TensorData { return this.unary("neg", this.hl.op.neg, a); }
  relu(a: TensorData): TensorData { return this.unary("relu", this.hl.op.relu, a); }
  exp(a: TensorData): TensorData { return this.unary("exp", this.hl.op.exp, a, Math.LOG2E); }
  log(a: TensorData): TensorData { return this.unary("log", this.hl.op.log, a, Math.LN2); }
  sqrt(a: TensorData): TensorData { return this.unary("sqrt", this.hl.op.sqrt, a); }
  scale(a: TensorData, s: number): TensorData { return this.unary("scale", this.hl.op.scale, a, s); }
  clamp(a: TensorData, lo: number, hi: number): TensorData {
    return this.unary("clamp", this.hl.op.clamp, a, lo, hi);
  }
  gelu(a: TensorData): TensorData {
    /* The kernel wants the folded constants, not the textbook ones -- see
     * elementwise_ops.c for why the tanh becomes one reciprocal. */
    return this.unary("gelu", this.hl.op.gelu, a,
                      2 * 0.7978845608028654 * Math.LOG2E, 0.044715, 1, 1);
  }
  silu(a: TensorData): TensorData {
    return this.unary("silu", this.hl.op.silu, a, -Math.LOG2E, 1);
  }
  pow(): TensorData { return this.unsupported("pow"); }

  // ── linear algebra and reduction ─────────────────────────────────────────

  matmul(a: TensorData, b: TensorData): TensorData {
    const M = a.shape[a.shape.length - 2] ?? 1;
    const K = a.shape[a.shape.length - 1] ?? 1;
    const N = b.shape[b.shape.length - 1] ?? 1;
    const da = this.device(a), db = this.device(b);
    const out = this.make([M, N], "f32");
    this.check(this.hl.matmul(out.buffer.handle, da.buffer.handle,
                              db.buffer.handle, M, N, K), "matmul");
    return out;
  }

  private reduceAll(name: string, mean: boolean, a: TensorData): TensorData {
    const da = this.device(a);
    const n = shapeSize(a.shape);
    const out = this.make([1], "f32");
    /* The scratch beyond the partial count must be zero: the second pass
     * reduces a power-of-two width and zero is the identity for a sum. */
    this.scratch.floats.fill(0);
    this.check(this.hl.reduce(mean ? 1 : 0, out.buffer.handle, da.buffer.handle,
                              this.scratch.handle, n), name);
    return out;
  }

  sum(a: TensorData, axis?: number): TensorData {
    if (axis !== undefined) this.unsupported("sum along an axis");
    return this.reduceAll("sum", false, a);
  }

  mean(a: TensorData, axis?: number): TensorData {
    if (axis !== undefined) this.unsupported("mean along an axis");
    return this.reduceAll("mean", true, a);
  }

  // ── nn ───────────────────────────────────────────────────────────────────

  private normalized(name: string, opId: number, x: TensorData, eps: number): NativeTensor {
    const width = x.shape[x.shape.length - 1] ?? 1;
    const rows = shapeSize(x.shape) / width;
    const dx = this.device(x);
    const out = this.make(x.shape, "f32");
    this.check(this.hl.normalize(opId, out.buffer.handle, dx.buffer.handle,
                                 width, rows, eps), name);
    return out;
  }

  rmsNorm(x: TensorData, weight: TensorData, eps: number): TensorData {
    return this.mul(this.normalized("rmsNorm", this.hl.op.rmsNorm, x, eps), weight);
  }

  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    const n = this.normalized("layerNorm", this.hl.op.layerNorm, x, eps);
    return this.add(this.mul(n, weight), bias);
  }

  softmax(a: TensorData, axis?: number): TensorData {
    if (axis !== undefined && axis !== a.shape.length - 1)
      this.unsupported("softmax over a non-final axis");
    return this.normalized("softmax", this.hl.op.softmax, a, 0);
  }

  logSoftmax(): TensorData { return this.unsupported("logSoftmax"); }

  embedding(weight: TensorData, indices: TensorData): TensorData {
    const dim = weight.shape[weight.shape.length - 1] ?? 1;
    const tokens = shapeSize(indices.shape);
    const dw = this.device(weight);
    /* Indices are integers; they go into device memory as raw words, not as
     * floats that happen to be whole numbers -- a float bit pattern used as an
     * index addresses somewhere absurd. */
    const ids = this.make([tokens], "i32");
    const src = indices.data as ArrayLike<number>;
    for (let i = 0; i < tokens; i++) ids.buffer.ints[i] = src[i] | 0;
    const out = this.make([tokens, dim], "f32");
    this.check(this.hl.embedding(out.buffer.handle, dw.buffer.handle,
                                 ids.buffer.handle, tokens, dim), "embedding");
    ids.buffer.release(this.hl);
    return out;
  }

  crossEntropy(logits: TensorData, targets: TensorData): TensorData {
    const classes = logits.shape[logits.shape.length - 1] ?? 1;
    const rows = shapeSize(logits.shape) / classes;
    const dl = this.device(logits);
    const ids = this.make([rows], "i32");
    const src = targets.data as ArrayLike<number>;
    for (let i = 0; i < rows; i++) ids.buffer.ints[i] = src[i] | 0;
    const out = this.make([rows], "f32");
    this.check(this.hl.crossEntropy(out.buffer.handle, dl.buffer.handle,
                                    ids.buffer.handle, rows, classes),
               "crossEntropy");
    ids.buffer.release(this.hl);
    return out;
  }

  transpose(a: TensorData): TensorData {
    const rows = a.shape[a.shape.length - 2] ?? 1;
    const cols = a.shape[a.shape.length - 1] ?? 1;
    const da = this.device(a);
    const out = this.make([cols, rows], "f32");
    this.check(this.hl.transpose(out.buffer.handle, da.buffer.handle, rows, cols),
               "transpose");
    return out;
  }

  // ── shape and comparison ─────────────────────────────────────────────────

  /**
   * A different view of the same memory. No copy, no kernel, no launch.
   *
   * This is the one operation that is free, and it is free because a reshape
   * changes only how the elements are INDEXED and the device stores them
   * contiguously either way. Copying here would be work done to produce a
   * tensor byte-identical to the one that already existed.
   */
  reshape(a: TensorData, shape: Shape): NativeTensor {
    if (shapeSize(shape) !== shapeSize(a.shape))
      throw new Error(
        `helios-native: reshape ${a.shape} -> ${shape} changes the element count`,
      );
    const da = this.device(a);
    const view: NativeTensor = {
      shape,
      dtype: da.dtype,
      data: da.data,
      buffer: da.buffer,
    };
    return view;
  }

  clone(a: TensorData): TensorData {
    return this.unary("clone", this.hl.op.copy, a);
  }

  slice(a: TensorData, starts: number[], ends: number[]): TensorData {
    /* One dimension, because that is what the kernel takes and what every
     * slice reduces to once the shape is flattened. A multi-dimensional slice
     * needs the offset and stride computed from the shapes, which is host
     * arithmetic that has not been written rather than a missing kernel. */
    if (starts.length !== 1 || ends.length !== 1)
      this.unsupported(`slice over ${starts.length} dimensions`);
    const count = ends[0] - starts[0];
    const da = this.device(a);
    const out = this.make([count], "f32");
    this.check(this.hl.slice(out.buffer.handle, da.buffer.handle, count,
                             starts[0], 1), "slice");
    return out;
  }

  causalMask(size: number): TensorData {
    /* The kernel masks an existing tensor rather than producing a bare mask, so
     * the identity it masks is zeros -- giving 0 below the diagonal and -inf
     * above, which is exactly the additive mask attention wants. */
    const zero = this.zeros([size, size]) as NativeTensor;
    const out = this.make([size, size], "f32");
    this.check(this.hl.causalMask(out.buffer.handle, zero.buffer.handle, size,
                                  size), "causalMask");
    zero.buffer.release(this.hl);
    return out;
  }

  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData {
    const dm = this.device(mask);
    const da = this.device(a);
    const out = this.make(a.shape, "f32");
    this.check(this.hl.maskedFill(out.buffer.handle, da.buffer.handle,
                                  dm.buffer.handle, shapeSize(a.shape), value),
               "maskedFill");
    return out;
  }

  /*
   * Comparisons run on the HOST, deliberately.
   *
   * They return a boolean, not a tensor, so the answer has to reach the host
   * anyway -- and a kernel would produce a per-element result that then needs
   * reducing and reading back, which is more launches and more latency to
   * deliver one bit. They are also test-path operations, not step-path ones.
   */
  equal(a: TensorData, b: TensorData): boolean {
    if (shapeSize(a.shape) !== shapeSize(b.shape)) return false;
    const x = this.device(a).data as ArrayLike<number>;
    const y = this.device(b).data as ArrayLike<number>;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  }

  allClose(a: TensorData, b: TensorData, atol = 1e-5, rtol = 1e-5): boolean {
    if (shapeSize(a.shape) !== shapeSize(b.shape)) return false;
    const x = this.device(a).data as ArrayLike<number>;
    const y = this.device(b).data as ArrayLike<number>;
    for (let i = 0; i < x.length; i++) {
      /* Written as !(<=) rather than >, so a NaN fails rather than passing --
       * every comparison with NaN is false, and `>` would accept it. The same
       * mistake was in three of the kernel checkers. */
      if (!(Math.abs(x[i] - y[i]) <= atol + rtol * Math.abs(y[i]))) return false;
    }
    return true;
  }

  cat(): TensorData { return this.unsupported("cat"); }
  argmax(): TensorData { return this.unsupported("argmax"); }
  topk(): { values: TensorData; indices: TensorData } { return this.unsupported("topk"); }
  gather(): TensorData { return this.unsupported("gather"); }

  /** Pool and program statistics, for confirming a step reuses rather than
   * reallocates. `allocations` should stop growing after the first step. */
  stats(): { live: number; pooled: number; allocations: number; programs: number } {
    return this.hl.stats();
  }
}
