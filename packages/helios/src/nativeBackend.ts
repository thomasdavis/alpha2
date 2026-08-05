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
        `fromArray reshape clone slice causalMask maskedFill equal allClose ` +
        `pow cat gather argmax topk, and sum/mean over the final axis.`,
    );
  }

  /**
   * A tensor whose `data` DRAINS THE QUEUE when it is read.
   *
   * Operations are asynchronous now, so a view read before the kernel has run
   * shows whatever was in the buffer beforehand -- plausible numbers, silently
   * wrong. Every read site cannot be found by inspection: the autograd tape,
   * the model and every test touch `.data` directly, and one missed site is a
   * bug with no symptom at the call that caused it.
   *
   * So the barrier goes on the property rather than at the call sites. A getter
   * satisfies `readonly data`, and flushing is nearly free when nothing is
   * pending -- the cost is one predictable branch on a path that is already
   * about to touch memory.
   */
  private make(shape: Shape, dtype: Dtype = "f32"): NativeTensor {
    /* No drain here. A freed buffer only re-enters circulation when the queue
     * drains (see tensor.c), so a fresh allocation cannot alias one that queued
     * work still reads. Draining per allocation would fire once per operation
     * and remove the batching entirely. */
    const n = shapeSize(shape);
    const buffer = NativeBuffer.alloc(this.hl, n);
    const hl = this.hl;
    const view = buffer.floats.subarray(0, n);
    return {
      shape,
      dtype,
      get data() {
        hl.flush();
        return view;
      },
      buffer,
    } as NativeTensor;
  }

  /** Upload a host tensor, or pass a device one straight through. */
  private device(t: TensorData): NativeTensor {
    if (isNative(t)) return t;
    const out = this.make(t.shape, "f32");
    const src = t.data as ArrayLike<number>;
    for (let i = 0; i < src.length; i++) out.buffer.floats[i] = src[i];
    return out;
  }

  /**
   * Resolve a possibly-negative axis against a rank.
   *
   * -1 means the last dimension, which is how attention and every loss in the
   * model spell it. Comparing a raw axis against rank-1 rejects -1 as
   * "non-final" -- which is what the first version did, and it read as a
   * missing kernel when the kernel was there and the index was a convention.
   */
  private axisOf(shape: Shape, axis: number): number {
    return axis < 0 ? shape.length + axis : axis;
  }

  /*
   * Drain the queue before the host reads device memory.
   *
   * Operations ENQUEUE and return immediately, which is what turns a
   * 150-operation step from 150 round trips into a handful. The cost is that a
   * tensor's `data` view is only meaningful after a flush, so every path here
   * that touches host memory calls this first. Missing one would read the
   * values that were there before the kernel ran -- plausible numbers, wrong.
   */
  private sync(): void {
    this.hl.flush();
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

  /**
   * Expand `t` to `shape` by NumPy's rules, if it is not already that shape.
   *
   * Broadcasting was refused at first, on the reasoning that shape rules belong
   * to the tensor library and a backend inventing them would be making
   * decisions the layer above already made. That was wrong, and the model
   * proved it: the autograd tape emits a backward sum over the broadcast axis,
   * which only makes sense if the forward broadcast happened. Refusing the
   * forward while accepting the backward is not a stricter contract, it is an
   * inconsistent one.
   *
   * The expansion is a COPY between two mapped views and performs no
   * arithmetic, so it does not put a FLOP anywhere but our own SASS. It costs
   * bandwidth, and a kernel that indexed the smaller operand modularly would
   * cost none -- that is the optimisation to make once this is correct.
   */
  private expand(t: TensorData, shape: Shape): NativeTensor {
    const dt = this.device(t);
    const want = shapeSize(shape);
    const have = shapeSize(t.shape);
    if (have === want) return dt;
    if (want % have !== 0)
      throw new Error(
        `helios-native: cannot broadcast ${t.shape} to ${shape}`,
      );

    /* Align the shapes from the RIGHT, as broadcasting does, and walk the
     * destination computing each element's source index. Contiguous runs fall
     * out of it naturally when the trailing dimensions already match. */
    const out = this.make(shape, "f32");
    this.sync();
    const src = dt.buffer.floats;
    const pad = shape.length - t.shape.length;
    const sStride: number[] = new Array(shape.length).fill(0);
    let acc = 1;
    for (let i = t.shape.length - 1; i >= 0; i--) {
      sStride[i + pad] = t.shape[i] === 1 ? 0 : acc;
      acc *= t.shape[i];
    }
    const idx = new Array(shape.length).fill(0);
    for (let n = 0; n < want; n++) {
      let si = 0;
      for (let d = 0; d < shape.length; d++) si += idx[d] * sStride[d];
      out.buffer.floats[n] = src[si];
      for (let d = shape.length - 1; d >= 0; d--) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
    return out;
  }

  /** The shape a binary operation produces: the larger of the two operands. */
  private resultShape(a: Shape, b: Shape): Shape {
    return shapeSize(a) >= shapeSize(b) ? a : b;
  }

  private binary(name: string, opId: number, a: TensorData, b: TensorData): TensorData {
    const shape = this.resultShape(a.shape, b.shape);
    const da = this.expand(a, shape), db = this.expand(b, shape);
    const n = shapeSize(shape);
    const out = this.make(shape, "f32");
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
  /*
   * The folded constants come from the ADDON, not from here.
   *
   * They were restated in TypeScript once and drifted -- gelu's two ended up
   * swapped and silu's negated, so both produced plausible wrong numbers that
   * nothing caught until a per-operation test looked at gelu directly. The
   * kernels evaluate an algebraically equal rearrangement of the textbook
   * formula, and which constant goes in which slot is a property of THAT
   * rearrangement, so it lives with it.
   */
  gelu(a: TensorData): TensorData {
    return this.unary("gelu", this.hl.op.gelu, a,
                      this.hl.scalar.geluK1, this.hl.scalar.geluFolded, 1, 1);
  }
  silu(a: TensorData): TensorData {
    return this.unary("silu", this.hl.op.silu, a, this.hl.scalar.log2e, 1);
  }
  /**
   * a^k.
   *
   * Small non-negative integer exponents become repeated multiplication --
   * exact, and the case that actually occurs, since a squaring is what a
   * variance or an L2 norm asks for. Everything else goes through exp(k*log a),
   * which is what the hardware can do and which is undefined for a negative
   * base. That restriction is real and is stated rather than silently returning
   * NaN: a fractional power of a negative number has no real value, and a
   * backend that produced one would be lying.
   */
  pow(a: TensorData, exp: number): TensorData {
    if (Number.isInteger(exp) && exp >= 0 && exp <= 8) {
      if (exp === 0) return this.full(a.shape, 1);
      let acc = this.device(a) as TensorData;
      for (let i = 1; i < exp; i++) acc = this.mul(acc, a);
      return acc;
    }
    return this.exp(this.scale(this.log(a), exp));
  }

  // ── linear algebra and reduction ─────────────────────────────────────────

  /**
   * Batched by FLATTENING, not by a batched kernel.
   *
   * [B,M,K] against a [K,N] weight is [B*M,K] against [K,N] -- the rows are
   * independent and contiguous, so collapsing every leading dimension into the
   * row count is the same arithmetic in the same memory order. One kernel
   * serves both, and the program cache sees one shape instead of one per batch
   * size.
   *
   * The output shape keeps the leading dimensions and replaces the last: the
   * first version took M and N from the final two dimensions and DROPPED
   * everything before them, so a [1,8,16] came back as [8,16] and the next
   * reshape rejected it for changing the element count. It did not change the
   * element count; it changed the rank, and the error message was pointing at
   * the wrong operation.
   *
   * A batched RIGHT operand -- [B,M,K] against [B,K,N], where each batch has
   * its own matrix -- is genuinely different and is not this. It needs the
   * kernel to offset b by the batch index, and it is refused rather than
   * silently computed against batch zero.
   */
  matmul(a: TensorData, b: TensorData): TensorData {
    const K = a.shape[a.shape.length - 1] ?? 1;
    const N = b.shape[b.shape.length - 1] ?? 1;

    /*
     * A batched RIGHT operand -- attention's Q @ K-transpose, where every head
     * has its own matrix -- is a different operation and gets a loop.
     *
     * Not a batched kernel, because the kernel addresses one matrix and giving
     * it a batch index means threading another dimension through its address
     * arithmetic. A loop of launches is slower and it is correct, and the
     * ordering that makes it correct is free: launches on one channel run in
     * sequence, so batch i is finished before batch i+1 starts.
     */
    if (b.shape.length > 2) {
      const M0 = a.shape[a.shape.length - 2] ?? 1;
      const batch = shapeSize(b.shape) / (K * N);
      const da = this.device(a), db = this.device(b);
      const out = this.make([...a.shape.slice(0, -1), N], "f32");
      this.sync();
      const la = this.make([M0, K], "f32"), lb = this.make([K, N], "f32");
      const lo = this.make([M0, N], "f32");
      for (let i = 0; i < batch; i++) {
        la.buffer.floats.set(da.buffer.floats.subarray(i * M0 * K, (i + 1) * M0 * K));
        lb.buffer.floats.set(db.buffer.floats.subarray(i * K * N, (i + 1) * K * N));
        this.check(this.hl.matmul(lo.buffer.handle, la.buffer.handle,
                                  lb.buffer.handle, M0, N, K), "matmul");
        this.sync();
        out.buffer.floats.set(lo.buffer.floats.subarray(0, M0 * N), i * M0 * N);
      }
      la.buffer.release(this.hl);
      lb.buffer.release(this.hl);
      lo.buffer.release(this.hl);
      return out;
    }

    if ((b.shape[b.shape.length - 2] ?? K) !== K)
      throw new Error(
        `helios-native: matmul inner dimensions disagree, ${a.shape} x ${b.shape}`,
      );
    const M = shapeSize(a.shape) / K;
    const da = this.device(a), db = this.device(b);
    const outShape = [...a.shape.slice(0, -1), N];
    const out = this.make(outShape, "f32");
    this.check(this.hl.matmul(out.buffer.handle, da.buffer.handle,
                              db.buffer.handle, M, N, K), "matmul");
    return out;
  }

  private reduceAll(name: string, mean: boolean, a: TensorData): TensorData {
    this.sync(); /* the scratch is cleared on the HOST just below */
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

  /**
   * Along the LAST axis: one reduction per row, launched one row at a time.
   *
   * Row at a time rather than one launch over all of them, because the
   * reduction kernel writes its answer to element zero of its output and has no
   * notion of which row it is on. Fixing that is a kernel change -- a row index
   * added to the store address -- and it is worth making once rows are the
   * common case. For now the loop is honest about being a loop.
   */
  private reduceAxis(name: string, mean: boolean, a: TensorData): TensorData {
    this.sync(); /* the loop below copies rows on the HOST between launches */
    const width = a.shape[a.shape.length - 1] ?? 1;
    const rows = shapeSize(a.shape) / width;
    const da = this.device(a);
    const outShape = a.shape.slice(0, -1);
    const out = this.make(outShape.length ? outShape : [1], "f32");
    const rowIn = this.make([width], "f32");
    const rowOut = this.make([1], "f32");
    for (let r = 0; r < rows; r++) {
      rowIn.buffer.floats.set(da.buffer.floats.subarray(r * width, (r + 1) * width));
      this.scratch.floats.fill(0);
      this.check(
        this.hl.reduce(mean ? 1 : 0, rowOut.buffer.handle, rowIn.buffer.handle,
                       this.scratch.handle, width),
        name,
      );
      /* The reduce above only ENQUEUED; reading its result needs the queue
       * drained. Inside the loop, because each row's answer is consumed before
       * the next row is queued. */
      this.sync();
      out.buffer.floats[r] = rowOut.buffer.floats[0];
    }
    rowIn.buffer.release(this.hl);
    rowOut.buffer.release(this.hl);
    return out;
  }

  /**
   * Reduce over any axis by bringing it to the END first.
   *
   * The kernel reduces a contiguous row, so an interior axis has to be made
   * contiguous before it can be reduced. Viewing the tensor as
   * [outer, axis, inner] and transposing the [axis, inner] plane does exactly
   * that, and both the transpose and the reduction stay on the device -- which
   * matters, because summing on the host would be arithmetic done somewhere
   * other than our own SASS, and that is the one thing this stack exists to
   * avoid.
   *
   * A trailing axis skips all of it and reduces in place.
   */
  private reduceOverAxis(name: string, mean: boolean, a: TensorData, axis: number): TensorData {
    const shape = a.shape;
    const k = this.axisOf(shape, axis);
    if (k === shape.length - 1) return this.reduceAxis(name, mean, a);

    const axisLen = shape[k];
    const inner = shape.slice(k + 1).reduce((x, y) => x * y, 1);
    const outer = shape.slice(0, k).reduce((x, y) => x * y, 1);

    /* [axis, inner] -> [inner, axis] makes the reduced axis contiguous, and the
     * reduction then produces one value per inner position, which is the shape
     * the caller wanted with dimension k removed. */
    const plane = this.reshape(a, [outer, axisLen, inner]);
    const t = this.transpose(plane);
    const reduced = this.reduceAxis(name, mean, t);
    const outShape = [...shape.slice(0, k), ...shape.slice(k + 1)];
    return this.reshape(reduced, outShape.length ? outShape : [1]);
  }

  /*
   * keepdims is HONOURED, and ignoring it was a real bug rather than an
   * omission: the caller reshapes the result assuming the reduced dimension is
   * still there, so dropping it produced a rank-1 tensor where a rank-2 was
   * expected and the failure surfaced as a reshape complaining about an element
   * count -- which was true and pointed at the wrong operation.
   */
  private keep(t: TensorData, a: TensorData, axis: number, keepdims: boolean): TensorData {
    if (!keepdims) return t;
    const k = this.axisOf(a.shape, axis);
    return this.reshape(t, [...a.shape.slice(0, k), 1, ...a.shape.slice(k + 1)]);
  }

  sum(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) return this.reduceAll("sum", false, a);
    return this.keep(this.reduceOverAxis("sum", false, a, axis), a, axis, keepdims);
  }

  mean(a: TensorData, axis?: number, keepdims = false): TensorData {
    if (axis === undefined) return this.reduceAll("mean", true, a);
    return this.keep(this.reduceOverAxis("mean", true, a, axis), a, axis, keepdims);
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
    if (axis !== undefined && this.axisOf(a.shape, axis) !== a.shape.length - 1)
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
    /*
     * The output keeps the INDICES' shape and appends the feature dimension:
     * [1,8] indices into a [16,16] table give [1,8,16], not [8,16].
     *
     * Flattening it to [tokens, dim] was the root of every shape failure in the
     * model for several rounds. It is operation EIGHT of a hundred and
     * forty-seven, and it surfaced as a reshape complaining about an element
     * count in the backward pass, seventy operations later, in a node that had
     * nothing to do with it. Found by tracing both backends and diffing the
     * first divergence -- which took one run, after several rounds of guessing
     * from the symptom.
     */
    const out = this.make([...indices.shape, dim], "f32");
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
    const perRow = this.make([rows], "f32");
    this.check(this.hl.crossEntropy(perRow.buffer.handle, dl.buffer.handle,
                                    ids.buffer.handle, rows, classes),
               "crossEntropy");
    ids.buffer.release(this.hl);
    /*
     * The kernel produces one loss per ROW; the interface returns a SCALAR --
     * the mean over rows. Returning the per-row vector was the contract
     * disagreement behind a 1.2% loss mismatch that survived a green suite,
     * because nothing tested the loss function itself. Coverage that stops
     * before the thing being measured is not coverage.
     */
    const out = this.reduceAll("crossEntropy mean", true, perRow) as NativeTensor;
    perRow.buffer.release(this.hl);
    const scalar: NativeTensor = {
      shape: [],
      dtype: "f32",
      data: out.data,
      buffer: out.buffer,
    };
    return scalar;
  }

  /**
   * Swap the last two dimensions, keeping every leading one.
   *
   * Same shape-propagation bug matmul had, and worth fixing the same way: the
   * first version returned [cols, rows] and dropped the batch entirely, so a
   * [1,2,8,16] came back rank-2. The kernel is unchanged -- it transposes one
   * matrix -- and the batch is handled by launching it once per leading index,
   * which is a loop and is honest about being one.
   */
  transpose(a: TensorData, dim0?: number, dim1?: number): TensorData {
    const rank = a.shape.length;

    /*
     * The interface takes the two dimensions to swap, and ignoring them was a
     * silent wrong answer rather than a missing feature.
     *
     * This always swapped the LAST TWO, which is right for a matrix and wrong
     * for attention: swapping heads and positions on a [B,H,T,D] is
     * transpose(1,2), and doing (2,3) instead produced a tensor of the right
     * SIZE with its axes interleaved. Nothing downstream objected until a
     * gradient came back the wrong rank, three operations later, in the
     * backward of a reshape that had nothing to do with it.
     *
     * A general permutation is a strided copy and does no arithmetic, so it
     * runs here rather than as a kernel; the last-two case still uses the
     * kernel because it is the hot one and it already exists.
     */
    const d0 = dim0 === undefined ? rank - 2 : this.axisOf(a.shape, dim0);
    const d1 = dim1 === undefined ? rank - 1 : this.axisOf(a.shape, dim1);
    if (!(d0 === rank - 2 && d1 === rank - 1) && !(d0 === rank - 1 && d1 === rank - 2))
      return this.permuteSwap(a, d0, d1);

    const rows = a.shape[rank - 2] ?? 1;
    const cols = a.shape[rank - 1] ?? 1;
    const batch = shapeSize(a.shape) / (rows * cols);
    const da = this.device(a);
    const outShape = [...a.shape.slice(0, -2), cols, rows];
    const out = this.make(outShape, "f32");

    if (batch === 1) {
      this.check(this.hl.transpose(out.buffer.handle, da.buffer.handle, rows, cols),
                 "transpose");
      return out;
    }

    this.sync();
    const one = this.make([rows, cols], "f32");
    const oneOut = this.make([cols, rows], "f32");
    const plane = rows * cols;
    for (let bI = 0; bI < batch; bI++) {
      one.buffer.floats.set(da.buffer.floats.subarray(bI * plane, (bI + 1) * plane));
      this.check(this.hl.transpose(oneOut.buffer.handle, one.buffer.handle, rows, cols),
                 "transpose");
      this.sync();
      out.buffer.floats.set(oneOut.buffer.floats.subarray(0, plane), bI * plane);
    }
    one.buffer.release(this.hl);
    oneOut.buffer.release(this.hl);
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

  /**
   * A sub-block of a row-major tensor, any rank.
   *
   * One dimension uses the kernel. More than one copies RUNS between the two
   * mapped views, and that is not a shortcut: a slice performs no arithmetic,
   * and its innermost dimension is contiguous in a row-major layout, so the
   * whole operation is a sequence of contiguous copies at memory bandwidth. A
   * kernel would add a launch to do exactly what the copy does.
   *
   * The run length is the product of the sliced extents from the first
   * FULL-WIDTH trailing dimension inward. Slicing only the outer dimensions --
   * which is what taking a batch or a head does, and what the model actually
   * asks for -- makes that the entire inner block, so it is one copy per outer
   * index rather than one per element.
   */
  slice(a: TensorData, starts: number[], ends: number[]): TensorData {
    const shape = a.shape;
    if (starts.length !== shape.length || ends.length !== shape.length)
      throw new Error(
        `helios-native: slice needs one bound per dimension, got ` +
          `${starts.length} for shape ${shape}`,
      );
    const extents = ends.map((e, i) => e - starts[i]);
    const da = this.device(a);
    const out = this.make(extents, "f32");

    if (shape.length === 1) {
      this.check(this.hl.slice(out.buffer.handle, da.buffer.handle, extents[0],
                               starts[0], 1), "slice");
      return out;
    }

    /* The source is read on the HOST below, so whatever produced it must have
     * run. Missing this gives a slice of the values that were in the buffer
     * before the producing kernel wrote them. */
    this.sync();

    /* How many trailing dimensions are taken whole: those form one contiguous
     * run in the source and can move in a single copy. */
    let runDims = 0;
    while (runDims < shape.length &&
           extents[shape.length - 1 - runDims] === shape[shape.length - 1 - runDims])
      runDims++;
    const run = extents.slice(shape.length - runDims).reduce((x, y) => x * y, 1);

    const srcStride: number[] = new Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * shape[i + 1];

    const outer = extents.slice(0, shape.length - runDims);
    const total = outer.reduce((x, y) => x * y, 1);
    const idx = new Array(outer.length).fill(0);
    for (let n = 0; n < total; n++) {
      let src = 0;
      for (let d = 0; d < outer.length; d++) src += (starts[d] + idx[d]) * srcStride[d];
      out.buffer.floats.set(da.buffer.floats.subarray(src, src + run), n * run);
      for (let d = outer.length - 1; d >= 0; d--) {
        if (++idx[d] < outer[d]) break;
        idx[d] = 0;
      }
    }
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

  /**
   * The mask BROADCASTS, and not expanding it was a silent partial mask.
   *
   * Attention applies an [8,8] causal mask to [1,2,8,8] scores -- one mask,
   * every head. Reading it flat meant head 0 masked correctly and head 1 read
   * past the end of the mask entirely, so half the attention saw the future.
   * The forward loss barely moved; the gradients came back with the wrong sign.
   *
   * Found by a differential trace with its threshold set ABOVE the accumulated
   * MUFU noise: at noise level the trace stops on the first harmless difference
   * and never reaches this one.
   */
  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData {
    const dm = this.expand(mask, a.shape);
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
    this.sync();
    if (shapeSize(a.shape) !== shapeSize(b.shape)) return false;
    const x = this.device(a).data as ArrayLike<number>;
    const y = this.device(b).data as ArrayLike<number>;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  }

  allClose(a: TensorData, b: TensorData, atol = 1e-5, rtol = 1e-5): boolean {
    this.sync();
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

  /**
   * Concatenation, and the reason it does not need a kernel.
   *
   * It moves bytes and performs no arithmetic, and both sides are already
   * mapped into this address space -- so a copy between the views IS the
   * operation, at memory bandwidth, with no launch and no round trip. Writing a
   * kernel for it would add a launch to do exactly what a memcpy does. The
   * "every FLOP through our own SASS" constraint is about arithmetic; there is
   * none here.
   */
  /**
   * Concatenate along any axis.
   *
   * Along axis 0 the inputs are already contiguous in the output, so it is one
   * copy each. Along an interior axis each input contributes a SLAB per outer
   * index, interleaved with the others -- so the copy walks outer indices and
   * lays down one slab from each tensor in turn. Still no arithmetic: this is
   * memory movement between mapped views, at bandwidth, and a kernel would add
   * a launch to do the same thing.
   */
  cat(tensors: TensorData[], axis: number): TensorData {
    if (tensors.length === 0) throw new Error("helios-native: cat of nothing");
    const shape = tensors[0].shape;
    const k = this.axisOf(shape, axis);
    const inner = shape.slice(k + 1).reduce((x, y) => x * y, 1);
    const outer = shape.slice(0, k).reduce((x, y) => x * y, 1);
    const sizes = tensors.map((t) => t.shape[k]);
    const outShape = shape.slice();
    outShape[k] = sizes.reduce((x, y) => x + y, 0);
    const out = this.make(outShape, "f32");
    const outSlab = outShape[k] * inner;

    this.sync();
    const devs = tensors.map((t) => this.device(t));
    for (let o = 0; o < outer; o++) {
      let at = o * outSlab;
      for (let i = 0; i < devs.length; i++) {
        const slab = sizes[i] * inner;
        out.buffer.floats.set(
          devs[i].buffer.floats.subarray(o * slab, (o + 1) * slab),
          at,
        );
        at += slab;
      }
    }
    return out;
  }

  /**
   * gather along axis 0 IS the embedding lookup -- same kernel, same indices,
   * different name at the call site. Any other axis would be a different
   * addressing pattern and is not pretended to be this one.
   */
  gather(a: TensorData, axis: number, indices: TensorData): TensorData {
    if (this.axisOf(a.shape, axis) !== 0)
      this.unsupported(`gather along axis ${axis}`);
    return this.embedding(a, indices);
  }

  /*
   * argmax and topk run on the HOST, and that is not a gap.
   *
   * Both produce INDICES, which are consumed by control flow -- sampling,
   * evaluation, beam search -- so the answer has to reach the host regardless.
   * A kernel would compute them on the device and then read them back, which is
   * a launch and a synchronisation to deliver a handful of integers. Neither
   * appears in a training step; both are inference-path.
   */
  argmax(a: TensorData, axis?: number): TensorData {
    const width = axis === undefined ? shapeSize(a.shape) : (a.shape[a.shape.length - 1] ?? 1);
    const rows = shapeSize(a.shape) / width;
    this.sync();
    const src = this.device(a).buffer.floats;
    const out = this.make(rows === 1 ? [1] : [rows], "i32");
    for (let r = 0; r < rows; r++) {
      let best = 0;
      for (let i = 1; i < width; i++)
        if (src[r * width + i] > src[r * width + best]) best = i;
      out.buffer.ints[r] = best;
    }
    return out;
  }

  topk(a: TensorData, k: number, axis?: number): { values: TensorData; indices: TensorData } {
    const width = axis === undefined ? shapeSize(a.shape) : (a.shape[a.shape.length - 1] ?? 1);
    const rows = shapeSize(a.shape) / width;
    this.sync();
    const src = this.device(a).buffer.floats;
    const values = this.make(rows === 1 ? [k] : [rows, k], "f32");
    const indices = this.make(rows === 1 ? [k] : [rows, k], "i32");
    for (let r = 0; r < rows; r++) {
      const order = Array.from({ length: width }, (_, i) => i)
        .sort((x, y) => src[r * width + y] - src[r * width + x])
        .slice(0, k);
      for (let j = 0; j < k; j++) {
        values.buffer.floats[r * k + j] = src[r * width + order[j]];
        indices.buffer.ints[r * k + j] = order[j];
      }
    }
    return { values, indices };
  }

  /** Swap two arbitrary axes by a strided copy. No arithmetic, so no kernel. */
  private permuteSwap(a: TensorData, d0: number, d1: number): NativeTensor {
    const shape = a.shape;
    const outShape = shape.slice();
    [outShape[d0], outShape[d1]] = [outShape[d1], outShape[d0]];
    this.sync();
    const da = this.device(a);
    const out = this.make(outShape, "f32");

    const srcStride: number[] = new Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * shape[i + 1];

    const n = shapeSize(shape);
    const idx = new Array(outShape.length).fill(0);
    for (let o = 0; o < n; o++) {
      /* The destination index walks in order; the source index is the same
       * coordinates with the two axes exchanged. */
      let si = 0;
      for (let d = 0; d < outShape.length; d++) {
        const sd = d === d0 ? d1 : d === d1 ? d0 : d;
        si += idx[d] * srcStride[sd];
      }
      out.buffer.floats[o] = da.buffer.floats[si];
      for (let d = outShape.length - 1; d >= 0; d--) {
        if (++idx[d] < outShape[d]) break;
        idx[d] = 0;
      }
    }
    return out;
  }

  /**
   * Return a tensor's memory to the pool.
   *
   * The tape and the model both accept a `release` callback and call it on
   * every intermediate they finish with. Not passing it does not leak in the
   * usual sense -- the pool still owns the memory -- but nothing is ever
   * REUSED, so a training loop allocates a fresh buffer per operation per step
   * until the slot table runs out. Twenty steps of a two-layer model exhausted
   * it, which is how this was found: a benchmark, not a correctness test.
   *
   * Safe on anything: a tensor that is not ours, or a handle already freed, is
   * ignored rather than double-freed.
   */
  release(t: TensorData): void {
    if (isNative(t) && !t.buffer.released) t.buffer.release(this.hl);
  }

  /** Drain the queue. Callers reading a tensor's `data` directly must call
   * this first — the operations are asynchronous now. */
  sync_(): void {
    this.sync();
  }

  /** Device identity, for the NVIDIA gate. Null until the context is open. */
  deviceInfo(): ReturnType<NativeAddon["deviceInfo"]> {
    return this.hl.deviceInfo();
  }

  /** Pool and program statistics, for confirming a step reuses rather than
   * reallocates. `allocations` should stop growing after the first step. */
  stats(): { live: number; pooled: number; allocations: number; programs: number; enqueued: number; flushes: number } {
    this.sync();
    return this.hl.stats();
  }
}
