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

/* Debug: drain and report after every dispatch, so an asynchronous MMU fault
 * is attributed to the kernel that caused it rather than the one that noticed. */
const TRACE_OPS = !!process.env.HELIOS_TRACE_OPS;

/*
 * Whether to use the FUSED A @ B^T kernel. Off, and the margin GREW.
 *
 * When first measured it was 3-6% slower and read as roughly neutral, so the
 * case for turning it on later was that it removes a fallback transpose and the
 * tensor nobody frees. Re-measured after the K-unroll, on the stable per-op
 * instrument:
 *
 *     105M seq 64, GPU ms/step   composed 276    fused 324.6  (3 runs, +-0.1)
 *     benchmark batch 128        60,402 tok/s    56,236 tok/s
 *
 * The unroll made the COMPOSED path faster — its transpose and its matmul both
 * benefit — while the fused one keeps an uncoalesced B read that unrolling
 * cannot help. And it does not even buy the memory: the step still leaks 356
 * tensors either way, so the fallback transposes were not the survivors the
 * allocation census pointed at.
 */
/*
 * ON when the shape can use the TENSOR CORES, and the refutation that turned it
 * off does not carry to that kernel.
 *
 * It was measured slower twice, both times for the scalar GEMM, and the reason
 * given each time was that a transposed B makes B[k][col] = col*K + k — a
 * column walk, uncoalesced, thirty-two transactions a warp — so the cost of the
 * fused form scales with block count while the saving does not.
 *
 * The tensor-core kernel reads B COLUMN-major by construction: m16n8k16 is
 * `row.col` and WANTS B transposed, which is why the transposed layout is the
 * CHEAPER of the two there (isa/hmma-sm86.md says so, from the capture). So the
 * composed path is now paying for a full-size transpose and a launch to hand
 * the kernel the orientation it did not want:
 *
 *     matmulTransposed  73 calls  209 us each   against matmul's 72
 *
 * The gate stays a constant rather than becoming an env var because a build
 * that half-takes it is an invalid launch, and those fault asynchronously.
 */
const FUSED_MATMUL_TRANSPOSED = true;

/*
 * Route the NT forward projection through the cp.async f16 GEMM, behind an env
 * flag so the gate can A/B it against the staged f32 path in ONE build. It is an
 * env var and not a compile constant ON PURPOSE — this is a MEASUREMENT, not a
 * shipped decision. The raw kernel is +10% (probe-cpasync-gemm.mjs), but wiring
 * it here adds a per-call cast of both operands to f16, and whether that cast
 * eats the win is exactly the question the gate answers. The real win needs the
 * operands ALREADY f16 (activations/weights f16 through the stack); this flag
 * measures how much a per-call cast leaves on the table. Numerically identical:
 * the staged path already rounds operands to f16 before the tensor cores.
 *
 * MEASURED 2026-08-06: with the per-call cast it is 16,283 tok/s against the
 * staged 19,926 — 0.82x, NET NEGATIVE, loss bit-identical. The cast is two extra
 * global passes over the operands plus two allocations per GEMM (held memory
 * 4.20 -> 4.67 GB), and that costs more than the raw kernel's +10% saves. So the
 * +10% is real but only capturable with PERSISTENT f16 operands — an f16
 * activation/weight dtype through the stack, which is the large change M5 tracks.
 * The flag stays OFF; it is kept as the instrument that proved this.
 */
const CPASYNC_MM = process.env.HELIOS_CPASYNC_MM === "1";

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

  /** Tensors already released, so a second offer of the same one is ignored
   * rather than decrementing its buffer's reference count twice. */
  private readonly freed = new WeakSet<object>();

  /*
   * A row of ones per length, kept for the life of the backend.
   *
   * `sum(x, 0)` is `ones[1,R] @ x[R,C]` — the only formulation that costs no
   * temporary larger than its result — and the fused layerNorm backward runs it
   * twice a layer, so a 105M step was allocating and FILLING thirty-seven of
   * them: 39 `full` launches, 2.1 ms of GPU, for a vector of 512 floats that is
   * the same vector every time.
   *
   * A cache is the thing that leaked in the Vulkan backend, so the terms are
   * explicit: keyed by LENGTH, never invalidated because a row of ones does not
   * depend on anything, never released, and bounded by the number of distinct
   * row counts a model uses — which is one. Two kilobytes, once.
   */
  private readonly onesRows = new Map<number, TensorData>();

  private onesRow(n: number): TensorData {
    let t = this.onesRows.get(n);
    if (!t) { t = this.full([1, n], 1); this.onesRows.set(n, t); }
    return t;
  }

  /*
   * IS THERE ANYTHING TO WAIT FOR? Tracked here so the answer is free.
   *
   * The barrier lives on every tensor's `data` getter, which is what makes it
   * impossible to miss a read site. The cost is that autograd's inner loops
   * touch it per ELEMENT -- `grad.data[i] += g.data[i]` is two getter calls an
   * iteration -- and at batch 128 a step made 1,052,759 of them. Each was a
   * napi crossing into a C function that looked at a counter and returned, so
   * ~100 ms of a 499 ms step was spent asking "anything pending?" across the
   * language boundary.
   *
   * The flag answers it on this side. It is set by check(), which every
   * dispatch goes through, and cleared when the queue is drained. Its error is
   * one-sided by construction: the C ring also flushes itself when it fills, so
   * the flag can be true when nothing is queued -- costing one wasted flush --
   * and cannot be false while work is outstanding, which is the direction that
   * would return stale data.
   */
  private pending = false;

  constructor(deviceIndex = 0) {
    this.hl = nativeAddon(deviceIndex);
    /*
     * HOST-MAPPED, deliberately, even when every other tensor is in video memory.
     *
     * reduceAll clears this on the host and then passes the HANDLE straight to
     * the kernel, bypassing `device()` — so a mirrored write would never be
     * committed and the second reduction pass would sum whatever the last one
     * left. It is 4 KB read once per reduction; keeping it in system memory
     * costs nothing measurable and keeps the host write direct.
     */
    this.scratch = NativeBuffer.allocHost(this.hl, 1024);
    /*
     * The fused layerNorm backward is ON. It was opt-in for three sessions
     * because it measured slower, and that measurement was right about the
     * machine it was taken on and wrong about this one.
     *
     *     105M, 18 layers, seq 64, batch 8, HELIOS_VIDMEM
     *       JavaScript fallback   172.1 ms/step   2,975 tok/s   host 91.6
     *       fused kernel           76.2 ms/step   6,721 tok/s   host 16.1
     *
     * Loss 9.6057 on both. The step goes from 47% GPU to 79% GPU, which is the
     * result rather than the 2.3x: the fallback reads device memory, so it
     * DRAINS THE QUEUE — 150 times a step — and the host and the device could
     * never overlap. Removing it deletes both the JavaScript loop over twelve
     * million elements and every mid-step barrier in front of it.
     *
     * What changed since the refutation: tensors moved to VIDEO memory, where
     * `xData.data` is a PCIe copy and a drain rather than a cached view; and the
     * three temporaries this arm allocated per layer became one, then none that
     * outlive the call. Until that last part it exhausted the card at 18 layers,
     * which is what kept the old measurement looking correct.
     *
     * ops.ts probes `if (B.layerNormBackward)`, so hiding the method is how the
     * arm is selected. HELIOS_FUSED_LNB=0 restores the JavaScript fallback for
     * a comparison without rebuilding.
     */
    if (process.env.HELIOS_FUSED_LNB === "0")
      (this as unknown as Record<string, unknown>).layerNormBackward = undefined;
    /*
     * The fused softCap backward is ON. It was gated off for one commit while
     * it was wrong, and what was wrong is worth keeping: the kernel consumed
     * the GRADIENT seven instructions after its load issued without waiting on
     * that load's barrier. It presented as some elements taking a plausible
     * value computed from another element's gradient — only on tensors big
     * enough for the latency to vary, and only where the answer was not already
     * saturated to zero, so a cap of 1.5 passed while 4 and 30 failed on the
     * same data.
     *
     * HELIOS_FUSED_SOFTCAP=0 restores the composed six-operation fallback.
     * packages/tests/diff-softcap-backward.mjs is the arbiter: 12/12 at three
     * caps and four shapes, worst 4.6e-7.
     */
    if (process.env.HELIOS_FUSED_SOFTCAP === "0")
      (this as unknown as Record<string, unknown>).softCapBackward = undefined;
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
    const self = this;
    /*
     * THE VIEW IS TAKEN LAZILY, and under video memory that is the difference
     * between working and not.
     *
     * `buffer.floats` on a device-resident tensor materialises a system-memory
     * mirror and copies into it. Taking it here would do that for every tensor
     * the model allocates -- hundreds a step, every one of which the GPU is
     * about to overwrite anyway -- so the copies would be pure loss and the
     * staging pool would hold a shadow of the entire model.
     *
     * It also has to stay a getter rather than a cached local, because the
     * mirror is refreshed against the device epoch: a view captured before a
     * kernel ran would be the bytes from before it ran.
     */
    /* Mapped memory can cache the view: it never moves. Device-resident memory
     * cannot, because the getter is what refreshes the mirror. */
    const cached = buffer.mapped ? buffer.floats.subarray(0, n) : null;
    return {
      shape,
      dtype,
      get data() {
        self.sync();
        return cached ?? buffer.floats.subarray(0, n);
      },
      buffer,
    } as NativeTensor;
  }

  /** Upload a host tensor, or pass a device one straight through. */
  private device(t: TensorData): NativeTensor {
    if (isNative(t)) {
      /*
       * THE WRITE-BACK POINT, and it is here because this is the only place
       * every operand passes through.
       *
       * Under video memory a host read hands back a system-memory mirror, and
       * nothing can observe a write into a Float32Array. So the writes are
       * pushed at the moment before a kernel could read them: an operation
       * resolves its inputs, and resolving is this. Anything earlier would miss
       * writes made after it; anything later would be after the launch.
       *
       * On the default mapped path this is a null check.
       */
      t.buffer.commit();
      return t;
    }
    const out = this.make(t.shape, "f32");
    const src = t.data as ArrayLike<number>;
    const dst = out.buffer.floats;
    for (let i = 0; i < src.length; i++) dst[i] = src[i];
    /* The host just wrote it; the device has to see that before anything reads
     * it, and `out` is returned as an operand rather than passing through the
     * branch above. */
    out.buffer.commit();
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
    if (!this.pending) return;
    this.hl.flush();
    this.pending = false;
  }

  /*
   * Say WHICH operand was dead, not just that the dispatch failed.
   *
   * A released buffer resolves to a null GPU address and the dispatch returns
   * -1, so every use-after-free arrives as "add failed on the device" -- an
   * error about the operation, from a bug in a lifetime, with nothing to say
   * whether the accumulator or the increment was the stale one. Naming the
   * operand is the difference between a day of bisecting and a minute.
   */
  private check(ok: boolean, op: string, ...operands: (TensorData | undefined)[]): void {
    /* Every dispatch in this file passes through here, which is what makes one
     * flag sufficient — and one invalidation. A kernel may have written any
     * buffer, so every staging mirror taken before now is stale. */
    this.pending = true;
    NativeBuffer.invalidateMirrors();
    /*
     * A drain after EVERY operation, when asked, because an MMU fault is
     * asynchronous and therefore mis-attributed by default.
     *
     * The channel reports the fault whenever the host next looks, which is at
     * some flush well after the kernel that caused it -- so "layerNorm failed
     * on the device (channel error 0x1f)" names the operation that noticed, not
     * the one that faulted. Draining per operation collapses the two.
     *
     * Debug only: it removes the batching entirely and costs an order of
     * magnitude.
     */
    if (TRACE_OPS) {
      const drained = this.hl.flush();
      const err = (this.hl.stats() as { lastError?: number }).lastError ?? 0;
      if (!drained || err)
        throw new Error(`helios-native: ${op} FAULTED (flush=${drained} channel error 0x${err.toString(16)}) ` +
                        `— drain-per-op attribution, this is the operation that caused it`);
    }
    if (ok) return;
    const dead = operands
      .map((t, i) => (t && isNative(t) && t.buffer.released ? `#${i} ${JSON.stringify(t.shape)}` : null))
      .filter(Boolean);
    /* The channel's error notifier separates a dispatch the driver refused from
     * one that was accepted and then never signalled -- a timeout. Without it
     * both arrive as the same sentence and the next hour goes to guessing. */
    const err = (this.hl.stats() as { lastError?: number }).lastError ?? 0;
    const why = err ? ` (channel error 0x${err.toString(16)})` : " (channel clean — dispatch refused or timed out)";
    throw new Error(
      dead.length
        ? `helios-native: ${op} was given RELEASED operand(s) ${dead.join(", ")} — ` +
          `something freed a tensor the graph still references`
        : `helios-native: ${op} failed on the device${why}`,
    );
  }

  // ── creation ─────────────────────────────────────────────────────────────

  /*
   * FILL ON THE DEVICE. Reading `.floats` to fill on the host is the most
   * expensive way to write a constant.
   *
   * Under video memory a host read hands back a system-memory MIRROR, so
   * `t.buffer.floats.fill(0)` allocated a staging buffer the size of the
   * tensor, wrote zeros into it with the CPU, and then copied the whole thing
   * back across PCIe on commit -- to produce a value the GPU can write for
   * itself in one launch. A 105M step calls this 185 times, and the backward
   * pass's gradient accumulators are the big ones: the LM head's is 640 x
   * 12,288, 31.5 MB, twice.
   *
   * PR_EW_FILL does not read its input (see elementwise.c reads_input), so the
   * output handle stands in for both operands and nothing is loaded. Skipping
   * commit is correct rather than merely cheaper: the host never wrote, so
   * there is no dirty mirror to push, and a later host read stages down from
   * the device as it does for any other kernel result.
   */
  zeros(shape: Shape, dtype: Dtype = "f32"): TensorData {
    return this.full(shape, 0, dtype);
  }

  ones(shape: Shape, dtype: Dtype = "f32"): TensorData {
    return this.full(shape, 1, dtype);
  }

  full(shape: Shape, value: number, dtype: Dtype = "f32"): TensorData {
    const out = this.make(shape, dtype);
    const n = shapeSize(shape);
    this.check(
      this.hl.elementwise(this.hl.op.fill, out.buffer.handle,
                          out.buffer.handle, out.buffer.handle, n,
                          value, 0, 0, 0, 0, 0, 1),
      "full",
    );
    return out;
  }

  randn(shape: Shape, dtype: Dtype = "f32"): TensorData {
    /* Host-side, because a normal deviate needs a Box-Muller pair and there is
     * no kernel for it. Initialisation happens once, so this costs nothing per
     * step -- unlike the operations above, which are on the hot path. */
    const t = this.make(shape, dtype);
    const n = shapeSize(shape);
    /* Hoisted: `floats` is a getter, and under video residency it is the one
     * that materialises the staging mirror. Reading it per element would copy
     * the buffer per element. */
    const dst = t.buffer.floats;
    for (let i = 0; i < n; i++) {
      const u = Math.random() || Number.EPSILON;
      dst[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
    }
    t.buffer.commit();
    return t;
  }

  fromArray(data: ArrayLike<number>, shape: Shape, dtype: Dtype = "f32"): TensorData {
    const t = this.make(shape, dtype);
    const dst = t.buffer.floats;
    for (let i = 0; i < data.length; i++) dst[i] = data[i];
    t.buffer.commit();
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

    /*
     * TWO SHAPES DO NOT NEED THE HOST AT ALL, and they are the two the model
     * asks for: a vector tiled down the rows (every bias, every norm weight)
     * and one value spread across each row (every mean and reciprocal deviation
     * in a normalisation). Both are a kernel of ten instructions once the launch
     * supplies the row.
     *
     * It matters beyond the copy. Broadcasting on the host READS device memory,
     * so it drains the queue, and it is one of the reads that keeps tensors in
     * system memory — where the GPU sees 19.7 GB/s against ~448 from its own.
     */
    const W = shape[shape.length - 1] ?? 1;
    const rowsOut = want / W;
    const srcLast = t.shape[t.shape.length - 1] ?? 1;

    /*
     * TILE A BLOCK, not just a row.
     *
     * Matching only a [C] vector repeated down the rows missed the causal mask:
     * [T,T] tiled across [B,H,T,T] repeats a whole 1,024-element BLOCK, and fell
     * to the host — 39.8 ms a step inside maskedFill.
     *
     * The kernel already does this. Tiling is `src = column`, and nothing in it
     * cares whether the column indexes a row of C or a block of T*T; only the
     * block has to fit one launch. So the condition is the general one: the
     * source repeats a whole number of times and its size is a legal block.
     */
    /*
     * 1024 WAS THE THREAD LIMIT, NOT THE KERNEL'S LIMIT, and the difference is
     * a queue drain per layer.
     *
     * The note above records the causal mask as "a 1,024-element BLOCK" — true
     * at sequence 32, where [T,T] is 32x32. At sequence 64 it is 4,096, the
     * guard rejects it, and the tile falls back to the host path below, which
     * READS device memory and therefore SYNCS. Profiling 105M put 692 ms of
     * fence-spin under maskedFill -> expand, a drain in the middle of the
     * attention block of every one of the eighteen layers.
     *
     * The kernel never had this restriction: pr_emit_broadcast computes
     * `chunks = ceil(W / pr_row_block(W))` and unrolls over them with
     * alternating value and address registers, exactly so a width can exceed
     * one block of threads. Only this constant said otherwise.
     *
     * 16384 rather than unbounded because the unroll is real instructions —
     * sixteen chunks of about eight against a 256-instruction program budget.
     * The vocabulary is 12,288, and embeddingBackward tiles a row that wide.
     */
    const MAX_BLOCK = 16384;
    /*
     * A TILE REQUIRES THE SOURCE TO MATCH THE DESTINATION'S TRAILING AXES.
     *
     * `want % have === 0` is not that condition, and the difference is a wrong
     * answer rather than a slow one: [B,T,1] -> [B,T,C] also divides evenly, and
     * it is a row SPREAD — each value repeated across a row, not the block
     * repeated end to end. Matching on divisibility alone sent it down the tile
     * path and the suite caught it, which is the whole reason these guards
     * describe the mapping rather than the sizes.
     */
    const tilePad = shape.length - t.shape.length;
    const isTile = tilePad >= 0 && t.shape.every((d, i) => d === shape[i + tilePad]);
    if (isTile && have > 0 && have <= MAX_BLOCK && want % have === 0) {
      this.check(
        this.hl.broadcastRows(out.buffer.handle, dt.buffer.handle, 0, have, want / have),
        "broadcast", dt);
      return out;
    }
    if (W > 0 && rowsOut > 0 && Number.isInteger(rowsOut) && W <= MAX_BLOCK &&
        have === rowsOut && srcLast === 1) {      /* [.., 1] spread along a row */
      this.check(this.hl.broadcastRows(out.buffer.handle, dt.buffer.handle, 1, W, rowsOut),
                 "broadcast", dt);
      return out;
    }

    this.sync();
    const src = dt.buffer.floats;
    const pad = shape.length - t.shape.length;
    const sStride: number[] = new Array(shape.length).fill(0);
    let acc = 1;
    for (let i = t.shape.length - 1; i >= 0; i--) {
      sStride[i + pad] = t.shape[i] === 1 ? 0 : acc;
      acc *= t.shape[i];
    }
    /*
     * A BROADCAST REPEATS A RUN. Copy the run, not its elements.
     *
     * The trailing axes that are NOT being broadcast are contiguous in the
     * source and land contiguously in the destination, so they move whole. Every
     * broadcast this model performs is of that form -- a bias or a norm weight
     * of shape [C] tiled across [B,T,C], or a per-row mean of [B,T,1] repeated
     * along C -- and the element loop was recomputing a full multi-dimensional
     * index for each of 262,144 elements to move runs of 64 that were already
     * in order.
     *
     * The run is the longest trailing span whose source strides are exactly the
     * contiguous ones; a zero stride marks a broadcast axis and ends it. When
     * the broadcast reaches the last axis the run is 1 and this degrades to the
     * old loop, which is correct and is what [B,T,1] -> [B,T,C] does.
     *
     * layerNorm and maskedFill are the callers that made this matter: 42.2 and
     * 39.9 ms a step at batch 128, all of it addressing.
     */
    let run = 1;
    let runDim = shape.length;
    while (runDim > 0 && sStride[runDim - 1] === run && shape[runDim - 1] > 0) {
      run *= shape[runDim - 1];
      runDim--;
    }

    const idx = new Array(shape.length).fill(0);
    const dst = out.buffer.floats;
    if (run > 1) {
      const total = want / run;
      for (let n = 0; n < total; n++) {
        let si = 0;
        for (let d = 0; d < runDim; d++) si += idx[d] * sStride[d];
        dst.set(src.subarray(si, si + run), n * run);
        for (let d = runDim - 1; d >= 0; d--) {
          if (++idx[d] < shape[d]) break;
          idx[d] = 0;
        }
      }
      /* Written on the host and returned straight to `binary`, which passes the
       * handle to a kernel without going through `device()`. Everything built
       * on the host commits before it is returned — that is the invariant, and
       * it is local so it cannot be forgotten somewhere else. */
      out.buffer.commit();
      return out;
    }

    for (let n = 0; n < want; n++) {
      let si = 0;
      for (let d = 0; d < shape.length; d++) si += idx[d] * sStride[d];
      dst[n] = src[si];
      for (let d = shape.length - 1; d >= 0; d--) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
    out.buffer.commit();
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
      name, da, db,
    );
    /*
     * A MATERIALISED BROADCAST IS THIS FUNCTION'S RUBBISH, and nobody else can
     * take it out.
     *
     * `expand` returns the operand untouched when the shapes already match and
     * a NEW tensor when they do not — every bias, every norm weight, every mask.
     * That tensor is never returned to the caller, so the release callback the
     * tape and the model pass around cannot see it, and it leaked once per
     * broadcast per step forever: 372 tensors a step at 18 layers, which is what
     * put a 105M model out of memory in two steps on an 8 GB card.
     *
     * Releasing straight after the dispatch is safe because a release here is
     * DEFERRED — helios_tensor_free only marks, and the memory stays valid and
     * untouched until helios_end_step. The kernel that was just enqueued still
     * reads it.
     */
    if (da !== a && da !== out) da.buffer.release(this.hl);
    if (db !== b && db !== da && db !== out) db.buffer.release(this.hl);
    return out;
  }

  /**
   * a += b, in place, with no new tensor.
   *
   * The tape probes for this when accumulating gradients. Without it, every
   * accumulation is `input.grad = backend.add(input.grad, g)` — a fresh tensor
   * the size of the parameter, written in full, and the old one released. A
   * 105M model accumulates into every parameter on every step, so it is a
   * carve and a full write per parameter per step for arithmetic that could
   * have gone straight into the buffer that already exists.
   *
   * The kernel takes its second operand from the OUTPUT pointer rather than a
   * third one — see reads_output in elementwise.c — so `out` and the addend are
   * the two arrays and the third handle is unused.
   */
  addInplace(a: TensorData, b: TensorData): void {
    const da = this.device(a), db = this.device(b);
    this.check(
      this.hl.elementwise(this.hl.op.addInPlace, da.buffer.handle,
                          db.buffer.handle, db.buffer.handle,
                          shapeSize(a.shape), 0, 0, 0, 0, 0, 0, 0),
      "addInplace", da, db,
    );
  }

  /**
   * out = a * b, into a buffer the caller already owns.
   *
   * The only caller aliases `out` onto `a`, which is why this is private: it is
   * sound for an ELEMENTWISE kernel, where a thread reads and writes the same
   * index, and unsound for anything whose output index differs from its input.
   * No broadcasting either — the shapes must already match, and the one caller
   * has reshaped them to agree.
   */
  private mulInto(out: TensorData, a: TensorData, b: TensorData): void {
    const dOut = out as NativeTensor, da = this.device(a), db = this.device(b);
    this.check(
      this.hl.elementwise(this.hl.op.mul, dOut.buffer.handle, da.buffer.handle,
                          db.buffer.handle, shapeSize(out.shape), 0, 0, 0, 0, 0,
                          0, 0),
      "mulInto", da, db,
    );
  }

  /**
   * Reshape and HAND OVER OWNERSHIP: the view becomes the only reference.
   *
   * `reshape` RETAINS, because a view and its source are two tensors pointing
   * at one buffer and the tape releases both. That is right when the caller
   * still holds the source and wrong — silently, permanently — when the source
   * was this function's own temporary: the caller releases the view, the count
   * falls from two to one, and the buffer is never freed.
   *
   * The corrected allocation census found exactly that, at two sites in this
   * file, on the DEFAULT path: `live` climbing 18 buffers a step with `pooled`
   * flat, so the pool carved a fresh slab every step forever. It is invisible
   * to a leak check that only asks "was release called" — release WAS called,
   * once, on a buffer that needed two.
   *
   * So: any function returning `this.reshape(x, ...)` where x is its own must
   * use this instead.
   */
  private reshapeOwned(t: TensorData, shape: Shape): NativeTensor {
    const view = this.reshape(t, shape);
    const src = t as NativeTensor;
    if (src.buffer && src.buffer !== view.buffer) return view;
    if (src.buffer && !src.buffer.released) src.buffer.release(this.hl);
    return view;
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
      name, da,
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

  /**
   * c * tanh(x/c), in one launch — the kernel was already built and had no
   * caller.
   *
   * ops.ts probes `if (ctx.backend.softCap)` and, not finding it, composes the
   * same function out of scale, clamp, exp, ones, sub, add, div and scale. That
   * fallback is not merely seven launches a layer instead of one: it allocates
   * a tensor of ONES the size of the attention scores, wraps it in a Variable
   * that requires no gradient, and therefore never hands it to the tape's
   * release callback.
   *
   *     4 layers, batch 8, 8 steps: 32 buffers, 40 MiB held, 5.00 MiB a step
   *     packages/autograd/ops.ts softCap  <-  model/gpt.ts attention
   *
   * — the largest entry in the allocation-site census, and the reason a 105M
   * run held 4.5 GB. 327,680 floats is exactly [8,10,64,64], which is exactly
   * the allocation that had been failing.
   *
   * The BACKWARD deliberately stays on ops.ts's composed path, which releases
   * every temporary it makes (it is handed the release callback and uses it).
   * A fused softCapBackward kernel is worth having and is not what this fixes.
   *
   * The folded constants come from the ADDON for the reason gelu's do: the
   * kernel evaluates c*(1 - 2/(exp2(s0*x) + 1)), and which constant belongs in
   * which slot is a property of that rearrangement rather than of the textbook
   * formula. Only s0 and s2 depend on the cap, so they are recomputed here from
   * the addon's value for c = PR_SOFTCAP_C rather than restated.
   */
  /**
   * g * (1 - tanh^2(x/cap)), in one launch.
   *
   * ops.ts composes this from SIX operations when it is absent — recompute
   * softCap, scale, mul, ones, sub, mul — each a full pass over the attention
   * scores, once per layer. The kernel evaluates it as 4r(1 - r) with
   * r = 1/(exp2(s0*x) + 1), which is the same r the forward already forms and
   * never builds tanh at all.
   *
   * Argument order matches the interface: gradient first, pre-activation
   * second. The kernel's `a` is x and its `b` is g, which is the binary
   * convention in elementwise_ops.c.
   */
  /*
   * `scale` folds the score scale in, for the fused attention chain.
   *
   * d/dx[cap*tanh(s*x/cap)] = s*(1 - tanh^2(s*x/cap)). The first factor rides
   * in s0, exactly as the forward's does; the leading s rides in the slot that
   * otherwise carries the constant 4. So the gradient can be taken against the
   * RAW scores, and the scaled intermediate never has to exist.
   */
  softCapBackward(g: TensorData, a: TensorData, cap: number, scale = 1): TensorData {
    /* s0 = 2*log2(e)/cap, s1 = 1, s2 = 4 — recovered from the addon's folded
     * pair for its own default c rather than restated here, for the reason
     * gelu's constants are: which constant belongs in which slot is a property
     * of the kernel's rearrangement, not of the textbook formula. */
    const twoLog2e = this.hl.scalar.softCapFolded * this.hl.scalar.softCapC;
    const da = this.device(a), dg = this.device(g);
    const out = this.make(a.shape, "f32");
    this.check(
      this.hl.elementwise(this.hl.op.softCapGrad, out.buffer.handle,
                          da.buffer.handle, dg.buffer.handle,
                          shapeSize(a.shape), twoLog2e * scale / cap,
                          4 * scale, 2, 0, 0, 0, 3),
      "softCapBackward", da, dg,
    );
    if (da !== a && da !== out) da.buffer.release(this.hl);
    if (dg !== g && dg !== out) dg.buffer.release(this.hl);
    return out;
  }

  softCap(a: TensorData, cap: number): TensorData {
    /* s0 = 2*log2(e)/cap, s1 = 1, s2 = cap, s3 = 2. The addon publishes the
     * folded pair for its own default c, and the ratio recovers 2*log2(e). */
    const twoLog2e = this.hl.scalar.softCapFolded * this.hl.scalar.softCapC;
    return this.unary("softCap", this.hl.op.softCap, a,
                      twoLog2e / cap, 1, cap, 2);
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
      /*
       * ONE launch for the whole batch, the plane from the block's Y index.
       *
       * This looped batch elements through the HOST -- copy both operands in,
       * launch, drain the queue, copy the result out -- so attention's four
       * heads were four round trips per matmul. The drains were the cost, not
       * the copies: with launches batched, a drain per operation is what stops
       * anything from queueing.
       */
      const M0 = a.shape[a.shape.length - 2] ?? 1;
      const batch = shapeSize(b.shape) / (K * N);
      const da = this.device(a), db = this.device(b);
      const out = this.make([...a.shape.slice(0, -1), N], "f32");
      this.check(
        this.hl.matmul(out.buffer.handle, da.buffer.handle, db.buffer.handle,
                       M0, N, K, batch),
        "matmul", da, db,
      );
      /* Uploads are this function's own — see the 2-D branch below. */
      if (da !== a && da !== out) da.buffer.release(this.hl);
      if (db !== b && db !== da && db !== out) db.buffer.release(this.hl);
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
                              db.buffer.handle, M, N, K, 1), "matmul", da, db);
    /*
     * AN UPLOAD IS THIS FUNCTION'S OWN, same as a materialised broadcast.
     *
     * `device()` returns its argument untouched when it is already on the
     * device and ALLOCATES when it is not — and the caller never sees that
     * tensor, so nobody downstream can free it. The free-gap profiler put 54
     * class-11 buffers a step here, which at this shape is weight-sized.
     *
     * Fifth instance of this shape, after binary's broadcasts, layerNorm's
     * intermediates, maskedFill's mask and matmulTransposed's transpose.
     */
    if (da !== a && da !== out) da.buffer.release(this.hl);
    if (db !== b && db !== da && db !== out) db.buffer.release(this.hl);
    return out;
  }

  /**
   * C = A @ B-transpose, with B stored [N, K].
   *
   * autograd probes for this and, not finding it, transposed B and called
   * matmul — a launch and a tensor nobody frees, on every projection. The
   * allocation census puts that at 225 MB a step across three call sites at 18
   * layers, the largest single entry in it.
   *
   * Only for a 2-D B, which is what a weight is. A batched right operand —
   * attention's Q @ K-transpose — keeps the composed path, because its plane
   * strides differ and getting that wrong is a wrong answer rather than a
   * missing feature.
   */
  /**
   * C[M,N] = A^T @ B, with A stored [K,M] — the WEIGHT GRADIENT.
   *
   * autograd probes for this name and, not finding it, transposes the incoming
   * gradient and calls matmul. That costs a full-size tensor and an extra
   * launch per weight per step — 54 of them at 18 layers, up to 24 MiB each for
   * the LM head — and the allocation census puts that path among the largest
   * consumers of a step. Peak memory here is total bytes ALLOCATED in a step,
   * not the live set, so a temporary that is freed is not thereby free.
   *
   * It costs nothing extra in the tensor-core kernel: a fragment register
   * already takes two loads and a pack, because the operands are f32 in memory
   * and f16 in the fragment, so a strided A is the same instruction count as a
   * contiguous one with different immediate offsets.
   *
   * TOTAL, not partial: when the shape does not divide the tensor-core tile it
   * transposes and multiplies right here, releasing the transpose. ops.ts
   * probes for the NAME and then calls it unconditionally, so a method that
   * could decline would have to be declined for by its caller — and the caller
   * is backend-agnostic.
   */
  matmulTransposedA(a: TensorData, b: TensorData): TensorData {
    /* a is [.., R, M], b is [.., R, N]; the result is [.., M, N]. */
    const ra = a.shape.length, rb = b.shape.length;
    const R = a.shape[ra - 2] ?? 1, M = a.shape[ra - 1] ?? 1;
    const N = b.shape[rb - 1] ?? 1;
    const shapesAgree = ra >= 2 && rb >= 2 && (b.shape[rb - 2] ?? 1) === R &&
      shapeSize(a.shape) / (R * M) === shapeSize(b.shape) / (R * N);
    if (!this.hl.matmulTransposedA || !shapesAgree) return this.transposeThenMatmul(a, b);
    const batch = shapeSize(a.shape) / (R * M);

    const da = this.device(a), db = this.device(b);
    const out = this.make([...a.shape.slice(0, -2), M, N], "f32");
    if (!this.hl.matmulTransposedA(out.buffer.handle, da.buffer.handle,
                                   db.buffer.handle, M, N, R, batch)) {
      /* The shape does not divide the tile. Give the buffer straight back —
       * an abandoned output would be a leak on exactly the path this method
       * exists to make cheaper — and take the old route. */
      out.buffer.release(this.hl);
      if (da !== a) da.buffer.release(this.hl);
      if (db !== b) db.buffer.release(this.hl);
      return this.transposeThenMatmul(a, b);
    }
    this.pending = true;
    NativeBuffer.invalidateMirrors();
    if (da !== a && da !== out) da.buffer.release(this.hl);
    if (db !== b && db !== out) db.buffer.release(this.hl);
    return out;
  }

  /**
   * dest += A @ B (layout 0), A @ B^T (1) or A^T @ B (2), in ONE launch.
   *
   * Returns false when the shape cannot reach the tensor cores — the scalar
   * kernel has no accumulating form — and the caller keeps its two-step route.
   *
   * WHY IT EXISTS. Gradient accumulation is ~24% of a step's GPU time and it is
   * already at the card's bandwidth ceiling (417 GB/s on a 0.33M-element
   * tensor, 342 on 1.31M, against 448), so it cannot be made faster; it can
   * only be made less. A matmul into a fresh tensor followed by an elementwise
   * add of it into the destination is four passes over the result and an
   * allocation. This is two passes and none.
   */
  matmulAccumulate(dest: TensorData, a: TensorData, b: TensorData,
                   layout: 0 | 1 | 2): boolean {
    if (!this.hl.matmulAccumulate) return false;
    const ra = a.shape.length, rb = b.shape.length;
    if (ra < 2 || rb < 2) return false;
    let M: number, N: number, K: number;
    if (layout === 2) {
      K = a.shape[ra - 2] ?? 1; M = a.shape[ra - 1] ?? 1; N = b.shape[rb - 1] ?? 1;
      if ((b.shape[rb - 2] ?? 1) !== K) return false;
    } else {
      M = a.shape[ra - 2] ?? 1; K = a.shape[ra - 1] ?? 1;
      N = layout === 1 ? (b.shape[rb - 2] ?? 1) : (b.shape[rb - 1] ?? 1);
      if ((layout === 1 ? (b.shape[rb - 1] ?? 1) : (b.shape[rb - 2] ?? 1)) !== K)
        return false;
    }
    if (shapeSize(dest.shape) !== M * N) return false;
    const batch = shapeSize(a.shape) / (layout === 2 ? K * M : M * K);
    if (batch !== 1) return false; /* the destination has no batch axis here */

    const dd = this.device(dest), da = this.device(a), db = this.device(b);
    if (!this.hl.matmulAccumulate(dd.buffer.handle, da.buffer.handle,
                                  db.buffer.handle, M, N, K, batch, layout))
      return false;
    this.pending = true;
    NativeBuffer.invalidateMirrors();
    if (da !== a && da !== dest) da.buffer.release(this.hl);
    if (db !== b && db !== dest) db.buffer.release(this.hl);
    return true;
  }

  /**
   * softmax's backward, fused: s * (g - sum_j g_j s_j) over each row.
   *
   * ops.ts composes it from mul, sum, broadcast, sub and mul — five passes over
   * a [B,H,T,T] tensor to express one row reduction and one write. The
   * elementwise half of a step already runs at 340-417 GB/s against a 448 GB/s
   * card, so it cannot be made faster; it can only be made LESS.
   *
   * Only over the LAST axis, which is the only one softmax is taken over here.
   */
  softmaxBackward(s: TensorData, g: TensorData): TensorData {
    const width = s.shape[s.shape.length - 1] ?? 1;
    const rows = shapeSize(s.shape) / width;
    const ds = this.device(s), dg = this.device(g);
    const out = this.make(s.shape, "f32");
    this.check(
      this.hl.softmaxBackward!(out.buffer.handle, ds.buffer.handle,
                               dg.buffer.handle, width, rows),
      "softmaxBackward", ds, dg,
    );
    if (ds !== s && ds !== out) ds.buffer.release(this.hl);
    if (dg !== g && dg !== out) dg.buffer.release(this.hl);
    return out;
  }

  /** A^T @ B the old way, for shapes the tensor-core tile does not divide. */
  private transposeThenMatmul(a: TensorData, b: TensorData): TensorData {
    const t = this.transpose(a, a.shape.length - 2, a.shape.length - 1);
    const out = this.matmul(t, b);
    (t as NativeTensor).buffer.release(this.hl);
    return out;
  }

  matmulTransposed(a: TensorData, b: TensorData): TensorData {
    const K = a.shape[a.shape.length - 1] ?? 1;
    /*
     * UNWIRED, AND MEASURED — this returns the composed path on purpose.
     *
     * The kernel is correct at every shape the model runs (qkv 1920, mlp 1728
     * and 640, lm head 12288, odd N 1025, all within 2.4e-4 of cpu_ref) and it
     * is SLOWER:
     *
     *     benchmark shape   54,575 -> 51,431 tok/s   (-5.8%)
     *     105M seq 64           78 -> 76 tok/s       (-3%, 7.31 -> 7.09 GB)
     *
     * RE-MEASURED at batch 8, once the step's memory was bounded and M was 512
     * instead of 64 — on the theory that eliminating the separate transpose
     * (11.7% of GPU time by then) would outweigh 5.8%. It does not, and the
     * gap is far wider than at batch 1:
     *
     *     105M seq 64 batch 8   767 -> 402 tok/s     (-48%, GPU 462 -> 1046 ms)
     *
     * The penalty GROWS with M, which is what the access pattern predicts:
     * every block issues 32 separate transactions where one would do, so the
     * cost scales with block count while the saving — one transpose launch per
     * call — does not. Unlike the row-tiled and column-blocked kernels, this
     * one was never parallelism-starved; it is simply uncoalesced.
     *
     * Because the access pattern inverts. Untransposed, B[k][col] is `k*N+col`
     * and adjacent threads read adjacent addresses — one coalesced transaction.
     * Transposed, it is `col*K+k`, so adjacent threads are K elements apart and
     * every warp issues 32 separate transactions. That costs more than the
     * transpose launch and the leaked tensor it saves.
     *
     * The fix is not in this kernel, it is the tiled GEMM: staging B's tile in
     * shared memory makes the global read coalesced whichever way the tile is
     * indexed, and then the fused form is strictly better. Kept and unwired so
     * that work starts from a kernel already proven correct.
     */
    if (!FUSED_MATMUL_TRANSPOSED || b.shape.length !== 2 ||
        (b.shape[1] ?? K) !== K || !this.hl.matmulTransposed) {
      /*
       * THE TRANSPOSE IS THIS FUNCTION'S OWN, and it is weight-sized.
       *
       * Written as `return this.matmul(a, this.transpose(b, ...))` it is
       * allocated, used as an operand and dropped — 4.9 MB for a qkv weight,
       * once per projection per layer, and nobody downstream can free it
       * because it never leaves this expression. The pool's size-class
       * histogram put 87% of a 2.5 GB-per-step leak in exactly this class.
       */
      const t = this.transpose(b, b.shape.length - 2, b.shape.length - 1);
      const out = this.matmul(a, t);
      if (isNative(t)) t.buffer.release(this.hl);
      return out;
    }
    const N = b.shape[0] ?? 1;
    const M = shapeSize(a.shape) / K;
    const da = this.device(a), db = this.device(b);
    const out = this.make([...a.shape.slice(0, -1), N], "f32");

    /*
     * The cp.async f16 path — MEASURED against the staged one via the flag.
     * Cast both operands to f16 (the weight B[N,K] is already the layout
     * cp.async wants, K-contiguous, so no transpose), then LDGSTS them. Only
     * where the tile divides — pr_hmma_applies is M%64, N%64, K%16 — otherwise
     * the staged path below. The f16 buffers hold packed pairs, so their size
     * in f32 slots is half the element count.
     */
    if (CPASYNC_MM && this.hl.matmulCpasync && this.hl.cast &&
        K % 16 === 0 && M % 64 === 0 && N % 64 === 0) {
      const aF16 = this.make([(M * K) >> 1], "f32");
      const bF16 = this.make([(N * K) >> 1], "f32");
      this.hl.cast(1, aF16.buffer.handle, da.buffer.handle, M * K);
      this.hl.cast(1, bF16.buffer.handle, db.buffer.handle, N * K);
      const rc = this.hl.matmulCpasync(out.buffer.handle, aF16.buffer.handle,
                                       bF16.buffer.handle, M, N, K, 1);
      aF16.buffer.release(this.hl);
      bF16.buffer.release(this.hl);
      if (rc) return out;
      /* false means the shape was refused after all — fall through to staged. */
    }

    this.check(this.hl.matmulTransposed!(out.buffer.handle, da.buffer.handle,
                                        db.buffer.handle, M, N, K, 1),
               "matmulTransposed", da, db);
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
   * Along the LAST axis, in ONE launch.
   *
   * This looped a row at a time, copying each into a scratch buffer and
   * draining the queue to read the answer back -- so an eight-row reduction was
   * eight round trips, and the drains were the single largest cost left after
   * batching went in.
   *
   * The kernel to do it in one launch already existed: the PARTIAL reduction
   * writes one value per BLOCK, so launching one block per row with the row
   * width as the block size is exactly a row-wise reduction. It was written for
   * the first pass of a whole-tensor sum and happens to be the same shape of
   * computation.
   *
   * A mean scales afterwards rather than inside, because the partial never
   * scales -- see reduction.c, where that restraint exists so a short final
   * block cannot make a whole-tensor mean wrong.
   */
  /*
   * THE BLOCK REDUCTION TREE ONLY SPANS A POWER OF TWO, so anything else goes
   * through a matmul with ones instead.
   *
   * pr_emit_tree walks strides width/2, /4, ... down to 1. Over 15 elements
   * those are 7, 3 and 1, and element 14 is never combined into anything: the
   * sum comes back finite, plausible, and short by one element. It is wrong for
   * EVERY non-power-of-two extent, on this axis and on any other, and it has
   * always been -- the model never met it because C is 64 and B*T is 4,096.
   *
   * A sum down an axis IS a matmul with a vector of ones, which is exactly the
   * identity reduceOverAxis already uses for axes too long for one block, and
   * that path is correct at any length (verified at 1,025 and 1,500). So route
   * the ragged extents through it rather than teaching the tree to pad: the
   * padding would need out-of-range threads to load nothing and contribute the
   * combiner's identity, which is a change to every kernel that reduces, for a
   * case none of them currently meets.
   */
  private reduceAxis(name: string, mean: boolean, a: TensorData): TensorData {
    const width = a.shape[a.shape.length - 1] ?? 1;
    const rows = shapeSize(a.shape) / width;
    const outShape = a.shape.slice(0, -1);

    if ((width & (width - 1)) !== 0) {
      const summed = this.matmul(this.reshape(a, [rows, width]), this.full([width, 1], 1));
      const flat = this.reshape(summed, outShape.length ? outShape : [1]);
      return mean ? this.scale(flat, 1 / width) : flat;
    }

    const da = this.device(a);
    const out = this.make(outShape.length ? outShape : [1], "f32");
    this.check(
      this.hl.reduceRows(out.buffer.handle, da.buffer.handle, width, rows),
      name,
    );
    return mean ? this.scale(out, 1 / width) : out;
  }

  /**
   * Reduce over any axis by bringing it to the END first.
   *
   * The kernel reduces a contiguous row, so an interior axis has to be made
   * contiguous before it can be reduced. Viewing the tensor as
   * [outer, axis, inner] and transposing the [axis, inner] plane does that, and
   * both steps stay on the device -- summing on the host would be arithmetic
   * done somewhere other than our own SASS.
   */
  private reduceOverAxis(name: string, mean: boolean, a: TensorData, axis: number): TensorData {
    const shape = a.shape;
    const k = this.axisOf(shape, axis);
    if (k === shape.length - 1) return this.reduceAxis(name, mean, a);

    const axisLen = shape[k];
    const inner = shape.slice(k + 1).reduce((x, y) => x * y, 1);
    const outer = shape.slice(0, k).reduce((x, y) => x * y, 1);

    /*
     * SUMMING DOWN A LONG AXIS IS A MATMUL WITH A VECTOR OF ONES.
     *
     * The route below transposes the axis to the end and reduces a row, and the
     * row reduction puts the whole axis in ONE BLOCK — so it fails outright
     * above PR_MAX_BLOCK. That is not hypothetical: `sum(x, 0)` over a
     * [B*T, C] gradient is 4,096 rows at batch 128, and it is what broke every
     * attempt at a fused layerNorm backward, in three separate sessions, always
     * as an unexplained "matmul failed on the device" one operation later.
     *
     * ones[1,R] x a[R,C] is the same sum, in one launch, at any size. It is only
     * used when the axis is too long for a block, because for short axes the
     * reduction is the cheaper shape -- a matmul of one row does far more work
     * than it needs to.
     */
    /* Too long for one block, OR not a power of two -- see reduceAxis for why
     * the second condition is not optional. */
    /*
     * THE INTERMEDIATES BELONG TO THIS FUNCTION AND NOTHING ELSE FREES THEM.
     *
     * Both routes allocate a full-size temporary that never leaves here: the
     * matmul route a vector of ones and the product, the transpose route a
     * whole transposed copy of the input. Neither reaches the tape, so neither
     * reaches its release callback, and `sum(g, 0)` is how EVERY bias and
     * weight gradient in the model is formed.
     *
     * Found through the fused layerNorm backward, which calls it twice a layer
     * and could not survive 94 warmup steps at four layers before exhausting
     * the card — read at first as "the fused kernel needs more memory than the
     * fallback" and really a leak in a function both arms share. The fallback
     * hid it by taking a different path.
     *
     * A release only MARKS: tensor.c keeps the buffer valid to any queued
     * launch and to any reader until helios_end_step, so releasing a temporary
     * that a queued kernel is still reading is safe by construction. What is
     * NOT safe is releasing a RESHAPE, which shares its source's buffer — hence
     * the care below about which handles own memory and which are views.
     */
    const drop = (t: TensorData | undefined) => {
      if (t && isNative(t) && !t.buffer.released) t.buffer.release(this.hl);
    };

    /* Too long for one block, OR not a power of two -- see reduceAxis for why
     * the second condition is not optional. */
    if (k === 0 && outer === 1 && (axisLen > 1024 || (axisLen & (axisLen - 1)) !== 0)) {
      const ones = this.full([1, axisLen], 1);
      const summed = this.matmul(ones, this.reshape(a, [axisLen, inner]));
      const flat = this.reshape(summed, [inner]);
      const outShape0 = shape.slice(1);
      const res = mean ? this.scale(flat, 1 / axisLen) : flat;
      drop(ones);
      /* `flat` is a VIEW of `summed` and holds a reference of its own, so the
       * source must be released whichever branch ran — see reshapeOwned. */
      drop(summed);
      return this.reshapeOwned(res, outShape0.length ? outShape0 : [1]);
    }

    const plane = this.reshape(a, [outer, axisLen, inner]); /* a view of `a` */
    /*
     * REDUCING THE LAST AXIS NEEDS NO TRANSPOSE, and it was doing one.
     *
     * The route below moves the reduced axis to the end so reduceAxis can walk
     * a row. When the axis IS the last one, `inner` is 1 — and transposing
     * [outer, axisLen, 1] into [outer, 1, axisLen] leaves every element exactly
     * where it was: (o, a, 0) is at o*axisLen + a either way. So the transpose
     * is a full-size COPY that changes nothing, plus a launch, plus a tensor.
     *
     * softmax's backward reduces the last axis once per layer, and at 18 layers
     * `transpose` was 218 calls a step against a ~5.7 us floor. A reshape is a
     * view: no copy, no launch, no allocation.
     */
    const t = inner === 1 ? this.reshape(plane, [outer, 1, axisLen])
                          : this.transpose(plane);
    const reduced = this.reduceAxis(name, mean, t);
    drop(t);
    const outShape = [...shape.slice(0, k), ...shape.slice(k + 1)];
    return this.reshapeOwned(reduced, outShape.length ? outShape : [1]);
  }

  /*
   * keepdims is HONOURED. Ignoring it was a real bug: the caller reshapes the
   * result assuming the reduced dimension is still there, so dropping it
   * produced a rank-1 tensor where rank-2 was expected and surfaced as a
   * reshape complaining about an element count -- true, and pointing at the
   * wrong operation.
   */
  private keep(t: TensorData, a: TensorData, axis: number, keepdims: boolean): TensorData {
    if (!keepdims) return t;
    const k = this.axisOf(a.shape, axis);
    /* `t` is reduceOverAxis's own result and nobody else holds it. */
    return this.reshapeOwned(t, [...a.shape.slice(0, k), 1, ...a.shape.slice(k + 1)]);
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

  /*
   * THE INTERMEDIATES OF A COMPOSITE ARE ITS OWN, and the caller never sees them.
   *
   * layerNorm is normalize, then multiply by the weight, then add the bias —
   * two tensors that exist only between those three calls. The release callback
   * the tape and the model pass around gets the RESULT; it has no way to know
   * the other two were ever made, so they leaked once per norm per step, which
   * at 18 layers is 74 tensors a step doing nothing.
   *
   * Safe to release immediately: helios_tensor_free only marks, and the memory
   * stays valid until helios_end_step, so the kernels just enqueued still read
   * it. Same reasoning as the materialised broadcasts in `binary`.
   */
  /*
   * THE AFFINE IS PART OF THE NORM, not two more passes over the tensor.
   *
   * These used to be three launches: normalise, broadcast-multiply by the
   * weight, broadcast-add the bias. The multiply and the add each read a full
   * [1536,640] tensor and write another — 7.9 MB of traffic apiece to apply one
   * number per COLUMN, which every thread of a row-per-block kernel already
   * holds. Thirty-seven layer norms a step is 74 of those passes and 74
   * launches, at a point where 69% of launches already wait for the queue to
   * drain.
   *
   * The intermediates went with them, which was the other cost: the release
   * callback the tape passes around only ever sees the RESULT, so the two
   * temporaries here leaked once per norm per step until they were released by
   * hand. A kernel that never allocates them cannot leak them.
   */
  private normalizedAffine(name: string, opId: number, x: TensorData,
                           weight: TensorData, bias: TensorData | null,
                           eps: number): NativeTensor {
    const width = x.shape[x.shape.length - 1] ?? 1;
    const rows = shapeSize(x.shape) / width;
    const dx = this.device(x), dw = this.device(weight);
    const db = bias ? this.device(bias) : dw;
    const out = this.make(x.shape, "f32");
    this.check(this.hl.normalizeAffine(opId, out.buffer.handle, dx.buffer.handle,
                                       dw.buffer.handle, db.buffer.handle,
                                       width, rows, eps), name);
    return out;
  }

  /*
   * ATTENTION'S SCORE CHAIN IN ONE LAUNCH: scale, causal mask, softmax.
   *
   * Composed, it is three kernels over the same [B,H,T,T] tensor — 23.6 MB of
   * traffic per layer to apply one multiply and one predicated move per
   * element, plus a materialised copy of the mask before the fix that tiled it.
   * Fused it is 7.9 MB and one launch, at a point where 72% of launches already
   * wait for the queue to drain.
   *
   * Returns null when the shape does not fit the kernel, and the caller keeps
   * the composed path: the mask index wraps at T*T with an AND, so T has to be
   * a power of two, and the row must fit one block.
   */
  softmaxMasked(a: TensorData, mask: TensorData, scale: number,
                cap = 0): TensorData | null {
    const width = a.shape[a.shape.length - 1] ?? 1;
    const rows = shapeSize(a.shape) / width;
    if (width > 1024 || (width & (width - 1)) !== 0) return null;
    if (shapeSize(mask.shape) !== width * width) return null;
    const da = this.device(a), dm = this.device(mask);
    const out = this.make(a.shape, "f32");
    this.check(this.hl.softmaxMasked(out.buffer.handle, da.buffer.handle,
                                     dm.buffer.handle, width, rows, scale, cap),
               "softmaxMasked", da, dm);
    return out;
  }

  rmsNorm(x: TensorData, weight: TensorData, eps: number): TensorData {
    return this.normalizedAffine("rmsNorm", this.hl.op.rmsNormAffine, x, weight, null, eps);
  }

  layerNorm(x: TensorData, weight: TensorData, bias: TensorData, eps: number): TensorData {
    return this.normalizedAffine("layerNorm", this.hl.op.layerNormAffine, x, weight, bias, eps);
  }

  /*
   * THE FUSED BACKWARDS ARE NOT HERE, AND THAT IS A MEASURED DECISION.
   *
   * ops.ts probes for layerNormBackward, geluBackward, clampBackward, broadcast
   * and a dozen more; Vulkan implements them, this backend does not, so each one
   * falls back to a JavaScript loop over the tensor. That looked like the
   * obvious next win -- at batch 32 those fallbacks held 37.8 ms of drain in a
   * 195 ms step.
   *
   * They were written, as compositions of kernels that already exist, and they
   * were SLOWER than the JavaScript they replaced:
   *
   *     batch  1     2911 -> 2466 tok/s
   *     batch 16     7204 -> 5941 tok/s
   *     batch 64     7739 -> failed
   *
   * The reason is the memory fix that came before them. A CPU fallback is a
   * drain plus a host loop, and once tensors were mapped cached that loop reads
   * at 1.4 us per 2048 elements instead of 226 -- so the fallback became cheap
   * while a composition still costs ~20 launches at 20-50 us each. Fixing the
   * memory removed the motivation for the kernels.
   *
   * RETESTED AT SCALE, because the first test was at batch 1 and 16 where the
   * fallback is cheapest and the comparison was therefore rigged in its favour.
   * At batch 128 the layerNorm fallback alone costs 48.8 ms of drain a step on a
   * JavaScript loop over 262,144 elements, so the economics should reverse: a
   * fallback grows with the batch, a fixed number of launches does not.
   *
   * They do not reverse. layerNormBackward composed ON ITS OWN still lost --
   * 3126 -> 2541 at batch 1, 7232 -> 6010 at batch 16 -- and failed outright at
   * batch 128 with a dispatch error, which is a second bug the composition
   * exposes somewhere in the large reductions it introduces.
   *
   * So: COMPOSED backwards are refuted at every batch measured, one at a time
   * and all together. What is NOT refuted is a single fused KERNEL per backward
   * -- one launch instead of twenty, which is a different proposition and the
   * only remaining route to the fallbacks' ~110 ms. The compositions are in the
   * history if they are wanted as a specification of the arithmetic.
   *
   * THAT ROUTE WAS TAKEN THREE TIMES AND WON TWICE. geluBackward and
   * clampBackward below are single elementwise kernels and they are on by
   * default. layerNormBackward is a real fused kernel too -- four reductions,
   * three inputs, one launch, in prometheus/normalize.c -- and it is OFF,
   * because measured against the JavaScript it replaces, in one process on one
   * card, it LOST:
   *
   *     batch 128   JS fallback   192.9 ms/step   21,235 tok/s   1.26 GB
   *     batch 128   fused kernel  215.8 ms/step   18,984 tok/s   1.72 GB
   *
   * WHY THE PROFILE SAID OTHERWISE, which is the part worth keeping. The step
   * profiler attributes 106.88 ms of a 280 ms step to five drains at
   * autograd/ops.js:637 -- `const xArr = xData.data` inside this very fallback,
   * 38% of the step, by far the largest single item. That number is real and it
   * is not the fallback's cost. A drain is the host WAITING FOR WORK ALREADY
   * QUEUED to finish, and that work is the forward and backward kernels, which
   * have to run either way. The fallback was merely where the queue happened to
   * get drained. Deleting the reader moves the wait; it does not remove the
   * work -- and the kernel arm then ADDS ~20 launches and five full-size
   * temporaries a step to do on the GPU what a cached host mapping was already
   * doing at 1.4 us per 2048 elements.
   *
   * So the earlier paragraph's conclusion holds and is now sharper: it is not
   * that compositions are slow and kernels are fast. It is that this fallback
   * is CHEAP, and anything replacing it must beat a host loop over mapped
   * memory rather than beat the drain sitting in front of it. Attributing a
   * drain to whoever triggered it is how three sessions in a row concluded
   * otherwise.
   *
   * The kernel is kept, correct and tested, behind HELIOS_FUSED_LNB=1. It is
   * the arm to re-measure if launches ever get cheaper or the queue stops being
   * the thing in front of the reader.
   */

  /*
   * out[c] = sum over rows of a[r][c] — the bias-shaped gradient.
   *
   * Public because it is the right answer for every reduction that collapses
   * the token axis and keeps the feature axis, not only for a layer norm's dw
   * and db. See pr_emit_column_sum for why this is a kernel rather than either
   * of the two compositions that came before it.
   */
  /* With `b`, the reduction is over the elementwise PRODUCT: out[c] =
   * sum_r a[r][c] * b[r][c]. That is a layer norm's dw, and forming the product
   * as its own pass costs a full read-add-write over the tensor to produce a
   * value consumed immediately and only as a sum. */
  columnSum(a: TensorData, rows: number, cols: number, b?: TensorData): TensorData {
    const da = this.device(a);
    const db = b ? this.device(b) : null;
    const out = this.make([cols], "f32");
    this.check(this.hl.columnSum(out.buffer.handle, da.buffer.handle,
                                 db ? db.buffer.handle : 0, rows, cols),
               "columnSum", da);
    return out;
  }

  /**
   * layerNorm's backward — dx and xhat in one launch, dw and db from three ops.
   *
   * OFF BY DEFAULT; see above. Verified against the definition element by
   * element at five shapes (packages/tests/diff-layernorm-backward.mjs), dx to
   * 3e-8 absolute and dw/db to 2e-6 relative.
   *
   * The KERNEL returns dx and xhat. dw and db are formed here because they
   * reduce across ROWS rather than within one, which is a different kernel
   * shape; xhat exists as an output precisely so that dw = sum_rows(g*xhat)
   * costs a multiply rather than the two reductions that produced it.
   */
  layerNormBackward(x: TensorData, weight: TensorData, g: TensorData, eps: number):
      { dx: TensorData; dw: TensorData; db: TensorData; xhat: TensorData } {
    const width = x.shape[x.shape.length - 1] ?? 1;
    const rows = shapeSize(x.shape) / width;
    const dxIn = this.device(x), dgIn = this.device(g), dwIn = this.device(weight);
    const dx = this.make(x.shape, "f32");
    const xhat = this.make(x.shape, "f32");
    this.check(
      this.hl.layerNormBackward(dx.buffer.handle, xhat.buffer.handle,
                                dxIn.buffer.handle, dgIn.buffer.handle,
                                dwIn.buffer.handle, width, rows, eps),
      "layerNormBackward", dxIn, dgIn, dwIn);

    /*
     * Down the row axis — ITS OWN KERNEL, after two other routes were measured.
     *
     * `sum(x, 0)` takes the transpose route, which allocates a full transposed
     * copy of its input. Two a layer, and a release only MARKS — the buffer is
     * not reusable until helios_end_step — so a step's peak is the bytes
     * ALLOCATED, and that is what exhausted the card here while the JavaScript
     * fallback did not.
     *
     * `ones[1,R] @ x[R,C]` replaced it and allocates nothing larger than the
     * result, which fixed the memory and hid a much larger cost. It is a GEMM
     * with ONE output row: it fills a 32x128 tensor-core tile to a
     * thirty-second and falls off the HMMA path entirely. Profiling by SHAPE
     * rather than by op name put it at 74 calls of [1,1536] x [1536,640],
     * 298 us each, 22.0 ms — 21% of the whole GPU step, and the largest single
     * line in the profile. The op-name view could not see it, because "matmul"
     * averaged it together with the m1536 projections that run at 20 TFLOP/s.
     *
     * It moves 3.9 MB and writes 2.5 KB, which is about 10 us of bandwidth.
     * pr_emit_column_sum is that kernel: coalesced along the axis it keeps, one
     * block per 32 columns, no temporary at all.
     */
    /*
     * EVERY VIEW TAKEN HERE MUST BE GIVEN BACK. `reshape` RETAINS, so a view of
     * a tensor this function does not own still raises its count, and dropping
     * the view on the floor leaves the source pinned forever — the caller's own
     * release then takes the count from two to one and frees nothing.
     *
     * That is how the incoming gradient `g` leaked out of this very function at
     * 11 MiB a step, and `xhat` beside it at 21, while both looked released.
     */
    const flatG = this.reshape(g, [rows, width]);
    const db = this.columnSum(flatG, rows, width);
    /*
     * dw = sum_rows(g * xhat) IN ONE PASS — the multiply lives inside the
     * reduction rather than before it.
     *
     * This used to form the product first, in place over xhat's own buffer, to
     * avoid allocating a third full-size tensor: a release only MARKS here, so
     * a step costs what it ALLOCATES and a temporary is not free even when it
     * is freed. That was the right trade against `mul`, and it still cost a
     * full read-add-write pass over [1536,640] — 11.8 MB — to produce a value
     * consumed immediately and only as a sum.
     *
     * The column sum was already loading one of the two operands, so the
     * multiply is one instruction inside a loop that was going to run anyway.
     * It also retires the in-place aliasing argument the old route needed: xhat
     * is left intact and nothing is overwritten.
     */
    const flatXhat = this.reshape(xhat, [rows, width]);
    const dw = this.columnSum(flatXhat, rows, width, flatG);
    /*
     * xhat AND the product are this method's OWN, and nothing else frees them.
     *
     * ops.ts destructures dx, dw and db and ignores the fourth field, so the
     * tape's release callback never sees xhat — and `prod` is not returned at
     * all. Two full-size tensors a layer, thirty-seven layer norms a step: the
     * fused arm could not survive eleven warmup steps at batch 8 before it
     * exhausted the card, which read as "the kernel needs more memory than the
     * fallback" and was really a leak.
     *
     * Releasing only MARKS — tensor.c keeps the buffer valid to the queued
     * launch and to any reader until helios_end_step — so returning xhat for
     * diagnosis stays sound.
     */
    (flatXhat as NativeTensor).buffer.release(this.hl);
    (flatG as NativeTensor).buffer.release(this.hl);
    (xhat as NativeTensor).buffer.release(this.hl);
    /*
     * xhat rides along beyond the interface's three gradients — but it now
     * holds g*xhat, not xhat, because the product overwrote it in place. The
     * field is kept because ops.ts's destructure names it and because the
     * kernel's only observable intermediate is worth having a handle on; the
     * diff harness checks dx, dw and db, which are what the interface promises.
     */
    return { dx, dw, db, xhat };
  }

  /**
   * Broadcast to `shape` — the name autograd probes for.
   *
   * Without it, ops.ts broadcasts in JavaScript: softmax's backward alone spent
   * 11 ms a step at batch 128 doing by hand what expand() does. Exposing it was
   * not worth doing before — expand walked elements too, so it would have been
   * the same loop under a different name. Now that expand copies runs, it is
   * simply the faster path, and the method the interface asks for.
   */
  broadcast(t: TensorData, shape: Shape): TensorData {
    return this.expand(t, shape);
  }

  /**
   * g * dgelu(x), in one launch.
   *
   * gelu is x*sigma(2u) with u = K0(x + K1 x^3), so the derivative reuses the
   * same sigma the forward already computes:
   *
   *   d/dx = s + x*s(1-s)*2K0*(1 + 3K1 x^2)
   *
   * which is the JavaScript fallback's expression rearranged — that one writes
   * 0.5(1+tanh u) + 0.5 x sech^2(u) K0 (1+3K1x^2), and s = (1+tanh u)/2 with
   * sech^2 u = 4s(1-s).
   *
   * Binary, because it needs the pre-activation and the incoming gradient
   * together. It replaces a JavaScript loop over the whole tensor behind a
   * drain — ~33 ms a step at batch 128, growing with the batch.
   *
   * The constants come from the ADDON. Restating them here is how gelu's two
   * ended up swapped once, giving plausible wrong numbers.
   */
  geluBackward(x: TensorData, g: TensorData): TensorData {
    const dx = this.device(x), dg = this.device(g);
    const out = this.make(x.shape, "f32");
    this.check(
      this.hl.elementwise(this.hl.op.geluGrad, out.buffer.handle, dx.buffer.handle,
                          dg.buffer.handle, shapeSize(x.shape), this.hl.scalar.geluK1,
                          this.hl.scalar.geluFolded, 1, this.hl.scalar.gelu3K1,
                          this.hl.scalar.gelu2K0, 0, 5),
      "geluBackward", dx, dg);
    return out;
  }

  /**
   * g where the forward did not clamp, zero where it did.
   *
   * No comparison kernel is needed: clamp(a) equals a EXACTLY in range, so the
   * difference is exactly zero there — not merely small, which is what makes
   * the indicator sound rather than a trick.
   */
  clampBackward(a: TensorData, g: TensorData, lo: number, hi: number): TensorData {
    const da = this.device(a), dg = this.device(g);
    const out = this.make(a.shape, "f32");
    this.check(
      this.hl.elementwise(this.hl.op.clampGrad, out.buffer.handle, da.buffer.handle,
                          dg.buffer.handle, shapeSize(a.shape), lo, hi, 1e30, 1, 0, 0, 4),
      "clampBackward", da, dg);
    return out;
  }

  softmax(a: TensorData, axis?: number): TensorData {
    if (axis !== undefined && this.axisOf(a.shape, axis) !== a.shape.length - 1)
      this.unsupported("softmax over a non-final axis");
    return this.normalized("softmax", this.hl.op.softmax, a, 0);
  }

  logSoftmax(): TensorData { return this.unsupported("logSoftmax"); }

  /*
   * THE EMBEDDING GRADIENT, ON THE DEVICE — built from kernels that already
   * exist rather than from a scatter that does not.
   *
   * Backend.embeddingBackward is OPTIONAL, the Vulkan backend implements it,
   * and this one did not — so autograd took its host fallback: a JavaScript
   * loop over every token accumulating into the table's gradient, which for
   * 105M is 640 x 12,288 and 31.5 MB. It reads three device tensors to do it,
   * and reading device memory DRAINS THE QUEUE, so it also serialised the
   * backward pass. Profiling put it at the largest single host cost left.
   *
   * The obvious kernel is a scatter-add, which needs atomics, which this ISA
   * encoder does not emit. It is not needed. The gradient is
   *
   *     dW[v][c] = sum over tokens i where indices[i] == v of g[i][c]
   *              = (onehot^T @ g)[v][c],   onehot[i][v] = (indices[i] == v)
   *
   * and the one-hot itself comes out of arithmetic, no scatter and no
   * comparison operator:
   *
   *     onehot = 1 - clamp((indices[i] - v)^2, 0, 1)
   *
   * which is 1 exactly when the difference is zero and 0 otherwise. Squaring
   * is safe in f32 despite overflowing 24 bits of mantissa at this vocabulary:
   * the value is only ever compared against 1, and rounding a large number
   * never yields zero. The difference itself is exact — both operands are
   * integers below 2^24.
   *
   * It computes a 12,288 x 768 product to obtain a mostly-zero table, which is
   * wasteful in FLOPs and much cheaper in seconds than the host loop it
   * replaces, because it neither drains the queue nor moves 31.5 MB across the
   * bus twice.
   */
  embeddingBackward(indices: TensorData, gradOutput: TensorData,
                    vocabSize: number): TensorData {
    const tokens = shapeSize(indices.shape);
    const dim = gradOutput.shape[gradOutput.shape.length - 1] ?? 1;
    const wide: Shape = [tokens, vocabSize];

    const drop = (t: TensorData) => { if (isNative(t)) t.buffer.release(this.hl); };

    /*
     * THE SCATTER, which is what this gradient actually is: for each token,
     * add its row of g into dW at the row its id names. Reads about 2 MB where
     * the one-hot below moves 75 MB seven times over and then multiplies out at
     * 24 GFLOP to recover a table that is zero in all but `tokens` of its rows.
     *
     * It needs an atomic — two tokens can share an id, and on this hardware the
     * loser of that race does not fault, it silently wins — which is the only
     * reason the cheap form was not written before RED.E.ADD.F32 was captured.
     *
     * dW is zeroed first because the kernel only ADDS and only visits rows that
     * appear in ids; every other row is never addressed, so whatever the pool
     * last left there would survive.
     *
     * Capped at a block, exactly as the forward gather is: one block per token,
     * one thread per column.
     */
    if (this.hl.embeddingScatter && dim <= 1024) {
      const out = this.zeros([vocabSize, dim], "f32") as NativeTensor;
      const ids = this.make([tokens], "i32");
      const src = indices.data as ArrayLike<number>;
      const idsInts = ids.buffer.ints;
      for (let i = 0; i < tokens; i++) idsInts[i] = src[i] | 0;
      ids.buffer.commit();
      const g = this.device(this.reshape(gradOutput, [tokens, dim]));
      this.check(this.hl.embeddingScatter(out.buffer.handle, g.buffer.handle,
                                          ids.buffer.handle, tokens, dim),
                 "embeddingScatter");
      ids.buffer.release(this.hl);
      return out;
    }

    /* Each token's id spread across its row, against the column's own index
     * tiled down every row. */
    const idxRow = this.expand(this.reshape(indices, [tokens, 1]), wide);
    const colRow = this.expand(this.arange(vocabSize), wide);
    const d = this.sub(idxRow, colRow);
    drop(idxRow); drop(colRow);
    const d2 = this.mul(d, d);
    drop(d);
    const c = this.clamp(d2, 0, 1);
    drop(d2);
    const ones = this.full(wide, 1);
    const oh = this.sub(ones, c);
    drop(ones); drop(c);

    /*
     * onehot^T @ g WITHOUT forming onehot^T. The tensor-core kernel reads a
     * transposed A by changing address immediates — a fragment register already
     * costs a load and a pack, so a strided read is the same instruction count
     * — where the explicit transpose is a whole extra pass over a
     * [tokens, vocab] tensor and a second one of it held at the same time. At
     * this vocabulary that tensor is 37.7 MB a step at batch 12 and 75 at batch
     * 24, and the allocation census named this function as four of the top
     * sites, 772 MiB held, which is what the batch runs out of.
     */
    const g2 = this.reshape(gradOutput, [tokens, dim]);
    if (this.matmulTransposedA) {
      const out = this.matmulTransposedA(oh, g2);
      drop(oh);
      return out;
    }
    const ohT = this.transpose(oh, 0, 1);
    drop(oh);
    const out = this.matmul(ohT, g2);
    drop(ohT);
    return out;
  }

  /* 0..n-1 on the device, built once. It is a constant of the vocabulary, so
   * rebuilding it per step would be the host cost this class is removing. */
  private arangeCache = new Map<number, TensorData>();
  private arange(n: number): TensorData {
    let t = this.arangeCache.get(n);
    if (!t) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = i;
      t = this.fromArray(a, [n]);
      this.arangeCache.set(n, t);
    }
    return t;
  }

  embedding(weight: TensorData, indices: TensorData): TensorData {
    const dim = weight.shape[weight.shape.length - 1] ?? 1;
    const tokens = shapeSize(indices.shape);
    const dw = this.device(weight);
    /* Indices are integers; they go into device memory as raw words, not as
     * floats that happen to be whole numbers -- a float bit pattern used as an
     * index addresses somewhere absurd. */
    const ids = this.make([tokens], "i32");
    const src = indices.data as ArrayLike<number>;
    const idsInts = ids.buffer.ints;
    for (let i = 0; i < tokens; i++) idsInts[i] = src[i] | 0;
    /*
     * COMMIT, because this handle goes to the kernel WITHOUT passing `device()`.
     *
     * `device()` is where host writes are pushed back under video residency, and
     * every other operand reaches a dispatch through it. This one does not: the
     * indices are built here and handed straight to `hl.embedding`. Without the
     * commit the kernel reads whatever was in that video memory, uses it as a
     * row index, and addresses somewhere absurd -- an MMU fault (channel error
     * 0x1f) reported at whichever LATER operation happened to flush, which is
     * how this arrived labelled "layerNorm failed on the device".
     */
    ids.buffer.commit();
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

  /*
   * The gradient, in one kernel: (softmax(logits) - onehot(targets)) * scale.
   *
   * WHY THIS OVERRIDE EXISTS: autograd's fallback for it is four operations,
   * and the first of them builds the one-hot term as a HOST array — at this
   * model's shape 512 x 12,288 floats, twenty-five megabytes written element by
   * element in JavaScript and uploaded every step — before a full-size softmax,
   * subtract and scale combine it with a tensor that is zero in all but 512
   * places. None of that is arithmetic the GPU could not do without being told.
   *
   * `g` is read on the host, as the composed path also does. It is the gradient
   * of a scalar loss: one float, and in practice the constant 1.
   */
  crossEntropyBackward(logits: TensorData, targets: TensorData,
                       g: TensorData): TensorData {
    const classes = logits.shape[logits.shape.length - 1] ?? 1;
    const rows = shapeSize(logits.shape) / classes;
    const dl = this.device(logits);
    const ids = this.make([rows], "i32");
    const src = targets.data as ArrayLike<number>;
    const idsInts = ids.buffer.ints;
    for (let i = 0; i < rows; i++) idsInts[i] = src[i] | 0;
    /* Built here and handed straight to the kernel, so the write-back is
     * explicit — same as the forward. */
    ids.buffer.commit();
    const gScalar = (g.data as ArrayLike<number>)[0] ?? 1;
    const out = this.make(logits.shape.slice(), "f32");
    this.check(this.hl.crossEntropyBackward(out.buffer.handle, dl.buffer.handle,
                                            ids.buffer.handle, rows, classes,
                                            gScalar / rows),
               "crossEntropyBackward");
    ids.buffer.release(this.hl);
    return out;
  }

  crossEntropy(logits: TensorData, targets: TensorData): TensorData {
    const classes = logits.shape[logits.shape.length - 1] ?? 1;
    const rows = shapeSize(logits.shape) / classes;
    const dl = this.device(logits);
    const ids = this.make([rows], "i32");
    const src = targets.data as ArrayLike<number>;
    const idsInts = ids.buffer.ints;
    for (let i = 0; i < rows; i++) idsInts[i] = src[i] | 0;
    /* Same as embedding: built here, handed straight to the kernel, so the
     * write-back has to be explicit. */
    ids.buffer.commit();
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
    /* Same memory, second tensor — see reshape. */
    out.buffer.retain();
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

    /*
     * ONE launch for the whole batch, the plane taken from the block's Y index.
     *
     * This looped planes through the HOST -- copy in, launch, drain the queue,
     * copy out -- so a four-head attention transpose was four round trips. With
     * batching on, those drains were most of what remained: queueing launches
     * between per-operation drains saves nothing.
     */
    /*
     * ABLATION (HELIOS_ABLATE_TRANSPOSE=1): skip the launch entirely and return
     * the correctly-shaped but UNFILLED output. This corrupts the result — the
     * loss is garbage — but every other kernel runs at the same shapes, so the
     * gate's tok/s measures the step WITH the transposes removed. That is the
     * real end-to-end value of folding them, which the drained profiler only
     * estimates (it says 4.1% but cannot see overlap). Never ship it; it exists
     * to decide whether the strided-GEMM fold is worth building. A win that does
     * not move the step is not a win.
     */
    if (process.env.HELIOS_ABLATE_TRANSPOSE === "1") return out;

    this.check(
      this.hl.transpose(out.buffer.handle, da.buffer.handle, rows, cols, batch),
      "transpose",
    );
    /*
     * AN UPLOAD IS THIS FUNCTION'S OWN — and this is where the leak actually
     * lived, one level below where it looked like it lived.
     *
     * matmulTransposed releases the transpose it creates, and the free-gap
     * profiler still put 309 MB a step on that line. The stack read
     * `alloc <- device() <- transpose() <- matmulTransposed()`: what leaked was
     * not the transpose's OUTPUT but the copy `device()` makes when its
     * argument is not resident. Releasing the visible temporary while its
     * hidden upload stays is why the first fix moved nothing.
     *
     * Safe to release here because no operation in this file returns what
     * `device()` allocated, and because a release only MARKS — the memory stays
     * valid to the queued launch until helios_end_step (tensor.c).
     */
    if (da !== a && da !== out) da.buffer.release(this.hl);
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
  /*
   * ...and it stopped being free by writing `data: da.data`.
   *
   * That spelling READS the source's getter, and the getter is where the queue
   * barrier lives -- so the one operation that launches nothing was draining
   * the queue, 50 times a step, 367 us each: 18.4 ms of an 86 ms step, more
   * than any real kernel. It also flattened the barrier away, since the result
   * held a resolved array rather than a property, so a later read of the view
   * would not have flushed at all.
   *
   * Carrying the getter across fixes both. The view is computed eagerly because
   * `subarray` allocates nothing and launches nothing; only the FLUSH is
   * deferred, to the moment somebody actually reads.
   */
  reshape(a: TensorData, shape: Shape): NativeTensor {
    if (shapeSize(shape) !== shapeSize(a.shape))
      throw new Error(
        `helios-native: reshape ${a.shape} -> ${shape} changes the element count`,
      );
    const da = this.device(a);
    const self = this;
    const buffer = da.buffer;
    /* A second tensor now points at this memory, and the tape will release both
     * of them. Without this the first release frees it under the other. */
    buffer.retain();
    /* Lazily, for the same reason `make` is: taking the view materialises a
     * system-memory mirror under video residency, and a reshape is the one
     * operation that is supposed to cost nothing at all. */
    const n = shapeSize(shape);
    const cached = buffer.mapped ? buffer.floats.subarray(0, n) : null;
    return {
      shape,
      dtype: da.dtype,
      get data() {
        self.sync();
        return cached ?? buffer.floats.subarray(0, n);
      },
      buffer,
    } as NativeTensor;
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

    /*
     * A LAST-AXIS SLICE HAS A KERNEL: it is what splitting qkv does, three
     * times a layer, and doing it on the host both copies and drains.
     *
     * Only when every outer dimension is taken whole — then the output rows are
     * exactly the source rows, offset by `starts[last]`, and the launch supplies
     * the row. A slice that also cuts an outer dimension is a different mapping
     * and keeps the host path.
     */
    const lastD = shape.length - 1;
    const outerWhole = starts.slice(0, lastD).every((st, i) => st === 0 && ends[i] === shape[i]);
    if (outerWhole && extents[lastD] > 0) {
      const rowsN = shapeSize(shape) / shape[lastD];
      this.check(
        this.hl.sliceRows(out.buffer.handle, da.buffer.handle, extents[lastD],
                          shape[lastD], starts[lastD], rowsN),
        "slice", da);
      return out;
    }

    /* The source is read on the HOST below, so whatever produced it must have
     * run. Missing this gives a slice of the values that were in the buffer
     * before the producing kernel wrote them. */
    this.sync();

    /*
     * The INNERMOST dimension is always contiguous, whether or not it is taken
     * whole, and that is the difference between one copy per row and one copy
     * per ELEMENT.
     *
     * This counted only trailing dimensions taken in full. Slicing the last
     * dimension -- which is what splitting qkv does, three times a layer -- made
     * that count zero, so the run length was 1 and the loop below issued a
     * separate `set` of a single float for every element. At batch 128 that is
     * 262,144 calls to move 262,144 contiguous floats: 12.5 ms per slice, 75 ms
     * a step, the second largest cost in the model.
     *
     * A partial extent along the last axis is still one contiguous span in the
     * source, so the run starts at the last axis and grows outward through
     * dimensions taken WHOLE -- only the outermost dimension of the run may be
     * partial, because a gap in any inner one would break the span.
     */
    let runDims = 1;
    while (runDims < shape.length &&
           extents[shape.length - runDims] === shape[shape.length - runDims])
      runDims++;
    const run = extents.slice(shape.length - runDims).reduce((x, y) => x * y, 1);

    const srcStride: number[] = new Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * shape[i + 1];

    /* The run may now begin part-way into its own dimensions, so its start
     * offset counts too. It is zero whenever every dimension in the run is
     * taken whole, which is the case this used to handle. */
    const runStart = starts
      .slice(shape.length - runDims)
      .reduce((acc, st, i) => acc + st * srcStride[shape.length - runDims + i], 0);

    const outer = extents.slice(0, shape.length - runDims);
    const total = outer.reduce((x, y) => x * y, 1);
    const idx = new Array(outer.length).fill(0);
    const dst = out.buffer.floats, srcArr = da.buffer.floats;
    for (let n = 0; n < total; n++) {
      let src = runStart;
      for (let d = 0; d < outer.length; d++) src += (starts[d] + idx[d]) * srcStride[d];
      dst.set(srcArr.subarray(src, src + run), n * run);
      for (let d = outer.length - 1; d >= 0; d--) {
        if (++idx[d] < outer[d]) break;
        idx[d] = 0;
      }
    }
    out.buffer.commit();
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
  /*
   * THE MASK IS TILED BY THE KERNEL, not by an allocation.
   *
   * This used to call `expand` first, because the kernel indexed the mask with
   * the same flat index as the value and therefore needed them the same size.
   * The causal mask is [T,T] — 16 KB at T=64, and permanently resident in L2 —
   * while the scores are [B,H,T,T] at 3.93 MB. So every call allocated 3.93 MB
   * and wrote it in full to reproduce 16 KB of data, thirty-six times a step,
   * in an allocator where a step's PEAK is the bytes it allocates rather than
   * the bytes it holds live.
   *
   * The kernel now wraps the mask index at the mask's own size. That needs the
   * size to be a power of two, so the wrap is an AND rather than a division —
   * a [T,T] mask with T a power of two always is. Anything else keeps the old
   * route, which is still correct and is what the general case wants.
   */
  maskedFill(a: TensorData, mask: TensorData, value: number): TensorData {
    const n = shapeSize(a.shape);
    const maskN = shapeSize(mask.shape);
    const tiled = maskN < n && n % maskN === 0 && (maskN & (maskN - 1)) === 0;
    const dm = tiled ? this.device(mask) : this.expand(mask, a.shape);
    const da = this.device(a);
    const out = this.make(a.shape, "f32");
    this.check(this.hl.maskedFill(out.buffer.handle, da.buffer.handle,
                                  dm.buffer.handle, n, value, tiled ? maskN : 0),
               "maskedFill");
    /* Only the materialised mask is this function's own. The tiled one IS the
     * caller's tensor and releasing it would take a live buffer away. */
    if (!tiled && dm !== mask) dm.buffer.release(this.hl);
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
  /*
   * Write `src` into a column range of `dest`, in place, and return `dest`.
   *
   * WHY: q, k and v are three DISJOINT column ranges of one qkv projection, so
   * their three gradients tile the same tensor without overlapping. The tape
   * was reconstructing that by having each of them build its own full-width
   * gradient — two zero tensors and a three-way cat apiece — and then adding
   * the three together: per layer, six zero fills, three full-width writes, and
   * two read-add-write passes, about 82 MB where the data is 6. Given the
   * destination each one simply writes its own third.
   *
   * It is the same kernel `cat` already launches per piece, which is what makes
   * this an exposure rather than a new capability: cat along the last axis IS a
   * sequence of these, and the only thing missing was the ability to name a
   * destination that already exists.
   */
  writeColumns(dest: TensorData, src: TensorData, colOffset: number): TensorData {
    const dd = this.device(dest), ds = this.device(src);
    const w = src.shape[src.shape.length - 1];
    const outW = dest.shape[dest.shape.length - 1];
    const rows = shapeSize(src.shape) / w;
    if (shapeSize(dest.shape) / outW !== rows)
      throw new Error("helios-native: writeColumns row mismatch");
    if (colOffset + w > outW)
      throw new Error("helios-native: writeColumns past the destination");
    this.check(this.hl.catRows(dd.buffer.handle, ds.buffer.handle, w, outW,
                               colOffset, rows),
               "writeColumns", ds);
    return dd;
  }

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

    /*
     * ALONG THE LAST AXIS THIS IS A KERNEL, one launch per piece.
     *
     * Each source contributes a contiguous column range of every output row, so
     * it is exactly sliceRows run backwards, and the launch supplies the row.
     * On the host it was 3.4 ms a call at batch 128 — 20 ms a step plus the
     * drains, because copying on the host means READING device memory.
     *
     * Interior axes keep the host path: there each input contributes a slab per
     * outer index rather than one column range, which is a different mapping.
     */
    if (k === shape.length - 1) {
      const rowsN = shapeSize(shape) / shape[k];
      let at = 0;
      for (const t of tensors) {
        const dt = this.device(t);
        const w = t.shape[k];
        this.check(
          this.hl.catRows(out.buffer.handle, dt.buffer.handle, w, outShape[k], at, rowsN),
          "cat", dt);
        at += w;
      }
      return out;
    }

    this.sync();
    const devs = tensors.map((t) => this.device(t));
    const dst = out.buffer.floats;
    const srcArrs = devs.map((d) => d.buffer.floats);
    for (let o = 0; o < outer; o++) {
      let at = o * outSlab;
      for (let i = 0; i < devs.length; i++) {
        const slab = sizes[i] * inner;
        dst.set(
          srcArrs[i].subarray(o * slab, (o + 1) * slab),
          at,
        );
        at += slab;
      }
    }
    out.buffer.commit();
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
    const outInts = out.buffer.ints;
    for (let r = 0; r < rows; r++) {
      let best = 0;
      for (let i = 1; i < width; i++)
        if (src[r * width + i] > src[r * width + best]) best = i;
      outInts[r] = best;
    }
    out.buffer.commit();
    return out;
  }

  topk(a: TensorData, k: number, axis?: number): { values: TensorData; indices: TensorData } {
    const width = axis === undefined ? shapeSize(a.shape) : (a.shape[a.shape.length - 1] ?? 1);
    const rows = shapeSize(a.shape) / width;
    this.sync();
    const src = this.device(a).buffer.floats;
    const values = this.make(rows === 1 ? [k] : [rows, k], "f32");
    const indices = this.make(rows === 1 ? [k] : [rows, k], "i32");
    const vals = values.buffer.floats, idxInts = indices.buffer.ints;
    for (let r = 0; r < rows; r++) {
      const order = Array.from({ length: width }, (_, i) => i)
        .sort((x, y) => src[r * width + y] - src[r * width + x])
        .slice(0, k);
      for (let j = 0; j < k; j++) {
        vals[r * k + j] = src[r * width + order[j]];
        idxInts[r * k + j] = order[j];
      }
    }
    values.buffer.commit();
    indices.buffer.commit();
    return { values, indices };
  }

  /** Swap two arbitrary axes by a strided copy. No arithmetic, so no kernel. */
  private permuteSwap(a: TensorData, d0: number, d1: number): NativeTensor {
    const shape = a.shape;
    const outShape = shape.slice();
    [outShape[d0], outShape[d1]] = [outShape[d1], outShape[d0]];
    /*
     * THE MIDDLE-AXIS SWAP HAS A KERNEL NOW, and taking it skips the drain.
     *
     * A host permute must READ device memory, so it costs a queue drain as well
     * as the copy: 75 ms a step at batch 128 for an operation that performs no
     * arithmetic. It is also one of the reads that keeps tensors in system
     * memory, where the GPU reads at a measured 19.7 GB/s against ~448 from its
     * own — so removing it is worth more than the 75 ms it costs directly.
     *
     * sm_86 has no integer divide, so the kernel decoded its indices with
     * shifts and masks — and that required first T, H AND D to be powers of
     * two, then (once h moved to the grid's X index) T alone. BOTH narrowings
     * were made after a shape broke the guard, and the second one left the
     * REVERSE permute still broken: attention goes [B,T,H,D] -> [B,H,T,D] on
     * the way in and back on the way out, and on the way back the axis sitting
     * in the T position is the HEAD COUNT. 105M has ten heads, so every return
     * permute took the host fallback below — 180 queue drains and 36.3 ms a
     * step at batch 4, the largest single entry in the host profile.
     *
     * The kernel now takes h, t and b from the three grid dimensions and
     * decodes nothing, so NO dimension has to be a power of two and one program
     * serves both directions. D bounds the block, which is the only limit left.
     *
     * The lesson worth keeping: a guard narrowed twice by shapes that broke it
     * is a guard that should have been deleted, and the way to delete it was to
     * stop dividing rather than to divide more cleverly.
     */
    if (d1 === d0 + 1 && shape.length === 4 && d0 === 1 && shape[3] <= 1024) {
      const [B, T, H, D] = shape;
      const dat = this.device(a);
      const o = this.make(outShape, "f32");
      this.check(this.hl.permute(o.buffer.handle, dat.buffer.handle, T, H, D, B),
                 "permute", dat);
      return o;
    }

    this.sync();
    const da = this.device(a);
    const out = this.make(outShape, "f32");

    const srcStride: number[] = new Array(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * shape[i + 1];

    /*
     * EVERYTHING PAST THE OUTERMOST SWAPPED AXIS IS UNTOUCHED, and therefore
     * contiguous in BOTH layouts — so it moves as one run rather than as
     * elements.
     *
     * This walked every element and recomputed a full multi-dimensional index
     * for each. Attention permutes [B,T,H,D] to [B,H,T,D], which leaves D
     * contiguous on both sides, so the element loop was doing D times the
     * addressing work and D times the copies it needed. At batch 128 transpose
     * was 144.6 ms a step, 3.1 ms a call, the largest single cost in the model
     * — five times every matmul.
     *
     * The same shape of fix as slice, for the same reason: a copy that costs
     * more than the arithmetic is a loop that has not noticed its own memory is
     * already in order.
     */
    const hi = Math.max(d0, d1);
    const run = shape.slice(hi + 1).reduce((x, y) => x * y, 1);
    const outer = outShape.slice(0, hi + 1);
    const total = outer.reduce((x, y) => x * y, 1);
    const idx = new Array(outer.length).fill(0);
    const dst = out.buffer.floats, srcArr = da.buffer.floats;
    for (let o = 0; o < total; o++) {
      /* The destination walks in order; the source is the same coordinates with
       * the two axes exchanged. Axes past `hi` contribute nothing here — they
       * are the run, and the run starts at coordinate zero along them. */
      let si = 0;
      for (let d = 0; d <= hi; d++) {
        const sd = d === d0 ? d1 : d === d1 ? d0 : d;
        si += idx[d] * srcStride[sd];
      }
      dst.set(srcArr.subarray(si, si + run), o * run);
      for (let d = outer.length - 1; d >= 0; d--) {
        if (++idx[d] < outer[d]) break;
        idx[d] = 0;
      }
    }
    out.buffer.commit();
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
    if (!isNative(t)) return;
    /* ONE decrement per TENSOR, however many times it is offered.
     *
     * The buffer counts how many tensors point at it, so a tensor released
     * twice would decrement for a reference that never existed and free memory
     * a live view still holds. The tape guards its own double-releases; this
     * guards everyone else's, and it is a WeakSet so remembering costs nothing
     * once the tensor is gone. */
    if (this.freed.has(t)) return;
    this.freed.add(t);
    t.buffer.release(this.hl);
  }

  /*
   * The name the TRAINER looks for, now that releasing is survivable.
   *
   * trainer.ts builds its release callback with
   * `typeof backend.releaseGpuTensor === "function"`, and this backend spelled
   * the method `release`, so the probe failed and a real training run reclaimed
   * nothing at all. Defining the alias was unsafe until reclamation moved to the
   * step boundary: the tape releases tensors it turns out to still reference,
   * and freeing on the next flush handed that memory to the next allocation.
   *
   * Deferred reclamation makes that harmless -- a released buffer stays valid
   * and untouched until `finishStepOps` -- so the alias can exist. It is only
   * meaningful if the caller marks step boundaries; one that never does simply
   * never recycles, which is what happened before.
   */
  releaseGpuTensor(t: TensorData): void {
    this.release(t);
  }

  /**
   * End of a training step: drain, then recycle everything released during it.
   *
   * `finishStepOps` is the name trainer.ts probes for, so this is the hook a
   * real run already calls once a step. Without it nothing is ever reused: a
   * batch-16 step carves 284 buffers and takes 12 fresh 4 MiB slabs from the
   * driver, 48 MB a step it never gives back.
   */
  finishStepOps(): void {
    this.hl.endStep();
    this.pending = false;
  }

  /** Also the trainer's spelling: it probes for `syncGpu`. */
  syncGpu(): void {
    this.sync();
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
   * reallocates. `carved` is the one to watch: it should stop growing after
   * the first step, and `allocations` cannot show that any more now that a
   * trip to the driver buys a whole slab. */
  stats(): { live: number; pooled: number; allocations: number; carved: number; programs: number; enqueued: number; flushes: number } {
    this.sync();
    return this.hl.stats();
  }
}
