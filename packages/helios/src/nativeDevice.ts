/**
 * nativeDevice.ts — loading the from-scratch addon, and the handle wrapper.
 *
 * WHAT: resolves `helios_native.node`, exposes its surface with types, and
 * wraps a tensor handle together with the zero-copy view over its memory.
 *
 * WHY THE VIEW AND THE HANDLE TRAVEL TOGETHER: the ArrayBuffer the addon hands
 * back is a window onto pooled device memory, and it stays valid only while the
 * tensor does. The handle is generation-checked on the native side; the view is
 * not, because checking on every element access would defeat the point of not
 * copying. Keeping them in one object is what lets `free` invalidate both at
 * once, so a use-after-free is a null dereference here rather than a silent
 * read of somebody else's tensor.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no fallback. If the addon is missing or
 * the device will not open, this throws. A backend that quietly degrades to the
 * CPU is how a test suite comes to pass while never exercising the GPU at all --
 * which happened here before, and is the reason the gate script refuses a bare
 * green exit.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

/** The operation selectors the addon exports by name. */
export interface NativeOps {
  readonly [name: string]: number;
}

export interface NativeAddon {
  open(index: number): boolean;
  close(): void;
  alloc(bytes: number): number;
  /* C[M,N] = A[K,M]^T @ B[K,N]. Returns false when the shape cannot use the
   * tensor-core path — the scalar kernel has no transposed-A form. */
  matmulTransposedA?(out: number, a: number, b: number, M: number, N: number,
                     K: number, batch: number): boolean;
  /* C += A@B. `layout` is 0 for A@B, 1 for A@B^T, 2 for A^T@B. */
  matmulAccumulate?(out: number, a: number, b: number, M: number, N: number,
                    K: number, batch: number, layout: number): boolean;
  /* out = s * (g - sum_j g_j s_j) over each row. */
  softmaxBackward?(out: number, s: number, g: number, width: number,
                   rows: number): boolean;
  /* Host-mapped whatever the residency policy is. Optional so a stale addon
   * still loads — without it VIDMEM simply has nowhere to stage. */
  allocHost?(bytes: number): number;
  hostVisible?(handle: number): boolean;
  free(handle: number): void;
  /* Null for a device-resident tensor: video memory is not mapped, by design. */
  view(handle: number): ArrayBuffer | null;
  stats(): {
    live: number;
    pooled: number;
    /** Trips to the driver — slabs, not tensors. Rare by design. */
    allocations: number;
    /** Requests the free list could not serve. This is the one that reveals a
     * pool that never recycles; `allocations` no longer can. */
    carved: number;
    programs: number;
    enqueued: number;
    flushes: number;
  };
  readonly op: NativeOps;
  /** The folded constants the kernels expect, by name — see nativeBackend. */
  readonly scalar: NativeOps;
  /** Drain the launch queue. Must precede any host read of device memory. */
  flush(): boolean;
  /** Drain, then return every buffer released during the step to the pool.
   * Reclamation lives here rather than in flush because a released buffer must
   * stay valid for the rest of the step — see helios_tensor_free. */
  endStep(): boolean;
  /** Device identity in the shape the NVIDIA gate checks: vendorId 0x10de, and
   * channelLive, which distinguishes "ran on a GPU" from "ran at all". */
  deviceInfo(): {
    vendorId: number;
    gpuId: number;
    minor: number;
    name: string;
    channelLive: boolean;
  } | null;

  elementwise(
    op: number, out: number, a: number, b: number, n: number,
    s0: number, s1: number, s2: number, s3: number, s4: number, s5: number,
    nscalars: number,
  ): boolean;
  reduce(mean: number, out: number, a: number, scratch: number, n: number): boolean;
  /** One value per row: one block per row, `width` threads each. */
  reduceRows(out: number, a: number, width: number, rows: number): boolean;
  normalize(op: number, out: number, a: number, width: number, rows: number, eps: number): boolean;
  /* dx and xhat together; dw and db reduce down the other axis and are the
   * caller's job. Slot order is the kernel's and is not checked. */
  layerNormBackward(dx: number, xhat: number, x: number, g: number, w: number,
                    width: number, rows: number, eps: number): boolean;
  matmul(out: number, a: number, b: number, M: number, N: number, K: number, batch: number): boolean;
  /* C = A @ B^T with B stored [N,K]. Optional so a stale addon still loads. */
  matmulTransposed?(out: number, a: number, b: number, M: number, N: number, K: number, batch: number): boolean;
  transpose(out: number, a: number, rows: number, cols: number, batch: number): boolean;
  /** out[b][h][t][d] = in[b][t][h][d]; `planes` is b*t*h. T, H and D must be
   * powers of two — the kernel decomposes the plane index with shifts. */
  permute(out: number, a: number, T: number, H: number, D: number, planes: number): boolean;
  /** out[r][c] = in[r][start + c], over `rows` rows. */
  sliceRows(out: number, a: number, W: number, srcW: number, start: number, rows: number): boolean;
  /** out[r][start + c] = in[r][c], over `rows` rows — the inverse of sliceRows. */
  catRows(out: number, a: number, W: number, dstW: number, start: number, rows: number): boolean;
  /** mode 0 tiles a [W] vector down `rows`; mode 1 spreads one value per row. */
  broadcastRows(out: number, a: number, mode: number, W: number, rows: number): boolean;
  embedding(out: number, table: number, ids: number, tokens: number, dim: number): boolean;
  slice(out: number, a: number, count: number, offset: number, stride: number): boolean;
  causalMask(out: number, a: number, rows: number, cols: number): boolean;
  maskedFill(out: number, a: number, mask: number, n: number, value: number): boolean;
  cast(toF16: number, out: number, a: number, n: number): boolean;
  dropoutMask(out: number, n: number, seed: number, counter: number, p: number): boolean;
  crossEntropy(out: number, logits: number, targets: number, rows: number, classes: number): boolean;
  crossEntropyBackward(out: number, logits: number, targets: number, rows: number, classes: number, scale: number): boolean;
  embeddingScatter(gradTable: number, gradOut: number, ids: number, tokens: number, dim: number): boolean;
  residualRms(out: number, x: number, res: number, w: number, width: number, rows: number, eps: number): boolean;
  residualDropout(out: number, x: number, res: number, mask: number, width: number, rows: number, scale: number): boolean;
  adamw(param: number, grad: number, m: number, v: number, n: number,
        b1: number, b2: number, lr: number, eps: number, wd: number): boolean;
}

let addon: NativeAddon | null = null;

/** Load the addon, opening device `index` on first use. Throws if it cannot. */
export function nativeAddon(index = 0): NativeAddon {
  if (addon) return addon;

  const here = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    join(here, "..", "native", "helios_native.node"),
    join(here, "..", "..", "native", "helios_native.node"),
    join(process.cwd(), "packages", "helios", "native", "helios_native.node"),
  ];
  const found = candidates.find((p) => existsSync(p));
  if (!found) {
    throw new Error(
      `helios: helios_native.node not found. Build it with ` +
        `\`node packages/helios/native/build-stack.mjs\`. Looked in:\n  ` +
        candidates.join("\n  "),
    );
  }

  const req = createRequire(import.meta.url);
  const mod = req(found) as NativeAddon;
  if (!mod.open(index)) throw new Error("helios: device would not open");
  addon = mod;
  return mod;
}

/**
 * A device tensor: the handle, and the view over the same memory.
 *
 * `bytes` is what was asked for, not what the pool rounded up to, so a view is
 * exactly the tensor and never spills into the slack at the end of its buffer.
 */
/*
 * A counter that moves whenever ANY kernel is enqueued.
 *
 * A staging mirror is a copy of device memory taken at a moment, and it stops
 * being true the moment a kernel writes anything. Tracking which buffer a
 * dispatch wrote would be exact and would require every one of forty operations
 * to declare its output; this is the conservative version of the same fact --
 * after any dispatch, assume every mirror is stale.
 *
 * It is what makes the mirror correct rather than merely fast. Without it a
 * buffer read on the host, written by a kernel, and read again returned the
 * FIRST read's bytes: the loss came back 4.1888 to 4.1910 across runs against
 * cpu_ref's 4.1904, varying with pool reuse order, which reads as a race and is
 * really a cache with no invalidation.
 *
 * Refreshing per dispatch would be the naive reading of that and would copy the
 * buffer once per element in every host loop. Per GENERATION is the right
 * granularity: a host loop enqueues nothing, so it sees one copy.
 */
let deviceGeneration = 1;

/*
 * THE SAFETY NET THE NATIVE BACKEND DID NOT HAVE.
 *
 * The Vulkan backend registers every buffer with a FinalizationRegistry
 * (gpuCleanup) so a tensor JavaScript drops without an explicit release still
 * returns its memory. This one had nothing, so a buffer nobody released held
 * its pool slot FOREVER — and "nobody released" is the normal case for any
 * temporary an operation makes internally, which is four separate bugs this
 * session and counting.
 *
 * It shows as a leak that garbage collection cannot touch: forcing gc() between
 * steps moved the pool's size-class histogram by exactly zero, because nothing
 * was ever registered to be collected.
 *
 * Freeing from a finalizer is as safe as freeing from release(): helios_tensor_
 * free only MARKS the slot, and the memory stays valid until helios_end_step —
 * so a kernel still holding the buffer is unaffected, which is the same
 * deferral the tape already relies on.
 *
 * This is a NET, not a policy. Explicit release is still what makes reclamation
 * prompt; the registry only stops a missed one from being permanent.
 */
const reclaim = new FinalizationRegistry<{ hl: NativeAddon; handle: number }>(
  (info) => { info.hl.free(info.handle); },
);

export class NativeBuffer {
  readonly handle: number;
  /*
   * The direct view, when the memory is host-mapped — and NULL when it is not.
   *
   * Under HELIOS_VIDMEM a tensor lives in video memory with no host mapping at
   * all, because the aperture that would provide one is 256 MiB against a
   * batch-128 step's 1.4 GB, and reads through it are uncached PCIe round trips
   * per element. The GPU gets 111.8 GB/s from video memory against 19.7 from
   * system memory, so the trade is: kernels get 5.7x, and the handful of places
   * that read on the host pay one sequential copy instead of a mapping.
   */
  private readonly direct: Float32Array | null;
  private readonly directInts: Int32Array | null;
  private readonly hl: NativeAddon;
  private readonly elements: number;

  /* The staging mirror, allocated only if somebody actually reads on the host. */
  private staging: NativeBuffer | null = null;
  /*
   * The generation the mirror was taken at; 0 means never.
   *
   * Comparing it against `deviceGeneration` is what makes a getter called in a
   * loop copy once rather than every turn: a host loop enqueues nothing, so the
   * generation does not move and the mirror stays current. Without that the host
   * permute fallback -- `out.buffer.floats.set(...)` once per plane -- would
   * fire one device-to-host copy per plane, which is how a correct staging
   * design becomes slower than the mapping it replaced.
   */
  private stagingGen = 0;
  /*
   * Set on every host read, because a Float32Array cannot report being written.
   *
   * The alternative is to trust callers to declare it, and the callers are
   * every autograd fallback and every host copy path in the backend. A
   * conservative write-back costs one sequential copy on a buffer the host
   * already touched; a missed one is a gradient computed from stale memory,
   * which is finite, plausible and wrong.
   */
  private mirrorDirty = false;
  /*
   * HOW MANY TENSORS POINT HERE, not whether one does.
   *
   * A reshape is a different view of the SAME memory, and so is the scalar
   * cross-entropy returns -- two TensorData objects, one buffer. A plain
   * live/dead flag made releasing either of them free the memory the other was
   * still using, which is why tensor.c says wiring the tape's release callback
   * "naively frees tensors the graph still references". It is not the tape
   * being naive; it is the buffer not knowing it is shared.
   *
   * The failure is at least loud: the generation check rejects the stale
   * handle, helios_tensor_addr returns 0 and the dispatch fails rather than
   * reading somebody else's tensor.
   */
  private rc = 1;

  private constructor(hl: NativeAddon, handle: number, buffer: ArrayBuffer | null,
                      elements: number) {
    this.hl = hl;
    this.handle = handle;
    this.elements = elements;
    this.direct = buffer ? new Float32Array(buffer) : null;
    this.directInts = buffer ? new Int32Array(buffer) : null;
    /*
     * SHADOW THE GETTERS WITH OWN FIELDS when the memory is mapped.
     *
     * `floats` has to be a getter for the device-resident case, and making it
     * one cost the DEFAULT path 10-20%: it is read inside every host copy loop
     * in the backend, and a prototype accessor is a call where a field is an
     * inline-cached load. Defining an own data property here shadows the
     * accessor for these instances, so the mapped path gets its field back and
     * the device-resident path keeps the getter.
     *
     * defineProperty rather than assignment: the prototype accessor has no
     * setter, so `this.floats = view` would throw.
     */
    if (buffer) {
      Object.defineProperty(this, "floats", { value: this.direct, enumerable: false });
      Object.defineProperty(this, "ints", { value: this.directInts, enumerable: false });
    }
    /* The token is `this`, so release() can unregister and the two paths cannot
     * both free the same handle. */
    reclaim.register(this, { hl, handle }, this);
  }

  static alloc(hl: NativeAddon, elements: number): NativeBuffer {
    const handle = hl.alloc(elements * 4);
    if (handle === 0) throw new Error(`helios: allocation of ${elements} floats failed`);
    /* Null is the normal answer for a device-resident tensor, not a failure —
     * `view` returns it precisely so nothing does pointer arithmetic on a slab
     * that was never mapped. */
    return new NativeBuffer(hl, handle, hl.view(handle), elements);
  }

  /** Host-mapped whatever the residency policy is: the staging pool. */
  static allocHost(hl: NativeAddon, elements: number): NativeBuffer {
    const handle = hl.allocHost ? hl.allocHost(elements * 4) : hl.alloc(elements * 4);
    if (handle === 0) throw new Error(`helios: host allocation of ${elements} floats failed`);
    const view = hl.view(handle);
    if (!view) throw new Error("helios: host-visible handle has no view");
    return new NativeBuffer(hl, handle, view, elements);
  }

  /** True when reads and writes need no copy — the default, non-VIDMEM path. */
  get mapped(): boolean {
    return this.direct !== null;
  }

  /**
   * A kernel may have written here, so the mirror is stale.
   *
   * Called where output buffers are made rather than at every launch: a buffer
   * is written by the op that allocates it, and an in-place op would have to say
   * so anyway.
   */
  /** Every dispatch invalidates every mirror. Called from the backend's `check`,
   * which is the one point every dispatch in the stack passes through. */
  static invalidateMirrors(): void {
    deviceGeneration++;
  }

  private mirror(): NativeBuffer {
    if (!this.staging) this.staging = NativeBuffer.allocHost(this.hl, this.elements);
    /*
     * A DIRTY MIRROR IS THE AUTHORITY, and refreshing it loses the writes.
     *
     * The autograd fallbacks ACCUMULATE: `B.zeros(...)` then `dw.data[j] += ...`
     * over many reads of `.data`. Every one of those reads is a mirror(), and
     * dispatches happen in between, so an unconditional refresh copied device
     * memory back over the partial sum -- and device memory still held the zeros
     * that `zeros` committed. The gradient came back exactly 0, which is why it
     * was the final LayerNorm's weight that reported it: `lnF.weight grad[1]:
     * device 0 vs reference -0.0054929466`.
     *
     * So once the host has written, the host owns the buffer until `device()`
     * commits it. The limitation this accepts, stated plainly: a buffer written
     * by the host AND by a kernel between two commits cannot be merged, and the
     * host's copy wins. No path in this backend does that -- a buffer is either
     * being accumulated on the host or produced by kernels -- and a merge would
     * need per-element ownership, which is a different design.
     */
    if (!this.mirrorDirty && this.stagingGen !== deviceGeneration) {
      /* One sequential copy, device to system, through the same elementwise
       * kernel everything else uses. The flush is not optional: the copy has to
       * have RUN before the bytes are read, and reading them is what happens
       * next by definition. */
      this.hl.elementwise(this.hl.op.copy, this.staging.handle, this.handle,
                          this.handle, this.elements, 0, 0, 0, 0, 0, 0, 0);
      /* A FAILED FLUSH HERE IS SILENT GARBAGE. The copy is the only thing
       * standing between device memory and a host loop, so a drain that timed
       * out or faulted must not be mistaken for one that completed — the reader
       * would get whatever the staging buffer happened to hold. */
      if (!this.hl.flush()) {
        const err = (this.hl.stats() as { lastError?: number }).lastError ?? 0;
        throw new Error(
          `helios: flush failed staging ${this.elements} floats (${(this.elements * 4 / 1048576).toFixed(1)} MB) ` +
          `for the host — channel error 0x${err.toString(16)}`);
      }
      /* The copy is itself a dispatch, so read the generation AFTER it: the
       * mirror is current as of the world that includes the copy. */
      this.stagingGen = deviceGeneration;
    }
    this.mirrorDirty = true;
    return this.staging;
  }

  /**
   * Push host writes back to the device.
   *
   * Called when a tensor is resolved as an operand, which is the moment before
   * any kernel could read it and after every host write that could have
   * happened. Cheap and silent when nothing was written or nothing is mirrored.
   */
  commit(): void {
    if (!this.mirrorDirty || !this.staging) return;
    this.mirrorDirty = false;
    this.hl.elementwise(this.hl.op.copy, this.handle, this.staging.handle,
                        this.staging.handle, this.elements, 0, 0, 0, 0, 0, 0, 0);
    /* The device now matches the mirror, so a read that follows must not copy
     * back over the writes it just made. */
    this.stagingGen = deviceGeneration;
  }

  get floats(): Float32Array {
    return this.direct ?? this.mirror().direct!;
  }

  get ints(): Int32Array {
    return this.directInts ?? this.mirror().directInts!;
  }

  /** Another tensor now points at this memory and must release it in turn. */
  retain(): void {
    if (this.rc > 0) this.rc++;
  }

  release(hl: NativeAddon): void {
    if (this.rc <= 0) return;
    if (--this.rc === 0) {
      /*
       * PUSH THE MIRROR FIRST: in this backend a release is deferred, so a
       * released buffer is still readable and still gets used.
       *
       * helios_tensor_free only MARKS; the handle stays valid and the memory
       * untouched until the step boundary, because the tape releases tensors it
       * turns out to still reference. So a buffer released with host writes
       * pending is not dead, it is a buffer whose writes have to land -- and
       * dropping the mirror here made `commit()` a no-op afterwards, so those
       * writes were silently lost and the loss came back near but not equal to
       * cpu_ref, differently on each run as the release order varied.
       *
       * The staging buffer is released, not forgotten: its handle is deferred by
       * exactly the same rule, so it stays valid for the rest of the step and a
       * later read through it is as safe as a read through the parent.
       */
      this.commit();
      if (this.staging) this.staging.release(hl);
      /* Explicit release wins; take it off the net so the finalizer cannot free
       * a handle the pool has already recycled to somebody else. */
      reclaim.unregister(this);
      hl.free(this.handle);
    }
  }

  get released(): boolean {
    return this.rc <= 0;
  }
}
