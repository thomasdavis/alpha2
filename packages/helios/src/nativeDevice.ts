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
  free(handle: number): void;
  view(handle: number): ArrayBuffer | null;
  stats(): {
    live: number;
    pooled: number;
    allocations: number;
    programs: number;
    enqueued: number;
    flushes: number;
  };
  readonly op: NativeOps;
  /** The folded constants the kernels expect, by name — see nativeBackend. */
  readonly scalar: NativeOps;
  /** Drain the launch queue. Must precede any host read of device memory. */
  flush(): boolean;
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
  matmul(out: number, a: number, b: number, M: number, N: number, K: number): boolean;
  transpose(out: number, a: number, rows: number, cols: number): boolean;
  embedding(out: number, table: number, ids: number, tokens: number, dim: number): boolean;
  slice(out: number, a: number, count: number, offset: number, stride: number): boolean;
  causalMask(out: number, a: number, rows: number, cols: number): boolean;
  maskedFill(out: number, a: number, mask: number, n: number, value: number): boolean;
  cast(toF16: number, out: number, a: number, n: number): boolean;
  dropoutMask(out: number, n: number, seed: number, counter: number, p: number): boolean;
  crossEntropy(out: number, logits: number, targets: number, rows: number, classes: number): boolean;
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
export class NativeBuffer {
  readonly handle: number;
  readonly floats: Float32Array;
  readonly ints: Int32Array;
  private live = true;

  private constructor(handle: number, buffer: ArrayBuffer) {
    this.handle = handle;
    this.floats = new Float32Array(buffer);
    this.ints = new Int32Array(buffer);
  }

  static alloc(hl: NativeAddon, elements: number): NativeBuffer {
    const handle = hl.alloc(elements * 4);
    if (handle === 0) throw new Error(`helios: allocation of ${elements} floats failed`);
    const view = hl.view(handle);
    if (!view) throw new Error("helios: allocated handle has no view");
    return new NativeBuffer(handle, view);
  }

  release(hl: NativeAddon): void {
    if (!this.live) return;
    this.live = false;
    hl.free(this.handle);
  }

  get released(): boolean {
    return !this.live;
  }
}
