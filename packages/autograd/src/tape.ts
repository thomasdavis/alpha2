/**
 * Tape-based autograd engine.
 *
 * Inspired by microgpt.py's Value class, but operating on TensorData.
 * Each operation records itself on a global tape. backward() walks the
 * tape in reverse, calling each op's backward function.
 */
import type { TensorData, Backend } from "@alpha/core";

// Retain a matched-control path for trajectory and throughput ablations.  The
// ownership-forwarding path is the default; setting this diagnostic variable
// restores the historical clone-every-first-gradient behavior.
const DISABLE_GRADIENT_BUFFER_MOVE = process.env.ALPHA_DISABLE_GRADIENT_BUFFER_MOVE === "1";

// ── TapeEntry ──────────────────────────────────────────────────────────────
export interface TapeEntry {
  /** The output variable of this op */
  readonly output: Variable;
  /** Inputs to this op */
  readonly inputs: readonly Variable[];
  /** Compute gradients for each input given the output grad.
   *  The optional release callback frees intermediate GPU tensors created
   *  within the backward closure (transposes, intermediates, etc.).
   *  The optional needsGrad array indicates which inputs actually need gradients,
   *  allowing closures to skip expensive computations for non-parameter inputs. */
  /*
   * `dests` is the gradient each input ALREADY has, when it has one.
   *
   * A closure may compute straight into it and return that same tensor; the
   * tape then knows the accumulation has happened and does not add again. It is
   * optional in both directions — a closure that ignores it is correct, and so
   * is a tape that passes nothing — because the alternative to using it is a
   * matmul into a fresh tensor followed by a separate read-add-write pass over
   * the destination, which is four passes where this is two.
   */
  backward(outGrad: TensorData, backend: Backend, release?: (td: TensorData) => void,
           needsGrad?: boolean[], dests?: (TensorData | null)[]): TensorData[];
  /** Optional release hook for non-output auxiliary tensors captured by the op. */
  cleanup?(release?: (td: TensorData) => void): void;
}

// ── Variable ───────────────────────────────────────────────────────────────
let _nextId = 0;

export class Variable {
  readonly id: number;
  data: TensorData;
  grad: TensorData | null = null;
  readonly requiresGrad: boolean;

  constructor(data: TensorData, requiresGrad = false) {
    this.id = _nextId++;
    this.data = data;
    this.requiresGrad = requiresGrad;
  }
}

const ACCUM_CENSUS = typeof process !== "undefined" && process.env?.HELIOS_ACCUM_CENSUS === "1";
const accumCensus = new Map<string, number>();

/** The accumulation census, biggest first, and it clears as it reports. */
export function drainAccumCensus(): Array<[string, number]> {
  const rows = [...accumCensus.entries()].sort((a, b) => b[1] - a[1]);
  accumCensus.clear();
  return rows;
}

// ── Tape ───────────────────────────────────────────────────────────────────
export class Tape {
  private entries: TapeEntry[] = [];

  record(entry: TapeEntry): void {
    this.entries.push(entry);
  }

  /**
   * Backward pass: topological reverse through the tape.
   * Sets .grad on every Variable that requiresGrad.
   *
   * @param releaseTensor — Optional callback to explicitly release GPU buffers
   * for intermediate gradient tensors as they're consumed. Without this,
   * backward creates hundreds of temporary GPU tensors per step that
   * accumulate across steps (FinalizationRegistry is too slow to collect them),
   * causing OOM during subsequent backward passes.
   */
  backward(loss: Variable, backend: Backend, releaseTensor?: (td: TensorData) => void, initialGrad?: TensorData): void {
    // Initialize loss grad: use provided gradient or default to ones
    loss.grad = initialGrad ?? backend.ones(loss.data.shape, loss.data.dtype);

    // Guard against double-release when backward closures alias gradients
    // across multiple inputs (e.g. returning the same TensorData object).
    const released = releaseTensor ? new Set<TensorData>() : null;
    const releaseOnce = (td: TensorData): void => {
      if (!releaseTensor || !released) return;
      if (released.has(td)) return;
      released.add(td);
      releaseTensor(td);
    };

    // Walk tape in reverse
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const entry = this.entries[i];
      const outGrad = entry.output.grad;
      if (!outGrad) {
        if (entry.cleanup) entry.cleanup(releaseTensor);
        continue;
      }

      // Compute needsGrad mask only when at least one input does not require grads.
      let needsGrad: boolean[] | undefined = undefined;
      for (const inp of entry.inputs) {
        if (!inp.requiresGrad) {
          needsGrad = entry.inputs.map(v => v.requiresGrad);
          break;
        }
      }
      /*
       * Offer each input's existing gradient as a destination. Only when it is
       * the input's LAST use, because a closure writing into a buffer another
       * pending accumulation still has to read would lose that one.
       */
      const dests = entry.inputs.map((v) =>
        v.requiresGrad && v.grad ? v.grad : null);
      const inputGrads = entry.backward(outGrad, backend, releaseTensor, needsGrad, dests);

      // Track how many trainable inputs consume each gradient tensor.
      // Some ops legitimately alias grads across inputs (e.g. add/sub paths).
      // Releasing the first consumer would force later consumers down a slow
      // readback path (or worse, use-after-release semantics).
      const gradUseCount = new Map<TensorData, number>();
      for (let j = 0; j < entry.inputs.length; j++) {
        if (!entry.inputs[j].requiresGrad) continue;
        const g = inputGrads[j];
        if (!g) continue;
        gradUseCount.set(g, (gradUseCount.get(g) ?? 0) + 1);
      }

      // A freshly produced gradient has a single eventual owner in the common
      // case.  Transfer that buffer to the last input consumer instead of
      // cloning it and then releasing the original.  For aliased gradients we
      // still clone all but the last consumer, preserving independent mutable
      // accumulation buffers.  This is an ownership optimization only: no
      // arithmetic or accumulation order changes.
      const retainedAsInputGrad = new Set<TensorData>();

      for (let j = 0; j < entry.inputs.length; j++) {
        const input = entry.inputs[j];
        const g = inputGrads[j];
        if (!g) continue;
        if (!input.requiresGrad) {
          if (!gradUseCount.has(g)) {
            releaseOnce(g);
          }
          continue;
        }

        const remainingUsesBefore = gradUseCount.get(g) ?? 0;
        let transferred = false;
        if (input.grad === g) {
          /* The closure accumulated into the destination itself: the gradient
           * is already where it belongs and adding it again would double it. */
          transferred = true;
          retainedAsInputGrad.add(g);
          gradUseCount.delete(g);
          continue;
        }
        if (input.grad) {
          // In-place accumulation: A += B without allocating a new tensor
          if (ACCUM_CENSUS) {
            /*
             * WHICH gradients accumulate, by shape.
             *
             * addInplace is the second-largest item on the GPU after the GEMM —
             * 166 calls and 17% of a step — and every one of them is a full
             * read-add-write pass over a tensor that some kernel had JUST
             * finished writing. matmul already avoids its share by computing
             * into the destination and returning it, which the tape detects
             * above; nothing else does. This says what "everything else" is,
             * so the next accumulate-into-destination is aimed rather than
             * guessed. Shape is the identifier because a tape entry carries no
             * name, and in a transformer the shape is nearly as good as one.
             */
            const k = `[${g.shape}]`;
            accumCensus.set(k, (accumCensus.get(k) ?? 0) + 1);
          }
          if (backend.addInplace) {
            backend.addInplace(input.grad, g);
          } else {
            const oldGrad = input.grad;
            input.grad = backend.add(input.grad, g);
            releaseOnce(oldGrad);
          }
        } else if (!DISABLE_GRADIENT_BUFFER_MOVE && remainingUsesBefore === 1) {
          // Move the final reference.  If g is also this entry's outGrad, the
          // retained set below prevents the generic outGrad cleanup from
          // releasing a buffer that now belongs to the input variable.
          input.grad = g;
          retainedAsInputGrad.add(g);
          transferred = true;
        } else {
          input.grad = backend.clone(g);
        }

        const remainingUses = remainingUsesBefore - 1;
        if (remainingUses <= 0) {
          gradUseCount.delete(g);
          if (!transferred) releaseOnce(g);
        } else {
          gradUseCount.set(g, remainingUses);
        }
      }

      // Release this entry's outGrad — it's been fully consumed.
      // (This frees GPU buffers from grad accumulation of previous entries.)
      if (outGrad) {
        if (!retainedAsInputGrad.has(outGrad)) releaseOnce(outGrad);
        entry.output.grad = null;
      }

      // Release this entry's forward activation data — all entries j > i have
      // already been processed (we walk backward), so no future backward closure
      // will reference this output. This dramatically reduces peak GPU memory
      // during backward (from O(tape_size) to O(current_entry) activations).
      // Null out .data so clear() won't double-free.
      if (releaseTensor) {
        releaseOnce(entry.output.data);
        (entry.output as { data: TensorData | null }).data = null!;
      }

      if (entry.cleanup) entry.cleanup(releaseTensor);
    }
  }

  /**
   * Clear the tape, releasing all recorded entries and their GPU resources.
   *
   * @param releaseTensor — Optional callback to explicitly release GPU buffers
   * for intermediate TensorData objects. Without this, GPU buffers are only
   * freed when V8's FinalizationRegistry fires, which is unreliable for
   * timely cleanup and leads to GPU OOM during multi-step training.
   * @param keepOutput — Optional output Variable whose data buffer should be
   * retained while clearing all other intermediates.
   *
   * All entry.output Variables are intermediates created by record() — never
   * model parameters — so releasing their .data and .grad buffers is safe
   * after backward() has completed.
   */
  clear(releaseTensor?: (td: TensorData) => void, keepOutput?: Variable): void {
    if (releaseTensor) {
      for (const entry of this.entries) {
        const keep = !!keepOutput && entry.output.id === keepOutput.id;
        // Release intermediate output GPU buffers (forward pass results)
        // Skip if already released by backward() (data nulled out)
        if (!keep && entry.output.data) {
          releaseTensor(entry.output.data);
        }
        // Release accumulated gradient GPU buffers on intermediates
        if (entry.output.grad) {
          releaseTensor(entry.output.grad);
        }
        if (entry.cleanup) entry.cleanup(releaseTensor);
        entry.output.grad = null;
      }
    } else {
      for (const entry of this.entries) {
        if (entry.cleanup) entry.cleanup();
        entry.output.grad = null;
      }
    }
    this.entries = [];
  }

  get size(): number {
    return this.entries.length;
  }
}
