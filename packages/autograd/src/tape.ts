/**
 * Tape-based autograd engine.
 *
 * Inspired by microgpt.py's Value class, but operating on TensorData.
 * Each operation records itself on a global tape. backward() walks the
 * tape in reverse, calling each op's backward function.
 */
import type { TensorData, Backend } from "@alpha/core";

// ── TapeEntry ──────────────────────────────────────────────────────────────
export interface TapeEntry {
  /** The output variable of this op */
  readonly output: Variable;
  /** Inputs to this op */
  readonly inputs: readonly Variable[];
  /** Compute gradients for each input given the output grad */
  backward(outGrad: TensorData, backend: Backend): TensorData[];
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

// ── Tape ───────────────────────────────────────────────────────────────────
export class Tape {
  private entries: TapeEntry[] = [];

  record(entry: TapeEntry): void {
    this.entries.push(entry);
  }

  /**
   * Backward pass: topological reverse through the tape.
   * Sets .grad on every Variable that requiresGrad.
   */
  backward(loss: Variable, backend: Backend): void {
    // Initialize loss grad to ones (scalar)
    loss.grad = backend.ones(loss.data.shape, loss.data.dtype);

    // Walk tape in reverse
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const entry = this.entries[i];
      const outGrad = entry.output.grad;
      if (!outGrad) continue;

      const inputGrads = entry.backward(outGrad, backend);

      for (let j = 0; j < entry.inputs.length; j++) {
        const input = entry.inputs[j];
        if (!input.requiresGrad) continue;
        const g = inputGrads[j];
        if (!g) continue;

        if (input.grad) {
          input.grad = backend.add(input.grad, g);
        } else {
          input.grad = backend.clone(g);
        }
      }
    }
  }

  clear(): void {
    this.entries = [];
  }

  get size(): number {
    return this.entries.length;
  }
}
