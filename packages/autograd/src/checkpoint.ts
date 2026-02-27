/**
 * Activation checkpointing: trades compute for memory.
 *
 * During forward, runs the segment function but discards intermediate
 * activations, keeping only the output value. During backward, recomputes
 * the segment's forward pass to reconstruct intermediates, then
 * backpropagates through them.
 *
 * Memory savings: O(layers * ops_per_layer) → O(layers + ops_per_layer)
 * Compute cost: ~33% increase (one extra forward per checkpointed segment)
 *
 * The backward closure re-runs fn() which uses captured parameter Variables.
 * Gradients for those parameters are accumulated directly by the inner
 * backward pass — the outer tape doesn't need to know about them.
 *
 * NOTE: Not compatible with non-deterministic ops (e.g., dropout with
 * Math.random()). Use only when dropout=0 or with deterministic RNG.
 */
import type { Backend, TensorData } from "@alpha/core";
import { Variable, Tape } from "./tape.js";

type Ctx = { tape: Tape; backend: Backend; release?: (td: TensorData) => void };

/**
 * Checkpoint a segment of computation.
 *
 * @param ctx - Outer autograd context (tape + backend)
 * @param fn - Segment function: takes (innerCtx, input) and returns output.
 *             May capture parameter Variables in its closure — their gradients
 *             are handled by the inner backward during recomputation.
 * @param input - Input variable to the segment (e.g., layer input x)
 * @returns Output variable recorded on the outer tape as a single entry
 */
export function checkpoint(
  ctx: Ctx,
  fn: (innerCtx: Ctx, input: Variable) => Variable,
  input: Variable,
): Variable {
  const B = ctx.backend;

  // Forward: run fn on a throwaway tape to get the output value.
  // The inner tape captures all intermediate ops, but we discard it.
  const tmpTape = new Tape();
  const tmpCtx: Ctx = { tape: tmpTape, backend: B };
  const tmpOutput = fn(tmpCtx, input);

  // Keep ownership of the computed output buffer and release only intermediates.
  const outputData = tmpOutput.data;

  // Discard the throwaway tape — explicitly release GPU buffers for
  // intermediate activations. Without passing release here, all
  // intermediates leak until FinalizationRegistry fires (too slow).
  tmpTape.clear(ctx.release, tmpOutput);

  // Record a single entry on the outer tape: saves only input + fn reference.
  // This replaces what would be ~30 entries per transformer layer.
  const out = new Variable(outputData, true);
  ctx.tape.record({
    output: out,
    inputs: [input],
    backward: (outGrad: TensorData, backend: Backend, release?: (td: TensorData) => void) => {
      // Recompute: create a fresh input variable and tape, re-run forward.
      // The fresh input variable wraps the same underlying data but has
      // requiresGrad=true so the inner tape tracks gradients through it.
      const reInput = new Variable(input.data, true);
      const reTape = new Tape();
      const reCtx: Ctx = { tape: reTape, backend };
      const reOutput = fn(reCtx, reInput);

      // Backward through the recomputed tape with the upstream gradient.
      // This sets gradients on:
      //   - reInput (which we extract and return to the outer tape)
      //   - All captured parameter Variables (accumulated directly on their .grad)
      reTape.backward(reOutput, backend, release, backend.clone(outGrad));

      // Extract the input gradient before clearing.
      const inputGrad = reInput.grad;

      // Clean up recomputed tape's intermediate buffers.
      reTape.clear(release);

      return [inputGrad ?? backend.zeros(input.data.shape, input.data.dtype)];
    },
  });

  return out;
}
