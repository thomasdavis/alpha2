/**
 * Free energy proxy: F = loss + beta * weight_entropy.
 * Inspired by symbiogenesis/training.py — written from scratch.
 */

/** Compute free energy proxy. */
export function computeFreeEnergy(loss: number, beta: number, weightEntropy: number): number {
  return loss + beta * weightEntropy;
}
