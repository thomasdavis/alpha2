/**
 * Shannon entropy of weight magnitude distributions.
 * Inspired by symbiogenesis/model.py:Unit.weight_entropy — written from scratch.
 *
 * Algorithm: flatten all model parameters, take absolute values,
 * discretize into 100 uniform bins between 0 and max(|w|),
 * compute normalized histogram, return Shannon entropy in bits.
 */
import type { TensorData } from "@alpha/core";

const NUM_BINS = 100;

/** Compute Shannon entropy (bits) of the weight magnitude distribution. */
export function computeWeightEntropy(params: Map<string, TensorData>): number {
  // Collect all weight magnitudes
  let totalSize = 0;
  for (const [, td] of params) totalSize += td.data.length;
  if (totalSize === 0) return 0;

  // Find max absolute value across all parameters
  let maxAbs = 0;
  for (const [, td] of params) {
    const arr = td.data;
    for (let i = 0; i < arr.length; i++) {
      const a = Math.abs(arr[i] as number);
      if (a > maxAbs) maxAbs = a;
    }
  }
  if (maxAbs === 0) return 0;

  // Build histogram with 100 bins
  const counts = new Float64Array(NUM_BINS);
  const binWidth = maxAbs / NUM_BINS;
  for (const [, td] of params) {
    const arr = td.data;
    for (let i = 0; i < arr.length; i++) {
      const a = Math.abs(arr[i] as number);
      const bin = Math.min(Math.floor(a / binWidth), NUM_BINS - 1);
      counts[bin]++;
    }
  }

  // Shannon entropy: H = -Σ p_i * log2(p_i)
  let entropy = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    if (counts[i] > 0) {
      const p = counts[i] / totalSize;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}
