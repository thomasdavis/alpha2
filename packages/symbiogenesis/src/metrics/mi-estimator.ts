/**
 * Mutual information estimation via binned entropy.
 * Inspired by symbiogenesis/mi_estimator.py — written from scratch.
 *
 * Per-neuron independence assumption, uniform binning, joint entropy estimation.
 * Adapted for transformer hidden states.
 */

export interface MIProfile {
  mi_input_repr: number;
  mi_repr_output: number;
  mi_compression: number;
}

/**
 * Estimate mutual information profile from input, hidden representation, and output.
 *
 * @param input - Flattened input activations [N]
 * @param representation - Flattened hidden representation [N]
 * @param output - Flattened output/label values [N]
 * @param numBins - Number of bins for discretization
 */
export function estimateMI(
  input: Float32Array,
  representation: Float32Array,
  output: Float32Array,
  numBins: number,
): MIProfile {
  const mi_input_repr = binMI(input, representation, numBins);
  const mi_repr_output = binMI(representation, output, numBins);
  const mi_compression = mi_input_repr > 1e-10 ? mi_repr_output / mi_input_repr : 0;

  return { mi_input_repr, mi_repr_output, mi_compression };
}

/** Estimate mutual information between two variables via binned histogram. */
function binMI(x: Float32Array, y: Float32Array, numBins: number): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  // Find ranges
  let xMin = x[0], xMax = x[0], yMin = y[0], yMax = y[0];
  for (let i = 1; i < n; i++) {
    if (x[i] < xMin) xMin = x[i];
    if (x[i] > xMax) xMax = x[i];
    if (y[i] < yMin) yMin = y[i];
    if (y[i] > yMax) yMax = y[i];
  }

  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  if (xRange < 1e-10 || yRange < 1e-10) return 0;

  const xBinWidth = xRange / numBins;
  const yBinWidth = yRange / numBins;

  // Joint histogram
  const joint = new Float64Array(numBins * numBins);
  const margX = new Float64Array(numBins);
  const margY = new Float64Array(numBins);

  for (let i = 0; i < n; i++) {
    const xBin = Math.min(Math.floor((x[i] - xMin) / xBinWidth), numBins - 1);
    const yBin = Math.min(Math.floor((y[i] - yMin) / yBinWidth), numBins - 1);
    joint[xBin * numBins + yBin]++;
    margX[xBin]++;
    margY[yBin]++;
  }

  // MI = Σ p(x,y) * log2(p(x,y) / (p(x) * p(y)))
  let mi = 0;
  for (let i = 0; i < numBins; i++) {
    if (margX[i] === 0) continue;
    const px = margX[i] / n;
    for (let j = 0; j < numBins; j++) {
      if (joint[i * numBins + j] === 0 || margY[j] === 0) continue;
      const pxy = joint[i * numBins + j] / n;
      const py = margY[j] / n;
      mi += pxy * Math.log2(pxy / (px * py));
    }
  }

  return Math.max(0, mi);
}
