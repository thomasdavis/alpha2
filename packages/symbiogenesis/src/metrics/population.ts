/**
 * Population entropy and diversity metrics.
 * Inspired by symbiogenesis/monitor.py:IterationMetrics — written from scratch.
 */

/**
 * Compute population entropy from a sliding window of recent loss values.
 * Uses softmax normalization to convert losses to a probability distribution,
 * then computes Shannon entropy in nats (natural log).
 */
export function computePopulationEntropy(recentLosses: number[]): number {
  const n = recentLosses.length;
  if (n <= 1) return 0;

  // Softmax normalization (with numerical stability via max subtraction)
  let maxLoss = recentLosses[0];
  for (let i = 1; i < n; i++) {
    if (recentLosses[i] > maxLoss) maxLoss = recentLosses[i];
  }

  // Negate losses: lower loss = higher "fitness" = higher probability
  const exps = new Float64Array(n);
  let expSum = 0;
  for (let i = 0; i < n; i++) {
    exps[i] = Math.exp(-(recentLosses[i] - maxLoss));
    expSum += exps[i];
  }

  if (expSum === 0) return 0;

  // Shannon entropy in nats: H = -Σ p_i * ln(p_i)
  let entropy = 0;
  for (let i = 0; i < n; i++) {
    const p = exps[i] / expSum;
    if (p > 0) entropy -= p * Math.log(p);
  }

  return entropy;
}

/**
 * Get the activation distribution as a JSON-serializable map.
 * For single-run mode: all layers use the same activation.
 * For search mode: aggregated across the candidate population.
 */
export function getActivationDistribution(
  activation: string,
  nLayers: number,
): Record<string, number> {
  return { [activation]: nLayers };
}

/**
 * Compute architecture diversity: fraction of unique architectures in a population.
 */
export function computeArchitectureDiversity(activations: string[]): number {
  if (activations.length === 0) return 0;
  const unique = new Set(activations);
  return unique.size / activations.length;
}
