/**
 * Multi-objective fitness functions.
 * Inspired by symbiogenesis/training.py:train_and_eval — written from scratch.
 *
 * Three complexity modes:
 * - "params": 1.0 / (1.0 + num_parameters / 1000.0)
 * - "entropy": weight_entropy / max(1.0, log2(num_parameters))
 * - "mdl": effective_rank * log2(max(2, num_parameters)) / num_parameters
 */

export interface FitnessInput {
  loss: number;
  numParams: number;
  weightEntropy: number;
  effectiveRank: number;
}

/**
 * Compute the complexity score based on the configured mode.
 */
export function computeComplexity(
  input: FitnessInput,
  mode: "params" | "entropy" | "mdl",
): number {
  switch (mode) {
    case "params":
      return 1.0 / (1.0 + input.numParams / 1000.0);
    case "entropy":
      return input.weightEntropy / Math.max(1.0, Math.log2(input.numParams));
    case "mdl":
      return (input.effectiveRank * Math.log2(Math.max(2, input.numParams))) / input.numParams;
  }
}

/**
 * Compute the overall fitness score.
 * fitness = alpha * accuracy - complexity_penalty
 * where accuracy = 1 / (1 + loss) to normalize into [0, 1]
 */
export function computeFitness(
  input: FitnessInput,
  alpha: number,
  complexityMode: "params" | "entropy" | "mdl",
): number {
  const accuracy = 1.0 / (1.0 + input.loss);
  const complexity = computeComplexity(input, complexityMode);
  return alpha * accuracy - complexity;
}
