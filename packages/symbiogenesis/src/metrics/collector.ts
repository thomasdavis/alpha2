/**
 * SymbioMetricsCollector: orchestrates all symbio metric computation.
 * Called by the trainer at metricsInterval steps.
 */
import type { TensorData } from "@alpha/core";
import type { SymbioConfig } from "../config/schema.js";
import type { SymbioStepMetrics } from "../types.js";
import { computeWeightEntropy } from "./weight-entropy.js";
import { computeEffectiveRank } from "./effective-rank.js";
import { computeFreeEnergy } from "./free-energy.js";
import { computePopulationEntropy, getActivationDistribution } from "./population.js";
import { computeComplexity, computeFitness } from "./fitness.js";
import { estimateMI } from "./mi-estimator.js";

export class SymbioMetricsCollector {
  private readonly config: SymbioConfig;
  private recentLosses: number[] = [];
  private readonly maxLossWindow = 50;

  constructor(config: SymbioConfig) {
    this.config = config;
  }

  /** Record a loss value (called every step for the sliding window). */
  recordLoss(loss: number): void {
    this.recentLosses.push(loss);
    if (this.recentLosses.length > this.maxLossWindow) {
      this.recentLosses.shift();
    }
  }

  /**
   * Collect expensive metrics. Called every metricsInterval steps.
   * @param params - Model parameters as TensorData map
   * @param loss - Current training loss
   * @param numParams - Total parameter count
   * @param activation - Current FFN activation name
   * @param nLayers - Number of transformer layers
   * @param valLoss - Validation loss (if available)
   * @param inputActivations - Input activations for MI estimation (optional)
   * @param hiddenActivations - Hidden activations for MI estimation (optional)
   * @param outputActivations - Output activations for MI estimation (optional)
   */
  collect(
    params: Map<string, TensorData>,
    loss: number,
    numParams: number,
    activation: string,
    nLayers: number,
    valLoss?: number,
    inputActivations?: Float32Array,
    hiddenActivations?: Float32Array,
    outputActivations?: Float32Array,
  ): Partial<SymbioStepMetrics> {
    const result: Partial<SymbioStepMetrics> = {};

    // Weight entropy
    let weightEntropy = 0;
    if (this.config.trackWeightEntropy) {
      weightEntropy = computeWeightEntropy(params);
      result.weight_entropy = weightEntropy;
    }

    // Effective rank
    let effectiveRank = 0;
    if (this.config.trackEffectiveRank) {
      effectiveRank = computeEffectiveRank(params);
      result.effective_rank = effectiveRank;
    }

    // Free energy
    if (this.config.trackFreeEnergy) {
      result.free_energy = computeFreeEnergy(loss, this.config.freeEnergyBeta, weightEntropy);
    }

    // Population entropy
    if (this.config.trackPopulationMetrics) {
      result.population_entropy = computePopulationEntropy(this.recentLosses);
      result.activation_distribution = JSON.stringify(getActivationDistribution(activation, nLayers));
    }

    // Complexity & fitness
    const fitnessInput = { loss, numParams, weightEntropy, effectiveRank };
    result.complexity_score = computeComplexity(fitnessInput, this.config.complexityMode);

    // Fitness score uses valLoss if available, otherwise training loss
    const fitnessLoss = valLoss ?? loss;
    result.fitness_score = computeFitness(
      { ...fitnessInput, loss: fitnessLoss },
      this.config.fitnessAlpha,
      this.config.complexityMode,
    );

    // MI profiles (expensive, off by default)
    if (this.config.trackMIProfiles && inputActivations && hiddenActivations && outputActivations) {
      const mi = estimateMI(inputActivations, hiddenActivations, outputActivations, this.config.miNumBins);
      result.mi_input_repr = mi.mi_input_repr;
      result.mi_repr_output = mi.mi_repr_output;
      result.mi_compression = mi.mi_compression;
    }

    return result;
  }
}
