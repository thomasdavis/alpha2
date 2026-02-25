/**
 * SymbioConfig type and defaults.
 * Inspired by symbiogenesis/config.py but written from scratch.
 */

export interface SymbioConfig {
  // -- CUSUM monitoring --
  readonly cusumSensitivity: number;
  readonly cusumBaselineWindow: number;

  // -- Metric collection --
  readonly metricsInterval: number;
  readonly trackWeightEntropy: boolean;
  readonly trackEffectiveRank: boolean;
  readonly trackFreeEnergy: boolean;
  readonly trackMIProfiles: boolean;
  readonly trackPopulationMetrics: boolean;
  readonly freeEnergyBeta: number;
  readonly miNumBins: number;

  // -- Adaptive batch sizing (deprecated; kept for backward compatibility telemetry) --
  readonly adaptiveBatch: boolean;
  readonly batchMin: number;
  readonly batchMax: number;
  readonly batchStep: number;
  readonly calmStepsBeforeRestore: number;

  // -- Population dynamics adaptation (replaces adaptive batch) --
  readonly populationAdaptation: boolean;
  /** Min/Max multiplier applied to base populationSize. */
  readonly populationScaleMin: number;
  readonly populationScaleMax: number;
  /** Step size for populationScale adjustments. */
  readonly populationScaleStep: number;
  /** Cooldown (steps) between population-scale changes. */
  readonly populationAdaptationCooldown: number;
  /** Clamp mutation rate during adaptation. */
  readonly mutationRateMin: number;
  readonly mutationRateMax: number;

  // -- Fitness / ranking --
  readonly fitnessAlpha: number;
  readonly complexityMode: "params" | "entropy" | "mdl";
  readonly diversityBonus: number;
  readonly diversityDecay: "none" | "linear" | "cosine";

  // -- FFN activation search --
  readonly searchMode: "ffn-activation-search" | "composed-activation-search" | "none";
  readonly activationPool: readonly string[];
  readonly searchStrategy: "evolutionary" | "exhaustive";
  readonly populationSize: number;
  readonly generations: number;
  readonly selectionStrategy: "topk" | "tournament";
  readonly tournamentK: number;
  readonly mutationRate: number;
  readonly stepsPerCandidate: number;
  readonly rankBy: "valLoss" | "fitness";
  readonly perfWeight: number;
  readonly stabilityWeight: number;

  // -- Candidate continuity / fusion --
  /** Preserve compatible weights when switching candidates instead of full re-init. */
  readonly preserveWeightsAcrossCandidates: boolean;
  /** Carry optimizer moment buffers across candidate switches where shapes are compatible. */
  readonly carryOptimizerStateAcrossCandidates: boolean;
  /** Use the same ffnDim across all candidates to maximize weight reuse. */
  readonly constantFfnDimAcrossCandidates: boolean;
  /** Apply step-wise parameter fusion toward a consensus shadow model. */
  readonly fuseWeightsEachStep: boolean;
  /** EMA rate for consensus shadow updates (0..1). */
  readonly fusionShadowEma: number;
  /** Minimum parameter fusion strength per step (0..1). */
  readonly fusionBaseStrength: number;
  /** Maximum parameter fusion strength per step (0..1). */
  readonly fusionMaxStrength: number;
  /** Kuramoto coupling strength used to modulate fusion strength. */
  readonly kuramotoCoupling: number;
  /** Kuramoto integration timestep. */
  readonly kuramotoDt: number;
  /** Damping applied to phase drift (0..1). */
  readonly kuramotoDamping: number;

  // -- Composed activation graph --
  readonly basisPool?: readonly string[];
  readonly maxGraphDepth?: number;
  readonly maxGraphNodes?: number;

  // -- Output --
  readonly writeReport: boolean;
  readonly writeCandidates: boolean;
  readonly writeSummary: boolean;
}

export const defaultSymbioConfig: SymbioConfig = {
  cusumSensitivity: 4.0,
  cusumBaselineWindow: 10,

  metricsInterval: 50,
  trackWeightEntropy: true,
  trackEffectiveRank: true,
  trackFreeEnergy: true,
  trackMIProfiles: false,
  trackPopulationMetrics: true,
  freeEnergyBeta: 0.01,
  miNumBins: 30,

  adaptiveBatch: false,
  batchMin: 8,
  batchMax: 64,
  batchStep: 4,
  calmStepsBeforeRestore: 200,

  populationAdaptation: true,
  populationScaleMin: 0.5,
  populationScaleMax: 2.0,
  populationScaleStep: 0.1,
  populationAdaptationCooldown: 20,
  mutationRateMin: 0.05,
  mutationRateMax: 0.9,

  fitnessAlpha: 1.0,
  complexityMode: "entropy",
  diversityBonus: 0.05,
  diversityDecay: "cosine",

  searchMode: "ffn-activation-search",
  activationPool: ["gelu", "relu", "silu", "swiglu", "universal", "kan_spline"],
  searchStrategy: "evolutionary",
  populationSize: 100000,
  generations: 50,
  selectionStrategy: "topk",
  tournamentK: 3,
  mutationRate: 0.25,
  stepsPerCandidate: 5000,
  rankBy: "valLoss",
  perfWeight: 0.0,
  stabilityWeight: 0.0,

  preserveWeightsAcrossCandidates: true,
  carryOptimizerStateAcrossCandidates: true,
  constantFfnDimAcrossCandidates: true,
  fuseWeightsEachStep: true,
  fusionShadowEma: 0.02,
  fusionBaseStrength: 0.001,
  fusionMaxStrength: 0.02,
  kuramotoCoupling: 0.4,
  kuramotoDt: 0.1,
  kuramotoDamping: 0.05,

  writeReport: true,
  writeCandidates: true,
  writeSummary: true,
};
