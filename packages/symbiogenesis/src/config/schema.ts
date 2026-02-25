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

  // -- Adaptive batch sizing --
  readonly adaptiveBatch: boolean;
  readonly batchMin: number;
  readonly batchMax: number;
  readonly batchStep: number;
  readonly calmStepsBeforeRestore: number;

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

  adaptiveBatch: true,
  batchMin: 8,
  batchMax: 64,
  batchStep: 4,
  calmStepsBeforeRestore: 200,

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

  writeReport: true,
  writeCandidates: true,
  writeSummary: true,
};
