/**
 * @alpha/symbiogenesis — Symbiogenesis-inspired training mode.
 *
 * Provides monitoring, metrics collection, adaptive behavior, and search
 * orchestration for the Alpha training system. Written from scratch in
 * TypeScript, inspired by the Symbiogenesis Python repo.
 */

// Config
export { type SymbioConfig, defaultSymbioConfig } from "./config/schema.js";
export { loadSymbioConfig, validateSymbioConfig } from "./config/load.js";
export { applySymbioModelPreset, applySymbioTrainPreset, ffnDimForActivation } from "./config/preset.js";

// Types
export {
  type SymbioStepMetrics,
  type TrainerStepInfo,
  CUSUM_GRAD,
  CUSUM_CLIP,
  CUSUM_TPS,
  CUSUM_VAL,
} from "./types.js";

// Monitor
export { CusumMonitor } from "./monitor/cusum.js";
export { CusumDashboard, type CusumResult } from "./monitor/dashboard.js";
export { AdaptiveBatch } from "./monitor/adaptive-batch.js";

// Metrics
export { computeWeightEntropy } from "./metrics/weight-entropy.js";
export { computeEffectiveRank } from "./metrics/effective-rank.js";
export { computeFreeEnergy } from "./metrics/free-energy.js";
export { estimateMI, type MIProfile } from "./metrics/mi-estimator.js";
export {
  computePopulationEntropy,
  getActivationDistribution,
  computeArchitectureDiversity,
} from "./metrics/population.js";
export { computeComplexity, computeFitness, type FitnessInput } from "./metrics/fitness.js";
export { SymbioMetricsCollector } from "./metrics/collector.js";

// Search
export {
  type SearchCandidate,
  createCandidate,
  generateInitialPopulation,
  mutateCandidate,
} from "./search/candidates.js";
export { rankCandidates, selectParents } from "./search/ranking.js";
export {
  SearchOrchestrator,
  type SearchState,
} from "./search/orchestrator.js";
export {
  generateSummary,
  generateCandidatesJSONL,
  generateReport,
  type SearchSummary,
} from "./search/report.js";
