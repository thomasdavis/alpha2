/**
 * Shared types for @alpha/symbiogenesis.
 */

/** CUSUM alert bitmask values. */
export const CUSUM_GRAD = 0x01;
export const CUSUM_CLIP = 0x02;
export const CUSUM_TPS  = 0x04;
export const CUSUM_VAL  = 0x08;

/** A snapshot of all symbio metrics for one training step. */
export interface SymbioStepMetrics {
  // CUSUM
  cusum_grad?: number;
  cusum_clip?: number;
  cusum_tps?: number;
  cusum_val?: number;
  cusum_alerts?: number;
  cusum_alert_reason?: string;

  // Adaptive batch
  adaptive_batch_size?: number;
  batch_change_reason?: string;

  // Expensive metrics (sparse — every metricsInterval steps)
  weight_entropy?: number;
  effective_rank?: number;
  free_energy?: number;
  population_entropy?: number;
  activation_distribution?: string;
  mi_input_repr?: number;
  mi_repr_output?: number;
  mi_compression?: number;
  fitness_score?: number;
  complexity_score?: number;

  // Search candidate tracking
  symbio_candidate_id?: string;
  symbio_candidate_activation?: string;
  symbio_generation?: number;
  architecture_diversity?: number;
}

/** Minimal step info passed into the symbio layer from the trainer. */
export interface TrainerStepInfo {
  step: number;
  loss: number;
  gradNorm: number;
  clipPct: number;
  tokensPerSec: number;
  valLoss?: number;
}
