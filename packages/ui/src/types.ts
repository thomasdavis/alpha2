export interface ChartMetric {
  step: number;
  loss: number;
  val_loss: number | null;
  symbio_candidate_id?: string | null;
  lr: number;
  grad_norm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  gpu_util_pct: number | null;
  gpu_vram_used_mb: number | null;
  gpu_vram_total_mb: number | null;
  timing_fwd_ms: number | null;
  timing_bwd_ms: number | null;
  timing_optim_ms: number | null;
  timing_data_ms: number | null;
  timing_flush_ms: number | null;
  timing_grad_norm_ms: number | null;
  timing_grad_clip_ms: number | null;
  gpu_ops_count: number | null;
  per_layer_grad_norms?: string | null;
}

export interface ChartCheckpoint {
  step: number;
}

export interface MiniSeries {
  data: { step: number; value: number }[];
  color: string;
  label: string;
  axis?: "left" | "right";
  format?: (v: number) => string;
}

export type MarkerType = "checkpoints" | "bestVal" | "warmupEnd" | "gradSpikes" | "lossSpikes" | "overfit" | "activationSwitch" | "evoValEnvelope" | "evoOverfit";
export type MarkerVisibility = Record<MarkerType, boolean>;

export interface ActivationSwitchEvent {
  step: number;
  fromActivation: string | null;
  toActivation: string;
  toGeneration: number;
  toCandidateId: string;
  lossAtSwitch: number;
}

export interface ComputedEvents {
  checkpointSteps: number[];
  bestValStep: number | null;
  bestValLoss: number | null;
  warmupEndStep: number | null;
  gradSpikeSteps: number[];
  lossSpikeSteps: number[];
  overfitStep: number | null;
  activationSwitches: ActivationSwitchEvent[];
  evoValEnvelope: { step: number; loss: number }[];
  evoOverfitRegions: { startStep: number; endStep: number }[];
}
