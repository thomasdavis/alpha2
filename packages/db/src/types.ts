/**
 * Database row types for @alpha/db.
 */

export interface DbRun {
  id: string;
  run_id: string;
  config_hash: string;
  domain: string;
  // denormalized model config
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_embd: number;
  n_head: number;
  dropout: number;
  // denormalized train config
  total_iters: number;
  batch_size: number;
  lr: number;
  seed: number;
  backend: string;
  tokenizer: string;
  optimizer: string;
  // full JSON blobs
  model_config: string;
  train_config: string;
  // status tracking
  status: "active" | "completed" | "stale" | "failed";
  latest_step: number;
  last_loss: number | null;
  best_val_loss: number | null;
  estimated_params: number | null;
  // timestamps
  created_at: string;
  updated_at: string;
  disk_mtime: number | null;
  // infrastructure metadata
  gpu_name: string | null;
  gpu_vendor: string | null;
  gpu_vram_mb: number | null;
  hostname: string | null;
  cpu_count: number | null;
  ram_total_mb: number | null;
  os_platform: string | null;
  // Symbio metadata
  symbio: number | null;
  symbio_config: string | null;
  ffn_activation: string | null;
  symbio_winner: string | null;
  symbio_mode: string | null;
}

export interface DbMetric {
  run_id: string;
  step: number;
  loss: number;
  val_loss: number | null;
  lr: number;
  grad_norm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  gpu_util_pct: number | null;
  gpu_vram_used_mb: number | null;
  gpu_vram_total_mb: number | null;
  gpu_mem_pool_mb: number | null;
  // Per-step timing breakdown (Phase 0 instrumentation)
  timing_fwd_ms: number | null;
  timing_bwd_ms: number | null;
  timing_optim_ms: number | null;
  timing_data_ms: number | null;
  timing_flush_ms: number | null;
  timing_grad_norm_ms: number | null;
  timing_grad_clip_ms: number | null;
  gpu_ops_count: number | null;
  // Clipping telemetry
  clip_coef: number | null;
  clip_pct: number | null;
  // CUSUM
  cusum_grad: number | null;
  cusum_clip: number | null;
  cusum_tps: number | null;
  cusum_val: number | null;
  cusum_alerts: number | null;
  cusum_alert_reason: string | null;
  // Symbio metrics
  weight_entropy: number | null;
  effective_rank: number | null;
  free_energy: number | null;
  population_entropy: number | null;
  activation_distribution: string | null;
  mi_input_repr: number | null;
  mi_repr_output: number | null;
  mi_compression: number | null;
  fitness_score: number | null;
  complexity_score: number | null;
  // Adaptive batch
  adaptive_batch_size: number | null;
  batch_change_reason: string | null;
  // Search candidate
  symbio_candidate_id: string | null;
  symbio_candidate_activation: string | null;
  symbio_generation: number | null;
  architecture_diversity: number | null;
}

export interface DbCheckpoint {
  run_id: string;
  step: number;
  filename: string;
  file_path: string;
  file_size: number | null;
  created_at: string;
}

export interface DbRunSummary extends DbRun {
  checkpoint_count: number;
  latest_checkpoint_step: number | null;
  metric_count: number;
}

export interface DbSample {
  run_id: string;
  idx: number;
  prompt: string;
  output: string;
  created_at: string;
}

export type RunStatus = "active" | "completed" | "stale" | "failed";
