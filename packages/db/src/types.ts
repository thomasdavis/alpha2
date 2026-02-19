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

export type RunStatus = "active" | "completed" | "stale" | "failed";
