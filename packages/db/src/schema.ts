/**
 * Database schema migrations.
 *
 * Each entry is a migration version. The migrate runner applies them
 * sequentially and tracks the current version in schema_version.
 */

export const migrations: string[][] = [
  // Version 1: initial schema
  [
    `CREATE TABLE IF NOT EXISTS runs (
      id              TEXT PRIMARY KEY,
      run_id          TEXT NOT NULL,
      config_hash     TEXT NOT NULL,
      domain          TEXT NOT NULL DEFAULT 'novels',
      vocab_size      INTEGER,
      block_size      INTEGER,
      n_layer         INTEGER,
      n_embd          INTEGER,
      n_head          INTEGER,
      dropout         REAL,
      total_iters     INTEGER,
      batch_size      INTEGER,
      lr              REAL,
      seed            INTEGER,
      backend         TEXT,
      tokenizer       TEXT,
      optimizer       TEXT,
      model_config    TEXT NOT NULL,
      train_config    TEXT NOT NULL,
      status          TEXT NOT NULL DEFAULT 'active'
                      CHECK(status IN ('active','completed','stale','failed')),
      latest_step     INTEGER NOT NULL DEFAULT 0,
      last_loss       REAL,
      best_val_loss   REAL,
      estimated_params INTEGER,
      created_at      TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
      disk_mtime      REAL
    )`,

    `CREATE TABLE IF NOT EXISTS metrics (
      run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      step        INTEGER NOT NULL,
      loss        REAL NOT NULL,
      val_loss    REAL,
      lr          REAL NOT NULL,
      grad_norm   REAL NOT NULL,
      elapsed_ms  REAL NOT NULL,
      tokens_per_sec REAL NOT NULL,
      ms_per_iter REAL NOT NULL,
      PRIMARY KEY (run_id, step)
    ) WITHOUT ROWID`,

    `CREATE TABLE IF NOT EXISTS checkpoints (
      run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      step      INTEGER NOT NULL,
      filename  TEXT NOT NULL,
      file_path TEXT NOT NULL,
      file_size INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (run_id, step)
    ) WITHOUT ROWID`,

    `CREATE TABLE IF NOT EXISTS domains (
      id             TEXT PRIMARY KEY,
      display_name   TEXT NOT NULL,
      tokenizer      TEXT NOT NULL,
      sample_prompts TEXT NOT NULL,
      model_defaults TEXT NOT NULL,
      train_defaults TEXT NOT NULL,
      created_at     TEXT NOT NULL DEFAULT (datetime('now'))
    )`,

    // Indices
    `CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)`,
    `CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs(domain)`,
    `CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC)`,
    `CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step DESC)`,

    // View
    `CREATE VIEW IF NOT EXISTS run_summary AS
     SELECT r.*,
       (SELECT COUNT(*) FROM checkpoints c WHERE c.run_id = r.id) AS checkpoint_count,
       (SELECT MAX(step) FROM checkpoints c WHERE c.run_id = r.id) AS latest_checkpoint_step,
       (SELECT COUNT(*) FROM metrics m WHERE m.run_id = r.id) AS metric_count
     FROM runs r`,
  ],

  // Version 2: samples table
  [
    `CREATE TABLE IF NOT EXISTS samples (
      run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      idx       INTEGER NOT NULL,
      prompt    TEXT NOT NULL,
      output    TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (run_id, idx)
    ) WITHOUT ROWID`,
  ],

  // Version 3: GPU metrics columns
  [
    `ALTER TABLE metrics ADD COLUMN gpu_util_pct REAL`,
    `ALTER TABLE metrics ADD COLUMN gpu_vram_used_mb REAL`,
    `ALTER TABLE metrics ADD COLUMN gpu_vram_total_mb REAL`,
    `ALTER TABLE metrics ADD COLUMN gpu_mem_pool_mb REAL`,
  ],

  // Version 4: Per-step timing breakdown (Phase 0 instrumentation)
  [
    `ALTER TABLE metrics ADD COLUMN timing_fwd_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_bwd_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_optim_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_data_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_flush_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_grad_norm_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN timing_grad_clip_ms REAL`,
    `ALTER TABLE metrics ADD COLUMN gpu_ops_count INTEGER`,
  ],

  // Version 5: Infrastructure metadata on runs
  [
    `ALTER TABLE runs ADD COLUMN gpu_name TEXT`,
    `ALTER TABLE runs ADD COLUMN gpu_vendor TEXT`,
    `ALTER TABLE runs ADD COLUMN gpu_vram_mb INTEGER`,
    `ALTER TABLE runs ADD COLUMN hostname TEXT`,
    `ALTER TABLE runs ADD COLUMN cpu_count INTEGER`,
    `ALTER TABLE runs ADD COLUMN ram_total_mb INTEGER`,
    `ALTER TABLE runs ADD COLUMN os_platform TEXT`,
  ],

  // Version 6: Symbio metrics, CUSUM, clipping telemetry, adaptive batch, search candidates
  [
    // Clipping telemetry (ALL runs)
    `ALTER TABLE metrics ADD COLUMN clip_coef REAL`,
    `ALTER TABLE metrics ADD COLUMN clip_pct REAL`,

    // CUSUM
    `ALTER TABLE metrics ADD COLUMN cusum_grad REAL`,
    `ALTER TABLE metrics ADD COLUMN cusum_clip REAL`,
    `ALTER TABLE metrics ADD COLUMN cusum_tps REAL`,
    `ALTER TABLE metrics ADD COLUMN cusum_val REAL`,
    `ALTER TABLE metrics ADD COLUMN cusum_alerts INTEGER`,
    `ALTER TABLE metrics ADD COLUMN cusum_alert_reason TEXT`,

    // Symbio metrics (sparse)
    `ALTER TABLE metrics ADD COLUMN weight_entropy REAL`,
    `ALTER TABLE metrics ADD COLUMN effective_rank REAL`,
    `ALTER TABLE metrics ADD COLUMN free_energy REAL`,
    `ALTER TABLE metrics ADD COLUMN population_entropy REAL`,
    `ALTER TABLE metrics ADD COLUMN activation_distribution TEXT`,
    `ALTER TABLE metrics ADD COLUMN mi_input_repr REAL`,
    `ALTER TABLE metrics ADD COLUMN mi_repr_output REAL`,
    `ALTER TABLE metrics ADD COLUMN mi_compression REAL`,
    `ALTER TABLE metrics ADD COLUMN fitness_score REAL`,
    `ALTER TABLE metrics ADD COLUMN complexity_score REAL`,

    // Adaptive batch
    `ALTER TABLE metrics ADD COLUMN adaptive_batch_size INTEGER`,
    `ALTER TABLE metrics ADD COLUMN batch_change_reason TEXT`,

    // Search candidate tracking
    `ALTER TABLE metrics ADD COLUMN symbio_candidate_id TEXT`,
    `ALTER TABLE metrics ADD COLUMN symbio_candidate_activation TEXT`,
    `ALTER TABLE metrics ADD COLUMN symbio_generation INTEGER`,
    `ALTER TABLE metrics ADD COLUMN architecture_diversity REAL`,

    // Run-level symbio metadata
    `ALTER TABLE runs ADD COLUMN symbio INTEGER DEFAULT 0`,
    `ALTER TABLE runs ADD COLUMN symbio_config TEXT`,
    `ALTER TABLE runs ADD COLUMN ffn_activation TEXT`,
    `ALTER TABLE runs ADD COLUMN symbio_winner TEXT`,
    `ALTER TABLE runs ADD COLUMN symbio_mode TEXT`,
  ],
];
