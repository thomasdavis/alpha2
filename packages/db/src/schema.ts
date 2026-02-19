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
];
