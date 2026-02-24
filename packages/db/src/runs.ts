/**
 * CRUD operations for the runs table.
 */
import type { Client, InValue } from "@libsql/client";
import type { DbRun, DbRunSummary, RunStatus } from "./types.js";

export interface UpsertRunInput {
  id: string;
  run_id: string;
  config_hash: string;
  domain: string;
  model_config: Record<string, unknown>;
  train_config: Record<string, unknown>;
  status?: RunStatus;
  latest_step?: number;
  last_loss?: number | null;
  best_val_loss?: number | null;
  estimated_params?: number | null;
  disk_mtime?: number | null;
  gpu_name?: string | null;
  gpu_vendor?: string | null;
  gpu_vram_mb?: number | null;
  hostname?: string | null;
  cpu_count?: number | null;
  ram_total_mb?: number | null;
  os_platform?: string | null;
}

export async function upsertRun(client: Client, input: UpsertRunInput): Promise<void> {
  const mc = input.model_config as any;
  const tc = input.train_config as any;

  await client.execute({
    sql: `INSERT INTO runs (
      id, run_id, config_hash, domain,
      vocab_size, block_size, n_layer, n_embd, n_head, dropout,
      total_iters, batch_size, lr, seed, backend, tokenizer, optimizer,
      model_config, train_config,
      status, latest_step, last_loss, best_val_loss, estimated_params,
      disk_mtime,
      gpu_name, gpu_vendor, gpu_vram_mb, hostname, cpu_count, ram_total_mb, os_platform,
      updated_at
    ) VALUES (
      ?, ?, ?, ?,
      ?, ?, ?, ?, ?, ?,
      ?, ?, ?, ?, ?, ?, ?,
      ?, ?,
      ?, ?, ?, ?, ?,
      ?,
      ?, ?, ?, ?, ?, ?, ?,
      datetime('now')
    )
    ON CONFLICT(id) DO UPDATE SET
      status = excluded.status,
      latest_step = excluded.latest_step,
      last_loss = excluded.last_loss,
      best_val_loss = excluded.best_val_loss,
      estimated_params = excluded.estimated_params,
      disk_mtime = excluded.disk_mtime,
      gpu_name = COALESCE(excluded.gpu_name, gpu_name),
      gpu_vendor = COALESCE(excluded.gpu_vendor, gpu_vendor),
      gpu_vram_mb = COALESCE(excluded.gpu_vram_mb, gpu_vram_mb),
      hostname = COALESCE(excluded.hostname, hostname),
      cpu_count = COALESCE(excluded.cpu_count, cpu_count),
      ram_total_mb = COALESCE(excluded.ram_total_mb, ram_total_mb),
      os_platform = COALESCE(excluded.os_platform, os_platform),
      updated_at = datetime('now')`,
    args: [
      input.id,
      input.run_id,
      input.config_hash,
      input.domain,
      mc.vocabSize ?? null,
      mc.blockSize ?? null,
      mc.nLayer ?? null,
      mc.nEmbd ?? null,
      mc.nHead ?? null,
      mc.dropout ?? null,
      tc.iters ?? null,
      tc.batchSize ?? null,
      tc.lr ?? null,
      tc.seed ?? null,
      tc.backend ?? null,
      tc.tokenizer ?? null,
      tc.optimizer ?? null,
      JSON.stringify(input.model_config),
      JSON.stringify(input.train_config),
      input.status ?? "active",
      input.latest_step ?? 0,
      input.last_loss ?? null,
      input.best_val_loss ?? null,
      input.estimated_params ?? null,
      input.disk_mtime ?? null,
      input.gpu_name ?? null,
      input.gpu_vendor ?? null,
      input.gpu_vram_mb ?? null,
      input.hostname ?? null,
      input.cpu_count ?? null,
      input.ram_total_mb ?? null,
      input.os_platform ?? null,
    ],
  });
}

export async function getRun(client: Client, id: string): Promise<DbRun | null> {
  const result = await client.execute({
    sql: "SELECT * FROM runs WHERE id = ?",
    args: [id],
  });
  return (result.rows[0] as unknown as DbRun) ?? null;
}

export async function listRuns(
  client: Client,
  opts?: { status?: RunStatus; domain?: string; limit?: number }
): Promise<DbRunSummary[]> {
  const conditions: string[] = [];
  const args: InValue[] = [];

  if (opts?.status) {
    conditions.push("r.status = ?");
    args.push(opts.status);
  }
  if (opts?.domain) {
    conditions.push("r.domain = ?");
    args.push(opts.domain);
  }

  const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = opts?.limit ?? 100;

  const result = await client.execute({
    sql: `SELECT * FROM run_summary r ${where} ORDER BY r.created_at DESC LIMIT ?`,
    args: [...args, limit],
  });
  return result.rows as unknown as DbRunSummary[];
}

export async function updateRunProgress(
  client: Client,
  id: string,
  update: {
    latest_step: number;
    last_loss?: number | null;
    best_val_loss?: number | null;
    status?: RunStatus;
  }
): Promise<void> {
  await client.execute({
    sql: `UPDATE runs SET
      latest_step = ?,
      last_loss = COALESCE(?, last_loss),
      best_val_loss = CASE
        WHEN ? IS NOT NULL AND (best_val_loss IS NULL OR ? < best_val_loss) THEN ?
        ELSE best_val_loss
      END,
      status = COALESCE(?, status),
      updated_at = datetime('now')
    WHERE id = ?`,
    args: [
      update.latest_step,
      update.last_loss ?? null,
      update.best_val_loss ?? null,
      update.best_val_loss ?? null,
      update.best_val_loss ?? null,
      update.status ?? null,
      id,
    ],
  });
}

export async function deleteRun(client: Client, id: string): Promise<void> {
  await client.execute({
    sql: "DELETE FROM runs WHERE id = ?",
    args: [id],
  });
}
