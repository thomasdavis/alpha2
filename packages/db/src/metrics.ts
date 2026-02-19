/**
 * CRUD operations for the metrics table.
 */
import type { Client, InValue } from "@libsql/client";
import type { DbMetric } from "./types.js";

const CHUNK_SIZE = 500;

export async function insertMetrics(
  client: Client,
  runId: string,
  metrics: Array<{
    step: number;
    loss: number;
    valLoss?: number | null;
    lr: number;
    gradNorm: number;
    elapsed_ms: number;
    tokens_per_sec: number;
    ms_per_iter: number;
  }>
): Promise<number> {
  if (metrics.length === 0) return 0;

  let inserted = 0;
  for (let i = 0; i < metrics.length; i += CHUNK_SIZE) {
    const chunk = metrics.slice(i, i + CHUNK_SIZE);
    await client.batch(
      chunk.map((m) => ({
        sql: `INSERT OR IGNORE INTO metrics
              (run_id, step, loss, val_loss, lr, grad_norm, elapsed_ms, tokens_per_sec, ms_per_iter)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          runId,
          m.step,
          m.loss,
          m.valLoss ?? null,
          m.lr,
          m.gradNorm,
          m.elapsed_ms,
          m.tokens_per_sec,
          m.ms_per_iter,
        ],
      })),
      "write"
    );
    inserted += chunk.length;
  }

  return inserted;
}

export async function getMetrics(
  client: Client,
  runId: string,
  opts?: { fromStep?: number; limit?: number }
): Promise<DbMetric[]> {
  const conditions = ["run_id = ?"];
  const args: InValue[] = [runId];

  if (opts?.fromStep != null) {
    conditions.push("step >= ?");
    args.push(opts.fromStep);
  }

  const limit = opts?.limit ?? 10000;

  const result = await client.execute({
    sql: `SELECT * FROM metrics WHERE ${conditions.join(" AND ")} ORDER BY step ASC LIMIT ?`,
    args: [...args, limit],
  });
  return result.rows as unknown as DbMetric[];
}

export async function getRecentMetrics(
  client: Client,
  runId: string,
  count: number
): Promise<DbMetric[]> {
  const result = await client.execute({
    sql: `SELECT * FROM (
      SELECT * FROM metrics WHERE run_id = ? ORDER BY step DESC LIMIT ?
    ) sub ORDER BY step ASC`,
    args: [runId, count],
  });
  return result.rows as unknown as DbMetric[];
}

export async function getMaxStep(client: Client, runId: string): Promise<number> {
  const result = await client.execute({
    sql: "SELECT COALESCE(MAX(step), -1) AS max_step FROM metrics WHERE run_id = ?",
    args: [runId],
  });
  return result.rows[0].max_step as number;
}
