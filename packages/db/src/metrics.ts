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
    gpu_util_pct?: number | null;
    gpu_vram_used_mb?: number | null;
    gpu_vram_total_mb?: number | null;
    gpu_mem_pool_mb?: number | null;
    // Per-step timing breakdown
    timing_fwd_ms?: number | null;
    timing_bwd_ms?: number | null;
    timing_optim_ms?: number | null;
    timing_data_ms?: number | null;
    timing_flush_ms?: number | null;
    timing_grad_norm_ms?: number | null;
    timing_grad_clip_ms?: number | null;
    gpu_ops_count?: number | null;
    clip_coef?: number | null;
    clip_pct?: number | null;
    cusum_grad?: number | null;
    cusum_clip?: number | null;
    cusum_tps?: number | null;
    cusum_val?: number | null;
    cusum_alerts?: number | null;
    cusum_alert_reason?: string | null;
    weight_entropy?: number | null;
    effective_rank?: number | null;
    free_energy?: number | null;
    population_entropy?: number | null;
    activation_distribution?: string | null;
    mi_input_repr?: number | null;
    mi_repr_output?: number | null;
    mi_compression?: number | null;
    fitness_score?: number | null;
    complexity_score?: number | null;
    adaptive_batch_size?: number | null;
    batch_change_reason?: string | null;
    symbio_candidate_id?: string | null;
    symbio_candidate_name?: string | null;
    symbio_candidate_activation?: string | null;
    symbio_candidate_parent_id?: string | null;
    symbio_candidate_parent_name?: string | null;
    symbio_generation?: number | null;
    architecture_diversity?: number | null;
  }>
): Promise<number> {
  if (metrics.length === 0) return 0;

  let inserted = 0;
  for (let i = 0; i < metrics.length; i += CHUNK_SIZE) {
    const chunk = metrics.slice(i, i + CHUNK_SIZE);
    await client.batch(
      chunk.map((m) => ({
        sql: `INSERT OR IGNORE INTO metrics
              (run_id, step, loss, val_loss, lr, grad_norm, elapsed_ms, tokens_per_sec, ms_per_iter,
               gpu_util_pct, gpu_vram_used_mb, gpu_vram_total_mb, gpu_mem_pool_mb,
               timing_fwd_ms, timing_bwd_ms, timing_optim_ms, timing_data_ms,
               timing_flush_ms, timing_grad_norm_ms, timing_grad_clip_ms, gpu_ops_count,
               clip_coef, clip_pct,
               cusum_grad, cusum_clip, cusum_tps, cusum_val, cusum_alerts, cusum_alert_reason,
               weight_entropy, effective_rank, free_energy, population_entropy, activation_distribution,
               mi_input_repr, mi_repr_output, mi_compression, fitness_score, complexity_score,
               adaptive_batch_size, batch_change_reason,
               symbio_candidate_id, symbio_candidate_name, symbio_candidate_activation,
               symbio_candidate_parent_id, symbio_candidate_parent_name,
               symbio_generation, architecture_diversity)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
          m.gpu_util_pct ?? null,
          m.gpu_vram_used_mb ?? null,
          m.gpu_vram_total_mb ?? null,
          m.gpu_mem_pool_mb ?? null,
          m.timing_fwd_ms ?? null,
          m.timing_bwd_ms ?? null,
          m.timing_optim_ms ?? null,
          m.timing_data_ms ?? null,
          m.timing_flush_ms ?? null,
          m.timing_grad_norm_ms ?? null,
          m.timing_grad_clip_ms ?? null,
          m.gpu_ops_count ?? null,
          m.clip_coef ?? null,
          m.clip_pct ?? null,
          m.cusum_grad ?? null,
          m.cusum_clip ?? null,
          m.cusum_tps ?? null,
          m.cusum_val ?? null,
          m.cusum_alerts ?? null,
          m.cusum_alert_reason ?? null,
          m.weight_entropy ?? null,
          m.effective_rank ?? null,
          m.free_energy ?? null,
          m.population_entropy ?? null,
          m.activation_distribution ?? null,
          m.mi_input_repr ?? null,
          m.mi_repr_output ?? null,
          m.mi_compression ?? null,
          m.fitness_score ?? null,
          m.complexity_score ?? null,
          m.adaptive_batch_size ?? null,
          m.batch_change_reason ?? null,
          m.symbio_candidate_id ?? null,
          m.symbio_candidate_name ?? null,
          m.symbio_candidate_activation ?? null,
          m.symbio_candidate_parent_id ?? null,
          m.symbio_candidate_parent_name ?? null,
          m.symbio_generation ?? null,
          m.architecture_diversity ?? null,
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
