import rawData from "../public/data.json";

export type RunConfig = typeof rawData.run;

export type Metric = {
  step: number;
  loss: number;
  val_loss: number | null;
  lr: number;
  grad_norm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  gpu_util_pct: number;
  gpu_vram_used_mb: number;
  gpu_vram_total_mb: number;
  gpu_mem_pool_mb: number;
  timing_fwd_ms: number;
  timing_bwd_ms: number;
  timing_optim_ms: number;
  timing_data_ms: number;
  timing_flush_ms: number;
  timing_grad_norm_ms: number;
  timing_grad_clip_ms: number;
  gpu_ops_count: number;
  clip_coef: number | null;
  clip_pct: number | null;
  cusum_grad: number;
  cusum_clip: number;
  cusum_tps: number;
  cusum_val: number;
  cusum_alerts: number;
  cusum_alert_reason: string | null;
  weight_entropy: number | null;
  effective_rank: number | null;
  free_energy: number | null;
  population_entropy: number | null;
  fitness_score: number | null;
  complexity_score: number | null;
  architecture_diversity: number | null;
  symbio_candidate_id: string | null;
  symbio_candidate_activation: string | null;
  symbio_generation: number | null;
  symbio_candidate_name: string | null;
  symbio_candidate_parent_name: string | null;
  symbio_activation_graph: string | null;
  symbio_mutation_applied: string | null;
};

export const run: RunConfig = rawData.run;
export const metrics: Metric[] = rawData.metrics as Metric[];

// Activation color palette
const ACTIVATION_COLORS: Record<string, string> = {
  silu: "#22d3ee",
  relu: "#f87171",
  gelu: "#a78bfa",
  id: "#94a3b8",
  sq: "#fbbf24",
  swiglu: "#34d399",
  universal: "#f472b6",
  kan_spline: "#fb923c",
};

export function getActivationColor(activation: string | null): string {
  if (!activation) return "#64748b";
  const lower = activation.toLowerCase();
  for (const [key, color] of Object.entries(ACTIVATION_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return "#64748b";
}

export function getActivationBaseColor(activation: string | null): string {
  if (!activation) return "#64748b";
  const lower = activation.toLowerCase();
  if (lower.startsWith("silu") || lower.includes("silu")) return ACTIVATION_COLORS.silu;
  if (lower.startsWith("relu") || lower.includes("relu")) return ACTIVATION_COLORS.relu;
  if (lower.startsWith("gelu") || lower.includes("gelu")) return ACTIVATION_COLORS.gelu;
  if (lower === "id") return ACTIVATION_COLORS.id;
  if (lower === "sq") return ACTIVATION_COLORS.sq;
  return "#64748b";
}

// Extract candidate switch events
export type SwitchEvent = {
  step: number;
  candidateName: string;
  activation: string;
  generation: number;
};

export function extractSwitchEvents(): SwitchEvent[] {
  const events: SwitchEvent[] = [];
  let prev = "";
  for (const m of metrics) {
    const name = m.symbio_candidate_name;
    if (name && name !== prev) {
      events.push({
        step: m.step,
        candidateName: name,
        activation: m.symbio_candidate_activation || "",
        generation: m.symbio_generation || 0,
      });
      prev = name;
    }
  }
  return events;
}

// Sample metrics at regular intervals for chart rendering
export function sampleMetrics(count: number): Metric[] {
  if (metrics.length <= count) return metrics;
  const step = (metrics.length - 1) / (count - 1);
  const result: Metric[] = [];
  for (let i = 0; i < count; i++) {
    result.push(metrics[Math.round(i * step)]);
  }
  return result;
}

// Get min/max for a field
export function getRange(field: keyof Metric): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const m of metrics) {
    const v = m[field] as number | null;
    if (v != null && isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return [min, max];
}

export function fmtNum(n: number, decimals = 1): string {
  if (n >= 1e9) return (n / 1e9).toFixed(decimals) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(decimals) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(decimals) + "K";
  return n.toFixed(decimals);
}
