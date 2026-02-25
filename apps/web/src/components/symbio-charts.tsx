"use client";

import { useEffect, useRef } from "react";
import { type ChartMetric, Stat } from "@/components/charts";

// ── Types ────────────────────────────────────────────────────

interface SymbioMetric extends ChartMetric {
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
  fitness_score?: number | null;
  complexity_score?: number | null;
  adaptive_batch_size?: number | null;
  batch_change_reason?: string | null;
  clip_coef?: number | null;
  clip_pct?: number | null;
  symbio_candidate_id?: string | null;
  symbio_candidate_activation?: string | null;
  symbio_generation?: number | null;
  architecture_diversity?: number | null;
}

// ── Symbio Stats Grid ────────────────────────────────────────

export function SymbioStatsGrid({ metrics }: { metrics: SymbioMetric[] }) {
  const symbioMetrics = metrics.filter(m => m.weight_entropy != null || m.cusum_grad != null);
  if (symbioMetrics.length === 0) return null;

  const last = symbioMetrics[symbioMetrics.length - 1];
  const lastWithEntropy = [...metrics].reverse().find(m => m.weight_entropy != null);
  const lastWithRank = [...metrics].reverse().find(m => m.effective_rank != null);
  const lastWithFE = [...metrics].reverse().find(m => m.free_energy != null);
  const lastWithPE = [...metrics].reverse().find(m => m.population_entropy != null);
  const lastWithFit = [...metrics].reverse().find(m => m.fitness_score != null);
  const lastWithComp = [...metrics].reverse().find(m => m.complexity_score != null);

  const totalAlerts = metrics.reduce((n, m) => n + ((m.cusum_alerts ?? 0) > 0 ? 1 : 0), 0);
  const lastAlertStep = [...metrics].reverse().find(m => (m.cusum_alerts ?? 0) > 0)?.step;

  return (
    <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
      <Stat label="Weight Entropy" value={lastWithEntropy?.weight_entropy != null ? lastWithEntropy.weight_entropy.toFixed(2) : "-"} sub="bits" color="text-purple-400" />
      <Stat label="Effective Rank" value={lastWithRank?.effective_rank != null ? lastWithRank.effective_rank.toFixed(1) : "-"} />
      <Stat label="Free Energy" value={lastWithFE?.free_energy != null ? lastWithFE.free_energy.toFixed(4) : "-"} />
      <Stat label="Pop Entropy" value={lastWithPE?.population_entropy != null ? lastWithPE.population_entropy.toFixed(3) : "-"} sub="nats" />
      <Stat label="Complexity" value={lastWithComp?.complexity_score != null ? lastWithComp.complexity_score.toFixed(4) : "-"} />
      <Stat label="Fitness" value={lastWithFit?.fitness_score != null ? lastWithFit.fitness_score.toFixed(4) : "-"} color="text-green" />
      <Stat label="CUSUM Alerts" value={String(totalAlerts)} sub={lastAlertStep ? `last: step ${lastAlertStep}` : undefined} color={totalAlerts > 0 ? "text-orange-400" : undefined} />
      <Stat label="Batch Size" value={last?.adaptive_batch_size != null ? String(last.adaptive_batch_size) : "-"} />
    </div>
  );
}

// ── Canvas helpers ────────────────────────────────────────────

function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  return ctx;
}

function drawGrid(ctx: CanvasRenderingContext2D, pad: { l: number; r: number; t: number; b: number }, w: number, h: number, rows: number) {
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= rows; i++) {
    const y = pad.t + (i / rows) * (h - pad.t - pad.b);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
  }
}

// ── CUSUM Chart ──────────────────────────────────────────────

export function CusumChart({ metrics, sensitivity = 4.0 }: { metrics: SymbioMetric[]; sensitivity?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.parentElement?.clientWidth ?? 600;
    const h = 200;
    const ctx = setupCanvas(canvas, w, h);
    const pad = { l: 50, r: 20, t: 20, b: 30 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, w, h);
    drawGrid(ctx, pad, w, h, 4);

    const cusumData = metrics.filter(m => m.cusum_grad != null);
    if (cusumData.length === 0) {
      ctx.fillStyle = "#666";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No CUSUM data (baseline accumulating...)", w / 2, h / 2);
      return;
    }

    // Find range
    let maxVal = sensitivity * 1.2;
    for (const m of cusumData) {
      maxVal = Math.max(maxVal, m.cusum_grad ?? 0, m.cusum_clip ?? 0, m.cusum_tps ?? 0, m.cusum_val ?? 0);
    }

    const minStep = cusumData[0].step;
    const maxStep = cusumData[cusumData.length - 1].step;
    const stepRange = maxStep - minStep || 1;

    const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
    const toY = (val: number) => pad.t + plotH - (val / maxVal) * plotH;

    // Draw sensitivity threshold
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, toY(sensitivity));
    ctx.lineTo(w - pad.r, toY(sensitivity));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`threshold=${sensitivity}`, pad.l + 4, toY(sensitivity) - 3);

    // Draw lines
    const series = [
      { key: "cusum_grad" as const, color: "#f59e0b", label: "gradNorm" },
      { key: "cusum_clip" as const, color: "#60a5fa", label: "clipPct" },
      { key: "cusum_tps" as const, color: "#34d399", label: "tokPerSec" },
      { key: "cusum_val" as const, color: "#f87171", label: "valLoss" },
    ];

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (const m of cusumData) {
        const val = m[s.key];
        if (val == null) continue;
        const x = toX(m.step);
        const y = toY(val);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Alert markers
    for (const m of cusumData) {
      if ((m.cusum_alerts ?? 0) > 0) {
        ctx.fillStyle = "rgba(248,113,113,0.3)";
        const x = toX(m.step);
        ctx.fillRect(x - 1, pad.t, 2, plotH);
      }
    }

    // Legend
    const legendX = pad.l + 10;
    let legendY = pad.t + 12;
    ctx.font = "10px monospace";
    for (const s of series) {
      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, legendY - 6, 10, 3);
      ctx.fillStyle = "#aaa";
      ctx.textAlign = "left";
      ctx.fillText(s.label, legendX + 14, legendY);
      legendY += 13;
    }

    // Axes labels
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`step ${minStep}`, pad.l, h - 8);
    ctx.fillText(`step ${maxStep}`, w - pad.r, h - 8);
    ctx.textAlign = "right";
    ctx.fillText("0", pad.l - 4, h - pad.b);
    ctx.fillText(maxVal.toFixed(1), pad.l - 4, pad.t + 10);
  }, [metrics, sensitivity]);

  return <canvas ref={canvasRef} className="w-full" />;
}

// ── Clip Telemetry Chart ─────────────────────────────────────

export function ClipChart({ metrics }: { metrics: SymbioMetric[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const clipData = metrics.filter(m => m.clip_coef != null);
    if (clipData.length === 0) return;

    const w = canvas.parentElement?.clientWidth ?? 600;
    const h = 160;
    const ctx = setupCanvas(canvas, w, h);
    const pad = { l: 50, r: 50, t: 15, b: 25 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, w, h);
    drawGrid(ctx, pad, w, h, 4);

    const minStep = clipData[0].step;
    const maxStep = clipData[clipData.length - 1].step;
    const stepRange = maxStep - minStep || 1;

    const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;

    // clip_coef (0-1, left axis)
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (const m of clipData) {
      const x = toX(m.step);
      const y = pad.t + plotH - ((m.clip_coef ?? 1) * plotH);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // clip_pct (0-100%, right axis)
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    started = false;
    for (const m of clipData) {
      const x = toX(m.step);
      const y = pad.t + plotH - ((m.clip_pct ?? 0) / 100 * plotH);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    ctx.font = "10px monospace";
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(pad.l + 10, pad.t + 6, 10, 3);
    ctx.fillStyle = "#aaa";
    ctx.textAlign = "left";
    ctx.fillText("clip_coef", pad.l + 24, pad.t + 12);
    ctx.fillStyle = "#60a5fa";
    ctx.fillRect(pad.l + 100, pad.t + 6, 10, 3);
    ctx.fillStyle = "#aaa";
    ctx.fillText("clip_pct", pad.l + 114, pad.t + 12);

    // Axis labels
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText("0", pad.l - 4, h - pad.b);
    ctx.fillText("1.0", pad.l - 4, pad.t + 10);
    ctx.textAlign = "left";
    ctx.fillText("0%", w - pad.r + 4, h - pad.b);
    ctx.fillText("100%", w - pad.r + 4, pad.t + 10);
  }, [metrics]);

  return <canvas ref={canvasRef} className="w-full" />;
}

// ── Symbio Metrics Mini Charts ───────────────────────────────

function SparseLineChart({
  metrics,
  getY,
  color = "#a78bfa",
  label,
  format = (v: number) => v.toFixed(2),
}: {
  metrics: SymbioMetric[];
  getY: (m: SymbioMetric) => number | null | undefined;
  color?: string;
  label: string;
  format?: (v: number) => string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const points = metrics
      .map(m => ({ step: m.step, val: getY(m) }))
      .filter((p): p is { step: number; val: number } => p.val != null);

    const w = canvas.parentElement?.clientWidth ?? 300;
    const h = 120;
    const ctx = setupCanvas(canvas, w, h);
    const pad = { l: 50, r: 10, t: 18, b: 20 };

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, w, h);

    if (points.length < 2) {
      ctx.fillStyle = "#666";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`No ${label} data`, w / 2, h / 2);
      return;
    }

    drawGrid(ctx, pad, w, h, 3);

    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;
    const minStep = points[0].step;
    const maxStep = points[points.length - 1].step;
    const stepRange = maxStep - minStep || 1;
    let minVal = Infinity, maxVal = -Infinity;
    for (const p of points) {
      if (p.val < minVal) minVal = p.val;
      if (p.val > maxVal) maxVal = p.val;
    }
    const valRange = maxVal - minVal || 1;

    const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
    const toY = (val: number) => pad.t + plotH - ((val - minVal) / valRange) * plotH;

    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toX(points[0].step), toY(points[0].val));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(toX(points[i].step), toY(points[i].val));
    }
    ctx.stroke();

    // Points
    ctx.fillStyle = color;
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(toX(p.step), toY(p.val), 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Label
    ctx.fillStyle = "#aaa";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(label, pad.l + 4, pad.t - 4);

    // Latest value
    const lastVal = points[points.length - 1].val;
    ctx.fillStyle = color;
    ctx.textAlign = "right";
    ctx.fillText(format(lastVal), w - pad.r - 4, pad.t - 4);

    // Y-axis
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(format(minVal), pad.l - 4, h - pad.b);
    ctx.fillText(format(maxVal), pad.l - 4, pad.t + 10);
  }, [metrics, getY, color, label, format]);

  return <canvas ref={canvasRef} className="w-full" />;
}

// ── Adaptive Batch Step Chart ────────────────────────────────

export function AdaptiveBatchChart({ metrics }: { metrics: SymbioMetric[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const batchData = metrics.filter(m => m.adaptive_batch_size != null);
    if (batchData.length === 0) return;

    const w = canvas.parentElement?.clientWidth ?? 300;
    const h = 120;
    const ctx = setupCanvas(canvas, w, h);
    const pad = { l: 40, r: 10, t: 18, b: 20 };
    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, w, h);
    drawGrid(ctx, pad, w, h, 3);

    const minStep = batchData[0].step;
    const maxStep = batchData[batchData.length - 1].step;
    const stepRange = maxStep - minStep || 1;
    let minBatch = Infinity, maxBatch = -Infinity;
    for (const m of batchData) {
      const b = m.adaptive_batch_size!;
      if (b < minBatch) minBatch = b;
      if (b > maxBatch) maxBatch = b;
    }
    const batchRange = maxBatch - minBatch || 1;

    const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
    const toY = (batch: number) => pad.t + plotH - ((batch - minBatch) / batchRange) * plotH;

    // Step chart
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < batchData.length; i++) {
      const x = toX(batchData[i].step);
      const y = toY(batchData[i].adaptive_batch_size!);
      if (i === 0) ctx.moveTo(x, y);
      else {
        ctx.lineTo(x, y); // horizontal then vertical (step chart)
      }
      if (i < batchData.length - 1) {
        ctx.lineTo(toX(batchData[i + 1].step), y);
      }
    }
    ctx.stroke();

    // Change annotations
    for (const m of batchData) {
      if (m.batch_change_reason) {
        const x = toX(m.step);
        ctx.fillStyle = "rgba(248,113,113,0.2)";
        ctx.fillRect(x - 1, pad.t, 2, plotH);
      }
    }

    // Labels
    ctx.fillStyle = "#aaa";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Adaptive Batch", pad.l + 4, pad.t - 4);
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(String(minBatch), pad.l - 4, h - pad.b);
    ctx.fillText(String(maxBatch), pad.l - 4, pad.t + 10);
  }, [metrics]);

  return <canvas ref={canvasRef} className="w-full" />;
}

// ── Search Candidate Comparison ──────────────────────────────

interface CandidateStats {
  id: string;
  activation: string;
  generation: number;
  steps: number;
  bestLoss: number;
  bestValLoss: number;
  avgLoss: number;
  bestFitness: number;
}

function extractCandidateStats(metrics: SymbioMetric[]): CandidateStats[] {
  const candidates = new Map<string, { activation: string; generation: number; losses: number[]; valLosses: number[]; fitnesses: number[]; steps: number }>();

  for (const m of metrics) {
    const id = m.symbio_candidate_id;
    if (!id) continue;
    let entry = candidates.get(id);
    if (!entry) {
      entry = { activation: m.symbio_candidate_activation ?? "?", generation: m.symbio_generation ?? 0, losses: [], valLosses: [], fitnesses: [], steps: 0 };
      candidates.set(id, entry);
    }
    entry.losses.push(m.loss);
    if (m.val_loss != null) entry.valLosses.push(m.val_loss);
    if (m.fitness_score != null) entry.fitnesses.push(m.fitness_score);
    entry.steps++;
  }

  return Array.from(candidates.entries()).map(([id, e]) => ({
    id,
    activation: e.activation,
    generation: e.generation,
    steps: e.steps,
    bestLoss: Math.min(...e.losses),
    bestValLoss: e.valLosses.length > 0 ? Math.min(...e.valLosses) : Infinity,
    avgLoss: e.losses.reduce((a, b) => a + b, 0) / e.losses.length,
    bestFitness: e.fitnesses.length > 0 ? Math.max(...e.fitnesses) : -Infinity,
  }));
}

function SearchCandidateTable({ metrics }: { metrics: SymbioMetric[] }) {
  const candidates = extractCandidateStats(metrics);
  if (candidates.length === 0) return null;

  // Sort by best val loss (or best loss if no val)
  const sorted = [...candidates].sort((a, b) => {
    if (a.bestValLoss !== Infinity && b.bestValLoss !== Infinity) return a.bestValLoss - b.bestValLoss;
    if (a.bestValLoss !== Infinity) return -1;
    if (b.bestValLoss !== Infinity) return 1;
    return a.bestLoss - b.bestLoss;
  });

  const bestId = sorted[0]?.id;

  const ACTIVATION_COLORS: Record<string, string> = {
    gelu: "text-blue-400",
    silu: "text-green-400",
    relu: "text-yellow-400",
    swiglu: "text-purple-400",
  };

  return (
    <div className="mb-4 rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-3">
        <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
          Search Candidates ({candidates.length})
        </span>
      </div>
      <div className="grid grid-cols-[40px_100px_80px_50px_60px_80px_80px_80px_70px] gap-2 border-b border-border/50 px-4 py-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
        <span>#</span><span>ID</span><span>Activation</span><span>Gen</span><span>Steps</span><span>Best Loss</span><span>Best Val</span><span>Avg Loss</span><span>Fitness</span>
      </div>
      {sorted.map((c, i) => (
        <div key={c.id} className={`grid grid-cols-[40px_100px_80px_50px_60px_80px_80px_80px_70px] gap-2 border-b border-border/30 px-4 py-1.5 text-xs last:border-0 ${c.id === bestId ? "bg-green-500/5" : ""}`}>
          <span className="font-mono font-semibold text-text-muted">{i + 1}</span>
          <span className="truncate font-mono text-text-secondary">{c.id.replace(/^gen\d+_/, "").replace(/_\d+$/, "")}</span>
          <span className={`font-semibold ${ACTIVATION_COLORS[c.activation] ?? "text-text-secondary"}`}>{c.activation}</span>
          <span className="text-text-muted">{c.generation}</span>
          <span className="text-text-muted">{c.steps}</span>
          <span className="font-mono text-white">{c.bestLoss.toFixed(4)}</span>
          <span className="font-mono text-blue-400">{c.bestValLoss === Infinity ? "-" : c.bestValLoss.toFixed(4)}</span>
          <span className="font-mono text-text-secondary">{c.avgLoss.toFixed(4)}</span>
          <span className="font-mono text-green">{c.bestFitness === -Infinity ? "-" : c.bestFitness.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

function ActivationDistributionChart({ metrics }: { metrics: SymbioMetric[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Count steps per activation across the entire search
    const counts = new Map<string, number>();
    for (const m of metrics) {
      const act = m.symbio_candidate_activation;
      if (!act) continue;
      counts.set(act, (counts.get(act) ?? 0) + 1);
    }

    if (counts.size === 0) return;

    const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((s, [, n]) => s + n, 0);

    const w = canvas.parentElement?.clientWidth ?? 300;
    const h = 100;
    const ctx = setupCanvas(canvas, w, h);
    const pad = { l: 60, r: 20, t: 10, b: 25 };
    const barH = 18;

    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, w, h);

    const colors: Record<string, string> = { gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa" };
    const maxW = w - pad.l - pad.r;

    for (let i = 0; i < entries.length && i < 4; i++) {
      const [act, count] = entries[i];
      const y = pad.t + i * (barH + 4);
      const bw = (count / total) * maxW;

      ctx.fillStyle = colors[act] ?? "#888";
      ctx.fillRect(pad.l, y, bw, barH);

      ctx.fillStyle = "#aaa";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(act, pad.l - 6, y + barH / 2 + 4);

      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.fillText(`${((count / total) * 100).toFixed(0)}%`, pad.l + bw + 6, y + barH / 2 + 4);
    }
  }, [metrics]);

  return <canvas ref={canvasRef} className="w-full" />;
}

// ── Symbio Section (composite) ───────────────────────────────

export function SymbioSection({ metrics, run }: { metrics: SymbioMetric[]; run: { symbio?: number | null; symbio_config?: string | null; ffn_activation?: string | null } }) {
  const isSymbio = (run.symbio ?? 0) === 1;
  const hasClipData = metrics.some(m => m.clip_coef != null);
  const hasCusumData = metrics.some(m => m.cusum_grad != null);
  const hasSymbioMetrics = metrics.some(m => m.weight_entropy != null);
  const hasAdaptiveBatch = metrics.some(m => m.adaptive_batch_size != null);
  const hasSearchData = metrics.some(m => m.symbio_candidate_id != null);

  let symbioConfig: Record<string, unknown> | null = null;
  try {
    if (run.symbio_config) symbioConfig = JSON.parse(run.symbio_config);
  } catch { /* ignore */ }

  if (!isSymbio && !hasClipData) return null;

  return (
    <>
      {/* Clip telemetry — shown for any run with clip data */}
      {hasClipData && (
        <div className="mb-4 rounded-lg border border-border bg-surface p-4">
          <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Clip Telemetry
          </div>
          <ClipChart metrics={metrics} />
        </div>
      )}

      {isSymbio && (
        <>
          {/* Symbio badge header */}
          <div className="mb-4 flex items-center gap-2">
            <span className="rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-wider text-purple-400">
              Symbio
            </span>
            {run.ffn_activation && (
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[0.6rem] font-medium text-cyan-400">
                {run.ffn_activation.toUpperCase()}
              </span>
            )}
          </div>

          {/* Symbio stats grid */}
          <SymbioStatsGrid metrics={metrics} />

          {/* CUSUM monitor chart */}
          {hasCusumData && (
            <div className="mb-4 rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                CUSUM Change-Point Monitor
              </div>
              <CusumChart
                metrics={metrics}
                sensitivity={(symbioConfig?.cusumSensitivity as number) ?? 4.0}
              />
            </div>
          )}

          {/* Symbio metric mini charts */}
          {hasSymbioMetrics && (
            <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-lg border border-border bg-surface p-3">
                <SparseLineChart metrics={metrics} getY={m => m.weight_entropy} color="#a78bfa" label="Weight Entropy" format={v => v.toFixed(2) + " bits"} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-3">
                <SparseLineChart metrics={metrics} getY={m => m.effective_rank} color="#f59e0b" label="Effective Rank" format={v => v.toFixed(1)} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-3">
                <SparseLineChart metrics={metrics} getY={m => m.free_energy} color="#34d399" label="Free Energy" format={v => v.toFixed(4)} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-3">
                <SparseLineChart metrics={metrics} getY={m => m.fitness_score} color="#60a5fa" label="Fitness" format={v => v.toFixed(4)} />
              </div>
            </div>
          )}

          {/* Adaptive batch chart */}
          {hasAdaptiveBatch && (
            <div className="mb-4 rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                Adaptive Batch Size
              </div>
              <AdaptiveBatchChart metrics={metrics} />
            </div>
          )}

          {/* Search candidate comparison */}
          {hasSearchData && (
            <>
              <SearchCandidateTable metrics={metrics} />
              <div className="mb-4 rounded-lg border border-border bg-surface p-4">
                <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                  Activation Distribution
                </div>
                <ActivationDistributionChart metrics={metrics} />
              </div>
            </>
          )}

          {/* Symbio config panel */}
          {symbioConfig && (
            <details className="mb-4 rounded-lg border border-border bg-surface">
              <summary className="cursor-pointer px-4 py-3 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted hover:text-text">
                Symbio Config
              </summary>
              <pre className="overflow-x-auto px-4 pb-3 text-[0.6rem] text-text-muted">
                {JSON.stringify(symbioConfig, null, 2)}
              </pre>
            </details>
          )}
        </>
      )}
    </>
  );
}
