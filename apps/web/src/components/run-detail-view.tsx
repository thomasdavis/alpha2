"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";

// ── Types ────────────────────────────────────────────────────────

interface MetricData {
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
  timing_fwd_ms: number | null;
  timing_bwd_ms: number | null;
  timing_optim_ms: number | null;
  timing_data_ms: number | null;
  timing_flush_ms: number | null;
  timing_grad_norm_ms: number | null;
  timing_grad_clip_ms: number | null;
  gpu_ops_count: number | null;
}

interface CheckpointData {
  step: number;
  filename: string;
  file_size: number | null;
  created_at: string;
}

interface SampleData {
  idx: number;
  prompt: string;
  output: string;
  created_at: string;
}

interface RunData {
  id: string;
  run_id: string;
  config_hash: string;
  domain: string;
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_embd: number;
  n_head: number;
  dropout: number;
  total_iters: number;
  batch_size: number;
  lr: number;
  seed: number;
  backend: string;
  tokenizer: string;
  optimizer: string;
  model_config: string;
  train_config: string;
  status: string;
  latest_step: number;
  last_loss: number | null;
  best_val_loss: number | null;
  estimated_params: number | null;
  created_at: string;
  updated_at: string;
  disk_mtime: number | null;
}

export interface RunDetailProps {
  run: RunData;
  metrics: MetricData[];
  checkpoints: CheckpointData[];
  samples: SampleData[];
}

// ── Constants ────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, { badge: string; bar: string; gradient: string }> = {
  active: {
    badge: "border-green/20 bg-green-bg text-green",
    bar: "bg-gradient-to-r from-green/80 to-green",
    gradient: "from-green-bg/50",
  },
  completed: {
    badge: "border-blue/20 bg-blue-bg text-blue",
    bar: "bg-gradient-to-r from-blue/80 to-blue",
    gradient: "from-blue-bg/50",
  },
  stale: {
    badge: "border-yellow/20 bg-yellow-bg text-yellow",
    bar: "bg-gradient-to-r from-yellow/80 to-yellow",
    gradient: "from-yellow-bg/50",
  },
  failed: {
    badge: "border-red/20 bg-red-bg text-red",
    bar: "bg-gradient-to-r from-red/80 to-red",
    gradient: "from-red-bg/50",
  },
};

const DOMAIN_STYLES: Record<string, string> = {
  novels: "border-blue/20 bg-blue-bg text-blue",
  chords: "border-yellow/20 bg-yellow-bg text-yellow",
  abc: "border-green/20 bg-green-bg text-green",
  dumb_finance: "border-red/20 bg-red-bg text-red",
  concordance: "border-cyan-500/20 bg-cyan-950 text-cyan-400",
};

const TIMING_PHASES = [
  { key: "timing_fwd_ms" as const, label: "Forward", color: "#22d3ee" },
  { key: "timing_bwd_ms" as const, label: "Backward", color: "#f97316" },
  { key: "timing_grad_norm_ms" as const, label: "Grad Norm", color: "#a78bfa" },
  { key: "timing_optim_ms" as const, label: "Optimizer", color: "#10b981" },
  { key: "timing_flush_ms" as const, label: "GPU Sync", color: "#f43f5e" },
  { key: "timing_data_ms" as const, label: "Data", color: "#64748b" },
];

// ── Helpers ──────────────────────────────────────────────────────

function fmtParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function fmtLoss(v: number | null): string {
  return v != null ? v.toFixed(4) : "-";
}

function fmtBytes(b: number | null): string {
  if (b == null) return "-";
  if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}

function fmtDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

function fmtNum(n: number, decimals = 0): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

function timeAgo(iso: string | null): string {
  if (!iso) return "-";
  const ms = Date.now() - new Date(iso + "Z").getTime();
  if (ms < 0) return "now";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function fmtDate(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso + "Z");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    + " " + d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

// ── Small Components ─────────────────────────────────────────────

function Stat({ label, value, sub, color, tip }: {
  label: string; value: string; sub?: string; color?: string; tip?: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-surface-2/80 px-3 py-2.5">
      <div className={`font-mono text-sm font-bold ${color ?? "text-white"}`}>{value}</div>
      <div className="text-[0.6rem] uppercase tracking-wider text-text-muted">
        {label}{tip && <Tip text={tip} />}
      </div>
      {sub && <div className="mt-0.5 text-[0.6rem] text-text-muted">{sub}</div>}
    </div>
  );
}

function DetailRow({ label, value, tip }: {
  label: string; value: string | number | null | undefined; tip?: string;
}) {
  return (
    <div className="flex justify-between border-b border-border/30 py-1.5 last:border-0">
      <span className="text-[0.7rem] text-text-muted">{label}{tip && <Tip text={tip} />}</span>
      <span className="font-mono text-[0.7rem] text-text-primary">{value ?? "-"}</span>
    </div>
  );
}

// ── Interactive Loss Chart ───────────────────────────────────────

interface LossTooltip {
  pointX: number;
  mouseY: number;
  metric: MetricData;
  containerWidth: number;
}

function drawLossChart(
  canvas: HTMLCanvasElement,
  metrics: MetricData[],
  hoverIdx: number | null,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || metrics.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 16, right: 20, bottom: 36, left: 56 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const losses = metrics.map((m) => m.loss);
  const valPts = metrics.filter((m) => m.val_loss != null);
  const allVals = [...losses, ...valPts.map((v) => v.val_loss!)];
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const rangeL = maxL - minL || 1;
  const paddedMin = minL - rangeL * 0.05;
  const paddedMax = maxL + rangeL * 0.05;
  const paddedRange = paddedMax - paddedMin;
  const minStep = metrics[0].step;
  const maxStep = metrics[metrics.length - 1].step;
  const rangeS = maxStep - minStep || 1;

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;
  const sy = (loss: number) => pad.top + (1 - (loss - paddedMin) / paddedRange) * ch;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (i / 5) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    const val = paddedMax - (i / 5) * paddedRange;
    ctx.fillStyle = "#444";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
  }

  // X labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#444";
  ctx.font = "10px monospace";
  const stepTicks = [minStep, Math.round(minStep + rangeS * 0.25), Math.round(minStep + rangeS * 0.5), Math.round(minStep + rangeS * 0.75), maxStep];
  for (const s of stepTicks) {
    ctx.fillText(fmtNum(s), sx(s), h - pad.bottom + 16);
  }

  // Axis labels
  ctx.fillStyle = "#333";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("step", pad.left + cw / 2, h - 4);
  ctx.save();
  ctx.translate(10, pad.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("loss", 0, 0);
  ctx.restore();

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0, "rgba(245, 158, 11, 0.15)");
  grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
  ctx.beginPath();
  ctx.moveTo(sx(metrics[0].step), pad.top + ch);
  for (const m of metrics) ctx.lineTo(sx(m.step), sy(m.loss));
  ctx.lineTo(sx(metrics[metrics.length - 1].step), pad.top + ch);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Train loss line
  ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  for (let i = 0; i < metrics.length; i++) {
    if (i === 0) ctx.moveTo(sx(metrics[i].step), sy(metrics[i].loss));
    else ctx.lineTo(sx(metrics[i].step), sy(metrics[i].loss));
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Val loss
  if (valPts.length > 0) {
    ctx.shadowColor = "rgba(96, 165, 250, 0.3)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    for (let i = 0; i < valPts.length; i++) {
      const x = sx(valPts[i].step);
      const y = sy(valPts[i].val_loss!);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#60a5fa";
    for (const v of valPts) {
      ctx.beginPath();
      ctx.arc(sx(v.step), sy(v.val_loss!), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Hover crosshair
  if (hoverIdx != null && hoverIdx >= 0 && hoverIdx < metrics.length) {
    const m = metrics[hoverIdx];
    const hx = sx(m.step);
    const hy = sy(m.loss);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, pad.top);
    ctx.lineTo(hx, pad.top + ch);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, hy);
    ctx.lineTo(pad.left + cw, hy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Train loss dot
    ctx.beginPath();
    ctx.arc(hx, hy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#f59e0b";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(hx, hy, 8, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Val loss dot
    if (m.val_loss != null) {
      const vy = sy(m.val_loss);
      ctx.beginPath();
      ctx.arc(hx, vy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#60a5fa";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(hx, vy, 8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(96, 165, 250, 0.4)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  } else {
    // Last point glow
    const last = metrics[metrics.length - 1];
    ctx.beginPath();
    ctx.arc(sx(last.step), sy(last.loss), 4, 0, Math.PI * 2);
    ctx.fillStyle = "#f59e0b";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx(last.step), sy(last.loss), 7, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.3)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Legend
  const ly = h - 8;
  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(pad.left, ly - 1, 14, 2);
  ctx.fillStyle = "#666";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("train", pad.left + 18, ly + 3);
  if (valPts.length > 0) {
    ctx.fillStyle = "#60a5fa";
    ctx.beginPath();
    ctx.arc(pad.left + 60, ly, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#666";
    ctx.fillText("val", pad.left + 67, ly + 3);
  }
}

function InteractiveLossChart({ metrics }: { metrics: MetricData[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<LossTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);

  const draw = useCallback((idx: number | null = null) => {
    if (canvasRef.current) drawLossChart(canvasRef.current, metrics, idx);
  }, [metrics]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => draw(hoverRef.current));
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [draw]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length < 2) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const w = canvas.clientWidth;
    const padL = 56, padR = 20;
    const cw = w - padL - padR;

    if (mouseX < padL || mouseX > w - padR) {
      hoverRef.current = null;
      setTooltip(null);
      draw();
      return;
    }

    const minStep = metrics[0].step;
    const maxStep = metrics[metrics.length - 1].step;
    const rangeS = maxStep - minStep || 1;
    const stepAt = minStep + ((mouseX - padL) / cw) * rangeS;

    let lo = 0, hi = metrics.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (metrics[mid].step < stepAt) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && Math.abs(metrics[lo - 1].step - stepAt) < Math.abs(metrics[lo].step - stepAt)) lo--;

    hoverRef.current = lo;
    const m = metrics[lo];
    const pointX = padL + ((m.step - minStep) / rangeS) * cw;
    setTooltip({ pointX, mouseY, metric: m, containerWidth: w });
    draw(lo);
  }, [metrics, draw]);

  const onLeave = useCallback(() => {
    hoverRef.current = null;
    setTooltip(null);
    draw();
  }, [draw]);

  if (metrics.length < 2) {
    return (
      <div className="flex h-72 items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-xs text-text-muted">
        {metrics.length === 0 ? "No metrics data" : "Waiting for more data..."}
      </div>
    );
  }

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="h-72 w-full cursor-crosshair rounded-lg"
        onMouseMove={onMove}
        onMouseLeave={onLeave}
      />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 min-w-[185px] rounded-lg border border-border-2 bg-surface-2 p-3 shadow-xl"
          style={{
            left: tooltip.pointX < tooltip.containerWidth * 0.65 ? tooltip.pointX + 16 : undefined,
            right: tooltip.pointX >= tooltip.containerWidth * 0.65 ? tooltip.containerWidth - tooltip.pointX + 16 : undefined,
            top: Math.max(4, tooltip.mouseY - 100),
          }}
        >
          <div className="mb-1.5 font-mono text-[0.7rem] font-bold text-white">
            Step {tooltip.metric.step.toLocaleString()}
          </div>
          <div className="space-y-1 text-[0.68rem]">
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">Train Loss</span>
              <span className="font-mono text-yellow">{tooltip.metric.loss.toFixed(4)}</span>
            </div>
            {tooltip.metric.val_loss != null && (
              <div className="flex justify-between gap-4">
                <span className="text-text-muted">Val Loss</span>
                <span className="font-mono text-blue">{tooltip.metric.val_loss.toFixed(4)}</span>
              </div>
            )}
            <div className="my-1 border-t border-border/50" />
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">LR</span>
              <span className="font-mono text-text-secondary">{tooltip.metric.lr.toExponential(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">Grad Norm</span>
              <span className="font-mono text-text-secondary">{tooltip.metric.grad_norm.toFixed(3)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">Tok/sec</span>
              <span className="font-mono text-text-secondary">{fmtNum(tooltip.metric.tokens_per_sec)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">ms/iter</span>
              <span className="font-mono text-text-secondary">{tooltip.metric.ms_per_iter.toFixed(0)}</span>
            </div>
            {tooltip.metric.gpu_util_pct != null && (
              <div className="flex justify-between gap-4">
                <span className="text-text-muted">GPU</span>
                <span className="font-mono text-text-secondary">{tooltip.metric.gpu_util_pct.toFixed(0)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Mini Charts ──────────────────────────────────────────────────

interface MiniSeries {
  data: { step: number; value: number }[];
  color: string;
  label: string;
  axis?: "left" | "right";
}

function drawMiniChart(
  canvas: HTMLCanvasElement,
  series: MiniSeries[],
  opts: { logScale?: boolean; formatLeft?: (v: number) => string; formatRight?: (v: number) => string },
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const hasRight = series.some((s) => s.axis === "right");
  const pad = { top: 8, right: hasRight ? 42 : 12, bottom: 22, left: 42 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const allSteps = series.flatMap((s) => s.data.map((d) => d.step));
  const minStep = allSteps.length > 0 ? Math.min(...allSteps) : 0;
  const maxStep = allSteps.length > 0 ? Math.max(...allSteps) : 1;
  const rangeS = maxStep - minStep || 1;
  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;

  function computeRange(items: MiniSeries[], log: boolean) {
    const vals = items.flatMap((s) => s.data.map((d) => d.value));
    if (vals.length === 0) return { min: 0, max: 1 };
    if (log) {
      const pos = vals.filter((v) => v > 0);
      if (pos.length === 0) return { min: -2, max: 2 };
      const logVals = pos.map((v) => Math.log10(v));
      const lo = Math.floor(Math.min(...logVals));
      const hi = Math.ceil(Math.max(...logVals));
      return { min: lo, max: hi === lo ? lo + 1 : hi };
    }
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    const r = hi - lo || 1;
    return { min: lo - r * 0.05, max: hi + r * 0.05 };
  }

  const leftSeries = series.filter((s) => s.axis !== "right");
  const rightSeries = series.filter((s) => s.axis === "right");
  const leftRange = computeRange(leftSeries, !!opts.logScale);
  const rightRange = computeRange(rightSeries, false);

  const syLeft = (v: number) => {
    const nv = opts.logScale ? Math.log10(Math.max(v, 1e-10)) : v;
    return pad.top + (1 - (nv - leftRange.min) / (leftRange.max - leftRange.min || 1)) * ch;
  };
  const syRight = (v: number) =>
    pad.top + (1 - (v - rightRange.min) / (rightRange.max - rightRange.min || 1)) * ch;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    if (leftSeries.length > 0) {
      const frac = 1 - i / 4;
      let val: number;
      if (opts.logScale) val = Math.pow(10, leftRange.min + frac * (leftRange.max - leftRange.min));
      else val = leftRange.min + frac * (leftRange.max - leftRange.min);
      ctx.fillStyle = "#555";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText(opts.formatLeft ? opts.formatLeft(val) : (opts.logScale ? val.toExponential(0) : val.toPrecision(3)), pad.left - 4, y + 3);
    }
    if (hasRight && rightSeries.length > 0) {
      const frac = 1 - i / 4;
      const val = rightRange.min + frac * (rightRange.max - rightRange.min);
      ctx.fillStyle = "#555";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(opts.formatRight ? opts.formatRight(val) : val.toPrecision(3), w - pad.right + 4, y + 3);
    }
  }

  // Step labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#444";
  ctx.font = "9px monospace";
  const ticks = [minStep, Math.round(minStep + rangeS * 0.5), maxStep];
  for (const s of ticks) ctx.fillText(String(s), sx(s), h - pad.bottom + 12);

  // Lines
  for (const s of series) {
    if (s.data.length === 0) continue;
    const toY = s.axis === "right" ? syRight : syLeft;
    ctx.shadowColor = s.color + "4d";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (let i = 0; i < s.data.length; i++) {
      const x = sx(s.data[i].step);
      const v = s.data[i].value;
      const y = toY(opts.logScale && s.axis !== "right" ? Math.max(v, 1e-10) : v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Legend
  let lx = pad.left;
  const ly = h - 4;
  ctx.font = "9px sans-serif";
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, ly - 1, 10, 2);
    ctx.fillStyle = "#666";
    ctx.textAlign = "left";
    ctx.fillText(s.label, lx + 13, ly + 2);
    lx += ctx.measureText(s.label).width + 24;
    if (lx > w - 30) break;
  }
}

function MiniChart({ metrics, title, buildSeries, logScale, formatLeft, formatRight, noDataMsg }: {
  metrics: MetricData[];
  title: string;
  buildSeries: (m: MetricData[]) => MiniSeries[];
  logScale?: boolean;
  formatLeft?: (v: number) => string;
  formatRight?: (v: number) => string;
  noDataMsg?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const series = useMemo(() => buildSeries(metrics), [metrics, buildSeries]);
  const hasData = series.some((s) => s.data.length >= 2);

  useEffect(() => {
    if (canvasRef.current && hasData) {
      drawMiniChart(canvasRef.current, series, { logScale, formatLeft, formatRight });
    }
  }, [series, hasData, logScale, formatLeft, formatRight]);

  return (
    <div>
      <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">{title}</div>
      {hasData ? (
        <canvas ref={canvasRef} className="h-[140px] w-full rounded-lg" />
      ) : (
        <div className="flex h-[140px] items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-[0.65rem] text-text-muted">
          {noDataMsg ?? "No data"}
        </div>
      )}
    </div>
  );
}

// ── Step Time Chart ──────────────────────────────────────────────

function drawStepTimeChart(canvas: HTMLCanvasElement, metrics: MetricData[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const data = metrics.filter((m) => m.timing_fwd_ms != null);
  if (data.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 10, right: 12, bottom: 26, left: 44 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  const minStep = data[0].step;
  const maxStep = data[data.length - 1].step;
  const rangeS = maxStep - minStep || 1;

  const stacked = data.map((m) => {
    const phases = TIMING_PHASES.map((p) => ((m as any)[p.key] as number) ?? 0);
    return { step: m.step, phases, total: phases.reduce((a, b) => a + b, 0) };
  });
  const maxTotal = Math.max(...stacked.map((s) => s.total), 1);

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;
  const sy = (ms: number) => pad.top + (1 - ms / maxTotal) * ch;

  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    const val = maxTotal * (1 - i / 4);
    ctx.fillStyle = "#444";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val >= 1000 ? (val / 1000).toFixed(1) + "s" : val.toFixed(0) + "ms", pad.left - 6, y + 3);
  }

  ctx.textAlign = "center";
  ctx.fillStyle = "#444";
  ctx.font = "9px monospace";
  for (const s of [minStep, Math.round(minStep + rangeS * 0.5), maxStep]) {
    ctx.fillText(String(s), sx(s), h - pad.bottom + 12);
  }

  for (let pi = TIMING_PHASES.length - 1; pi >= 0; pi--) {
    ctx.beginPath();
    for (let i = 0; i < stacked.length; i++) {
      const cum = stacked[i].phases.slice(0, pi + 1).reduce((a, b) => a + b, 0);
      const x = sx(stacked[i].step);
      if (i === 0) ctx.moveTo(x, sy(cum)); else ctx.lineTo(x, sy(cum));
    }
    for (let i = stacked.length - 1; i >= 0; i--) {
      const below = pi > 0 ? stacked[i].phases.slice(0, pi).reduce((a, b) => a + b, 0) : 0;
      ctx.lineTo(sx(stacked[i].step), sy(below));
    }
    ctx.closePath();
    ctx.fillStyle = TIMING_PHASES[pi].color + "80";
    ctx.fill();
  }

  // Legend
  ctx.font = "8px monospace";
  let lx = pad.left;
  const ly = h - 3;
  for (const p of TIMING_PHASES) {
    ctx.fillStyle = p.color;
    ctx.fillRect(lx, ly - 4, 6, 4);
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText(p.label, lx + 8, ly);
    lx += ctx.measureText(p.label).width + 16;
    if (lx > w - 30) break;
  }
}

function StepTimeChart({ metrics }: { metrics: MetricData[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hasData = metrics.some((m) => m.timing_fwd_ms != null);

  useEffect(() => {
    if (canvasRef.current && hasData) drawStepTimeChart(canvasRef.current, metrics);
  }, [metrics, hasData]);

  return (
    <div>
      <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Step Time Breakdown</div>
      {hasData ? (
        <canvas ref={canvasRef} className="h-[140px] w-full rounded-lg" />
      ) : (
        <div className="flex h-[140px] items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-[0.65rem] text-text-muted">
          No timing data
        </div>
      )}
    </div>
  );
}

// ── Build Series Helpers ─────────────────────────────────────────

const buildGpuSeries = (m: MetricData[]): MiniSeries[] => {
  const vram = m.filter((x) => x.gpu_vram_used_mb != null).map((x) => ({ step: x.step, value: x.gpu_vram_used_mb! }));
  const util = m.filter((x) => x.gpu_util_pct != null).map((x) => ({ step: x.step, value: x.gpu_util_pct! }));
  return [
    { data: vram, color: "#10b981", label: "VRAM", axis: "left" as const },
    { data: util, color: "#f59e0b", label: "GPU%", axis: "right" as const },
  ];
};

const buildLrSeries = (m: MetricData[]): MiniSeries[] => {
  const lr = m.filter((x) => x.lr > 0).map((x) => ({ step: x.step, value: x.lr }));
  return [{ data: lr, color: "#22d3ee", label: "LR" }];
};

const buildGradNormSeries = (m: MetricData[]): MiniSeries[] => {
  const gn = m.filter((x) => x.grad_norm > 0 && isFinite(x.grad_norm)).map((x) => ({ step: x.step, value: x.grad_norm }));
  return [{ data: gn, color: "#f97316", label: "norm" }];
};

// ── Main Component ───────────────────────────────────────────────

export function RunDetailView({ run, metrics, checkpoints, samples }: RunDetailProps) {
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [showTrainConfig, setShowTrainConfig] = useState(false);

  const modelConfig = useMemo(() => {
    try { return JSON.parse(run.model_config); } catch { return null; }
  }, [run.model_config]);

  const trainConfig = useMemo(() => {
    try { return JSON.parse(run.train_config); } catch { return null; }
  }, [run.train_config]);

  const last = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const isActive = run.status === "active";
  const ss = STATUS_STYLES[run.status] ?? STATUS_STYLES.stale;
  const progress = run.total_iters > 0 ? Math.min(100, (run.latest_step / run.total_iters) * 100) : 0;

  const stats = useMemo(() => {
    if (metrics.length === 0) return null;
    const losses = metrics.map((m) => m.loss);
    const minLoss = Math.min(...losses);
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    const lossDropPct = firstLoss > 0 ? ((firstLoss - lastLoss) / firstLoss) * 100 : 0;

    const recent = metrics.slice(-20);
    const avgTps = recent.reduce((s, m) => s + m.tokens_per_sec, 0) / recent.length;
    const avgMs = recent.reduce((s, m) => s + m.ms_per_iter, 0) / recent.length;
    const avgGradNorm = recent.reduce((s, m) => s + m.grad_norm, 0) / recent.length;

    const totalElapsed = metrics.reduce((s, m) => s + m.elapsed_ms, 0);
    const stepsRemaining = run.total_iters - (last?.step ?? 0);
    const eta = stepsRemaining > 0 && avgMs > 0 ? avgMs * stepsRemaining : 0;

    const valLosses = metrics.filter((m) => m.val_loss != null);
    const bestVal = valLosses.length > 0 ? Math.min(...valLosses.map((m) => m.val_loss!)) : null;
    const lastVal = valLosses.length > 0 ? valLosses[valLosses.length - 1].val_loss : null;

    const totalTokens = metrics.reduce((s, m) => s + (m.tokens_per_sec * m.elapsed_ms / 1000), 0);

    // Timing stats
    const timed = recent.filter((m) => m.timing_fwd_ms != null);
    const avgFwd = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_fwd_ms ?? 0), 0) / timed.length : null;
    const avgBwd = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_bwd_ms ?? 0), 0) / timed.length : null;
    const avgFlush = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_flush_ms ?? 0), 0) / timed.length : null;
    const avgGpuOps = timed.length > 0 ? timed.reduce((s, m) => s + (m.gpu_ops_count ?? 0), 0) / timed.length : null;

    // MFU: 6 * params * tokens_per_step / (step_time * peak_flops)
    const totalParams = run.estimated_params ?? 0;
    const tokensPerStep = run.batch_size * run.block_size;
    const flopsPerStep = 6 * totalParams * tokensPerStep;
    const mfu = avgMs > 0 && totalParams > 0 ? (flopsPerStep / (avgMs / 1000)) / 30.3e12 * 100 : null;

    return {
      minLoss, lossDropPct, avgTps, avgMs, avgGradNorm, totalElapsed, eta,
      bestVal, lastVal, totalTokens,
      avgFwd, avgBwd, avgFlush, avgGpuOps, mfu,
    };
  }, [metrics, run, last]);

  return (
    <>
      {/* Breadcrumb */}
      <nav className="mb-4 flex items-center gap-1.5 text-xs text-text-muted">
        <Link href="/runs" className="hover:text-text-secondary">Training Runs</Link>
        <span>/</span>
        <span className="text-text-secondary">{run.id}</span>
      </nav>

      {/* Header */}
      <div className={`mb-6 overflow-hidden rounded-xl border ${isActive ? "border-green/20" : "border-border"} bg-surface`}>
        <div className={`h-0.5 ${ss.bar}`} />
        <div className={`border-b border-border/50 bg-gradient-to-r ${ss.gradient} to-transparent px-5 py-4`}>
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="font-mono text-lg font-bold text-white">{run.id}</span>
            <span className={`flex items-center gap-1 rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${ss.badge}`}>
              {isActive && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green" />}
              {run.status}
            </span>
            <span className={`rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${DOMAIN_STYLES[run.domain] ?? "border-border bg-surface-2 text-text-secondary"}`}>
              {run.domain}
            </span>
            <span className="rounded-md border border-border bg-surface-2 px-2 py-0.5 text-[0.65rem] font-semibold text-text-secondary">
              {fmtParams(run.estimated_params)} params
            </span>
            <span className="ml-auto text-xs text-text-muted">
              {stats && stats.totalElapsed > 0 && <>{fmtDuration(stats.totalElapsed)} elapsed</>}
              {isActive && stats && stats.eta > 0 && <> &middot; ~{fmtDuration(stats.eta)} remaining</>}
              {!isActive && <> &middot; Updated {timeAgo(run.updated_at)}</>}
            </span>
          </div>
          <div className="mt-1.5 text-[0.7rem] text-text-muted">
            {run.n_layer}L / {run.n_embd}D / {run.n_head}H &middot; {run.backend} &middot; {run.tokenizer} &middot; {run.optimizer}
            &middot; Created {fmtDate(run.created_at)}
          </div>
        </div>

        {/* Progress */}
        <div className="px-5 py-4">
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="text-xs text-text-secondary">
              Step <span className="font-mono font-bold text-white">{fmtNum(run.latest_step)}</span>
              <span className="text-text-muted"> / {fmtNum(run.total_iters)}</span>
            </span>
            <span className="font-mono text-sm font-bold text-white">{progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-2">
            <div
              className={`h-full rounded-full transition-all duration-700 ${ss.bar}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
        <Stat label="Loss" value={last ? fmtLoss(last.loss) : "-"} color="text-yellow" tip={tips.loss} />
        <Stat
          label="Best Loss"
          value={stats && isFinite(stats.minLoss) ? fmtLoss(stats.minLoss) : "-"}
          sub={stats && isFinite(stats.lossDropPct) ? `${stats.lossDropPct > 0 ? "-" : ""}${Math.abs(stats.lossDropPct).toFixed(1)}% from start` : undefined}
          tip={tips.lastLoss}
        />
        <Stat
          label="Val Loss"
          value={stats?.lastVal != null ? fmtLoss(stats.lastVal) : "-"}
          sub={stats?.bestVal != null ? `best: ${fmtLoss(stats.bestVal)}` : undefined}
          color={stats?.lastVal != null ? "text-blue" : undefined}
          tip={tips.valLoss}
        />
        <Stat label="Learning Rate" value={last?.lr != null ? last.lr.toExponential(2) : "-"} tip={tips.lr} />
        <Stat label="Throughput" value={stats ? fmtNum(stats.avgTps) : "-"} sub="tok/s (avg)" color="text-green" tip={tips.throughput} />
        <Stat label="Speed" value={stats ? fmtNum(stats.avgMs) : "-"} sub="ms/iter (avg)" tip={tips.msPerIter} />
        <Stat
          label="Grad Norm"
          value={last?.grad_norm != null ? last.grad_norm.toFixed(3) : "-"}
          sub={stats && isFinite(stats.avgGradNorm) ? `avg: ${stats.avgGradNorm.toFixed(3)}` : undefined}
          tip={tips.gradNorm}
        />
        <Stat label="Tokens" value={stats ? fmtParams(Math.round(stats.totalTokens)) : "-"} sub="processed" />
      </div>

      {/* Timing stats */}
      {stats?.avgFwd != null && (
        <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Forward" value={`${stats.avgFwd.toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${(stats.avgFwd / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-cyan-400" />
          <Stat label="Backward" value={`${(stats.avgBwd ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgBwd ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-orange-400" />
          <Stat label="GPU Sync" value={`${(stats.avgFlush ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgFlush ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-rose-400" />
          <Stat label="GPU Ops" value={stats.avgGpuOps != null ? fmtNum(stats.avgGpuOps) : "-"} sub="per step" />
          <Stat
            label="MFU"
            value={stats.mfu != null ? `${stats.mfu.toFixed(1)}%` : "-"}
            sub="model FLOPS util"
            color={stats.mfu != null && stats.mfu > 50 ? "text-green" : stats.mfu != null && stats.mfu > 10 ? "text-yellow" : "text-red"}
          />
          <Stat
            label="Bwd/Fwd"
            value={stats.avgFwd != null && stats.avgBwd != null ? `${(stats.avgBwd / stats.avgFwd).toFixed(1)}x` : "-"}
            sub="ratio"
          />
        </div>
      )}

      {/* Chart + config sidebar */}
      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_280px]">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Loss Curve <Tip text={tips.lossChart} />
          </div>
          <InteractiveLossChart metrics={metrics} />
        </div>

        <div className="space-y-3">
          {/* Architecture */}
          <div className="rounded-lg border border-border/60 bg-surface p-3">
            <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Architecture</div>
            <DetailRow label="Layers" value={run.n_layer} tip={tips.nLayer} />
            <DetailRow label="Embedding" value={run.n_embd} tip={tips.nEmbd} />
            <DetailRow label="Heads" value={run.n_head} tip={tips.nHead} />
            <DetailRow label="Vocab" value={fmtNum(run.vocab_size)} tip={tips.vocabSize} />
            <DetailRow label="Context" value={run.block_size} tip={tips.blockSize} />
            <DetailRow label="Dropout" value={run.dropout} tip={tips.dropout} />
            <DetailRow label="Parameters" value={fmtParams(run.estimated_params)} tip={tips.params} />
          </div>

          {/* Training config */}
          <div className="rounded-lg border border-border/60 bg-surface p-3">
            <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Training Config</div>
            <DetailRow label="Total iters" value={fmtNum(run.total_iters)} tip={tips.totalIters} />
            <DetailRow label="Batch size" value={run.batch_size} tip={tips.batchSize} />
            <DetailRow label="Max LR" value={run.lr} tip={tips.lr} />
            <DetailRow label="Optimizer" value={run.optimizer} tip={tips.optimizer} />
            <DetailRow label="Backend" value={run.backend} tip={tips.backend} />
            <DetailRow label="Tokenizer" value={run.tokenizer} tip={tips.tokenizer} />
            <DetailRow label="Seed" value={run.seed} tip={tips.seed} />
            {trainConfig?.weightDecay != null && <DetailRow label="Weight decay" value={trainConfig.weightDecay} tip={tips.weightDecay} />}
            {trainConfig?.gradClip != null && <DetailRow label="Grad clip" value={trainConfig.gradClip} tip={tips.gradClip} />}
            {trainConfig?.evalInterval != null && <DetailRow label="Eval interval" value={trainConfig.evalInterval} tip={tips.evalInterval} />}
          </div>
        </div>
      </div>

      {/* Mini charts */}
      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MiniChart
          metrics={metrics}
          title="GPU & VRAM"
          noDataMsg="No GPU data"
          formatLeft={(v) => (v / 1024).toFixed(1) + "G"}
          formatRight={(v) => v.toFixed(0) + "%"}
          buildSeries={buildGpuSeries}
        />
        <MiniChart
          metrics={metrics}
          title="Learning Rate"
          formatLeft={(v) => v.toExponential(1)}
          buildSeries={buildLrSeries}
        />
        <MiniChart
          metrics={metrics}
          title="Grad Norm"
          logScale
          formatLeft={(v) => v.toExponential(0)}
          buildSeries={buildGradNormSeries}
        />
        <StepTimeChart metrics={metrics} />
      </div>

      {/* Checkpoints */}
      <div className="mb-6 rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-3">
          <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Checkpoints ({checkpoints.length}) <Tip text={tips.checkpoint} />
          </span>
        </div>
        {checkpoints.length === 0 ? (
          <div className="px-4 py-6 text-center text-xs text-text-muted">No checkpoints saved</div>
        ) : (
          <>
            <div className="grid grid-cols-[80px_1fr_90px_100px_52px] gap-4 border-b border-border/50 px-4 py-2 text-[0.62rem] font-semibold uppercase tracking-wider text-text-muted">
              <span>Step</span><span>Filename</span><span>Size</span><span>Created</span><span></span>
            </div>
            {checkpoints.map((c) => (
              <div key={c.step} className="grid grid-cols-[80px_1fr_90px_100px_52px] gap-4 border-b border-border/30 px-4 py-2 text-xs last:border-0">
                <span className="font-mono font-semibold text-white">{fmtNum(c.step)}</span>
                <span className="truncate font-mono text-text-secondary">{c.filename}</span>
                <span className="text-text-muted">{fmtBytes(c.file_size)}</span>
                <span className="text-text-muted">{c.created_at ? timeAgo(c.created_at) : "-"}</span>
                <a
                  href={`/api/runs/${encodeURIComponent(run.id)}/download/${encodeURIComponent(c.filename)}`}
                  className="text-text-muted transition-colors hover:text-accent"
                  title="Download checkpoint"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" />
                    <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                  </svg>
                </a>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Samples */}
      {samples.length > 0 && (
        <div className="mb-6">
          <div className="mb-3 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Sample Generations ({samples.length})
          </div>
          <div className="space-y-3">
            {samples.map((s) => (
              <div key={s.idx} className="overflow-hidden rounded-lg border border-border bg-surface">
                <div className="flex items-center gap-2 border-b border-border/50 bg-surface-2/50 px-4 py-2">
                  <span className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.62rem] font-bold text-text-secondary">#{s.idx + 1}</span>
                  <span className="text-[0.62rem] text-text-muted">{s.created_at ? timeAgo(s.created_at) : ""}</span>
                </div>
                <div className="p-4">
                  <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Prompt</div>
                  <div className="mb-3 rounded border border-border/50 bg-[#0d0d0d] px-3 py-2 font-mono text-xs leading-relaxed text-text-secondary">
                    {s.prompt}
                  </div>
                  <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Output</div>
                  <div className="whitespace-pre-wrap rounded border border-border/50 bg-[#0d0d0d] px-3 py-2 font-mono text-xs leading-relaxed text-text-primary">
                    {s.output}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Raw config */}
      <div className="mb-6 space-y-2">
        <button
          onClick={() => setShowModelConfig(!showModelConfig)}
          className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-left text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted transition-colors hover:border-border-2"
        >
          <span className="text-[0.7rem]">{showModelConfig ? "\u25BC" : "\u25B6"}</span>
          Model Config JSON <Tip text={tips.rawConfig} />
        </button>
        {showModelConfig && modelConfig && (
          <pre className="overflow-x-auto rounded-lg border border-border bg-[#0d0d0d] p-4 font-mono text-xs leading-relaxed text-text-secondary">
            {JSON.stringify(modelConfig, null, 2)}
          </pre>
        )}

        <button
          onClick={() => setShowTrainConfig(!showTrainConfig)}
          className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-left text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted transition-colors hover:border-border-2"
        >
          <span className="text-[0.7rem]">{showTrainConfig ? "\u25BC" : "\u25B6"}</span>
          Train Config JSON <Tip text={tips.rawConfig} />
        </button>
        {showTrainConfig && trainConfig && (
          <pre className="overflow-x-auto rounded-lg border border-border bg-[#0d0d0d] p-4 font-mono text-xs leading-relaxed text-text-secondary">
            {JSON.stringify(trainConfig, null, 2)}
          </pre>
        )}
      </div>
    </>
  );
}
