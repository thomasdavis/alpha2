"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Tip } from "@/components/tooltip";

// ── Types ────────────────────────────────────────────────────────

export interface ChartMetric {
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
  timing_fwd_ms: number | null;
  timing_bwd_ms: number | null;
  timing_optim_ms: number | null;
  timing_data_ms: number | null;
  timing_flush_ms: number | null;
  timing_grad_norm_ms: number | null;
  timing_grad_clip_ms: number | null;
  gpu_ops_count: number | null;
}

export interface ChartCheckpoint {
  step: number;
}

export interface MiniSeries {
  data: { step: number; value: number }[];
  color: string;
  label: string;
  axis?: "left" | "right";
  format?: (v: number) => string;
}

interface MiniChartTooltip {
  pointX: number;
  mouseY: number;
  step: number;
  values: { label: string; value: number; color: string; formatted: string }[];
  containerWidth: number;
}

interface StepTimeTooltip {
  pointX: number;
  mouseY: number;
  step: number;
  phases: { label: string; value: number; color: string }[];
  total: number;
  containerWidth: number;
}

// ── Event Marker Types ──────────────────────────────────────────

export type MarkerType = "checkpoints" | "bestVal" | "warmupEnd" | "gradSpikes" | "lossSpikes" | "overfit";
export type MarkerVisibility = Record<MarkerType, boolean>;

export const DEFAULT_MARKERS: MarkerVisibility = {
  checkpoints: true,
  bestVal: true,
  warmupEnd: true,
  overfit: true,
  gradSpikes: false,
  lossSpikes: false,
};

export const MARKER_COLORS: Record<MarkerType, string> = {
  checkpoints: "#34d399",
  bestVal: "#22d3ee",
  warmupEnd: "#a78bfa",
  gradSpikes: "#fb923c",
  lossSpikes: "#f472b6",
  overfit: "#ef4444",
};

export const MARKER_LABELS: Record<MarkerType, string> = {
  checkpoints: "Checkpoints",
  bestVal: "Best Val",
  warmupEnd: "Warmup End",
  gradSpikes: "Grad Spikes",
  lossSpikes: "Loss Spikes",
  overfit: "Overfit",
};

export interface ComputedEvents {
  checkpointSteps: number[];
  bestValStep: number | null;
  bestValLoss: number | null;
  warmupEndStep: number | null;
  gradSpikeSteps: number[];
  lossSpikeSteps: number[];
  overfitStep: number | null;
}

export const MARKERS_STORAGE_KEY = "alpha-chart-markers";

// ── Constants ────────────────────────────────────────────────────

export const TIMING_PHASES = [
  { key: "timing_fwd_ms" as const, label: "Forward", color: "#22d3ee" },
  { key: "timing_bwd_ms" as const, label: "Backward", color: "#f97316" },
  { key: "timing_grad_norm_ms" as const, label: "Grad Norm", color: "#a78bfa" },
  { key: "timing_optim_ms" as const, label: "Optimizer", color: "#10b981" },
  { key: "timing_flush_ms" as const, label: "GPU Sync", color: "#f43f5e" },
  { key: "timing_data_ms" as const, label: "Data", color: "#64748b" },
];

// ── Formatting Helpers ──────────────────────────────────────────

export function fmtParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

export function fmtLoss(v: number | null): string {
  return v != null ? v.toFixed(4) : "-";
}

export function fmtBytes(b: number | null): string {
  if (b == null) return "-";
  if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}

export function fmtDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

export function fmtNum(n: number | null | undefined, decimals = 0): string {
  if (n == null) return "0";
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

export function timeAgo(iso: string | null): string {
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

export function fmtDate(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso + "Z");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    + " " + d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

// ── Marker Persistence ──────────────────────────────────────────

function loadMarkerPrefs(): MarkerVisibility {
  if (typeof window === "undefined") return DEFAULT_MARKERS;
  try {
    const raw = localStorage.getItem(MARKERS_STORAGE_KEY);
    if (!raw) return DEFAULT_MARKERS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_MARKERS, ...parsed };
  } catch {
    return DEFAULT_MARKERS;
  }
}

function saveMarkerPrefs(m: MarkerVisibility) {
  try { localStorage.setItem(MARKERS_STORAGE_KEY, JSON.stringify(m)); } catch {}
}

// ── Event Detection ─────────────────────────────────────────────

function detectBestValStep(metrics: ChartMetric[]): { step: number; loss: number } | null {
  const valPts = metrics.filter((m) => m.val_loss != null);
  if (valPts.length === 0) return null;
  let best = valPts[0];
  for (let i = 1; i < valPts.length; i++) {
    if (valPts[i].val_loss! < best.val_loss!) best = valPts[i];
  }
  return { step: best.step, loss: best.val_loss! };
}

function detectWarmupEnd(metrics: ChartMetric[]): number | null {
  if (metrics.length < 3) return null;
  let peakIdx = 0;
  for (let i = 1; i < metrics.length; i++) {
    if (metrics[i].lr > metrics[peakIdx].lr) peakIdx = i;
  }
  if (peakIdx <= 1 || peakIdx >= metrics.length - 2) return null;
  return metrics[peakIdx].step;
}

function detectGradNormSpikes(metrics: ChartMetric[]): number[] {
  if (metrics.length < 15) return [];
  const spikes: number[] = [];
  let ema = metrics[0].grad_norm;
  const alpha = 0.05;
  for (let i = 1; i < metrics.length; i++) {
    const gn = metrics[i].grad_norm;
    ema = alpha * gn + (1 - alpha) * ema;
    if (i >= 10 && gn > 3 * ema && gn > 0.5) {
      spikes.push(metrics[i].step);
    }
  }
  return spikes;
}

function detectLossSpikes(metrics: ChartMetric[]): number[] {
  if (metrics.length < 25) return [];
  const spikes: number[] = [];
  const winSize = 20;
  for (let i = winSize; i < metrics.length; i++) {
    let sum = 0;
    for (let j = i - winSize; j < i; j++) sum += metrics[j].loss;
    const mean = sum / winSize;
    const cur = metrics[i].loss;
    const prev = metrics[i - 1].loss;
    if (cur > 1.5 * mean || (cur - prev) > 0.3) {
      spikes.push(metrics[i].step);
    }
  }
  return spikes;
}

function detectOverfitStep(metrics: ChartMetric[]): number | null {
  const valPts = metrics.filter((m) => m.val_loss != null);
  if (valPts.length < 3) return null;

  let minIdx = 0;
  for (let i = 1; i < valPts.length; i++) {
    if (valPts[i].val_loss! < valPts[minIdx].val_loss!) minIdx = i;
  }

  if (minIdx >= valPts.length - 2) return null;

  const minVal = valPts[minIdx].val_loss!;
  const afterMin = valPts.slice(minIdx + 1);
  const avgAfter = afterMin.reduce((s, m) => s + m.val_loss!, 0) / afterMin.length;
  const pctRise = (avgAfter - minVal) / minVal;

  let consecutive = 0;
  let prev = minVal;
  for (const m of afterMin) {
    if (m.val_loss! > prev) { consecutive++; prev = m.val_loss!; }
    else break;
  }

  if (pctRise < 0.02 && consecutive < 3) return null;

  return valPts[minIdx].step;
}

function computeEvents(metrics: ChartMetric[], checkpoints: ChartCheckpoint[]): ComputedEvents {
  const bestVal = detectBestValStep(metrics);
  return {
    checkpointSteps: checkpoints.map((c) => c.step),
    bestValStep: bestVal?.step ?? null,
    bestValLoss: bestVal?.loss ?? null,
    warmupEndStep: detectWarmupEnd(metrics),
    gradSpikeSteps: detectGradNormSpikes(metrics),
    lossSpikeSteps: detectLossSpikes(metrics),
    overfitStep: detectOverfitStep(metrics),
  };
}

// ── Small Components ─────────────────────────────────────────────

export function Stat({ label, value, sub, color, tip }: {
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

export function DetailRow({ label, value, tip }: {
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
  metric: ChartMetric;
  containerWidth: number;
}

function drawLossChart(
  canvas: HTMLCanvasElement,
  metrics: ChartMetric[],
  hoverIdx: number | null,
  events: ComputedEvents,
  markers: MarkerVisibility,
  pinnedStep: number | null = null,
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
  const minStep = 0;
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

  // ── Event Markers ──────────────────────────────────────────────

  // Checkpoint vertical lines
  if (markers.checkpoints && events.checkpointSteps.length > 0) {
    ctx.save();
    ctx.strokeStyle = MARKER_COLORS.checkpoints + "80";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (const step of events.checkpointSteps) {
      if (step < minStep || step > maxStep) continue;
      const cx2 = sx(step);
      ctx.beginPath();
      ctx.moveTo(cx2, pad.top);
      ctx.lineTo(cx2, pad.top + ch);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
  }

  // LR warmup end
  if (markers.warmupEnd && events.warmupEndStep != null) {
    const wx = sx(events.warmupEndStep);
    ctx.save();
    ctx.strokeStyle = MARKER_COLORS.warmupEnd + "99";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(wx, pad.top);
    ctx.lineTo(wx, pad.top + ch);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = MARKER_COLORS.warmupEnd;
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("warmup", wx, pad.top - 4);
    ctx.restore();
  }

  // Overfit onset
  if (markers.overfit && events.overfitStep != null) {
    const ox = sx(events.overfitStep);
    ctx.save();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(ox, pad.top);
    ctx.lineTo(ox, pad.top + ch);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("overfit", ox, pad.top - 4);
    ctx.restore();
  }

  // Grad norm spike markers (downward triangles at top)
  if (markers.gradSpikes && events.gradSpikeSteps.length > 0) {
    ctx.fillStyle = MARKER_COLORS.gradSpikes;
    for (const step of events.gradSpikeSteps) {
      if (step < minStep || step > maxStep) continue;
      const tx = sx(step);
      const ty = pad.top + 6;
      ctx.beginPath();
      ctx.moveTo(tx - 4, ty - 5);
      ctx.lineTo(tx + 4, ty - 5);
      ctx.lineTo(tx, ty + 3);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Loss spike markers (downward triangles, offset below grad spikes)
  if (markers.lossSpikes && events.lossSpikeSteps.length > 0) {
    ctx.fillStyle = MARKER_COLORS.lossSpikes;
    for (const step of events.lossSpikeSteps) {
      if (step < minStep || step > maxStep) continue;
      const tx = sx(step);
      const ty = pad.top + 18;
      ctx.beginPath();
      ctx.moveTo(tx - 4, ty - 5);
      ctx.lineTo(tx + 4, ty - 5);
      ctx.lineTo(tx, ty + 3);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Best val loss diamond
  if (markers.bestVal && events.bestValStep != null && events.bestValLoss != null) {
    const bx = sx(events.bestValStep);
    const by = sy(events.bestValLoss);
    ctx.fillStyle = MARKER_COLORS.bestVal;
    ctx.beginPath();
    ctx.moveTo(bx, by - 6);
    ctx.lineTo(bx + 5, by);
    ctx.lineTo(bx, by + 6);
    ctx.lineTo(bx - 5, by);
    ctx.closePath();
    ctx.fill();
  }

  // Pinned step marker
  if (pinnedStep != null) {
    const padObj = { top: pad.top, right: pad.right, bottom: pad.bottom, left: pad.left };
    drawPinnedStep(ctx, pinnedStep, sx, padObj, ch, w);
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

  // Legend (train + val only; markers are the toggle chips)
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

export function InteractiveLossChart({ metrics, checkpoints, pinnedStep, onPinStep }: { metrics: ChartMetric[]; checkpoints: ChartCheckpoint[]; pinnedStep?: number | null; onPinStep?: (step: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<LossTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const [markers, setMarkers] = useState<MarkerVisibility>(DEFAULT_MARKERS);

  // Load from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    setMarkers(loadMarkerPrefs());
  }, []);

  const toggleMarker = useCallback((key: MarkerType) => {
    setMarkers((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      saveMarkerPrefs(next);
      return next;
    });
  }, []);

  const events = useMemo(() => computeEvents(metrics, checkpoints), [metrics, checkpoints]);

  const draw = useCallback((idx: number | null = null) => {
    if (canvasRef.current) drawLossChart(canvasRef.current, metrics, idx, events, markers, pinnedStep ?? null);
  }, [metrics, events, markers, pinnedStep]);

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

    const minStep = 0;
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

  const onClick = useCallback((e: React.MouseEvent) => {
    if (!onPinStep || !canvasRef.current || metrics.length < 2) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const w2 = canvasRef.current.clientWidth;
    const padL = 56, padR = 20;
    const cw2 = w2 - padL - padR;
    if (mouseX < padL || mouseX > w2 - padR) return;
    const maxStep = metrics[metrics.length - 1].step;
    const rangeS = maxStep || 1;
    const stepAt = (mouseX - padL) / cw2 * rangeS;
    let lo = 0, hi = metrics.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (metrics[mid].step < stepAt) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(metrics[lo - 1].step - stepAt) < Math.abs(metrics[lo].step - stepAt)) lo--;
    onPinStep(metrics[lo].step);
  }, [metrics, onPinStep]);

  if (metrics.length < 2) {
    return (
      <div className="flex h-72 items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-xs text-text-muted">
        {metrics.length === 0 ? "No metrics data" : "Waiting for more data..."}
      </div>
    );
  }

  const markerKeys: MarkerType[] = ["checkpoints", "bestVal", "warmupEnd", "gradSpikes", "lossSpikes", "overfit"];

  return (
    <div className="relative">
      <div className="mb-2 flex flex-wrap gap-1.5">
        {markerKeys.map((key) => (
          <button
            key={key}
            onClick={() => toggleMarker(key)}
            className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.62rem] font-medium transition-colors ${
              markers[key]
                ? "border-border-2 bg-surface-2 text-text-primary"
                : "border-border/40 bg-transparent text-text-muted opacity-50"
            }`}
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: MARKER_COLORS[key] }}
            />
            {MARKER_LABELS[key]}
          </button>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        className="h-72 w-full cursor-crosshair rounded-lg"
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        onClick={onClick}
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

// ── Pinned Step Drawing Helper ────────────────────────────────────

function drawPinnedStep(
  ctx: CanvasRenderingContext2D,
  pinnedStep: number,
  sx: (step: number) => number,
  pad: { top: number; right: number; bottom: number; left: number },
  ch: number,
  w: number,
) {
  const px = sx(pinnedStep);
  if (px < pad.left || px > w - pad.right) return;
  ctx.save();
  ctx.strokeStyle = "rgba(168, 85, 247, 0.7)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(px, pad.top);
  ctx.lineTo(px, pad.top + ch);
  ctx.stroke();
  ctx.fillStyle = "rgba(168, 85, 247, 0.85)";
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`${pinnedStep.toLocaleString()}`, px, pad.top - 3);
  ctx.restore();
}

// ── Mini Charts ──────────────────────────────────────────────────

function drawMiniChart(
  canvas: HTMLCanvasElement,
  series: MiniSeries[],
  opts: { logScale?: boolean; formatLeft?: (v: number) => string; formatRight?: (v: number) => string },
  hoverStep: number | null = null,
  pinnedStep: number | null = null,
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
  const pad = { top: 12, right: hasRight ? 42 : 16, bottom: 28, left: 48 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const allSteps = series.flatMap((s) => s.data.map((d) => d.step));
  const minStep = 0;
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
  const ticks = [minStep, Math.round(minStep + rangeS * 0.25), Math.round(minStep + rangeS * 0.5), Math.round(minStep + rangeS * 0.75), maxStep];
  for (const s of ticks) ctx.fillText(fmtNum(s), sx(s), h - pad.bottom + 14);

  // Axis label
  ctx.fillStyle = "#333";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("step", pad.left + cw / 2, h - 4);

  // Gradient fill under first left-axis series
  if (leftSeries.length > 0 && leftSeries[0].data.length > 1) {
    const s0 = leftSeries[0];
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, s0.color + "20");
    grad.addColorStop(1, s0.color + "00");
    ctx.beginPath();
    ctx.moveTo(sx(s0.data[0].step), pad.top + ch);
    for (const d of s0.data) {
      const yv = opts.logScale ? Math.max(d.value, 1e-10) : d.value;
      ctx.lineTo(sx(d.step), syLeft(yv));
    }
    ctx.lineTo(sx(s0.data[s0.data.length - 1].step), pad.top + ch);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

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

  // Pinned step marker
  if (pinnedStep != null) drawPinnedStep(ctx, pinnedStep, sx, pad, ch, w);

  // Hover crosshair + dots
  if (hoverStep != null) {
    const hx = sx(hoverStep);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, pad.top);
    ctx.lineTo(hx, pad.top + ch);
    ctx.stroke();
    ctx.setLineDash([]);

    for (const s of series) {
      let nearest = s.data[0];
      let bestDist = Infinity;
      for (const d of s.data) {
        const dist = Math.abs(d.step - hoverStep);
        if (dist < bestDist) { bestDist = dist; nearest = d; }
      }
      if (!nearest || bestDist > rangeS * 0.05) continue;
      const toY = s.axis === "right" ? syRight : syLeft;
      const y = toY(opts.logScale && s.axis !== "right" ? Math.max(nearest.value, 1e-10) : nearest.value);
      ctx.beginPath();
      ctx.arc(hx, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(hx, y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = s.color + "60";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  } else {
    // Last-point glow
    for (const s of series) {
      if (s.data.length === 0) continue;
      const last = s.data[s.data.length - 1];
      const toY = s.axis === "right" ? syRight : syLeft;
      const y = toY(opts.logScale && s.axis !== "right" ? Math.max(last.value, 1e-10) : last.value);
      const lx2 = sx(last.step);
      ctx.beginPath();
      ctx.arc(lx2, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lx2, y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = s.color + "30";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Legend
  let lx = pad.left;
  const ly = h - 6;
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

export function MiniChart({ metrics, title, buildSeries, logScale, formatLeft, formatRight, noDataMsg, pinnedStep, onPinStep }: {
  metrics: ChartMetric[];
  title: string;
  buildSeries: (m: ChartMetric[]) => MiniSeries[];
  logScale?: boolean;
  formatLeft?: (v: number) => string;
  formatRight?: (v: number) => string;
  noDataMsg?: string;
  pinnedStep?: number | null;
  onPinStep?: (step: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<MiniChartTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const series = useMemo(() => buildSeries(metrics), [metrics, buildSeries]);
  const hasData = series.some((s) => s.data.length >= 2);

  const allSteps = useMemo(() => {
    const stepSet = new Set<number>();
    for (const s of series) for (const d of s.data) stepSet.add(d.step);
    return [...stepSet].sort((a, b) => a - b);
  }, [series]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current && hasData) {
      drawMiniChart(canvasRef.current, series, { logScale, formatLeft, formatRight }, step, pinnedStep ?? null);
    }
  }, [series, hasData, logScale, formatLeft, formatRight, pinnedStep]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => draw(hoverRef.current));
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const canvas = canvasRef.current;
    if (!canvas || allSteps.length < 2) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const w = canvas.clientWidth;
    const hasRight = series.some((s) => s.axis === "right");
    const padL = 48, padR = hasRight ? 42 : 16;
    const cw2 = w - padL - padR;
    if (mouseX < padL || mouseX > w - padR) return null;
    const maxStep = allSteps[allSteps.length - 1];
    const rangeS = maxStep || 1;
    const stepAt = (mouseX - padL) / cw2 * rangeS;
    let lo = 0, hi = allSteps.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (allSteps[mid] < stepAt) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(allSteps[lo - 1] - stepAt) < Math.abs(allSteps[lo] - stepAt)) lo--;
    return allSteps[lo];
  }, [allSteps, series]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e);
    if (step == null) {
      hoverRef.current = null;
      setTooltip(null);
      draw();
      return;
    }
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const w = canvas.clientWidth;
    const hasRight = series.some((s) => s.axis === "right");
    const padL = 48, padR = hasRight ? 42 : 16;
    const maxStep = allSteps[allSteps.length - 1];
    const rangeS = maxStep || 1;
    const pointX = padL + (step / rangeS) * (w - padL - padR);

    hoverRef.current = step;
    const values: MiniChartTooltip["values"] = [];
    for (const s of series) {
      let nearest = s.data[0];
      let bestDist = Infinity;
      for (const d of s.data) {
        const dist = Math.abs(d.step - step);
        if (dist < bestDist) { bestDist = dist; nearest = d; }
      }
      if (nearest && bestDist <= rangeS * 0.05) {
        const fmt = s.format ?? ((v: number) => v.toPrecision(4));
        values.push({ label: s.label, value: nearest.value, color: s.color, formatted: fmt(nearest.value) });
      }
    }
    setTooltip({ pointX, mouseY, step, values, containerWidth: w });
    draw(step);
  }, [allSteps, series, draw, findStep]);

  const onLeave = useCallback(() => {
    hoverRef.current = null;
    setTooltip(null);
    draw();
  }, [draw]);

  const onClick = useCallback((e: React.MouseEvent) => {
    const step = findStep(e);
    if (step != null && onPinStep) onPinStep(step);
  }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <div className="mb-1.5 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">{title}</div>
      {hasData ? (
        <canvas
          ref={canvasRef}
          className="h-[220px] w-full cursor-crosshair rounded-lg"
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          onClick={onClick}
        />
      ) : (
        <div className="flex h-[220px] items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-[0.65rem] text-text-muted">
          {noDataMsg ?? "No data"}
        </div>
      )}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 min-w-[140px] rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl"
          style={{
            left: tooltip.pointX < tooltip.containerWidth * 0.65 ? tooltip.pointX + 12 : undefined,
            right: tooltip.pointX >= tooltip.containerWidth * 0.65 ? tooltip.containerWidth - tooltip.pointX + 12 : undefined,
            top: Math.max(24, tooltip.mouseY - 60),
          }}
        >
          <div className="mb-1.5 font-mono text-[0.68rem] font-bold text-white">
            Step {tooltip.step.toLocaleString()}
          </div>
          <div className="space-y-0.5 text-[0.64rem]">
            {tooltip.values.map((v, i) => (
              <div key={i} className="flex justify-between gap-3">
                <span className="text-text-muted">{v.label}</span>
                <span className="font-mono" style={{ color: v.color }}>{v.formatted}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Step Time Chart ──────────────────────────────────────────────

function drawStepTimeChart(canvas: HTMLCanvasElement, metrics: ChartMetric[], hoverStep: number | null = null, pinnedStep: number | null = null) {
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

  const pad = { top: 12, right: 16, bottom: 32, left: 48 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  const minStep = 0;
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
  const stepTicks = [minStep, Math.round(minStep + rangeS * 0.25), Math.round(minStep + rangeS * 0.5), Math.round(minStep + rangeS * 0.75), maxStep];
  for (const s of stepTicks) ctx.fillText(fmtNum(s), sx(s), h - pad.bottom + 14);

  ctx.fillStyle = "#333";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("step", pad.left + cw / 2, h - 4);

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

  // Pinned step marker
  if (pinnedStep != null) drawPinnedStep(ctx, pinnedStep, sx, pad, ch, w);

  // Hover crosshair
  if (hoverStep != null) {
    const hx = sx(hoverStep);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, pad.top);
    ctx.lineTo(hx, pad.top + ch);
    ctx.stroke();
    ctx.setLineDash([]);
    let nearest = stacked[0];
    let bestDist = Infinity;
    for (const s of stacked) {
      const dist = Math.abs(s.step - hoverStep);
      if (dist < bestDist) { bestDist = dist; nearest = s; }
    }
    if (nearest) {
      let cumVal = 0;
      for (let pi = 0; pi < TIMING_PHASES.length; pi++) {
        cumVal += nearest.phases[pi];
        if (nearest.phases[pi] > 0) {
          ctx.beginPath();
          ctx.arc(hx, sy(cumVal), 3, 0, Math.PI * 2);
          ctx.fillStyle = TIMING_PHASES[pi].color;
          ctx.fill();
        }
      }
    }
  }

  // Legend
  ctx.font = "9px monospace";
  let lx = pad.left;
  const ly = h - 6;
  for (const p of TIMING_PHASES) {
    ctx.fillStyle = p.color;
    ctx.fillRect(lx, ly - 4, 8, 4);
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText(p.label, lx + 10, ly);
    lx += ctx.measureText(p.label).width + 20;
    if (lx > w - 30) break;
  }
}

export function StepTimeChart({ metrics, pinnedStep, onPinStep }: { metrics: ChartMetric[]; pinnedStep?: number | null; onPinStep?: (step: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<StepTimeTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const hasData = metrics.some((m) => m.timing_fwd_ms != null);
  const data = useMemo(() => metrics.filter((m) => m.timing_fwd_ms != null), [metrics]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current && hasData) drawStepTimeChart(canvasRef.current, metrics, step, pinnedStep ?? null);
  }, [metrics, hasData, pinnedStep]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => draw(hoverRef.current));
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return null;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const w = canvas.clientWidth;
    const padL = 48, padR = 16;
    const cw2 = w - padL - padR;
    if (mouseX < padL || mouseX > w - padR) return null;
    const maxStep = data[data.length - 1].step;
    const rangeS = maxStep || 1;
    const stepAt = (mouseX - padL) / cw2 * rangeS;
    let lo = 0, hi = data.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (data[mid].step < stepAt) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(data[lo - 1].step - stepAt) < Math.abs(data[lo].step - stepAt)) lo--;
    return data[lo].step;
  }, [data]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e);
    if (step == null) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const w = canvas.clientWidth;
    const padL = 48, padR = 16;
    const maxStep = data[data.length - 1].step;
    const rangeS = maxStep || 1;
    const pointX = padL + (step / rangeS) * (w - padL - padR);

    hoverRef.current = step;
    const m = data.find(d => d.step === step) ?? data[0];
    const phases = TIMING_PHASES.map((p) => ({ label: p.label, value: ((m as any)[p.key] as number) ?? 0, color: p.color }));
    const total = phases.reduce((s, p) => s + p.value, 0);
    setTooltip({ pointX, mouseY, step, phases, total, containerWidth: w });
    draw(step);
  }, [data, draw, findStep]);

  const onLeave = useCallback(() => { hoverRef.current = null; setTooltip(null); draw(); }, [draw]);

  const onClick = useCallback((e: React.MouseEvent) => {
    const step = findStep(e);
    if (step != null && onPinStep) onPinStep(step);
  }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <div className="mb-1.5 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">Step Time Breakdown</div>
      {hasData ? (
        <canvas ref={canvasRef} className="h-[220px] w-full cursor-crosshair rounded-lg" onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />
      ) : (
        <div className="flex h-[220px] items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-[0.65rem] text-text-muted">
          No timing data
        </div>
      )}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 min-w-[160px] rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl"
          style={{
            left: tooltip.pointX < tooltip.containerWidth * 0.65 ? tooltip.pointX + 12 : undefined,
            right: tooltip.pointX >= tooltip.containerWidth * 0.65 ? tooltip.containerWidth - tooltip.pointX + 12 : undefined,
            top: Math.max(24, tooltip.mouseY - 80),
          }}
        >
          <div className="mb-1.5 font-mono text-[0.68rem] font-bold text-white">
            Step {tooltip.step.toLocaleString()}
          </div>
          <div className="space-y-0.5 text-[0.64rem]">
            {tooltip.phases.filter(p => p.value > 0).map((p, i) => (
              <div key={i} className="flex justify-between gap-3">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: p.color }} />
                  <span className="text-text-muted">{p.label}</span>
                </span>
                <span className="font-mono text-text-secondary">{p.value.toFixed(0)}ms</span>
              </div>
            ))}
            <div className="mt-1 border-t border-border/50 pt-1">
              <div className="flex justify-between gap-3 font-semibold">
                <span className="text-text-muted">Total</span>
                <span className="font-mono text-white">{tooltip.total.toFixed(0)}ms</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Build Series Helpers ─────────────────────────────────────────

export const buildGpuSeries = (m: ChartMetric[]): MiniSeries[] => {
  const vram = m.filter((x) => x.gpu_vram_used_mb != null).map((x) => ({ step: x.step, value: x.gpu_vram_used_mb! }));
  const util = m.filter((x) => x.gpu_util_pct != null).map((x) => ({ step: x.step, value: x.gpu_util_pct! }));
  return [
    { data: vram, color: "#10b981", label: "VRAM", axis: "left" as const, format: (v: number) => v >= 1024 ? (v / 1024).toFixed(1) + " GB" : v.toFixed(0) + " MB" },
    { data: util, color: "#f59e0b", label: "GPU%", axis: "right" as const, format: (v: number) => v.toFixed(0) + "%" },
  ];
};

export const buildLrSeries = (m: ChartMetric[]): MiniSeries[] => {
  const lr = m.filter((x) => x.lr > 0).map((x) => ({ step: x.step, value: x.lr }));
  return [{ data: lr, color: "#22d3ee", label: "LR", format: (v: number) => v.toExponential(2) }];
};

export const buildGradNormSeries = (m: ChartMetric[]): MiniSeries[] => {
  const gn = m.filter((x) => x.grad_norm > 0 && isFinite(x.grad_norm)).map((x) => ({ step: x.step, value: x.grad_norm }));
  return [{ data: gn, color: "#f97316", label: "norm", format: (v: number) => v.toFixed(4) }];
};
