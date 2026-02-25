"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ChartMetric, Stat, fmtNum } from "@/components/charts";

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

interface ChartTooltip {
  pointX: number;
  mouseY: number;
  step: number;
  lines: { label: string; value: string; color?: string }[];
  containerWidth: number;
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

function drawPinnedMarker(ctx: CanvasRenderingContext2D, pinnedStep: number, toX: (s: number) => number, padL: number, padR: number, padT: number, plotH: number, w: number) {
  const px = toX(pinnedStep);
  if (px < padL || px > w - padR) return;
  ctx.save();
  ctx.strokeStyle = "rgba(168, 85, 247, 0.7)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px, padT);
  ctx.lineTo(px, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = "rgba(168, 85, 247, 0.85)";
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`${pinnedStep.toLocaleString()}`, px, padT - 3);
  ctx.restore();
}

function drawHoverCrosshair(ctx: CanvasRenderingContext2D, hx: number, padT: number, plotH: number) {
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(hx, padT);
  ctx.lineTo(hx, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
}

function ChartTooltipDiv({ tooltip }: { tooltip: ChartTooltip }) {
  return (
    <div
      className="pointer-events-none absolute z-20 min-w-[150px] rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl"
      style={{
        left: tooltip.pointX < tooltip.containerWidth * 0.65 ? tooltip.pointX + 12 : undefined,
        right: tooltip.pointX >= tooltip.containerWidth * 0.65 ? tooltip.containerWidth - tooltip.pointX + 12 : undefined,
        top: Math.max(24, tooltip.mouseY - 70),
      }}
    >
      <div className="mb-1.5 font-mono text-[0.68rem] font-bold text-white">
        Step {tooltip.step.toLocaleString()}
      </div>
      <div className="space-y-0.5 text-[0.64rem]">
        {tooltip.lines.map((l, i) => (
          <div key={i} className="flex justify-between gap-3">
            <span className="text-text-muted">{l.label}</span>
            <span className="font-mono" style={{ color: l.color ?? "#ccc" }}>{l.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Hover helpers ─────────────────────────────────────────────

function binarySearchStep(data: { step: number }[], stepAt: number): number {
  let lo = 0, hi = data.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (data[mid].step < stepAt) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(data[lo - 1].step - stepAt) < Math.abs(data[lo].step - stepAt)) lo--;
  return lo;
}

// ── CUSUM Chart ──────────────────────────────────────────────

const CUSUM_SERIES = [
  { key: "cusum_grad" as const, color: "#f59e0b", label: "gradNorm" },
  { key: "cusum_clip" as const, color: "#60a5fa", label: "clipPct" },
  { key: "cusum_tps" as const, color: "#34d399", label: "tokPerSec" },
  { key: "cusum_val" as const, color: "#f87171", label: "valLoss" },
];

function drawCusum(canvas: HTMLCanvasElement, metrics: SymbioMetric[], sensitivity: number, hoverStep: number | null, pinnedStep: number | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { l: 50, r: 20, t: 20, b: 30 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, pad, w, h, 4);

  const cusumData = metrics.filter(m => m.cusum_grad != null || m.cusum_clip != null || m.cusum_tps != null || m.cusum_val != null);
  if (cusumData.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("No CUSUM data (baseline accumulating...)", w / 2, h / 2);
    return;
  }

  let maxVal = sensitivity * 1.2;
  for (const m of cusumData) maxVal = Math.max(maxVal, m.cusum_grad ?? 0, m.cusum_clip ?? 0, m.cusum_tps ?? 0, m.cusum_val ?? 0);

  const minStep = cusumData[0].step;
  const maxStep = cusumData[cusumData.length - 1].step;
  const stepRange = maxStep - minStep || 1;
  const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
  const toY = (val: number) => pad.t + plotH - (val / maxVal) * plotH;

  // Threshold
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

  // Lines
  for (const s of CUSUM_SERIES) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (const m of cusumData) {
      const val = m[s.key];
      if (val == null) continue;
      if (!started) { ctx.moveTo(toX(m.step), toY(val)); started = true; }
      else ctx.lineTo(toX(m.step), toY(val));
    }
    ctx.stroke();
  }

  // Alert markers
  for (const m of cusumData) {
    if ((m.cusum_alerts ?? 0) > 0) {
      ctx.fillStyle = "rgba(248,113,113,0.3)";
      ctx.fillRect(toX(m.step) - 1, pad.t, 2, plotH);
    }
  }

  // Pinned step
  if (pinnedStep != null) drawPinnedMarker(ctx, pinnedStep, toX, pad.l, pad.r, pad.t, plotH, w);

  // Hover
  if (hoverStep != null) {
    const hx = toX(hoverStep);
    drawHoverCrosshair(ctx, hx, pad.t, plotH);
    for (const s of CUSUM_SERIES) {
      let nearest: SymbioMetric | null = null;
      let bestDist = Infinity;
      for (const m of cusumData) {
        if (m[s.key] == null) continue;
        const d = Math.abs(m.step - hoverStep);
        if (d < bestDist) { bestDist = d; nearest = m; }
      }
      if (nearest && bestDist <= stepRange * 0.05) {
        const val = nearest[s.key]!;
        ctx.beginPath();
        ctx.arc(hx, toY(val), 4, 0, Math.PI * 2);
        ctx.fillStyle = s.color;
        ctx.fill();
      }
    }
  }

  // Legend
  let legendY = pad.t + 12;
  ctx.font = "10px monospace";
  for (const s of CUSUM_SERIES) {
    ctx.fillStyle = s.color;
    ctx.fillRect(pad.l + 10, legendY - 6, 10, 3);
    ctx.fillStyle = "#aaa";
    ctx.textAlign = "left";
    ctx.fillText(s.label, pad.l + 24, legendY);
    legendY += 13;
  }

  // Axes
  ctx.fillStyle = "#888";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.fillText(fmtNum(minStep), pad.l, h - 8);
  ctx.fillText(fmtNum(maxStep), w - pad.r, h - 8);
  ctx.textAlign = "right";
  ctx.fillText("0", pad.l - 4, h - pad.b);
  ctx.fillText(maxVal.toFixed(1), pad.l - 4, pad.t + 10);
}

export function CusumChart({ metrics, sensitivity = 4.0, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; sensitivity?: number; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const cusumData = useMemo(() => metrics.filter(m => m.cusum_grad != null || m.cusum_clip != null || m.cusum_tps != null || m.cusum_val != null), [metrics]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current) drawCusum(canvasRef.current, metrics, sensitivity, step, pinnedStep ?? null);
  }, [metrics, sensitivity, pinnedStep]);

  useEffect(() => { draw(); const c = canvasRef.current; if (!c) return; const o = new ResizeObserver(() => draw(hoverRef.current)); o.observe(c); return () => o.disconnect(); }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const c = canvasRef.current; if (!c || cusumData.length < 2) return null;
    const rect = c.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
    const padL = 50, padR = 20, cw = c.clientWidth - padL - padR;
    if (mouseX < padL || mouseX > c.clientWidth - padR) return null;
    const minS = cusumData[0].step, maxS = cusumData[cusumData.length - 1].step, rng = maxS - minS || 1;
    const stepAt = minS + ((mouseX - padL) / cw) * rng;
    const idx = binarySearchStep(cusumData, stepAt);
    return cusumData[idx].step;
  }, [cusumData]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e); if (step == null) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const c = canvasRef.current!; const rect = c.getBoundingClientRect();
    const w = c.clientWidth; const padL = 50, padR = 20;
    const minS = cusumData[0].step, maxS = cusumData[cusumData.length - 1].step, rng = maxS - minS || 1;
    const pointX = padL + ((step - minS) / rng) * (w - padL - padR);
    hoverRef.current = step;
    const m = cusumData.find(d => d.step === step);
    const lines: ChartTooltip["lines"] = [];
    if (m) for (const s of CUSUM_SERIES) { const v = m[s.key]; if (v != null) lines.push({ label: s.label, value: v.toFixed(2), color: s.color }); }
    if (m && (m.cusum_alerts ?? 0) > 0) lines.push({ label: "Alert", value: m.cusum_alert_reason ?? "yes", color: "#f87171" });
    setTooltip({ pointX, mouseY: e.clientY - rect.top, step, lines, containerWidth: w });
    draw(step);
  }, [cusumData, draw, findStep]);

  const onLeave = useCallback(() => { hoverRef.current = null; setTooltip(null); draw(); }, [draw]);
  const onClick = useCallback((e: React.MouseEvent) => { const s = findStep(e); if (s != null && onPinStep) onPinStep(s); }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="h-[220px] w-full cursor-crosshair rounded-lg" onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />
      {tooltip && <ChartTooltipDiv tooltip={tooltip} />}
    </div>
  );
}

// ── Clip Telemetry Chart ─────────────────────────────────────

function drawClip(canvas: HTMLCanvasElement, metrics: SymbioMetric[], hoverStep: number | null, pinnedStep: number | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const clipData = metrics.filter(m => m.clip_coef != null);
  if (clipData.length === 0) return;

  const pad = { l: 50, r: 50, t: 15, b: 28 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, { l: pad.l, r: pad.r, t: pad.t, b: pad.b }, w, h, 4);

  const minStep = clipData[0].step;
  const maxStep = clipData[clipData.length - 1].step;
  const stepRange = maxStep - minStep || 1;
  const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;

  // Gradient fill for clip_coef
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
  grad.addColorStop(0, "rgba(245, 158, 11, 0.15)");
  grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
  ctx.beginPath();
  ctx.moveTo(toX(clipData[0].step), pad.t + plotH);
  for (const m of clipData) ctx.lineTo(toX(m.step), pad.t + plotH - ((m.clip_coef ?? 1) * plotH));
  ctx.lineTo(toX(clipData[clipData.length - 1].step), pad.t + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // clip_coef line
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (const m of clipData) {
    const x = toX(m.step), y = pad.t + plotH - ((m.clip_coef ?? 1) * plotH);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // clip_pct line
  ctx.strokeStyle = "#60a5fa";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  started = false;
  for (const m of clipData) {
    const x = toX(m.step), y = pad.t + plotH - ((m.clip_pct ?? 0) / 100 * plotH);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Pinned
  if (pinnedStep != null) drawPinnedMarker(ctx, pinnedStep, toX, pad.l, pad.r, pad.t, plotH, w);

  // Hover
  if (hoverStep != null) {
    const hx = toX(hoverStep);
    drawHoverCrosshair(ctx, hx, pad.t, plotH);
    let nearest = clipData[0]; let bestDist = Infinity;
    for (const m of clipData) { const d = Math.abs(m.step - hoverStep); if (d < bestDist) { bestDist = d; nearest = m; } }
    if (bestDist <= stepRange * 0.05) {
      const yCoef = pad.t + plotH - ((nearest.clip_coef ?? 1) * plotH);
      ctx.beginPath(); ctx.arc(hx, yCoef, 4, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
      const yPct = pad.t + plotH - ((nearest.clip_pct ?? 0) / 100 * plotH);
      ctx.beginPath(); ctx.arc(hx, yPct, 4, 0, Math.PI * 2); ctx.fillStyle = "#60a5fa"; ctx.fill();
    }
  }

  // Legend + axes
  ctx.font = "10px monospace";
  ctx.fillStyle = "#f59e0b"; ctx.fillRect(pad.l + 10, pad.t + 6, 10, 3);
  ctx.fillStyle = "#aaa"; ctx.textAlign = "left"; ctx.fillText("clip_coef", pad.l + 24, pad.t + 12);
  ctx.fillStyle = "#60a5fa"; ctx.fillRect(pad.l + 100, pad.t + 6, 10, 3);
  ctx.fillStyle = "#aaa"; ctx.fillText("clip_pct", pad.l + 114, pad.t + 12);
  ctx.fillStyle = "#888"; ctx.font = "9px monospace";
  ctx.textAlign = "right"; ctx.fillText("0", pad.l - 4, h - pad.b); ctx.fillText("1.0", pad.l - 4, pad.t + 10);
  ctx.textAlign = "left"; ctx.fillText("0%", w - pad.r + 4, h - pad.b); ctx.fillText("100%", w - pad.r + 4, pad.t + 10);
  ctx.textAlign = "center"; ctx.fillText(fmtNum(minStep), pad.l, h - 6); ctx.fillText(fmtNum(maxStep), w - pad.r, h - 6);
}

export function ClipChart({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const clipData = useMemo(() => metrics.filter(m => m.clip_coef != null), [metrics]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current) drawClip(canvasRef.current, metrics, step, pinnedStep ?? null);
  }, [metrics, pinnedStep]);

  useEffect(() => { draw(); const c = canvasRef.current; if (!c) return; const o = new ResizeObserver(() => draw(hoverRef.current)); o.observe(c); return () => o.disconnect(); }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const c = canvasRef.current; if (!c || clipData.length < 2) return null;
    const rect = c.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
    const padL = 50, padR = 50, cw = c.clientWidth - padL - padR;
    if (mouseX < padL || mouseX > c.clientWidth - padR) return null;
    const minS = clipData[0].step, maxS = clipData[clipData.length - 1].step, rng = maxS - minS || 1;
    const stepAt = minS + ((mouseX - padL) / cw) * rng;
    return clipData[binarySearchStep(clipData, stepAt)].step;
  }, [clipData]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e); if (step == null) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const c = canvasRef.current!; const rect = c.getBoundingClientRect();
    const w = c.clientWidth; const padL = 50, padR = 50;
    const minS = clipData[0].step, maxS = clipData[clipData.length - 1].step, rng = maxS - minS || 1;
    const pointX = padL + ((step - minS) / rng) * (w - padL - padR);
    hoverRef.current = step;
    const m = clipData.find(d => d.step === step);
    const lines: ChartTooltip["lines"] = [];
    if (m) {
      lines.push({ label: "Clip Coef", value: (m.clip_coef ?? 1).toFixed(4), color: "#f59e0b" });
      lines.push({ label: "Clip %", value: (m.clip_pct ?? 0).toFixed(1) + "%", color: "#60a5fa" });
    }
    setTooltip({ pointX, mouseY: e.clientY - rect.top, step, lines, containerWidth: w });
    draw(step);
  }, [clipData, draw, findStep]);

  const onLeave = useCallback(() => { hoverRef.current = null; setTooltip(null); draw(); }, [draw]);
  const onClick = useCallback((e: React.MouseEvent) => { const s = findStep(e); if (s != null && onPinStep) onPinStep(s); }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="h-[220px] w-full cursor-crosshair rounded-lg" onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />
      {tooltip && <ChartTooltipDiv tooltip={tooltip} />}
    </div>
  );
}

// ── Symbio Metrics Mini Charts ───────────────────────────────

function drawSparseLine(canvas: HTMLCanvasElement, points: { step: number; val: number }[], color: string, label: string, format: (v: number) => string, hoverStep: number | null, pinnedStep: number | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { l: 50, r: 14, t: 18, b: 24 };
  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  if (points.length < 2) {
    ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "center";
    ctx.fillText(`No ${label} data`, w / 2, h / 2);
    return;
  }

  drawGrid(ctx, pad, w, h, 3);
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const minStep = points[0].step, maxStep = points[points.length - 1].step, stepRange = maxStep - minStep || 1;
  let minVal = Infinity, maxVal = -Infinity;
  for (const p of points) { if (p.val < minVal) minVal = p.val; if (p.val > maxVal) maxVal = p.val; }
  const valRange = maxVal - minVal || 1;

  const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
  const toY = (val: number) => pad.t + plotH - ((val - minVal) / valRange) * plotH;

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
  grad.addColorStop(0, color + "20");
  grad.addColorStop(1, color + "00");
  ctx.beginPath();
  ctx.moveTo(toX(points[0].step), pad.t + plotH);
  for (const p of points) ctx.lineTo(toX(p.step), toY(p.val));
  ctx.lineTo(toX(points[points.length - 1].step), pad.t + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.shadowColor = color + "4d";
  ctx.shadowBlur = 4;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(toX(points[0].step), toY(points[0].val));
  for (let i = 1; i < points.length; i++) ctx.lineTo(toX(points[i].step), toY(points[i].val));
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Pinned
  if (pinnedStep != null) drawPinnedMarker(ctx, pinnedStep, toX, pad.l, pad.r, pad.t, plotH, w);

  // Hover
  if (hoverStep != null) {
    const hx = toX(hoverStep);
    drawHoverCrosshair(ctx, hx, pad.t, plotH);
    let nearest = points[0]; let bestDist = Infinity;
    for (const p of points) { const d = Math.abs(p.step - hoverStep); if (d < bestDist) { bestDist = d; nearest = p; } }
    if (bestDist <= stepRange * 0.05) {
      ctx.beginPath(); ctx.arc(hx, toY(nearest.val), 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      ctx.beginPath(); ctx.arc(hx, toY(nearest.val), 7, 0, Math.PI * 2); ctx.strokeStyle = color + "60"; ctx.lineWidth = 2; ctx.stroke();
    }
  } else {
    // Last-point glow
    const last = points[points.length - 1];
    ctx.beginPath(); ctx.arc(toX(last.step), toY(last.val), 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  }

  // Label + value
  ctx.fillStyle = "#aaa"; ctx.font = "10px monospace"; ctx.textAlign = "left"; ctx.fillText(label, pad.l + 4, pad.t - 4);
  ctx.fillStyle = color; ctx.textAlign = "right"; ctx.fillText(format(points[points.length - 1].val), w - pad.r - 4, pad.t - 4);
  // Y-axis
  ctx.fillStyle = "#888"; ctx.font = "9px monospace"; ctx.textAlign = "right";
  ctx.fillText(format(minVal), pad.l - 4, h - pad.b); ctx.fillText(format(maxVal), pad.l - 4, pad.t + 10);
  // X-axis
  ctx.textAlign = "center"; ctx.fillText(fmtNum(minStep), pad.l, h - 6); ctx.fillText(fmtNum(maxStep), w - pad.r, h - 6);
}

function SparseLineChart({ metrics, getY, color = "#a78bfa", label, format = (v: number) => v.toFixed(2), pinnedStep, onPinStep }: {
  metrics: SymbioMetric[]; getY: (m: SymbioMetric) => number | null | undefined; color?: string; label: string; format?: (v: number) => string;
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const points = useMemo(() => metrics.map(m => ({ step: m.step, val: getY(m) })).filter((p): p is { step: number; val: number } => p.val != null), [metrics, getY]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current) drawSparseLine(canvasRef.current, points, color, label, format, step, pinnedStep ?? null);
  }, [points, color, label, format, pinnedStep]);

  useEffect(() => { draw(); const c = canvasRef.current; if (!c) return; const o = new ResizeObserver(() => draw(hoverRef.current)); o.observe(c); return () => o.disconnect(); }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const c = canvasRef.current; if (!c || points.length < 2) return null;
    const rect = c.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
    const padL = 50, padR = 14, cw = c.clientWidth - padL - padR;
    if (mouseX < padL || mouseX > c.clientWidth - padR) return null;
    const minS = points[0].step, maxS = points[points.length - 1].step, rng = maxS - minS || 1;
    const stepAt = minS + ((mouseX - padL) / cw) * rng;
    return points[binarySearchStep(points, stepAt)].step;
  }, [points]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e); if (step == null) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const c = canvasRef.current!; const rect = c.getBoundingClientRect();
    const w2 = c.clientWidth; const padL = 50, padR = 14;
    const minS = points[0].step, maxS = points[points.length - 1].step, rng = maxS - minS || 1;
    const pointX = padL + ((step - minS) / rng) * (w2 - padL - padR);
    hoverRef.current = step;
    const pt = points.find(d => d.step === step);
    const lines: ChartTooltip["lines"] = [];
    if (pt) lines.push({ label, value: format(pt.val), color });
    setTooltip({ pointX, mouseY: e.clientY - rect.top, step, lines, containerWidth: w2 });
    draw(step);
  }, [points, draw, findStep, color, label, format]);

  const onLeave = useCallback(() => { hoverRef.current = null; setTooltip(null); draw(); }, [draw]);
  const onClick = useCallback((e: React.MouseEvent) => { const s = findStep(e); if (s != null && onPinStep) onPinStep(s); }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="h-[200px] w-full cursor-crosshair rounded-lg" onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />
      {tooltip && <ChartTooltipDiv tooltip={tooltip} />}
    </div>
  );
}

// ── Adaptive Batch Step Chart ────────────────────────────────

function drawAdaptiveBatch(canvas: HTMLCanvasElement, metrics: SymbioMetric[], hoverStep: number | null, pinnedStep: number | null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const batchData = metrics.filter(m => m.adaptive_batch_size != null);
  if (batchData.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { l: 48, r: 14, t: 18, b: 24 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);
  drawGrid(ctx, pad, w, h, 3);

  const minStep = batchData[0].step, maxStep = batchData[batchData.length - 1].step, stepRange = maxStep - minStep || 1;
  let minBatch = Infinity, maxBatch = -Infinity;
  for (const m of batchData) { const b = m.adaptive_batch_size!; if (b < minBatch) minBatch = b; if (b > maxBatch) maxBatch = b; }
  const batchRange = maxBatch - minBatch || 1;

  const toX = (step: number) => pad.l + ((step - minStep) / stepRange) * plotW;
  const toY = (batch: number) => pad.t + plotH - ((batch - minBatch) / batchRange) * plotH;

  // Step chart fill
  ctx.fillStyle = "rgba(52, 211, 153, 0.1)";
  ctx.beginPath();
  ctx.moveTo(toX(batchData[0].step), pad.t + plotH);
  for (let i = 0; i < batchData.length; i++) {
    const x = toX(batchData[i].step), y = toY(batchData[i].adaptive_batch_size!);
    ctx.lineTo(x, y);
    if (i < batchData.length - 1) ctx.lineTo(toX(batchData[i + 1].step), y);
  }
  ctx.lineTo(toX(batchData[batchData.length - 1].step), pad.t + plotH);
  ctx.closePath();
  ctx.fill();

  // Step chart line
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < batchData.length; i++) {
    const x = toX(batchData[i].step), y = toY(batchData[i].adaptive_batch_size!);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    if (i < batchData.length - 1) ctx.lineTo(toX(batchData[i + 1].step), y);
  }
  ctx.stroke();

  // Change annotations
  for (const m of batchData) {
    if (m.batch_change_reason) {
      ctx.fillStyle = "rgba(248,113,113,0.2)";
      ctx.fillRect(toX(m.step) - 1, pad.t, 2, plotH);
    }
  }

  // Pinned
  if (pinnedStep != null) drawPinnedMarker(ctx, pinnedStep, toX, pad.l, pad.r, pad.t, plotH, w);

  // Hover
  if (hoverStep != null) {
    const hx = toX(hoverStep);
    drawHoverCrosshair(ctx, hx, pad.t, plotH);
    let nearest = batchData[0]; let bestDist = Infinity;
    for (const m of batchData) { const d = Math.abs(m.step - hoverStep); if (d < bestDist) { bestDist = d; nearest = m; } }
    if (bestDist <= stepRange * 0.05) {
      const y = toY(nearest.adaptive_batch_size!);
      ctx.beginPath(); ctx.arc(hx, y, 4, 0, Math.PI * 2); ctx.fillStyle = "#34d399"; ctx.fill();
    }
  }

  // Labels
  ctx.fillStyle = "#aaa"; ctx.font = "10px monospace"; ctx.textAlign = "left"; ctx.fillText("Adaptive Batch", pad.l + 4, pad.t - 4);
  ctx.fillStyle = "#888"; ctx.font = "9px monospace"; ctx.textAlign = "right";
  ctx.fillText(String(minBatch), pad.l - 4, h - pad.b); ctx.fillText(String(maxBatch), pad.l - 4, pad.t + 10);
  ctx.textAlign = "center"; ctx.fillText(fmtNum(minStep), pad.l, h - 6); ctx.fillText(fmtNum(maxStep), w - pad.r, h - 6);
}

export function AdaptiveBatchChart({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const batchData = useMemo(() => metrics.filter(m => m.adaptive_batch_size != null), [metrics]);

  const draw = useCallback((step: number | null = null) => {
    if (canvasRef.current) drawAdaptiveBatch(canvasRef.current, metrics, step, pinnedStep ?? null);
  }, [metrics, pinnedStep]);

  useEffect(() => { draw(); const c = canvasRef.current; if (!c) return; const o = new ResizeObserver(() => draw(hoverRef.current)); o.observe(c); return () => o.disconnect(); }, [draw]);

  const findStep = useCallback((e: React.MouseEvent): number | null => {
    const c = canvasRef.current; if (!c || batchData.length < 2) return null;
    const rect = c.getBoundingClientRect(); const mouseX = e.clientX - rect.left;
    const padL = 48, padR = 14, cw = c.clientWidth - padL - padR;
    if (mouseX < padL || mouseX > c.clientWidth - padR) return null;
    const minS = batchData[0].step, maxS = batchData[batchData.length - 1].step, rng = maxS - minS || 1;
    const stepAt = minS + ((mouseX - padL) / cw) * rng;
    return batchData[binarySearchStep(batchData, stepAt)].step;
  }, [batchData]);

  const onMove = useCallback((e: React.MouseEvent) => {
    const step = findStep(e); if (step == null) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const c = canvasRef.current!; const rect = c.getBoundingClientRect();
    const w2 = c.clientWidth; const padL = 48, padR = 14;
    const minS = batchData[0].step, maxS = batchData[batchData.length - 1].step, rng = maxS - minS || 1;
    const pointX = padL + ((step - minS) / rng) * (w2 - padL - padR);
    hoverRef.current = step;
    const m = batchData.find(d => d.step === step);
    const lines: ChartTooltip["lines"] = [];
    if (m) {
      lines.push({ label: "Batch Size", value: String(m.adaptive_batch_size), color: "#34d399" });
      if (m.batch_change_reason) lines.push({ label: "Reason", value: m.batch_change_reason, color: "#f87171" });
    }
    setTooltip({ pointX, mouseY: e.clientY - rect.top, step, lines, containerWidth: w2 });
    draw(step);
  }, [batchData, draw, findStep]);

  const onLeave = useCallback(() => { hoverRef.current = null; setTooltip(null); draw(); }, [draw]);
  const onClick = useCallback((e: React.MouseEvent) => { const s = findStep(e); if (s != null && onPinStep) onPinStep(s); }, [findStep, onPinStep]);

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="h-[200px] w-full cursor-crosshair rounded-lg" onMouseMove={onMove} onMouseLeave={onLeave} onClick={onClick} />
      {tooltip && <ChartTooltipDiv tooltip={tooltip} />}
    </div>
  );
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

  const sorted = [...candidates].sort((a, b) => {
    if (a.bestValLoss !== Infinity && b.bestValLoss !== Infinity) return a.bestValLoss - b.bestValLoss;
    if (a.bestValLoss !== Infinity) return -1;
    if (b.bestValLoss !== Infinity) return 1;
    return a.bestLoss - b.bestLoss;
  });

  const bestId = sorted[0]?.id;

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

// ── Activation Switch Log ─────────────────────────────────────

interface SwitchEvent {
  step: number;
  fromId: string | null;
  fromActivation: string | null;
  fromGeneration: number | null;
  toId: string;
  toActivation: string;
  toGeneration: number;
  fromSteps: number;
  fromBestLoss: number | null;
  fromFinalLoss: number | null;
  fromBestFitness: number | null;
  fromAvgTokPerSec: number | null;
  fromCusumAlerts: number;
  fromLastAlertReason: string | null;
  fromGradSpikes: number;
  fromClipPctAtEnd: number | null;
  lossAtSwitch: number;
}

function extractSwitchEvents(metrics: SymbioMetric[]): SwitchEvent[] {
  const events: SwitchEvent[] = [];
  let prevId: string | null = null;
  let prevActivation: string | null = null;
  let prevGeneration: number | null = null;
  let candSteps = 0;
  let candLosses: number[] = [];
  let candFitnesses: number[] = [];
  let candTps: number[] = [];
  let candCusumAlerts = 0;
  let candLastAlertReason: string | null = null;
  let candGradSpikes = 0;
  let candLastClipPct: number | null = null;

  for (const m of metrics) {
    const id = m.symbio_candidate_id;
    if (!id) continue;
    if (id !== prevId) {
      events.push({
        step: m.step, fromId: prevId, fromActivation: prevActivation, fromGeneration: prevGeneration,
        toId: id, toActivation: m.symbio_candidate_activation ?? "?", toGeneration: m.symbio_generation ?? 0,
        fromSteps: candSteps, fromBestLoss: candLosses.length > 0 ? Math.min(...candLosses) : null,
        fromFinalLoss: candLosses.length > 0 ? candLosses[candLosses.length - 1] : null,
        fromBestFitness: candFitnesses.length > 0 ? Math.max(...candFitnesses) : null,
        fromAvgTokPerSec: candTps.length > 0 ? candTps.reduce((a, b) => a + b, 0) / candTps.length : null,
        fromCusumAlerts: candCusumAlerts, fromLastAlertReason: candLastAlertReason,
        fromGradSpikes: candGradSpikes, fromClipPctAtEnd: candLastClipPct, lossAtSwitch: m.loss,
      });
      prevId = id; prevActivation = m.symbio_candidate_activation ?? "?"; prevGeneration = m.symbio_generation ?? 0;
      candSteps = 0; candLosses = []; candFitnesses = []; candTps = [];
      candCusumAlerts = 0; candLastAlertReason = null; candGradSpikes = 0; candLastClipPct = null;
    }
    candSteps++;
    candLosses.push(m.loss);
    if (m.fitness_score != null) candFitnesses.push(m.fitness_score);
    if (m.tokens_per_sec != null) candTps.push(m.tokens_per_sec);
    if ((m.cusum_alerts ?? 0) > 0) { candCusumAlerts++; if (m.cusum_alert_reason) candLastAlertReason = m.cusum_alert_reason; }
    if (m.grad_norm > 10) candGradSpikes++;
    if (m.clip_pct != null) candLastClipPct = m.clip_pct;
  }
  return events;
}

const ACTIVATION_COLORS: Record<string, string> = {
  gelu: "text-blue-400", silu: "text-green-400", relu: "text-yellow-400", swiglu: "text-purple-400",
};

const ACTIVATION_BG: Record<string, string> = {
  gelu: "bg-blue-500/10 border-blue-500/20", silu: "bg-green-500/10 border-green-500/20",
  relu: "bg-yellow-500/10 border-yellow-500/20", swiglu: "bg-purple-500/10 border-purple-500/20",
};

function ActivationSwitchLog({ metrics }: { metrics: SymbioMetric[] }) {
  const events = extractSwitchEvents(metrics);
  if (events.length === 0) return null;
  return (
    <div className="mb-4 rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-3">
        <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">Activation Switch Log ({events.length} transitions)</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border/50 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
              <th className="px-3 py-2 text-left">Step</th><th className="px-3 py-2 text-left">From</th><th className="px-3 py-2 text-center"></th>
              <th className="px-3 py-2 text-left">To</th><th className="px-3 py-2 text-left">Gen</th><th className="px-3 py-2 text-right">Prev Steps</th>
              <th className="px-3 py-2 text-right">Best Loss</th><th className="px-3 py-2 text-right">Final Loss</th><th className="px-3 py-2 text-right">Fitness</th>
              <th className="px-3 py-2 text-right">Avg tok/s</th><th className="px-3 py-2 text-right">Clip %</th><th className="px-3 py-2 text-left">Alerts</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} className={`border-b border-border/20 last:border-0 ${i === 0 ? "bg-surface-2/30" : ""}`}>
                <td className="px-3 py-2 font-mono font-semibold text-white">{e.step}</td>
                <td className="px-3 py-2">{e.fromActivation ? <span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[e.fromActivation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[e.fromActivation] ?? "text-text-secondary"}`}>{e.fromActivation}</span> : <span className="text-text-muted">-</span>}</td>
                <td className="px-1 py-2 text-center text-text-muted">&rarr;</td>
                <td className="px-3 py-2"><span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[e.toActivation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[e.toActivation] ?? "text-text-secondary"}`}>{e.toActivation}</span></td>
                <td className="px-3 py-2 text-text-muted">{e.toGeneration}</td>
                <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromSteps > 0 ? e.fromSteps : "-"}</td>
                <td className="px-3 py-2 text-right font-mono text-white">{e.fromBestLoss != null ? e.fromBestLoss.toFixed(4) : "-"}</td>
                <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromFinalLoss != null ? e.fromFinalLoss.toFixed(4) : "-"}</td>
                <td className="px-3 py-2 text-right font-mono text-green">{e.fromBestFitness != null ? e.fromBestFitness.toFixed(4) : "-"}</td>
                <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromAvgTokPerSec != null ? Math.round(e.fromAvgTokPerSec).toLocaleString() : "-"}</td>
                <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromClipPctAtEnd != null ? `${e.fromClipPctAtEnd.toFixed(0)}%` : "-"}</td>
                <td className="px-3 py-2">{(e.fromCusumAlerts > 0 || e.fromGradSpikes > 0) ? <div className="flex flex-wrap gap-1">{e.fromCusumAlerts > 0 && <span className="rounded bg-orange-500/10 px-1.5 py-0.5 text-[0.58rem] text-orange-400" title={e.fromLastAlertReason ?? undefined}>{e.fromCusumAlerts} cusum</span>}{e.fromGradSpikes > 0 && <span className="rounded bg-red-500/10 px-1.5 py-0.5 text-[0.58rem] text-red-400">{e.fromGradSpikes} spike{e.fromGradSpikes > 1 ? "s" : ""}</span>}</div> : <span className="text-text-muted">-</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ActivationDistributionChart({ metrics }: { metrics: SymbioMetric[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const counts = new Map<string, number>();
    for (const m of metrics) { const act = m.symbio_candidate_activation; if (!act) continue; counts.set(act, (counts.get(act) ?? 0) + 1); }
    if (counts.size === 0) return;
    const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((s, [, n]) => s + n, 0);
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth; const h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d")!; ctx.scale(dpr, dpr);
    const pad = { l: 60, r: 20, t: 10, b: 10 }; const barH = 18;
    ctx.fillStyle = "#0d0d0d"; ctx.fillRect(0, 0, w, h);
    const colors: Record<string, string> = { gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa" };
    const maxW = w - pad.l - pad.r;
    for (let i = 0; i < entries.length && i < 4; i++) {
      const [act, count] = entries[i]; const y = pad.t + i * (barH + 4); const bw = (count / total) * maxW;
      ctx.fillStyle = colors[act] ?? "#888"; ctx.fillRect(pad.l, y, bw, barH);
      ctx.fillStyle = "#aaa"; ctx.font = "10px monospace"; ctx.textAlign = "right"; ctx.fillText(act, pad.l - 6, y + barH / 2 + 4);
      ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(`${((count / total) * 100).toFixed(0)}%`, pad.l + bw + 6, y + barH / 2 + 4);
    }
  }, [metrics]);
  return <canvas ref={canvasRef} className="h-[100px] w-full rounded-lg" />;
}

// ── Symbio Section (composite) ───────────────────────────────

export function SymbioSection({ metrics, run, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  run: { symbio?: number | null; symbio_config?: string | null; ffn_activation?: string | null };
  pinnedStep?: number | null;
  onPinStep?: (s: number) => void;
}) {
  const isSymbio = (run.symbio ?? 0) === 1;
  const hasClipData = metrics.some(m => m.clip_coef != null);
  const hasCusumData = metrics.some(m => m.cusum_grad != null || m.cusum_clip != null || m.cusum_tps != null || m.cusum_val != null);
  const hasSymbioMetrics = metrics.some(m => m.weight_entropy != null);
  const hasAdaptiveBatch = metrics.some(m => m.adaptive_batch_size != null);
  const hasSearchData = metrics.some(m => m.symbio_candidate_id != null);

  let symbioConfig: Record<string, unknown> | null = null;
  try { if (run.symbio_config) symbioConfig = JSON.parse(run.symbio_config); } catch { /* ignore */ }

  if (!isSymbio && !hasClipData) return null;

  return (
    <>
      {hasClipData && (
        <div className="mb-4 rounded-lg border border-border bg-surface p-4">
          <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">Clip Telemetry</div>
          <ClipChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
        </div>
      )}

      {isSymbio && (
        <>
          <div className="mb-4 flex items-center gap-2">
            <span className="rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-wider text-purple-400">Symbio</span>
            {run.ffn_activation && <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[0.6rem] font-medium text-cyan-400">{run.ffn_activation.toUpperCase()}</span>}
          </div>

          <SymbioStatsGrid metrics={metrics} />

          {hasCusumData && (
            <div className="mb-4 rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">CUSUM Change-Point Monitor</div>
              <CusumChart metrics={metrics} sensitivity={(symbioConfig?.cusumSensitivity as number) ?? 4.0} pinnedStep={pinnedStep} onPinStep={onPinStep} />
            </div>
          )}

          {hasSymbioMetrics && (
            <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="rounded-lg border border-border bg-surface p-4">
                <SparseLineChart metrics={metrics} getY={m => m.weight_entropy} color="#a78bfa" label="Weight Entropy" format={v => v.toFixed(2) + " bits"} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-4">
                <SparseLineChart metrics={metrics} getY={m => m.effective_rank} color="#f59e0b" label="Effective Rank" format={v => v.toFixed(1)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-4">
                <SparseLineChart metrics={metrics} getY={m => m.free_energy} color="#34d399" label="Free Energy" format={v => v.toFixed(4)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </div>
              <div className="rounded-lg border border-border bg-surface p-4">
                <SparseLineChart metrics={metrics} getY={m => m.fitness_score} color="#60a5fa" label="Fitness" format={v => v.toFixed(4)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </div>
            </div>
          )}

          {hasAdaptiveBatch && (
            <div className="mb-4 rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">Adaptive Batch Size</div>
              <AdaptiveBatchChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
            </div>
          )}

          {hasSearchData && (
            <>
              <ActivationSwitchLog metrics={metrics} />
              <SearchCandidateTable metrics={metrics} />
              <div className="mb-4 rounded-lg border border-border bg-surface p-4">
                <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">Activation Distribution</div>
                <ActivationDistributionChart metrics={metrics} />
              </div>
            </>
          )}

          {symbioConfig && (
            <details className="mb-4 rounded-lg border border-border bg-surface">
              <summary className="cursor-pointer px-4 py-3 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted hover:text-text">Symbio Config</summary>
              <pre className="overflow-x-auto px-4 pb-3 text-[0.6rem] text-text-muted">{JSON.stringify(symbioConfig, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </>
  );
}
