"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Tip } from "@/components/tooltip";
import {
  type ChartMetric,
  type ChartCheckpoint,
  type MiniSeries,
  type MarkerType,
  type MarkerVisibility,
  type ActivationSwitchEvent,
  type ComputedEvents,
  MARKER_COLORS,
  MARKER_LABELS,
  MARKER_HELP_TEXTS,
  TIMING_PHASES,
  fmtNum,
  cn,
} from "@alpha/ui";

// ── Shared Detection Logic ───────────────────────────────────────

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

function computeEvoValEnvelope(metrics: ChartMetric[]): { step: number; loss: number }[] {
  if (metrics.length < 10) return [];
  const decay = 1 - 1 / 200;
  const smoothAlpha = 0.05;
  let best = metrics[0].loss;
  const raw: number[] = [];
  for (const m of metrics) {
    if (m.loss < best) best = m.loss;
    else best = best * decay + m.loss * (1 - decay);
    raw.push(best);
  }
  const out: { step: number; loss: number }[] = [];
  let ema = raw[0];
  for (let i = 0; i < raw.length; i++) {
    ema = smoothAlpha * raw[i] + (1 - smoothAlpha) * ema;
    out.push({ step: metrics[i].step, loss: ema });
  }
  return out;
}

function computeEvoOverfitRegions(metrics: ChartMetric[]): { startStep: number; endStep: number }[] {
  const byCandidate = new Map<string, ChartMetric[]>();
  for (const m of metrics) {
    const cid = m.symbio_candidate_id;
    if (!cid) continue;
    const arr = byCandidate.get(cid) ?? [];
    arr.push(m);
    byCandidate.set(cid, arr);
  }
  const regions: { startStep: number; endStep: number }[] = [];
  for (const [, cMetrics] of byCandidate) {
    if (cMetrics.length < 8) continue;
    let minLoss = Infinity, minIdx = 0;
    for (let i = 0; i < cMetrics.length; i++) {
      if (cMetrics[i].loss < minLoss) { minLoss = cMetrics[i].loss; minIdx = i; }
    }
    if (minIdx >= cMetrics.length - 3) continue;
    let consecutive = 0, regionStart = -1, prev = minLoss;
    for (let i = minIdx + 1; i < cMetrics.length; i++) {
      if (cMetrics[i].loss > prev + 0.0001) {
        consecutive++;
        if (consecutive === 5 && regionStart < 0) regionStart = cMetrics[i - 4].step;
        prev = cMetrics[i].loss;
      } else {
        if (regionStart >= 0 && consecutive >= 5) regions.push({ startStep: regionStart, endStep: cMetrics[i - 1].step });
        consecutive = 0; regionStart = -1; prev = cMetrics[i].loss;
      }
    }
    if (regionStart >= 0 && consecutive >= 5) regions.push({ startStep: regionStart, endStep: cMetrics[cMetrics.length - 1].step });
  }
  return regions;
}

function computeEvents(metrics: ChartMetric[], checkpoints: ChartCheckpoint[], activationSwitches?: ActivationSwitchEvent[]): ComputedEvents {
  const bestVal = detectBestValStep(metrics);
  return {
    checkpointSteps: checkpoints.map((c) => c.step),
    bestValStep: bestVal?.step ?? null,
    bestValLoss: bestVal?.loss ?? null,
    warmupEndStep: detectWarmupEnd(metrics),
    gradSpikeSteps: detectGradNormSpikes(metrics),
    lossSpikeSteps: detectLossSpikes(metrics),
    overfitStep: detectOverfitStep(metrics),
    activationSwitches: activationSwitches ?? [],
    evoValEnvelope: computeEvoValEnvelope(metrics),
    evoOverfitRegions: computeEvoOverfitRegions(metrics),
  };
}

// ── Small Components ─────────────────────────────────────────────

export { ChartHelpIcon, ChartPanel, Stat, DetailRow } from "@alpha/ui";

// ── Interactive Loss Chart ───────────────────────────────────────

interface LossTooltip {
  pointX: number;
  mouseY: number;
  metric: ChartMetric;
  containerWidth: number;
  nearbySwitch?: ActivationSwitchEvent | null;
}

type LossChartPreset = "traditional" | "evolutionary" | "unified";

function markersForLossPreset(markers: MarkerVisibility, preset: LossChartPreset): MarkerVisibility {
  if (preset === "unified") return markers;
  if (preset === "traditional") {
    return { ...markers, evoValEnvelope: false, evoOverfit: false, activationSwitch: false };
  }
  return { ...markers, bestVal: false, warmupEnd: false, gradSpikes: false, lossSpikes: false, overfit: false };
}

function splitIntoCandidateSegments(metrics: ChartMetric[]): Array<{ start: number; end: number }> {
  if (metrics.length === 0) return [];
  const segments: Array<{ start: number; end: number }> = [];
  let start = 0;
  for (let i = 1; i < metrics.length; i++) {
    const prevId = metrics[i - 1].symbio_candidate_id ?? null;
    const curId = metrics[i].symbio_candidate_id ?? null;
    if (curId !== prevId) { segments.push({ start, end: i - 1 }); start = i; }
  }
  segments.push({ start, end: metrics.length - 1 });
  return segments;
}

function drawLossChart(canvas: HTMLCanvasElement, metrics: ChartMetric[], hoverIdx: number | null, events: ComputedEvents, markers: MarkerVisibility, pinnedStep: number | null = null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  // Theme-aware colors
  const style = getComputedStyle(document.documentElement);
  const bg = style.getPropertyValue("--bg").trim() || "#0a0a0a";
  const surface2 = style.getPropertyValue("--surface-2").trim() || "#1a1a1a";
  const border = style.getPropertyValue("--border").trim() || "#222222";
  const textPrimary = style.getPropertyValue("--text-primary").trim() || "#e0e0e0";
  const textMuted = style.getPropertyValue("--text-muted").trim() || "#555";

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 20, right: 20, bottom: 28, left: 56 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const minStep = 0;
  const maxStep = metrics[metrics.length - 1].step;
  const rangeS = maxStep - minStep || 1;
  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;

  const losses = metrics.map((m) => m.loss);
  const valLosses = metrics.filter((m) => m.val_loss != null).map((m) => m.val_loss!);
  let minL = Math.min(...losses, ...valLosses);
  let maxL = Math.max(...losses, ...valLosses);
  if (minL === maxL) { minL -= 0.1; maxL += 0.1; }
  const rangeL = maxL - minL || 1;
  const sy = (loss: number) => pad.top + (1 - (loss - minL) / rangeL) * ch;

  // Grid
  ctx.strokeStyle = border;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    const val = maxL - (i / 4) * rangeL;
    ctx.fillStyle = textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), pad.left - 6, y + 3);
  }

  // X-axis (Steps)
  ctx.textAlign = "center";
  const ticks = [minStep, Math.round(minStep + rangeS * 0.25), Math.round(minStep + rangeS * 0.5), Math.round(minStep + rangeS * 0.75), maxStep];
  for (const s of ticks) ctx.fillText(fmtNum(s), sx(s), h - 10);

  // Candidate Segments
  const segments = splitIntoCandidateSegments(metrics);
  segments.forEach((seg, si) => {
    // Shaded area for even segments
    if (si % 2 === 1) {
      ctx.fillStyle = surface2 + "40";
      ctx.fillRect(sx(metrics[seg.start].step), pad.top, sx(metrics[seg.end].step) - sx(metrics[seg.start].step), ch);
    }
    // Loss line
    ctx.beginPath();
    ctx.strokeStyle = "#f59e0b"; // yellow
    ctx.lineWidth = 1.5;
    for (let i = seg.start; i <= seg.end; i++) {
      const x = sx(metrics[i].step);
      const y = sy(metrics[i].loss);
      if (i === seg.start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Val loss
  const vpts = metrics.filter(m => m.val_loss != null);
  if (vpts.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = "#2563eb"; // blue
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 2]);
    vpts.forEach((m, i) => {
      const x = sx(m.step);
      const y = sy(m.val_loss!);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Hover crosshair
  if (hoverIdx != null) {
    const m = metrics[hoverIdx];
    const hx = sx(m.step);
    ctx.strokeStyle = textMuted + "80";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(hx, pad.top); ctx.lineTo(hx, h - pad.bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(hx, sy(m.loss), 4, 0, Math.PI * 2); ctx.fillStyle = "#f59e0b"; ctx.fill();
  }
}

function MarkerPillHelp({ marker }: { marker: MarkerType }) {
  return (
    <Tip text={MARKER_HELP_TEXTS[marker]}>
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-border/60 bg-surface-2 text-[0.5rem] font-bold text-text-muted hover:border-text-muted transition-colors">
        ?
      </span>
    </Tip>
  );
}

function MarkerTogglePill({ marker, enabled, onToggle }: { marker: MarkerType; enabled: boolean; onToggle: (marker: MarkerType) => void }) {
  return (
    <div className={cn("flex items-center gap-1 rounded-full border pr-1 transition-all", enabled ? "border-border-2 bg-surface-2 shadow-sm" : "border-border/40 bg-transparent opacity-60")}>
      <button
        type="button"
        onClick={() => onToggle(marker)}
        className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.62rem] font-semibold transition-colors", enabled ? "text-text-primary" : "text-text-muted")}
      >
        <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: MARKER_COLORS[marker] }} />
        {MARKER_LABELS[marker]}
      </button>
      <MarkerPillHelp marker={marker} />
    </div>
  );
}

function LossChartPanel({ title, subtitle, preset, metrics, events, markers, pinnedStep, onPinStep }: { title: string; subtitle: string; preset: LossChartPreset; metrics: ChartMetric[]; events: ComputedEvents; markers: MarkerVisibility; pinnedStep?: number | null; onPinStep?: (step: number) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<LossTooltip | null>(null);
  const hoverRef = useRef<number | null>(null);
  const panelMarkers = useMemo(() => markersForLossPreset(markers, preset), [markers, preset]);
  const modeBadge = preset === "traditional" ? "Classic" : preset === "evolutionary" ? "Evolution" : "Unified";

  const draw = useCallback((idx: number | null = null) => {
    if (canvasRef.current) drawLossChart(canvasRef.current, metrics, idx, events, panelMarkers, pinnedStep ?? null);
  }, [metrics, events, panelMarkers, pinnedStep]);

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
    if (mouseX < padL || mouseX > w - padR) { hoverRef.current = null; setTooltip(null); draw(); return; }
    const minStep = 0;
    const maxStep = metrics[metrics.length - 1].step;
    const rangeS = maxStep - minStep || 1;
    const stepAt = minStep + ((mouseX - padL) / cw) * rangeS;
    let lo = 0, hi = metrics.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (metrics[mid].step < stepAt) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(metrics[lo - 1].step - stepAt) < Math.abs(metrics[lo].step - stepAt)) lo--;
    hoverRef.current = lo;
    const m = metrics[lo];
    const pointX = padL + ((m.step - minStep) / rangeS) * cw;
    const nearbySwitch = panelMarkers.activationSwitch ? (events.activationSwitches.find((sw) => Math.abs(sw.step - m.step) <= rangeS * 0.02) ?? null) : null;
    setTooltip({ pointX, mouseY, metric: m, containerWidth: w, nearbySwitch });
    draw(lo);
  }, [metrics, draw, events, panelMarkers.activationSwitch]);

  return (
    <div className="relative">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="text-[0.7rem] font-bold uppercase tracking-widest text-text-primary">{title}</div>
          <div className="text-[0.65rem] text-text-muted">{subtitle}</div>
        </div>
        <div className="rounded-full border border-border/50 bg-surface-2/80 px-2.5 py-0.5 text-[0.55rem] font-bold uppercase tracking-widest text-text-muted">
          {modeBadge}
        </div>
      </div>
      <canvas ref={canvasRef} className="h-64 w-full cursor-crosshair rounded-xl border border-border/50 bg-surface shadow-inner" onMouseMove={onMove} onMouseLeave={() => { hoverRef.current = null; setTooltip(null); draw(); }} />
      {tooltip && (
        <div className="pointer-events-none absolute z-20 min-w-[180px] rounded-xl border border-border-2 bg-surface-2/95 p-3 shadow-2xl backdrop-blur-sm" style={{ left: tooltip.pointX < tooltip.containerWidth * 0.65 ? tooltip.pointX + 16 : undefined, right: tooltip.pointX >= tooltip.containerWidth * 0.65 ? tooltip.containerWidth - tooltip.pointX + 16 : undefined, top: Math.max(4, tooltip.mouseY - 100) }}>
          <div className="mb-2 font-mono text-[0.75rem] font-bold text-text-primary">Step {tooltip.metric.step.toLocaleString()}</div>
          <div className="space-y-1.5 text-[0.7rem]">
            <div className="flex justify-between gap-4"><span className="text-text-muted">Train Loss</span><span className="font-mono font-bold text-yellow">{tooltip.metric.loss.toFixed(4)}</span></div>
            {tooltip.metric.val_loss != null && <div className="flex justify-between gap-4"><span className="text-text-muted">Val Loss</span><span className="font-mono font-bold text-blue">{tooltip.metric.val_loss.toFixed(4)}</span></div>}
            <div className="my-1.5 border-t border-border/50" />
            <div className="flex justify-between gap-4"><span className="text-text-muted">Tok/sec</span><span className="font-mono text-text-primary">{fmtNum(tooltip.metric.tokens_per_sec)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}

export function InteractiveLossChart({ metrics, checkpoints, pinnedStep, onPinStep, activationSwitches }: { metrics: ChartMetric[]; checkpoints: ChartCheckpoint[]; pinnedStep?: number | null; onPinStep?: (step: number) => void; activationSwitches?: ActivationSwitchEvent[] }) {
  const [markers, setMarkers] = useState<MarkerVisibility>(() => {
    if (typeof window === "undefined") return {} as any;
    try { return JSON.parse(localStorage.getItem("alpha-chart-markers") || "{}"); } catch { return {}; }
  });

  useEffect(() => {
    localStorage.setItem("alpha-chart-markers", JSON.stringify(markers));
  }, [markers]);

  const events = useMemo(() => computeEvents(metrics, checkpoints, activationSwitches), [metrics, checkpoints, activationSwitches]);
  const hasSymbio = metrics.some((m) => m.symbio_candidate_id != null);

  if (metrics.length < 2) return <div className="flex h-64 items-center justify-center rounded-xl border border-border/50 bg-surface-2/30 text-[0.7rem] text-text-muted uppercase tracking-widest font-semibold">Waiting for telemetry...</div>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 py-1">
        {(["checkpoints", "bestVal", "warmupEnd", "overfit"] as MarkerType[]).map(key => (
          <MarkerTogglePill key={key} marker={key} enabled={!!markers[key]} onToggle={k => setMarkers(p => ({ ...p, [k]: !p[k] }))} />
        ))}
      </div>
      <LossChartPanel title={hasSymbio ? "Evolutionary Frontier" : "Training Performance"} subtitle={hasSymbio ? "Evaluating candidates on shared step axis." : "Real-time loss and optimization telemetry."} preset="unified" metrics={metrics} events={events} markers={markers as any} pinnedStep={pinnedStep} onPinStep={onPinStep} />
    </div>
  );
}

// ── Mini Charts ──────────────────────────────────────────────────

function drawMiniChart(canvas: HTMLCanvasElement, series: MiniSeries[], opts: { logScale?: boolean; formatLeft?: (v: number) => string; formatRight?: (v: number) => string }, hoverStep: number | null = null, pinnedStep: number | null = null) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth; const h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const style = getComputedStyle(document.documentElement);
  const bg = style.getPropertyValue("--bg").trim() || "#0a0a0a";
  const border = style.getPropertyValue("--border").trim() || "#222222";
  const textMuted = style.getPropertyValue("--text-muted").trim() || "#555";

  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);

  const hasRight = series.some((s) => s.axis === "right");
  const pad = { top: 12, right: hasRight ? 42 : 16, bottom: 28, left: 48 };
  const cw = w - pad.left - pad.right; const ch = h - pad.top - pad.bottom;

  const allSteps = series.flatMap((s) => s.data.map((d) => d.step));
  const maxStep = allSteps.length > 0 ? Math.max(...allSteps) : 1;
  const sx = (step: number) => pad.left + (step / maxStep) * cw;

  ctx.strokeStyle = border; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
  }

  series.forEach(s => {
    ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
    s.data.forEach((d, i) => {
      const x = sx(d.step);
      const y = pad.top + (1 - (d.value / 100)) * ch; // Simple normalization for now
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

export function MiniChart({ metrics, title, buildSeries, logScale, formatLeft, formatRight, noDataMsg, pinnedStep, onPinStep }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const series = useMemo(() => buildSeries(metrics), [metrics, buildSeries]);
  const hasData = series.some((s: any) => s.data.length >= 2);

  useEffect(() => {
    if (canvasRef.current && hasData) drawMiniChart(canvasRef.current, series, { logScale, formatLeft, formatRight }, null, pinnedStep);
  }, [series, hasData, logScale, formatLeft, formatRight, pinnedStep]);

  return (
    <div className="relative">
      <div className="mb-2 text-[0.65rem] font-bold uppercase tracking-widest text-text-primary">{title}</div>
      {hasData ? <canvas ref={canvasRef} className="h-48 w-full rounded-xl border border-border/50 bg-surface shadow-inner" /> : <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-border/50 bg-surface-2/20 text-[0.65rem] text-text-muted uppercase tracking-widest">No Telemetry</div>}
    </div>
  );
}

export function StepTimeChart({ metrics, pinnedStep, onPinStep }: any) {
  return <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-border/50 bg-surface-2/20 text-[0.65rem] text-text-muted uppercase tracking-widest">Step Analysis Pending</div>;
}

export const buildGpuSeries = (m: ChartMetric[]): MiniSeries[] => {
  const vram = m.filter((x) => x.gpu_vram_used_mb != null).map((x) => ({ step: x.step, value: x.gpu_vram_used_mb! }));
  const util = m.filter((x) => x.gpu_util_pct != null).map((x) => ({ step: x.step, value: x.gpu_util_pct! }));
  return [
    { data: vram, color: "#10b981", label: "VRAM", axis: "left" as const },
    { data: util, color: "#f59e0b", label: "GPU%", axis: "right" as const },
  ];
};

export const buildLrSeries = (m: ChartMetric[]): MiniSeries[] => [{ data: m.map(x => ({ step: x.step, value: x.lr })), color: "#22d3ee", label: "LR" }];
export const buildGradNormSeries = (m: ChartMetric[]): MiniSeries[] => [{ data: m.map(x => ({ step: x.step, value: x.grad_norm })), color: "#f97316", label: "norm" }];
