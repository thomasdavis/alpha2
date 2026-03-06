"use client";

import * as React from "react";
import { Tooltip } from "./tooltip.js";
import { 
  ChartMetric, 
  ChartCheckpoint, 
  ActivationSwitchEvent, 
  ComputedEvents, 
  MarkerType, 
  MarkerVisibility, 
  MiniSeries 
} from "../types.js";
import { fmtNum, cn } from "../utils.js";

// ── Constants & Config ──────────────────────────────────────────

export const DEFAULT_MARKERS: MarkerVisibility = {
  checkpoints: true,
  bestVal: true,
  warmupEnd: true,
  overfit: true,
  gradSpikes: false,
  lossSpikes: false,
  activationSwitch: true,
  evoValEnvelope: true,
  evoOverfit: true,
};

export const MARKER_COLORS: Record<MarkerType, string> = {
  checkpoints: "#34d399",
  bestVal: "#22d3ee",
  warmupEnd: "#a78bfa",
  gradSpikes: "#fb923c",
  lossSpikes: "#f472b6",
  overfit: "#ef4444",
  activationSwitch: "#e879f9",
  evoValEnvelope: "#10b981",
  evoOverfit: "#f97316",
};

export const MARKER_LABELS: Record<MarkerType, string> = {
  checkpoints: "Checkpoints",
  bestVal: "Best Val",
  warmupEnd: "Warmup End",
  gradSpikes: "Grad Spikes",
  lossSpikes: "Loss Spikes",
  overfit: "Overfit",
  activationSwitch: "Activation Switch",
  evoValEnvelope: "Evo Best",
  evoOverfit: "Evo Overfit",
};

export const MARKER_HELP_TEXTS: Record<MarkerType, string> = {
  checkpoints: "Significant training states saved to disk.",
  bestVal: "Point of lowest validation loss achieved.",
  warmupEnd: "End of initial learning rate linear warmup.",
  gradSpikes: "Sudden jumps in gradient norm (potential instability).",
  lossSpikes: "Sudden jumps in training loss.",
  overfit: "Detected onset of training/validation divergence.",
  activationSwitch: "Evolutionary step: activation function was mutated/swapped.",
  evoValEnvelope: "Rolling best validation loss across all candidates.",
  evoOverfit: "Localized overfit regions for specific candidates.",
};

export const TIMING_PHASES = [
  { key: "timing_fwd_ms" as const, label: "Forward", color: "#22d3ee" },
  { key: "timing_bwd_ms" as const, label: "Backward", color: "#f97316" },
  { key: "timing_grad_norm_ms" as const, label: "Grad Norm", color: "#a78bfa" },
  { key: "timing_optim_ms" as const, label: "Optimizer", color: "#10b981" },
  { key: "timing_flush_ms" as const, label: "GPU Sync", color: "#f43f5e" },
  { key: "timing_data_ms" as const, label: "Data", color: "#64748b" },
];

// ── Small Components ─────────────────────────────────────────────

export function ChartHelpIcon({ text }: { text: string }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen(!open)}
        className="flex h-5 w-5 items-center justify-center rounded-full border border-border/60 bg-surface-2 text-[0.6rem] text-text-muted transition-colors hover:border-border-2 hover:text-text-secondary"
        title="What is this?"
      >
        ?
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-7 z-40 w-72 rounded-lg border border-border-2 bg-surface-2 p-3 shadow-xl text-[0.68rem] leading-relaxed text-text-secondary">
            {text}
          </div>
        </>
      )}
    </div>
  );
}

export function ChartPanel({ title, helpText, children }: { title: string; helpText?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[0.65rem] font-bold uppercase tracking-widest text-text-primary">{title}</span>
        {helpText && <ChartHelpIcon text={helpText} />}
      </div>
      {children}
    </div>
  );
}

// ── Shared Detection Logic ───────────────────────────────────────

export function detectBestValStep(metrics: ChartMetric[]): { step: number; loss: number } | null {
  const valPts = metrics.filter((m) => m.val_loss != null);
  if (valPts.length === 0) return null;
  let best = valPts[0];
  for (let i = 1; i < valPts.length; i++) {
    if (valPts[i].val_loss! < best.val_loss!) best = valPts[i];
  }
  return { step: best.step, loss: best.val_loss! };
}

export function detectWarmupEnd(metrics: ChartMetric[]): number | null {
  if (metrics.length < 3) return null;
  let peakIdx = 0;
  for (let i = 1; i < metrics.length; i++) {
    if (metrics[i].lr > metrics[peakIdx].lr) peakIdx = i;
  }
  if (peakIdx <= 1 || peakIdx >= metrics.length - 2) return null;
  return metrics[peakIdx].step;
}

export function detectGradNormSpikes(metrics: ChartMetric[]): number[] {
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

export function detectLossSpikes(metrics: ChartMetric[]): number[] {
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

export function detectOverfitStep(metrics: ChartMetric[]): number | null {
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
  if ((avgAfter - minVal) / minVal < 0.02) return null;
  return valPts[minIdx].step;
}

export function computeEvoValEnvelope(metrics: ChartMetric[]): { step: number; loss: number }[] {
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

export function computeEvoOverfitRegions(metrics: ChartMetric[]): { startStep: number; endStep: number }[] {
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

export function computeEvents(metrics: ChartMetric[], checkpoints: ChartCheckpoint[], activationSwitches?: ActivationSwitchEvent[]): ComputedEvents {
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

const PINNED_MARKER_COLOR = "#a855f7";

function drawPinnedMarkers(ctx: CanvasRenderingContext2D, pinnedSteps: number[], sx: (step: number) => number, top: number, bottom: number, w: number) {
  if (pinnedSteps.length === 0) return;
  ctx.save();
  ctx.strokeStyle = PINNED_MARKER_COLOR;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.fillStyle = PINNED_MARKER_COLOR;
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  for (const step of pinnedSteps) {
    const x = sx(step);
    if (x < 0 || x > w) continue;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    // Step label at top
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.85;
    const label = fmtNum(step);
    const tw = ctx.measureText(label).width + 6;
    ctx.fillStyle = PINNED_MARKER_COLOR + "30";
    ctx.fillRect(x - tw / 2, top - 1, tw, 13);
    ctx.fillStyle = PINNED_MARKER_COLOR;
    ctx.fillText(label, x, top + 9);
    ctx.globalAlpha = 1;
    ctx.setLineDash([4, 3]);
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function findNearestStep(metrics: ChartMetric[], mouseX: number, padL: number, cw: number, maxStep: number): number | null {
  if (metrics.length < 2 || cw <= 0) return null;
  const stepAt = ((mouseX - padL) / cw) * maxStep;
  let lo = 0, hi = metrics.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (metrics[mid].step < stepAt) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(metrics[lo - 1].step - stepAt) < Math.abs(metrics[lo].step - stepAt)) lo--;
  return metrics[lo].step;
}

function drawLossChart(canvas: HTMLCanvasElement, metrics: ChartMetric[], hoverIdx: number | null, events: ComputedEvents, markers: MarkerVisibility, pinnedSteps: number[] = []) {
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
  const textMuted = style.getPropertyValue("--text-muted").trim() || "#555";

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const pad = { top: 20, right: 20, bottom: 28, left: 56 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  if (metrics.length < 2) return;

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

  // Markers (Checkpoints etc)
  if (markers.checkpoints) {
    ctx.strokeStyle = MARKER_COLORS.checkpoints;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    events.checkpointSteps.forEach(s => {
      const x = sx(s);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, h - pad.bottom); ctx.stroke();
    });
    ctx.setLineDash([]);
  }

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

  // Pinned step markers
  drawPinnedMarkers(ctx, pinnedSteps, sx, pad.top, h - pad.bottom, w);

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
    <Tooltip text={MARKER_HELP_TEXTS[marker]}>
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-border/60 bg-surface-2 text-[0.5rem] font-bold text-text-muted hover:border-text-muted transition-colors">
        ?
      </span>
    </Tooltip>
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

export function InteractiveLossChart({ metrics, checkpoints, pinnedSteps, onPinStep, activationSwitches }: { metrics: ChartMetric[]; checkpoints: ChartCheckpoint[]; pinnedSteps?: number[]; onPinStep?: (step: number) => void; activationSwitches?: ActivationSwitchEvent[] }) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = React.useState<LossTooltip | null>(null);
  const hoverRef = React.useRef<number | null>(null);
  const [markers, setMarkers] = React.useState<MarkerVisibility>(DEFAULT_MARKERS);

  const events = React.useMemo(() => computeEvents(metrics, checkpoints, activationSwitches), [metrics, checkpoints, activationSwitches]);

  const draw = React.useCallback((idx: number | null = null) => {
    if (canvasRef.current) drawLossChart(canvasRef.current, metrics, idx, events, markers, pinnedSteps ?? []);
  }, [metrics, events, markers, pinnedSteps]);

  React.useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const obs = new ResizeObserver(() => draw(hoverRef.current));
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [draw]);

  const onMove = React.useCallback((e: React.MouseEvent) => {
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
    setTooltip({ pointX, mouseY, metric: m, containerWidth: w });
    draw(lo);
  }, [metrics, draw]);

  if (metrics.length < 2) return <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-border/50 bg-surface-2/20 text-[0.7rem] text-text-muted uppercase tracking-widest font-semibold">Waiting for telemetry...</div>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 py-1">
        {(["checkpoints", "bestVal", "warmupEnd", "overfit"] as MarkerType[]).map(key => (
          <MarkerTogglePill key={key} marker={key} enabled={!!markers[key]} onToggle={k => setMarkers(p => ({ ...p, [k]: !p[k] }))} />
        ))}
      </div>
      <div className="relative">
        <canvas ref={canvasRef} className="h-64 w-full cursor-crosshair rounded-xl border border-border/50 bg-surface shadow-inner" onMouseMove={onMove} onMouseLeave={() => { hoverRef.current = null; setTooltip(null); draw(); }} onClick={(e) => {
          const canvas = canvasRef.current;
          if (!canvas || metrics.length < 2 || !onPinStep) return;
          const rect = canvas.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const padL = 56, padR = 20, cw = canvas.clientWidth - padL - padR;
          if (mouseX < padL || mouseX > canvas.clientWidth - padR) return;
          const step = findNearestStep(metrics, mouseX, padL, cw, metrics[metrics.length - 1].step);
          if (step != null) onPinStep(step);
        }} />
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
    </div>
  );
}

// ── Mini Charts ──────────────────────────────────────────────────

function drawMiniChart(canvas: HTMLCanvasElement, series: MiniSeries[], opts: { logScale?: boolean; formatLeft?: (v: number) => string; formatRight?: (v: number) => string }, hoverStep: number | null = null, pinnedSteps: number[] = []) {
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
    const values = s.data.map(dv => dv.value);
    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    const rangeV = maxV - minV || 1;
    s.data.forEach((d, i) => {
      const x = sx(d.step);
      const y = pad.top + (1 - (d.value - minV) / rangeV) * ch;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Pinned markers
  drawPinnedMarkers(ctx, pinnedSteps, sx, pad.top, h - pad.bottom, w);
}

export interface MiniChartProps {
  metrics: ChartMetric[];
  title: string;
  buildSeries: (m: ChartMetric[]) => MiniSeries[];
  logScale?: boolean;
  formatLeft?: (v: number) => string;
  formatRight?: (v: number) => string;
  noDataMsg?: string;
  pinnedSteps?: number[];
  onPinStep?: (step: number) => void;
}

export function MiniChart({ metrics, title, buildSeries, logScale, formatLeft, formatRight, noDataMsg, pinnedSteps, onPinStep }: MiniChartProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const series = React.useMemo(() => buildSeries(metrics), [metrics, buildSeries]);
  const hasData = series.some((s) => s.data.length >= 2);

  React.useEffect(() => {
    if (canvasRef.current && hasData) drawMiniChart(canvasRef.current, series, { logScale, formatLeft, formatRight }, null, pinnedSteps ?? []);
  }, [series, hasData, logScale, formatLeft, formatRight, pinnedSteps]);

  const handleClick = React.useCallback((e: React.MouseEvent) => {
    if (!onPinStep || metrics.length < 2) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const hasRight = series.some(s => s.axis === "right");
    const padL = 48, padR = hasRight ? 42 : 16;
    const cw = canvas.clientWidth - padL - padR;
    if (mouseX < padL || mouseX > canvas.clientWidth - padR) return;
    const step = findNearestStep(metrics, mouseX, padL, cw, metrics[metrics.length - 1].step);
    if (step != null) onPinStep(step);
  }, [metrics, series, onPinStep]);

  return (
    <div className="relative">
      <div className="mb-2 text-[0.65rem] font-bold uppercase tracking-widest text-text-primary">{title}</div>
      {hasData ? <canvas ref={canvasRef} className="h-48 w-full cursor-crosshair rounded-xl border border-border/50 bg-surface shadow-inner" onClick={handleClick} /> : <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-border/50 bg-surface-2/20 text-[0.65rem] text-text-muted uppercase tracking-widest">{noDataMsg || "No Telemetry"}</div>}
    </div>
  );
}

export function StepTimeChart({ metrics, pinnedSteps, onPinStep }: { metrics: ChartMetric[]; pinnedSteps?: number[]; onPinStep?: (step: number) => void }) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = React.useState<{ x: number; y: number; step: number; phases: { label: string; color: string; ms: number }[] } | null>(null);

  const timed = React.useMemo(() => {
    const pts = metrics.filter(m => m.timing_fwd_ms != null);
    if (pts.length <= 400) return pts;
    const stride = Math.ceil(pts.length / 400);
    return pts.filter((_, i) => i % stride === 0);
  }, [metrics]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || timed.length < 2) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = 192;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);

    const style = getComputedStyle(document.documentElement);
    const bg = style.getPropertyValue("--bg").trim() || "#0a0a0a";
    const border = style.getPropertyValue("--border").trim() || "#222";
    const textMuted = style.getPropertyValue("--text-muted").trim() || "#555";

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const pad = { top: 10, right: 16, bottom: 24, left: 48 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    const maxStep = timed[timed.length - 1].step;
    const sx = (step: number) => pad.left + (step / (maxStep || 1)) * cw;

    // Compute stacked totals to find max
    let maxTotal = 0;
    for (const m of timed) {
      const total = (m.timing_fwd_ms ?? 0) + (m.timing_bwd_ms ?? 0) + (m.timing_optim_ms ?? 0) + (m.timing_flush_ms ?? 0) + (m.timing_data_ms ?? 0);
      if (total > maxTotal) maxTotal = total;
    }
    maxTotal = maxTotal || 1;
    const sy = (ms: number) => ch - (ms / maxTotal) * ch;

    // Grid
    ctx.strokeStyle = border;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * ch;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = textMuted;
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText((maxTotal * (1 - i / 4)).toFixed(0) + "ms", pad.left - 4, y + 3);
    }

    // Stacked areas (bottom to top: data, gpu sync, optimizer, backward, forward)
    const phases = TIMING_PHASES.slice().reverse();
    const barW = Math.max(1, cw / timed.length);

    for (let pi = 0; pi < phases.length; pi++) {
      ctx.fillStyle = phases[pi].color + "cc";
      for (let i = 0; i < timed.length; i++) {
        const m = timed[i];
        let cumBelow = 0;
        for (let j = pi + 1; j < phases.length; j++) {
          cumBelow += (m[phases[j].key] as number) ?? 0;
        }
        const val = (m[phases[pi].key] as number) ?? 0;
        const x = sx(m.step);
        const yTop = pad.top + sy(cumBelow + val);
        const yBot = pad.top + sy(cumBelow);
        ctx.fillRect(x, yTop, barW, yBot - yTop);
      }
    }

    // Pinned markers
    drawPinnedMarkers(ctx, pinnedSteps ?? [], sx, pad.top, h - pad.bottom, w);

    // X labels
    ctx.fillStyle = textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    const ticks = [0, Math.round(maxStep * 0.25), Math.round(maxStep * 0.5), Math.round(maxStep * 0.75), maxStep];
    for (const s of ticks) ctx.fillText(fmtNum(s), sx(s), h - 6);
  }, [timed, pinnedSteps]);

  if (timed.length < 2) {
    return <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-border/50 bg-surface-2/20 text-[0.65rem] text-text-muted uppercase tracking-widest">No timing data</div>;
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas
        ref={canvasRef}
        className="w-full rounded-xl border border-border/50 bg-surface shadow-inner cursor-crosshair"
        onMouseMove={(e) => {
          const canvas = canvasRef.current;
          if (!canvas || timed.length < 2) return;
          const rect = canvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const padL = 48, padR = 16;
          const cw = rect.width - padL - padR;
          if (mx < padL || mx > rect.width - padR) { setTooltip(null); return; }
          const maxStep = timed[timed.length - 1].step;
          const stepAt = ((mx - padL) / cw) * maxStep;
          let idx = 0;
          for (let i = 1; i < timed.length; i++) {
            if (Math.abs(timed[i].step - stepAt) < Math.abs(timed[idx].step - stepAt)) idx = i;
          }
          const m = timed[idx];
          setTooltip({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            step: m.step,
            phases: TIMING_PHASES.map(p => ({ label: p.label, color: p.color, ms: (m[p.key] as number) ?? 0 })),
          });
        }}
        onMouseLeave={() => setTooltip(null)}
        onClick={(e) => {
          if (!onPinStep || timed.length < 2) return;
          const canvas = canvasRef.current;
          if (!canvas) return;
          const rect = canvas.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const padL = 48, padR = 16, cw = rect.width - padL - padR;
          if (mx < padL || mx > rect.width - padR) return;
          const step = findNearestStep(timed, mx, padL, cw, timed[timed.length - 1].step);
          if (step != null) onPinStep(step);
        }}
      />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 min-w-[160px] rounded-lg border border-border-2 bg-surface-2/95 p-2.5 shadow-xl backdrop-blur-sm text-[0.65rem]"
          style={{ left: tooltip.x < 200 ? tooltip.x + 12 : undefined, right: tooltip.x >= 200 ? (containerRef.current?.clientWidth ?? 400) - tooltip.x + 12 : undefined, top: Math.max(4, tooltip.y - 80) }}
        >
          <div className="mb-1.5 font-mono font-bold text-text-primary">Step {fmtNum(tooltip.step)}</div>
          {tooltip.phases.filter(p => p.ms > 0).map(p => (
            <div key={p.label} className="flex justify-between gap-3">
              <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.color }} />{p.label}</span>
              <span className="font-mono font-bold text-text-primary">{p.ms.toFixed(1)}ms</span>
            </div>
          ))}
          <div className="mt-1 border-t border-border/50 pt-1 flex justify-between gap-3">
            <span className="text-text-muted">Total</span>
            <span className="font-mono font-bold text-text-primary">{tooltip.phases.reduce((s, p) => s + p.ms, 0).toFixed(1)}ms</span>
          </div>
        </div>
      )}
      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-3 justify-center">
        {TIMING_PHASES.map(p => (
          <div key={p.key} className="flex items-center gap-1 text-[0.6rem] text-text-muted">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.color }} />
            {p.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Core Charting Components (Base Implementations) ──────────────

export function BaseMiniChart({ 
  data, 
  title, 
  height = 200,
  logScale = false,
  formatValue = (v: number) => v.toFixed(2)
}: { 
  data: { step: number; value: number }[]; 
  title: string;
  height?: number;
  logScale?: boolean;
  formatValue?: (v: number) => string;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const style = getComputedStyle(document.documentElement);
    const bg = style.getPropertyValue("--bg").trim() || "#0a0a0a";
    const border = style.getPropertyValue("--border").trim() || "#222222";
    const textMuted = style.getPropertyValue("--text-muted").trim() || "#555";
    const accent = style.getPropertyValue("--accent").trim() || "#2563eb";

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    const pad = { top: 10, right: 10, bottom: 20, left: 40 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    const minStep = data[0].step;
    const maxStep = data[data.length - 1].step;
    const rangeS = maxStep - minStep || 1;

    const values = data.map(d => d.value);
    let minV = Math.min(...values);
    let maxV = Math.max(...values);
    if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
    const rangeV = maxV - minV || 1;

    const sx = (s: number) => pad.left + ((s - minStep) / rangeS) * cw;
    const sy = (v: number) => pad.top + (1 - (v - minV) / rangeV) * ch;

    // Grid
    ctx.strokeStyle = border;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * ch;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    data.forEach((d, i) => {
      const x = sx(d.step);
      const y = sy(d.value);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = textMuted;
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(formatValue(maxV), pad.left - 4, pad.top + 3);
    ctx.fillText(formatValue(minV), pad.left - 4, pad.top + ch + 3);
  }, [data, logScale, formatValue]);

  return (
    <div className="space-y-2">
      <div className="text-[0.6rem] font-bold uppercase tracking-widest text-text-muted">{title}</div>
      <canvas ref={canvasRef} className="w-full rounded-lg border border-border/50 shadow-inner" style={{ height }} />
    </div>
  );
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

export const buildThroughputSeries = (m: ChartMetric[]): MiniSeries[] => [
  { data: m.map(x => ({ step: x.step, value: x.tokens_per_sec })), color: "#10b981", label: "tok/s" },
];

export const buildStepTimeSeries = (m: ChartMetric[]): MiniSeries[] => [
  { data: m.map(x => ({ step: x.step, value: x.ms_per_iter })), color: "#f472b6", label: "ms/iter" },
];

export const buildClipSeries = (m: ChartMetric[]): MiniSeries[] => {
  const coef = m.filter(x => x.clip_coef != null).map(x => ({ step: x.step, value: x.clip_coef! }));
  const pct = m.filter(x => x.clip_pct != null).map(x => ({ step: x.step, value: x.clip_pct! }));
  return [
    { data: coef, color: "#f59e0b", label: "Clip Coef", axis: "left" as const },
    { data: pct, color: "#f472b6", label: "Clip %", axis: "right" as const },
  ];
};

export const buildGpuOpsSeries = (m: ChartMetric[]): MiniSeries[] => {
  const ops = m.filter(x => x.gpu_ops_count != null).map(x => ({ step: x.step, value: x.gpu_ops_count! }));
  return [{ data: ops, color: "#a78bfa", label: "ops" }];
};

export const buildPerplexitySeries = (m: ChartMetric[]): MiniSeries[] => {
  const valPts = m.filter(x => x.val_loss != null);
  return [
    { data: m.map(x => ({ step: x.step, value: Math.exp(x.loss) })), color: "#f59e0b", label: "Train PPL" },
    ...(valPts.length > 1 ? [{ data: valPts.map(x => ({ step: x.step, value: Math.exp(x.val_loss!) })), color: "#2563eb", label: "Val PPL" }] : []),
  ];
};

export const buildTrainValGapSeries = (m: ChartMetric[]): MiniSeries[] => {
  const valPts = m.filter(x => x.val_loss != null);
  if (valPts.length < 2) return [];
  return [{ data: valPts.map(x => ({ step: x.step, value: x.val_loss! - x.loss })), color: "#ef4444", label: "Gap" }];
};

export const buildLossVelocitySeries = (m: ChartMetric[]): MiniSeries[] => {
  if (m.length < 10) return [];
  const windowSize = 10;
  const velocity: { step: number; value: number }[] = [];
  for (let i = windowSize; i < m.length; i++) {
    const dLoss = m[i].loss - m[i - windowSize].loss;
    const dStep = m[i].step - m[i - windowSize].step || 1;
    velocity.push({ step: m[i].step, value: (dLoss / dStep) * 1000 });
  }
  return [{ data: velocity, color: "#22d3ee", label: "dL/dStep" }];
};

export const buildSmoothedLossSeries = (m: ChartMetric[]): MiniSeries[] => {
  if (m.length < 5) return [];
  const alpha = 0.05;
  let ema = m[0].loss;
  const smoothed = m.map(x => {
    ema = alpha * x.loss + (1 - alpha) * ema;
    return { step: x.step, value: ema };
  });
  const raw = m.map(x => ({ step: x.step, value: x.loss }));
  return [
    { data: raw, color: "#f59e0b40", label: "Raw" },
    { data: smoothed, color: "#f59e0b", label: "EMA" },
  ];
};

export const buildFwdBwdRatioSeries = (m: ChartMetric[]): MiniSeries[] => {
  const pts = m.filter(x => x.timing_fwd_ms != null && x.timing_bwd_ms != null && x.timing_fwd_ms! > 0);
  if (pts.length < 2) return [];
  return [{ data: pts.map(x => ({ step: x.step, value: x.timing_bwd_ms! / x.timing_fwd_ms! })), color: "#a78bfa", label: "Bwd/Fwd" }];
};

export const buildTimingPhaseSeries = (m: ChartMetric[]): MiniSeries[] => {
  const pts = m.filter(x => x.timing_fwd_ms != null);
  if (pts.length < 2) return [];
  return [
    { data: pts.map(x => ({ step: x.step, value: x.timing_fwd_ms! })), color: "#22d3ee", label: "Forward" },
    { data: pts.map(x => ({ step: x.step, value: x.timing_bwd_ms ?? 0 })), color: "#f97316", label: "Backward" },
    { data: pts.map(x => ({ step: x.step, value: x.timing_optim_ms ?? 0 })), color: "#10b981", label: "Optimizer" },
    { data: pts.map(x => ({ step: x.step, value: x.timing_flush_ms ?? 0 })), color: "#f43f5e", label: "GPU Sync" },
    { data: pts.map(x => ({ step: x.step, value: x.timing_data_ms ?? 0 })), color: "#64748b", label: "Data" },
  ];
};
