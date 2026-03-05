"use client";

import * as React from "react";
import { Tip } from "./tooltip.js";
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

export function computeEvents(metrics: ChartMetric[], checkpoints: ChartCheckpoint[], activationSwitches?: ActivationSwitchEvent[]): ComputedEvents {
  const bestVal = detectBestValStep(metrics);
  return {
    checkpointSteps: checkpoints.map((c) => c.step),
    bestValStep: bestVal?.step ?? null,
    bestValLoss: bestVal?.loss ?? null,
    warmupEndStep: detectWarmupEnd(metrics),
    gradSpikeSteps: [], 
    lossSpikeSteps: [],
    overfitStep: detectOverfitStep(metrics),
    activationSwitches: activationSwitches ?? [],
    evoValEnvelope: [],
    evoOverfitRegions: [],
  };
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
