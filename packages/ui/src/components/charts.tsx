"use client";

import * as React from "react";
import { Tip } from "./tooltip.js";
import { ChartMetric, ChartCheckpoint, ActivationSwitchEvent, ComputedEvents, MarkerType, MarkerVisibility, MiniSeries } from "../types.js";
import { fmtNum } from "../utils.js";

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
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">{title}</span>
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

// ... other detection functions would go here if needed in multiple places ...
