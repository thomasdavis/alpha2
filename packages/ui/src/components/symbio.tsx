"use client";

import * as React from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine, Legend
} from "recharts";
import { ChartMetric, ActivationSwitchEvent } from "../types.js";
import { Stat } from "./stat.js";
import { ChartPanel } from "./charts.js";
import { fmtNum } from "../utils.js";

// ── Types ────────────────────────────────────────────────────

export interface SymbioMetric extends ChartMetric {
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
}

// ── Constants ────────────────────────────────────────────────

export const ACTIVATION_HEX: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
  universal: "#f472b6", kan_spline: "#22d3ee", composed: "#e879f9",
  identity: "#94a3b8", square: "#fb923c",
};

export const SYMBIO_HELP = {
  cusum: "CUSUM (Cumulative Sum) is a statistical method for detecting sudden shifts in time series data. Four independent monitors track gradient norms, clipping rates, throughput, and validation loss.",
  mi: "Mutual Information (MI) estimates measure how much information flows through the model.",
  weightEntropy: "Shannon entropy of the weight magnitude distribution, measured in bits.",
  effectiveRank: "Measures how many dimensions (singular values) of each weight matrix are actively being used.",
  fitness: "Multi-objective score combining accuracy and complexity. Higher is better.",
};

// ── Shared Dynamic Theme ─────────────────────────────────────

function useChartTheme() {
  const [theme, setTheme] = React.useState({ grid: "#222", text: "#555" });
  React.useEffect(() => {
    const style = getComputedStyle(document.documentElement);
    setTheme({
      grid: style.getPropertyValue("--border").trim() || "#222",
      text: style.getPropertyValue("--text-muted").trim() || "#555",
    });
  }, []);
  return theme;
}

function CustomTooltipContent({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border-2 bg-surface-2/95 p-2.5 shadow-xl text-[0.64rem] backdrop-blur-sm">
      <div className="mb-1 font-mono font-bold text-text-primary">Step {Number(label).toLocaleString()}</div>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex justify-between gap-3">
          <span className="text-text-muted">{p.name}</span>
          <span className="font-mono font-bold" style={{ color: p.color }}>{typeof p.value === "number" ? p.value.toFixed(4) : String(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

// ── Components ───────────────────────────────────────────────

export function SymbioStatsGrid({ metrics }: { metrics: SymbioMetric[] }) {
  const last = [...metrics].reverse().find(m => m.weight_entropy != null);
  if (!last) return null;

  return (
    <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
      <Stat label="Wt Entropy" value={last.weight_entropy?.toFixed(2) ?? "-"} sub="bits" color="text-purple-400" />
      <Stat label="Eff. Rank" value={last.effective_rank?.toFixed(1) ?? "-"} color="text-amber-400" />
      <Stat label="Free Energy" value={last.free_energy?.toFixed(4) ?? "-"} color="text-green" />
      <Stat label="Pop Entropy" value={last.population_entropy?.toFixed(3) ?? "-"} sub="nats" color="text-cyan-400" />
      <Stat label="Complexity" value={last.complexity_score?.toFixed(4) ?? "-"} color="text-rose-400" />
      <Stat label="Fitness" value={last.fitness_score?.toFixed(4) ?? "-"} color="text-blue" />
      <Stat label="CUSUM" value={String(metrics.filter(m => (m.cusum_alerts ?? 0) > 0).length)} sub="alerts" />
      <Stat label="Batch Size" value={last.adaptive_batch_size?.toFixed(0) ?? "-"} sub="adaptive" color="text-teal-400" />
    </div>
  );
}

export function CusumChart({ metrics, sensitivity = 5.0, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; sensitivity?: number; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const theme = useChartTheme();
  const data = metrics.filter((m) => m.cusum_grad != null).map((m) => ({ step: m.step, grad: m.cusum_grad, clip: m.cusum_clip, tps: m.cusum_tps, val: m.cusum_val }));
  
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} onClick={(e: any) => e?.activeLabel && onPinStep?.(Number(e.activeLabel))}>
        <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" stroke={theme.text} tick={{ fontSize: 10 }} tickFormatter={fmtNum} />
        <YAxis stroke={theme.text} tick={{ fontSize: 10 }} />
        <RTooltip content={<CustomTooltipContent />} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <ReferenceLine y={sensitivity} stroke="#ef4444" strokeDasharray="6 3" />
        <Line type="monotone" dataKey="grad" name="Gradient" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="clip" name="Clipping" stroke="#f472b6" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="tps" name="Throughput" stroke="#34d399" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="val" name="Val Loss" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
        <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export interface SymbioRunData {
  symbio?: number | null;
  symbio_config?: string | null;
  ffn_activation?: string | null;
  symbio_mode?: string | null;
}

export function SymbioSection({ metrics, run, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; run?: SymbioRunData; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const hasSymbio = metrics.some((m) => m.weight_entropy != null || m.cusum_grad != null);
  if (!hasSymbio && !run?.symbio) return null;

  return (
    <div className="mt-8 space-y-6">
      <div className="border-b border-border pb-2">
        <h2 className="text-lg font-bold text-text-primary uppercase tracking-wider">Evolutionary Analysis (Symbiogenesis)</h2>
      </div>
      
      <SymbioStatsGrid metrics={metrics} />

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="CUSUM Statistical Monitors" helpText={SYMBIO_HELP.cusum}>
          <CusumChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
        </ChartPanel>
        
        <ChartPanel title="Information Bottleneck (MI)" helpText={SYMBIO_HELP.mi}>
          <div className="flex h-[220px] items-center justify-center text-[0.7rem] text-text-muted uppercase tracking-widest border border-dashed border-border/50 rounded-xl">MI Analysis Pending</div>
        </ChartPanel>
      </div>
    </div>
  );
}
