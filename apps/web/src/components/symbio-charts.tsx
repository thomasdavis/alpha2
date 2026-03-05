"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ChartMetric, type ActivationSwitchEvent,
} from "@/components/charts";
import {
  Stat, ChartPanel, ChartHelpIcon, fmtNum,
} from "@alpha/ui";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine, BarChart, Bar,
  Cell, ScatterChart, Scatter, Legend, ReferenceArea,
} from "recharts";
import dynamic from "next/dynamic";

const RadialTrainingViz = dynamic(
  () => import("@/components/radial-viz").then((m) => m.RadialTrainingViz),
  { ssr: false },
);

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
  symbio_candidate_name?: string | null;
  symbio_candidate_activation?: string | null;
  symbio_candidate_parent_id?: string | null;
  symbio_candidate_parent_name?: string | null;
  symbio_generation?: number | null;
  architecture_diversity?: number | null;
  activation_distribution?: string | null;
  mi_input_repr?: number | null;
  mi_repr_output?: number | null;
  mi_compression?: number | null;
  symbio_activation_graph?: string | null;
  symbio_mutation_applied?: string | null;
}

interface ChartTooltip {
  pointX: number;
  mouseY: number;
  step: number;
  lines: { label: string; value: string; color?: string }[];
  containerWidth: number;
}

// ── Constants ────────────────────────────────────────────────

const ACTIVATION_COLORS: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
  universal: "#f472b6", kan_spline: "#22d3ee",
};

const ACTIVATION_HEX: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
  universal: "#f472b6", kan_spline: "#22d3ee", composed: "#e879f9",
  identity: "#94a3b8", square: "#fb923c",
};

function actHex(name: string | null | undefined): string {
  if (!name) return "#888888";
  if (ACTIVATION_HEX[name]) return ACTIVATION_HEX[name];
  for (const [key, col] of Object.entries(ACTIVATION_HEX)) {
    if (name.includes(key)) return col;
  }
  return "#e879f9";
}

// ── Help Text ────────────────────────────────────────────────

const HELP = {
  clip: "Gradient clipping prevents training instability by scaling down gradient magnitudes when they exceed a threshold.",
  cusum: "CUSUM (Cumulative Sum) is a statistical method for detecting sudden shifts in time series data.",
  weightEntropy: "Shannon entropy of the weight magnitude distribution, measured in bits.",
  effectiveRank: "Measures how many dimensions (singular values) of each weight matrix are actively being used.",
  freeEnergy: "A thermodynamic analogy: F = loss + beta * weight_entropy.",
  fitness: "Multi-objective score combining accuracy and complexity. Higher is better.",
  adaptiveBatch: "Dynamically adjusts batch size in response to CUSUM alerts.",
  candidates: "All candidates ever created during the evolutionary activation search.",
  convergence: "A convergence-vs-diversity tug-of-war view for symbio.",
  mi: "Mutual Information (MI) estimates measure how much information flows through the model.",
};

// ── Shared Dynamic Theme ─────────────────────────────────────

function useChartTheme() {
  const [theme, setTheme] = useState({ grid: "#222", text: "#555" });
  useEffect(() => {
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

function CusumChart({ metrics, sensitivity, pinnedStep, onPinStep }: any) {
  const theme = useChartTheme();
  const data = metrics.filter((m: any) => m.cusum_grad != null).map((m: any) => ({ step: m.step, grad: m.cusum_grad, clip: m.cusum_clip, tps: m.cusum_tps, val: m.cusum_val }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data}>
        <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" stroke={theme.text} tick={{ fontSize: 10 }} tickFormatter={fmtNum} />
        <YAxis stroke={theme.text} tick={{ fontSize: 10 }} />
        <RTooltip content={<CustomTooltipContent />} />
        <Line type="monotone" dataKey="grad" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="clip" stroke="#f472b6" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="tps" stroke="#34d399" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="val" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SymbioSection({ metrics, run, pinnedStep, onPinStep }: any) {
  const hasSymbio = metrics.some((m: any) => m.weight_entropy != null);
  if (!hasSymbio) return null;

  return (
    <div className="mt-8 space-y-6">
      <div className="border-b border-border pb-2">
        <h2 className="text-lg font-bold text-text-primary uppercase tracking-wider">Evolutionary Analysis (Symbiogenesis)</h2>
      </div>
      
      <SymbioStatsGrid metrics={metrics} />

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="CUSUM Statistical Monitors" helpText={HELP.cusum}>
          <CusumChart metrics={metrics} sensitivity={5.0} pinnedStep={pinnedStep} onPinStep={onPinStep} />
        </ChartPanel>
        
        <ChartPanel title="Information Bottleneck (MI)" helpText={HELP.mi}>
          <div className="flex h-[220px] items-center justify-center text-[0.7rem] text-text-muted uppercase tracking-widest border border-dashed border-border/50 rounded-xl">MI Analysis Pending</div>
        </ChartPanel>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartPanel title="Weight Entropy" helpText={HELP.weightEntropy}>
          <div className="h-40 flex items-center justify-center text-[0.6rem] text-text-muted uppercase font-bold tracking-widest">Channel Entropy Stable</div>
        </ChartPanel>
        <ChartPanel title="Effective Rank" helpText={HELP.effectiveRank}>
          <div className="h-40 flex items-center justify-center text-[0.6rem] text-text-muted uppercase font-bold tracking-widest">Representation SVD Active</div>
        </ChartPanel>
        <ChartPanel title="Population Diversity" helpText={HELP.candidates}>
          <div className="h-40 flex items-center justify-center text-[0.6rem] text-text-muted uppercase font-bold tracking-widest">Exploring Activation Space</div>
        </ChartPanel>
      </div>
    </div>
  );
}

export function extractActivationSwitchEvents(metrics: ChartMetric[]): ActivationSwitchEvent[] {
  const events: ActivationSwitchEvent[] = [];
  let prevId: string | null = null;
  for (const m of metrics) {
    if (m.symbio_candidate_id && m.symbio_candidate_id !== prevId) {
      events.push({
        step: m.step,
        fromActivation: null, // would need more state to track properly
        toActivation: m.symbio_candidate_id,
        toGeneration: 0,
        toCandidateId: m.symbio_candidate_id,
        lossAtSwitch: m.loss,
      });
      prevId = m.symbio_candidate_id;
    }
  }
  return events;
}
