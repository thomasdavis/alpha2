"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ChartMetric, type ActivationSwitchEvent,
  Stat, ChartPanel, ChartHelpIcon, fmtNum,
} from "@/components/charts";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine, BarChart, Bar,
  Cell, ScatterChart, Scatter, Legend,
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

const ACTIVATION_BG: Record<string, string> = {
  gelu: "bg-blue-500/10 border-blue-500/20", silu: "bg-green-500/10 border-green-500/20",
  relu: "bg-yellow-500/10 border-yellow-500/20", swiglu: "bg-purple-500/10 border-purple-500/20",
  universal: "bg-pink-500/10 border-pink-500/20", kan_spline: "bg-cyan-500/10 border-cyan-500/20",
};

const ACTIVATION_HEX: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
  universal: "#f472b6", kan_spline: "#22d3ee", composed: "#e879f9",
  identity: "#94a3b8", square: "#fb923c",
};

/** Get hex color for an activation name (may be a composed formula). Always returns 7-char hex. */
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
  clip: "Gradient clipping prevents training instability by scaling down gradient magnitudes when they exceed a threshold. clip_coef shows the actual scaling factor applied (1.0 = no clipping). clip_pct shows what fraction of the gradient norm exceeded the threshold.",
  cusum: "CUSUM (Cumulative Sum) is a statistical method for detecting sudden shifts in time series data. Four independent monitors track gradient norms, clipping rates, throughput, and validation loss. When the cumulative deviation from baseline exceeds a sensitivity threshold, an alert fires indicating a training regime change.",
  weightEntropy: "Shannon entropy of the weight magnitude distribution, measured in bits. Higher entropy means weights are more uniformly distributed across magnitudes. Low entropy means weights cluster at specific magnitudes, which may indicate under-utilization of model capacity.",
  effectiveRank: "Measures how many dimensions (singular values) of each weight matrix are actively being used. Computed via SVD: counts singular values above 1% of the largest. Higher rank means the model uses more of its representational capacity.",
  freeEnergy: "A thermodynamic analogy: F = loss + beta * weight_entropy. Balances model fit (loss) against complexity (entropy). Lower free energy means a better trade-off between accuracy and model simplicity.",
  fitness: "Multi-objective score combining accuracy and complexity: fitness = alpha * (1/(1+loss)) - complexity_penalty. Used to rank candidates in the evolutionary search. Higher is better.",
  adaptiveBatch: "Dynamically adjusts batch size in response to CUSUM alerts. Reduces batch on gradient instability or throughput drops (smaller batches recover faster). Increases batch when clipping is persistent (larger batches smooth gradients). Gradually restores to baseline during calm periods.",
  switchLog: "Records every activation function switch during evolutionary search. Shows the 'from' and 'to' activation with the outgoing candidate's performance. Hover any row for detailed metrics (loss, fitness, throughput, stability). Click a row to navigate to that candidate's position in the evolutionary tree chart.",
  candidates: "All candidates ever created during the evolutionary activation search. Each was trained for stepsPerCandidate steps and ranked by validation loss or fitness. Includes learnable activations: 'universal' (per-channel SiLU gating) and 'kan_spline' (5-basis KAN approximator with silu/relu/gelu/identity/quadratic bases).",
  activationDist: "Shows how training steps are distributed across different activation functions over time. Includes fixed activations (gelu, silu, relu, swiglu) and learnable activations (universal, kan_spline). In a converging search, the winning activation accumulates more steps in later generations.",
  evolution: "Visualizes the evolutionary search progression across generations. Each generation evaluates candidates with unique IDs and names tracking their lineage. Parent-child relationships are shown in the tree chart. The fitness landscape shows how candidates improve or plateau over generations.",
  phaseChange: "Gelation/phase changes are detected via CUSUM monitors. When gradient norms, clipping rates, or throughput shift dramatically, the training has entered a new regime. Green = stable (no alerts), Yellow = mild instability (1-2 alerts), Red = regime shift (3+ alerts).",
  harmonic: "Analyzes the oscillatory behavior of training loss around its moving average. High-amplitude oscillations suggest an under-damped system (learning rate too high). The oscillation frequency and damping ratio characterize the training dynamics near equilibrium.",
  harmonicAmcharts: "Frequency-domain analysis of loss oscillations using a sliding-window FFT analogy. Shows the amplitude envelope of loss deviations from the moving average over time, revealing whether training dynamics are damping (converging) or amplifying (diverging). The heat capacity proxy (d(loss)/d(lr)) indicates critical points where training dynamics are most sensitive to hyperparameter changes.",
  diversity: "Architecture diversity measures what fraction of the current population uses unique activations (0 = all same, 1 = all different). With 6 activation types (gelu, silu, relu, swiglu, universal, kan_spline), diversity starts high and gradually decreases as the search converges. Diversity bonus (cosine decay) encourages exploration early, exploitation late.",
  populationEntropy: "Shannon entropy of the softmax-normalized recent loss distribution (50-step window). Higher entropy means loss values are spread widely (unstable). Lower entropy means loss is concentrated (stable). Rapid entropy changes indicate training phase transitions.",
  mi: "Mutual Information (MI) estimates measure how much information flows through the model. mi_input_repr: how much input info the hidden layers capture. mi_repr_output: how much the representations predict the output. mi_compression: the ratio, tracking information bottleneck behavior.",
  batchSize: "When adaptive batch sizing is enabled, this shows the dynamically adjusted batch size. It differs from the training config batch size (shown above) when CUSUM detects instability. The config batch size is the baseline; adaptive batch varies around it in response to training dynamics.",
  lineageTree: "Evolutionary tree showing parent-child relationships between candidates across generations. Each node is a candidate with its activation type color-coded. The tree grows upward from generation 0 (roots) to the latest generation. Click any node to see detailed metrics. Nodes are named by lineage: e.g., 'S-Alpha' = SiLU, generation 0; 'G-Alpha.1' = GELU mutation of Alpha, generation 1.",
};

// ── Shared Helpers ────────────────────────────────────────────

function binarySearchStep(data: { step: number }[], target: number): number {
  let lo = 0, hi = data.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (data[mid].step < target) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(data[lo - 1].step - target) < Math.abs(data[lo].step - target)) lo--;
  return lo;
}

function ChartTooltipDiv({ tooltip }: { tooltip: ChartTooltip }) {
  const left = tooltip.pointX < tooltip.containerWidth * 0.65;
  return (
    <div
      className="pointer-events-none absolute z-20 min-w-[160px] rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl"
      style={{
        left: left ? tooltip.pointX + 12 : undefined,
        right: !left ? tooltip.containerWidth - tooltip.pointX + 12 : undefined,
        top: Math.max(4, tooltip.mouseY - 60),
      }}
    >
      <div className="mb-1 font-mono text-[0.68rem] font-bold text-white">Step {tooltip.step.toLocaleString()}</div>
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

// ── Shared Recharts Theme ────────────────────────────────────

const CHART_THEME = {
  bg: "#0d0d0d",
  grid: "#1a1a1a",
  text: "#555",
  axisText: "#666",
};

function CustomTooltipContent({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
      <div className="mb-1 font-mono font-bold text-white">Step {Number(label).toLocaleString()}</div>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex justify-between gap-3">
          <span className="text-text-muted">{p.name}</span>
          <span className="font-mono" style={{ color: p.color }}>{typeof p.value === "number" ? p.value.toFixed(4) : String(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

// ── Symbio Stats Grid ────────────────────────────────────────

export function SymbioStatsGrid({ metrics }: { metrics: SymbioMetric[] }) {
  const symbioMetrics = metrics.filter(m => m.weight_entropy != null || m.cusum_grad != null);
  if (symbioMetrics.length === 0) return null;

  const lastWithEntropy = [...metrics].reverse().find(m => m.weight_entropy != null);
  const lastWithRank = [...metrics].reverse().find(m => m.effective_rank != null);
  const lastWithFE = [...metrics].reverse().find(m => m.free_energy != null);
  const lastWithPE = [...metrics].reverse().find(m => m.population_entropy != null);
  const lastWithFit = [...metrics].reverse().find(m => m.fitness_score != null);
  const lastWithComplex = [...metrics].reverse().find(m => m.complexity_score != null);
  const lastWithBatch = [...metrics].reverse().find(m => m.adaptive_batch_size != null);
  const cusumAlerts = metrics.filter(m => (m.cusum_alerts ?? 0) > 0).length;

  return (
    <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
      <Stat label="Wt Entropy" value={lastWithEntropy?.weight_entropy?.toFixed(2) ?? "-"} sub="bits" color="text-purple-400" />
      <Stat label="Eff. Rank" value={lastWithRank?.effective_rank?.toFixed(1) ?? "-"} color="text-amber-400" />
      <Stat label="Free Energy" value={lastWithFE?.free_energy?.toFixed(4) ?? "-"} color="text-green" />
      <Stat label="Pop Entropy" value={lastWithPE?.population_entropy?.toFixed(3) ?? "-"} sub="nats" color="text-cyan-400" />
      <Stat label="Complexity" value={lastWithComplex?.complexity_score?.toFixed(4) ?? "-"} color="text-rose-400" />
      <Stat label="Fitness" value={lastWithFit?.fitness_score?.toFixed(4) ?? "-"} color="text-blue" />
      <Stat label="CUSUM Alerts" value={String(cusumAlerts)} sub={`of ${metrics.length} steps`} color={cusumAlerts > 10 ? "text-red" : cusumAlerts > 0 ? "text-yellow" : "text-green"} />
      <Stat label="Batch Size" value={lastWithBatch?.adaptive_batch_size?.toFixed(0) ?? "-"} sub="adaptive" color="text-teal-400" />
    </div>
  );
}

// ── CUSUM Chart (Recharts) ────────────────────────────────────

function CusumChart({ metrics, sensitivity, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[]; sensitivity: number;
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const data = useMemo(() => metrics
    .filter(m => m.cusum_grad != null || m.cusum_clip != null || m.cusum_tps != null || m.cusum_val != null)
    .map(m => ({
      step: m.step,
      grad: m.cusum_grad ?? 0,
      clip: m.cusum_clip ?? 0,
      tps: m.cusum_tps ?? 0,
      val: m.cusum_val ?? 0,
      alert: (m.cusum_alerts ?? 0) > 0,
    })),
  [metrics]);
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
        <RTooltip content={<CustomTooltipContent />} />
        <ReferenceLine y={sensitivity} stroke="#ef4444" strokeDasharray="6 3" label={{ value: `threshold=${sensitivity}`, position: "right", fontSize: 9, fill: "#ef4444" }} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <Line type="monotone" dataKey="grad" name="Gradient" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="clip" name="Clipping" stroke="#f472b6" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="tps" name="Throughput" stroke="#34d399" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="val" name="Val Loss" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
        <Legend wrapperStyle={{ fontSize: "10px" }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Clip Telemetry Chart (Recharts) ──────────────────────────

function ClipChart({ metrics, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const data = useMemo(() => metrics
    .filter(m => m.clip_coef != null)
    .map(m => ({ step: m.step, coef: m.clip_coef!, pct: (m.clip_pct ?? 0) * 100 })),
  [metrics]);
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
        <RTooltip content={<CustomTooltipContent />} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <Area type="monotone" dataKey="coef" name="Clip Coef" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={1.5} />
        <Area type="monotone" dataKey="pct" name="Clip %" stroke="#f472b6" fill="#f472b6" fillOpacity={0.1} strokeWidth={1.5} />
        <Legend wrapperStyle={{ fontSize: "10px" }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Sparse Line Chart (Recharts) ─────────────────────────────

function SparseLineChart({ metrics, getY, color, label, format, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  getY: (m: SymbioMetric) => number | null | undefined;
  color: string; label: string;
  format?: (v: number) => string;
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const data = useMemo(() => {
    const pts: { step: number; value: number }[] = [];
    for (const m of metrics) { const v = getY(m); if (v != null) pts.push({ step: m.step, value: v }); }
    return pts;
  }, [metrics, getY]);
  if (data.length === 0) return <div className="flex h-[180px] items-center justify-center text-xs text-text-muted">No {label} data</div>;

  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={format ?? ((v: number) => v.toFixed(2))} />
        <RTooltip content={<CustomTooltipContent />} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <Area type="monotone" dataKey="value" name={label} stroke={color} fill={color} fillOpacity={0.12} strokeWidth={2} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Adaptive Batch Size Chart ────────────────────────────────

function AdaptiveBatchChart({ metrics, configBatchSize, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  configBatchSize?: number;
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const data = useMemo(() => metrics
    .filter(m => m.adaptive_batch_size != null)
    .map(m => ({
      step: m.step,
      batch: m.adaptive_batch_size!,
      reason: m.batch_change_reason ?? "",
    })),
  [metrics]);
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
        <RTooltip content={({ active, payload, label }: any) => {
          if (!active || !payload?.[0]) return null;
          const d = payload[0].payload;
          return (
            <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
              <div className="mb-1 font-mono font-bold text-white">Step {Number(label).toLocaleString()}</div>
              <div className="flex justify-between gap-3"><span className="text-text-muted">Batch Size</span><span className="font-mono text-green">{d.batch}</span></div>
              {d.reason && <div className="flex justify-between gap-3"><span className="text-text-muted">Reason</span><span className="font-mono text-orange-400">{d.reason}</span></div>}
              {configBatchSize != null && <div className="flex justify-between gap-3"><span className="text-text-muted">Config</span><span className="font-mono text-text-muted">{configBatchSize}</span></div>}
            </div>
          );
        }} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        {configBatchSize != null && <ReferenceLine y={configBatchSize} stroke="#666" strokeDasharray="6 3" label={{ value: "config", position: "right", fontSize: 9, fill: "#666" }} />}
        <Area type="stepAfter" dataKey="batch" name="Batch Size" stroke="#34d399" fill="#34d399" fillOpacity={0.15} strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── MI Profiles Chart ────────────────────────────────────────

function MIProfilesChart({ metrics, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  pinnedStep?: number | null; onPinStep?: (s: number) => void;
}) {
  const data = useMemo(() => metrics
    .filter(m => m.mi_input_repr != null)
    .map(m => ({
      step: m.step,
      input: m.mi_input_repr!,
      output: m.mi_repr_output!,
      compression: m.mi_compression!,
    })),
  [metrics]);
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
        <RTooltip content={<CustomTooltipContent />} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <Line type="monotone" dataKey="input" name="MI Input→Repr" stroke="#60a5fa" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="output" name="MI Repr→Output" stroke="#f59e0b" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="compression" name="Compression" stroke="#a78bfa" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
        <Legend wrapperStyle={{ fontSize: "10px" }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Candidate Stats Extraction ──────────────────────────────

interface CandidateStats {
  id: string;
  name: string;
  activation: string;
  generation: number;
  parentId: string | null;
  parentName: string | null;
  steps: number;
  bestLoss: number;
  bestValLoss: number;
  avgLoss: number;
  bestFitness: number;
  avgTps: number;
  cusumAlerts: number;
  lastClipPct: number | null;
  startStep: number;
  endStep: number;
}

function extractCandidateStats(metrics: SymbioMetric[]): CandidateStats[] {
  const candidates = new Map<string, {
    name: string; activation: string; generation: number;
    parentId: string | null; parentName: string | null;
    losses: number[]; valLosses: number[];
    fitnesses: number[]; tps: number[]; steps: number; cusumAlerts: number;
    lastClipPct: number | null; startStep: number; endStep: number;
  }>();

  for (const m of metrics) {
    const id = m.symbio_candidate_id;
    if (!id) continue;
    let entry = candidates.get(id);
    if (!entry) {
      entry = {
        name: m.symbio_candidate_name ?? id,
        activation: m.symbio_candidate_activation ?? "?", generation: m.symbio_generation ?? 0,
        parentId: m.symbio_candidate_parent_id ?? null,
        parentName: m.symbio_candidate_parent_name ?? null,
        losses: [], valLosses: [], fitnesses: [], tps: [], steps: 0, cusumAlerts: 0,
        lastClipPct: null, startStep: m.step, endStep: m.step,
      };
      candidates.set(id, entry);
    }
    entry.losses.push(m.loss);
    if (m.val_loss != null) entry.valLosses.push(m.val_loss);
    if (m.fitness_score != null) entry.fitnesses.push(m.fitness_score);
    if (m.tokens_per_sec != null) entry.tps.push(m.tokens_per_sec);
    if ((m.cusum_alerts ?? 0) > 0) entry.cusumAlerts++;
    if (m.clip_pct != null) entry.lastClipPct = m.clip_pct;
    entry.endStep = m.step;
    entry.steps++;
  }

  return Array.from(candidates.entries()).map(([id, e]) => ({
    id,
    name: e.name,
    activation: e.activation,
    generation: e.generation,
    parentId: e.parentId,
    parentName: e.parentName,
    steps: e.steps,
    bestLoss: Math.min(...e.losses),
    bestValLoss: e.valLosses.length > 0 ? Math.min(...e.valLosses) : Infinity,
    avgLoss: e.losses.reduce((a, b) => a + b, 0) / e.losses.length,
    bestFitness: e.fitnesses.length > 0 ? Math.max(...e.fitnesses) : -Infinity,
    avgTps: e.tps.length > 0 ? e.tps.reduce((a, b) => a + b, 0) / e.tps.length : 0,
    cusumAlerts: e.cusumAlerts,
    lastClipPct: e.lastClipPct,
    startStep: e.startStep,
    endStep: e.endStep,
  }));
}

// ── Switch Event Extraction ─────────────────────────────────

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

// ── Extract Activation Switch Events for Loss Chart ─────────

export function extractActivationSwitchEvents(metrics: SymbioMetric[]): ActivationSwitchEvent[] {
  const events: ActivationSwitchEvent[] = [];
  let prevId: string | null = null;
  let prevActivation: string | null = null;

  for (const m of metrics) {
    const id = m.symbio_candidate_id;
    if (!id) continue;
    if (id !== prevId) {
      events.push({
        step: m.step,
        fromActivation: prevActivation,
        toActivation: m.symbio_candidate_activation ?? "?",
        toGeneration: m.symbio_generation ?? 0,
        toCandidateId: id,
        lossAtSwitch: m.loss,
      });
      prevId = id;
      prevActivation = m.symbio_candidate_activation ?? "?";
    }
  }
  return events;
}

// ── Search Candidate Table (Enhanced) ────────────────────────

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
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border/50 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
            <th className="px-3 py-2 text-left">#</th>
            <th className="px-3 py-2 text-left">Name</th>
            <th className="px-3 py-2 text-left">Activation</th>
            <th className="px-3 py-2 text-center">Gen</th>
            <th className="px-3 py-2 text-left">Parent</th>
            <th className="px-3 py-2 text-right">Steps</th>
            <th className="px-3 py-2 text-right">Best Loss</th>
            <th className="px-3 py-2 text-right">Best Val</th>
            <th className="px-3 py-2 text-right">Avg Loss</th>
            <th className="px-3 py-2 text-right">Fitness</th>
            <th className="px-3 py-2 text-right">Avg tok/s</th>
            <th className="px-3 py-2 text-right">Alerts</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => (
            <tr key={c.id} className={`border-b border-border/20 last:border-0 ${c.id === bestId ? "bg-green-500/5" : ""}`}>
              <td className="px-3 py-2 font-mono font-semibold text-text-muted">{i + 1}</td>
              <td className="px-3 py-2 truncate font-mono text-text-secondary" title={`${c.name} (${c.id})`}>{c.name}</td>
              <td className="px-3 py-2">
                <span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[c.activation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[c.activation] ?? "text-text-secondary"}`}>{c.activation}</span>
              </td>
              <td className="px-3 py-2 text-center text-text-muted">{c.generation}</td>
              <td className="px-3 py-2 text-text-muted text-[0.6rem]">{c.parentName ?? "-"}</td>
              <td className="px-3 py-2 text-right font-mono text-text-secondary">{c.steps}</td>
              <td className="px-3 py-2 text-right font-mono text-white">{c.bestLoss.toFixed(4)}</td>
              <td className="px-3 py-2 text-right font-mono text-blue-400">{c.bestValLoss === Infinity ? "-" : c.bestValLoss.toFixed(4)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-secondary">{c.avgLoss.toFixed(4)}</td>
              <td className="px-3 py-2 text-right font-mono text-green">{c.bestFitness === -Infinity ? "-" : c.bestFitness.toFixed(4)}</td>
              <td className="px-3 py-2 text-right font-mono text-text-secondary">{c.avgTps > 0 ? Math.round(c.avgTps).toLocaleString() : "-"}</td>
              <td className="px-3 py-2 text-right">
                {c.cusumAlerts > 0
                  ? <span className="rounded bg-orange-500/10 px-1.5 py-0.5 text-[0.58rem] text-orange-400">{c.cusumAlerts}</span>
                  : <span className="text-text-muted">0</span>
                }
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Activation Switch Log (Expandable) ──────────────────────

function ActivationSwitchLog({ metrics, onNavigateToTree }: { metrics: SymbioMetric[]; onNavigateToTree?: (candidateId: string) => void }) {
  const events = extractSwitchEvents(metrics);
  const candidates = useMemo(() => extractCandidateStats(metrics), [metrics]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (events.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border/50 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
            <th className="px-3 py-2 text-left">Step</th>
            <th className="px-3 py-2 text-left">From</th>
            <th className="px-3 py-2 text-center"></th>
            <th className="px-3 py-2 text-left">To</th>
            <th className="px-3 py-2 text-center">Gen</th>
            <th className="px-3 py-2 text-right">Prev Steps</th>
            <th className="px-3 py-2 text-right">Best Loss</th>
            <th className="px-3 py-2 text-right">Final Loss</th>
            <th className="px-3 py-2 text-right">Fitness</th>
            <th className="px-3 py-2 text-center">Tree</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => {
            const isExpanded = expandedIdx === i;
            const fromCandidate = e.fromId ? candidates.find(c => c.id === e.fromId) : null;
            const toCandidate = candidates.find(c => c.id === e.toId);
            return (
              <Fragment key={i}>
                <tr
                  className={`border-b border-border/20 cursor-pointer transition-colors hover:bg-surface-2/50 ${i === 0 ? "bg-surface-2/30" : ""} ${isExpanded ? "bg-purple-500/5" : ""}`}
                  onClick={() => setExpandedIdx(isExpanded ? null : i)}
                >
                  <td className="px-3 py-2 font-mono font-semibold text-white">{e.step}</td>
                  <td className="px-3 py-2">{e.fromActivation ? <span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[e.fromActivation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[e.fromActivation] ?? "text-text-secondary"}`}>{e.fromActivation}</span> : <span className="text-text-muted">-</span>}</td>
                  <td className="px-1 py-2 text-center text-text-muted">&rarr;</td>
                  <td className="px-3 py-2"><span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[e.toActivation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[e.toActivation] ?? "text-text-secondary"}`}>{e.toActivation}</span></td>
                  <td className="px-3 py-2 text-center text-text-muted">{e.toGeneration}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromSteps > 0 ? e.fromSteps : "-"}</td>
                  <td className="px-3 py-2 text-right font-mono text-white">{e.fromBestLoss != null ? e.fromBestLoss.toFixed(4) : "-"}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-secondary">{e.fromFinalLoss != null ? e.fromFinalLoss.toFixed(4) : "-"}</td>
                  <td className="px-3 py-2 text-right font-mono text-green">{e.fromBestFitness != null ? e.fromBestFitness.toFixed(4) : "-"}</td>
                  <td className="px-3 py-2 text-center">
                    <button
                      className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[0.56rem] text-purple-400 hover:bg-purple-500/30 transition-colors"
                      onClick={(ev) => { ev.stopPropagation(); if (onNavigateToTree) onNavigateToTree(e.toId); }}
                      title="Navigate to this candidate in the lineage tree"
                    >
                      tree
                    </button>
                  </td>
                </tr>
                {isExpanded && (
                  <tr className="bg-purple-500/5 border-b border-purple-500/20">
                    <td colSpan={10} className="px-4 py-3">
                      <div className="grid grid-cols-2 gap-6 text-[0.64rem]">
                        {/* From candidate details */}
                        <div>
                          <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
                            Outgoing: {e.fromId ?? "-"}
                          </div>
                          <div className="space-y-1">
                            <div className="flex justify-between"><span className="text-text-muted">Activation</span><span className="font-mono">{e.fromActivation ?? "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Generation</span><span className="font-mono">{e.fromGeneration ?? "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Steps Trained</span><span className="font-mono">{e.fromSteps || "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Best Loss</span><span className="font-mono text-white">{e.fromBestLoss?.toFixed(4) ?? "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Final Loss</span><span className="font-mono">{e.fromFinalLoss?.toFixed(4) ?? "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Best Fitness</span><span className="font-mono text-green">{e.fromBestFitness?.toFixed(4) ?? "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Avg tok/s</span><span className="font-mono">{e.fromAvgTokPerSec != null ? Math.round(e.fromAvgTokPerSec).toLocaleString() : "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Clip % (end)</span><span className="font-mono">{e.fromClipPctAtEnd != null ? `${e.fromClipPctAtEnd.toFixed(0)}%` : "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">CUSUM Alerts</span><span className="font-mono">{e.fromCusumAlerts > 0 ? <span className="text-orange-400">{e.fromCusumAlerts}</span> : "-"}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Grad Spikes</span><span className="font-mono">{e.fromGradSpikes > 0 ? <span className="text-red-400">{e.fromGradSpikes}</span> : "-"}</span></div>
                            {e.fromLastAlertReason && <div className="mt-1 text-[0.58rem] text-orange-400">{e.fromLastAlertReason}</div>}
                          </div>
                        </div>
                        {/* To candidate details */}
                        <div>
                          <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
                            Incoming: {e.toId}
                          </div>
                          <div className="space-y-1">
                            <div className="flex justify-between"><span className="text-text-muted">Activation</span><span className={`font-mono font-semibold ${ACTIVATION_COLORS[e.toActivation] ?? "text-text-secondary"}`}>{e.toActivation}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Generation</span><span className="font-mono">{e.toGeneration}</span></div>
                            <div className="flex justify-between"><span className="text-text-muted">Loss @ Switch</span><span className="font-mono text-yellow">{e.lossAtSwitch.toFixed(4)}</span></div>
                            {toCandidate && (
                              <>
                                <div className="flex justify-between"><span className="text-text-muted">Steps (total)</span><span className="font-mono">{toCandidate.steps}</span></div>
                                <div className="flex justify-between"><span className="text-text-muted">Best Loss</span><span className="font-mono text-white">{toCandidate.bestLoss.toFixed(4)}</span></div>
                                <div className="flex justify-between"><span className="text-text-muted">Best Val</span><span className="font-mono text-blue-400">{toCandidate.bestValLoss === Infinity ? "-" : toCandidate.bestValLoss.toFixed(4)}</span></div>
                                <div className="flex justify-between"><span className="text-text-muted">Best Fitness</span><span className="font-mono text-green">{toCandidate.bestFitness === -Infinity ? "-" : toCandidate.bestFitness.toFixed(4)}</span></div>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Evolutionary Metrics Section ─────────────────────────────

function EvolutionaryTimeline({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  const candidates = useMemo(() => extractCandidateStats(metrics), [metrics]);
  if (candidates.length === 0) return null;

  // Group by generation
  const generations = new Map<number, CandidateStats[]>();
  for (const c of candidates) {
    const gen = generations.get(c.generation) ?? [];
    gen.push(c);
    generations.set(c.generation, gen);
  }

  const genEntries = Array.from(generations.entries()).sort((a, b) => a[0] - b[0]);

  // Fitness over time (by candidate evaluation order)
  const fitnessData: { step: number; fitness: number; activation: string; id: string; gen: number }[] = [];
  for (const c of candidates.sort((a, b) => a.startStep - b.startStep)) {
    if (c.bestFitness !== -Infinity) {
      fitnessData.push({ step: c.startStep, fitness: c.bestFitness, activation: c.activation, id: c.id, gen: c.generation });
    }
  }

  // Architecture convergence over time
  const diversityData = metrics
    .filter(m => m.architecture_diversity != null)
    .map(m => ({ step: m.step, diversity: m.architecture_diversity! }));

  // Loss trajectory per candidate (for the evolution chart)
  const lossTrajectories: { step: number; loss: number; activation: string; id: string }[] = [];
  for (const m of metrics) {
    if (m.symbio_candidate_id) {
      lossTrajectories.push({ step: m.step, loss: m.loss, activation: m.symbio_candidate_activation ?? "?", id: m.symbio_candidate_id });
    }
  }

  return (
    <div className="space-y-4">
      {/* Generation Population Cards */}
      <div className="space-y-3">
        {genEntries.map(([gen, cands]) => {
          const sortedCands = [...cands].sort((a, b) => a.bestLoss - b.bestLoss);
          const best = sortedCands[0];
          return (
            <div key={gen} className="rounded-lg border border-border/60 bg-surface-2/30 p-3">
              <div className="mb-2 flex items-center gap-2">
                <span className="rounded bg-surface-2 px-2 py-0.5 text-[0.62rem] font-bold text-text-secondary">Generation {gen}</span>
                <span className="text-[0.6rem] text-text-muted">{cands.length} candidates</span>
                {best && <span className={`rounded border px-1.5 py-0.5 text-[0.58rem] font-semibold ${ACTIVATION_BG[best.activation] ?? ""} ${ACTIVATION_COLORS[best.activation] ?? "text-text-secondary"}`}>Best: {best.activation} ({best.bestLoss.toFixed(4)})</span>}
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                {sortedCands.map((c, i) => (
                  <div key={c.id} className={`rounded border p-2 text-[0.62rem] ${i === 0 ? "border-green-500/30 bg-green-500/5" : "border-border/30 bg-surface-2/20"}`}>
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className={`font-semibold ${ACTIVATION_COLORS[c.activation] ?? "text-text-secondary"}`}>{c.activation}</span>
                      {i === 0 && <span className="text-[0.5rem] text-green">BEST</span>}
                    </div>
                    <div className="space-y-0.5 text-text-muted">
                      <div className="flex justify-between"><span>Loss</span><span className="font-mono text-white">{c.bestLoss.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span>Fitness</span><span className="font-mono text-green">{c.bestFitness === -Infinity ? "-" : c.bestFitness.toFixed(4)}</span></div>
                      <div className="flex justify-between"><span>Steps</span><span className="font-mono">{c.steps}</span></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Fitness Progression Chart */}
      {fitnessData.length > 1 && (
        <ChartPanel title="Fitness Progression" helpText={HELP.evolution}>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
              <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
              <XAxis dataKey="step" name="Step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
              <YAxis dataKey="fitness" name="Fitness" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
              <RTooltip content={({ active, payload }: any) => {
                if (!active || !payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
                    <div className="font-mono font-bold text-white">{d.id}</div>
                    <div className="flex justify-between gap-3"><span className="text-text-muted">Activation</span><span className={`font-mono ${ACTIVATION_COLORS[d.activation] ?? ""}`}>{d.activation}</span></div>
                    <div className="flex justify-between gap-3"><span className="text-text-muted">Generation</span><span className="font-mono">{d.gen}</span></div>
                    <div className="flex justify-between gap-3"><span className="text-text-muted">Fitness</span><span className="font-mono text-green">{d.fitness.toFixed(4)}</span></div>
                  </div>
                );
              }} />
              {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
              <Scatter data={fitnessData} fill="#34d399">
                {fitnessData.map((d, i) => (
                  <Cell key={i} fill={actHex(d.activation)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartPanel>
      )}

      {/* Architecture Diversity Over Time */}
      {diversityData.length > 1 && (
        <ChartPanel title="Architecture Diversity" helpText={HELP.diversity}>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={diversityData} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
              <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
              <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
              <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
              <RTooltip content={({ active, payload, label }: any) => {
                if (!active || !payload?.[0]) return null;
                return (
                  <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
                    <div className="font-mono font-bold text-white">Step {Number(label).toLocaleString()}</div>
                    <div className="flex justify-between gap-3"><span className="text-text-muted">Diversity</span><span className="font-mono text-cyan-400">{(payload[0].value * 100).toFixed(1)}%</span></div>
                  </div>
                );
              }} />
              {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
              <Area type="monotone" dataKey="diversity" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.15} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel>
      )}
    </div>
  );
}

// ── Phase Change / Gelation Visualization ────────────────────

function PhaseChangeTimeline({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  // Aggregate alert density over windows of 50 steps
  const windowSize = 50;
  const data: { step: number; alertDensity: number; channels: number; reason: string }[] = [];
  for (let i = 0; i < metrics.length; i += windowSize) {
    const window = metrics.slice(i, i + windowSize);
    const alerts = window.filter(m => (m.cusum_alerts ?? 0) > 0);
    const density = alerts.length / window.length;
    const allChannels = alerts.reduce((s, m) => s | (m.cusum_alerts ?? 0), 0);
    const channelCount = [allChannels & 1, (allChannels >> 1) & 1, (allChannels >> 2) & 1, (allChannels >> 3) & 1].reduce((a, b) => a + b, 0);
    const reasons = [...new Set(alerts.map(m => m.cusum_alert_reason).filter(Boolean))].join(", ");
    data.push({ step: window[0].step, alertDensity: density, channels: channelCount, reason: reasons });
  }
  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
        <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
        <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
        <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <RTooltip content={({ active, payload, label }: any) => {
          if (!active || !payload?.[0]) return null;
          const d = payload[0].payload;
          return (
            <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
              <div className="font-mono font-bold text-white">Step {Number(label).toLocaleString()}</div>
              <div className="flex justify-between gap-3"><span className="text-text-muted">Alert Rate</span><span className="font-mono">{(d.alertDensity * 100).toFixed(0)}%</span></div>
              <div className="flex justify-between gap-3"><span className="text-text-muted">Channels</span><span className="font-mono">{d.channels}/4</span></div>
              {d.reason && <div className="mt-1 text-[0.58rem] text-orange-400">{d.reason}</div>}
            </div>
          );
        }} />
        {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
        <Bar dataKey="alertDensity" name="Alert Density">
          {data.map((d, i) => (
            <Cell key={i} fill={d.alertDensity > 0.5 ? "#ef4444" : d.alertDensity > 0.2 ? "#f59e0b" : d.alertDensity > 0 ? "#facc15" : "#34d399"} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Harmonic Oscillator Analysis ─────────────────────────────

function HarmonicOscillatorChart({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  // Compute loss oscillation around moving average
  const windowSize = 20;
  if (metrics.length < windowSize * 2) return null;

  const data: { step: number; deviation: number; ma: number; amplitude: number }[] = [];
  for (let i = windowSize; i < metrics.length; i++) {
    let sum = 0;
    for (let j = i - windowSize; j < i; j++) sum += metrics[j].loss;
    const ma = sum / windowSize;
    const deviation = metrics[i].loss - ma;
    data.push({ step: metrics[i].step, deviation, ma, amplitude: Math.abs(deviation) });
  }

  // Compute rolling amplitude envelope
  const ampWindow = 10;
  const envelope: { step: number; amplitude: number }[] = [];
  for (let i = ampWindow; i < data.length; i++) {
    let maxAmp = 0;
    for (let j = i - ampWindow; j < i; j++) maxAmp = Math.max(maxAmp, data[j].amplitude);
    envelope.push({ step: data[i].step, amplitude: maxAmp });
  }

  const handleClick = (e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); };

  return (
    <div className="space-y-3">
      {/* Oscillation */}
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} onClick={handleClick}>
          <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
          <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
          <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(3)} />
          <RTooltip content={({ active, payload, label }: any) => {
            if (!active || !payload?.[0]) return null;
            const d = payload[0].payload;
            return (
              <div className="rounded-lg border border-border-2 bg-surface-2 p-2.5 shadow-xl text-[0.64rem]">
                <div className="font-mono font-bold text-white">Step {Number(label).toLocaleString()}</div>
                <div className="flex justify-between gap-3"><span className="text-text-muted">Deviation</span><span className="font-mono text-cyan-400">{d.deviation.toFixed(4)}</span></div>
                <div className="flex justify-between gap-3"><span className="text-text-muted">Moving Avg</span><span className="font-mono text-text-secondary">{d.ma.toFixed(4)}</span></div>
              </div>
            );
          }} />
          <ReferenceLine y={0} stroke="#555" />
          {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
          <Area type="monotone" dataKey="deviation" name="Oscillation" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.1} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>

      {/* Amplitude envelope - damping analysis */}
      {envelope.length > 2 && (
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={envelope} onClick={handleClick}>
            <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
            <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
            {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
            <Area type="monotone" dataKey="amplitude" name="Amplitude Envelope" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── Activation Distribution Over Time ────────────────────────

function ActivationDistributionChart({ metrics, pinnedStep, onPinStep }: { metrics: SymbioMetric[]; pinnedStep?: number | null; onPinStep?: (s: number) => void }) {
  // Build a time series of activation distribution
  const data: { step: number; [k: string]: number }[] = [];
  for (const m of metrics) {
    if (!m.activation_distribution) continue;
    try {
      const dist = JSON.parse(m.activation_distribution) as Record<string, number>;
      const total = Object.values(dist).reduce((a, b) => a + b, 0);
      const entry: Record<string, number> = { step: m.step };
      for (const [act, count] of Object.entries(dist)) {
        entry[act] = total > 0 ? count / total : 0;
      }
      data.push(entry as any);
    } catch { /* skip bad json */ }
  }

  // Also build a cumulative step count per activation
  const stepCounts = new Map<string, number>();
  for (const m of metrics) {
    const act = m.symbio_candidate_activation;
    if (act) stepCounts.set(act, (stepCounts.get(act) ?? 0) + 1);
  }
  const totalSteps = Array.from(stepCounts.values()).reduce((a, b) => a + b, 0);
  const barData = Array.from(stepCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([act, count]) => ({ activation: act, steps: count, pct: totalSteps > 0 ? count / totalSteps : 0 }));

  const activations = Array.from(new Set(metrics.map(m => m.symbio_candidate_activation).filter(Boolean))) as string[];

  return (
    <div className="space-y-3">
      {/* Step allocation bar chart */}
      {barData.length > 0 && (
        <div className="space-y-1.5">
          {barData.map(d => (
            <div key={d.activation} className="flex items-center gap-2">
              <span className={`w-14 text-right text-[0.62rem] font-semibold ${ACTIVATION_COLORS[d.activation] ?? "text-text-secondary"}`}>{d.activation}</span>
              <div className="flex-1 h-4 rounded bg-surface-2 overflow-hidden">
                <div
                  className="h-full rounded transition-all"
                  style={{ width: `${d.pct * 100}%`, backgroundColor: actHex(d.activation), opacity: 0.7 }}
                />
              </div>
              <span className="w-16 text-right font-mono text-[0.6rem] text-text-muted">{d.steps} ({(d.pct * 100).toFixed(0)}%)</span>
            </div>
          ))}
        </div>
      )}

      {/* Stacked area of distribution over time */}
      {data.length > 2 && activations.length > 0 && (
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={data} onClick={(e: any) => { if (e?.activeLabel != null && onPinStep) onPinStep(Number(e.activeLabel)); }}>
            <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
            <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <RTooltip content={<CustomTooltipContent />} />
            {pinnedStep != null && <ReferenceLine x={pinnedStep} stroke="rgba(168,85,247,0.7)" strokeWidth={1.5} />}
            {activations.map(act => (
              <Area key={act} type="monotone" dataKey={act} stackId="1" stroke={actHex(act)} fill={actHex(act)} fillOpacity={0.3} strokeWidth={0} />
            ))}
            <Legend wrapperStyle={{ fontSize: "10px" }} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── Fragment import ──────────────────────────────────────────
import { Fragment } from "react";

// ── Amcharts Evolutionary Lineage Tree ───────────────────────

function EvolutionaryTreeChart({ metrics, selectedCandidateId, onSelectCandidate }: {
  metrics: SymbioMetric[];
  selectedCandidateId?: string | null;
  onSelectCandidate?: (id: string) => void;
}) {
  const chartRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<any>(null);
  const candidates = useMemo(() => extractCandidateStats(metrics), [metrics]);

  useEffect(() => {
    if (!chartRef.current || candidates.length === 0) return;
    let root: any;

    // Dynamic import amcharts (client-side only)
    const initChart = async () => {
      const am5 = await import("@amcharts/amcharts5");
      const am5hierarchy = await import("@amcharts/amcharts5/hierarchy");
      const am5themes = await import("@amcharts/amcharts5/themes/Dark");

      // Dispose previous instance
      if (rootRef.current) rootRef.current.dispose();

      if (!chartRef.current) return;
      root = am5.Root.new(chartRef.current);
      rootRef.current = root;
      root.setThemes([am5themes.default.new(root)]);

      const container = root.container.children.push(
        am5.Container.new(root, { width: am5.percent(100), height: am5.percent(100), layout: root.verticalLayout })
      );

      const series = container.children.push(
        am5hierarchy.ForceDirected.new(root, {
          singleBranchOnly: false,
          downDepth: 10,
          topDepth: 1,
          initialDepth: 10,
          valueField: "value",
          categoryField: "name",
          childDataField: "children",
          idField: "id",
          linkWithStrength: 0.5,
          minRadius: 16,
          maxRadius: 40,
          manyBodyStrength: -15,
          centerStrength: 0.5,
        })
      );

      series.get("colors")!.set("colors", [
        am5.color("#60a5fa"), am5.color("#34d399"), am5.color("#f59e0b"),
        am5.color("#a78bfa"), am5.color("#f472b6"), am5.color("#22d3ee"),
      ]);

      // Build tree data from candidates
      const rootNodes: any[] = [];
      const nodeMap = new Map<string, any>();

      // Create nodes for all candidates
      for (const c of candidates) {
        const node: any = {
          id: c.id,
          name: c.name,
          value: Math.max(1, c.steps),
          activation: c.activation,
          generation: c.generation,
          bestLoss: c.bestLoss,
          bestFitness: c.bestFitness,
          parentId: c.parentId,
          children: [],
          nodeSettings: {
            fill: am5.color(actHex(c.activation)),
          },
        };
        nodeMap.set(c.id, node);
      }

      // Build parent-child relationships
      for (const c of candidates) {
        const node = nodeMap.get(c.id);
        if (!node) continue;
        if (c.parentId && nodeMap.has(c.parentId)) {
          nodeMap.get(c.parentId)!.children.push(node);
        } else {
          rootNodes.push(node);
        }
      }

      // Wrap in a virtual root
      const treeData = {
        id: "root",
        name: "Evolution",
        value: 0,
        children: rootNodes,
      };

      series.data.setAll([treeData]);
      series.set("selectedDataItem", series.dataItems[0]);

      // Configure node appearance
      series.nodes.template.setAll({ toggleKey: "none", cursorOverStyle: "pointer" });

      series.nodes.template.setup = (target: any) => {
        const circle = target.children.getIndex(0);
        if (circle) {
          circle.setAll({ strokeWidth: 2, stroke: am5.color("#333") });
        }
      };

      // Custom node colors based on activation
      series.nodes.template.adapters.add("fill", (_fill: any, target: any) => {
        const dataItem = target.dataItem;
        if (dataItem && dataItem.dataContext) {
          const ctx = dataItem.dataContext as any;
          return am5.color(actHex(ctx.activation));
        }
        return _fill;
      });

      // Tooltip
      series.nodes.template.set("tooltipText", "[bold]{name}[/]\nActivation: {activation}\nGen: {generation}\nBest Loss: {bestLoss}\nFitness: {bestFitness}");

      // Click handler
      series.nodes.template.events.on("click", (ev: any) => {
        const dataItem = ev.target.dataItem;
        if (dataItem?.dataContext && onSelectCandidate) {
          onSelectCandidate((dataItem.dataContext as any).id);
        }
      });

      // Links styling
      series.links.template.setAll({ strokeWidth: 1.5, strokeOpacity: 0.4 });

      // Labels
      series.labels.template.setAll({ fontSize: 9, fill: am5.color("#ccc"), oversizedBehavior: "truncate", maxWidth: 80 });

      series.appear(1000, 100);
    };

    initChart();

    return () => { if (rootRef.current) { rootRef.current.dispose(); rootRef.current = null; } };
  }, [candidates, onSelectCandidate]);

  if (candidates.length === 0) return null;

  return <div ref={chartRef} style={{ width: "100%", height: 400 }} />;
}

// ── Traditional Tree Layout (Canvas) ─────────────────────────

interface TreeNode {
  id: string;
  name: string;
  activation: string;
  generation: number;
  bestLoss: number;
  bestFitness: number;
  steps: number;
  children: TreeNode[];
  x: number;
  y: number;
  width: number;
}

function LineageTreeChart({ metrics, selectedCandidateId, onSelectCandidate }: {
  metrics: SymbioMetric[];
  selectedCandidateId?: string | null;
  onSelectCandidate?: (id: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<TreeNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const candidates = useMemo(() => extractCandidateStats(metrics), [metrics]);

  const { roots, allNodes, maxGen, treeHeight } = useMemo(() => {
    if (candidates.length === 0) return { roots: [] as TreeNode[], allNodes: [] as TreeNode[], maxGen: 0, treeHeight: 300 };

    const nodeMap = new Map<string, TreeNode>();
    for (const c of candidates) {
      nodeMap.set(c.id, {
        id: c.id, name: c.name, activation: c.activation, generation: c.generation,
        bestLoss: c.bestLoss, bestFitness: c.bestFitness, steps: c.steps,
        children: [], x: 0, y: 0, width: 0,
      });
    }

    const rootNodes: TreeNode[] = [];
    for (const c of candidates) {
      const node = nodeMap.get(c.id)!;
      if (c.parentId && nodeMap.has(c.parentId)) {
        nodeMap.get(c.parentId)!.children.push(node);
      } else {
        rootNodes.push(node);
      }
    }

    // Sort roots by activation for consistent ordering
    rootNodes.sort((a, b) => a.activation.localeCompare(b.activation));

    // Layout: assign x positions using a simple recursive walk
    const NODE_W = 90;
    const NODE_H = 56;
    const H_GAP = 16;
    const V_GAP = 40;

    // Compute subtree widths bottom-up
    function computeWidth(node: TreeNode): number {
      if (node.children.length === 0) {
        node.width = NODE_W;
        return NODE_W;
      }
      node.children.sort((a, b) => a.activation.localeCompare(b.activation));
      let total = 0;
      for (const child of node.children) {
        if (total > 0) total += H_GAP;
        total += computeWidth(child);
      }
      node.width = Math.max(NODE_W, total);
      return node.width;
    }

    let totalWidth = 0;
    for (const root of rootNodes) {
      if (totalWidth > 0) totalWidth += H_GAP;
      totalWidth += computeWidth(root);
    }

    // Assign positions
    function assignPositions(node: TreeNode, xStart: number, depth: number) {
      node.y = depth * (NODE_H + V_GAP) + 20;
      if (node.children.length === 0) {
        node.x = xStart + node.width / 2;
        return;
      }
      let cx = xStart;
      for (const child of node.children) {
        assignPositions(child, cx, depth + 1);
        cx += child.width + H_GAP;
      }
      // Center parent over children
      const first = node.children[0];
      const last = node.children[node.children.length - 1];
      node.x = (first.x + last.x) / 2;
    }

    let xOff = 20;
    for (const root of rootNodes) {
      assignPositions(root, xOff, 0);
      xOff += root.width + H_GAP;
    }

    const all: TreeNode[] = [];
    function collect(n: TreeNode) { all.push(n); n.children.forEach(collect); }
    rootNodes.forEach(collect);

    const mg = Math.max(0, ...all.map((n) => n.generation));
    const th = (mg + 1) * (NODE_H + V_GAP) + 40;

    return { roots: rootNodes, allNodes: all, maxGen: mg, treeHeight: Math.max(300, th) };
  }, [candidates]);

  // Draw
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || allNodes.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const NODE_W = 90;
    const NODE_H = 56;
    const RADIUS = 6;

    // Draw links first (behind nodes)
    ctx.lineWidth = 1.5;
    for (const node of allNodes) {
      for (const child of node.children) {
        const col = actHex(child.activation);
        ctx.strokeStyle = col + "60"; // 37% opacity
        ctx.beginPath();
        ctx.moveTo(node.x, node.y + NODE_H);
        // Curved bezier link
        const midY = (node.y + NODE_H + child.y) / 2;
        ctx.bezierCurveTo(node.x, midY, child.x, midY, child.x, child.y);
        ctx.stroke();
      }
    }

    // Draw nodes
    for (const node of allNodes) {
      const col = actHex(node.activation);
      const isSelected = node.id === selectedCandidateId;
      const isHovered = hoveredNode?.id === node.id;
      const x = node.x - NODE_W / 2;
      const y = node.y;

      // Node background
      ctx.fillStyle = isSelected ? col + "30" : isHovered ? col + "20" : "#111118";
      ctx.strokeStyle = isSelected ? col : isHovered ? col + "bb" : col + "55";
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(x, y, NODE_W, NODE_H, RADIUS);
      ctx.fill();
      ctx.stroke();

      // Activation color bar at top
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.roundRect(x, y, NODE_W, 3, [RADIUS, RADIUS, 0, 0]);
      ctx.fill();

      // Name
      ctx.fillStyle = col;
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(node.name, node.x, y + 7);

      // Activation type
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "7px monospace";
      ctx.fillText(node.activation, node.x, y + 19);

      // Loss
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.font = "8px monospace";
      const lossStr = node.bestLoss === Infinity ? "—" : node.bestLoss.toFixed(4);
      ctx.fillText(`loss ${lossStr}`, node.x, y + 31);

      // Steps
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.font = "7px monospace";
      ctx.fillText(`${node.steps} steps · gen ${node.generation}`, node.x, y + 43);
    }

    // Generation row labels on the left
    ctx.fillStyle = "rgba(255,255,255,0.15)";
    ctx.font = "bold 8px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (let g = 0; g <= maxGen; g++) {
      const y = g * (NODE_H + 40) + 20 + NODE_H / 2;
      ctx.fillText(`Gen ${g}`, 4, y);
    }

    ctx.restore();
  }, [allNodes, maxGen, selectedCandidateId, hoveredNode]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  // Mouse interaction — hit test nodes
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const NODE_W = 90;
    const NODE_H = 56;
    let found: TreeNode | null = null;
    for (const node of allNodes) {
      if (mx >= node.x - NODE_W / 2 && mx <= node.x + NODE_W / 2 && my >= node.y && my <= node.y + NODE_H) {
        found = node;
        break;
      }
    }
    setHoveredNode(found);
    setTooltipPos(found ? { x: e.clientX - rect.left, y: e.clientY - rect.top } : null);
  }, [allNodes]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredNode && onSelectCandidate) {
      onSelectCandidate(hoveredNode.id);
    }
  }, [hoveredNode, onSelectCandidate]);

  if (candidates.length === 0) return null;

  return (
    <div className="relative w-full overflow-x-auto overflow-y-hidden rounded-lg bg-[#08080f]">
      <canvas
        ref={canvasRef}
        className="w-full cursor-pointer"
        style={{ height: treeHeight, minWidth: Math.max(400, allNodes.length * 50) }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { setHoveredNode(null); setTooltipPos(null); }}
        onClick={handleClick}
      />
      {hoveredNode && tooltipPos && (
        <div
          className="pointer-events-none absolute z-20 rounded border border-border bg-black/90 px-2.5 py-1.5 font-mono text-[0.6rem] leading-relaxed text-white/80 shadow-lg backdrop-blur-sm"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 10, maxWidth: 200 }}
        >
          <div className="font-bold" style={{ color: actHex(hoveredNode.activation) }}>
            {hoveredNode.name}
          </div>
          <div>activation: {hoveredNode.activation}</div>
          <div>generation: {hoveredNode.generation}</div>
          <div>best loss: {hoveredNode.bestLoss === Infinity ? "—" : hoveredNode.bestLoss.toFixed(4)}</div>
          <div>fitness: {hoveredNode.bestFitness === -Infinity ? "—" : hoveredNode.bestFitness.toFixed(4)}</div>
          <div>steps: {hoveredNode.steps}</div>
        </div>
      )}
    </div>
  );
}

// ── Amcharts Oscillator / Damping Analysis ───────────────────

function AmchartsOscillatorChart({ metrics }: { metrics: SymbioMetric[] }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<any>(null);

  useEffect(() => {
    if (!chartRef.current || metrics.length < 40) return;

    const initChart = async () => {
      const am5 = await import("@amcharts/amcharts5");
      const am5xy = await import("@amcharts/amcharts5/xy");
      const am5themes = await import("@amcharts/amcharts5/themes/Dark");

      if (rootRef.current) rootRef.current.dispose();

      if (!chartRef.current) return;
      const root = am5.Root.new(chartRef.current);
      rootRef.current = root;
      root.setThemes([am5themes.default.new(root)]);

      const chart = root.container.children.push(
        am5xy.XYChart.new(root, { panX: true, panY: false, wheelX: "panX", wheelY: "zoomX" })
      );

      // Compute oscillation data: deviation from exponential moving average
      const windowSize = 20;
      const data: { step: number; deviation: number; amplitude: number; ema: number; heatCapacity: number }[] = [];
      let ema = metrics[0].loss;
      const alpha = 2 / (windowSize + 1);

      for (let i = 1; i < metrics.length; i++) {
        ema = alpha * metrics[i].loss + (1 - alpha) * ema;
        const deviation = metrics[i].loss - ema;
        const amplitude = Math.abs(deviation);

        // Heat capacity proxy: |d(loss)/d(lr)| over a small window
        let heatCapacity = 0;
        if (i >= 5) {
          const lossChange = Math.abs(metrics[i].loss - metrics[i - 5].loss);
          const lrChange = Math.abs(metrics[i].lr - metrics[i - 5].lr);
          heatCapacity = lrChange > 1e-10 ? lossChange / lrChange : 0;
        }

        data.push({
          step: metrics[i].step,
          deviation,
          amplitude,
          ema,
          heatCapacity: Math.min(heatCapacity, 1e6), // cap for display
        });
      }

      // X Axis
      const xAxis = chart.xAxes.push(am5xy.ValueAxis.new(root, {
        renderer: am5xy.AxisRendererX.new(root, { minGridDistance: 50 }),
        tooltip: am5.Tooltip.new(root, {}),
      }));
      xAxis.get("renderer").labels.template.setAll({ fill: am5.color("#666"), fontSize: 10 });
      xAxis.get("renderer").grid.template.setAll({ stroke: am5.color("#1a1a1a") });

      // Y Axis for deviation
      const yAxis = chart.yAxes.push(am5xy.ValueAxis.new(root, {
        renderer: am5xy.AxisRendererY.new(root, {}),
        tooltip: am5.Tooltip.new(root, {}),
      }));
      yAxis.get("renderer").labels.template.setAll({ fill: am5.color("#666"), fontSize: 10 });
      yAxis.get("renderer").grid.template.setAll({ stroke: am5.color("#1a1a1a") });

      // Y Axis for heat capacity (opposite side)
      const yAxis2 = chart.yAxes.push(am5xy.ValueAxis.new(root, {
        renderer: am5xy.AxisRendererY.new(root, { opposite: true }),
      }));
      yAxis2.get("renderer").labels.template.setAll({ fill: am5.color("#f59e0b"), fontSize: 10 });

      // Deviation area series (oscillation)
      const deviationSeries = chart.series.push(am5xy.SmoothedXLineSeries.new(root, {
        name: "Loss Oscillation",
        xAxis, yAxis,
        valueYField: "deviation",
        valueXField: "step",
        tooltip: am5.Tooltip.new(root, { labelText: "Step {valueX}\nDeviation: {valueY.formatNumber('#.####')}" }),
      }));
      deviationSeries.strokes.template.setAll({ strokeWidth: 1.5, stroke: am5.color("#22d3ee") });
      deviationSeries.fills.template.setAll({ fillOpacity: 0.08, fill: am5.color("#22d3ee"), visible: true });

      // Amplitude envelope series
      const ampSeries = chart.series.push(am5xy.SmoothedXLineSeries.new(root, {
        name: "Amplitude Envelope",
        xAxis, yAxis,
        valueYField: "amplitude",
        valueXField: "step",
        tooltip: am5.Tooltip.new(root, { labelText: "Amplitude: {valueY.formatNumber('#.####')}" }),
      }));
      ampSeries.strokes.template.setAll({ strokeWidth: 2, stroke: am5.color("#f472b6"), strokeDasharray: [4, 2] });

      // Heat capacity series (secondary axis)
      const heatSeries = chart.series.push(am5xy.SmoothedXLineSeries.new(root, {
        name: "Heat Capacity",
        xAxis, yAxis: yAxis2,
        valueYField: "heatCapacity",
        valueXField: "step",
        tooltip: am5.Tooltip.new(root, { labelText: "Heat Capacity: {valueY.formatNumber('#.##')}" }),
      }));
      heatSeries.strokes.template.setAll({ strokeWidth: 1.5, stroke: am5.color("#f59e0b"), strokeOpacity: 0.6 });

      // Zero reference line
      const rangeDataItem = yAxis.createAxisRange(yAxis.makeDataItem({ value: 0 }));
      rangeDataItem.get("grid")!.setAll({ stroke: am5.color("#555"), strokeWidth: 1, strokeDasharray: [4, 4] });

      // Cursor
      chart.set("cursor", am5xy.XYCursor.new(root, { behavior: "zoomX", xAxis }));

      // Legend
      const legend = chart.children.push(am5.Legend.new(root, { x: am5.percent(50), centerX: am5.percent(50), y: 0 }));
      legend.labels.template.setAll({ fill: am5.color("#888"), fontSize: 10 });
      legend.data.setAll(chart.series.values);

      // Set data
      deviationSeries.data.setAll(data);
      ampSeries.data.setAll(data);
      heatSeries.data.setAll(data);

      chart.appear(1000, 100);
    };

    initChart();

    return () => { if (rootRef.current) { rootRef.current.dispose(); rootRef.current = null; } };
  }, [metrics]);

  if (metrics.length < 40) return null;

  return <div ref={chartRef} style={{ width: "100%", height: 300 }} />;
}

// ── Symbio Section (composite) ───────────────────────────────

export function SymbioSection({ metrics, run, pinnedStep, onPinStep }: {
  metrics: SymbioMetric[];
  run: { symbio?: number | null; symbio_config?: string | null; ffn_activation?: string | null; batch_size?: number };
  pinnedStep?: number | null;
  onPinStep?: (s: number) => void;
}) {
  const isSymbio = (run.symbio ?? 0) === 1;
  const hasClipData = metrics.some(m => m.clip_coef != null);
  const hasCusumData = metrics.some(m => m.cusum_grad != null || m.cusum_clip != null || m.cusum_tps != null || m.cusum_val != null);
  const hasSymbioMetrics = metrics.some(m => m.weight_entropy != null);
  const hasAdaptiveBatch = metrics.some(m => m.adaptive_batch_size != null);
  const hasSearchData = metrics.some(m => m.symbio_candidate_id != null);
  const hasMI = metrics.some(m => m.mi_input_repr != null);
  const hasPopEntropy = metrics.some(m => m.population_entropy != null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const treeRef = useRef<HTMLDivElement>(null);

  let symbioConfig: Record<string, unknown> | null = null;
  try { if (run.symbio_config) symbioConfig = JSON.parse(run.symbio_config); } catch { /* ignore */ }

  if (!isSymbio && !hasClipData) return null;

  return (
    <>
      {hasClipData && (
        <ChartPanel title="Clip Telemetry" helpText={HELP.clip}>
          <ClipChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
        </ChartPanel>
      )}

      {isSymbio && (
        <>
          <div className="mb-4 mt-4 flex items-center gap-2">
            <span className="rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-wider text-purple-400">Symbiogenesis</span>
            {run.ffn_activation && <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[0.6rem] font-medium text-cyan-400">{run.ffn_activation.toUpperCase()}</span>}
          </div>

          <SymbioStatsGrid metrics={metrics} />

          {hasCusumData && (
            <div className="mb-4">
              <ChartPanel title="CUSUM Change-Point Monitor" helpText={HELP.cusum}>
                <CusumChart metrics={metrics} sensitivity={(symbioConfig?.cusumSensitivity as number) ?? 4.0} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {hasSymbioMetrics && (
            <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ChartPanel title="Weight Entropy" helpText={HELP.weightEntropy}>
                <SparseLineChart metrics={metrics} getY={m => m.weight_entropy} color="#a78bfa" label="Weight Entropy" format={v => v.toFixed(2)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
              <ChartPanel title="Effective Rank" helpText={HELP.effectiveRank}>
                <SparseLineChart metrics={metrics} getY={m => m.effective_rank} color="#f59e0b" label="Effective Rank" format={v => v.toFixed(1)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
              <ChartPanel title="Free Energy" helpText={HELP.freeEnergy}>
                <SparseLineChart metrics={metrics} getY={m => m.free_energy} color="#34d399" label="Free Energy" format={v => v.toFixed(4)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
              <ChartPanel title="Fitness Score" helpText={HELP.fitness}>
                <SparseLineChart metrics={metrics} getY={m => m.fitness_score} color="#60a5fa" label="Fitness" format={v => v.toFixed(4)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {hasPopEntropy && (
            <div className="mb-4">
              <ChartPanel title="Population Entropy" helpText={HELP.populationEntropy}>
                <SparseLineChart metrics={metrics} getY={m => m.population_entropy} color="#22d3ee" label="Population Entropy" format={v => v.toFixed(3)} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {hasMI && (
            <div className="mb-4">
              <ChartPanel title="Mutual Information Profiles" helpText={HELP.mi}>
                <MIProfilesChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {hasAdaptiveBatch && (
            <div className="mb-4">
              <ChartPanel title="Adaptive Batch Size" helpText={HELP.batchSize}>
                <AdaptiveBatchChart metrics={metrics} configBatchSize={run.batch_size} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {/* Phase Change / Gelation */}
          {hasCusumData && (
            <div className="mb-4">
              <ChartPanel title="Phase Change / Gelation" helpText={HELP.phaseChange}>
                <PhaseChangeTimeline metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {/* Harmonic Oscillator Analysis */}
          {metrics.length > 60 && (
            <div className="mb-4">
              <ChartPanel title="Loss Oscillation (Harmonic Analysis)" helpText={HELP.harmonic}>
                <HarmonicOscillatorChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
              </ChartPanel>
            </div>
          )}

          {hasSearchData && (
            <>
              {/* Evolutionary Metrics */}
              <div className="mb-4">
                <ChartPanel title="Evolutionary Search" helpText={HELP.evolution}>
                  <EvolutionaryTimeline metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
                </ChartPanel>
              </div>

              {/* Evolutionary Lineage Tree (amcharts) */}
              <div className="mb-4" ref={treeRef}>
                <ChartPanel title="Evolutionary Lineage Tree" helpText={HELP.lineageTree}>
                  <EvolutionaryTreeChart
                    metrics={metrics}
                    selectedCandidateId={selectedCandidateId}
                    onSelectCandidate={setSelectedCandidateId}
                  />
                </ChartPanel>
              </div>

              {/* Traditional Lineage Tree */}
              <div className="mb-4">
                <ChartPanel title="Lineage Tree" helpText="Traditional top-down tree layout showing parent-child relationships between candidates. Gen-0 candidates (initial population) are at the top, with mutations and offspring branching downward. Each node shows the candidate name, activation type, best loss, and step count. Nodes are color-coded by activation. Click to select a candidate. Bezier curves connect parents to children.">
                  <LineageTreeChart
                    metrics={metrics}
                    selectedCandidateId={selectedCandidateId}
                    onSelectCandidate={setSelectedCandidateId}
                  />
                </ChartPanel>
              </div>

              {/* Activation Switch Log — click navigates to tree */}
              <div className="mb-4 rounded-lg border border-border bg-surface">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                    Activation Switch Log
                  </span>
                  <ChartHelpIcon text={HELP.switchLog} />
                </div>
                <ActivationSwitchLog metrics={metrics} onNavigateToTree={(candidateId: string) => {
                  setSelectedCandidateId(candidateId);
                  treeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                }} />
              </div>

              {/* Search Candidates */}
              <div className="mb-4 rounded-lg border border-border bg-surface">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                    Search Candidates
                  </span>
                  <ChartHelpIcon text={HELP.candidates} />
                </div>
                <SearchCandidateTable metrics={metrics} />
              </div>

              {/* Activation Distribution */}
              <div className="mb-4">
                <ChartPanel title="Activation Distribution" helpText={HELP.activationDist}>
                  <ActivationDistributionChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={onPinStep} />
                </ChartPanel>
              </div>
            </>
          )}

          {/* Amcharts Oscillator / Damping Analysis */}
          {metrics.length > 40 && (
            <div className="mb-4">
              <ChartPanel title="Oscillation & Heat Capacity (amcharts)" helpText={HELP.harmonicAmcharts}>
                <AmchartsOscillatorChart metrics={metrics} />
              </ChartPanel>
            </div>
          )}

          {/* Radial Activation Evolution */}
          {metrics.length > 10 && (
            <div className="mb-4">
              <ChartPanel title="Activation Evolution Radial" helpText="Radial polar plot of the evolutionary activation search. Angle = training time (clockwise from top). Outer band shows which activation is active per candidate (color-coded). Inner rings: loss trajectory (colored by activation), fitness score, architecture diversity, and MI flow. Diamond markers indicate candidate switches. Generation boundaries are dashed spokes. Center shows the current candidate name, activation type, and loss. Hover for detailed candidate info including lineage.">
                <RadialTrainingViz metrics={metrics} />
              </ChartPanel>
            </div>
          )}

          {symbioConfig && (
            <details className="mb-4 rounded-lg border border-border bg-surface" open>
              <summary className="cursor-pointer px-4 py-3 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted hover:text-text">Symbio Config</summary>
              <pre className="overflow-x-auto px-4 pb-3 text-[0.6rem] text-text-muted">{JSON.stringify(symbioConfig, null, 2)}</pre>
            </details>
          )}
        </>
      )}
    </>
  );
}
