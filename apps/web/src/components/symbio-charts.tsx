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
};

const ACTIVATION_BG: Record<string, string> = {
  gelu: "bg-blue-500/10 border-blue-500/20", silu: "bg-green-500/10 border-green-500/20",
  relu: "bg-yellow-500/10 border-yellow-500/20", swiglu: "bg-purple-500/10 border-purple-500/20",
};

const ACTIVATION_HEX: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
};

// ── Help Text ────────────────────────────────────────────────

const HELP = {
  clip: "Gradient clipping prevents training instability by scaling down gradient magnitudes when they exceed a threshold. clip_coef shows the actual scaling factor applied (1.0 = no clipping). clip_pct shows what fraction of the gradient norm exceeded the threshold.",
  cusum: "CUSUM (Cumulative Sum) is a statistical method for detecting sudden shifts in time series data. Four independent monitors track gradient norms, clipping rates, throughput, and validation loss. When the cumulative deviation from baseline exceeds a sensitivity threshold, an alert fires indicating a training regime change.",
  weightEntropy: "Shannon entropy of the weight magnitude distribution, measured in bits. Higher entropy means weights are more uniformly distributed across magnitudes. Low entropy means weights cluster at specific magnitudes, which may indicate under-utilization of model capacity.",
  effectiveRank: "Measures how many dimensions (singular values) of each weight matrix are actively being used. Computed via SVD: counts singular values above 1% of the largest. Higher rank means the model uses more of its representational capacity.",
  freeEnergy: "A thermodynamic analogy: F = loss + beta * weight_entropy. Balances model fit (loss) against complexity (entropy). Lower free energy means a better trade-off between accuracy and model simplicity.",
  fitness: "Multi-objective score combining accuracy and complexity: fitness = alpha * (1/(1+loss)) - complexity_penalty. Used to rank candidates in the evolutionary search. Higher is better.",
  adaptiveBatch: "Dynamically adjusts batch size in response to CUSUM alerts. Reduces batch on gradient instability or throughput drops (smaller batches recover faster). Increases batch when clipping is persistent (larger batches smooth gradients). Gradually restores to baseline during calm periods.",
  switchLog: "Records every time the evolutionary search switches from one activation function candidate to another. Shows the outgoing candidate's performance metrics (loss, fitness, throughput) and any stability issues detected during its evaluation.",
  candidates: "All candidates ever created during the evolutionary activation search. Each was trained for a fixed number of steps and ranked by validation loss or fitness score. Duplicates occur when elite parents are cloned into the next generation or when mutation selects the same activation.",
  activationDist: "Shows how training steps are distributed across different activation functions over time. In a converging search, the winning activation should accumulate more steps in later generations.",
  evolution: "Visualizes the evolutionary search progression across generations. Each generation evaluates multiple candidates, selects the best performers as parents, and generates offspring with potential mutations. The fitness landscape shows how candidates improve or plateau over generations.",
  phaseChange: "Gelation/phase changes are detected via CUSUM monitors. When gradient norms, clipping rates, or throughput shift dramatically, the training has entered a new regime. Green = stable (no alerts), Yellow = mild instability (1-2 alerts), Red = regime shift (3+ alerts).",
  harmonic: "Analyzes the oscillatory behavior of training loss around its moving average. High-amplitude oscillations suggest an under-damped system (learning rate too high). The oscillation frequency and damping ratio characterize the training dynamics near equilibrium.",
  diversity: "Architecture diversity measures what fraction of the current population uses unique activations (0 = all same, 1 = all different). In a healthy search, diversity starts high and gradually decreases as the search converges on the best-performing activation.",
  populationEntropy: "Shannon entropy of the softmax-normalized recent loss distribution (50-step window). Higher entropy means loss values are spread widely (unstable). Lower entropy means loss is concentrated (stable). Rapid entropy changes indicate training phase transitions.",
  mi: "Mutual Information (MI) estimates measure how much information flows through the model. mi_input_repr: how much input info the hidden layers capture. mi_repr_output: how much the representations predict the output. mi_compression: the ratio, tracking information bottleneck behavior.",
  batchSize: "When adaptive batch sizing is enabled, this shows the dynamically adjusted batch size. It differs from the training config batch size (shown above) when CUSUM detects instability. The config batch size is the baseline; adaptive batch varies around it in response to training dynamics.",
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
  activation: string;
  generation: number;
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
    activation: string; generation: number; losses: number[]; valLosses: number[];
    fitnesses: number[]; tps: number[]; steps: number; cusumAlerts: number;
    lastClipPct: number | null; startStep: number; endStep: number;
  }>();

  for (const m of metrics) {
    const id = m.symbio_candidate_id;
    if (!id) continue;
    let entry = candidates.get(id);
    if (!entry) {
      entry = {
        activation: m.symbio_candidate_activation ?? "?", generation: m.symbio_generation ?? 0,
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
    activation: e.activation,
    generation: e.generation,
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
            <th className="px-3 py-2 text-left">ID</th>
            <th className="px-3 py-2 text-left">Activation</th>
            <th className="px-3 py-2 text-center">Gen</th>
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
              <td className="px-3 py-2 truncate font-mono text-text-secondary" title={c.id}>{c.id}</td>
              <td className="px-3 py-2">
                <span className={`inline-block rounded border px-1.5 py-0.5 text-[0.62rem] font-semibold ${ACTIVATION_BG[c.activation] ?? "bg-surface-2 border-border"} ${ACTIVATION_COLORS[c.activation] ?? "text-text-secondary"}`}>{c.activation}</span>
              </td>
              <td className="px-3 py-2 text-center text-text-muted">{c.generation}</td>
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

function ActivationSwitchLog({ metrics }: { metrics: SymbioMetric[] }) {
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
                </tr>
                {isExpanded && (
                  <tr className="bg-purple-500/5 border-b border-purple-500/20">
                    <td colSpan={9} className="px-4 py-3">
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

function EvolutionaryTimeline({ metrics }: { metrics: SymbioMetric[] }) {
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
            <ScatterChart>
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
              <Scatter data={fitnessData} fill="#34d399">
                {fitnessData.map((d, i) => (
                  <Cell key={i} fill={ACTIVATION_HEX[d.activation] ?? "#888"} />
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
            <AreaChart data={diversityData}>
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
              <Area type="monotone" dataKey="diversity" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.15} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel>
      )}
    </div>
  );
}

// ── Phase Change / Gelation Visualization ────────────────────

function PhaseChangeTimeline({ metrics }: { metrics: SymbioMetric[] }) {
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
      <BarChart data={data}>
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

function HarmonicOscillatorChart({ metrics }: { metrics: SymbioMetric[] }) {
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

  return (
    <div className="space-y-3">
      {/* Oscillation */}
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data}>
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
          <Area type="monotone" dataKey="deviation" name="Oscillation" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.1} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>

      {/* Amplitude envelope - damping analysis */}
      {envelope.length > 2 && (
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={envelope}>
            <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
            <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} />
            <Area type="monotone" dataKey="amplitude" name="Amplitude Envelope" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── Activation Distribution Over Time ────────────────────────

function ActivationDistributionChart({ metrics }: { metrics: SymbioMetric[] }) {
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
                  style={{ width: `${d.pct * 100}%`, backgroundColor: ACTIVATION_HEX[d.activation] ?? "#888", opacity: 0.7 }}
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
          <AreaChart data={data}>
            <CartesianGrid stroke={CHART_THEME.grid} strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} tickFormatter={(v: number) => fmtNum(v)} />
            <YAxis stroke={CHART_THEME.axisText} tick={{ fontSize: 10 }} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <RTooltip content={<CustomTooltipContent />} />
            {activations.map(act => (
              <Area key={act} type="monotone" dataKey={act} stackId="1" stroke={ACTIVATION_HEX[act] ?? "#888"} fill={ACTIVATION_HEX[act] ?? "#888"} fillOpacity={0.3} strokeWidth={0} />
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
                <PhaseChangeTimeline metrics={metrics} />
              </ChartPanel>
            </div>
          )}

          {/* Harmonic Oscillator Analysis */}
          {metrics.length > 60 && (
            <div className="mb-4">
              <ChartPanel title="Loss Oscillation (Harmonic Analysis)" helpText={HELP.harmonic}>
                <HarmonicOscillatorChart metrics={metrics} />
              </ChartPanel>
            </div>
          )}

          {hasSearchData && (
            <>
              {/* Evolutionary Metrics */}
              <div className="mb-4">
                <ChartPanel title="Evolutionary Search" helpText={HELP.evolution}>
                  <EvolutionaryTimeline metrics={metrics} />
                </ChartPanel>
              </div>

              {/* Activation Switch Log */}
              <div className="mb-4 rounded-lg border border-border bg-surface">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
                    Activation Switch Log
                  </span>
                  <ChartHelpIcon text={HELP.switchLog} />
                </div>
                <ActivationSwitchLog metrics={metrics} />
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
                  <ActivationDistributionChart metrics={metrics} />
                </ChartPanel>
              </div>
            </>
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
