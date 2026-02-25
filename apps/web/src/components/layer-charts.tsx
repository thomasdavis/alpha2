"use client";

import { useMemo, useRef, useEffect, useState, useCallback } from "react";
import {
  type ChartMetric,
  ChartPanel, fmtNum,
} from "@/components/charts";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, AreaChart, Area, Legend,
} from "recharts";

// ── Types ────────────────────────────────────────────────────

interface LayerMetric extends ChartMetric {
  per_layer_grad_norms: string | null;
}

interface ParsedLayerNorms {
  step: number;
  layers: Record<string, number>; // "embed" | "0" | "1" | ... | "head"
}

// ── Colors ────────────────────────────────────────────────────

const LAYER_PALETTE = [
  "#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f472b6",
  "#22d3ee", "#fb923c", "#e879f9", "#4ade80", "#fbbf24",
  "#818cf8", "#f87171", "#2dd4bf", "#c084fc", "#38bdf8",
  "#a3e635",
];

function layerColor(key: string, idx: number): string {
  if (key === "embed") return "#94a3b8";
  if (key === "head") return "#e2e8f0";
  return LAYER_PALETTE[idx % LAYER_PALETTE.length];
}

// ── Parsing ────────────────────────────────────────────────────

function parseLayerNorms(metrics: LayerMetric[]): ParsedLayerNorms[] {
  const out: ParsedLayerNorms[] = [];
  for (const m of metrics) {
    if (!m.per_layer_grad_norms) continue;
    try {
      const layers = JSON.parse(m.per_layer_grad_norms) as Record<string, number>;
      out.push({ step: m.step, layers });
    } catch { /* skip malformed */ }
  }
  return out;
}

function getAllLayerKeys(parsed: ParsedLayerNorms[]): string[] {
  const keySet = new Set<string>();
  for (const p of parsed) {
    for (const k of Object.keys(p.layers)) keySet.add(k);
  }
  // Sort: embed first, then numeric, then head
  const keys = [...keySet];
  keys.sort((a, b) => {
    const ai = a === "embed" ? -1 : a === "head" ? 999 : parseInt(a, 10);
    const bi = b === "embed" ? -1 : b === "head" ? 999 : parseInt(b, 10);
    return ai - bi;
  });
  return keys;
}

// ── Heatmap (Canvas) ────────────────────────────────────────

function GradNormHeatmap({ data, layerKeys }: { data: ParsedLayerNorms[]; layerKeys: string[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; layer: string; norm: number } | null>(null);

  const maxNorm = useMemo(() => {
    let max = 0;
    for (const d of data) {
      for (const v of Object.values(d.layers)) {
        if (v > max) max = v;
      }
    }
    return max || 1;
  }, [data]);

  // Subsample for display (max ~600 columns)
  const displayData = useMemo(() => {
    if (data.length <= 600) return data;
    const stride = Math.ceil(data.length / 600);
    return data.filter((_, i) => i % stride === 0);
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || displayData.length === 0 || layerKeys.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const cellH = 20;
    const labelW = 48;
    const chartW = width - labelW;
    const height = layerKeys.length * cellH;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Layer labels
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "#6b7280";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let li = 0; li < layerKeys.length; li++) {
      const label = layerKeys[li] === "embed" ? "emb" : layerKeys[li] === "head" ? "head" : `L${layerKeys[li]}`;
      ctx.fillText(label, labelW - 6, li * cellH + cellH / 2);
    }

    // Heatmap cells
    const cellW = Math.max(1, chartW / displayData.length);
    // Use log scale for better contrast
    const logMax = Math.log(maxNorm + 1e-8);
    for (let si = 0; si < displayData.length; si++) {
      const d = displayData[si];
      const x = labelW + si * cellW;
      for (let li = 0; li < layerKeys.length; li++) {
        const norm = d.layers[layerKeys[li]] ?? 0;
        const logNorm = Math.log(norm + 1e-8);
        const intensity = Math.max(0, Math.min(1, logNorm / logMax));
        // Cool (dark blue) to warm (yellow/white) via HSL
        const h = 240 - intensity * 180; // 240 (blue) → 60 (yellow)
        const s = 50 + intensity * 30;
        const l = 8 + intensity * 55;
        ctx.fillStyle = `hsl(${h}, ${s}%, ${l}%)`;
        ctx.fillRect(x, li * cellH, Math.ceil(cellW) + 1, cellH - 1);
      }
    }
  }, [displayData, layerKeys, maxNorm]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || displayData.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const labelW = 48;
    const cellH = 20;
    const chartW = rect.width - labelW;
    const cellW = chartW / displayData.length;
    const col = Math.floor((mx - labelW) / cellW);
    const row = Math.floor(my / cellH);
    if (col < 0 || col >= displayData.length || row < 0 || row >= layerKeys.length) {
      setTooltip(null);
      return;
    }
    const d = displayData[col];
    const key = layerKeys[row];
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, step: d.step, layer: key, norm: d.layers[key] ?? 0 });
  }, [displayData, layerKeys]);

  return (
    <div ref={containerRef} className="relative" onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)}>
      <canvas ref={canvasRef} className="w-full rounded" />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 rounded border border-border-2 bg-surface-2 px-2 py-1 text-[0.62rem] shadow-lg"
          style={{ left: Math.min(tooltip.x + 12, (containerRef.current?.clientWidth ?? 300) - 140), top: tooltip.y - 36 }}
        >
          <span className="text-text-muted">Step {fmtNum(tooltip.step)}</span>
          {" "}
          <span className="font-semibold text-white">
            {tooltip.layer === "embed" ? "Embed" : tooltip.layer === "head" ? "Head" : `Layer ${tooltip.layer}`}
          </span>
          {" "}
          <span className="font-mono text-yellow-400">{tooltip.norm.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}

// ── Line chart data builder ────────────────────────────────

function buildLineData(parsed: ParsedLayerNorms[], layerKeys: string[]) {
  // Subsample for recharts (max 500 points)
  const stride = Math.max(1, Math.ceil(parsed.length / 500));
  const out: Record<string, number | string>[] = [];
  for (let i = 0; i < parsed.length; i += stride) {
    const d = parsed[i];
    const row: Record<string, number | string> = { step: d.step };
    for (const k of layerKeys) {
      row[k] = d.layers[k] ?? 0;
    }
    out.push(row);
  }
  return out;
}

// ── Stacked area (proportion) ────────────────────────────

function buildProportionData(parsed: ParsedLayerNorms[], layerKeys: string[]) {
  const stride = Math.max(1, Math.ceil(parsed.length / 500));
  const out: Record<string, number | string>[] = [];
  for (let i = 0; i < parsed.length; i += stride) {
    const d = parsed[i];
    const total = Object.values(d.layers).reduce((s, v) => s + v, 0) || 1;
    const row: Record<string, number | string> = { step: d.step };
    for (const k of layerKeys) {
      row[k] = ((d.layers[k] ?? 0) / total) * 100;
    }
    out.push(row);
  }
  return out;
}

// ── Layer stats table ────────────────────────────────────

interface LayerStat {
  key: string;
  label: string;
  avgNorm: number;
  maxNorm: number;
  minNorm: number;
  recentNorm: number;
  trend: "up" | "down" | "stable";
  color: string;
}

function computeLayerStats(parsed: ParsedLayerNorms[], layerKeys: string[]): LayerStat[] {
  if (parsed.length === 0) return [];
  const stats: LayerStat[] = [];
  const numericKeys = layerKeys.filter(k => k !== "embed" && k !== "head");

  for (let ki = 0; ki < layerKeys.length; ki++) {
    const k = layerKeys[ki];
    const values = parsed.map(p => p.layers[k] ?? 0).filter(v => v > 0);
    if (values.length === 0) continue;
    const avg = values.reduce((s, v) => s + v, 0) / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);
    const recent = values.slice(-20);
    const recentAvg = recent.reduce((s, v) => s + v, 0) / recent.length;
    const earlier = values.slice(0, Math.max(1, values.length - 20));
    const earlierAvg = earlier.reduce((s, v) => s + v, 0) / earlier.length;
    const ratio = earlierAvg > 0 ? recentAvg / earlierAvg : 1;
    const trend = ratio > 1.2 ? "up" : ratio < 0.8 ? "down" : "stable";

    const label = k === "embed" ? "Embed" : k === "head" ? "Head" : `Layer ${k}`;
    const colorIdx = k === "embed" ? 0 : k === "head" ? 0 : numericKeys.indexOf(k);
    stats.push({
      key: k,
      label,
      avgNorm: avg,
      maxNorm: max,
      minNorm: min,
      recentNorm: recentAvg,
      trend,
      color: layerColor(k, colorIdx),
    });
  }
  return stats;
}

// ── Custom tooltip ────────────────────────────────────────

function LayerTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a: any, b: any) => (b.value ?? 0) - (a.value ?? 0));
  return (
    <div className="rounded border border-border-2 bg-surface-2 px-3 py-2 shadow-xl text-[0.62rem]">
      <div className="mb-1 font-semibold text-text-muted">Step {fmtNum(label)}</div>
      {sorted.slice(0, 10).map((entry: any) => (
        <div key={entry.dataKey} className="flex justify-between gap-3">
          <span style={{ color: entry.color }}>
            {entry.dataKey === "embed" ? "Embed" : entry.dataKey === "head" ? "Head" : `L${entry.dataKey}`}
          </span>
          <span className="font-mono text-white">{(entry.value as number).toFixed(4)}</span>
        </div>
      ))}
      {sorted.length > 10 && <div className="text-text-muted">+{sorted.length - 10} more</div>}
    </div>
  );
}

function ProportionTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a: any, b: any) => (b.value ?? 0) - (a.value ?? 0));
  return (
    <div className="rounded border border-border-2 bg-surface-2 px-3 py-2 shadow-xl text-[0.62rem]">
      <div className="mb-1 font-semibold text-text-muted">Step {fmtNum(label)}</div>
      {sorted.slice(0, 10).map((entry: any) => (
        <div key={entry.dataKey} className="flex justify-between gap-3">
          <span style={{ color: entry.color }}>
            {entry.dataKey === "embed" ? "Embed" : entry.dataKey === "head" ? "Head" : `L${entry.dataKey}`}
          </span>
          <span className="font-mono text-white">{(entry.value as number).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────

export function LayersSection({ metrics }: { metrics: LayerMetric[] }) {
  const parsed = useMemo(() => parseLayerNorms(metrics), [metrics]);
  const layerKeys = useMemo(() => getAllLayerKeys(parsed), [parsed]);
  const lineData = useMemo(() => buildLineData(parsed, layerKeys), [parsed, layerKeys]);
  const proportionData = useMemo(() => buildProportionData(parsed, layerKeys), [parsed, layerKeys]);
  const layerStats = useMemo(() => computeLayerStats(parsed, layerKeys), [parsed, layerKeys]);
  const [showTable, setShowTable] = useState(false);

  // No layer data yet
  if (parsed.length < 2) return null;

  const numericKeys = layerKeys.filter(k => k !== "embed" && k !== "head");

  return (
    <div className="mb-6 space-y-4">
      {/* Section header */}
      <div className="flex items-center gap-2">
        <h3 className="text-[0.7rem] font-bold uppercase tracking-wider text-text-muted">
          Per-Layer Analysis
        </h3>
        <span className="rounded bg-cyan-500/10 border border-cyan-500/20 px-1.5 py-0.5 text-[0.58rem] font-semibold text-cyan-400">
          {layerKeys.length} layers
        </span>
        <span className="text-[0.58rem] text-text-muted">
          {fmtNum(parsed.length)} data points
        </span>
      </div>

      {/* Heatmap */}
      <ChartPanel
        title="Gradient Norm Heatmap"
        helpText="Color intensity shows gradient norm magnitude per layer over training steps (log scale). Blue = low, yellow = high. Helps identify which layers are actively learning vs stagnant, and spots gradient concentration or vanishing."
      >
        <GradNormHeatmap data={parsed} layerKeys={layerKeys} />
        <div className="mt-2 flex items-center gap-4 text-[0.58rem] text-text-muted">
          <span>Low</span>
          <div className="flex h-2.5 flex-1 rounded overflow-hidden">
            {Array.from({ length: 20 }, (_, i) => {
              const t = i / 19;
              const h = 240 - t * 180;
              const s = 50 + t * 30;
              const l = 8 + t * 55;
              return <div key={i} className="flex-1" style={{ background: `hsl(${h}, ${s}%, ${l}%)` }} />;
            })}
          </div>
          <span>High</span>
        </div>
      </ChartPanel>

      {/* Per-layer grad norm lines */}
      <ChartPanel
        title="Per-Layer Gradient Norms"
        helpText="Line chart showing the gradient norm of each transformer layer over training. Healthy training shows all layers receiving gradients of similar magnitude. Divergence between layers can indicate vanishing/exploding gradients or architectural imbalance."
      >
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={lineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="step" tick={{ fill: "#6b7280", fontSize: 10 }} tickFormatter={v => fmtNum(v)} />
            <YAxis
              tick={{ fill: "#6b7280", fontSize: 10 }}
              tickFormatter={v => v < 0.01 ? v.toExponential(0) : v.toFixed(2)}
              scale="log"
              domain={["auto", "auto"]}
              allowDataOverflow
            />
            <RTooltip content={<LayerTooltip />} />
            {layerKeys.map((k, i) => (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={layerColor(k, numericKeys.indexOf(k))}
                strokeWidth={k === "embed" || k === "head" ? 1.5 : 1}
                dot={false}
                strokeOpacity={0.8}
                name={k === "embed" ? "Embed" : k === "head" ? "Head" : `L${k}`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Gradient distribution (stacked area) */}
      <ChartPanel
        title="Gradient Distribution Across Layers"
        helpText="Shows what proportion of total gradient norm each layer accounts for over time. Ideally, gradient should be distributed somewhat evenly. If one layer dominates, it may be learning faster or experiencing instability while others stagnate."
      >
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={proportionData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }} stackOffset="expand">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="step" tick={{ fill: "#6b7280", fontSize: 10 }} tickFormatter={v => fmtNum(v)} />
            <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <RTooltip content={<ProportionTooltip />} />
            {[...layerKeys].reverse().map((k, i) => (
              <Area
                key={k}
                type="monotone"
                dataKey={k}
                stackId="1"
                fill={layerColor(k, numericKeys.indexOf(k))}
                stroke={layerColor(k, numericKeys.indexOf(k))}
                fillOpacity={0.7}
                strokeWidth={0}
                name={k === "embed" ? "Embed" : k === "head" ? "Head" : `L${k}`}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Layer stats table */}
      <div className="rounded-lg border border-border bg-surface">
        <button
          onClick={() => setShowTable(!showTable)}
          className="flex w-full items-center gap-2 px-4 py-3 text-left text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted transition-colors hover:bg-surface-2/30"
        >
          <span className="text-[0.7rem]">{showTable ? "\u25BC" : "\u25B6"}</span>
          Layer Statistics
          <span className="ml-auto text-[0.58rem] font-normal normal-case">
            {layerStats.length} layers tracked
          </span>
        </button>
        {showTable && layerStats.length > 0 && (
          <div className="overflow-x-auto border-t border-border/50">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/50 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
                  <th className="px-3 py-2 text-left">Layer</th>
                  <th className="px-3 py-2 text-right">Avg Norm</th>
                  <th className="px-3 py-2 text-right">Recent Norm</th>
                  <th className="px-3 py-2 text-right">Max</th>
                  <th className="px-3 py-2 text-right">Min</th>
                  <th className="px-3 py-2 text-center">Trend</th>
                </tr>
              </thead>
              <tbody>
                {layerStats.map(s => (
                  <tr key={s.key} className="border-b border-border/20 last:border-0 hover:bg-surface-2/20">
                    <td className="px-3 py-2 font-mono font-semibold" style={{ color: s.color }}>{s.label}</td>
                    <td className="px-3 py-2 text-right font-mono text-text-secondary">{s.avgNorm.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right font-mono text-white">{s.recentNorm.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right font-mono text-text-muted">{s.maxNorm.toFixed(4)}</td>
                    <td className="px-3 py-2 text-right font-mono text-text-muted">{s.minNorm.toFixed(4)}</td>
                    <td className="px-3 py-2 text-center">
                      {s.trend === "up" && <span className="text-red-400" title="Gradient norm increasing">&#9650;</span>}
                      {s.trend === "down" && <span className="text-green-400" title="Gradient norm decreasing">&#9660;</span>}
                      {s.trend === "stable" && <span className="text-text-muted" title="Gradient norm stable">&#8212;</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
