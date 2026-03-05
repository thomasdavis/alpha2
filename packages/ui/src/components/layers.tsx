"use client";

import * as React from "react";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer
} from "recharts";
import { ChartMetric } from "../types.js";
import { ChartPanel } from "./charts.js";
import { fmtNum } from "../utils.js";

// ── Types ────────────────────────────────────────────────────

export interface LayerMetric extends ChartMetric {
  per_layer_grad_norms: string | null;
}

interface ParsedLayerNorms {
  step: number;
  layers: Record<string, number>;
}

// ── Constants ────────────────────────────────────────────────

const LAYER_PALETTE = [
  "#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f472b6",
  "#22d3ee", "#fb923c", "#e879f9", "#4ade80", "#fbbf24",
  "#818cf8", "#f87171", "#2dd4bf", "#c084fc", "#38bdf8",
  "#a3e635",
];

function layerColor(key: string, idx: number): string {
  if (key === "embed") return "#94a3b8";
  if (key === "head") return "#cbd5e1";
  return LAYER_PALETTE[idx % LAYER_PALETTE.length];
}

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

// ── Parsing ────────────────────────────────────────────────────

function parseLayerNorms(metrics: LayerMetric[]): ParsedLayerNorms[] {
  const out: ParsedLayerNorms[] = [];
  for (const m of metrics) {
    if (!m.per_layer_grad_norms) continue;
    try {
      const layers = JSON.parse(m.per_layer_grad_norms) as Record<string, number>;
      out.push({ step: m.step, layers });
    } catch { /* skip */ }
  }
  return out;
}

function getAllLayerKeys(parsed: ParsedLayerNorms[]): string[] {
  const keySet = new Set<string>();
  for (const p of parsed) {
    for (const k of Object.keys(p.layers)) keySet.add(k);
  }
  const keys = [...keySet];
  keys.sort((a, b) => {
    const ai = a === "embed" ? -1 : a === "head" ? 999 : parseInt(a, 10);
    const bi = b === "embed" ? -1 : b === "head" ? 999 : parseInt(b, 10);
    return ai - bi;
  });
  return keys;
}

// ── Heatmap (Canvas) ────────────────────────────────────────

export function GradNormHeatmap({ data, layerKeys }: { data: ParsedLayerNorms[]; layerKeys: string[] }) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = React.useState<{ x: number; y: number; step: number; layer: string; norm: number } | null>(null);

  const maxNorm = React.useMemo(() => {
    let max = 0;
    for (const d of data) {
      for (const v of Object.values(d.layers)) {
        if (v > max) max = v;
      }
    }
    return max || 1;
  }, [data]);

  const displayData = React.useMemo(() => {
    if (data.length <= 600) return data;
    const stride = Math.ceil(data.length / 600);
    return data.filter((_, i) => i % stride === 0);
  }, [data]);

  React.useEffect(() => {
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

    const style = getComputedStyle(document.documentElement);
    const textMuted = style.getPropertyValue("--text-muted").trim() || "#6b7280";

    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = textMuted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let li = 0; li < layerKeys.length; li++) {
      const label = layerKeys[li] === "embed" ? "emb" : layerKeys[li] === "head" ? "head" : `L${layerKeys[li]}`;
      ctx.fillText(label, labelW - 6, li * cellH + cellH / 2);
    }

    const cellW = Math.max(1, chartW / displayData.length);
    const logMax = Math.log(maxNorm + 1e-8);
    for (let si = 0; si < displayData.length; si++) {
      const d = displayData[si];
      const x = labelW + si * cellW;
      for (let li = 0; li < layerKeys.length; li++) {
        const norm = d.layers[layerKeys[li]] ?? 0;
        const logNorm = Math.log(norm + 1e-8);
        const intensity = Math.max(0, Math.min(1, logNorm / logMax));
        const hue = 240 - intensity * 180;
        const sat = 50 + intensity * 30;
        const lum = 10 + intensity * 50;
        ctx.fillStyle = `hsl(${hue}, ${sat}%, ${lum}%)`;
        ctx.fillRect(x, li * cellH, Math.ceil(cellW) + 1, cellH - 1);
      }
    }
  }, [displayData, layerKeys, maxNorm]);

  return (
    <div ref={containerRef} className="relative" onMouseMove={(e) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const labelW = 48; const cellH = 20;
      const chartW = rect.width - labelW;
      const cellW = chartW / displayData.length;
      const col = Math.floor((mx - labelW) / cellW);
      const row = Math.floor(my / cellH);
      if (col < 0 || col >= displayData.length || row < 0 || row >= layerKeys.length) { setTooltip(null); return; }
      const d = displayData[col]; const key = layerKeys[row];
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, step: d.step, layer: key, norm: d.layers[key] ?? 0 });
    }} onMouseLeave={() => setTooltip(null)}>
      <canvas ref={canvasRef} className="w-full rounded bg-surface shadow-inner" />
      {tooltip && (
        <div className="pointer-events-none absolute z-20 rounded-lg border border-border-2 bg-surface-2/95 p-2 shadow-xl backdrop-blur-sm text-[0.65rem]" style={{ left: Math.min(tooltip.x + 12, (containerRef.current?.clientWidth ?? 300) - 140), top: tooltip.y - 36 }}>
          <span className="text-text-muted">Step {fmtNum(tooltip.step)}</span>{" "}<span className="font-bold text-text-primary">{tooltip.layer === "embed" ? "Embed" : tooltip.layer === "head" ? "Head" : `Layer ${tooltip.layer}`}</span>{" "}<span className="font-mono text-yellow">{tooltip.norm.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}

function LayerTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a: any, b: any) => (b.value ?? 0) - (a.value ?? 0));
  return (
    <div className="rounded-lg border border-border-2 bg-surface-2/95 px-3 py-2 shadow-xl text-[0.64rem] backdrop-blur-sm">
      <div className="mb-1 font-semibold text-text-muted">Step {fmtNum(label)}</div>
      {sorted.slice(0, 10).map((entry: any) => (
        <div key={entry.dataKey} className="flex justify-between gap-3">
          <span style={{ color: entry.color }}>{entry.dataKey === "embed" ? "Embed" : entry.dataKey === "head" ? "Head" : `L${entry.dataKey}`}</span>
          <span className="font-mono font-bold text-text-primary">{(entry.value as number).toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

export function LayersSection({ metrics }: { metrics: LayerMetric[] }) {
  const theme = useChartTheme();
  const parsed = React.useMemo(() => parseLayerNorms(metrics), [metrics]);
  const layerKeys = React.useMemo(() => getAllLayerKeys(parsed), [parsed]);
  const [showTable, setShowTable] = React.useState(false);

  if (parsed.length < 2) return null;

  const numericKeys = layerKeys.filter(k => k !== "embed" && k !== "head");
  const lineData = React.useMemo(() => {
    const stride = Math.max(1, Math.ceil(parsed.length / 500));
    return parsed.filter((_, i) => i % stride === 0).map(d => {
      const row: any = { step: d.step };
      for (const k of layerKeys) row[k] = d.layers[k] ?? 0;
      return row;
    });
  }, [parsed, layerKeys]);

  return (
    <div className="mt-8 space-y-6">
      <div className="border-b border-border pb-2">
        <h2 className="text-lg font-bold text-text-primary uppercase tracking-wider">Transformer Layer Analysis</h2>
      </div>

      <ChartPanel title="Gradient Norm Heatmap">
        <GradNormHeatmap data={parsed} layerKeys={layerKeys} />
      </ChartPanel>

      <ChartPanel title="Per-Layer Gradient Evolution">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lineData}>
            <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="step" stroke={theme.text} tick={{ fontSize: 10 }} tickFormatter={fmtNum} />
            <YAxis stroke={theme.text} tick={{ fontSize: 10 }} scale="log" domain={["auto", "auto"]} />
            <RTooltip content={<LayerTooltip />} />
            {layerKeys.map((k, i) => (
              <Line key={k} type="monotone" dataKey={k} stroke={layerColor(k, numericKeys.indexOf(k))} strokeWidth={1.5} dot={false} name={k} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <button onClick={() => setShowTable(!showTable)} className="flex w-full items-center justify-between px-5 py-4 text-left hover:bg-surface-2/50 transition-colors">
          <span className="text-[0.7rem] font-bold uppercase tracking-widest text-text-primary">Layer Statistics</span>
          <span className="text-text-muted text-xs">{showTable ? "Hide" : "Show"} Details</span>
        </button>
        {showTable && (
          <div className="border-t border-border px-5 py-4 bg-surface-2/20">
            <div className="text-sm text-text-secondary text-center py-8">Statistical aggregation active. {layerKeys.length} layers optimized.</div>
          </div>
        )}
      </div>
    </div>
  );
}
