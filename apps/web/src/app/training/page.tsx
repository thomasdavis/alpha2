"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";

const SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL || "";

// ── Types ──────────────────────────────────────────────────────

interface StepMetric {
  step: number;
  loss: number;
  valLoss?: number | null;
  lr: number;
  gradNorm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
}

interface TrainConfig {
  iters: number;
  batchSize: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  gradClip: number;
  evalInterval: number;
  evalIters: number;
  seed: number;
  backend: string;
  tokenizer: string;
  optimizer: string;
}

interface ModelConfig {
  nLayer: number;
  nEmbd: number;
  nHead: number;
  vocabSize: number;
  blockSize: number;
  dropout?: number;
}

interface LiveRun {
  id: string;
  domain: string;
  status: string;
  totalIters: number;
  modelConfig: ModelConfig | null;
  trainConfig: TrainConfig | null;
  totalParams: number | null;
  metrics: StepMetric[];
  startedAt: string | null;
}

// ── Helpers ────────────────────────────────────────────────────

function formatParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

function formatNumber(n: number, decimals = 0): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

// ── Canvas chart ───────────────────────────────────────────────

function drawLossChart(canvas: HTMLCanvasElement, metrics: StepMetric[], totalIters: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx || metrics.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 12, right: 16, bottom: 32, left: 52 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const losses = metrics.map((m) => m.loss);
  const valPts = metrics.filter((m) => m.valLoss != null) as Array<StepMetric & { valLoss: number }>;
  const allVals = [...losses, ...valPts.map((v) => v.valLoss!)];
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const rangeL = maxL - minL || 1;
  const paddedMin = minL - rangeL * 0.05;
  const paddedMax = maxL + rangeL * 0.05;
  const paddedRange = paddedMax - paddedMin;
  const minStep = metrics[0].step;
  const maxStep = Math.max(metrics[metrics.length - 1].step, totalIters);
  const rangeS = maxStep - minStep || 1;

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;
  const sy = (loss: number) => pad.top + (1 - (loss - paddedMin) / paddedRange) * ch;

  // Background
  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (i / 5) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    const val = paddedMax - (i / 5) * paddedRange;
    ctx.fillStyle = "#444";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
  }

  // Step labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#444";
  ctx.font = "10px monospace";
  const stepTicks = [minStep, Math.round(minStep + rangeS * 0.25), Math.round(minStep + rangeS * 0.5), Math.round(minStep + rangeS * 0.75), maxStep];
  for (const s of stepTicks) {
    ctx.fillText(String(s), sx(s), h - pad.bottom + 16);
  }

  // Axis labels
  ctx.fillStyle = "#333";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("step", pad.left + cw / 2, h - 4);
  ctx.save();
  ctx.translate(10, pad.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("loss", 0, 0);
  ctx.restore();

  // Gradient fill under loss line
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
  grad.addColorStop(0, "rgba(245, 158, 11, 0.15)");
  grad.addColorStop(1, "rgba(245, 158, 11, 0.0)");
  ctx.beginPath();
  ctx.moveTo(sx(metrics[0].step), pad.top + ch);
  for (let i = 0; i < metrics.length; i++) {
    ctx.lineTo(sx(metrics[i].step), sy(metrics[i].loss));
  }
  ctx.lineTo(sx(metrics[metrics.length - 1].step), pad.top + ch);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Train loss line with glow
  ctx.shadowColor = "rgba(245, 158, 11, 0.4)";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  for (let i = 0; i < metrics.length; i++) {
    const x = sx(metrics[i].step);
    const y = sy(metrics[i].loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Highlight last point
  const lastM = metrics[metrics.length - 1];
  ctx.beginPath();
  ctx.arc(sx(lastM.step), sy(lastM.loss), 4, 0, Math.PI * 2);
  ctx.fillStyle = "#f59e0b";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(sx(lastM.step), sy(lastM.loss), 6, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(245, 158, 11, 0.3)";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Val loss line + dots
  if (valPts.length > 0) {
    ctx.shadowColor = "rgba(96, 165, 250, 0.3)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    for (let i = 0; i < valPts.length; i++) {
      const x = sx(valPts[i].step);
      const y = sy(valPts[i].valLoss!);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#60a5fa";
    for (const v of valPts) {
      ctx.beginPath();
      ctx.arc(sx(v.step), sy(v.valLoss!), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Legend
  const ly = h - 8;
  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(pad.left, ly - 1, 14, 2);
  ctx.fillStyle = "#666";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("train", pad.left + 18, ly + 3);
  if (valPts.length > 0) {
    ctx.fillStyle = "#60a5fa";
    ctx.beginPath();
    ctx.arc(pad.left + 60, ly, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#666";
    ctx.fillText("val", pad.left + 67, ly + 3);
  }
}

function LiveLossChart({ metrics, totalIters }: { metrics: StepMetric[]; totalIters: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current && metrics.length >= 2) {
      drawLossChart(canvasRef.current, metrics, totalIters);
    }
  }, [metrics, totalIters]);

  if (metrics.length < 2) {
    return (
      <div className="flex h-56 items-center justify-center rounded-lg border border-border/50 bg-[#0d0d0d] text-xs text-text-muted">
        <div className="flex flex-col items-center gap-2">
          <svg className="h-5 w-5 animate-pulse text-text-muted" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="1.5,12 5,6 8.5,9 14.5,3" /></svg>
          Waiting for data...
        </div>
      </div>
    );
  }

  return <canvas ref={canvasRef} className="h-56 w-full rounded-lg" />;
}

// ── Stat card ──────────────────────────────────────────────────

function Stat({ label, value, sub, color, tip }: { label: string; value: string; sub?: string; color?: string; tip?: string }) {
  return (
    <div className="rounded-lg border border-border/60 bg-surface-2/80 px-3 py-2.5">
      <div className={`font-mono text-sm font-bold ${color ?? "text-white"}`}>{value}</div>
      <div className="text-[0.6rem] uppercase tracking-wider text-text-muted">
        {label}
        {tip && <Tip text={tip} />}
      </div>
      {sub && <div className="mt-0.5 text-[0.6rem] text-text-muted">{sub}</div>}
    </div>
  );
}

// ── Detail row ─────────────────────────────────────────────────

function DetailRow({ label, value, tip }: { label: string; value: string | number | null | undefined; tip?: string }) {
  return (
    <div className="flex justify-between border-b border-border/30 py-1.5 last:border-0">
      <span className="text-[0.7rem] text-text-muted">
        {label}
        {tip && <Tip text={tip} />}
      </span>
      <span className="font-mono text-[0.7rem] text-text-primary">{value ?? "-"}</span>
    </div>
  );
}

// ── LiveRunCard ────────────────────────────────────────────────

function LiveRunCard({ run }: { run: LiveRun }) {
  const last = run.metrics[run.metrics.length - 1];
  const pct = last && run.totalIters > 0 ? Math.min(100, (last.step / run.totalIters) * 100) : 0;
  const mc = run.modelConfig;
  const tc = run.trainConfig;
  const isActive = run.status === "active";

  // Computed stats
  const stats = useMemo(() => {
    if (run.metrics.length === 0) return null;
    const losses = run.metrics.map((m) => m.loss);
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    const lossDrop = firstLoss - lastLoss;
    const lossDropPct = firstLoss > 0 ? (lossDrop / firstLoss) * 100 : 0;

    const recent = run.metrics.slice(-10);
    const avgTps = recent.reduce((s, m) => s + m.tokens_per_sec, 0) / recent.length;
    const avgMs = recent.reduce((s, m) => s + m.ms_per_iter, 0) / recent.length;
    const avgGradNorm = recent.reduce((s, m) => s + m.gradNorm, 0) / recent.length;

    const totalElapsed = run.metrics.reduce((s, m) => s + m.elapsed_ms, 0);
    const stepsRemaining = run.totalIters - (last?.step ?? 0);
    const eta = stepsRemaining > 0 ? avgMs * stepsRemaining : 0;

    const valLosses = run.metrics.filter((m) => m.valLoss != null);
    const bestVal = valLosses.length > 0 ? Math.min(...valLosses.map((m) => m.valLoss!)) : null;
    const lastVal = valLosses.length > 0 ? valLosses[valLosses.length - 1].valLoss : null;

    return {
      minLoss, maxLoss, firstLoss, lastLoss, lossDrop, lossDropPct,
      avgTps, avgMs, avgGradNorm,
      totalElapsed, eta,
      bestVal, lastVal,
      totalTokens: run.metrics.reduce((s, m) => s + (m.tokens_per_sec * m.elapsed_ms / 1000), 0),
    };
  }, [run.metrics, run.totalIters, last]);

  const domainColors: Record<string, string> = {
    novels: "bg-blue-bg text-blue border-blue/20",
    chords: "bg-yellow-bg text-yellow border-yellow/20",
    abc: "bg-green-bg text-green border-green/20",
    dumb_finance: "bg-red-bg text-red border-red/20",
  };

  return (
    <div className={`overflow-hidden rounded-xl border ${isActive ? "border-green/20" : "border-border"} bg-surface`}>
      {/* Header with gradient */}
      <div className={`border-b px-5 py-4 ${isActive ? "border-green/10 bg-gradient-to-r from-green-bg/50 to-transparent" : "border-border bg-surface"}`}>
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="font-mono text-base font-bold text-white">{run.id}</span>
          <span className={`rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${domainColors[run.domain] ?? "border-border bg-surface-2 text-text-secondary"}`}>
            {run.domain}
          </span>
          <span className={`flex items-center gap-1 rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${isActive ? "border-green/20 bg-green-bg text-green" : "border-blue/20 bg-blue-bg text-blue"}`}>
            {isActive && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green" />}
            {run.status}
          </span>
          {run.totalParams != null && (
            <span className="rounded-md border border-border bg-surface-2 px-2 py-0.5 text-[0.65rem] font-semibold text-text-secondary">
              {formatParams(run.totalParams)} params
            </span>
          )}
          {stats && (
            <span className="ml-auto text-xs text-text-muted">
              {formatDuration(stats.totalElapsed)} elapsed
              {isActive && stats.eta > 0 && <> &middot; ~{formatDuration(stats.eta)} remaining</>}
            </span>
          )}
        </div>
      </div>

      <div className="p-5">
        {/* Progress bar */}
        <div className="mb-5">
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="text-xs text-text-secondary">
              Step <span className="font-mono font-bold text-white">{last ? formatNumber(last.step) : "0"}</span>
              <span className="text-text-muted"> / {formatNumber(run.totalIters)}</span>
            </span>
            <span className="font-mono text-sm font-bold text-white">{pct.toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-2">
            <div
              className={`h-full rounded-full transition-all duration-700 ease-out ${isActive ? "bg-gradient-to-r from-green/80 to-green" : "bg-gradient-to-r from-blue/80 to-blue"}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {/* Primary metrics - two rows */}
        <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
          <Stat label="Loss" value={last ? last.loss.toFixed(4) : "-"} color="text-yellow" tip={tips.loss} />
          <Stat label="Best Loss" value={stats ? stats.minLoss.toFixed(4) : "-"} sub={stats ? `${stats.lossDropPct > 0 ? "-" : ""}${Math.abs(stats.lossDropPct).toFixed(1)}% from start` : undefined} tip={tips.lastLoss} />
          <Stat label="Val Loss" value={stats?.lastVal != null ? stats.lastVal.toFixed(4) : "-"} sub={stats?.bestVal != null ? `best: ${stats.bestVal.toFixed(4)}` : undefined} color={stats?.lastVal != null ? "text-blue" : undefined} tip={tips.valLoss} />
          <Stat label="Learning Rate" value={last ? last.lr.toExponential(2) : "-"} tip={tips.lr} />
          <Stat label="Throughput" value={stats ? `${formatNumber(stats.avgTps, 0)}` : "-"} sub="tok/s (avg)" color="text-green" tip={tips.throughput} />
          <Stat label="Speed" value={stats ? `${formatNumber(stats.avgMs, 0)}` : "-"} sub="ms/iter (avg)" tip={tips.msPerIter} />
          <Stat label="Grad Norm" value={last ? last.gradNorm.toFixed(3) : "-"} sub={stats ? `avg: ${stats.avgGradNorm.toFixed(3)}` : undefined} tip={tips.gradNorm} />
          <Stat label="Tokens" value={stats ? formatParams(Math.round(stats.totalTokens)) : "-"} sub="processed" />
        </div>

        {/* Chart + side panels */}
        <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
          {/* Loss chart */}
          <div>
            <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
              Loss Curve <Tip text={tips.lossChart} />
            </div>
            <LiveLossChart metrics={run.metrics} totalIters={run.totalIters} />
          </div>

          {/* Architecture & config */}
          <div className="space-y-3">
            {mc && (
              <div className="rounded-lg border border-border/60 bg-surface-2/50 p-3">
                <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
                  Architecture
                </div>
                <DetailRow label="Layers" value={mc.nLayer} tip={tips.nLayer} />
                <DetailRow label="Embedding" value={mc.nEmbd} tip={tips.nEmbd} />
                <DetailRow label="Heads" value={mc.nHead} tip={tips.nHead} />
                <DetailRow label="Vocab" value={formatNumber(mc.vocabSize)} tip={tips.vocabSize} />
                <DetailRow label="Context" value={mc.blockSize} tip={tips.blockSize} />
                {mc.dropout != null && <DetailRow label="Dropout" value={mc.dropout} tip={tips.dropout} />}
                {run.totalParams != null && <DetailRow label="Parameters" value={formatParams(run.totalParams)} tip={tips.params} />}
              </div>
            )}

            {tc && (
              <div className="rounded-lg border border-border/60 bg-surface-2/50 p-3">
                <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
                  Training Config
                </div>
                <DetailRow label="Batch size" value={tc.batchSize} tip={tips.batchSize} />
                <DetailRow label="Max LR" value={tc.lr} tip={tips.lr} />
                <DetailRow label="Optimizer" value={tc.optimizer} tip={tips.optimizer} />
                <DetailRow label="Weight decay" value={tc.weightDecay} tip={tips.weightDecay} />
                <DetailRow label="Grad clip" value={tc.gradClip} tip={tips.gradClip} />
                <DetailRow label="Backend" value={tc.backend} tip={tips.backend} />
                <DetailRow label="Tokenizer" value={tc.tokenizer} tip={tips.tokenizer} />
                <DetailRow label="Seed" value={tc.seed} tip={tips.seed} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────

export default function TrainingPage() {
  const [runs, setRuns] = useState<Map<string, LiveRun>>(new Map());
  const [connected, setConnected] = useState(false);
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    const es = new EventSource(`${SERVER_URL}/api/training/live`);

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.addEventListener("snapshot", (e) => {
      const data = JSON.parse(e.data) as Array<Record<string, any>>;
      const next = new Map<string, LiveRun>();
      for (const r of data) {
        const parsedModel = r.model_config ? JSON.parse(r.model_config) : null;
        const parsedTrain = r.train_config ? JSON.parse(r.train_config) : null;
        const metrics: StepMetric[] = (r.metrics ?? []).map((m: any) => ({
          step: m.step,
          loss: m.loss,
          valLoss: m.val_loss ?? m.valLoss ?? null,
          lr: m.lr,
          gradNorm: m.grad_norm ?? m.gradNorm ?? 0,
          elapsed_ms: m.elapsed_ms ?? 0,
          tokens_per_sec: m.tokens_per_sec ?? 0,
          ms_per_iter: m.ms_per_iter ?? 0,
        }));
        next.set(r.id, {
          id: r.id,
          domain: r.domain ?? "unknown",
          status: r.status ?? "active",
          totalIters: r.total_iters ?? parsedTrain?.iters ?? 0,
          modelConfig: parsedModel ? {
            nLayer: parsedModel.nLayer,
            nEmbd: parsedModel.nEmbd,
            nHead: parsedModel.nHead,
            vocabSize: parsedModel.vocabSize,
            blockSize: parsedModel.blockSize,
            dropout: parsedModel.dropout,
          } : null,
          trainConfig: parsedTrain,
          totalParams: r.estimated_params ?? null,
          metrics,
          startedAt: r.created_at ?? null,
        });
      }
      setRuns(next);
    });

    es.addEventListener("run_start", (e) => {
      const data = JSON.parse(e.data);
      setRuns((prev) => {
        const next = new Map(prev);
        next.set(data.runId, {
          id: data.runId,
          domain: data.domain ?? "unknown",
          status: "active",
          totalIters: data.trainConfig?.iters ?? 0,
          modelConfig: data.modelConfig ?? null,
          trainConfig: data.trainConfig ?? null,
          totalParams: data.totalParams ?? null,
          metrics: [],
          startedAt: new Date().toISOString(),
        });
        return next;
      });
    });

    es.addEventListener("metrics", (e) => {
      const data = JSON.parse(e.data);
      const { runId, metrics } = data as { runId: string; metrics: StepMetric[] };
      setRuns((prev) => {
        const next = new Map(prev);
        const existing = next.get(runId);
        if (existing) {
          next.set(runId, {
            ...existing,
            metrics: [...existing.metrics, ...metrics],
          });
        }
        return next;
      });
    });

    es.addEventListener("run_complete", (e) => {
      const data = JSON.parse(e.data);
      setRuns((prev) => {
        const next = new Map(prev);
        const existing = next.get(data.runId);
        if (existing) {
          next.set(data.runId, { ...existing, status: "completed" });
        }
        return next;
      });
      setCompletedIds((prev) => new Set(prev).add(data.runId));
    });

    return () => es.close();
  }, []);

  const activeRuns = Array.from(runs.values()).filter((r) => r.status === "active");
  const recentlyCompleted = Array.from(runs.values()).filter(
    (r) => r.status === "completed" && completedIds.has(r.id)
  );

  return (
    <>
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <h1 className="text-xl font-bold text-white">Live Training</h1>
        <span
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
            connected
              ? "border-green/20 bg-green-bg text-green"
              : "border-red/20 bg-red-bg text-red"
          }`}
        >
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              connected ? "animate-pulse bg-green" : "bg-red"
            }`}
          />
          {connected ? "Connected" : "Disconnected"}
        </span>
        {activeRuns.length > 0 && (
          <span className="text-xs text-text-muted">
            {activeRuns.length} active run{activeRuns.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Active runs */}
      {activeRuns.length === 0 && recentlyCompleted.length === 0 ? (
        <div className="rounded-xl border border-border bg-surface p-10 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-border bg-surface-2">
            <svg className="h-6 w-6 text-text-muted" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="8" cy="8" r="3" />
              <path d="M4.5 4.5L2 2M11.5 4.5L14 2M4.5 11.5L2 14M11.5 11.5L14 14" />
            </svg>
          </div>
          <div className="mb-1 text-sm font-medium text-text-secondary">No active training runs</div>
          <div className="mb-4 text-xs text-text-muted">
            Set these environment variables on the training machine to stream metrics here:
          </div>
          <pre className="mx-auto inline-block rounded-lg border border-border bg-[#0d0d0d] px-5 py-3 text-left font-mono text-xs leading-relaxed text-text-primary">
            <span className="text-text-muted">export </span><span className="text-green">ALPHA_REMOTE_URL</span>=https://alpha.omegaai.dev{"\n"}
            <span className="text-text-muted">export </span><span className="text-green">ALPHA_REMOTE_SECRET</span>=&lt;your-secret&gt;
          </pre>
          <div className="mt-4 text-xs text-text-muted">
            Then run <code className="rounded bg-surface-2 px-1.5 py-0.5 text-text-secondary">alpha train --data=... --domain=novels</code> and watch it here.
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          {activeRuns.map((run) => (
            <LiveRunCard key={run.id} run={run} />
          ))}
        </div>
      )}

      {/* Recently completed */}
      {recentlyCompleted.length > 0 && (
        <div className="mt-8">
          <h2 className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wider text-text-muted">
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="2,8 6,12 14,4" /></svg>
            Recently Completed
          </h2>
          <div className="space-y-5">
            {recentlyCompleted.map((run) => (
              <LiveRunCard key={run.id} run={run} />
            ))}
          </div>
        </div>
      )}
    </>
  );
}
