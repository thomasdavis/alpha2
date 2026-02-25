"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";
import {
  type ChartMetric,
  Stat, DetailRow, InteractiveLossChart, MiniChart, StepTimeChart,
  buildGpuSeries, buildLrSeries, buildGradNormSeries,
  fmtParams, fmtDuration, fmtNum,
} from "@/components/charts";

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
  gpu_util_pct?: number | null;
  gpu_vram_used_mb?: number | null;
  gpu_vram_total_mb?: number | null;
  gpu_mem_pool_mb?: number | null;
  // Per-step timing breakdown
  timing_fwd_ms?: number | null;
  timing_bwd_ms?: number | null;
  timing_optim_ms?: number | null;
  timing_data_ms?: number | null;
  timing_flush_ms?: number | null;
  timing_grad_norm_ms?: number | null;
  timing_grad_clip_ms?: number | null;
  gpu_ops_count?: number | null;
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

// ── StepMetric → ChartMetric converter ─────────────────────────

function toChartMetric(m: StepMetric): ChartMetric {
  return {
    step: m.step,
    loss: m.loss,
    val_loss: m.valLoss ?? null,
    lr: m.lr,
    grad_norm: m.gradNorm,
    elapsed_ms: m.elapsed_ms,
    tokens_per_sec: m.tokens_per_sec,
    ms_per_iter: m.ms_per_iter,
    gpu_util_pct: m.gpu_util_pct ?? null,
    gpu_vram_used_mb: m.gpu_vram_used_mb ?? null,
    gpu_vram_total_mb: m.gpu_vram_total_mb ?? null,
    timing_fwd_ms: m.timing_fwd_ms ?? null,
    timing_bwd_ms: m.timing_bwd_ms ?? null,
    timing_optim_ms: m.timing_optim_ms ?? null,
    timing_data_ms: m.timing_data_ms ?? null,
    timing_flush_ms: m.timing_flush_ms ?? null,
    timing_grad_norm_ms: m.timing_grad_norm_ms ?? null,
    timing_grad_clip_ms: m.timing_grad_clip_ms ?? null,
    gpu_ops_count: m.gpu_ops_count ?? null,
  };
}

// ── LiveRunCard ────────────────────────────────────────────────

function LiveRunCard({ run }: { run: LiveRun }) {
  const last = run.metrics[run.metrics.length - 1];
  const pct = last && run.totalIters > 0 ? Math.min(100, (last.step / run.totalIters) * 100) : 0;
  const mc = run.modelConfig;
  const tc = run.trainConfig;
  const isActive = run.status === "active";

  const chartMetrics = useMemo(() => run.metrics.map(toChartMetric), [run.metrics]);

  // Computed stats
  const stats = useMemo(() => {
    if (run.metrics.length === 0) return null;
    const losses = run.metrics.map((m) => m.loss);
    const minLoss = Math.min(...losses);
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    const lossDropPct = firstLoss > 0 ? ((firstLoss - lastLoss) / firstLoss) * 100 : 0;

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

    // Timing breakdown (from recent steps with timing data)
    const timedSteps = recent.filter((m) => m.timing_fwd_ms != null);
    const avgFwd = timedSteps.length > 0 ? timedSteps.reduce((s, m) => s + (m.timing_fwd_ms ?? 0), 0) / timedSteps.length : null;
    const avgBwd = timedSteps.length > 0 ? timedSteps.reduce((s, m) => s + (m.timing_bwd_ms ?? 0), 0) / timedSteps.length : null;
    const avgFlush = timedSteps.length > 0 ? timedSteps.reduce((s, m) => s + (m.timing_flush_ms ?? 0), 0) / timedSteps.length : null;
    const avgGpuOps = timedSteps.length > 0 ? timedSteps.reduce((s, m) => s + (m.gpu_ops_count ?? 0), 0) / timedSteps.length : null;

    // MFU estimation: 6 * params * tokens_per_step / (step_time * peak_flops)
    const totalParams = run.totalParams ?? 0;
    const tokensPerStep = (run.trainConfig?.batchSize ?? 1) * (run.modelConfig?.blockSize ?? 256);
    const flopsPerStep = 6 * totalParams * tokensPerStep;
    const mfu = avgMs > 0 && totalParams > 0 ? (flopsPerStep / (avgMs / 1000)) / 30.3e12 * 100 : null; // L4 FP32

    return {
      minLoss, lossDropPct,
      avgTps, avgMs, avgGradNorm,
      totalElapsed, eta,
      bestVal, lastVal,
      totalTokens: run.metrics.reduce((s, m) => s + (m.tokens_per_sec * m.elapsed_ms / 1000), 0),
      avgFwd, avgBwd, avgFlush, avgGpuOps, mfu,
    };
  }, [run.metrics, run.totalIters, last]);

  const domainColors: Record<string, string> = {
    novels: "bg-blue-bg text-blue border-blue/20",
    chords: "bg-yellow-bg text-yellow border-yellow/20",
    abc: "bg-green-bg text-green border-green/20",
    dumb_finance: "bg-red-bg text-red border-red/20",
    chaos: "bg-purple-bg text-purple border-purple/20",
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
              {fmtParams(run.totalParams)} params
            </span>
          )}
          {stats && (
            <span className="ml-auto text-xs text-text-muted">
              {fmtDuration(stats.totalElapsed)} elapsed
              {isActive && stats.eta > 0 && <> &middot; ~{fmtDuration(stats.eta)} remaining</>}
            </span>
          )}
        </div>
      </div>

      <div className="p-5">
        {/* Progress bar */}
        <div className="mb-5">
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="text-xs text-text-secondary">
              Step <span className="font-mono font-bold text-white">{last ? fmtNum(last.step) : "0"}</span>
              <span className="text-text-muted"> / {fmtNum(run.totalIters)}</span>
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
          <Stat label="Loss" value={last?.loss != null ? last.loss.toFixed(4) : "-"} color="text-yellow" tip={tips.loss} />
          <Stat label="Best Loss" value={stats && isFinite(stats.minLoss) ? stats.minLoss.toFixed(4) : "-"} sub={stats && isFinite(stats.lossDropPct) ? `${stats.lossDropPct > 0 ? "-" : ""}${Math.abs(stats.lossDropPct).toFixed(1)}% from start` : undefined} tip={tips.lastLoss} />
          <Stat label="Val Loss" value={stats?.lastVal != null ? stats.lastVal.toFixed(4) : "-"} sub={stats?.bestVal != null ? `best: ${stats.bestVal.toFixed(4)}` : undefined} color={stats?.lastVal != null ? "text-blue" : undefined} tip={tips.valLoss} />
          <Stat label="Learning Rate" value={last?.lr != null ? last.lr.toExponential(2) : "-"} tip={tips.lr} />
          <Stat label="Throughput" value={stats ? `${fmtNum(stats.avgTps, 0)}` : "-"} sub="tok/s (avg)" color="text-green" tip={tips.throughput} />
          <Stat label="Speed" value={stats ? `${fmtNum(stats.avgMs, 0)}` : "-"} sub="ms/iter (avg)" tip={tips.msPerIter} />
          <Stat label="Grad Norm" value={last?.gradNorm != null ? last.gradNorm.toFixed(3) : "-"} sub={stats && isFinite(stats.avgGradNorm) ? `avg: ${stats.avgGradNorm.toFixed(3)}` : undefined} tip={tips.gradNorm} />
          <Stat label="Tokens" value={stats ? fmtParams(Math.round(stats.totalTokens)) : "-"} sub="processed" />
        </div>

        {/* Timing stats row (only shows when timing data is available) */}
        {stats?.avgFwd != null && (
          <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
            <Stat label="Forward" value={`${stats.avgFwd.toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${(stats.avgFwd / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-cyan-400" />
            <Stat label="Backward" value={`${(stats.avgBwd ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgBwd ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-orange-400" />
            <Stat label="GPU Sync" value={`${(stats.avgFlush ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgFlush ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-rose-400" />
            <Stat label="GPU Ops" value={stats.avgGpuOps != null ? fmtNum(stats.avgGpuOps, 0) : "-"} sub="per step" />
            <Stat label="MFU" value={stats.mfu != null ? `${stats.mfu.toFixed(1)}%` : "-"} sub="model FLOPS util" color={stats.mfu != null && stats.mfu > 50 ? "text-green" : stats.mfu != null && stats.mfu > 10 ? "text-yellow" : "text-red"} />
            <Stat label="Bwd/Fwd" value={stats.avgFwd != null && stats.avgBwd != null ? `${(stats.avgBwd / stats.avgFwd).toFixed(1)}x` : "-"} sub="ratio" />
          </div>
        )}

        {/* Chart + side panels */}
        <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
          {/* Loss chart */}
          <div>
            <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
              Loss Curve <Tip text={tips.lossChart} />
            </div>
            <InteractiveLossChart metrics={chartMetrics} checkpoints={[]} />
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
                <DetailRow label="Vocab" value={fmtNum(mc.vocabSize)} tip={tips.vocabSize} />
                <DetailRow label="Context" value={mc.blockSize} tip={tips.blockSize} />
                {mc.dropout != null && <DetailRow label="Dropout" value={mc.dropout} tip={tips.dropout} />}
                {run.totalParams != null && <DetailRow label="Parameters" value={fmtParams(run.totalParams)} tip={tips.params} />}
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

        {/* Mini charts row */}
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MiniChart
            metrics={chartMetrics}
            title="GPU & VRAM"
            noDataMsg="No GPU data"
            formatLeft={(v) => (v / 1024).toFixed(1) + "G"}
            formatRight={(v) => v.toFixed(0) + "%"}
            buildSeries={buildGpuSeries}
          />
          <MiniChart
            metrics={chartMetrics}
            title="Learning Rate"
            formatLeft={(v) => v.toExponential(1)}
            buildSeries={buildLrSeries}
          />
          <MiniChart
            metrics={chartMetrics}
            title="Grad Norm"
            logScale
            formatLeft={(v) => v.toExponential(0)}
            buildSeries={buildGradNormSeries}
          />
          <StepTimeChart metrics={chartMetrics} />
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
          step: m.step ?? 0,
          loss: m.loss ?? 0,
          valLoss: m.val_loss ?? m.valLoss ?? null,
          lr: m.lr ?? 0,
          gradNorm: m.grad_norm ?? m.gradNorm ?? 0,
          elapsed_ms: m.elapsed_ms ?? 0,
          tokens_per_sec: m.tokens_per_sec ?? 0,
          ms_per_iter: m.ms_per_iter ?? 0,
          gpu_util_pct: m.gpu_util_pct ?? null,
          gpu_vram_used_mb: m.gpu_vram_used_mb ?? null,
          gpu_vram_total_mb: m.gpu_vram_total_mb ?? null,
          gpu_mem_pool_mb: m.gpu_mem_pool_mb ?? null,
          timing_fwd_ms: m.timing_fwd_ms ?? null,
          timing_bwd_ms: m.timing_bwd_ms ?? null,
          timing_optim_ms: m.timing_optim_ms ?? null,
          timing_data_ms: m.timing_data_ms ?? null,
          timing_flush_ms: m.timing_flush_ms ?? null,
          timing_grad_norm_ms: m.timing_grad_norm_ms ?? null,
          timing_grad_clip_ms: m.timing_grad_clip_ms ?? null,
          gpu_ops_count: m.gpu_ops_count ?? null,
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
      const { runId, metrics: rawMetrics } = data as { runId: string; metrics: any[] };
      const normalized: StepMetric[] = (rawMetrics ?? []).map((m: any) => ({
        step: m.step ?? 0,
        loss: m.loss ?? 0,
        valLoss: m.val_loss ?? m.valLoss ?? null,
        lr: m.lr ?? 0,
        gradNorm: m.grad_norm ?? m.gradNorm ?? 0,
        elapsed_ms: m.elapsed_ms ?? 0,
        tokens_per_sec: m.tokens_per_sec ?? 0,
        ms_per_iter: m.ms_per_iter ?? 0,
        gpu_util_pct: m.gpu_util_pct ?? null,
        gpu_vram_used_mb: m.gpu_vram_used_mb ?? null,
        gpu_vram_total_mb: m.gpu_vram_total_mb ?? null,
        gpu_mem_pool_mb: m.gpu_mem_pool_mb ?? null,
        timing_fwd_ms: m.timing_fwd_ms ?? null,
        timing_bwd_ms: m.timing_bwd_ms ?? null,
        timing_optim_ms: m.timing_optim_ms ?? null,
        timing_data_ms: m.timing_data_ms ?? null,
        timing_flush_ms: m.timing_flush_ms ?? null,
        timing_grad_norm_ms: m.timing_grad_norm_ms ?? null,
        timing_grad_clip_ms: m.timing_grad_clip_ms ?? null,
        gpu_ops_count: m.gpu_ops_count ?? null,
      }));
      setRuns((prev) => {
        const next = new Map(prev);
        const existing = next.get(runId);
        if (existing) {
          next.set(runId, {
            ...existing,
            metrics: [...existing.metrics, ...normalized],
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
