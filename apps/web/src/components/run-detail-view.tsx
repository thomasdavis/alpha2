"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";
import {
  type ChartMetric,
  Stat, DetailRow, InteractiveLossChart, MiniChart, StepTimeChart, ChartPanel,
  buildGpuSeries, buildLrSeries, buildGradNormSeries,
  fmtParams, fmtLoss, fmtBytes, fmtDuration, fmtNum, timeAgo, fmtDate,
} from "@/components/charts";
import { SymbioSection, extractActivationSwitchEvents } from "@/components/symbio-charts";

// ── Types ────────────────────────────────────────────────────────

interface MetricData extends ChartMetric {
  gpu_mem_pool_mb: number | null;
}

interface CheckpointData {
  step: number;
  filename: string;
  file_size: number | null;
  created_at: string;
}

interface SampleData {
  idx: number;
  prompt: string;
  output: string;
  created_at: string;
}

interface RunData {
  id: string;
  run_id: string;
  config_hash: string;
  domain: string;
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_embd: number;
  n_head: number;
  dropout: number;
  total_iters: number;
  batch_size: number;
  lr: number;
  seed: number;
  backend: string;
  tokenizer: string;
  optimizer: string;
  model_config: string;
  train_config: string;
  status: string;
  latest_step: number;
  last_loss: number | null;
  best_val_loss: number | null;
  estimated_params: number | null;
  created_at: string;
  updated_at: string;
  disk_mtime: number | null;
  symbio?: number | null;
  symbio_config?: string | null;
  ffn_activation?: string | null;
  symbio_mode?: string | null;
}

export interface RunDetailProps {
  run: RunData;
  metrics: MetricData[];
  checkpoints: CheckpointData[];
  samples: SampleData[];
}

// ── Constants ────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, { badge: string; bar: string; gradient: string }> = {
  active: {
    badge: "border-green/20 bg-green-bg text-green",
    bar: "bg-gradient-to-r from-green/80 to-green",
    gradient: "from-green-bg/50",
  },
  completed: {
    badge: "border-blue/20 bg-blue-bg text-blue",
    bar: "bg-gradient-to-r from-blue/80 to-blue",
    gradient: "from-blue-bg/50",
  },
  stale: {
    badge: "border-yellow/20 bg-yellow-bg text-yellow",
    bar: "bg-gradient-to-r from-yellow/80 to-yellow",
    gradient: "from-yellow-bg/50",
  },
  failed: {
    badge: "border-red/20 bg-red-bg text-red",
    bar: "bg-gradient-to-r from-red/80 to-red",
    gradient: "from-red-bg/50",
  },
};

const DOMAIN_STYLES: Record<string, string> = {
  novels: "border-blue/20 bg-blue-bg text-blue",
  chords: "border-yellow/20 bg-yellow-bg text-yellow",
  abc: "border-green/20 bg-green-bg text-green",
  dumb_finance: "border-red/20 bg-red-bg text-red",
  concordance: "border-cyan-500/20 bg-cyan-950 text-cyan-400",
};

// ── Main Component ───────────────────────────────────────────────

const POLL_INTERVAL = 15_000;

export function RunDetailView({ run, metrics: initialMetrics, checkpoints: initialCheckpoints, samples: initialSamples }: RunDetailProps) {
  const [metrics, setMetrics] = useState(initialMetrics);
  const [checkpoints, setCheckpoints] = useState(initialCheckpoints);
  const [samples, setSamples] = useState(initialSamples);
  const [showModelConfig, setShowModelConfig] = useState(true);
  const [showTrainConfig, setShowTrainConfig] = useState(true);
  const [copiedJson, setCopiedJson] = useState(false);
  const [pinnedStep, setPinnedStep] = useState<number | null>(null);

  const handlePinStep = useCallback((step: number) => {
    setPinnedStep(prev => prev === step ? null : step);
  }, []);

  const poll = useCallback(async () => {
    try {
      const base = `/api/runs/${encodeURIComponent(run.id)}`;
      const [m, c, s] = await Promise.all([
        fetch(`${base}/metrics`).then((r) => r.ok ? r.json() : null),
        fetch(`${base}/checkpoints`).then((r) => r.ok ? r.json() : null),
        fetch(`${base}/samples`).then((r) => r.ok ? r.json() : null),
      ]);
      if (m) setMetrics(m);
      if (c) setCheckpoints(c);
      if (s) setSamples(s);
    } catch {}
  }, [run.id]);

  useEffect(() => {
    const id = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [poll]);

  const modelConfig = useMemo(() => {
    try { return JSON.parse(run.model_config); } catch { return null; }
  }, [run.model_config]);

  const trainConfig = useMemo(() => {
    try { return JSON.parse(run.train_config); } catch { return null; }
  }, [run.train_config]);

  const activationSwitches = useMemo(() => extractActivationSwitchEvents(metrics as any), [metrics]);

  const copyMetricsJson = useCallback(() => {
    const data = { run, metrics, checkpoints, samples };
    navigator.clipboard.writeText(JSON.stringify(data, null, 2)).then(() => {
      setCopiedJson(true);
      setTimeout(() => setCopiedJson(false), 2000);
    });
  }, [run, metrics, checkpoints, samples]);

  const last = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const isActive = run.status === "active";
  const ss = STATUS_STYLES[run.status] ?? STATUS_STYLES.stale;
  const progress = run.total_iters > 0 ? Math.min(100, (run.latest_step / run.total_iters) * 100) : 0;

  const stats = useMemo(() => {
    if (metrics.length === 0) return null;
    const losses = metrics.map((m) => m.loss);
    const minLoss = Math.min(...losses);
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    const lossDropPct = firstLoss > 0 ? ((firstLoss - lastLoss) / firstLoss) * 100 : 0;

    const recent = metrics.slice(-20);
    const avgTps = recent.reduce((s, m) => s + m.tokens_per_sec, 0) / recent.length;
    const avgMs = recent.reduce((s, m) => s + m.ms_per_iter, 0) / recent.length;
    const avgGradNorm = recent.reduce((s, m) => s + m.grad_norm, 0) / recent.length;

    const totalElapsed = metrics.reduce((s, m) => s + m.elapsed_ms, 0);
    const stepsRemaining = run.total_iters - (last?.step ?? 0);
    const eta = stepsRemaining > 0 && avgMs > 0 ? avgMs * stepsRemaining : 0;

    const valLosses = metrics.filter((m) => m.val_loss != null);
    const bestVal = valLosses.length > 0 ? Math.min(...valLosses.map((m) => m.val_loss!)) : null;
    const lastVal = valLosses.length > 0 ? valLosses[valLosses.length - 1].val_loss : null;

    const totalTokens = metrics.reduce((s, m) => s + (m.tokens_per_sec * m.elapsed_ms / 1000), 0);

    // Timing stats
    const timed = recent.filter((m) => m.timing_fwd_ms != null);
    const avgFwd = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_fwd_ms ?? 0), 0) / timed.length : null;
    const avgBwd = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_bwd_ms ?? 0), 0) / timed.length : null;
    const avgFlush = timed.length > 0 ? timed.reduce((s, m) => s + (m.timing_flush_ms ?? 0), 0) / timed.length : null;
    const avgGpuOps = timed.length > 0 ? timed.reduce((s, m) => s + (m.gpu_ops_count ?? 0), 0) / timed.length : null;

    // MFU: 6 * params * tokens_per_step / (step_time * peak_flops)
    const totalParams = run.estimated_params ?? 0;
    const tokensPerStep = run.batch_size * run.block_size;
    const flopsPerStep = 6 * totalParams * tokensPerStep;
    const mfu = avgMs > 0 && totalParams > 0 ? (flopsPerStep / (avgMs / 1000)) / 30.3e12 * 100 : null;

    return {
      minLoss, lossDropPct, avgTps, avgMs, avgGradNorm, totalElapsed, eta,
      bestVal, lastVal, totalTokens,
      avgFwd, avgBwd, avgFlush, avgGpuOps, mfu,
    };
  }, [metrics, run, last]);

  return (
    <>
      {/* Breadcrumb */}
      <nav className="mb-4 flex items-center gap-1.5 text-xs text-text-muted">
        <Link href="/runs" className="hover:text-text-secondary">Training Runs</Link>
        <span>/</span>
        <span className="text-text-secondary">{run.id}</span>
      </nav>

      {/* Header */}
      <div className={`mb-6 overflow-hidden rounded-xl border ${isActive ? "border-green/20" : "border-border"} bg-surface`}>
        <div className={`h-0.5 ${ss.bar}`} />
        <div className={`border-b border-border/50 bg-gradient-to-r ${ss.gradient} to-transparent px-5 py-4`}>
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="font-mono text-lg font-bold text-white">{run.id}</span>
            <span className={`flex items-center gap-1 rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${ss.badge}`}>
              {isActive && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green" />}
              {run.status}
            </span>
            <span className={`rounded-md border px-2 py-0.5 text-[0.65rem] font-bold uppercase ${DOMAIN_STYLES[run.domain] ?? "border-border bg-surface-2 text-text-secondary"}`}>
              {run.domain}
            </span>
            <span className="rounded-md border border-border bg-surface-2 px-2 py-0.5 text-[0.65rem] font-semibold text-text-secondary">
              {fmtParams(run.estimated_params)} params
            </span>
            <button
              onClick={copyMetricsJson}
              className="rounded-md border border-border bg-surface-2 px-2.5 py-1 text-[0.62rem] font-medium text-text-muted transition-colors hover:border-border-2 hover:text-text-secondary"
              title="Copy all run data as JSON"
            >
              {copiedJson ? "Copied!" : "Copy as JSON"}
            </button>
            <span className="ml-auto text-xs text-text-muted">
              {stats && stats.totalElapsed > 0 && <>{fmtDuration(stats.totalElapsed)} elapsed</>}
              {isActive && stats && stats.eta > 0 && <> &middot; ~{fmtDuration(stats.eta)} remaining</>}
              {!isActive && <> &middot; Updated {timeAgo(run.updated_at)}</>}
            </span>
          </div>
          <div className="mt-1.5 text-[0.7rem] text-text-muted">
            {run.n_layer}L / {run.n_embd}D / {run.n_head}H &middot; {run.backend} &middot; {run.tokenizer} &middot; {run.optimizer}
            &middot; Created {fmtDate(run.created_at)}
          </div>
        </div>

        {/* Progress */}
        <div className="px-5 py-4">
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="text-xs text-text-secondary">
              Step <span className="font-mono font-bold text-white">{fmtNum(run.latest_step)}</span>
              <span className="text-text-muted"> / {fmtNum(run.total_iters)}</span>
            </span>
            <span className="font-mono text-sm font-bold text-white">{progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-2">
            <div
              className={`h-full rounded-full transition-all duration-700 ${ss.bar}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
        <Stat label="Loss" value={last ? fmtLoss(last.loss) : "-"} color="text-yellow" tip={tips.loss} />
        <Stat
          label="Best Loss"
          value={stats && isFinite(stats.minLoss) ? fmtLoss(stats.minLoss) : "-"}
          sub={stats && isFinite(stats.lossDropPct) ? `${stats.lossDropPct > 0 ? "-" : ""}${Math.abs(stats.lossDropPct).toFixed(1)}% from start` : undefined}
          tip={tips.lastLoss}
        />
        <Stat
          label="Val Loss"
          value={stats?.lastVal != null ? fmtLoss(stats.lastVal) : "-"}
          sub={stats?.bestVal != null ? `best: ${fmtLoss(stats.bestVal)}` : undefined}
          color={stats?.lastVal != null ? "text-blue" : undefined}
          tip={tips.valLoss}
        />
        <Stat label="Learning Rate" value={last?.lr != null ? last.lr.toExponential(2) : "-"} tip={tips.lr} />
        <Stat label="Throughput" value={stats ? fmtNum(stats.avgTps) : "-"} sub="tok/s (avg)" color="text-green" tip={tips.throughput} />
        <Stat label="Speed" value={stats ? fmtNum(stats.avgMs) : "-"} sub="ms/iter (avg)" tip={tips.msPerIter} />
        <Stat
          label="Grad Norm"
          value={last?.grad_norm != null ? last.grad_norm.toFixed(3) : "-"}
          sub={stats && isFinite(stats.avgGradNorm) ? `avg: ${stats.avgGradNorm.toFixed(3)}` : undefined}
          tip={tips.gradNorm}
        />
        <Stat label="Tokens" value={stats ? fmtParams(Math.round(stats.totalTokens)) : "-"} sub="processed" />
      </div>

      {/* Timing stats */}
      {stats?.avgFwd != null && (
        <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Forward" value={`${stats.avgFwd.toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${(stats.avgFwd / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-cyan-400" />
          <Stat label="Backward" value={`${(stats.avgBwd ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgBwd ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-orange-400" />
          <Stat label="GPU Sync" value={`${(stats.avgFlush ?? 0).toFixed(0)}ms`} sub={stats.avgMs > 0 ? `${((stats.avgFlush ?? 0) / stats.avgMs * 100).toFixed(0)}% of step` : undefined} color="text-rose-400" />
          <Stat label="GPU Ops" value={stats.avgGpuOps != null ? fmtNum(stats.avgGpuOps) : "-"} sub="per step" />
          <Stat
            label="MFU"
            value={stats.mfu != null ? `${stats.mfu.toFixed(1)}%` : "-"}
            sub="model FLOPS util"
            color={stats.mfu != null && stats.mfu > 50 ? "text-green" : stats.mfu != null && stats.mfu > 10 ? "text-yellow" : "text-red"}
          />
          <Stat
            label="Bwd/Fwd"
            value={stats.avgFwd != null && stats.avgBwd != null ? `${(stats.avgBwd / stats.avgFwd).toFixed(1)}x` : "-"}
            sub="ratio"
          />
        </div>
      )}

      {/* Chart + config sidebar */}
      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_280px]">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="mb-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Loss Curve <Tip text={tips.lossChart} />
          </div>
          <InteractiveLossChart metrics={metrics} checkpoints={checkpoints} pinnedStep={pinnedStep} onPinStep={handlePinStep} activationSwitches={activationSwitches} />
        </div>

        <div className="space-y-3">
          {/* Architecture */}
          <div className="rounded-lg border border-border/60 bg-surface p-3">
            <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Architecture</div>
            <DetailRow label="Layers" value={run.n_layer} tip={tips.nLayer} />
            <DetailRow label="Embedding" value={run.n_embd} tip={tips.nEmbd} />
            <DetailRow label="Heads" value={run.n_head} tip={tips.nHead} />
            <DetailRow label="Vocab" value={fmtNum(run.vocab_size)} tip={tips.vocabSize} />
            <DetailRow label="Context" value={run.block_size} tip={tips.blockSize} />
            <DetailRow label="Dropout" value={run.dropout} tip={tips.dropout} />
            <DetailRow label="Parameters" value={fmtParams(run.estimated_params)} tip={tips.params} />
          </div>

          {/* Training config */}
          <div className="rounded-lg border border-border/60 bg-surface p-3">
            <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Training Config</div>
            <DetailRow label="Total iters" value={fmtNum(run.total_iters)} tip={tips.totalIters} />
            <DetailRow label="Batch size" value={run.batch_size} tip={tips.batchSize} />
            <DetailRow label="Max LR" value={run.lr} tip={tips.lr} />
            <DetailRow label="Optimizer" value={run.optimizer} tip={tips.optimizer} />
            <DetailRow label="Backend" value={run.backend} tip={tips.backend} />
            <DetailRow label="Tokenizer" value={run.tokenizer} tip={tips.tokenizer} />
            <DetailRow label="Seed" value={run.seed} tip={tips.seed} />
            {trainConfig?.weightDecay != null && <DetailRow label="Weight decay" value={trainConfig.weightDecay} tip={tips.weightDecay} />}
            {trainConfig?.gradClip != null && <DetailRow label="Grad clip" value={trainConfig.gradClip} tip={tips.gradClip} />}
            {trainConfig?.evalInterval != null && <DetailRow label="Eval interval" value={trainConfig.evalInterval} tip={tips.evalInterval} />}
          </div>
        </div>
      </div>

      {/* Mini charts */}
      <div className="mb-6 space-y-4">
        <ChartPanel title="GPU & VRAM" helpText="GPU utilization percentage and video RAM usage over training time. High GPU utilization (>80%) means the hardware is being used efficiently. VRAM tracks memory consumption to detect leaks or pressure.">
          <MiniChart
            metrics={metrics}
            title=""
            noDataMsg="No GPU data"
            formatLeft={(v) => (v / 1024).toFixed(1) + "G"}
            formatRight={(v) => v.toFixed(0) + "%"}
            buildSeries={buildGpuSeries}
            pinnedStep={pinnedStep}
            onPinStep={handlePinStep}
          />
        </ChartPanel>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ChartPanel title="Learning Rate" helpText="The learning rate schedule over training. Typically starts low (warmup), peaks, then decays via cosine annealing. The shape of this curve directly affects convergence speed and final model quality.">
            <MiniChart
              metrics={metrics}
              title=""
              formatLeft={(v) => v.toExponential(1)}
              buildSeries={buildLrSeries}
              pinnedStep={pinnedStep}
              onPinStep={handlePinStep}
            />
          </ChartPanel>
          <ChartPanel title="Grad Norm" helpText="The L2 norm of all gradients at each step (log scale). Stable training shows consistent gradient norms. Sudden spikes indicate training instability, exploding gradients, or data anomalies. Values that trend to zero indicate vanishing gradients.">
            <MiniChart
              metrics={metrics}
              title=""
              logScale
              formatLeft={(v) => v.toExponential(0)}
              buildSeries={buildGradNormSeries}
              pinnedStep={pinnedStep}
              onPinStep={handlePinStep}
            />
          </ChartPanel>
        </div>
        <ChartPanel title="Step Time Breakdown" helpText="How time is spent within each training step. Forward pass computes the loss, backward pass computes gradients, optimizer updates weights, GPU sync flushes compute queues, and data loading prepares the next batch. Bottlenecks appear as dominant phases.">
          <StepTimeChart metrics={metrics} pinnedStep={pinnedStep} onPinStep={handlePinStep} />
        </ChartPanel>
      </div>

      {/* Symbio section */}
      <SymbioSection metrics={metrics as any} run={run as any} pinnedStep={pinnedStep} onPinStep={handlePinStep} />

      {/* Checkpoints */}
      <div className="mb-6 rounded-lg border border-border bg-surface">
        <div className="border-b border-border px-4 py-3">
          <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
            Checkpoints ({checkpoints.length}) <Tip text={tips.checkpoint} />
          </span>
        </div>
        {checkpoints.length === 0 ? (
          <div className="px-4 py-6 text-center text-xs text-text-muted">No checkpoints saved</div>
        ) : (
          <>
            <div className="grid grid-cols-[80px_1fr_90px_100px_52px] gap-4 border-b border-border/50 px-4 py-2 text-[0.62rem] font-semibold uppercase tracking-wider text-text-muted">
              <span>Step</span><span>Filename</span><span>Size</span><span>Created</span><span></span>
            </div>
            {checkpoints.map((c) => (
              <div key={c.step} className="grid grid-cols-[80px_1fr_90px_100px_52px] gap-4 border-b border-border/30 px-4 py-2 text-xs last:border-0">
                <span className="font-mono font-semibold text-white">{fmtNum(c.step)}</span>
                <span className="truncate font-mono text-text-secondary">{c.filename}</span>
                <span className="text-text-muted">{fmtBytes(c.file_size)}</span>
                <span className="text-text-muted">{c.created_at ? timeAgo(c.created_at) : "-"}</span>
                <a
                  href={`/api/runs/${encodeURIComponent(run.id)}/download/${encodeURIComponent(c.filename)}`}
                  className="text-text-muted transition-colors hover:text-accent"
                  title="Download checkpoint"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" />
                    <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                  </svg>
                </a>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Samples */}
      {samples.length > 0 && (
        <div className="mb-6 rounded-lg border border-border bg-surface">
          <div className="border-b border-border px-4 py-3">
            <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
              Sample Generations ({samples.length})
            </span>
          </div>
          {/* Summary table */}
          <div className="border-b border-border/50">
            <div className="grid grid-cols-[50px_80px_1fr_100px] gap-4 border-b border-border/30 px-4 py-2 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">
              <span>#</span><span>Checkpoint</span><span>Prompt (preview)</span><span>Generated</span>
            </div>
            {samples.map((s, i) => {
              // Find nearest checkpoint that was created before this sample
              const sampleTime = s.created_at ? new Date(s.created_at + "Z").getTime() : 0;
              const nearestCp = [...checkpoints].reverse().find(c => {
                const cpTime = c.created_at ? new Date(c.created_at + "Z").getTime() : 0;
                return cpTime <= sampleTime + 60000; // within 1 min
              });
              return (
                <details key={s.idx} className="border-b border-border/20 last:border-0 group">
                  <summary className="grid grid-cols-[50px_80px_1fr_100px] gap-4 px-4 py-2 text-xs cursor-pointer hover:bg-surface-2/30">
                    <span className="font-mono font-semibold text-text-secondary">{s.idx + 1}</span>
                    <span className="font-mono text-green">{nearestCp ? `step ${fmtNum(nearestCp.step)}` : "-"}</span>
                    <span className="truncate text-text-muted">{s.prompt.slice(0, 80)}{s.prompt.length > 80 ? "..." : ""}</span>
                    <span className="text-text-muted">{s.created_at ? timeAgo(s.created_at) : "-"}</span>
                  </summary>
                  <div className="px-4 pb-3 pt-1">
                    <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Prompt</div>
                    <div className="mb-3 rounded border border-border/50 bg-[#0d0d0d] px-3 py-2 font-mono text-xs leading-relaxed text-text-secondary">
                      {s.prompt}
                    </div>
                    <div className="mb-1 text-[0.6rem] font-semibold uppercase tracking-wider text-text-muted">Output</div>
                    <div className="whitespace-pre-wrap rounded border border-border/50 bg-[#0d0d0d] px-3 py-2 font-mono text-xs leading-relaxed text-text-primary">
                      {s.output}
                    </div>
                  </div>
                </details>
              );
            })}
          </div>
        </div>
      )}

      {/* Raw config */}
      <div className="mb-6 space-y-2">
        <button
          onClick={() => setShowModelConfig(!showModelConfig)}
          className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-left text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted transition-colors hover:border-border-2"
        >
          <span className="text-[0.7rem]">{showModelConfig ? "\u25BC" : "\u25B6"}</span>
          Model Config JSON <Tip text={tips.rawConfig} />
        </button>
        {showModelConfig && modelConfig && (
          <pre className="overflow-x-auto rounded-lg border border-border bg-[#0d0d0d] p-4 font-mono text-xs leading-relaxed text-text-secondary">
            {JSON.stringify(modelConfig, null, 2)}
          </pre>
        )}

        <button
          onClick={() => setShowTrainConfig(!showTrainConfig)}
          className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-left text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted transition-colors hover:border-border-2"
        >
          <span className="text-[0.7rem]">{showTrainConfig ? "\u25BC" : "\u25B6"}</span>
          Train Config JSON <Tip text={tips.rawConfig} />
        </button>
        {showTrainConfig && trainConfig && (
          <pre className="overflow-x-auto rounded-lg border border-border bg-[#0d0d0d] p-4 font-mono text-xs leading-relaxed text-text-secondary">
            {JSON.stringify(trainConfig, null, 2)}
          </pre>
        )}
      </div>
    </>
  );
}
