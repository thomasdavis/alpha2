"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  type ChartMetric,
  Stat, DetailRow, InteractiveLossChart, MiniChart, ChartPanel,
  buildLrSeries, buildGradNormSeries, buildThroughputSeries,
  buildPerplexitySeries, buildSmoothedLossSeries,
  buildTimingPhaseSeries,
  fmtParams, fmtLoss, fmtNum, fmtDuration, fmtDate, fmtBytes,
  Card, CardContent, CardHeader, CardTitle, Button, Badge, Progress,
} from "@/components/charts";

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
  ffn_activation?: string | null;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  run: RunData;
  metrics: MetricData[];
  checkpoints: CheckpointData[];
  samples: SampleData[];
}

// ── Helpers ──────────────────────────────────────────────────────

function estimateParams(r: RunData): number {
  if (r.estimated_params) return r.estimated_params;
  const V = r.vocab_size, D = r.n_embd, L = r.n_layer, C = r.block_size;
  const ffnDim = D * 4;
  return V * D + C * D + L * (4 * D * D + 2 * D + ffnDim * D + D * ffnDim + 2 * D) + D + D + V * D;
}

function perplexity(loss: number | null): string {
  if (loss == null) return "—";
  return Math.exp(loss).toFixed(1);
}

function progressPct(step: number, total: number): number {
  return total > 0 ? Math.min(100, (step / total) * 100) : 0;
}

function tokensProcessed(step: number, batch: number, block: number): number {
  return step * batch * block;
}

type Section = "overview" | "training" | "architecture" | "samples" | "checkpoints" | "chat";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "training", label: "Training Metrics" },
  { id: "architecture", label: "Architecture" },
  { id: "samples", label: "Samples" },
  { id: "checkpoints", label: "Checkpoints" },
  { id: "chat", label: "Chat" },
];

// ── Stage classification ─────────────────────────────────────────

interface TrainingStage {
  name: string;
  color: string;
  description: string;
  startStep: number;
  endStep: number;
  startLoss: number | null;
  endLoss: number | null;
  technique: string;
}

function classifyStages(metrics: MetricData[], totalIters: number): TrainingStage[] {
  if (metrics.length === 0) return [];
  const stages: TrainingStage[] = [];

  // Warmup phase
  const warmupEnd = metrics.findIndex(m => m.step > totalIters * 0.01);
  const warmupMetrics = metrics.slice(0, Math.max(warmupEnd, 1));

  stages.push({
    name: "Warmup",
    color: "bg-cyan-500",
    description: "Learning rate warmup — model weights adjusting to data distribution",
    startStep: metrics[0]?.step ?? 0,
    endStep: warmupMetrics[warmupMetrics.length - 1]?.step ?? 0,
    startLoss: metrics[0]?.loss ?? null,
    endLoss: warmupMetrics[warmupMetrics.length - 1]?.loss ?? null,
    technique: "Linear LR warmup, gradient clipping",
  });

  // Rapid descent phase (loss dropping fastest)
  const rapidEnd = metrics.findIndex(m => m.step > totalIters * 0.3);
  if (rapidEnd > 0) {
    stages.push({
      name: "Rapid Descent",
      color: "bg-emerald-500",
      description: "Steepest loss reduction — model learning primary patterns",
      startStep: warmupMetrics[warmupMetrics.length - 1]?.step ?? 0,
      endStep: metrics[Math.min(rapidEnd, metrics.length - 1)]?.step ?? 0,
      startLoss: warmupMetrics[warmupMetrics.length - 1]?.loss ?? null,
      endLoss: metrics[Math.min(rapidEnd, metrics.length - 1)]?.loss ?? null,
      technique: "Cosine LR schedule, AdamW optimization",
    });
  }

  // Refinement phase
  const refineEnd = metrics.findIndex(m => m.step > totalIters * 0.7);
  if (refineEnd > 0 && rapidEnd > 0) {
    stages.push({
      name: "Refinement",
      color: "bg-amber-500",
      description: "Diminishing returns — model fine-tuning subtler patterns",
      startStep: metrics[rapidEnd]?.step ?? 0,
      endStep: metrics[Math.min(refineEnd, metrics.length - 1)]?.step ?? 0,
      startLoss: metrics[rapidEnd]?.loss ?? null,
      endLoss: metrics[Math.min(refineEnd, metrics.length - 1)]?.loss ?? null,
      technique: "Lower LR, gradient accumulation",
    });
  }

  // Convergence phase
  if (refineEnd > 0 && refineEnd < metrics.length) {
    stages.push({
      name: "Convergence",
      color: "bg-violet-500",
      description: "Approaching minimum — model capacity saturation",
      startStep: metrics[refineEnd]?.step ?? 0,
      endStep: metrics[metrics.length - 1]?.step ?? 0,
      startLoss: metrics[refineEnd]?.loss ?? null,
      endLoss: metrics[metrics.length - 1]?.loss ?? null,
      technique: "Minimum LR, weight decay regularization",
    });
  }

  return stages;
}

// ── Component ────────────────────────────────────────────────────

export function ModelReport({ run, metrics, checkpoints, samples }: Props) {
  const [activeSection, setActiveSection] = useState<Section>("overview");

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatGenerating, setChatGenerating] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Parse configs
  const modelConfig = useMemo(() => {
    try { return JSON.parse(run.model_config); } catch { return {}; }
  }, [run.model_config]);
  const trainConfig = useMemo(() => {
    try { return JSON.parse(run.train_config); } catch { return {}; }
  }, [run.train_config]);

  // Derived data
  const params = estimateParams(run);
  const latestMetric = metrics[metrics.length - 1];
  const firstMetric = metrics[0];
  const totalTokens = tokensProcessed(run.latest_step, run.batch_size, run.block_size);
  const stages = useMemo(() => classifyStages(metrics, run.total_iters), [metrics, run.total_iters]);

  // Best val loss
  const bestVal = useMemo(() => {
    let best: MetricData | null = null;
    for (const m of metrics) {
      if (m.val_loss != null && (best == null || m.val_loss < best.val_loss!)) best = m;
    }
    return best;
  }, [metrics]);

  // Loss reduction
  const lossReduction = firstMetric && latestMetric
    ? ((firstMetric.loss - latestMetric.loss) / firstMetric.loss * 100).toFixed(1)
    : null;

  // Avg throughput
  const avgThroughput = useMemo(() => {
    const tps = metrics.filter(m => m.tokens_per_sec != null).map(m => m.tokens_per_sec!);
    return tps.length > 0 ? tps.reduce((a, b) => a + b, 0) / tps.length : 0;
  }, [metrics]);

  // Training time estimate
  const totalTimeMs = useMemo(() => {
    return metrics.reduce((sum, m) => sum + (m.elapsed_ms ?? 0), 0);
  }, [metrics]);

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Section scroll observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id as Section);
          }
        }
      },
      { threshold: 0.3 },
    );
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  // Chat handler
  const sendChat = useCallback(async () => {
    const text = chatInput.trim();
    if (!text || chatGenerating) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput("");
    setChatGenerating(true);

    const assistantMsg: ChatMessage = { role: "assistant", content: "" };
    setChatMessages(prev => [...prev, assistantMsg]);

    try {
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: run.id,
          messages: [...chatMessages, userMsg].slice(-8).map(m => ({
            role: m.role,
            content: m.content,
          })),
          max_tokens: 200,
          temperature: 0.8,
          top_k: 40,
        }),
        signal: ctrl.signal,
      });

      if (!res.ok || !res.body) throw new Error("Chat failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value, { stream: true });
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: accumulated };
          return updated;
        });
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: "(Error generating response)" };
          return updated;
        });
      }
    } finally {
      setChatGenerating(false);
      abortRef.current = null;
    }
  }, [chatInput, chatGenerating, chatMessages, run.id]);

  return (
    <div className="flex gap-6">
      {/* Sticky sidebar nav */}
      <nav className="hidden xl:block w-48 shrink-0">
        <div className="sticky top-6 space-y-1">
          <div className="text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted mb-3">
            Report
          </div>
          {SECTIONS.map(s => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={`block rounded-md px-3 py-1.5 text-[0.8rem] transition-colors hover:no-underline ${
                activeSection === s.id
                  ? "bg-surface-2 text-text-primary font-medium"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {s.label}
            </a>
          ))}
        </div>
      </nav>

      {/* Main content */}
      <div className="flex-1 min-w-0 space-y-8">
        {/* Header */}
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Badge variant={run.status === "completed" ? "success" : run.status === "running" ? "blue" : "default"}>
              {run.status}
            </Badge>
            <span className="text-text-muted text-xs">{run.config_hash}</span>
          </div>
          <h1 className="text-2xl font-bold text-text-primary">{run.run_id || run.id}</h1>
          <p className="text-text-secondary mt-1">
            {fmtParams(params)} parameter {run.domain} model — {run.tokenizer} tokenizer, {run.n_layer}L/{run.n_embd}D/{run.n_head}H
          </p>
          <div className="flex items-center gap-4 mt-3">
            <Link href={`/runs/${run.id}`} className="text-xs text-accent hover:underline">
              Full run details
            </Link>
            <Link href={`/chat?model=${run.id}`} className="text-xs text-accent hover:underline">
              Open in Chat
            </Link>
          </div>
        </div>

        {/* ── Overview ──────────────────────────────────────────── */}
        <section id="overview">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Overview</h2>

          {/* Key metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <Stat label="Parameters" value={fmtParams(params)} />
            <Stat label="Final Loss" value={fmtLoss(latestMetric?.loss ?? run.last_loss)} />
            <Stat label="Best Val Loss" value={fmtLoss(bestVal?.val_loss ?? run.best_val_loss)} />
            <Stat label="Perplexity" value={perplexity(latestMetric?.loss ?? run.last_loss)} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <Stat label="Tokens Processed" value={fmtNum(totalTokens)} />
            <Stat label="Tokens/Param" value={(totalTokens / Math.max(params, 1)).toFixed(1)} />
            <Stat label="Avg Throughput" value={`${fmtNum(avgThroughput)} tok/s`} />
            <Stat label="Training Time" value={fmtDuration(totalTimeMs)} />
          </div>

          {/* Progress bar */}
          <Card className="mb-6">
            <CardContent className="py-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-secondary">Training Progress</span>
                <span className="text-sm font-mono text-text-primary">
                  {fmtNum(run.latest_step)} / {fmtNum(run.total_iters)} steps ({progressPct(run.latest_step, run.total_iters).toFixed(1)}%)
                </span>
              </div>
              <Progress value={progressPct(run.latest_step, run.total_iters)} />
              {lossReduction && (
                <div className="mt-2 text-xs text-text-muted">
                  Loss reduced by {lossReduction}% from initial {fmtLoss(firstMetric?.loss ?? null)}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Dataset & Training summary */}
          <Card className="mb-6">
            <CardHeader><CardTitle>Dataset & Training</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
                <DetailRow label="Domain" value={run.domain} />
                <DetailRow label="Tokenizer" value={run.tokenizer} />
                <DetailRow label="Total Iterations" value={fmtNum(run.total_iters)} />
                <DetailRow label="Batch Size" value={String(run.batch_size)} />
                <DetailRow label="Context Length" value={`${run.block_size} tokens`} />
                <DetailRow label="Tokens per Batch" value={fmtNum(run.batch_size * run.block_size)} />
                <DetailRow label="Dataset Passes" value={run.total_iters > 0 ? `~${((totalTokens / Math.max(1, run.vocab_size * 100)) || 0).toFixed(0)}` : "—"} />
                <DetailRow label="Effective Tokens" value={fmtNum(totalTokens)} />
              </div>
            </CardContent>
          </Card>

          {/* Training pipeline stages */}
          {stages.length > 0 && (
            <Card className="mb-6">
              <CardHeader><CardTitle>Training Pipeline</CardTitle></CardHeader>
              <CardContent>
                <div className="relative">
                  {stages.map((stage, i) => (
                    <div key={stage.name} className="flex gap-4 pb-6 last:pb-0">
                      {/* Timeline connector */}
                      <div className="flex flex-col items-center">
                        <div className={`w-3 h-3 rounded-full ${stage.color} shrink-0`} />
                        {i < stages.length - 1 && (
                          <div className="w-px flex-1 bg-border mt-1" />
                        )}
                      </div>
                      {/* Content */}
                      <div className="flex-1 -mt-0.5">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-semibold text-text-primary">{stage.name}</span>
                          <span className="text-xs text-text-muted">
                            steps {fmtNum(stage.startStep)}–{fmtNum(stage.endStep)}
                          </span>
                        </div>
                        <p className="text-xs text-text-secondary mb-1">{stage.description}</p>
                        <div className="flex gap-4 text-xs text-text-muted">
                          {stage.startLoss != null && stage.endLoss != null && (
                            <span>Loss: {stage.startLoss.toFixed(3)} → {stage.endLoss.toFixed(3)}</span>
                          )}
                          <span>{stage.technique}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </section>

        {/* ── Training Metrics ────────────────────────────────── */}
        <section id="training">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Training Metrics</h2>

          {metrics.length > 0 ? (
            <div className="space-y-4">
              {/* Loss chart */}
              <ChartPanel title="Loss Curve">
                <InteractiveLossChart
                  metrics={metrics}
                  checkpoints={checkpoints.map(c => ({ step: c.step, filename: c.filename }))}
                />
              </ChartPanel>

              {/* Mini charts grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ChartPanel title="Smoothed Loss">
                  <MiniChart metrics={metrics} title="" buildSeries={buildSmoothedLossSeries} />
                </ChartPanel>
                <ChartPanel title="Perplexity">
                  <MiniChart metrics={metrics} title="" buildSeries={buildPerplexitySeries} />
                </ChartPanel>
                <ChartPanel title="Learning Rate">
                  <MiniChart metrics={metrics} title="" buildSeries={buildLrSeries} />
                </ChartPanel>
                <ChartPanel title="Gradient Norm">
                  <MiniChart metrics={metrics} title="" buildSeries={buildGradNormSeries} />
                </ChartPanel>
                <ChartPanel title="Throughput (tok/s)">
                  <MiniChart metrics={metrics} title="" formatLeft={(v: number) => (v / 1000).toFixed(1) + "k"} buildSeries={buildThroughputSeries} />
                </ChartPanel>
                <ChartPanel title="Timing Breakdown">
                  <MiniChart metrics={metrics} title="" formatLeft={(v: number) => v.toFixed(0) + "ms"} buildSeries={buildTimingPhaseSeries} />
                </ChartPanel>
              </div>
            </div>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-text-muted">
                No metrics recorded yet.
              </CardContent>
            </Card>
          )}
        </section>

        {/* ── Architecture ────────────────────────────────────── */}
        <section id="architecture">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Model Architecture</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader><CardTitle>Model Configuration</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <DetailRow label="Architecture" value="GPT (decoder-only transformer)" />
                <DetailRow label="Parameters" value={fmtParams(params)} />
                <DetailRow label="Layers" value={String(run.n_layer)} />
                <DetailRow label="Embedding Dim" value={String(run.n_embd)} />
                <DetailRow label="Attention Heads" value={String(run.n_head)} />
                <DetailRow label="Head Dim" value={String(Math.floor(run.n_embd / run.n_head))} />
                <DetailRow label="FFN Dim" value={String(modelConfig.ffnDim || run.n_embd * 4)} />
                <DetailRow label="FFN Activation" value={run.ffn_activation || modelConfig.ffnActivation || "gelu"} />
                <DetailRow label="Vocab Size" value={fmtNum(run.vocab_size)} />
                <DetailRow label="Context Length" value={`${run.block_size} tokens`} />
                <DetailRow label="Dropout" value={String(run.dropout)} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Training Configuration</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <DetailRow label="Optimizer" value={run.optimizer || "adamw"} />
                <DetailRow label="Learning Rate" value={String(run.lr)} />
                <DetailRow label="LR Min" value={String(trainConfig.lrMin ?? "—")} />
                <DetailRow label="LR Schedule" value="Cosine decay" />
                <DetailRow label="Warmup Steps" value={fmtNum(trainConfig.warmupIters ?? 0)} />
                <DetailRow label="Batch Size" value={String(run.batch_size)} />
                <DetailRow label="Grad Accum Steps" value={String(trainConfig.gradAccumSteps ?? 1)} />
                <DetailRow label="Effective Batch" value={String(run.batch_size * (trainConfig.gradAccumSteps ?? 1))} />
                <DetailRow label="Grad Clip" value={String(trainConfig.gradClip ?? "—")} />
                <DetailRow label="Weight Decay" value={String(trainConfig.weightDecay ?? "—")} />
                <DetailRow label="Backend" value={run.backend} />
                <DetailRow label="Tokenizer" value={run.tokenizer} />
                <DetailRow label="Seed" value={String(run.seed)} />
              </CardContent>
            </Card>
          </div>

          {/* Layer diagram */}
          <Card className="mt-4">
            <CardHeader><CardTitle>Layer Structure</CardTitle></CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 overflow-x-auto py-2">
                <LayerBlock label="Token Embed" sub={`${fmtNum(run.vocab_size)}×${run.n_embd}`} color="bg-blue-500/20 border-blue-500/40" />
                <Arrow />
                <LayerBlock label="Pos Embed" sub={`${run.block_size}×${run.n_embd}`} color="bg-blue-500/20 border-blue-500/40" />
                <Arrow />
                {Array.from({ length: Math.min(run.n_layer, 6) }).map((_, i) => (
                  <span key={i} className="flex items-center gap-2">
                    <LayerBlock
                      label={`Block ${i}`}
                      sub={`Attn+FFN`}
                      color="bg-emerald-500/20 border-emerald-500/40"
                    />
                    <Arrow />
                  </span>
                ))}
                {run.n_layer > 6 && (
                  <>
                    <span className="text-text-muted text-xs px-2">...{run.n_layer - 6} more</span>
                    <Arrow />
                  </>
                )}
                <LayerBlock label="LayerNorm" sub={`${run.n_embd}`} color="bg-amber-500/20 border-amber-500/40" />
                <Arrow />
                <LayerBlock label="LM Head" sub={`${run.n_embd}×${fmtNum(run.vocab_size)}`} color="bg-violet-500/20 border-violet-500/40" />
              </div>
            </CardContent>
          </Card>
        </section>

        {/* ── Samples ─────────────────────────────────────────── */}
        <section id="samples">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Generated Samples</h2>

          {samples.length > 0 ? (
            <div className="space-y-3">
              {samples.slice(-10).map((s, i) => (
                <Card key={i}>
                  <CardContent className="py-3">
                    <div className="text-xs text-text-muted mb-1">
                      Step {fmtNum(s.idx)} — {fmtDate(s.created_at)}
                    </div>
                    <div className="text-xs text-text-secondary mb-2">
                      Prompt: <span className="font-mono text-accent">{s.prompt}</span>
                    </div>
                    <div className="text-sm text-text-primary font-mono whitespace-pre-wrap bg-surface-2 rounded-md p-3 max-h-40 overflow-y-auto">
                      {s.output}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-text-muted">
                No samples generated yet. Samples appear at configured intervals during training.
              </CardContent>
            </Card>
          )}
        </section>

        {/* ── Checkpoints ─────────────────────────────────────── */}
        <section id="checkpoints">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Checkpoints</h2>

          {checkpoints.length > 0 ? (
            <Card>
              <CardContent className="py-0">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs text-text-muted">
                      <th className="py-3 pr-4">Step</th>
                      <th className="py-3 pr-4">File</th>
                      <th className="py-3 pr-4">Size</th>
                      <th className="py-3">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {checkpoints.map((c) => (
                      <tr key={c.step} className="border-b border-border/50 last:border-0">
                        <td className="py-2.5 pr-4 font-mono text-text-primary">{fmtNum(c.step)}</td>
                        <td className="py-2.5 pr-4 text-text-secondary font-mono text-xs">{c.filename}</td>
                        <td className="py-2.5 pr-4 text-text-secondary">{fmtBytes(c.file_size)}</td>
                        <td className="py-2.5 text-text-muted text-xs">{fmtDate(c.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-text-muted">
                No checkpoints saved yet.
              </CardContent>
            </Card>
          )}
        </section>

        {/* ── Chat ─────────────────────────────────────────────── */}
        <section id="chat">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Chat with Model</h2>

          <Card>
            <CardContent className="p-0">
              {/* Messages */}
              <div className="h-80 overflow-y-auto p-4 space-y-3">
                {chatMessages.length === 0 && (
                  <div className="flex items-center justify-center h-full text-text-muted text-sm">
                    Send a message to chat with this model
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                        msg.role === "user"
                          ? "bg-accent text-white"
                          : "bg-surface-2 text-text-primary"
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content || (chatGenerating && i === chatMessages.length - 1 ? "..." : "")}</div>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Input */}
              <div className="border-t border-border p-3 flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendChat()}
                  placeholder="Type a message..."
                  className="flex-1 rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-accent"
                  disabled={chatGenerating}
                />
                <Button
                  onClick={chatGenerating ? () => abortRef.current?.abort() : sendChat}
                  variant={chatGenerating ? "outline" : "primary"}
                  size="sm"
                >
                  {chatGenerating ? "Stop" : "Send"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Footer */}
        <div className="border-t border-border pt-4 pb-8 text-xs text-text-muted">
          <div className="flex items-center justify-between">
            <span>Generated {fmtDate(new Date().toISOString())} — Alpha Training System</span>
            <span>Config hash: {run.config_hash}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Small helper components ──────────────────────────────────────

function LayerBlock({ label, sub, color }: { label: string; sub: string; color: string }) {
  return (
    <div className={`shrink-0 rounded-md border px-3 py-2 text-center ${color}`}>
      <div className="text-xs font-medium text-text-primary">{label}</div>
      <div className="text-[0.6rem] text-text-muted">{sub}</div>
    </div>
  );
}

function Arrow() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="shrink-0 text-text-muted">
      <path d="M3 8h10M10 5l3 3-3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
