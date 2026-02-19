"use client";

import { useEffect, useRef, useState, useCallback } from "react";

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

interface LiveRun {
  id: string;
  domain: string;
  status: string;
  totalIters: number;
  modelConfig: {
    nLayer: number;
    nEmbd: number;
    nHead: number;
    vocabSize: number;
    blockSize: number;
  } | null;
  totalParams: number | null;
  metrics: StepMetric[];
}

// ── Canvas chart ───────────────────────────────────────────────

function drawLossChart(canvas: HTMLCanvasElement, metrics: StepMetric[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx || metrics.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 8, right: 10, bottom: 24, left: 46 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const losses = metrics.map((m) => m.loss);
  const minL = Math.min(...losses);
  const maxL = Math.max(...losses);
  const rangeL = maxL - minL || 1;
  const minStep = metrics[0].step;
  const maxStep = metrics[metrics.length - 1].step;
  const rangeS = maxStep - minStep || 1;

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;
  const sy = (loss: number) => pad.top + (1 - (loss - minL) / rangeL) * ch;

  // Grid
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    const val = maxL - (i / 4) * rangeL;
    ctx.fillStyle = "#555";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), pad.left - 6, y + 3);
  }

  // Step labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#555";
  ctx.font = "10px monospace";
  ctx.fillText(String(minStep), sx(minStep), h - pad.bottom + 14);
  ctx.fillText(String(maxStep), sx(maxStep), h - pad.bottom + 14);

  // Loss line
  ctx.beginPath();
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  for (let i = 0; i < metrics.length; i++) {
    const x = sx(metrics[i].step);
    const y = sy(metrics[i].loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Val loss dots
  const valPts = metrics.filter((m) => m.valLoss != null);
  if (valPts.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 1;
    for (let i = 0; i < valPts.length; i++) {
      const x = sx(valPts[i].step);
      const y = sy(valPts[i].valLoss!);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#60a5fa";
    for (const v of valPts) {
      ctx.beginPath();
      ctx.arc(sx(v.step), sy(v.valLoss!), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function LiveLossChart({ metrics }: { metrics: StepMetric[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current && metrics.length >= 2) {
      drawLossChart(canvasRef.current, metrics);
    }
  }, [metrics]);

  if (metrics.length < 2) {
    return (
      <div className="flex h-40 items-center justify-center text-xs text-text-muted">
        Waiting for data...
      </div>
    );
  }

  return <canvas ref={canvasRef} className="h-40 w-full" />;
}

// ── LiveRunCard ────────────────────────────────────────────────

function LiveRunCard({ run }: { run: LiveRun }) {
  const last = run.metrics[run.metrics.length - 1];
  const pct = last && run.totalIters > 0 ? Math.min(100, (last.step / run.totalIters) * 100) : 0;
  const mc = run.modelConfig;
  const arch = mc ? `${mc.nLayer}L ${mc.nEmbd}D ${mc.nHead}H` : "";

  const domainColors: Record<string, string> = {
    novels: "bg-blue-bg text-blue",
    chords: "bg-yellow-bg text-yellow",
    abc: "bg-green-bg text-green",
  };

  const isActive = run.status === "active";

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-white">{run.id}</span>
        <span className={`rounded px-2 py-0.5 text-[0.65rem] font-semibold uppercase ${domainColors[run.domain] ?? "bg-surface-2 text-text-secondary"}`}>
          {run.domain}
        </span>
        <span className={`rounded px-2 py-0.5 text-[0.65rem] font-semibold uppercase ${isActive ? "bg-green-bg text-green" : "bg-blue-bg text-blue"}`}>
          {isActive && <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green" />}
          {run.status}
        </span>
        {arch && <span className="text-xs text-text-muted">{arch}</span>}
      </div>

      {/* Stats grid */}
      <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <div className="rounded border border-border bg-surface-2 px-3 py-2">
          <div className="text-sm font-bold text-white">
            {last ? `${last.step}/${run.totalIters}` : "-"}
          </div>
          <div className="text-[0.6rem] uppercase text-text-muted">Step</div>
        </div>
        <div className="rounded border border-border bg-surface-2 px-3 py-2">
          <div className="text-sm font-bold text-white">
            {last ? last.loss.toFixed(4) : "-"}
          </div>
          <div className="text-[0.6rem] uppercase text-text-muted">Loss</div>
        </div>
        <div className="rounded border border-border bg-surface-2 px-3 py-2">
          <div className="text-sm font-bold text-white">
            {last ? last.lr.toExponential(2) : "-"}
          </div>
          <div className="text-[0.6rem] uppercase text-text-muted">LR</div>
        </div>
        <div className="rounded border border-border bg-surface-2 px-3 py-2">
          <div className="text-sm font-bold text-white">
            {last ? last.tokens_per_sec.toFixed(0) : "-"}
          </div>
          <div className="text-[0.6rem] uppercase text-text-muted">tok/s</div>
        </div>
        <div className="rounded border border-border bg-surface-2 px-3 py-2">
          <div className="text-sm font-bold text-white">
            {last ? last.ms_per_iter.toFixed(0) : "-"}
          </div>
          <div className="text-[0.6rem] uppercase text-text-muted">ms/iter</div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="mb-1 flex justify-between text-[0.7rem] text-text-muted">
          <span>Progress</span>
          <span>{pct.toFixed(1)}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-surface-2">
          <div
            className={`h-full rounded-full transition-all duration-500 ${isActive ? "bg-green" : "bg-blue"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Loss chart */}
      <LiveLossChart metrics={run.metrics} />
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────

export default function TrainingPage() {
  const [runs, setRuns] = useState<Map<string, LiveRun>>(new Map());
  const [connected, setConnected] = useState(false);
  const [completedIds, setCompletedIds] = useState<Set<string>>(new Set());
  const runsRef = useRef(runs);
  runsRef.current = runs;

  const updateRun = useCallback((id: string, updater: (run: LiveRun) => LiveRun) => {
    setRuns((prev) => {
      const next = new Map(prev);
      const existing = next.get(id);
      if (existing) {
        next.set(id, updater(existing));
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const es = new EventSource("/api/training/live");

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.addEventListener("snapshot", (e) => {
      const data = JSON.parse(e.data) as Array<Record<string, any>>;
      const next = new Map<string, LiveRun>();
      for (const r of data) {
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
          totalIters: r.total_iters ?? 0,
          modelConfig: r.model_config ? JSON.parse(r.model_config) : null,
          totalParams: r.estimated_params ?? null,
          metrics,
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
          totalParams: data.totalParams ?? null,
          metrics: [],
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
          className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
            connected
              ? "bg-green-bg text-green"
              : "bg-red-bg text-red"
          }`}
        >
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              connected ? "animate-pulse bg-green" : "bg-red"
            }`}
          />
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Active runs */}
      {activeRuns.length === 0 && recentlyCompleted.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-8 text-center">
          <div className="mb-2 text-text-secondary">No active training runs</div>
          <div className="text-xs text-text-muted">
            To stream metrics here, set environment variables on the training machine:
          </div>
          <pre className="mx-auto mt-3 inline-block rounded border border-border bg-surface-2 px-4 py-2 text-left font-mono text-xs text-text-primary">
{`ALPHA_REMOTE_URL=https://alpha.omegaai.dev
ALPHA_REMOTE_SECRET=<your-secret>`}
          </pre>
          <div className="mt-3 text-xs text-text-muted">
            Then start training with the CLI and metrics will appear here in real-time.
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {activeRuns.map((run) => (
            <LiveRunCard key={run.id} run={run} />
          ))}
        </div>
      )}

      {/* Recently completed */}
      {recentlyCompleted.length > 0 && (
        <div className="mt-8">
          <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Recently Completed
          </h2>
          <div className="space-y-4">
            {recentlyCompleted.map((run) => (
              <LiveRunCard key={run.id} run={run} />
            ))}
          </div>
        </div>
      )}
    </>
  );
}
