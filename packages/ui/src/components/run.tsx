import * as React from "react";
import { Card } from "./card.js";
import { Progress } from "./progress.js";
import { Badge, StatusBadge, DomainBadge } from "./badge.js";
import { Tip } from "./tooltip.js";
import { fmtParams, fmtLoss, timeAgo, pct, fmtNum } from "../utils.js";
import { RunSummaryData, LiveRunData } from "../types.js";

// Mock tips for UI package, could also be passed down or centralized
const tips = {
  params: "Estimated total parameters",
  architecture: "Layers / Embedding Dim / Heads",
  loss: "Lowest validation loss (or training if no val)",
  throughput: "Tokens processed per second",
  msPerIter: "Average milliseconds per training step",
};

export function RunProgressBar({ step, total, status, className }: { step: number | null; total: number | null; status: string; className?: string }) {
  const p = pct(step ?? 0, total ?? 0);
  const variantMap: Record<string, any> = {
    active: "success",
    completed: "blue",
    stale: "warning",
    failed: "danger",
  };
  return (
    <div className={`mt-3 ${className || ""}`}>
      <Progress
        value={step ?? 0}
        max={total ?? 0}
        variant={variantMap[status] || "default"}
        className="h-1.5"
      />
      <div className="mt-1.5 flex justify-between text-[0.68rem] text-text-muted">
        <span>
          Step {(step ?? 0).toLocaleString()} / {(total ?? 0).toLocaleString()}
        </span>
        <span className="font-mono">{p}%</span>
      </div>
    </div>
  );
}

export function RunCard({ 
  run, 
  renderSparkline 
}: { 
  run: RunSummaryData;
  renderSparkline?: (runId: string, status: string) => React.ReactNode;
}) {
  const bestLoss = run.best_val_loss ?? run.last_loss;
  return (
    <a
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="block hover:no-underline transition-transform duration-200 active:scale-[0.99]"
    >
      <Card className="hover:border-border-2 transition-colors overflow-hidden group">
        <div className="flex items-start gap-4 p-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[0.95rem] font-bold text-text-primary group-hover:text-accent transition-colors">
                {run.id}
              </span>
              <StatusBadge status={run.status} />
              <DomainBadge domain={run.domain} />
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-text-secondary">
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">PARAM</span>
                {fmtParams(run.estimated_params)} <Tip text={tips.params} />
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">ARCH</span>
                {run.n_layer}L {run.n_embd}D {run.n_head}H <Tip text={tips.architecture} />
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">LOSS</span>
                <span className="font-mono text-text-primary">{fmtLoss(bestLoss)}</span> <Tip text={tips.loss} />
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1.5 text-text-muted">
                {timeAgo(run.updated_at)}
              </span>
            </div>
            <RunProgressBar
              step={run.latest_step}
              total={run.total_iters}
              status={run.status}
            />
          </div>
          {renderSparkline && (
            <div className="hidden sm:block opacity-80 group-hover:opacity-100 transition-opacity">
              {renderSparkline(run.id, run.status)}
            </div>
          )}
        </div>
      </Card>
    </a>
  );
}

export function RunRow({ 
  run,
  renderSparkline
}: { 
  run: RunSummaryData;
  renderSparkline?: (runId: string, status: string) => React.ReactNode;
}) {
  const bestLoss = run.best_val_loss ?? run.last_loss;
  return (
    <a
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="flex items-center gap-4 border-b border-border px-4 py-3.5 transition-colors hover:bg-surface-2/50 hover:no-underline last:border-0 group"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors">
            {run.id}
          </span>
          <StatusBadge status={run.status} />
        </div>
        <div className="mt-1 flex items-center gap-2 text-[0.68rem] text-text-muted">
          <span className="text-text-secondary">{fmtParams(run.estimated_params)}</span>
          <span>&middot;</span>
          <span>{run.n_layer}L/{run.n_embd}D/{run.n_head}H</span>
          <span>&middot;</span>
          <span>loss <span className="font-mono text-text-primary/70">{fmtLoss(bestLoss)}</span></span>
          <span>&middot;</span>
          <span>{timeAgo(run.updated_at)}</span>
        </div>
      </div>
      {renderSparkline && (
        <div className="hidden sm:block opacity-70 group-hover:opacity-100 transition-opacity">
          {renderSparkline(run.id, run.status)}
        </div>
      )}
      <div className="w-16 text-right text-[0.68rem] font-mono text-text-muted">
        {pct(run.latest_step, run.total_iters)}%
      </div>
    </a>
  );
}

export function LiveRunCard({ run }: { run: LiveRunData }) {
  const m = run.metrics.length > 0 ? run.metrics[run.metrics.length - 1] : null;
  const isLossSpike = run.metrics.length > 5 && (m?.loss ?? 0) > run.metrics[run.metrics.length - 2].loss * 1.5;
  const isGradSpike = run.metrics.length > 5 && (m?.grad_norm ?? 0) > run.metrics[run.metrics.length - 2].grad_norm * 3;

  return (
    <a href={`/runs/${encodeURIComponent(run.id)}`} className="block hover:no-underline transition-transform active:scale-[0.99]">
      <Card className="hover:border-border-2 transition-colors overflow-hidden group relative">
        <div className="absolute top-0 left-0 w-full h-1 bg-surface-2">
          <div className="h-full bg-green transition-all" style={{ width: `${pct(m?.step ?? 0, run.totalIters)}%` }} />
        </div>
        <div className="p-4 pt-5">
          <div className="flex justify-between items-start mb-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-text-primary group-hover:text-accent transition-colors">{run.id}</h3>
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green"></span>
                </span>
              </div>
              <div className="text-[0.65rem] text-text-muted mt-0.5 flex gap-2">
                <span className="uppercase tracking-wider font-semibold">{run.domain}</span>
                <span>&middot;</span>
                <span>{fmtParams(run.totalParams)} params</span>
              </div>
            </div>
            <div className="text-right">
              <div className="font-mono text-lg font-bold text-yellow">{m ? m.loss.toFixed(4) : "-"}</div>
              <div className="text-[0.6rem] uppercase tracking-wider text-text-muted">Train Loss</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 mb-3 bg-surface-2/50 rounded p-2">
            <div>
              <div className="text-[0.55rem] uppercase tracking-wider text-text-muted font-bold">Step</div>
              <div className="font-mono text-xs text-text-primary">{m ? fmtNum(m.step) : "-"}</div>
            </div>
            <div>
              <div className="text-[0.55rem] uppercase tracking-wider text-text-muted font-bold">Tokens/s</div>
              <div className="font-mono text-xs text-green">{m ? fmtNum(m.tokens_per_sec) : "-"}</div>
            </div>
            <div>
              <div className="text-[0.55rem] uppercase tracking-wider text-text-muted font-bold">Grad Norm</div>
              <div className={`font-mono text-xs ${isGradSpike ? 'text-red font-bold' : 'text-text-primary'}`}>{m ? m.grad_norm.toFixed(3) : "-"}</div>
            </div>
          </div>

          <div className="flex gap-2">
            {isLossSpike && <Badge variant="danger" className="text-[0.55rem] py-0">Loss Spike</Badge>}
            {isGradSpike && <Badge variant="warning" className="text-[0.55rem] py-0">Grad Spike</Badge>}
          </div>
        </div>
      </Card>
    </a>
  );
}
