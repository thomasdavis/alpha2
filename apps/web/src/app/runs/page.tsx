import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, type DbRunSummary } from "@alpha/db";
import { formatParams, formatLoss, formatNumber, timeAgo, pct } from "@/lib/format";
import { Sparkline } from "@/components/sparkline";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";

export const dynamic = "force-dynamic";

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "bg-green-bg text-green",
    completed: "bg-blue-bg text-blue",
    stale: "bg-yellow-bg text-yellow",
    failed: "bg-red-bg text-red",
  };
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide ${colors[status] ?? "bg-surface-2 text-text-secondary"}`}
    >
      {status}
    </span>
  );
}

function DomainBadge({ domain }: { domain: string }) {
  const colors: Record<string, string> = {
    novels: "bg-blue-bg text-blue",
    chords: "bg-yellow-bg text-yellow",
    abc: "bg-green-bg text-green",
    dumb_finance: "bg-red-bg text-red",
  };
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide ${colors[domain] ?? "bg-surface-2 text-text-secondary"}`}
    >
      {domain}
    </span>
  );
}

function ProgressBar({ step, total, status }: { step: number; total: number; status: string }) {
  const p = pct(step, total);
  const colors: Record<string, string> = {
    active: "bg-green",
    completed: "bg-blue",
    stale: "bg-yellow",
    failed: "bg-red",
  };
  return (
    <div className="mt-2">
      <div className="h-1 rounded-full bg-surface-2">
        <div
          className={`h-full rounded-full transition-all ${colors[status] ?? "bg-text-muted"}`}
          style={{ width: `${p}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[0.68rem] text-text-muted">
        <span>
          Step {step.toLocaleString()} / {total.toLocaleString()}
        </span>
        <span>{p}%</span>
      </div>
    </div>
  );
}

function RunCard({ run }: { run: DbRunSummary }) {
  const bestLoss = run.best_val_loss ?? run.last_loss;
  return (
    <Link
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="block rounded-lg border border-border bg-surface transition-colors hover:border-border-2 hover:no-underline"
    >
      <div className="flex items-start gap-4 p-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[0.95rem] font-semibold text-white">
              {run.id}
            </span>
            <StatusBadge status={run.status} />
            <DomainBadge domain={run.domain} />
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-text-secondary">
            <span>{formatParams(run.estimated_params)} params <Tip text={tips.params} /></span>
            <span className="text-text-muted">|</span>
            <span>
              {run.n_layer}L {run.n_embd}D {run.n_head}H <Tip text={tips.architecture} />
            </span>
            <span className="text-text-muted">|</span>
            <span>loss {formatLoss(bestLoss)} <Tip text={tips.loss} /></span>
            <span className="text-text-muted">|</span>
            <span>{run.metric_count} metrics <Tip text={tips.metricCount} /></span>
            <span className="text-text-muted">|</span>
            <span>{run.checkpoint_count} ckpts <Tip text={tips.checkpointCount} /></span>
            <span className="text-text-muted">|</span>
            <span>{timeAgo(run.updated_at)}</span>
          </div>
          <ProgressBar
            step={run.latest_step}
            total={run.total_iters}
            status={run.status}
          />
        </div>
        <Sparkline runId={run.id} status={run.status} />
      </div>
    </Link>
  );
}

export default async function RunsPage() {
  const client = getClient();
  const runs = await listRuns(client);

  const active = runs.filter((r) => r.status === "active").length;
  const completed = runs.filter((r) => r.status === "completed").length;
  const stale = runs.filter((r) => r.status === "stale").length;

  return (
    <>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">Training Runs</h1>
          <p className="mt-0.5 text-xs text-text-muted">
            {runs.length} runs &middot; {active} active &middot; {completed} completed &middot; {stale} stale
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        {runs.map((run) => (
          <RunCard key={run.id} run={run} />
        ))}
      </div>

      {runs.length === 0 && (
        <div className="py-12 text-center text-sm text-text-muted">
          No training runs found. Sync from disk or start a training run.
        </div>
      )}
    </>
  );
}
