import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, listDomains, type DbRunSummary } from "@alpha/db";
import { formatParams, formatLoss, formatNumber, timeAgo, pct } from "@/lib/format";
import { Sparkline } from "@/components/sparkline";

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

function StatCard({
  value,
  label,
  href,
}: {
  value: string | number;
  label: string;
  href?: string;
}) {
  const content = (
    <div className="rounded-lg border border-border bg-surface px-4 py-3 transition-colors hover:border-border-2">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="mt-0.5 text-[0.68rem] uppercase tracking-wider text-text-muted">
        {label}
      </div>
    </div>
  );
  if (href) {
    return (
      <Link href={href} className="hover:no-underline">
        {content}
      </Link>
    );
  }
  return content;
}

function RunRow({ run }: { run: DbRunSummary }) {
  const bestLoss = run.best_val_loss ?? run.last_loss;
  return (
    <Link
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="flex items-center gap-4 border-b border-border px-4 py-3 transition-colors hover:bg-surface-2 hover:no-underline last:border-0"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">{run.id}</span>
          <StatusBadge status={run.status} />
        </div>
        <div className="mt-0.5 text-xs text-text-muted">
          {formatParams(run.estimated_params)} &middot; {run.n_layer}L/{run.n_embd}D/{run.n_head}H &middot; loss {formatLoss(bestLoss)} &middot; {timeAgo(run.updated_at)}
        </div>
      </div>
      <div className="hidden sm:block">
        <Sparkline runId={run.id} status={run.status} />
      </div>
      <div className="w-16 text-right text-xs text-text-muted">
        {pct(run.latest_step, run.total_iters)}%
      </div>
    </Link>
  );
}

export default async function DashboardPage() {
  const client = await getClient();
  const [runs, domains] = await Promise.all([
    listRuns(client),
    listDomains(client),
  ]);

  const active = runs.filter((r) => r.status === "active").length;
  const completed = runs.filter((r) => r.status === "completed").length;
  const stale = runs.filter((r) => r.status === "stale").length;
  const totalSteps = runs.reduce((s, r) => s + (r.latest_step || 0), 0);
  const totalMetrics = runs.reduce((s, r) => s + (r.metric_count || 0), 0);
  const totalCheckpoints = runs.reduce((s, r) => s + (r.checkpoint_count || 0), 0);
  const totalParams = runs.reduce((s, r) => s + (r.estimated_params || 0), 0);
  const losses = runs
    .map((r) => r.best_val_loss ?? r.last_loss)
    .filter((v): v is number => v != null);
  const bestLoss = losses.length > 0 ? Math.min(...losses) : null;

  const recentRuns = runs.slice(0, 5);

  return (
    <>
      <h1 className="mb-1 text-lg font-bold text-white">Dashboard</h1>
      <p className="mb-6 text-xs text-text-muted">
        Overview of all training activity
      </p>

      {/* Stats */}
      <div className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard value={runs.length} label="Total Runs" href="/runs" />
        <StatCard value={active} label="Active" />
        <StatCard value={completed} label="Completed" />
        <StatCard value={stale} label="Stale" />
        <StatCard value={formatNumber(totalSteps)} label="Total Steps" />
        <StatCard value={formatNumber(totalMetrics)} label="Metrics" />
        <StatCard value={totalCheckpoints} label="Checkpoints" href="/checkpoints" />
        <StatCard
          value={bestLoss != null ? bestLoss.toFixed(4) : "-"}
          label="Best Loss"
        />
      </div>

      {/* Two-column: Recent runs + domains */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent runs */}
        <div className="rounded-lg border border-border bg-surface lg:col-span-2">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Recent Runs
            </h2>
            <Link
              href="/runs"
              className="text-xs text-text-muted hover:text-accent"
            >
              View all &rarr;
            </Link>
          </div>
          {recentRuns.length > 0 ? (
            recentRuns.map((run) => <RunRow key={run.id} run={run} />)
          ) : (
            <div className="px-4 py-8 text-center text-xs text-text-muted">
              No training runs found.
            </div>
          )}
        </div>

        {/* Domains summary */}
        <div className="rounded-lg border border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Domains
            </h2>
            <Link
              href="/domains"
              className="text-xs text-text-muted hover:text-accent"
            >
              View all &rarr;
            </Link>
          </div>
          <div className="divide-y divide-border">
            {domains.map((d) => {
              const domainRuns = runs.filter((r) => r.domain === d.id);
              const colorMap: Record<string, string> = {
                novels: "bg-blue",
                chords: "bg-yellow",
                abc: "bg-green",
              };
              return (
                <Link
                  key={d.id}
                  href="/domains"
                  className="flex items-center gap-3 px-4 py-3 hover:bg-surface-2 hover:no-underline"
                >
                  <div
                    className={`h-2 w-2 rounded-full ${colorMap[d.id] ?? "bg-text-muted"}`}
                  />
                  <div className="flex-1">
                    <div className="text-sm text-white">{d.display_name}</div>
                    <div className="text-[0.68rem] text-text-muted">
                      {d.tokenizer} tokenizer
                    </div>
                  </div>
                  <div className="text-xs text-text-secondary">
                    {domainRuns.length} runs
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
