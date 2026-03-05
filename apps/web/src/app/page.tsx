import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, listDomains, type DbRunSummary } from "@alpha/db";
import { formatParams, formatLoss, formatNumber, timeAgo, pct } from "@/lib/format";
import { Sparkline } from "@/components/sparkline";
import { Badge, Card, CardHeader, CardTitle, CardContent } from "@alpha/ui";

export const dynamic = "force-dynamic";

function StatusBadge({ status }: { status: string }) {
  const variantMap: Record<string, "success" | "blue" | "warning" | "danger"> = {
    active: "success",
    completed: "blue",
    stale: "warning",
    failed: "danger",
  };
  return (
    <Badge variant={variantMap[status] || "secondary"}>
      {status}
    </Badge>
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
    <Card className="px-4 py-3 transition-colors hover:border-border-2">
      <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
      <div className="mt-1 text-[0.62rem] uppercase font-semibold tracking-widest text-text-muted">
        {label}
      </div>
    </Card>
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
      className="flex items-center gap-4 border-b border-border px-4 py-3.5 transition-colors hover:bg-surface-2/50 hover:no-underline last:border-0 group"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-white group-hover:text-accent transition-colors">
            {run.id}
          </span>
          <StatusBadge status={run.status} />
        </div>
        <div className="mt-1 flex items-center gap-2 text-[0.68rem] text-text-muted">
          <span className="text-text-secondary">{formatParams(run.estimated_params)}</span>
          <span>&middot;</span>
          <span>{run.n_layer}L/{run.n_embd}D/{run.n_head}H</span>
          <span>&middot;</span>
          <span>loss <span className="font-mono text-white/70">{formatLoss(bestLoss)}</span></span>
          <span>&middot;</span>
          <span>{timeAgo(run.updated_at)}</span>
        </div>
      </div>
      <div className="hidden sm:block opacity-70 group-hover:opacity-100 transition-opacity">
        <Sparkline runId={run.id} status={run.status} />
      </div>
      <div className="w-16 text-right text-[0.68rem] font-mono text-text-muted">
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
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">Dashboard</h1>
        <p className="mt-1 text-sm text-text-muted">
          Overview of all training activity and engine health.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
        <Card className="lg:col-span-2 overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border py-3">
            <CardTitle>Recent Runs</CardTitle>
            <Link
              href="/runs"
              className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted hover:text-accent transition-colors"
            >
              View all &rarr;
            </Link>
          </CardHeader>
          <div className="flex flex-col">
            {recentRuns.length > 0 ? (
              recentRuns.map((run) => <RunRow key={run.id} run={run} />)
            ) : (
              <div className="px-4 py-12 text-center text-xs text-text-muted">
                No training runs found.
              </div>
            )}
          </div>
        </Card>

        {/* Domains summary */}
        <Card className="overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between border-b border-border py-3">
            <CardTitle>Domains</CardTitle>
            <Link
              href="/domains"
              className="text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted hover:text-accent transition-colors"
            >
              View all &rarr;
            </Link>
          </CardHeader>
          <div className="flex flex-col divide-y divide-border/50">
            {domains.map((d) => {
              const domainRuns = runs.filter((r) => r.domain === d.id);
              const variantMap: Record<string, "blue" | "warning" | "success" | "danger"> = {
                novels: "blue",
                chords: "warning",
                abc: "success",
              };
              return (
                <Link
                  key={d.id}
                  href="/domains"
                  className="flex items-center gap-3 px-4 py-3.5 hover:bg-surface-2/50 hover:no-underline group"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white group-hover:text-accent transition-colors">
                        {d.display_name}
                      </span>
                      <Badge variant={variantMap[d.id] || "outline"}>
                        {d.id}
                      </Badge>
                    </div>
                    <div className="mt-0.5 text-[0.68rem] text-text-muted">
                      {d.tokenizer} tokenizer
                    </div>
                  </div>
                  <div className="text-[0.68rem] font-mono text-text-secondary bg-surface-2 px-2 py-0.5 rounded border border-border/50">
                    {domainRuns.length} runs
                  </div>
                </Link>
              );
            })}
          </div>
        </Card>
      </div>
    </div>
  );
}
