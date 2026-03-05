import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, listDomains } from "@alpha/db";
import { formatNumber } from "@/lib/format";
import { Sparkline } from "@/components/sparkline";
import { Badge, Card, CardHeader, CardTitle, StatCard, RunRow } from "@alpha/ui";

export const dynamic = "force-dynamic";

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
        <h1 className="text-2xl font-bold tracking-tight text-text-primary">Dashboard</h1>
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
              recentRuns.map((run) => (
                <RunRow 
                  key={run.id} 
                  run={run as any} 
                  renderSparkline={(runId, status) => <Sparkline runId={runId} status={status} />}
                />
              ))
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
                      <span className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors">
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
