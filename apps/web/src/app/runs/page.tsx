import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, type DbRunSummary } from "@alpha/db";
import { formatParams, formatLoss, formatNumber, timeAgo, pct } from "@/lib/format";
import { Sparkline } from "@/components/sparkline";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";
import { Badge, Card, Progress } from "@alpha/ui";

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

function DomainBadge({ domain }: { domain: string }) {
  const variantMap: Record<string, "blue" | "warning" | "success" | "danger"> = {
    novels: "blue",
    chords: "warning",
    abc: "success",
    dumb_finance: "danger",
    chaos: "secondary" as any,
  };
  return (
    <Badge variant={variantMap[domain] || "outline"}>
      {domain}
    </Badge>
  );
}

function RunProgressBar({ step, total, status }: { step: number | null; total: number | null; status: string }) {
  const p = pct(step ?? 0, total ?? 0);
  const variantMap: Record<string, "success" | "blue" | "warning" | "danger"> = {
    active: "success",
    completed: "blue",
    stale: "warning",
    failed: "danger",
  };
  return (
    <div className="mt-3">
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

function RunCard({ run }: { run: DbRunSummary }) {
  const bestLoss = run.best_val_loss ?? run.last_loss;
  return (
    <Link
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="block hover:no-underline transition-transform duration-200 active:scale-[0.99]"
    >
      <Card className="hover:border-border-2 transition-colors overflow-hidden group">
        <div className="flex items-start gap-4 p-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[0.95rem] font-bold text-white group-hover:text-accent transition-colors">
                {run.id}
              </span>
              <StatusBadge status={run.status} />
              <DomainBadge domain={run.domain} />
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[0.65rem] text-text-secondary">
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">PARAM</span>
                {formatParams(run.estimated_params)} <Tip text={tips.params} />
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">ARCH</span>
                {run.n_layer}L {run.n_embd}D {run.n_head}H <Tip text={tips.architecture} />
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1.5">
                <span className="text-text-muted">LOSS</span>
                <span className="font-mono text-white">{formatLoss(bestLoss)}</span> <Tip text={tips.loss} />
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
          <div className="hidden sm:block opacity-80 group-hover:opacity-100 transition-opacity">
            <Sparkline runId={run.id} status={run.status} />
          </div>
        </div>
      </Card>
    </Link>
  );
}

export default async function RunsPage() {
  const client = await getClient();
  const runs = await listRuns(client, { limit: 500 });

  const active = runs.filter((r) => r.status === "active").length;
  const completed = runs.filter((r) => r.status === "completed").length;
  const stale = runs.filter((r) => r.status === "stale").length;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Training Runs</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="outline" className="border-border/50 text-text-muted font-normal lowercase tracking-normal">
              {runs.length} total
            </Badge>
            <span className="text-border">/</span>
            <div className="flex items-center gap-3 text-text-muted">
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-green" />
                {active} active
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-blue" />
                {completed} completed
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-yellow" />
                {stale} stale
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
           {/* Future actions: New Run, etc */}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        {runs.map((run) => (
          <RunCard key={run.id} run={run} />
        ))}
      </div>

      {runs.length === 0 && (
        <Card className="py-16 text-center border-dashed">
          <p className="text-sm text-text-muted">
            No training runs found. Sync from disk or start a training run.
          </p>
        </Card>
      )}
    </div>
  );
}
