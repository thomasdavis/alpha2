import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns } from "@alpha/db";
import { Sparkline } from "@/components/sparkline";
import { Badge, Card, RunCard } from "@alpha/ui";

export const dynamic = "force-dynamic";

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
          <h1 className="text-2xl font-bold tracking-tight text-text-primary">Training Runs</h1>
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
          <RunCard 
            key={run.id} 
            run={run as any} 
            renderSparkline={(runId, status) => <Sparkline runId={runId} status={status} />}
          />
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
