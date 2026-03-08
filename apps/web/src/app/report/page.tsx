import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns } from "@alpha/db";
import { Card, CardContent, Badge, fmtParams, fmtLoss, fmtNum, fmtDate } from "@alpha/ui";

export const dynamic = "force-dynamic";

function estimateParams(r: any): number {
  if (r.estimated_params) return r.estimated_params;
  const V = r.vocab_size, D = r.n_embd, L = r.n_layer, C = r.block_size;
  const ffnDim = D * 4;
  return V * D + C * D + L * (4 * D * D + 2 * D + ffnDim * D + D * ffnDim + 2 * D) + D + D + V * D;
}

export default async function ReportIndexPage() {
  const client = await getClient();
  const runs = await listRuns(client, {});

  return (
    <div>
      <h1 className="text-2xl font-bold text-text-primary mb-2">Model Reports</h1>
      <p className="text-text-secondary mb-6">
        Auto-generated training reports for every run. Each report includes metrics, architecture details, samples, and a live chat widget.
      </p>

      {runs.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-text-muted">
            No training runs found. Start training to generate reports.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(runs as any[]).map((run) => {
            const params = estimateParams(run);
            return (
              <Link key={run.id} href={`/report/${run.id}`} className="hover:no-underline">
                <Card className="hover:border-accent/50 transition-colors h-full">
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-text-primary truncate">
                        {run.run_id || run.id}
                      </span>
                      <Badge variant={run.status === "completed" ? "success" : run.status === "running" ? "blue" : "default"}>
                        {run.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-text-muted mb-3">
                      <span>{run.domain}</span>
                      <span>{fmtParams(params)}</span>
                      <span>{run.n_layer}L/{run.n_embd}D/{run.n_head}H</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <div className="text-text-muted">Loss</div>
                        <div className="text-text-primary font-mono">{fmtLoss(run.last_loss)}</div>
                      </div>
                      <div>
                        <div className="text-text-muted">Val Loss</div>
                        <div className="text-text-primary font-mono">{fmtLoss(run.best_val_loss)}</div>
                      </div>
                      <div>
                        <div className="text-text-muted">Step</div>
                        <div className="text-text-primary font-mono">{fmtNum(run.latest_step)} / {fmtNum(run.total_iters)}</div>
                      </div>
                    </div>
                    <div className="mt-3 text-[0.6rem] text-text-muted">
                      {fmtDate(run.created_at)}
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
