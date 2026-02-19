import { getClient } from "@/lib/db";
import { listDomains, listRuns } from "@alpha/db";

export const dynamic = "force-dynamic";

export default async function DomainsPage() {
  const client = getClient();
  const [domains, runs] = await Promise.all([
    listDomains(client),
    listRuns(client),
  ]);

  return (
    <>
      <h1 className="mb-6 text-xl font-bold text-white">Domains</h1>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {domains.map((d) => {
          const domainRuns = runs.filter((r) => r.domain === d.id);
          const completed = domainRuns.filter(
            (r) => r.status === "completed"
          ).length;
          const active = domainRuns.filter(
            (r) => r.status === "active"
          ).length;
          const totalMetrics = domainRuns.reduce(
            (s, r) => s + (r.metric_count || 0),
            0
          );
          const losses = domainRuns
            .map((r) => r.best_val_loss ?? r.last_loss)
            .filter((v): v is number => v != null);
          const bestLoss =
            losses.length > 0 ? Math.min(...losses) : null;

          const colorMap: Record<string, string> = {
            novels: "border-l-blue",
            chords: "border-l-yellow",
            abc: "border-l-green",
          };

          return (
            <div
              key={d.id}
              className={`rounded-lg border border-border border-l-4 bg-surface p-5 ${colorMap[d.id] ?? "border-l-text-muted"}`}
            >
              <h2 className="mb-1 text-base font-semibold text-white">
                {d.display_name}
              </h2>
              <div className="mb-3 text-xs text-text-muted">
                {d.tokenizer} tokenizer
              </div>

              <div className="mb-3 space-y-1 text-xs text-text-secondary">
                <div className="flex justify-between">
                  <span>Runs</span>
                  <span className="text-white">{domainRuns.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Active / Completed</span>
                  <span className="text-white">
                    {active} / {completed}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Total metrics</span>
                  <span className="text-white">
                    {totalMetrics.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Best loss</span>
                  <span className="text-white">
                    {bestLoss != null ? bestLoss.toFixed(4) : "-"}
                  </span>
                </div>
              </div>

              <div className="border-t border-border pt-3">
                <div className="mb-1.5 text-[0.68rem] uppercase text-text-muted">
                  Sample prompts
                </div>
                <div className="space-y-1">
                  {d.sample_prompts.slice(0, 3).map((p, i) => (
                    <div
                      key={i}
                      className="truncate rounded bg-surface-2 px-2 py-1 font-mono text-[0.72rem] text-text-secondary"
                    >
                      {p}
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-3 border-t border-border pt-3">
                <div className="mb-1.5 text-[0.68rem] uppercase text-text-muted">
                  Model defaults
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(d.model_defaults).map(([k, v]) => (
                    <span
                      key={k}
                      className="rounded bg-surface-2 px-1.5 py-0.5 text-[0.68rem] text-text-secondary"
                    >
                      {k}: {String(v)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
