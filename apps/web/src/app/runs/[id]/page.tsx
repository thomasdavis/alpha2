import Link from "next/link";
import { notFound } from "next/navigation";
import { getClient } from "@/lib/db";
import { getRun, getMetrics, listCheckpoints } from "@alpha/db";
import { formatParams, formatLoss, formatBytes, timeAgo, pct } from "@/lib/format";
import { LossChart } from "@/components/loss-chart";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";

export const dynamic = "force-dynamic";

function Detail({ label, value, tip }: { label: string; value: string | number | null; tip?: string }) {
  return (
    <tr>
      <td className="py-0.5 pr-4 text-xs text-text-muted">
        {label}
        {tip && <Tip text={tip} />}
      </td>
      <td className="py-0.5 font-mono text-xs text-text-primary">
        {value ?? "-"}
      </td>
    </tr>
  );
}

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const client = getClient();
  const run = await getRun(client, id);
  if (!run) notFound();

  const [metrics, checkpoints] = await Promise.all([
    getMetrics(client, id),
    listCheckpoints(client, id),
  ]);

  const p = pct(run.latest_step, run.total_iters);
  const statusColors: Record<string, string> = {
    active: "bg-green-bg text-green",
    completed: "bg-blue-bg text-blue",
    stale: "bg-yellow-bg text-yellow",
    failed: "bg-red-bg text-red",
  };
  const domainColors: Record<string, string> = {
    novels: "bg-blue-bg text-blue",
    chords: "bg-yellow-bg text-yellow",
    abc: "bg-green-bg text-green",
    dumb_finance: "bg-red-bg text-red",
  };
  const barColors: Record<string, string> = {
    active: "bg-green",
    completed: "bg-blue",
    stale: "bg-yellow",
    failed: "bg-red",
  };

  const recent = metrics.slice(-20);
  const avgTps = recent.length > 0
    ? recent.reduce((s, m) => s + (m.tokens_per_sec as number), 0) / recent.length
    : 0;
  const avgMsPerIter = recent.length > 0
    ? recent.reduce((s, m) => s + (m.ms_per_iter as number), 0) / recent.length
    : 0;

  return (
    <>
      {/* Breadcrumb */}
      <nav className="mb-4 flex items-center gap-1.5 text-xs text-text-muted">
        <Link href="/runs" className="hover:text-text-secondary">
          Training Runs
        </Link>
        <span>/</span>
        <span className="text-text-secondary">{run.id}</span>
      </nav>

      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-bold text-white">{run.id}</h1>
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${statusColors[run.status] ?? "bg-surface-2 text-text-secondary"}`}
        >
          {run.status}
        </span>
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${domainColors[run.domain] ?? "bg-surface-2 text-text-secondary"}`}
        >
          {run.domain}
        </span>
        <span className="ml-auto text-xs text-text-muted">
          Updated {timeAgo(run.updated_at)}
        </span>
      </div>

      {/* Progress */}
      <div className="mb-6 rounded-lg border border-border bg-surface p-4">
        <div className="mb-2 flex justify-between text-sm text-text-secondary">
          <span>
            Step {run.latest_step.toLocaleString()} /{" "}
            {run.total_iters.toLocaleString()}
          </span>
          <span>{p}%</span>
        </div>
        <div className="h-2 rounded-full bg-surface-2">
          <div
            className={`h-full rounded-full ${barColors[run.status] ?? "bg-text-muted"}`}
            style={{ width: `${p}%` }}
          />
        </div>
      </div>

      {/* Stats grid */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="text-lg font-bold text-white">
            {formatParams(run.estimated_params)}
          </div>
          <div className="text-[0.68rem] uppercase text-text-muted">Params <Tip text={tips.params} /></div>
        </div>
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="text-lg font-bold text-white">
            {formatLoss(run.last_loss)}
          </div>
          <div className="text-[0.68rem] uppercase text-text-muted">
            Last loss <Tip text={tips.lastLoss} />
          </div>
        </div>
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="text-lg font-bold text-white">
            {formatLoss(run.best_val_loss)}
          </div>
          <div className="text-[0.68rem] uppercase text-text-muted">
            Best val loss <Tip text={tips.bestValLoss} />
          </div>
        </div>
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="text-lg font-bold text-white">
            {avgTps > 0 ? avgTps.toFixed(0) : "-"}
          </div>
          <div className="text-[0.68rem] uppercase text-text-muted">
            Avg tok/sec <Tip text={tips.tokPerSec} />
          </div>
        </div>
      </div>

      {/* Loss chart */}
      <div className="mb-6 rounded-lg border border-border bg-surface p-4">
        <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
          Loss over training <Tip text={tips.lossChart} />
        </h2>
        <LossChart runId={run.id} />
      </div>

      {/* Config + checkpoints */}
      <div className="grid gap-6 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Model Architecture <Tip text={tips.architecture} />
          </h2>
          <table className="w-full">
            <tbody>
              <Detail label="Vocab size" value={run.vocab_size} tip={tips.vocabSize} />
              <Detail label="Block size" value={run.block_size} tip={tips.blockSize} />
              <Detail label="Layers" value={run.n_layer} tip={tips.nLayer} />
              <Detail label="Embedding dim" value={run.n_embd} tip={tips.nEmbd} />
              <Detail label="Heads" value={run.n_head} tip={tips.nHead} />
              <Detail label="Dropout" value={run.dropout} tip={tips.dropout} />
              <Detail label="Est. params" value={formatParams(run.estimated_params)} tip={tips.params} />
            </tbody>
          </table>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4">
          <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Training Config
          </h2>
          <table className="w-full">
            <tbody>
              <Detail label="Total iters" value={run.total_iters} tip={tips.totalIters} />
              <Detail label="Batch size" value={run.batch_size} tip={tips.batchSize} />
              <Detail label="Learning rate" value={run.lr} tip={tips.lr} />
              <Detail label="Seed" value={run.seed} tip={tips.seed} />
              <Detail label="Backend" value={run.backend} tip={tips.backend} />
              <Detail label="Tokenizer" value={run.tokenizer} tip={tips.tokenizer} />
              <Detail label="Optimizer" value={run.optimizer} tip={tips.optimizer} />
              <Detail label="Avg ms/iter" value={avgMsPerIter > 0 ? avgMsPerIter.toFixed(1) + " ms" : "-"} tip={tips.msPerIter} />
            </tbody>
          </table>
        </div>

        <div className="rounded-lg border border-border bg-surface p-4 sm:col-span-2">
          <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Checkpoints ({checkpoints.length}) <Tip text={tips.checkpoint} />
          </h2>
          {checkpoints.length === 0 ? (
            <p className="text-xs text-text-muted">No checkpoints saved</p>
          ) : (
            <div className="space-y-1">
              {checkpoints.map((c: any) => (
                <div
                  key={c.step}
                  className="flex justify-between border-b border-border py-1.5 text-xs last:border-0"
                >
                  <span className="font-mono text-text-primary">
                    checkpoint-{c.step}.json
                  </span>
                  <span className="text-text-muted">
                    {formatBytes(c.file_size)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
