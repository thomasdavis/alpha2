import Link from "next/link";
import { notFound } from "next/navigation";
import { getClient } from "@/lib/db";
import { getRun, getMetrics, listCheckpoints } from "@alpha/db";
import { formatParams, formatLoss, formatBytes, timeAgo, pct } from "@/lib/format";
import { LossChart } from "@/components/loss-chart";
import { TrainingDataPreview } from "./training-data";
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

// Check if model is loaded in the inference engine (direct import, no HTTP)
async function checkInferenceStatus(id: string): Promise<boolean> {
  try {
    const { getRuns } = await import("@/lib/engine");
    const runs = getRuns();
    return runs.some((r) => r.id === id);
  } catch {
    return false;
  }
}

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const client = await getClient();
  const run = await getRun(client, id);
  if (!run) notFound();

  const [metrics, checkpoints, inferenceAvailable] = await Promise.all([
    getMetrics(client, id),
    listCheckpoints(client, id),
    checkInferenceStatus(id),
  ]);

  const p = pct(run.latest_step ?? 0, run.total_iters ?? 0);
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
    chaos: "bg-purple-bg text-purple",
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

  // Parse stored JSON configs if available
  let modelConfigJson: string | null = null;
  let trainConfigJson: string | null = null;
  try {
    if ((run as any).model_config) modelConfigJson = JSON.stringify(JSON.parse((run as any).model_config), null, 2);
  } catch { /* not available */ }
  try {
    if ((run as any).train_config) trainConfigJson = JSON.stringify(JSON.parse((run as any).train_config), null, 2);
  } catch { /* not available */ }

  return (
    <>
      {/* Breadcrumb */}
      <nav className="mb-4 flex items-center gap-1.5 text-xs text-text-muted">
        <Link href="/models" className="hover:text-text-secondary">
          Models
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
        <Tip text={run.status === "active" ? tips.statusActive : run.status === "completed" ? tips.statusCompleted : run.status === "stale" ? tips.statusStale : tips.statusFailed} />
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold uppercase ${domainColors[run.domain] ?? "bg-surface-2 text-text-secondary"}`}
        >
          {run.domain}
        </span>
        <Tip text={tips.domain} />
        <span className="ml-auto text-xs text-text-muted">
          Updated {timeAgo(run.updated_at)}
        </span>
      </div>

      {/* Inference status */}
      <div className="mb-6 rounded-lg border border-border bg-surface p-4">
        {inferenceAvailable ? (
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-green-bg px-3 py-1 text-xs font-semibold text-green">
              <span className="h-1.5 w-1.5 rounded-full bg-green" />
              Available for inference
            </span>
            <Tip text={tips.inferenceAvailable} />
            <Link
              href={`/chat?model=${id}`}
              className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent/80 hover:no-underline"
            >
              Open in Chat
            </Link>
            <Link
              href={`/inference?model=${id}`}
              className="rounded-md border border-border-2 bg-surface-2 px-3 py-1.5 text-xs text-text-secondary transition-colors hover:bg-border hover:text-text-primary hover:no-underline"
            >
              Inference
            </Link>
          </div>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-xs text-text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-text-muted" />
            Not loaded in inference engine
            <Tip text={tips.inferenceUnavailable} />
          </span>
        )}
      </div>

      {/* Stats grid */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-border bg-surface px-4 py-3">
          <div className="text-lg font-bold text-white">
            {formatParams(run.estimated_params)}
          </div>
          <div className="text-[0.68rem] uppercase text-text-muted">
            Params <Tip text={tips.params} />
          </div>
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

      {/* Progress */}
      <div className="mb-6 rounded-lg border border-border bg-surface p-4">
        <div className="mb-2 flex justify-between text-sm text-text-secondary">
          <span>
            Step {(run.latest_step ?? 0).toLocaleString()} /{" "}
            {(run.total_iters ?? 0).toLocaleString()}
            <Tip text={tips.step} />
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

        {/* Raw JSON configs */}
        {(modelConfigJson || trainConfigJson) && (
          <div className="rounded-lg border border-border bg-surface p-4 sm:col-span-2">
            <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
              Raw Config JSON <Tip text={tips.rawConfig} />
            </h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {modelConfigJson && (
                <div>
                  <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-text-muted">model_config</div>
                  <pre className="max-h-48 overflow-auto rounded bg-[#0d0d0d] p-3 text-[0.7rem] leading-relaxed text-text-secondary">{modelConfigJson}</pre>
                </div>
              )}
              {trainConfigJson && (
                <div>
                  <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-text-muted">train_config</div>
                  <pre className="max-h-48 overflow-auto rounded bg-[#0d0d0d] p-3 text-[0.7rem] leading-relaxed text-text-secondary">{trainConfigJson}</pre>
                </div>
              )}
            </div>
          </div>
        )}

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

        {/* Training data preview */}
        <div className="rounded-lg border border-border bg-surface p-4 sm:col-span-2">
          <h2 className="mb-3 text-xs uppercase tracking-wider text-text-muted">
            Training Data <Tip text={tips.trainingData} />
          </h2>
          <TrainingDataPreview runId={run.id} />
        </div>
      </div>
    </>
  );
}
