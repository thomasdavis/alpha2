import Link from "next/link";
import { getCorpusReader } from "@/lib/corpus";
import type { CorpusReviewCampaignProgress } from "@alpha/corpus";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const runtime = "nodejs";

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC"
  }).format(date) + " UTC";
}

type PipelineStage = {
  label: string;
  detail: string;
  completed: number;
  total: number;
  assigned?: number;
  unlocked: boolean;
};

function stageStatus(stage: PipelineStage): "complete" | "current" | "locked" {
  if (stage.total > 0 && stage.completed >= stage.total) return "complete";
  return stage.unlocked ? "current" : "locked";
}

function nextReviewAction(progress: CorpusReviewCampaignProgress): string {
  if (progress.passA.completed < progress.passA.total) {
    return progress.passA.assigned > 0
      ? `Complete and locally import the ${progress.passA.assigned} open Pass A assignments. They are one session within the ${progress.candidates}-candidate census.`
      : `Prepare the next blinded Pass A session; ${progress.passA.completed} of ${progress.passA.total} candidates are sealed.`;
  }
  if (progress.hiddenRepeats.completed < progress.hiddenRepeats.total) {
    return `Complete the hidden Pass A repeat presentations before revealing contracts; ${progress.hiddenRepeats.completed} of ${progress.hiddenRepeats.total} are sealed.`;
  }
  if (progress.passB.completed < progress.passB.total) {
    return `Complete contract-aware Pass B for all ${progress.passB.total} candidates without altering the sealed Pass A evidence.`;
  }
  if (progress.passC.completed < progress.passC.total
    || progress.structuralDispositions.completed < progress.structuralDispositions.total) {
    return `Complete all ${progress.passC.total} family syntheses and ${progress.structuralDispositions.total} separate structural dispositions.`;
  }
  if (progress.passD.completed < progress.passD.total) {
    return "Complete the non-binding Pass D campaign closeout. Its recommendations cannot authorize generation, release, training, or compute.";
  }
  return "D5 evidence is closed. Any next experiment still requires a separately bounded operator authorization.";
}

function ReviewPipeline({ progress }: { progress: CorpusReviewCampaignProgress }) {
  const passAComplete = progress.passA.completed >= progress.passA.total;
  const repeatsComplete = progress.hiddenRepeats.completed >= progress.hiddenRepeats.total;
  const passBComplete = progress.passB.completed >= progress.passB.total;
  const passCComplete = progress.passC.completed >= progress.passC.total;
  const structuralComplete = progress.structuralDispositions.completed >= progress.structuralDispositions.total;
  const stages: PipelineStage[] = [
    {
      label: "Pass A",
      detail: "Blind conversation",
      ...progress.passA,
      unlocked: true
    },
    {
      label: "Repeats",
      detail: "Blind stability",
      completed: progress.hiddenRepeats.completed,
      total: progress.hiddenRepeats.total,
      assigned: progress.hiddenRepeats.assigned,
      unlocked: passAComplete
    },
    {
      label: "Pass B",
      detail: "Contract aware",
      ...progress.passB,
      unlocked: passAComplete && repeatsComplete
    },
    {
      label: "Pass C",
      detail: "Family synthesis",
      ...progress.passC,
      unlocked: passAComplete && repeatsComplete && passBComplete
    },
    {
      label: "Structural",
      detail: "Rejected cases",
      completed: progress.structuralDispositions.completed,
      total: progress.structuralDispositions.total,
      unlocked: passAComplete && repeatsComplete && passBComplete
    },
    {
      label: "Pass D",
      detail: "Campaign closeout",
      ...progress.passD,
      unlocked: passAComplete && repeatsComplete && passBComplete && passCComplete && structuralComplete
    }
  ];

  return (
    <article className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-4">
        <div>
          <p className="font-mono text-xs font-semibold text-text-primary">{progress.campaignSlug}</p>
          <p className="mt-1 text-xs text-text-muted">
            reviewer {progress.reviewerAlias} · {progress.candidates} candidates · {progress.families} families
          </p>
        </div>
        <span className={`rounded border px-2 py-1 text-[0.68rem] font-semibold ${
          progress.passD.executionAuthorizations === 0
            ? "border-green/30 bg-green-bg text-text-primary"
            : "border-red/30 bg-red-bg text-text-primary"
        }`}>
          {progress.passD.executionAuthorizations === 0 ? "No execution authority" : "Authority anomaly"}
        </span>
      </div>
      <div className="grid gap-px bg-border sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {stages.map((stage) => {
          const status = stageStatus(stage);
          return (
            <div key={stage.label} className="bg-surface px-4 py-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-text-primary">{stage.label}</p>
                <span className={`text-[0.64rem] font-semibold uppercase tracking-[0.08em] ${
                  status === "complete" ? "text-text-primary" : status === "current" ? "text-blue" : "text-text-muted"
                }`}>
                  {status}
                </span>
              </div>
              <p className="mt-1 text-[0.68rem] text-text-muted">{stage.detail}</p>
              <p className="mt-4 font-mono text-lg font-semibold tabular-nums text-text-primary">
                {stage.completed}<span className="text-text-muted"> / {stage.total}</span>
              </p>
              {stage.assigned !== undefined && stage.assigned > 0 ? (
                <p className="mt-1 text-[0.68rem] text-text-secondary">{stage.assigned} open</p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="border-t border-border bg-surface-2 px-4 py-4">
        <p className="text-xs font-semibold uppercase tracking-[0.08em] text-text-muted">Current gate</p>
        <p className="mt-1 text-sm leading-6 text-text-primary">{nextReviewAction(progress)}</p>
        <p className="mt-1 text-xs leading-5 text-text-muted">
          Browser drafts are not evidence until the downloaded packet passes the local importer.
        </p>
      </div>
    </article>
  );
}

export default function CorpusReviewPage() {
  const reader = getCorpusReader();
  const packets = reader.listReviewPackets();
  const progress = [...new Map(
    packets.map((packet) => [`${packet.campaignSlug}\0${packet.reviewerAlias}`, packet] as const)
  ).values()]
    .map((packet) => reader.reviewCampaignProgress(packet.campaignSlug, packet.reviewerAlias))
    .filter((entry): entry is CorpusReviewCampaignProgress => entry !== null);

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">Human review workspace</h1>
            <span className="rounded border border-blue/30 bg-blue-bg px-2 py-1 text-[0.68rem] font-semibold text-text-primary">
              Local draft only
            </span>
          </div>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-text-secondary">
            Review blinded Alpha Corpus candidates with the versioned D5 rubric. Answers autosave in this
            browser and leave the server only when you download the completed JSON packet.
          </p>
        </div>
        <Link
          href="/corpus"
          className="min-h-11 rounded-md border border-border-2 px-3 py-2.5 text-sm font-medium text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          Browse ledger
        </Link>
      </header>

      <section aria-labelledby="review-security-heading" className="grid gap-4 border-b border-border pb-8 md:grid-cols-3">
        <div>
          <h2 id="review-security-heading" className="text-xs font-semibold uppercase tracking-[0.1em] text-text-muted">
            Review boundary
          </h2>
        </div>
        <div className="md:col-span-2">
          <p className="text-sm leading-6 text-text-primary">
            This public page has no mutation endpoint. Downloaded packets must be imported locally with the
            validated <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs">review-submit</code> command.
            Import preserves the exact submission as a content-addressed artifact and does not automatically
            promote any candidate into a release or training run.
          </p>
        </div>
      </section>

      {progress.length > 0 ? (
        <section aria-labelledby="campaign-pipeline-heading">
          <div>
            <h2 id="campaign-pipeline-heading" className="text-base font-semibold text-text-primary">
              D5 campaign pipeline
            </h2>
            <p className="mt-1 max-w-3xl text-xs leading-5 text-text-muted">
              Reviewer-scoped counts derived from the public ledger. No candidate IDs, family labels, hidden
              contracts, structural status, or repeat identity are exposed here.
            </p>
          </div>
          <div className="mt-4 space-y-4">
            {progress.map((entry) => (
              <ReviewPipeline key={`${entry.campaignSlug}:${entry.reviewerAlias}`} progress={entry} />
            ))}
          </div>
        </section>
      ) : null}

      <section aria-labelledby="open-sessions-heading">
        <div className="flex items-baseline justify-between gap-4">
          <div>
            <h2 id="open-sessions-heading" className="text-base font-semibold text-text-primary">Review sessions</h2>
            <p className="mt-1 text-xs text-text-muted">Latest verified packet for each immutable session.</p>
          </div>
          <span className="font-mono text-xs tabular-nums text-text-muted">{packets.length}</span>
        </div>

        {packets.length === 0 ? (
          <p className="mt-4 border-y border-border py-10 text-center text-sm text-text-muted">
            No human-review packets have been exported yet.
          </p>
        ) : (
          <div className="mt-4 overflow-hidden rounded-lg border border-border bg-surface">
            <div className="hidden grid-cols-[minmax(0,1fr)_7rem_8rem_12rem_7rem] gap-4 border-b border-border bg-surface-2 px-4 py-2 text-[0.68rem] font-semibold uppercase tracking-[0.08em] text-text-muted md:grid">
              <span>Session</span><span>Pass</span><span>Assignments</span><span>Exported</span><span className="text-right">Action</span>
            </div>
            <ul className="divide-y divide-border">
              {packets.map((packet) => (
                <li key={packet.sessionId} className="grid gap-3 px-4 py-4 md:grid-cols-[minmax(0,1fr)_7rem_8rem_12rem_7rem] md:items-center md:gap-4">
                  <div className="min-w-0">
                    <p className="truncate font-mono text-xs font-medium text-text-primary" title={packet.sessionId}>{packet.sessionId}</p>
                    <p className="mt-1 text-xs text-text-muted">{packet.campaignSlug} · reviewer {packet.reviewerAlias}</p>
                  </div>
                  <div>
                    <span className="rounded border border-border bg-surface-2 px-2 py-1 text-xs font-semibold text-text-secondary">Pass {packet.pass}</span>
                  </div>
                  <p className="font-mono text-xs tabular-nums text-text-secondary">
                    {packet.completedCount} complete · {packet.assignedCount} open
                  </p>
                  <p className="text-xs text-text-muted">{formatTimestamp(packet.createdAt)}</p>
                  <div className="md:text-right">
                    <Link
                      href={`/corpus/review/${encodeURIComponent(packet.sessionId)}`}
                      className="inline-flex min-h-11 items-center rounded-md bg-accent px-3 py-2 text-sm font-semibold text-bg hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    >
                      Review
                    </Link>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}
