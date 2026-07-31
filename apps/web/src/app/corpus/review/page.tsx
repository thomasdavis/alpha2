import Link from "next/link";
import { getCorpusReader } from "@/lib/corpus";

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

export default function CorpusReviewPage() {
  const reader = getCorpusReader();
  const packets = reader.listReviewPackets();

  return (
    <main className="mx-auto w-full max-w-6xl space-y-8">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">Human review workspace</h1>
            <span className="rounded border border-blue/30 bg-blue-bg px-2 py-1 text-[0.68rem] font-semibold text-blue">
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
                      className="inline-flex min-h-11 items-center rounded-md bg-accent px-3 py-2 text-sm font-semibold text-white hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
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
    </main>
  );
}
