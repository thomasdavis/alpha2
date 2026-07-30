export default function CorpusLoading() {
  return (
    <div className="space-y-4" aria-label="Loading corpus ledger">
      <div className="h-8 w-52 animate-pulse rounded bg-surface-2" />
      <div className="grid min-h-[70vh] gap-4 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <div className="animate-pulse rounded-lg border border-border bg-surface" />
        <div className="space-y-3 rounded-lg border border-border bg-surface p-4">
          <div className="h-10 animate-pulse rounded bg-surface-2" />
          <div className="h-96 animate-pulse rounded bg-surface-2" />
        </div>
      </div>
      <span className="sr-only">Reading the public corpus ledger…</span>
    </div>
  );
}
