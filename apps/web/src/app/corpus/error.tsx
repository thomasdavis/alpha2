"use client";

export default function CorpusError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section className="rounded-lg border border-red/30 bg-red-bg/30 px-6 py-12">
      <h1 className="text-lg font-semibold text-text-primary">The corpus ledger could not be read</h1>
      <p className="mt-2 max-w-2xl text-sm text-text-secondary">
        The explorer is read-only and will not repair or rewrite the database. Try the read again; if it still
        fails, the operator should verify the configured ledger path and SQLite integrity.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-5 min-h-11 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        Read the ledger again
      </button>
    </section>
  );
}
