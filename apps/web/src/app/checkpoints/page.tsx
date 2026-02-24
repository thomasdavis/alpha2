import Link from "next/link";
import { getClient } from "@/lib/db";
import { listRuns, listCheckpoints } from "@alpha/db";
import { formatBytes, timeAgo } from "@/lib/format";

export const dynamic = "force-dynamic";

interface CheckpointRow {
  runId: string;
  step: number;
  filename: string;
  file_size: number | null;
  created_at: string | null;
}

export default async function CheckpointsPage() {
  const client = await getClient();
  const runs = await listRuns(client);

  // Gather all checkpoints across runs
  const allCheckpoints: CheckpointRow[] = [];
  for (const run of runs) {
    const ckpts = await listCheckpoints(client, run.id);
    for (const c of ckpts) {
      allCheckpoints.push({
        runId: run.id,
        step: c.step as number,
        filename: c.filename as string,
        file_size: c.file_size as number | null,
        created_at: c.created_at as string | null,
      });
    }
  }

  // Sort by created_at descending
  allCheckpoints.sort((a, b) => {
    if (!a.created_at || !b.created_at) return 0;
    return b.created_at.localeCompare(a.created_at);
  });

  const totalSize = allCheckpoints.reduce((s, c) => s + (c.file_size ?? 0), 0);

  return (
    <>
      <div className="mb-6">
        <h1 className="text-lg font-bold text-white">Checkpoints</h1>
        <p className="mt-0.5 text-xs text-text-muted">
          {allCheckpoints.length} checkpoints across {runs.length} runs &middot; {formatBytes(totalSize)} total
        </p>
      </div>

      <div className="rounded-lg border border-border bg-surface">
        {/* Table header */}
        <div className="grid grid-cols-[1fr_80px_100px_80px_44px] gap-4 border-b border-border px-4 py-2.5 text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted sm:grid-cols-[1fr_1fr_100px_100px_100px_52px]">
          <span>Run</span>
          <span className="hidden sm:block">Filename</span>
          <span>Step</span>
          <span>Size</span>
          <span>Created</span>
          <span></span>
        </div>

        {allCheckpoints.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-text-muted">
            No checkpoints found.
          </div>
        ) : (
          allCheckpoints.map((c, i) => (
            <div
              key={`${c.runId}-${c.step}`}
              className="grid grid-cols-[1fr_80px_100px_80px_44px] gap-4 border-b border-border px-4 py-2.5 text-xs last:border-0 sm:grid-cols-[1fr_1fr_100px_100px_100px_52px]"
            >
              <Link
                href={`/runs/${encodeURIComponent(c.runId)}`}
                className="truncate font-medium text-white hover:text-accent"
              >
                {c.runId}
              </Link>
              <span className="hidden truncate font-mono text-text-secondary sm:block">
                {c.filename}
              </span>
              <span className="font-mono text-text-secondary">
                {c.step.toLocaleString()}
              </span>
              <span className="text-text-muted">
                {formatBytes(c.file_size)}
              </span>
              <span className="text-text-muted">
                {c.created_at ? timeAgo(c.created_at) : "-"}
              </span>
              <a
                href={`/api/runs/${encodeURIComponent(c.runId)}/download/${encodeURIComponent(c.filename)}`}
                className="text-text-muted transition-colors hover:text-accent"
                title="Download checkpoint"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                  <path d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z" />
                  <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                </svg>
              </a>
            </div>
          ))
        )}
      </div>
    </>
  );
}
