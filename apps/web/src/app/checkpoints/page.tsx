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
        <div className="grid grid-cols-[1fr_80px_100px_80px] gap-4 border-b border-border px-4 py-2.5 text-[0.68rem] font-semibold uppercase tracking-wider text-text-muted sm:grid-cols-[1fr_1fr_100px_100px_100px]">
          <span>Run</span>
          <span className="hidden sm:block">Filename</span>
          <span>Step</span>
          <span>Size</span>
          <span>Created</span>
        </div>

        {allCheckpoints.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-text-muted">
            No checkpoints found.
          </div>
        ) : (
          allCheckpoints.map((c, i) => (
            <div
              key={`${c.runId}-${c.step}`}
              className="grid grid-cols-[1fr_80px_100px_80px] gap-4 border-b border-border px-4 py-2.5 text-xs last:border-0 sm:grid-cols-[1fr_1fr_100px_100px_100px]"
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
            </div>
          ))
        )}
      </div>
    </>
  );
}
