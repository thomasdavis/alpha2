/**
 * Health check and recovery for inconsistent state.
 *
 * - Finds orphaned temp files
 * - Finds pending batches (never committed)
 * - Cleans up stale state
 */
import { existsSync, readdirSync, unlinkSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { getPendingBatches, rollbackBatch, listRuns, getRun, updateRunStatus } from "./db.js";

export interface DoctorResult {
  orphanedTempFiles: string[];
  pendingBatches: string[];
  staleRuns: string[];
  actions: string[];
}

export async function doctor(fix: boolean = false): Promise<DoctorResult> {
  const baseDir = resolve(process.cwd(), ".histchat");
  const tmpDir = resolve(baseDir, ".tmp");
  const result: DoctorResult = {
    orphanedTempFiles: [],
    pendingBatches: [],
    staleRuns: [],
    actions: [],
  };

  // 1. Check for orphaned temp files
  if (existsSync(tmpDir)) {
    const tmpFiles = readdirSync(tmpDir);
    for (const f of tmpFiles) {
      if (f.endsWith(".tmp")) {
        result.orphanedTempFiles.push(f);
        if (fix) {
          unlinkSync(resolve(tmpDir, f));
          result.actions.push(`Deleted orphaned temp file: ${f}`);
        }
      }
    }
  }

  // 2. Check for pending batches
  const runs = await listRuns();
  for (const run of runs) {
    const pending = await getPendingBatches(run.id);
    for (const batch of pending) {
      result.pendingBatches.push(`${batch.id} (run ${run.id}, index ${batch.index})`);
      if (fix) {
        await rollbackBatch(batch.id);
        result.actions.push(`Rolled back pending batch: ${batch.id}`);
      }
    }

    // 3. Check for stale active runs (no updates in 1 hour)
    if (run.status === "active") {
      const updatedAt = new Date(run.updatedAt).getTime();
      const oneHourAgo = Date.now() - 3600_000;
      if (updatedAt < oneHourAgo) {
        result.staleRuns.push(run.id);
        if (fix) {
          await updateRunStatus(run.id, "paused");
          result.actions.push(`Paused stale run: ${run.id}`);
        }
      }
    }
  }

  return result;
}
