/**
 * CRUD operations for the checkpoints table.
 */
import type { Client } from "@libsql/client";
import type { DbCheckpoint } from "./types.js";

export async function upsertCheckpoint(
  client: Client,
  input: {
    run_id: string;
    step: number;
    filename: string;
    file_path: string;
    file_size?: number | null;
  }
): Promise<void> {
  await client.execute({
    sql: `INSERT INTO checkpoints (run_id, step, filename, file_path, file_size)
          VALUES (?, ?, ?, ?, ?)
          ON CONFLICT(run_id, step) DO UPDATE SET
            filename = excluded.filename,
            file_path = excluded.file_path,
            file_size = excluded.file_size`,
    args: [
      input.run_id,
      input.step,
      input.filename,
      input.file_path,
      input.file_size ?? null,
    ],
  });
}

export async function listCheckpoints(
  client: Client,
  runId: string
): Promise<DbCheckpoint[]> {
  const result = await client.execute({
    sql: "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC",
    args: [runId],
  });
  return result.rows as unknown as DbCheckpoint[];
}

export async function getLatestCheckpoint(
  client: Client,
  runId: string
): Promise<DbCheckpoint | null> {
  const result = await client.execute({
    sql: "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC LIMIT 1",
    args: [runId],
  });
  return (result.rows[0] as unknown as DbCheckpoint) ?? null;
}
