/**
 * CRUD operations for the samples table.
 */
import type { Client } from "@libsql/client";
import type { DbSample } from "./types.js";

export async function insertSamples(
  client: Client,
  runId: string,
  samples: Array<{ prompt: string; output: string }>
): Promise<number> {
  if (samples.length === 0) return 0;

  await client.batch(
    samples.map((s, idx) => ({
      sql: `INSERT OR IGNORE INTO samples (run_id, idx, prompt, output)
            VALUES (?, ?, ?, ?)`,
      args: [runId, idx, s.prompt, s.output],
    })),
    "write"
  );

  return samples.length;
}

export async function getSamples(
  client: Client,
  runId: string,
): Promise<DbSample[]> {
  const result = await client.execute({
    sql: "SELECT * FROM samples WHERE run_id = ? ORDER BY idx ASC",
    args: [runId],
  });
  return result.rows as unknown as DbSample[];
}
