/**
 * CRUD operations for the samples table.
 */
import type { Client } from "@libsql/client";
import type { DbSample } from "./types.js";

export async function insertSamples(
  client: Client,
  runId: string,
  samples: Array<{ prompt: string; output: string }>,
  step?: number | null,
): Promise<number> {
  if (samples.length === 0) return 0;
  const sampleStep = Number.isFinite(step as number) ? Math.max(0, Math.floor(step as number)) : null;

  await client.batch(
    samples.map((s, idx) => ({
      sql: `INSERT INTO samples (run_id, idx, step, prompt, output)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, idx) DO UPDATE SET
              step = excluded.step,
              prompt = excluded.prompt,
              output = excluded.output,
              created_at = datetime('now')`,
      args: [runId, idx, sampleStep, s.prompt, s.output],
    })),
    "write"
  );

  return samples.length;
}

export async function getSamples(
  client: Client,
  runId: string,
  options?: { step?: number | null },
): Promise<DbSample[]> {
  const explicitStep = options?.step;
  if (Number.isFinite(explicitStep as number)) {
    const step = Math.max(0, Math.floor(explicitStep as number));
    const byStep = await client.execute({
      sql: "SELECT * FROM samples WHERE run_id = ? AND step = ? ORDER BY idx ASC",
      args: [runId, step],
    });
    return byStep.rows as unknown as DbSample[];
  }

  const latestStep = await client.execute({
    sql: "SELECT step FROM samples WHERE run_id = ? AND step IS NOT NULL ORDER BY step DESC LIMIT 1",
    args: [runId],
  });
  const stepValue = latestStep.rows[0]?.step;
  let result;
  if (typeof stepValue === "number") {
    result = await client.execute({
      sql: "SELECT * FROM samples WHERE run_id = ? AND step = ? ORDER BY idx ASC",
      args: [runId, stepValue],
    });
  } else {
    // Legacy rows without step metadata.
    result = await client.execute({
      sql: "SELECT * FROM samples WHERE run_id = ? ORDER BY idx ASC",
      args: [runId],
    });
  }
  return result.rows as unknown as DbSample[];
}
