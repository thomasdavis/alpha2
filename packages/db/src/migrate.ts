/**
 * Version-based migration runner.
 *
 * Tracks applied versions in a schema_version table and applies
 * pending migrations via client.batch().
 */
import type { Client } from "@libsql/client";
import { migrations } from "./schema.js";

export async function migrate(client: Client): Promise<number> {
  // Ensure schema_version table exists
  await client.execute(
    `CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER PRIMARY KEY,
      applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    )`
  );

  // Get current version
  const result = await client.execute(
    "SELECT COALESCE(MAX(version), 0) AS v FROM schema_version"
  );
  const current = result.rows[0].v as number;

  let applied = 0;
  for (let i = current; i < migrations.length; i++) {
    const version = i + 1;
    const stmts = migrations[i];

    await client.batch(
      [
        ...stmts.map((sql) => ({ sql, args: [] })),
        {
          sql: "INSERT INTO schema_version (version) VALUES (?)",
          args: [version],
        },
      ],
      "write"
    );
    applied++;
  }

  return applied;
}
