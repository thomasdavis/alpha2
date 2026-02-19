/**
 * Database client lifecycle: createDb / getDb / closeDb.
 *
 * Reads TURSO_DATABASE_URL and TURSO_AUTH_TOKEN from environment by default.
 */
import { createClient, type Client } from "@libsql/client";
import { migrate } from "./migrate.js";
import { seedDomains } from "./domains.js";

export interface DbOptions {
  url?: string;
  authToken?: string;
}

let _client: Client | null = null;

export async function createDb(opts?: DbOptions): Promise<Client> {
  const url = opts?.url ?? process.env.TURSO_DATABASE_URL;
  const authToken = opts?.authToken ?? process.env.TURSO_AUTH_TOKEN;

  if (!url) {
    throw new Error(
      "No database URL — set TURSO_DATABASE_URL or pass opts.url"
    );
  }

  const isRemote = url.startsWith("libsql://") || url.startsWith("https://");

  const client = createClient({ url, authToken });

  // WAL + FK only apply to local SQLite files
  if (!isRemote) {
    await client.batch(
      [
        { sql: "PRAGMA journal_mode=WAL", args: [] },
        { sql: "PRAGMA foreign_keys=ON", args: [] },
      ],
      "write"
    );
  }

  await migrate(client);
  await seedDomains(client);

  _client = client;
  return client;
}

export function getDb(): Client {
  if (!_client) throw new Error("Database not initialized — call createDb() first");
  return _client;
}

export function closeDb(): void {
  if (_client) {
    _client.close();
    _client = null;
  }
}
