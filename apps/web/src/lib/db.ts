import { createClient, type Client } from "@libsql/client";
import { createDb, getDb } from "@alpha/db";

let _client: Client | null = null;
let _dbReady: Promise<void> | null = null;

/** Ensure @alpha/db is initialized (migrations + domain seeds). */
function ensureDb(): Promise<void> {
  if (!_dbReady) {
    _dbReady = createDb().then(() => {}).catch((e) => {
      console.warn("createDb failed:", (e as Error).message);
      _dbReady = null;
    });
  }
  return _dbReady ?? Promise.resolve();
}

/**
 * Get a DB client. Used by both SSR pages and API route handlers.
 * Ensures @alpha/db is initialized on first call.
 */
export async function getClient(): Promise<Client> {
  await ensureDb();
  try {
    return getDb();
  } catch {
    // Fallback: create a raw client if createDb didn't set the singleton
    if (_client) return _client;
    const url = process.env.TURSO_DATABASE_URL;
    const authToken = process.env.TURSO_AUTH_TOKEN;
    if (!url) throw new Error("TURSO_DATABASE_URL not set");
    _client = createClient({ url, authToken });
    return _client;
  }
}
