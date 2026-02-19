import { createClient, type Client } from "@libsql/client";

let _client: Client | null = null;

export function getClient(): Client {
  if (_client) return _client;
  const url = process.env.TURSO_DATABASE_URL;
  const authToken = process.env.TURSO_AUTH_TOKEN;
  if (!url) throw new Error("TURSO_DATABASE_URL not set");
  _client = createClient({ url, authToken });
  return _client;
}
