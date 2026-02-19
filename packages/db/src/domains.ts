/**
 * Domain seeding and queries.
 *
 * Seeds the domains table from @alpha/core's domain definitions.
 */
import type { Client } from "@libsql/client";
import { domains as coreDomains } from "@alpha/core";

export async function seedDomains(client: Client): Promise<void> {
  const stmts = Array.from(coreDomains.values()).map((d) => ({
    sql: `INSERT OR REPLACE INTO domains
          (id, display_name, tokenizer, sample_prompts, model_defaults, train_defaults)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [
      d.id,
      d.displayName,
      d.tokenizer,
      JSON.stringify(d.samplePrompts),
      JSON.stringify(d.modelDefaults),
      JSON.stringify(d.trainDefaults),
    ],
  }));

  if (stmts.length > 0) {
    await client.batch(stmts, "write");
  }
}

export async function listDomains(
  client: Client
): Promise<
  Array<{
    id: string;
    display_name: string;
    tokenizer: string;
    sample_prompts: string[];
    model_defaults: Record<string, unknown>;
    train_defaults: Record<string, unknown>;
  }>
> {
  const result = await client.execute("SELECT * FROM domains ORDER BY id");
  return result.rows.map((row: any) => ({
    id: row.id as string,
    display_name: row.display_name as string,
    tokenizer: row.tokenizer as string,
    sample_prompts: JSON.parse(row.sample_prompts as string),
    model_defaults: JSON.parse(row.model_defaults as string),
    train_defaults: JSON.parse(row.train_defaults as string),
  }));
}
