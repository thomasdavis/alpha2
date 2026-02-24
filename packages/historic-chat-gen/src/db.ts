/**
 * SQLite state ledger for historic-chat-gen.
 *
 * Local file: .histchat/state.db
 * Uses @libsql/client for SQLite access.
 */
import { createClient, type Client } from "@libsql/client";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { genId, now } from "./util.js";
import type {
  Run, Batch, GeneratedConversation,
  RunStatus, BatchStatus, StatsResult,
} from "./types.js";

const SCHEMA_VERSION = 1;

const migrations: string[][] = [
  // Version 1: initial schema
  [
    `CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER NOT NULL
    )`,

    `CREATE TABLE IF NOT EXISTS runs (
      id               TEXT PRIMARY KEY,
      target_count     INTEGER NOT NULL,
      budget_limit     REAL NOT NULL,
      concurrency      INTEGER NOT NULL DEFAULT 5,
      batch_size       INTEGER NOT NULL DEFAULT 25,
      status           TEXT NOT NULL DEFAULT 'active'
                       CHECK(status IN ('active','completed','paused','failed')),
      completed_count  INTEGER NOT NULL DEFAULT 0,
      failed_count     INTEGER NOT NULL DEFAULT 0,
      total_input_tokens  INTEGER NOT NULL DEFAULT 0,
      total_output_tokens INTEGER NOT NULL DEFAULT 0,
      total_cost       REAL NOT NULL DEFAULT 0,
      seed             INTEGER NOT NULL,
      created_at       TEXT NOT NULL,
      updated_at       TEXT NOT NULL
    )`,

    `CREATE TABLE IF NOT EXISTS conversations (
      id            TEXT PRIMARY KEY,
      run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      batch_id      TEXT NOT NULL,
      figure_a      TEXT NOT NULL,
      figure_b      TEXT NOT NULL,
      topic         TEXT NOT NULL,
      tone          TEXT NOT NULL,
      turn_count    INTEGER NOT NULL,
      turns_json    TEXT NOT NULL,
      input_tokens  INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      cost          REAL NOT NULL,
      created_at    TEXT NOT NULL
    ) WITHOUT ROWID`,

    `CREATE TABLE IF NOT EXISTS batches (
      id                 TEXT PRIMARY KEY,
      run_id             TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      batch_index        INTEGER NOT NULL,
      status             TEXT NOT NULL DEFAULT 'pending'
                         CHECK(status IN ('pending','committed','rolled_back')),
      conversation_count INTEGER NOT NULL DEFAULT 0,
      created_at         TEXT NOT NULL
    )`,

    `CREATE INDEX IF NOT EXISTS idx_conversations_run ON conversations(run_id)`,
    `CREATE INDEX IF NOT EXISTS idx_conversations_batch ON conversations(batch_id)`,
    `CREATE INDEX IF NOT EXISTS idx_batches_run ON batches(run_id)`,
  ],
];

let _client: Client | null = null;

export async function createDb(dbDir?: string): Promise<Client> {
  const dir = dbDir ?? resolve(process.cwd(), ".histchat");
  mkdirSync(dir, { recursive: true });
  const dbPath = resolve(dir, "state.db");

  const client = createClient({ url: `file:${dbPath}` });

  await client.execute({ sql: "PRAGMA journal_mode=WAL", args: [] });
  await client.execute({ sql: "PRAGMA foreign_keys=ON", args: [] });

  await migrate(client);
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

async function migrate(client: Client): Promise<void> {
  // Ensure schema_version table exists
  await client.execute({
    sql: `CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)`,
    args: [],
  });

  const row = await client.execute({ sql: "SELECT version FROM schema_version LIMIT 1", args: [] });
  let currentVersion = row.rows.length > 0 ? (row.rows[0]!["version"] as number) : 0;

  for (let v = currentVersion; v < migrations.length; v++) {
    const stmts = migrations[v]!;
    await client.batch(
      stmts.map((sql) => ({ sql, args: [] })),
      "write"
    );
  }

  if (currentVersion === 0) {
    await client.execute({ sql: "INSERT INTO schema_version (version) VALUES (?)", args: [migrations.length] });
  } else if (currentVersion < migrations.length) {
    await client.execute({ sql: "UPDATE schema_version SET version = ?", args: [migrations.length] });
  }
}

// ── Run CRUD ──

export async function createRun(opts: {
  targetCount: number;
  budgetLimit: number;
  concurrency: number;
  batchSize: number;
  seed: number;
}): Promise<Run> {
  const id = genId();
  const ts = now();
  const db = getDb();
  await db.execute({
    sql: `INSERT INTO runs (id, target_count, budget_limit, concurrency, batch_size, seed, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [id, opts.targetCount, opts.budgetLimit, opts.concurrency, opts.batchSize, opts.seed, ts, ts],
  });
  return {
    id, targetCount: opts.targetCount, budgetLimit: opts.budgetLimit,
    concurrency: opts.concurrency, batchSize: opts.batchSize,
    status: "active", completedCount: 0, failedCount: 0,
    totalInputTokens: 0, totalOutputTokens: 0, totalCost: 0,
    seed: opts.seed, createdAt: ts, updatedAt: ts,
  };
}

export async function getRun(id: string): Promise<Run | null> {
  const db = getDb();
  const res = await db.execute({ sql: "SELECT * FROM runs WHERE id = ?", args: [id] });
  if (res.rows.length === 0) return null;
  return rowToRun(res.rows[0]!);
}

export async function getLatestRun(): Promise<Run | null> {
  const db = getDb();
  const res = await db.execute({ sql: "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1", args: [] });
  if (res.rows.length === 0) return null;
  return rowToRun(res.rows[0]!);
}

export async function getActiveRun(): Promise<Run | null> {
  const db = getDb();
  const res = await db.execute({
    sql: "SELECT * FROM runs WHERE status IN ('active', 'paused') ORDER BY created_at DESC LIMIT 1",
    args: [],
  });
  if (res.rows.length === 0) return null;
  return rowToRun(res.rows[0]!);
}

export async function updateRunStatus(id: string, status: RunStatus): Promise<void> {
  const db = getDb();
  await db.execute({ sql: "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?", args: [status, now(), id] });
}

export async function updateRunTotals(
  id: string,
  completedDelta: number,
  failedDelta: number,
  inputTokensDelta: number,
  outputTokensDelta: number,
  costDelta: number
): Promise<void> {
  const db = getDb();
  await db.execute({
    sql: `UPDATE runs SET
            completed_count = completed_count + ?,
            failed_count = failed_count + ?,
            total_input_tokens = total_input_tokens + ?,
            total_output_tokens = total_output_tokens + ?,
            total_cost = total_cost + ?,
            updated_at = ?
          WHERE id = ?`,
    args: [completedDelta, failedDelta, inputTokensDelta, outputTokensDelta, costDelta, now(), id],
  });
}

// ── Batch CRUD ──

export async function createBatch(runId: string, batchIndex: number): Promise<Batch> {
  const id = genId();
  const ts = now();
  const db = getDb();
  await db.execute({
    sql: `INSERT INTO batches (id, run_id, batch_index, status, created_at) VALUES (?, ?, ?, 'pending', ?)`,
    args: [id, runId, batchIndex, ts],
  });
  return { id, runId, index: batchIndex, status: "pending", conversationCount: 0, createdAt: ts };
}

export async function commitBatch(
  batchId: string,
  conversations: GeneratedConversation[]
): Promise<void> {
  const db = getDb();
  const stmts: Array<{ sql: string; args: unknown[] }> = [];

  for (const c of conversations) {
    stmts.push({
      sql: `INSERT OR IGNORE INTO conversations
            (id, run_id, batch_id, figure_a, figure_b, topic, tone, turn_count, turns_json, input_tokens, output_tokens, cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [c.id, c.runId, c.batchId, c.figureA, c.figureB, c.topic, c.tone, c.turnCount,
             JSON.stringify(c.turns), c.inputTokens, c.outputTokens, c.cost, c.createdAt],
    });
  }

  stmts.push({
    sql: `UPDATE batches SET status = 'committed', conversation_count = ? WHERE id = ?`,
    args: [conversations.length, batchId],
  });

  const totalInput = conversations.reduce((s, c) => s + c.inputTokens, 0);
  const totalOutput = conversations.reduce((s, c) => s + c.outputTokens, 0);
  const totalCost = conversations.reduce((s, c) => s + c.cost, 0);

  stmts.push({
    sql: `UPDATE runs SET
            completed_count = completed_count + ?,
            total_input_tokens = total_input_tokens + ?,
            total_output_tokens = total_output_tokens + ?,
            total_cost = total_cost + ?,
            updated_at = ?
          WHERE id = (SELECT run_id FROM batches WHERE id = ?)`,
    args: [conversations.length, totalInput, totalOutput, totalCost, now(), batchId],
  });

  await db.batch(stmts as any, "write");
}

export async function rollbackBatch(batchId: string): Promise<void> {
  const db = getDb();
  await db.batch([
    { sql: "DELETE FROM conversations WHERE batch_id = ?", args: [batchId] },
    { sql: "UPDATE batches SET status = 'rolled_back' WHERE id = ?", args: [batchId] },
  ] as any, "write");
}

export async function getPendingBatches(runId: string): Promise<Batch[]> {
  const db = getDb();
  const res = await db.execute({
    sql: "SELECT * FROM batches WHERE run_id = ? AND status = 'pending' ORDER BY batch_index",
    args: [runId],
  });
  return res.rows.map(rowToBatch);
}

// ── Conversation queries ──

export async function getCommittedConversations(runId: string): Promise<GeneratedConversation[]> {
  const db = getDb();
  const res = await db.execute({
    sql: `SELECT c.* FROM conversations c
          JOIN batches b ON c.batch_id = b.id
          WHERE c.run_id = ? AND b.status = 'committed'
          ORDER BY c.created_at`,
    args: [runId],
  });
  return res.rows.map(rowToConversation);
}

export async function getCompletedAssignmentKeys(runId: string): Promise<Set<string>> {
  const db = getDb();
  const res = await db.execute({
    sql: `SELECT figure_a, figure_b, topic, tone FROM conversations
          WHERE run_id = ? AND batch_id IN (SELECT id FROM batches WHERE status = 'committed')`,
    args: [runId],
  });
  const keys = new Set<string>();
  for (const row of res.rows) {
    keys.add(`${row["figure_a"]}|${row["figure_b"]}|${row["topic"]}|${row["tone"]}`);
  }
  return keys;
}

// ── Stats ──

export async function getRunStats(runId: string): Promise<StatsResult | null> {
  const run = await getRun(runId);
  if (!run) return null;

  const db = getDb();
  const figRes = await db.execute({
    sql: `SELECT figure_a AS fig, COUNT(*) AS cnt FROM conversations WHERE run_id = ? GROUP BY figure_a
          UNION ALL
          SELECT figure_b AS fig, COUNT(*) AS cnt FROM conversations WHERE run_id = ? GROUP BY figure_b`,
    args: [runId, runId],
  });
  const figDist: Record<string, number> = {};
  for (const row of figRes.rows) {
    const fig = row["fig"] as string;
    figDist[fig] = (figDist[fig] ?? 0) + (row["cnt"] as number);
  }

  const topicRes = await db.execute({
    sql: "SELECT topic, COUNT(*) AS cnt FROM conversations WHERE run_id = ? GROUP BY topic",
    args: [runId],
  });
  const topicDist: Record<string, number> = {};
  for (const row of topicRes.rows) {
    topicDist[row["topic"] as string] = row["cnt"] as number;
  }

  const avgCost = run.completedCount > 0 ? run.totalCost / run.completedCount : 0;

  return {
    runId: run.id,
    status: run.status,
    completed: run.completedCount,
    failed: run.failedCount,
    target: run.targetCount,
    totalCost: run.totalCost,
    budgetLimit: run.budgetLimit,
    budgetRemaining: run.budgetLimit - run.totalCost,
    avgCostPerConversation: avgCost,
    inputTokens: run.totalInputTokens,
    outputTokens: run.totalOutputTokens,
    figureDistribution: figDist,
    topicDistribution: topicDist,
  };
}

// ── Row mappers ──

function rowToRun(row: Record<string, unknown>): Run {
  return {
    id: row["id"] as string,
    targetCount: row["target_count"] as number,
    budgetLimit: row["budget_limit"] as number,
    concurrency: row["concurrency"] as number,
    batchSize: row["batch_size"] as number,
    status: row["status"] as RunStatus,
    completedCount: row["completed_count"] as number,
    failedCount: row["failed_count"] as number,
    totalInputTokens: row["total_input_tokens"] as number,
    totalOutputTokens: row["total_output_tokens"] as number,
    totalCost: row["total_cost"] as number,
    seed: row["seed"] as number,
    createdAt: row["created_at"] as string,
    updatedAt: row["updated_at"] as string,
  };
}

function rowToBatch(row: Record<string, unknown>): Batch {
  return {
    id: row["id"] as string,
    runId: row["run_id"] as string,
    index: row["batch_index"] as number,
    status: row["status"] as BatchStatus,
    conversationCount: row["conversation_count"] as number,
    createdAt: row["created_at"] as string,
  };
}

function rowToConversation(row: Record<string, unknown>): GeneratedConversation {
  return {
    id: row["id"] as string,
    runId: row["run_id"] as string,
    batchId: row["batch_id"] as string,
    figureA: row["figure_a"] as string,
    figureB: row["figure_b"] as string,
    topic: row["topic"] as string,
    tone: row["tone"] as string as any,
    turnCount: row["turn_count"] as number,
    turns: JSON.parse(row["turns_json"] as string),
    inputTokens: row["input_tokens"] as number,
    outputTokens: row["output_tokens"] as number,
    cost: row["cost"] as number,
    createdAt: row["created_at"] as string,
  };
}

// ── All runs ──

export async function listRuns(): Promise<Run[]> {
  const db = getDb();
  const res = await db.execute({ sql: "SELECT * FROM runs ORDER BY created_at DESC", args: [] });
  return res.rows.map(rowToRun);
}
