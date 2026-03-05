/**
 * CRUD operations for the events table.
 */
import type { Client, InValue } from "@libsql/client";
import type { DbEvent, EventLevel } from "./types.js";

const CHUNK_SIZE = 500;
const DEFAULT_LIMIT = 500;
const MAX_LIMIT = 5000;
const MAX_MESSAGE_LENGTH = 4096;

export interface InsertEventInput {
  step?: number | null;
  level?: EventLevel;
  kind: string;
  message: string;
  payload?: unknown;
  createdAt?: string | null;
}

export interface GetEventsOptions {
  last?: number;
  fromId?: number;
  fromStep?: number;
  level?: EventLevel;
  kind?: string;
  limit?: number;
}

function clampLimit(limit?: number): number {
  if (!Number.isFinite(limit as number) || (limit as number) <= 0) return DEFAULT_LIMIT;
  return Math.max(1, Math.min(MAX_LIMIT, Math.floor(limit as number)));
}

function normalizeLevel(level?: EventLevel): EventLevel {
  if (level === "debug" || level === "info" || level === "warn" || level === "error") return level;
  return "info";
}

function normalizeMessage(message: string): string {
  const trimmed = String(message ?? "").trim();
  if (trimmed.length === 0) return "(no message)";
  if (trimmed.length <= MAX_MESSAGE_LENGTH) return trimmed;
  return `${trimmed.slice(0, MAX_MESSAGE_LENGTH - 3)}...`;
}

function serializePayload(payload: unknown): string | null {
  if (payload == null) return null;
  try {
    return JSON.stringify(payload);
  } catch {
    return JSON.stringify({ error: "payload_not_serializable" });
  }
}

function normalizeEventInput(input: InsertEventInput): Required<Omit<InsertEventInput, "step" | "createdAt" | "payload">> & {
  step: number | null;
  createdAt: string | null;
  payloadJson: string | null;
} {
  return {
    step: Number.isFinite(input.step as number) ? Math.floor(input.step as number) : null,
    level: normalizeLevel(input.level),
    kind: String(input.kind ?? "").trim() || "generic",
    message: normalizeMessage(input.message),
    createdAt: input.createdAt ? String(input.createdAt) : null,
    payloadJson: serializePayload(input.payload),
  };
}

export async function insertEvents(
  client: Client,
  runId: string,
  events: InsertEventInput[],
): Promise<number> {
  if (events.length === 0) return 0;

  let inserted = 0;
  for (let i = 0; i < events.length; i += CHUNK_SIZE) {
    const chunk = events.slice(i, i + CHUNK_SIZE);
    await client.batch(
      chunk.map((event) => {
        const normalized = normalizeEventInput(event);
        return {
          sql: `INSERT INTO events (run_id, step, level, kind, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')))`,
          args: [
            runId,
            normalized.step,
            normalized.level,
            normalized.kind,
            normalized.message,
            normalized.payloadJson,
            normalized.createdAt,
          ],
        };
      }),
      "write",
    );
    inserted += chunk.length;
  }

  return inserted;
}

export async function insertEvent(
  client: Client,
  runId: string,
  event: InsertEventInput,
): Promise<void> {
  await insertEvents(client, runId, [event]);
}

export async function getEvents(
  client: Client,
  runId: string,
  opts?: GetEventsOptions,
): Promise<DbEvent[]> {
  const conditions = ["run_id = ?"];
  const args: InValue[] = [runId];

  if (opts?.fromId != null) {
    conditions.push("id > ?");
    args.push(opts.fromId);
  }
  if (opts?.fromStep != null) {
    conditions.push("step >= ?");
    args.push(opts.fromStep);
  }
  if (opts?.level) {
    conditions.push("level = ?");
    args.push(opts.level);
  }
  if (opts?.kind) {
    conditions.push("kind = ?");
    args.push(opts.kind);
  }

  const where = conditions.join(" AND ");
  const last = Number.isFinite(opts?.last as number) ? Math.max(0, Math.floor(opts!.last as number)) : 0;
  const limit = clampLimit(last > 0 ? last : opts?.limit);

  const result = last > 0
    ? await client.execute({
      sql: `SELECT * FROM (
              SELECT * FROM events
              WHERE ${where}
              ORDER BY id DESC
              LIMIT ?
            ) sub
            ORDER BY id ASC`,
      args: [...args, limit],
    })
    : await client.execute({
      sql: `SELECT * FROM events
            WHERE ${where}
            ORDER BY id ASC
            LIMIT ?`,
      args: [...args, limit],
    });

  return result.rows as unknown as DbEvent[];
}
