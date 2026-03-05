/**
 * Command: alpha events
 *
 * Tail run event logs from the remote metrics API.
 */
import { boolArg, intArg, parseKV, requireArg, strArg } from "../parse.js";

interface RunEvent {
  id: number;
  run_id: string;
  step: number | null;
  level: "debug" | "info" | "warn" | "error";
  kind: string;
  message: string;
  payload_json?: string | null;
  payload?: unknown;
  created_at: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeBaseUrl(raw: string): string {
  return raw.replace(/\/+$/, "");
}

function parsePayload(event: RunEvent): unknown {
  if (event.payload !== undefined) return event.payload;
  if (!event.payload_json) return null;
  try {
    return JSON.parse(event.payload_json);
  } catch {
    return event.payload_json;
  }
}

function formatPayload(payload: unknown): string {
  if (payload == null) return "";
  const text = JSON.stringify(payload);
  if (text.length <= 220) return text;
  return `${text.slice(0, 217)}...`;
}

function formatEventLine(event: RunEvent): string {
  const ts = event.created_at ?? new Date().toISOString();
  const step = Number.isFinite(event.step as number) ? `step=${event.step}` : "step=-";
  const payloadStr = formatPayload(parsePayload(event));
  const payloadSuffix = payloadStr ? ` payload=${payloadStr}` : "";
  return `${ts} id=${event.id} ${step} ${event.level.toUpperCase()} ${event.kind}: ${event.message}${payloadSuffix}`;
}

export async function eventsCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const runId = requireArg(kv, "run", "run id");
  const baseUrl = normalizeBaseUrl(strArg(kv, "url", process.env.ALPHA_REMOTE_URL ?? "http://127.0.0.1:3001"));
  const pollSec = Math.max(0, intArg(kv, "poll", 2));
  const pollMs = pollSec * 1000;
  const once = boolArg(kv, "once", false) || pollSec === 0;
  const asJson = boolArg(kv, "json", false);
  const level = kv["level"];
  const kind = kv["kind"];
  const limit = Math.max(1, intArg(kv, "limit", 200));
  const last = Math.max(1, intArg(kv, "last", 50));
  let fromId = kv["fromId"] ? Math.max(0, Number.parseInt(kv["fromId"], 10) || 0) : 0;
  let firstFetch = true;

  const authHeader = process.env.ALPHA_REMOTE_SECRET
    ? `Bearer ${process.env.ALPHA_REMOTE_SECRET}`
    : null;

  const fetchEvents = async (): Promise<RunEvent[]> => {
    const params = new URLSearchParams();
    if (firstFetch) {
      params.set("last", String(last));
    } else {
      params.set("fromId", String(fromId));
      params.set("limit", String(limit));
    }
    if (level) params.set("level", level);
    if (kind) params.set("kind", kind);
    const url = `${baseUrl}/api/runs/${encodeURIComponent(runId)}/events?${params.toString()}`;
    const headers = new Headers({ Accept: "application/json" });
    if (authHeader) headers.set("Authorization", authHeader);
    const res = await fetch(url, {
      headers,
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) {
      throw new Error(`GET ${url} failed: ${res.status} ${await res.text()}`);
    }
    return (await res.json()) as RunEvent[];
  };

  let keepRunning = true;
  process.on("SIGINT", () => { keepRunning = false; });

  while (keepRunning) {
    try {
      const events = await fetchEvents();
      if (events.length > 0) {
        for (const event of events) {
          if (asJson) {
            console.log(JSON.stringify(event));
          } else {
            console.log(formatEventLine(event));
          }
          fromId = Math.max(fromId, event.id ?? 0);
        }
      }
      firstFetch = false;
    } catch (e) {
      console.error(`[events] ${(e as Error).message}`);
    }

    if (once) break;
    await sleep(pollMs);
  }
}
