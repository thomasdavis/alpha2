import { NextRequest } from "next/server";
import { getClient } from "@/lib/db";
import { getEvents } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

function parseIntParam(value: string | null): number | undefined {
  if (!value) return undefined;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const query = request.nextUrl.searchParams;

  const last = parseIntParam(query.get("last"));
  const fromId = parseIntParam(query.get("fromId"));
  const fromStep = parseIntParam(query.get("fromStep"));
  const limit = parseIntParam(query.get("limit"));
  const levelRaw = query.get("level");
  const kindRaw = query.get("kind");
  const level = levelRaw === "debug" || levelRaw === "info" || levelRaw === "warn" || levelRaw === "error"
    ? levelRaw
    : undefined;
  const kind = kindRaw && kindRaw.trim().length > 0 ? kindRaw.trim() : undefined;

  const client = await getClient();
  const events = await getEvents(client, id, {
    last,
    fromId,
    fromStep,
    limit,
    level,
    kind,
  });

  return jsonResponse(events.map((event) => ({
    ...event,
    payload: event.payload_json ? safeJsonParse(event.payload_json) : null,
  })));
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}
