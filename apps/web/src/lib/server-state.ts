/**
 * Shared in-memory server state for the consolidated Next.js app.
 *
 * Module-scoped state persists across requests in standalone mode.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";

// ── Outputs directory ─────────────────────────────────────────────────────

export const OUTPUTS_DIR = process.env.OUTPUTS_DIR
  ? path.resolve(process.env.OUTPUTS_DIR)
  : path.resolve(process.cwd(), "outputs");

// ── Build info ────────────────────────────────────────────────────────────

export const BUILD_INFO = (() => {
  // Try build-info.json (baked in during Docker build)
  const candidates = [
    path.resolve(process.cwd(), "build-info.json"),
    path.resolve(process.cwd(), "apps/web/build-info.json"),
  ];
  for (const infoPath of candidates) {
    try {
      const info = JSON.parse(fs.readFileSync(infoPath, "utf8"));
      return { ...info, startedAt: new Date().toISOString() };
    } catch { /* not found */ }
  }
  // Try local git
  try {
    const sha = execSync("git rev-parse --short HEAD", { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] }).trim();
    const msg = execSync("git log -1 --format=%s", { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] }).trim();
    return { sha, message: msg, startedAt: new Date().toISOString() };
  } catch { /* no git */ }
  // Fallback to env
  const sha = process.env.RAILWAY_GIT_COMMIT_SHA?.slice(0, 7)
    ?? process.env.COMMIT_SHA
    ?? "unknown";
  return { sha, message: "", startedAt: new Date().toISOString() };
})();

// ── Models cache (30s TTL) ────────────────────────────────────────────────

export let modelsCache: { json: string; ts: number } | null = null;
export const MODELS_CACHE_TTL = 30_000;

export function invalidateModelsCache(): void { modelsCache = null; }
export function setModelsCache(json: string): void {
  modelsCache = { json, ts: Date.now() };
}

// ── SSE live training clients ─────────────────────────────────────────────

export const liveClients = new Set<ReadableStreamDefaultController>();

export function broadcastLive(event: string, data: unknown): void {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const controller of liveClients) {
    try { controller.enqueue(new TextEncoder().encode(payload)); } catch { liveClients.delete(controller); }
  }
}

// ── Chunked uploads ───────────────────────────────────────────────────────

export const pendingChunks = new Map<string, { chunks: Map<number, Buffer>; total: number; receivedAt: number }>();

// Clean up stale pending uploads every 5 minutes
if (typeof setInterval !== "undefined") {
  setInterval(() => {
    const now = Date.now();
    for (const [id, entry] of pendingChunks) {
      if (now - entry.receivedAt > 60 * 60 * 1000) pendingChunks.delete(id);
    }
  }, 5 * 60 * 1000);
}

// ── Auth helper ───────────────────────────────────────────────────────────

export function checkAuth(request: Request): Response | null {
  const secret = process.env.UPLOAD_SECRET;
  if (!secret) {
    return Response.json({ error: "UPLOAD_SECRET not configured" }, { status: 500 });
  }
  const auth = request.headers.get("authorization");
  if (auth !== `Bearer ${secret}`) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  return null;
}

// ── CORS helper ───────────────────────────────────────────────────────────

export function corsHeaders(): HeadersInit {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding",
  };
}

export function jsonResponse(data: unknown, status = 200): Response {
  return Response.json(data, { status, headers: corsHeaders() });
}
