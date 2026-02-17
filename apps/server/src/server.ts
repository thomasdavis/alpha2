/**
 * Local dev server for inference + chat.
 *
 * Endpoints match Vercel routing:
 *   GET  /              → Inference UI
 *   GET  /chat          → Chat UI
 *   GET  /api/models    → JSON list of available runs
 *   GET  /api/inference → SSE stream of generated tokens
 *   POST /api/chat      → AI SDK streaming chat endpoint
 *
 * In dev, also serves /models and /inference as aliases.
 */
import * as http from "node:http";
import * as fs from "node:fs";
import * as path from "node:path";
import { streamText } from "ai";
import {
  initEngine, getRuns, ensureModel, generateTokens, AlphaLanguageModel,
} from "./lib/engine.js";

const PORT = parseInt(process.env.PORT ?? "3000", 10);
const OUTPUTS_DIR = path.resolve(import.meta.dirname, "../../../outputs");

// ── HTTP helpers ──────────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}

function serveStatic(res: http.ServerResponse, filePath: string, contentType: string): void {
  const publicDir = path.resolve(import.meta.dirname, "../../../public");
  const full = path.join(publicDir, filePath);
  if (fs.existsSync(full)) {
    res.writeHead(200, { "Content-Type": contentType });
    res.end(fs.readFileSync(full));
  } else {
    res.writeHead(404);
    res.end("Not found");
  }
}

// ── Handlers ──────────────────────────────────────────────────────────────

function handleModels(_req: http.IncomingMessage, res: http.ServerResponse): void {
  res.writeHead(200, { "Content-Type": "application/json" });
  const payload = getRuns().map((r) => ({
    id: r.id, name: r.name, step: r.step, mtime: r.mtime, lastLoss: r.lastLoss,
    modelConfig: r.config.modelConfig, trainConfig: r.config.trainConfig,
  }));
  res.end(JSON.stringify(payload));
}

async function handleInference(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const runs = getRuns();
  const query = url.searchParams.get("query") ?? "";
  const modelId = url.searchParams.get("model") ?? runs[0]?.id;
  const steps = Math.min(parseInt(url.searchParams.get("steps") ?? "200", 10), 500);
  const temperature = parseFloat(url.searchParams.get("temp") ?? "0.8");
  const topk = parseInt(url.searchParams.get("topk") ?? "40", 10);

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unknown model" }));
    return;
  }

  const model = await ensureModel(modelId);
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });

  let aborted = false;
  req.on("close", () => { aborted = true; });

  const gen = generateTokens(model, query, steps, temperature, topk);
  function nextToken(): void {
    if (aborted) return;
    const result = gen.next();
    if (result.done) { res.write("data: [DONE]\n\n"); res.end(); return; }
    res.write(`data: ${JSON.stringify({ token: result.value })}\n\n`);
    setImmediate(nextToken);
  }
  nextToken();
}

async function handleChat(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const body = JSON.parse(await readBody(req));
  const runs = getRuns();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const modelId: string = body.model ?? runs[0]?.id;
  const maxTokens: number = Math.min(body.maxTokens ?? 200, 500);
  const temperature: number = body.temperature ?? 0.8;
  const topk: number = body.topk ?? 40;

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unknown model" }));
    return;
  }

  await ensureModel(modelId);
  const model = new AlphaLanguageModel(modelId, { steps: maxTokens, temperature, topk });

  const result = streamText({
    model,
    messages: messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content })),
    temperature,
    maxOutputTokens: maxTokens,
    topK: topk,
  });

  result.pipeTextStreamToResponse(res);
}

// ── Router ────────────────────────────────────────────────────────────────

async function route(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const p = url.pathname;

  // API routes
  if (p === "/api/models" || p === "/models") { handleModels(req, res); return; }
  if (p === "/api/inference" || p === "/inference") { await handleInference(req, res); return; }
  if ((p === "/api/chat") && req.method === "POST") { await handleChat(req, res); return; }

  // Static pages
  if (p === "/") { serveStatic(res, "index.html", "text/html"); return; }
  if (p === "/chat") { serveStatic(res, "chat.html", "text/html"); return; }

  res.writeHead(404);
  res.end("Not found");
}

const server = http.createServer((req, res) => {
  route(req, res).catch((e) => {
    console.error("Error:", e);
    if (!res.headersSent) { res.writeHead(500); res.end("Internal error"); }
  });
});

// ── Start ─────────────────────────────────────────────────────────────────

await initEngine(OUTPUTS_DIR);

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
  console.log(`  Inference UI: http://localhost:${PORT}/`);
  console.log(`  Chat UI:      http://localhost:${PORT}/chat`);
});
