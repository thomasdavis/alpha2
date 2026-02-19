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
import * as zlib from "node:zlib";
import * as crypto from "node:crypto";
import { streamText } from "ai";
import {
  initEngine, getRuns, ensureModel, generateTokens, sampleNextToken, AlphaLanguageModel, resetEngine,
} from "./lib/engine.js";
import { SeededRng } from "@alpha/core";
import {
  createDb, getDb, syncFromDisk, listRuns as dbListRuns,
  getRecentMetrics, getMetrics, listCheckpoints, listDomains,
  type DbRunSummary,
} from "@alpha/db";

const PORT = parseInt(process.env.PORT ?? "3000", 10);
const OUTPUTS_DIR = process.env.OUTPUTS_DIR
  ? path.resolve(process.env.OUTPUTS_DIR)
  : path.resolve(import.meta.dirname, "../../../outputs");

// ── HTTP helpers ──────────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks);
      if (req.headers["content-encoding"] === "gzip") {
        zlib.gunzip(raw, (err, result) => {
          if (err) reject(err);
          else resolve(result.toString());
        });
      } else {
        resolve(raw.toString());
      }
    });
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
    domain: r.domain,
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

// ── Generate handler (contract endpoint) ─────────────────────────────────

async function handleGenerate(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const body = req.method === "POST" ? JSON.parse(await readBody(req)) : {};
  const runs = getRuns();
  const prompt: string = url.searchParams.get("prompt") ?? body.prompt ?? "";
  const maxTokens: number = Math.min(parseInt(url.searchParams.get("max_tokens") ?? "", 10) || (body.max_tokens ?? 2048), 2048);
  const temperature: number = parseFloat(url.searchParams.get("temperature") ?? "") || (body.temperature ?? 0.7);
  const modelId: string = url.searchParams.get("model") ?? body.model ?? runs[0]?.id;

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unknown model" }));
    return;
  }

  const model = await ensureModel(modelId);
  const { config, tokenizer } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);

  const promptTokens = tokenizer.encode(prompt);
  const maxLen = Math.min(promptTokens.length + maxTokens, config.blockSize);
  const tokens = new Int32Array(maxLen);
  tokens.set(promptTokens);
  let currentLen = promptTokens.length;
  let completionCount = 0;

  for (let i = 0; i < maxTokens && currentLen < config.blockSize; i++) {
    const next = sampleNextToken(model, tokens, currentLen, temperature, 40, rng);
    tokens[currentLen] = next;
    currentLen++;
    completionCount++;
  }

  const text = tokenizer.decode(tokens.slice(promptTokens.length, currentLen));

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    text,
    model: modelId,
    usage: {
      prompt_tokens: promptTokens.length,
      completion_tokens: completionCount,
    },
  }));
}

// ── OpenAI-compatible endpoints ───────────────────────────────────────────

function handleOpenAIModels(_req: http.IncomingMessage, res: http.ServerResponse): void {
  const runs = getRuns();
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    object: "list",
    data: runs.map((r) => ({
      id: r.id,
      object: "model",
      created: Math.floor(r.mtime / 1000),
      owned_by: "alpha",
    })),
  }));
}

function messagesToPrompt(messages: Array<{ role: string; content: string }>): string {
  return messages.map((m) => m.content).join("\n");
}

async function handleChatCompletions(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const body = JSON.parse(await readBody(req));
  const runs = getRuns();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const modelId: string = body.model ?? runs[0]?.id;
  const maxTokens: number = Math.min(body.max_tokens ?? body.max_completion_tokens ?? 2048, 2048);
  const temperature: number = body.temperature ?? 0.7;
  const stream: boolean = body.stream === true;

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: { message: "Unknown model: " + modelId, type: "invalid_request_error" } }));
    return;
  }

  const model = await ensureModel(modelId);
  const { config, tokenizer } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const prompt = messagesToPrompt(messages);
  const promptTokens = tokenizer.encode(prompt);
  const maxLen = Math.min(promptTokens.length + maxTokens, config.blockSize);
  const tokens = new Int32Array(maxLen);
  tokens.set(promptTokens);
  let currentLen = promptTokens.length;
  const completionId = "chatcmpl-" + crypto.randomBytes(12).toString("hex");
  const created = Math.floor(Date.now() / 1000);

  if (stream) {
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });

    let aborted = false;
    req.on("close", () => { aborted = true; });
    let completionCount = 0;

    // Role chunk
    res.write(`data: ${JSON.stringify({
      id: completionId, object: "chat.completion.chunk", created, model: modelId,
      choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
    })}\n\n`);

    function nextChunk(): void {
      if (aborted) return;
      if (completionCount >= maxTokens || currentLen >= config.blockSize) {
        res.write(`data: ${JSON.stringify({
          id: completionId, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: completionCount >= maxTokens ? "length" : "stop" }],
          usage: { prompt_tokens: promptTokens.length, completion_tokens: completionCount, total_tokens: promptTokens.length + completionCount },
        })}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
        return;
      }

      const next = sampleNextToken(model, tokens, currentLen, temperature, 40, rng);
      tokens[currentLen] = next;
      currentLen++;
      completionCount++;
      const raw = tokenizer.decode(new Int32Array([next]));
      const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";

      res.write(`data: ${JSON.stringify({
        id: completionId, object: "chat.completion.chunk", created, model: modelId,
        choices: [{ index: 0, delta: { content: sep + raw }, finish_reason: null }],
      })}\n\n`);
      setImmediate(nextChunk);
    }
    nextChunk();
  } else {
    let completionCount = 0;
    for (let i = 0; i < maxTokens && currentLen < config.blockSize; i++) {
      const next = sampleNextToken(model, tokens, currentLen, temperature, 40, rng);
      tokens[currentLen] = next;
      currentLen++;
      completionCount++;
    }

    const text = tokenizer.decode(tokens.slice(promptTokens.length, currentLen));

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      id: completionId,
      object: "chat.completion",
      created,
      model: modelId,
      choices: [{
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: completionCount >= maxTokens ? "length" : "stop",
      }],
      usage: {
        prompt_tokens: promptTokens.length,
        completion_tokens: completionCount,
        total_tokens: promptTokens.length + completionCount,
      },
    }));
  }
}

// ── Upload handler ───────────────────────────────────────────────────────

async function handleUpload(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const secret = process.env.UPLOAD_SECRET;
  if (!secret) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "UPLOAD_SECRET not configured" }));
    return;
  }

  const auth = req.headers.authorization;
  if (auth !== `Bearer ${secret}`) {
    res.writeHead(401, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Unauthorized" }));
    return;
  }

  const body = JSON.parse(await readBody(req));
  const { name, config, checkpoint, step, metrics } = body as {
    name: string;
    config: Record<string, unknown>;
    checkpoint: unknown;
    step: number;
    metrics?: string;
  };

  if (!name || !config || !checkpoint || !step) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Missing required fields: name, config, checkpoint, step" }));
    return;
  }

  const runDir = path.join(OUTPUTS_DIR, name);
  fs.mkdirSync(runDir, { recursive: true });

  fs.writeFileSync(path.join(runDir, "config.json"), JSON.stringify(config, null, 2));
  fs.writeFileSync(path.join(runDir, `checkpoint-${step}.json`), JSON.stringify(checkpoint));
  if (metrics) {
    fs.writeFileSync(path.join(runDir, "metrics.jsonl"), metrics);
  }

  // Rescan engine to pick up new model
  resetEngine();
  await initEngine(OUTPUTS_DIR);

  console.log(`Uploaded model: ${name} (step ${step})`);
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ ok: true, name, step }));
}

// ── Dashboard API handlers ────────────────────────────────────────────────

async function handleDbRuns(_req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const client = getDb();
  const url = new URL(_req.url!, `http://${_req.headers.host}`);
  const status = url.searchParams.get("status") as any;
  const domain = url.searchParams.get("domain") ?? undefined;
  const runs = await dbListRuns(client, { status: status || undefined, domain });
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(runs));
}

async function handleDbRunMetrics(req: http.IncomingMessage, res: http.ServerResponse, runId: string): Promise<void> {
  const client = getDb();
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const last = parseInt(url.searchParams.get("last") ?? "0", 10);

  const metrics = last > 0
    ? await getRecentMetrics(client, runId, last)
    : await getMetrics(client, runId);

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(metrics));
}

async function handleDbRunCheckpoints(_req: http.IncomingMessage, res: http.ServerResponse, runId: string): Promise<void> {
  const client = getDb();
  const checkpoints = await listCheckpoints(client, runId);
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(checkpoints));
}

async function handleDbDomains(_req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const client = getDb();
  const domains = await listDomains(client);
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(domains));
}

async function handleDbSync(_req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const client = getDb();
  const result = await syncFromDisk(client, OUTPUTS_DIR);
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(result));
}

// ── Router ────────────────────────────────────────────────────────────────

function setCors(res: http.ServerResponse): void {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Content-Encoding");
}

async function route(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const p = url.pathname;

  setCors(res);

  // CORS preflight
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  // API routes
  if (p === "/api/models") { handleModels(req, res); return; }
  if (p === "/api/inference" || p === "/inference") { await handleInference(req, res); return; }
  if ((p === "/api/chat") && req.method === "POST") { await handleChat(req, res); return; }
  if (p === "/api/generate") { await handleGenerate(req, res); return; }
  if (p === "/v1/models") { handleOpenAIModels(req, res); return; }
  if (p === "/v1/chat/completions" || p === "/chat/completions") { await handleChatCompletions(req, res); return; }
  if (p === "/api/upload" && req.method === "POST") { await handleUpload(req, res); return; }

  // Dashboard API routes
  if (p === "/api/runs") { await handleDbRuns(req, res); return; }
  if (p === "/api/domains") { await handleDbDomains(req, res); return; }
  if (p === "/api/sync" && req.method === "POST") { await handleDbSync(req, res); return; }
  const runMetricsMatch = p.match(/^\/api\/runs\/([^/]+)\/metrics$/);
  if (runMetricsMatch) { await handleDbRunMetrics(req, res, decodeURIComponent(runMetricsMatch[1])); return; }
  const runCheckpointsMatch = p.match(/^\/api\/runs\/([^/]+)\/checkpoints$/);
  if (runCheckpointsMatch) { await handleDbRunCheckpoints(req, res, decodeURIComponent(runCheckpointsMatch[1])); return; }

  // Static pages
  if (p === "/") { serveStatic(res, "index.html", "text/html"); return; }
  if (p === "/chat") { serveStatic(res, "chat.html", "text/html"); return; }
  if (p === "/docs") { serveStatic(res, "docs.html", "text/html"); return; }
  if (p === "/models") { serveStatic(res, "models.html", "text/html"); return; }

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

// Initialize database and sync runs
try {
  await createDb();
  const syncResult = await syncFromDisk(getDb(), OUTPUTS_DIR);
  console.log(`DB synced: ${syncResult.runsUpserted} runs, ${syncResult.metricsInserted} new metrics`);
} catch (e) {
  console.warn("DB init failed (dashboard will be unavailable):", (e as Error).message);
}

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
  console.log(`  Inference UI: http://localhost:${PORT}/`);
  console.log(`  Chat UI:      http://localhost:${PORT}/chat`);
  console.log(`  Dashboard:    http://localhost:${PORT}/dashboard`);
});
