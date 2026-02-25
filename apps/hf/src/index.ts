/**
 * Minimal HF Spaces inference server.
 *
 * Serves /v1/chat/completions and /v1/models for a single baked-in checkpoint.
 * No Next.js, no DB, no dashboard — just inference.
 *
 * Uses a dedicated inference engine (inference.ts) that bypasses the autograd
 * training path for 10-20× faster CPU inference via KV cache, tiled matmul,
 * and zero-allocation decode loop.
 */
import * as fs from "node:fs";
import * as crypto from "node:crypto";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import type { ModelConfig, Tokenizer, TokenizerArtifacts } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import { BpeTokenizer, CharTokenizer, WordTokenizer } from "@alpha/tokenizers";
import {
  type InferenceWeights,
  type InferenceSession,
  prepareInferenceWeights, createSession, resetCache, prefill, decodeStep, sampleFromLogits,
  countModelParams,
  SessionPool,
} from "@alpha/inference";

// ── Checkpoint loading (inlined — no @alpha/train dependency) ──────────────

interface LoadedCheckpoint {
  modelConfig: ModelConfig;
  params: Record<string, { shape: number[]; data: Float32Array | number[] }>;
  tokenizerArtifacts?: TokenizerArtifacts;
  step: number;
}

function loadCheckpoint(filePath: string): LoadedCheckpoint {
  const raw = fs.readFileSync(filePath);

  // Binary format: magic "ALPH" (0x41 0x4c 0x50 0x48)
  if (raw.length >= 4 && raw[0] === 0x41 && raw[1] === 0x4c && raw[2] === 0x50 && raw[3] === 0x48) {
    let offset = 4;
    const headerLen = raw.readUInt32LE(offset); offset += 4;
    const header = JSON.parse(raw.subarray(offset, offset + headerLen).toString("utf-8"));
    offset += headerLen;

    const params: Record<string, { shape: number[]; data: Float32Array }> = {};
    for (const t of header.tensors as Array<{ name: string; shape: number[]; elements: number }>) {
      const byteLen = t.elements * 4;
      const f32 = new Float32Array(raw.buffer.slice(raw.byteOffset + offset, raw.byteOffset + offset + byteLen));
      offset += byteLen;
      if (t.name.startsWith("p.")) {
        params[t.name.slice(2)] = { shape: t.shape, data: f32 };
      }
      // Skip optimizer buffers (o.*)
    }

    return {
      modelConfig: header.modelConfig,
      params,
      tokenizerArtifacts: header.tokenizerArtifacts,
      step: header.step,
    };
  }

  // Legacy JSON format
  const parsed = JSON.parse(raw.toString("utf-8"));
  return {
    modelConfig: parsed.modelConfig,
    params: parsed.params,
    tokenizerArtifacts: parsed.tokenizerArtifacts,
    step: parsed.step,
  };
}

// ── Build tokenizer ───────────────────────────────────────────────────────

function buildTokenizer(artifacts: TokenizerArtifacts): Tokenizer {
  if (artifacts.type === "bpe") {
    const tok = new BpeTokenizer();
    tok.loadArtifacts(artifacts);
    return tok;
  } else if (artifacts.type === "word") {
    const tok = new WordTokenizer();
    tok.loadArtifacts(artifacts);
    return tok;
  } else {
    const tok = new CharTokenizer();
    tok.loadArtifacts(artifacts);
    return tok;
  }
}

// ── Load model at startup ──────────────────────────────────────────────────

const MODEL_ID = "alpha-v0-historic";
const CHECKPOINT_PATH = process.env.CHECKPOINT_PATH ?? "/app/checkpoint.alph";
const PORT = parseInt(process.env.PORT ?? "7860", 10);

console.log(`Loading checkpoint from ${CHECKPOINT_PATH}...`);
const t0 = Date.now();
const ckpt = loadCheckpoint(CHECKPOINT_PATH);

if (!ckpt.tokenizerArtifacts) throw new Error("Checkpoint has no tokenizer artifacts");
const tokenizer = buildTokenizer(ckpt.tokenizerArtifacts);
const paramCount = countModelParams(ckpt.params);
const inferenceWeights = prepareInferenceWeights(ckpt.modelConfig, ckpt.params);
const sessionPool = new SessionPool(inferenceWeights);

console.log(
  `Loaded ${MODEL_ID} (${(paramCount / 1e6).toFixed(2)}M params, step ${ckpt.step}) in ${Date.now() - t0}ms`,
);

// ── Hono server ────────────────────────────────────────────────────────────

const app = new Hono();

// CORS
app.use("*", async (c, next) => {
  await next();
  c.header("Access-Control-Allow-Origin", "*");
  c.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  c.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
});
app.options("*", (c) => c.body(null, 204));

// Health
app.get("/", (c) =>
  c.json({ status: "ok", model: MODEL_ID, params: paramCount, step: ckpt.step }),
);

// GET /v1/models
app.get("/v1/models", (c) =>
  c.json({
    object: "list",
    data: [{
      id: MODEL_ID,
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "alpha",
    }],
  }),
);

// POST /v1/chat/completions
app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const temperature: number = body.temperature ?? 0.7;
  const stream: boolean = body.stream === true;

  const prompt = messages.map((m) => m.content).join("\n");
  if (!prompt) return c.json({ error: { message: "Empty prompt", type: "invalid_request_error" } }, 400);

  const config = inferenceWeights.config;
  const requestedMax = body.max_tokens ?? body.max_completion_tokens ?? 200;
  const maxTokens = Math.min(requestedMax, config.blockSize);
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const session = sessionPool.acquire();

  const allPromptTokens = tokenizer.encode(prompt);
  const maxPrompt = Math.max(1, config.blockSize - 1);
  const promptTokens = allPromptTokens.length > maxPrompt
    ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
    : allPromptTokens;

  let logits = prefill(inferenceWeights, session, new Int32Array(promptTokens));
  let currentPos = promptTokens.length;

  const completionId = "chatcmpl-" + crypto.randomBytes(12).toString("hex");
  const created = Math.floor(Date.now() / 1000);
  const generatedTokens: number[] = [];

  if (stream) {
    const enc = new TextEncoder();
    const readable = new ReadableStream({
      start(controller) {
        let count = 0;

        controller.enqueue(enc.encode(`data: ${JSON.stringify({
          id: completionId, object: "chat.completion.chunk", created, model: MODEL_ID,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        })}\n\n`));

        function next() {
          if (count >= maxTokens || currentPos >= config.blockSize) {
            controller.enqueue(enc.encode(`data: ${JSON.stringify({
              id: completionId, object: "chat.completion.chunk", created, model: MODEL_ID,
              choices: [{ index: 0, delta: {}, finish_reason: count >= maxTokens ? "length" : "stop" }],
              usage: { prompt_tokens: promptTokens.length, completion_tokens: count, total_tokens: promptTokens.length + count },
            })}\n\n`));
            controller.enqueue(enc.encode("data: [DONE]\n\n"));
            controller.close();
            sessionPool.release(session);
            return;
          }

          const tok = sampleFromLogits(session, logits, temperature, 40, rng);
          generatedTokens.push(tok);
          count++;

          const raw = tokenizer.decode(new Int32Array([tok]));
          const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";

          controller.enqueue(enc.encode(`data: ${JSON.stringify({
            id: completionId, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: { content: sep + raw }, finish_reason: null }],
          })}\n\n`));

          logits = decodeStep(inferenceWeights, session, tok, currentPos);
          currentPos++;

          setImmediate(next);
        }
        next();
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  // Non-streaming
  let count = 0;
  for (let i = 0; i < maxTokens && currentPos < config.blockSize; i++) {
    const tok = sampleFromLogits(session, logits, temperature, 40, rng);
    generatedTokens.push(tok);
    count++;

    logits = decodeStep(inferenceWeights, session, tok, currentPos);
    currentPos++;

    if (count % 8 === 0) await new Promise((r) => setImmediate(r));
  }

  sessionPool.release(session);
  const text = tokenizer.decode(new Int32Array(generatedTokens));

  return c.json({
    id: completionId,
    object: "chat.completion",
    created,
    model: MODEL_ID,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: count >= maxTokens ? "length" : "stop",
    }],
    usage: {
      prompt_tokens: promptTokens.length,
      completion_tokens: count,
      total_tokens: promptTokens.length + count,
    },
  });
});

// ── Start ──────────────────────────────────────────────────────────────────

serve({ fetch: app.fetch, port: PORT }, () => {
  console.log(`Alpha HF inference server listening on :${PORT}`);
});
