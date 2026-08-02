/** Public inference service for Alpha's selected conversational-repair checkpoint. */
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import { serve } from "@hono/node-server";
import type { ModelConfig, Tokenizer, TokenizerArtifacts } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import {
  type InferenceSession,
  countModelParams,
  decodeStep,
  prefill,
  prepareInferenceWeights,
  sampleFromLogits,
  SessionPool,
} from "@alpha/inference";
import { BpeTokenizer, ByteBpeTokenizer, CharTokenizer, WordTokenizer } from "@alpha/tokenizers";
import { Hono } from "hono";
import { finiteNumber, formatChatPrompt, MODEL_ID, parseMessages, positiveInteger, resolveChatStopTokenIds } from "./protocol.js";
import { renderUi, SELECTED_EVIDENCE, type UiEvidence } from "./ui.js";

interface LoadedCheckpoint {
  readonly modelConfig: ModelConfig;
  readonly params: Record<string, { shape: number[]; data: Float32Array | number[] }>;
  readonly tokenizerArtifacts?: TokenizerArtifacts;
  readonly step: number;
}

function loadCheckpoint(filePath: string): LoadedCheckpoint {
  const raw = fs.readFileSync(filePath);
  if (raw.length >= 4 && raw.subarray(0, 4).toString("ascii") === "ALPH") {
    let offset = 4;
    const headerLength = raw.readUInt32LE(offset);
    offset += 4;
    if (headerLength <= 0 || headerLength > 16 * 1024 * 1024 || offset + headerLength > raw.length) {
      throw new Error(`invalid ALPH header length ${headerLength}`);
    }
    const header = JSON.parse(raw.subarray(offset, offset + headerLength).toString("utf-8"));
    offset += headerLength;
    const params: Record<string, { shape: number[]; data: Float32Array }> = {};
    for (const tensor of header.tensors as Array<{ name: string; shape: number[]; elements: number }>) {
      const byteLength = tensor.elements * 4;
      if (offset + byteLength > raw.length) throw new Error(`truncated tensor ${tensor.name}`);
      if (tensor.name.startsWith("p.")) {
        const bytes = raw.subarray(offset, offset + byteLength);
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        params[tensor.name.slice(2)] = { shape: tensor.shape, data: new Float32Array(copy) };
      }
      offset += byteLength;
    }
    return {
      modelConfig: header.modelConfig,
      params,
      tokenizerArtifacts: header.tokenizerArtifacts,
      step: header.step,
    };
  }
  const parsed = JSON.parse(raw.toString("utf-8"));
  return {
    modelConfig: parsed.modelConfig,
    params: parsed.params,
    tokenizerArtifacts: parsed.tokenizerArtifacts,
    step: parsed.step,
  };
}

function buildTokenizer(artifacts: TokenizerArtifacts): Tokenizer {
  if (artifacts.type === "bpe" || artifacts.type === "byte_bpe") {
    if (artifacts.type === "byte_bpe") {
      const byteTokenizer = new ByteBpeTokenizer();
      byteTokenizer.loadArtifacts(artifacts);
      return byteTokenizer;
    }
    const tokenizer = new BpeTokenizer();
    tokenizer.loadArtifacts(artifacts);
    return tokenizer;
  }
  if (artifacts.type === "word") {
    const tokenizer = new WordTokenizer();
    tokenizer.loadArtifacts(artifacts);
    return tokenizer;
  }
  const tokenizer = new CharTokenizer();
  tokenizer.loadArtifacts(artifacts);
  return tokenizer;
}

const CHECKPOINT_PATH = process.env.CHECKPOINT_PATH ?? "/app/checkpoint.alph";
const PORT = Number.parseInt(process.env.PORT ?? "7860", 10);
const HOST = process.env.HOST ?? "127.0.0.1";
if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65_535) throw new Error(`invalid PORT: ${process.env.PORT}`);

function loadEvidence(path: string | undefined): UiEvidence {
  if (!path) return SELECTED_EVIDENCE;
  const value = JSON.parse(fs.readFileSync(path, "utf8")) as Partial<UiEvidence>;
  for (const field of ["totalChat", "structuralPass", "emptyReplies", "degenerateLoops", "qaExact", "qaTotal"] as const) {
    if (!Number.isSafeInteger(value[field]) || Number(value[field]) < 0) throw new Error(`invalid evidence field: ${field}`);
  }
  for (const field of ["meanRepeat", "maxRepeat", "exportTop1", "checkpointSha256"] as const) {
    if (typeof value[field] !== "string" || value[field]!.length === 0) throw new Error(`invalid evidence field: ${field}`);
  }
  if (!Number.isFinite(value.exportMaxLogitDifference) || Number(value.exportMaxLogitDifference) < 0) {
    throw new Error("invalid evidence field: exportMaxLogitDifference");
  }
  if (value.qualityGate !== "PASS" && value.qualityGate !== "FAIL") throw new Error("invalid evidence qualityGate");
  return value as UiEvidence;
}

const RUNTIME_EVIDENCE = loadEvidence(process.env.EVIDENCE_PATH);

console.log(`Loading checkpoint from ${CHECKPOINT_PATH}...`);
const loadStarted = Date.now();
const checkpoint = loadCheckpoint(CHECKPOINT_PATH);
if (!checkpoint.tokenizerArtifacts) throw new Error("checkpoint has no tokenizer artifacts");
const tokenizer = buildTokenizer(checkpoint.tokenizerArtifacts);
const chatStopTokenIds = resolveChatStopTokenIds((text) => tokenizer.encode(text));
const eosId = chatStopTokenIds.eos;
const paramCount = countModelParams(checkpoint.params);
const inferenceWeights = prepareInferenceWeights(checkpoint.modelConfig, checkpoint.params);
const sessionPool = new SessionPool(inferenceWeights);
const startedAt = new Date().toISOString();
console.log(`Loaded ${MODEL_ID} (${(paramCount / 1e6).toFixed(2)}M params, step ${checkpoint.step}) in ${Date.now() - loadStarted}ms`);

const app = new Hono();

app.use("*", async (context, next) => {
  await next();
  context.header("Access-Control-Allow-Origin", "*");
  context.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  context.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  context.header("X-Alpha-Quality-Gate", RUNTIME_EVIDENCE.qualityGate);
});
app.options("*", (context) => context.body(null, 204));
app.onError((error, context) => {
  console.error(error);
  return context.json({ error: { message: "inference request failed", type: "server_error" } }, 500);
});

app.get("/", (context) => context.html(renderUi(paramCount, checkpoint.step, "", RUNTIME_EVIDENCE)));
app.get("/health", (context) => context.json({
  status: "ok",
  model: MODEL_ID,
  parameters: paramCount,
  checkpoint_step: checkpoint.step,
  quality_gate: RUNTIME_EVIDENCE.qualityGate,
  started_at: startedAt,
}));
app.get("/evidence", (context) => context.json({
  model: MODEL_ID,
  quality_gate: RUNTIME_EVIDENCE.qualityGate,
  frozen_chat: {
    structural_pass: RUNTIME_EVIDENCE.structuralPass,
    total: RUNTIME_EVIDENCE.totalChat,
    empty: RUNTIME_EVIDENCE.emptyReplies,
    degenerate_loops: RUNTIME_EVIDENCE.degenerateLoops,
    mean_four_gram_repeat_rate: RUNTIME_EVIDENCE.meanRepeat,
    max_four_gram_repeat_rate: RUNTIME_EVIDENCE.maxRepeat,
  },
  frozen_qa: { exact_match: RUNTIME_EVIDENCE.qaExact, total: RUNTIME_EVIDENCE.qaTotal },
  export_parity: {
    result: "PASS",
    top1: RUNTIME_EVIDENCE.exportTop1,
    max_logit_difference: RUNTIME_EVIDENCE.exportMaxLogitDifference,
  },
  checkpoint_sha256: RUNTIME_EVIDENCE.checkpointSha256,
}));
app.get("/v1/models", (context) => context.json({
  object: "list",
  data: [{ id: MODEL_ID, object: "model", created: 0, owned_by: "ajaxdavis" }],
}));

function completionChunk(id: string, created: number, delta: Record<string, unknown>, finishReason: string | null, usage?: Record<string, number>) {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model: MODEL_ID,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage ? { usage } : {}),
  };
}

function releaseOnce(session: InferenceSession): () => void {
  let released = false;
  return () => {
    if (!released) {
      released = true;
      sessionPool.release(session);
    }
  };
}

app.post("/v1/chat/completions", async (context) => {
  let body: Record<string, unknown>;
  try {
    const candidate = await context.req.json();
    if (typeof candidate !== "object" || candidate === null || Array.isArray(candidate)) throw new Error("request body must be an object");
    body = candidate as Record<string, unknown>;
  } catch (error) {
    const message = error instanceof Error ? error.message : "invalid JSON body";
    return context.json({ error: { message, type: "invalid_request_error" } }, 400);
  }

  let prompt: string;
  let temperature: number;
  let topK: number;
  let topP: number;
  let maxTokens: number;
  try {
    prompt = formatChatPrompt(parseMessages(body.messages));
    temperature = finiteNumber(body.temperature, 0, 0, 2);
    topK = Math.floor(finiteNumber(body.top_k ?? body.topk, 40, 0, inferenceWeights.config.vocabSize));
    topP = finiteNumber(body.top_p ?? body.topp, 1, 0, 1);
    maxTokens = positiveInteger(body.max_completion_tokens ?? body.max_tokens, 96, 256);
  } catch (error) {
    const message = error instanceof Error ? error.message : "invalid request";
    return context.json({ error: { message, type: "invalid_request_error" } }, 400);
  }

  const promptTokens = tokenizer.encode(prompt);
  if (promptTokens.length >= inferenceWeights.config.blockSize) {
    return context.json({
      error: {
        message: `formatted prompt is ${promptTokens.length} tokens; this checkpoint supports fewer than ${inferenceWeights.config.blockSize}`,
        type: "context_length_exceeded",
      },
    }, 400);
  }
  maxTokens = Math.min(maxTokens, inferenceWeights.config.blockSize - promptTokens.length);
  const stream = body.stream === true;
  const rng = new SeededRng(Date.now() & 0xffff_ffff);
  const session = sessionPool.acquire();
  const release = releaseOnce(session);
  let logits: Float32Array;
  try {
    logits = prefill(inferenceWeights, session, promptTokens);
  } catch (error) {
    release();
    throw error;
  }
  let currentPosition = promptTokens.length;
  const completionId = `chatcmpl-${crypto.randomBytes(12).toString("hex")}`;
  const created = Math.floor(Date.now() / 1000);
  const generated: number[] = [];

  if (stream) {
    const encoder = new TextEncoder();
    let closed = false;
    const readable = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(completionChunk(completionId, created, { role: "assistant", content: "" }, null))}\n\n`));
        let emittedText = "";

        const finish = (reason: "stop" | "length") => {
          if (closed) return;
          const finalText = tokenizer.decode(new Int32Array(generated));
          if (finalText.length > emittedText.length && finalText.startsWith(emittedText)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(completionChunk(completionId, created, { content: finalText.slice(emittedText.length) }, null))}\n\n`));
          }
          const usage = { prompt_tokens: promptTokens.length, completion_tokens: generated.length, total_tokens: promptTokens.length + generated.length };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(completionChunk(completionId, created, {}, reason, usage))}\n\n`));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          closed = true;
          controller.close();
          release();
        };

        const next = () => {
          if (closed) return;
          try {
            if (generated.length >= maxTokens || currentPosition >= inferenceWeights.config.blockSize) {
              finish("length");
              return;
            }
            const token = sampleFromLogits(session, logits, temperature, topK, rng, topP);
            if (chatStopTokenIds.all.has(token)) {
              finish("stop");
              return;
            }
            generated.push(token);
            const decoded = tokenizer.decode(new Int32Array(generated));
            const stable = decoded.endsWith("�") ? decoded.slice(0, -1) : decoded;
            if (stable.length > emittedText.length && stable.startsWith(emittedText)) {
              const delta = stable.slice(emittedText.length);
              emittedText = stable;
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(completionChunk(completionId, created, { content: delta }, null))}\n\n`));
            }
            logits = decodeStep(inferenceWeights, session, token, currentPosition);
            currentPosition++;
            setImmediate(next);
          } catch (error) {
            closed = true;
            release();
            controller.error(error);
          }
        };
        setImmediate(next);
      },
      cancel() {
        closed = true;
        release();
      },
    });
    return new Response(readable, {
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform", Connection: "keep-alive" },
    });
  }

  let finishReason: "stop" | "length" = "length";
  try {
    for (let index = 0; index < maxTokens && currentPosition < inferenceWeights.config.blockSize; index++) {
      const token = sampleFromLogits(session, logits, temperature, topK, rng, topP);
      if (chatStopTokenIds.all.has(token)) {
        finishReason = "stop";
        break;
      }
      generated.push(token);
      logits = decodeStep(inferenceWeights, session, token, currentPosition);
      currentPosition++;
      if (generated.length % 8 === 0) await new Promise<void>((resolve) => setImmediate(resolve));
    }
  } finally {
    release();
  }
  const text = tokenizer.decode(new Int32Array(generated));
  return context.json({
    id: completionId,
    object: "chat.completion",
    created,
    model: MODEL_ID,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: finishReason }],
    usage: { prompt_tokens: promptTokens.length, completion_tokens: generated.length, total_tokens: promptTokens.length + generated.length },
    alpha: { quality_gate: RUNTIME_EVIDENCE.qualityGate, empty_eos: finishReason === "stop" && generated.length === 0 },
  });
});

serve({ fetch: app.fetch, hostname: HOST, port: PORT }, () => {
  console.log(`Alpha HF Space listening on ${HOST}:${PORT}`);
});
