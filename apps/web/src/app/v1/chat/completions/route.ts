import * as crypto from "node:crypto";
import { SeededRng } from "@alpha/core";
import { prefill, decodeStep, sampleFromLogits } from "@alpha/inference";
import { getRuns, ensureModel } from "@/lib/engine";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

function messagesToPrompt(messages: Array<{ role: string; content: string }>): string {
  return messages.map((m) => m.content).join("\n");
}

export async function POST(request: Request) {
  const body = await request.json();
  const runs = getRuns();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const modelId: string = body.model ?? runs[0]?.id;
  const maxTokens: number = Math.min(body.max_tokens ?? body.max_completion_tokens ?? 2048, 20_000);
  const temperature: number = body.temperature ?? 0.7;
  const topk: number = body.topk ?? body.top_k ?? 40;
  const topp: number = body.top_p ?? body.topp ?? 1.0;
  const stream: boolean = body.stream === true;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return jsonResponse({ error: { message: "Unknown model: " + modelId, type: "invalid_request_error" } }, 400);
  }

  const prompt = messagesToPrompt(messages);
  if (!prompt) {
    return jsonResponse({ error: { message: "Empty prompt", type: "invalid_request_error" } }, 400);
  }

  const model = await ensureModel(modelId);
  const { config, tokenizer, weights, sessionPool } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const session = sessionPool.acquire();

  const allPromptTokens = tokenizer.encode(prompt);
  const maxPrompt = Math.max(1, config.blockSize - 1);
  const promptTokens = allPromptTokens.length > maxPrompt
    ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
    : allPromptTokens;

  let logits = prefill(weights, session, Int32Array.from(promptTokens));
  let currentPos = promptTokens.length;

  const completionId = "chatcmpl-" + crypto.randomBytes(12).toString("hex");
  const created = Math.floor(Date.now() / 1000);
  const generatedTokens: number[] = [];

  if (stream) {
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      start(controller) {
        let completionCount = 0;

        controller.enqueue(encoder.encode(`data: ${JSON.stringify({
          id: completionId, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        })}\n\n`));

        function nextChunk() {
          if (request.signal.aborted || completionCount >= maxTokens || currentPos >= config.blockSize) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({
              id: completionId, object: "chat.completion.chunk", created, model: modelId,
              choices: [{ index: 0, delta: {}, finish_reason: completionCount >= maxTokens ? "length" : "stop" }],
              usage: { prompt_tokens: promptTokens.length, completion_tokens: completionCount, total_tokens: promptTokens.length + completionCount },
            })}\n\n`));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
            sessionPool.release(session);
            return;
          }

          const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
          generatedTokens.push(tok);
          completionCount++;
          const raw = tokenizer.decode(new Int32Array([tok]));
          const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            id: completionId, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: { content: sep + raw }, finish_reason: null }],
          })}\n\n`));

          logits = decodeStep(weights, session, tok, currentPos);
          currentPos++;

          setImmediate(nextChunk);
        }
        nextChunk();
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
  let completionCount = 0;
  for (let i = 0; i < maxTokens && currentPos < config.blockSize; i++) {
    const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
    generatedTokens.push(tok);
    completionCount++;

    logits = decodeStep(weights, session, tok, currentPos);
    currentPos++;
  }

  sessionPool.release(session);

  const text = tokenizer.decode(new Int32Array(generatedTokens));

  return jsonResponse({
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
  });
}
