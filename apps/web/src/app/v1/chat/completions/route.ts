import * as crypto from "node:crypto";
import { SeededRng } from "@alpha/core";
import { getRuns, ensureModel, sampleNextToken } from "@/lib/engine";
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
  const stream: boolean = body.stream === true;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return jsonResponse({ error: { message: "Unknown model: " + modelId, type: "invalid_request_error" } }, 400);
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
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      start(controller) {
        let completionCount = 0;

        // Role chunk
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({
          id: completionId, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        })}\n\n`));

        function nextChunk() {
          if (request.signal.aborted) return;
          if (completionCount >= maxTokens || currentLen >= config.blockSize) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({
              id: completionId, object: "chat.completion.chunk", created, model: modelId,
              choices: [{ index: 0, delta: {}, finish_reason: completionCount >= maxTokens ? "length" : "stop" }],
              usage: { prompt_tokens: promptTokens.length, completion_tokens: completionCount, total_tokens: promptTokens.length + completionCount },
            })}\n\n`));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
            return;
          }

          const next = sampleNextToken(model, tokens, currentLen, temperature, 40, rng);
          tokens[currentLen] = next;
          currentLen++;
          completionCount++;
          const raw = tokenizer.decode(new Int32Array([next]));
          const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            id: completionId, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: { content: sep + raw }, finish_reason: null }],
          })}\n\n`));
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
  for (let i = 0; i < maxTokens && currentLen < config.blockSize; i++) {
    const next = sampleNextToken(model, tokens, currentLen, temperature, 40, rng);
    tokens[currentLen] = next;
    currentLen++;
    completionCount++;
  }

  const text = tokenizer.decode(tokens.slice(promptTokens.length, currentLen));

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
