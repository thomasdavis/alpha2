import * as crypto from "node:crypto";
import { SeededRng } from "@alpha/core";
import { resetCache, prefill, decodeStep, sampleFromLogits } from "@alpha/inference";
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
  const stream: boolean = body.stream === true;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return jsonResponse({ error: { message: "Unknown model: " + modelId, type: "invalid_request_error" } }, 400);
  }

  const model = await ensureModel(modelId);
  const { config, tokenizer, inference } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const prompt = messagesToPrompt(messages);

  const allPromptTokens = tokenizer.encode(prompt);
  const maxPrompt = Math.max(1, config.blockSize - 1);
  const promptTokens = allPromptTokens.length > maxPrompt
    ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
    : allPromptTokens;

  // Reset KV cache and prefill prompt
  resetCache(inference);
  let logits = prefill(inference, Int32Array.from(promptTokens));
  let currentPos = promptTokens.length;

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
          if (completionCount >= maxTokens || currentPos >= config.blockSize) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({
              id: completionId, object: "chat.completion.chunk", created, model: modelId,
              choices: [{ index: 0, delta: {}, finish_reason: completionCount >= maxTokens ? "length" : "stop" }],
              usage: { prompt_tokens: promptTokens.length, completion_tokens: completionCount, total_tokens: promptTokens.length + completionCount },
            })}\n\n`));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
            return;
          }

          const tok = sampleFromLogits(inference, logits, temperature, 40, rng);
          completionCount++;
          const raw = tokenizer.decode(new Int32Array([tok]));
          const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";

          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            id: completionId, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: { content: sep + raw }, finish_reason: null }],
          })}\n\n`));

          logits = decodeStep(inference, tok, currentPos);
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
  const generatedTokens: number[] = [];
  for (let i = 0; i < maxTokens && currentPos < config.blockSize; i++) {
    const tok = sampleFromLogits(inference, logits, temperature, 40, rng);
    generatedTokens.push(tok);
    completionCount++;

    logits = decodeStep(inference, tok, currentPos);
    currentPos++;
  }

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
