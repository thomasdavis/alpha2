import { NextRequest } from "next/server";
import { SeededRng } from "@alpha/core";
import { prefill, decodeStep, sampleFromLogits } from "@alpha/inference";
import { getRuns, ensureModel } from "@/lib/engine";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  return handleGenerate(request);
}

export async function POST(request: NextRequest) {
  return handleGenerate(request);
}

async function handleGenerate(request: NextRequest) {
  const runs = getRuns();
  const body = request.method === "POST" ? await request.json() : {};
  const prompt: string = request.nextUrl.searchParams.get("prompt") ?? body.prompt ?? "";
  const maxTokens: number = Math.min(
    parseInt(request.nextUrl.searchParams.get("max_tokens") ?? "", 10) || (body.max_tokens ?? 2048),
    20_000,
  );
  const temperature: number = parseFloat(request.nextUrl.searchParams.get("temperature") ?? "") || (body.temperature ?? 0.7);
  const topkParam = request.nextUrl.searchParams.get("topk");
  const topkParsed = topkParam !== null ? Number.parseInt(topkParam, 10) : Number.NaN;
  const topk: number = Number.isFinite(topkParsed) ? topkParsed : (body.topk ?? body.top_k ?? 40);
  const toppParam = request.nextUrl.searchParams.get("top_p");
  const toppParsed = toppParam !== null ? Number.parseFloat(toppParam) : Number.NaN;
  const topp: number = Number.isFinite(toppParsed) ? toppParsed : (body.top_p ?? body.topp ?? 1.0);
  const modelId: string = request.nextUrl.searchParams.get("model") ?? body.model ?? runs[0]?.id;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return jsonResponse({ error: "Unknown model" }, 400);
  }

  if (!prompt) {
    return jsonResponse({ error: "Empty prompt" }, 400);
  }

  const model = await ensureModel(modelId);
  const { config, tokenizer, weights, sessionPool } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const session = sessionPool.acquire();

  try {
    const allPromptTokens = tokenizer.encode(prompt);
    const maxPrompt = Math.max(1, config.blockSize - 1);
    const promptTokens = allPromptTokens.length > maxPrompt
      ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
      : allPromptTokens;

    let logits = prefill(weights, session, Int32Array.from(promptTokens));
    let currentPos = promptTokens.length;
    let completionCount = 0;
    const generatedTokens: number[] = [];

    for (let i = 0; i < maxTokens && currentPos < config.blockSize; i++) {
      const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
      generatedTokens.push(tok);
      completionCount++;

      logits = decodeStep(weights, session, tok, currentPos);
      currentPos++;
    }

    const text = tokenizer.decode(new Int32Array(generatedTokens));

    return jsonResponse({
      text,
      model: modelId,
      usage: {
        prompt_tokens: promptTokens.length,
        completion_tokens: completionCount,
      },
    });
  } finally {
    sessionPool.release(session);
  }
}
