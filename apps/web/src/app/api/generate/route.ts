import { NextRequest } from "next/server";
import { SeededRng } from "@alpha/core";
import { getRuns, ensureModel, sampleNextToken } from "@/lib/engine";
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
  const modelId: string = request.nextUrl.searchParams.get("model") ?? body.model ?? runs[0]?.id;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return jsonResponse({ error: "Unknown model" }, 400);
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

  return jsonResponse({
    text,
    model: modelId,
    usage: {
      prompt_tokens: promptTokens.length,
      completion_tokens: completionCount,
    },
  });
}
