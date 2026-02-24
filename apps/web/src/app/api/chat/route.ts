import { streamText } from "ai";
import { getRuns, ensureModel, AlphaLanguageModel } from "@/lib/engine";
import { ensureInit } from "@/lib/init";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  await ensureInit();
  const body = await request.json();
  const runs = getRuns();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const modelId: string = body.model ?? runs[0]?.id;
  const maxTokens: number = Math.min(body.maxTokens ?? 200, 20_000);
  const temperature: number = body.temperature ?? 0.8;
  const topk: number = body.topk ?? 40;

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return Response.json({ error: "Unknown model" }, { status: 400 });
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

  return result.toTextStreamResponse();
}
