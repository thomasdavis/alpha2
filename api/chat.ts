import type { VercelRequest, VercelResponse } from "@vercel/node";
import { streamText } from "ai";
import { initEngine, getRuns, ensureModel, AlphaLanguageModel } from "../apps/server/src/lib/engine.js";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  await initEngine();
  const runs = getRuns();

  const body = req.body;
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];
  const modelId: string = body.model ?? runs[0]?.id;
  const maxTokens: number = Math.min(body.maxTokens ?? 200, 500);
  const temperature: number = body.temperature ?? 0.8;
  const topk: number = body.topk ?? 40;

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.status(400).json({ error: "Unknown model" });
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
