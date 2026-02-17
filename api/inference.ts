import type { VercelRequest, VercelResponse } from "@vercel/node";
import * as path from "node:path";
import { initEngine, getRuns, ensureModel, generateTokens } from "../apps/server/src/lib/engine.js";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  await initEngine(path.resolve(process.cwd(), "outputs"));
  const runs = getRuns();

  const query = (req.query.query as string) ?? "";
  const modelId = (req.query.model as string) ?? runs[0]?.id;
  const steps = Math.min(parseInt((req.query.steps as string) ?? "200", 10), 500);
  const temperature = parseFloat((req.query.temp as string) ?? "0.8");
  const topk = parseInt((req.query.topk as string) ?? "40", 10);

  if (!modelId || !runs.find((r) => r.id === modelId)) {
    res.status(400).json({ error: "Unknown model" });
    return;
  }

  const model = await ensureModel(modelId);

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const gen = generateTokens(model, query, steps, temperature, topk);

  for (const token of gen) {
    res.write(`data: ${JSON.stringify({ token })}\n\n`);
  }
  res.write("data: [DONE]\n\n");
  res.end();
}
