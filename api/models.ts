import type { VercelRequest, VercelResponse } from "@vercel/node";
import { initEngine, getRuns } from "../apps/server/src/lib/engine.js";

export default async function handler(_req: VercelRequest, res: VercelResponse) {
  await initEngine();

  const payload = getRuns().map((r) => ({
    id: r.id,
    name: r.name,
    step: r.step,
    mtime: r.mtime,
    lastLoss: r.lastLoss,
    modelConfig: r.config.modelConfig,
    trainConfig: r.config.trainConfig,
    domain: r.domain,
  }));
  res.status(200).json(payload);
}
