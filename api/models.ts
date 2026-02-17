import type { VercelRequest, VercelResponse } from "@vercel/node";
import * as path from "node:path";
import { initEngine, getRuns } from "../apps/server/src/lib/engine.js";

export default async function handler(_req: VercelRequest, res: VercelResponse) {
  await initEngine(path.resolve(process.cwd(), "outputs"));

  const payload = getRuns().map((r) => ({
    id: r.id,
    name: r.name,
    step: r.step,
    mtime: r.mtime,
    lastLoss: r.lastLoss,
    modelConfig: r.config.modelConfig,
    trainConfig: r.config.trainConfig,
  }));
  res.status(200).json(payload);
}
