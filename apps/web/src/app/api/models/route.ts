import * as fs from "node:fs";
import * as path from "node:path";
import { NextRequest } from "next/server";
import { getClient } from "@/lib/db";
import { listRuns as dbListRuns } from "@alpha/db";
import { getRuns, resetEngine, initEngine } from "@/lib/engine";
import {
  OUTPUTS_DIR, modelsCache, MODELS_CACHE_TTL, invalidateModelsCache, setModelsCache,
  checkAuth, jsonResponse,
} from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET() {
  if (modelsCache && Date.now() - modelsCache.ts < MODELS_CACHE_TTL) {
    return new Response(modelsCache.json, {
      headers: { "Content-Type": "application/json" },
    });
  }

  const localRuns = getRuns().map((r) => ({
    id: r.id, runId: r.config?.runId ?? r.id, name: r.name, step: r.step, mtime: r.mtime, lastLoss: r.lastLoss,
    modelConfig: r.config.modelConfig, trainConfig: r.config.trainConfig,
    domain: r.domain,
  }));

  const seen = new Set(localRuns.map((r) => r.id));
  try {
    const client = await getClient();
    const dbRuns = await dbListRuns(client, {});
    for (const r of dbRuns) {
      if (seen.has(r.id)) continue;
      let modelConfig, trainConfig;
      try { modelConfig = JSON.parse(r.model_config); } catch { modelConfig = {}; }
      try { trainConfig = JSON.parse(r.train_config); } catch { trainConfig = {}; }
      localRuns.push({
        id: r.id,
        runId: r.run_id || r.id,
        name: r.run_id || r.id,
        step: r.latest_step ?? 0,
        mtime: r.disk_mtime ?? new Date(r.updated_at).getTime(),
        lastLoss: r.last_loss ?? undefined,
        modelConfig,
        trainConfig,
        domain: r.domain,
      });
    }
  } catch { /* DB unavailable — return local runs only */ }

  const json = JSON.stringify(localRuns);
  setModelsCache(json);
  return new Response(json, {
    headers: { "Content-Type": "application/json" },
  });
}

export async function DELETE(request: NextRequest) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const name = request.nextUrl.searchParams.get("name");
  if (!name) {
    return jsonResponse({ error: "Missing ?name= parameter" }, 400);
  }

  const runDir = path.join(OUTPUTS_DIR, name);
  if (!fs.existsSync(runDir)) {
    return jsonResponse({ error: "Model not found" }, 404);
  }

  fs.rmSync(runDir, { recursive: true, force: true });
  resetEngine();
  await initEngine(OUTPUTS_DIR);
  invalidateModelsCache();

  console.log(`Deleted model: ${name}`);
  return jsonResponse({ ok: true, deleted: name });
}
