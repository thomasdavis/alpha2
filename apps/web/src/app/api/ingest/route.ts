import { getDb, upsertRun, insertMetrics, updateRunProgress, insertSamples } from "@alpha/db";
import { checkAuth, invalidateModelsCache, broadcastLive, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const body = await request.json();
  const { type } = body as { type: string };

  if (type === "run_start") {
    const { runId, domain, modelConfig, trainConfig, totalParams } = body;
    try {
      const client = getDb();
      await upsertRun(client, {
        id: runId,
        run_id: runId,
        config_hash: "",
        domain: domain || "unknown",
        model_config: modelConfig,
        train_config: trainConfig,
        status: "active",
        estimated_params: totalParams ?? null,
      });
    } catch (e) {
      console.warn("Ingest run_start DB error:", (e as Error).message);
    }
    invalidateModelsCache();
    broadcastLive("run_start", { runId, domain, modelConfig, trainConfig, totalParams });
    return jsonResponse({ ok: true });
  }

  if (type === "metrics") {
    const { runId, metrics } = body as { runId: string; metrics: Array<Record<string, unknown>> };
    try {
      const client = getDb();
      await insertMetrics(client, runId, metrics as any);
      const last = metrics[metrics.length - 1] as any;
      if (last) {
        await updateRunProgress(client, runId, {
          latest_step: last.step,
          last_loss: last.loss,
          best_val_loss: last.valLoss ?? null,
        });
      }
    } catch (e) {
      console.warn("Ingest metrics DB error:", (e as Error).message);
    }
    broadcastLive("metrics", { runId, metrics });
    return jsonResponse({ ok: true });
  }

  if (type === "samples") {
    const { runId, samples } = body as { runId: string; samples: Array<{ prompt: string; output: string }> };
    try {
      const client = getDb();
      await insertSamples(client, runId, samples);
    } catch (e) {
      console.warn("Ingest samples DB error:", (e as Error).message);
    }
    broadcastLive("samples", { runId, samples });
    return jsonResponse({ ok: true });
  }

  return jsonResponse({ error: "Unknown ingest type" }, 400);
}
