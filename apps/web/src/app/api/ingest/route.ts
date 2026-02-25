import { getClient } from "@/lib/db";
import { upsertRun, insertMetrics, updateRunProgress, insertSamples } from "@alpha/db";
import { checkAuth, invalidateModelsCache, broadcastLive, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const body = await request.json();
  const { type } = body as { type: string };

  if (type === "run_start") {
    const { runId, domain, modelConfig, trainConfig, totalParams, infra } = body;
    try {
      const client = await getClient();
      const tc = trainConfig ?? {};
      const mc = modelConfig ?? {};
      await upsertRun(client, {
        id: runId,
        run_id: runId,
        config_hash: "",
        domain: domain || "unknown",
        model_config: modelConfig,
        train_config: trainConfig,
        status: "active",
        estimated_params: totalParams ?? null,
        gpu_name: infra?.gpuName ?? null,
        gpu_vendor: infra?.gpuVendor ?? null,
        gpu_vram_mb: infra?.gpuVramMb ?? null,
        hostname: infra?.hostname ?? null,
        cpu_count: infra?.cpuCount ?? null,
        ram_total_mb: infra?.ramTotalMb ?? null,
        os_platform: infra?.osPlatform ?? null,
        symbio: tc.symbio ? 1 : 0,
        symbio_config: tc.symbioConfig ? JSON.stringify(tc.symbioConfig) : null,
        ffn_activation: mc.ffnActivation ?? null,
        symbio_mode: tc.symbio ? (tc.symbioConfig?.searchMode ?? "ffn-activation-search") : null,
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
      const client = await getClient();
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
      const client = await getClient();
      await insertSamples(client, runId, samples);
    } catch (e) {
      console.warn("Ingest samples DB error:", (e as Error).message);
    }
    broadcastLive("samples", { runId, samples });
    return jsonResponse({ ok: true });
  }

  return jsonResponse({ error: "Unknown ingest type" }, 400);
}
