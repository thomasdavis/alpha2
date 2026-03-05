import { getClient } from "@/lib/db";
import { upsertRun, insertMetrics, updateRunProgress, insertSamples, insertEvent, insertEvents } from "@alpha/db";
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
        let batchLatestStep = -1;
        let batchBestVal: number | null = null;
        for (const metric of metrics as any[]) {
          if (typeof metric.step === "number" && metric.step > batchLatestStep) {
            batchLatestStep = metric.step;
          }
          const v = typeof metric.valLoss === "number"
            ? metric.valLoss
            : typeof metric.val_loss === "number"
              ? metric.val_loss
              : null;
          if (v !== null && Number.isFinite(v)) {
            batchBestVal = batchBestVal === null ? v : Math.min(batchBestVal, v);
          }
        }
        if (batchLatestStep >= 0) {
          await updateRunProgress(client, runId, {
            latest_step: batchLatestStep,
            last_loss: typeof last.loss === "number" ? last.loss : null,
            best_val_loss: batchBestVal,
            status: "active",
          });
        }
      }
    } catch (e) {
      console.warn("Ingest metrics DB error:", (e as Error).message);
    }
    broadcastLive("metrics", { runId, metrics });
    return jsonResponse({ ok: true });
  }

  if (type === "heartbeat") {
    const { runId, step } = body as { runId: string; step?: number };
    try {
      if (Number.isFinite(step as number)) {
        const client = await getClient();
        await updateRunProgress(client, runId, {
          latest_step: step as number,
          status: "active",
        });
      }
    } catch (e) {
      console.warn("Ingest heartbeat DB error:", (e as Error).message);
    }
    broadcastLive("heartbeat", { runId, step: Number.isFinite(step as number) ? step : null });
    return jsonResponse({ ok: true });
  }

  if (type === "samples") {
    const { runId, samples, step, trend } = body as {
      runId: string;
      samples: Array<{ prompt: string; output: string }>;
      step?: number;
      trend?: unknown;
    };
    try {
      const client = await getClient();
      await insertSamples(client, runId, samples);
      if (Number.isFinite(step as number)) {
        await updateRunProgress(client, runId, {
          latest_step: step as number,
          status: "active",
        });
      }
    } catch (e) {
      console.warn("Ingest samples DB error:", (e as Error).message);
    }
    broadcastLive("samples", { runId, samples, step: Number.isFinite(step as number) ? step : null, trend: trend ?? null });
    return jsonResponse({ ok: true });
  }

  if (type === "event") {
    const {
      runId,
      event,
    } = body as {
      runId: string;
      event: {
        step?: number;
        level?: "debug" | "info" | "warn" | "error";
        kind?: string;
        message?: string;
        payload?: unknown;
        timestamp?: string;
      };
    };
    try {
      const client = await getClient();
      await insertEvent(client, runId, {
        step: typeof event?.step === "number" ? event.step : null,
        level: event?.level,
        kind: event?.kind ?? "generic",
        message: event?.message ?? "event",
        payload: event?.payload ?? null,
        createdAt: event?.timestamp ?? null,
      });
      if (Number.isFinite(event?.step as number)) {
        await updateRunProgress(client, runId, {
          latest_step: event.step as number,
          status: "active",
        });
      }
    } catch (e) {
      console.warn("Ingest event DB error:", (e as Error).message);
    }
    broadcastLive("event", { runId, event: event ?? null });
    return jsonResponse({ ok: true });
  }

  if (type === "events") {
    const {
      runId,
      events,
    } = body as {
      runId: string;
      events: Array<{
        step?: number;
        level?: "debug" | "info" | "warn" | "error";
        kind?: string;
        message?: string;
        payload?: unknown;
        timestamp?: string;
      }>;
    };
    try {
      const client = await getClient();
      const validEvents = Array.isArray(events) ? events : [];
      if (validEvents.length > 0) {
        await insertEvents(client, runId, validEvents.map((event) => ({
          step: typeof event?.step === "number" ? event.step : null,
          level: event?.level,
          kind: event?.kind ?? "generic",
          message: event?.message ?? "event",
          payload: event?.payload ?? null,
          createdAt: event?.timestamp ?? null,
        })));
        let latestStep: number | null = null;
        for (const event of validEvents) {
          if (typeof event.step === "number" && Number.isFinite(event.step)) {
            latestStep = latestStep == null ? event.step : Math.max(latestStep, event.step);
          }
        }
        if (latestStep != null) {
          await updateRunProgress(client, runId, {
            latest_step: latestStep,
            status: "active",
          });
        }
      }
    } catch (e) {
      console.warn("Ingest events DB error:", (e as Error).message);
    }
    broadcastLive("events", { runId, events: Array.isArray(events) ? events : [] });
    return jsonResponse({ ok: true });
  }

  return jsonResponse({ error: "Unknown ingest type" }, 400);
}
