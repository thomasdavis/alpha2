import * as fs from "node:fs";
import * as path from "node:path";
import * as zlib from "node:zlib";
import { resetEngine, initEngine } from "@/lib/engine";
import { OUTPUTS_DIR, checkAuth, invalidateModelsCache, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const contentType = request.headers.get("content-type") || "";

  let name: string;
  let config: Record<string, unknown>;
  let step: number;
  let metricsStr: string | undefined;
  let trainingDataStr: string | undefined;
  let checkpointBuf: Buffer;

  if (contentType === "application/octet-stream") {
    // Binary checkpoint upload — metadata in headers
    name = request.headers.get("x-checkpoint-name") as string;
    step = parseInt(request.headers.get("x-checkpoint-step") as string, 10);
    const configB64 = request.headers.get("x-checkpoint-config") as string;
    const metricsB64 = request.headers.get("x-checkpoint-metrics") as string;
    const dataB64 = request.headers.get("x-checkpoint-trainingdata") as string;

    config = configB64 ? JSON.parse(Buffer.from(configB64, "base64").toString("utf-8")) : {};
    metricsStr = metricsB64 ? Buffer.from(metricsB64, "base64").toString("utf-8") : undefined;
    trainingDataStr = dataB64 ? Buffer.from(dataB64, "base64").toString("utf-8") : undefined;

    const raw = await request.arrayBuffer();
    let buf = Buffer.from(raw);
    if (request.headers.get("content-encoding") === "gzip") {
      buf = zlib.gunzipSync(buf);
    }
    checkpointBuf = buf;

    if (!name || !step) {
      return jsonResponse({ error: "Missing X-Checkpoint-Name or X-Checkpoint-Step headers" }, 400);
    }
  } else {
    // Legacy JSON checkpoint upload
    const body = await request.json();
    name = body.name;
    config = body.config;
    step = body.step;
    metricsStr = body.metrics;
    trainingDataStr = body.trainingData;

    const checkpoint = body.checkpoint;
    if (!name || !config || !checkpoint || !step) {
      return jsonResponse({ error: "Missing required fields: name, config, checkpoint, step" }, 400);
    }

    const ckpt = checkpoint as Record<string, unknown>;
    if (!ckpt.modelConfig || !ckpt.params) {
      return jsonResponse({ error: "Invalid checkpoint: missing modelConfig or params" }, 400);
    }

    checkpointBuf = Buffer.from(JSON.stringify(checkpoint));
  }

  const runDir = path.join(OUTPUTS_DIR, name);
  fs.mkdirSync(runDir, { recursive: true });

  fs.writeFileSync(path.join(runDir, "config.json"), JSON.stringify(config, null, 2));
  fs.writeFileSync(path.join(runDir, `checkpoint-${step}.json`), checkpointBuf);
  if (metricsStr) {
    fs.writeFileSync(path.join(runDir, "metrics.jsonl"), metricsStr);
  }
  if (trainingDataStr) {
    fs.writeFileSync(path.join(runDir, "training-data.txt"), trainingDataStr);
  }

  resetEngine();
  await initEngine(OUTPUTS_DIR);
  invalidateModelsCache();

  console.log(`Uploaded model: ${name} (step ${step})`);
  return jsonResponse({ ok: true, name, step });
}
