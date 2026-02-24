import * as fs from "node:fs";
import * as path from "node:path";
import * as zlib from "node:zlib";
import { resetEngine, initEngine } from "@/lib/engine";
import { OUTPUTS_DIR, checkAuth, invalidateModelsCache, pendingChunks, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const body = await request.json();
  const { uploadId, name, step, totalChunks, config, metrics, trainingData } = body;

  if (!uploadId || !name || !step || !totalChunks) {
    return jsonResponse({ error: "Missing required fields" }, 400);
  }

  const entry = pendingChunks.get(uploadId);
  if (!entry || entry.chunks.size < totalChunks) {
    const received = entry?.chunks.size ?? 0;
    return jsonResponse({ error: `Incomplete upload: ${received}/${totalChunks} chunks received` }, 400);
  }

  // Reassemble chunks in order
  const ordered: Buffer[] = [];
  for (let i = 0; i < totalChunks; i++) {
    const chunk = entry.chunks.get(i);
    if (!chunk) {
      return jsonResponse({ error: `Missing chunk ${i}` }, 400);
    }
    ordered.push(chunk);
  }
  pendingChunks.delete(uploadId);

  // Decompress the reassembled gzipped checkpoint
  const compressed = Buffer.concat(ordered);
  const checkpointBuf = zlib.gunzipSync(compressed);

  // Write to outputs directory
  const runDir = path.join(OUTPUTS_DIR, name);
  fs.mkdirSync(runDir, { recursive: true });

  fs.writeFileSync(path.join(runDir, "config.json"), JSON.stringify(config, null, 2));
  fs.writeFileSync(path.join(runDir, `checkpoint-${step}.json`), checkpointBuf);
  if (metrics) {
    fs.writeFileSync(path.join(runDir, "metrics.jsonl"), metrics);
  }
  if (trainingData) {
    fs.writeFileSync(path.join(runDir, "training-data.txt"), trainingData);
  }

  resetEngine();
  await initEngine(OUTPUTS_DIR);
  invalidateModelsCache();

  console.log(`Uploaded model (chunked): ${name} (step ${step}, ${totalChunks} chunks, ${compressed.length} bytes compressed)`);
  return jsonResponse({ ok: true, name, step });
}
