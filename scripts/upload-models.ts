#!/usr/bin/env npx tsx
/**
 * Upload model checkpoints to Vercel Blob storage.
 *
 * Usage:
 *   BLOB_READ_WRITE_TOKEN=vercel_blob_... npx tsx scripts/upload-models.ts [run-name]
 *
 * If run-name is provided, only that run is uploaded.
 * Otherwise, all runs in outputs/ are uploaded.
 *
 * For each run, uploads:
 *   models/{name}/manifest.json  — metadata (config, step, loss, checkpoint URL)
 *   models/{name}/checkpoint.json — the highest-numbered checkpoint file
 *
 * The manifest points to the checkpoint blob URL so the serverless engine
 * can fetch it at runtime.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { put, list, del } from "@vercel/blob";

const OUTPUTS_DIR = path.resolve(new URL(".", import.meta.url).pathname, "../outputs");

interface RunMeta {
  id: string;
  name: string;
  step: number;
  checkpointPath: string;
  checkpointSize: number;
  config: Record<string, unknown>;
  lastLoss?: number;
}

function discoverRuns(filter?: string): RunMeta[] {
  const entries = fs.readdirSync(OUTPUTS_DIR, { withFileTypes: true });
  const results: RunMeta[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (filter && entry.name !== filter) continue;

    const dirPath = path.join(OUTPUTS_DIR, entry.name);
    const configPath = path.join(dirPath, "config.json");
    if (!fs.existsSync(configPath)) continue;

    const files = fs.readdirSync(dirPath);
    const checkpoints = files
      .filter((f) => /^checkpoint-\d+\.json$/.test(f))
      .map((f) => ({
        file: f,
        step: parseInt(f.match(/checkpoint-(\d+)\.json/)![1], 10),
      }))
      .sort((a, b) => b.step - a.step);

    if (checkpoints.length === 0) continue;

    const best = checkpoints[0];
    const ckptPath = path.join(dirPath, best.file);
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    const stat = fs.statSync(ckptPath);

    let lastLoss: number | undefined;
    const metricsPath = path.join(dirPath, "metrics.jsonl");
    if (fs.existsSync(metricsPath)) {
      const lines = fs.readFileSync(metricsPath, "utf-8").trim().split("\n");
      if (lines.length > 0) {
        try {
          const last = JSON.parse(lines[lines.length - 1]);
          lastLoss = last.valLoss ?? last.loss;
        } catch { /* ignore */ }
      }
    }

    results.push({
      id: entry.name,
      name: entry.name,
      step: best.step,
      checkpointPath: ckptPath,
      checkpointSize: stat.size,
      config,
      lastLoss,
    });
  }

  return results;
}

async function uploadRun(run: RunMeta): Promise<void> {
  const sizeMB = (run.checkpointSize / 1e6).toFixed(1);
  console.log(`\nUploading ${run.name} (step ${run.step}, ${sizeMB}MB)...`);

  // Upload checkpoint
  console.log(`  Uploading checkpoint...`);
  const ckptData = fs.readFileSync(run.checkpointPath);
  const ckptBlob = await put(`models/${run.name}/checkpoint.json`, ckptData, {
    access: "public",
    contentType: "application/json",
    addRandomSuffix: false,
    allowOverwrite: true,
  });
  console.log(`  Checkpoint: ${ckptBlob.url}`);

  // Upload manifest
  const domain = (run.config as any).domain ?? "novels";
  const manifest = {
    id: run.id,
    name: run.name,
    step: run.step,
    checkpointUrl: ckptBlob.url,
    config: run.config,
    lastLoss: run.lastLoss,
    domain,
    uploadedAt: new Date().toISOString(),
  };

  const manifestBlob = await put(
    `models/${run.name}/manifest.json`,
    JSON.stringify(manifest, null, 2),
    { access: "public", contentType: "application/json", addRandomSuffix: false, allowOverwrite: true },
  );
  console.log(`  Manifest:   ${manifestBlob.url}`);
}

async function main() {
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error("Error: BLOB_READ_WRITE_TOKEN environment variable is required.");
    console.error("Get it from: Vercel Dashboard → Project → Storage → Blob → Tokens");
    process.exit(1);
  }

  const filter = process.argv[2];
  const runsToUpload = discoverRuns(filter);

  if (runsToUpload.length === 0) {
    console.log(filter
      ? `No run found matching "${filter}" in ${OUTPUTS_DIR}`
      : `No runs found in ${OUTPUTS_DIR}`);
    process.exit(1);
  }

  console.log(`Found ${runsToUpload.length} run(s) to upload:`);
  for (const r of runsToUpload) {
    const sizeMB = (r.checkpointSize / 1e6).toFixed(1);
    console.log(`  ${r.name} — step ${r.step}, ${sizeMB}MB`);
  }

  for (const run of runsToUpload) {
    await uploadRun(run);
  }

  // List all models in blob storage
  console.log("\n--- All models in blob storage ---");
  const { blobs } = await list({ prefix: "models/" });
  const manifests = blobs.filter((b) => b.pathname.endsWith("/manifest.json"));
  for (const m of manifests) {
    const name = m.pathname.split("/")[1];
    console.log(`  ${name}: ${m.url}`);
  }

  console.log("\nDone!");
}

main().catch((e) => { console.error(e); process.exit(1); });
