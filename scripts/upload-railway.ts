#!/usr/bin/env npx tsx
/**
 * Upload model checkpoints to a Railway-hosted server.
 *
 * Usage:
 *   RAILWAY_URL=https://... UPLOAD_SECRET=... npx tsx scripts/upload-railway.ts [run-name]
 *
 * If run-name is provided, only that run is uploaded.
 * Otherwise, all runs in outputs/ are uploaded.
 *
 * For each run, POSTs to /api/upload with { name, config, checkpoint, step, metrics }.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import * as zlib from "node:zlib";

const OUTPUTS_DIR = path.resolve(new URL(".", import.meta.url).pathname, "../outputs");

interface RunMeta {
  name: string;
  step: number;
  configPath: string;
  checkpointPath: string;
  checkpointSize: number;
  metricsPath?: string;
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
    const stat = fs.statSync(ckptPath);
    const metricsPath = path.join(dirPath, "metrics.jsonl");

    results.push({
      name: entry.name,
      step: best.step,
      configPath,
      checkpointPath: ckptPath,
      checkpointSize: stat.size,
      metricsPath: fs.existsSync(metricsPath) ? metricsPath : undefined,
    });
  }

  return results;
}

async function uploadRun(baseUrl: string, secret: string, run: RunMeta): Promise<void> {
  const sizeMB = (run.checkpointSize / 1e6).toFixed(1);
  console.log(`\nUploading ${run.name} (step ${run.step}, ${sizeMB}MB)...`);

  const config = JSON.parse(fs.readFileSync(run.configPath, "utf-8"));
  const checkpoint = JSON.parse(fs.readFileSync(run.checkpointPath, "utf-8"));
  const metrics = run.metricsPath ? fs.readFileSync(run.metricsPath, "utf-8") : undefined;

  const json = JSON.stringify({ name: run.name, config, checkpoint, step: run.step, metrics });
  const compressed = zlib.gzipSync(Buffer.from(json));
  const compressedMB = (compressed.length / 1e6).toFixed(1);
  console.log(`  Compressed: ${compressedMB}MB`);

  const res = await fetch(`${baseUrl}/api/upload`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Encoding": "gzip",
      Authorization: `Bearer ${secret}`,
    },
    body: compressed,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  const result = await res.json();
  console.log(`  Done:`, result);
}

async function main() {
  const baseUrl = process.env.RAILWAY_URL;
  const secret = process.env.UPLOAD_SECRET;

  if (!baseUrl) {
    console.error("Error: RAILWAY_URL environment variable is required.");
    console.error("Example: RAILWAY_URL=https://alpha-server-production.up.railway.app");
    process.exit(1);
  }
  if (!secret) {
    console.error("Error: UPLOAD_SECRET environment variable is required.");
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

  console.log(`Uploading to: ${baseUrl}`);
  console.log(`Found ${runsToUpload.length} run(s) to upload:`);
  for (const r of runsToUpload) {
    const sizeMB = (r.checkpointSize / 1e6).toFixed(1);
    console.log(`  ${r.name} — step ${r.step}, ${sizeMB}MB`);
  }

  for (const run of runsToUpload) {
    await uploadRun(baseUrl, secret, run);
  }

  console.log("\nDone!");
}

main().catch((e) => { console.error(e); process.exit(1); });
