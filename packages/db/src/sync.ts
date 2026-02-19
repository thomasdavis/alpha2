/**
 * Sync training runs from disk into the database.
 *
 * Replicates the scanning logic from apps/tui/src/lib/scanner.ts but
 * writes into SQLite instead of building in-memory RunState objects.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import type { Client } from "@libsql/client";
import type { ModelConfig } from "@alpha/core";
import { upsertRun } from "./runs.js";
import { insertMetrics, getMaxStep } from "./metrics.js";
import { upsertCheckpoint } from "./checkpoints.js";
import { updateRunProgress } from "./runs.js";
import type { RunStatus } from "./types.js";

export interface SyncResult {
  runsScanned: number;
  runsUpserted: number;
  metricsInserted: number;
  checkpointsUpserted: number;
  errors: Array<{ run: string; error: string }>;
}

/** Estimate parameter count from model config (matches TUI formula). */
function estimateParams(mc: ModelConfig): number {
  const E = mc.nEmbd, L = mc.nLayer, V = mc.vocabSize, B = mc.blockSize;
  return V * E + B * E + L * (4 * E * E + 4 * E + 4 * 4 * E * E + 4 * E + 2 * E) + 2 * E + V * E;
}

/** Detect run status from step progress and filesystem mtime. */
function detectStatus(latestStep: number, totalIters: number, mtime: number): RunStatus {
  if (latestStep >= totalIters) return "completed";
  if (Date.now() - mtime < 60_000) return "active";
  return "stale";
}

interface RunConfig {
  modelConfig: ModelConfig;
  trainConfig: Record<string, unknown>;
  configHash: string;
  runId: string;
  domain?: string;
}

interface MetricPoint {
  step: number;
  loss: number;
  valLoss?: number;
  lr: number;
  gradNorm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
}

function readConfig(configPath: string): RunConfig | null {
  try {
    return JSON.parse(fs.readFileSync(configPath, "utf-8"));
  } catch {
    return null;
  }
}

function readMetrics(metricsPath: string): MetricPoint[] {
  if (!fs.existsSync(metricsPath)) return [];
  const content = fs.readFileSync(metricsPath, "utf-8").trim();
  if (!content) return [];

  const points: MetricPoint[] = [];
  for (const line of content.split("\n")) {
    try {
      points.push(JSON.parse(line));
    } catch { /* skip malformed */ }
  }
  return points;
}

function findCheckpoints(dirPath: string): Array<{ filename: string; step: number; size: number }> {
  const files = fs.readdirSync(dirPath);
  return files
    .filter((f) => /^checkpoint-\d+\.json$/.test(f))
    .map((f) => {
      const step = parseInt(f.match(/checkpoint-(\d+)\.json/)![1], 10);
      let size = 0;
      try {
        size = fs.statSync(path.join(dirPath, f)).size;
      } catch { /* ignore */ }
      return { filename: f, step, size };
    })
    .sort((a, b) => b.step - a.step);
}

/** Sync a single run directory into the database. */
export async function syncRun(
  client: Client,
  outputsDir: string,
  runName: string
): Promise<{ metricsInserted: number; checkpointsUpserted: number }> {
  const dirPath = path.join(outputsDir, runName);
  const configPath = path.join(dirPath, "config.json");

  const config = readConfig(configPath);
  if (!config) throw new Error(`Cannot read config: ${configPath}`);

  const metricsPath = path.join(dirPath, "metrics.jsonl");
  const allMetrics = readMetrics(metricsPath);
  const checkpoints = findCheckpoints(dirPath);

  // Get mtime from metrics file or config
  let mtime = 0;
  try {
    if (fs.existsSync(metricsPath)) {
      mtime = fs.statSync(metricsPath).mtimeMs;
    }
  } catch { /* ignore */ }

  const last = allMetrics.length > 0 ? allMetrics[allMetrics.length - 1] : undefined;
  const latestStep = last?.step ?? 0;
  const totalIters = (config.trainConfig as any).iters ?? 0;

  const valLosses = allMetrics.filter((m) => m.valLoss != null).map((m) => m.valLoss!);
  const bestValLoss = valLosses.length > 0 ? Math.min(...valLosses) : null;

  const status = detectStatus(latestStep, totalIters, mtime);

  // Upsert the run
  await upsertRun(client, {
    id: runName,
    run_id: config.runId,
    config_hash: config.configHash,
    domain: config.domain ?? "novels",
    model_config: config.modelConfig as any,
    train_config: config.trainConfig,
    status,
    latest_step: latestStep,
    last_loss: last?.loss ?? null,
    best_val_loss: bestValLoss,
    estimated_params: estimateParams(config.modelConfig),
    disk_mtime: mtime,
  });

  // Incremental metrics insert: only insert metrics after max step already in DB
  const maxStepInDb = await getMaxStep(client, runName);
  const newMetrics = allMetrics.filter((m) => m.step > maxStepInDb);
  let metricsInserted = 0;
  if (newMetrics.length > 0) {
    metricsInserted = await insertMetrics(client, runName, newMetrics);
  }

  // Update run progress after metrics insert
  await updateRunProgress(client, runName, {
    latest_step: latestStep,
    last_loss: last?.loss ?? null,
    best_val_loss: bestValLoss,
    status,
  });

  // Upsert checkpoints
  let checkpointsUpserted = 0;
  for (const cp of checkpoints) {
    await upsertCheckpoint(client, {
      run_id: runName,
      step: cp.step,
      filename: cp.filename,
      file_path: path.join(dirPath, cp.filename),
      file_size: cp.size,
    });
    checkpointsUpserted++;
  }

  return { metricsInserted, checkpointsUpserted };
}

/** Full sync: scan all run directories in outputsDir. */
export async function syncFromDisk(
  client: Client,
  outputsDir: string
): Promise<SyncResult> {
  const result: SyncResult = {
    runsScanned: 0,
    runsUpserted: 0,
    metricsInserted: 0,
    checkpointsUpserted: 0,
    errors: [],
  };

  if (!fs.existsSync(outputsDir)) return result;

  const entries = fs.readdirSync(outputsDir, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const configPath = path.join(outputsDir, entry.name, "config.json");
    if (!fs.existsSync(configPath)) continue;

    result.runsScanned++;

    try {
      const { metricsInserted, checkpointsUpserted } = await syncRun(
        client,
        outputsDir,
        entry.name
      );
      result.runsUpserted++;
      result.metricsInserted += metricsInserted;
      result.checkpointsUpserted += checkpointsUpserted;
    } catch (err) {
      result.errors.push({
        run: entry.name,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return result;
}
