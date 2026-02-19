import * as fs from "node:fs";
import * as path from "node:path";
import type { RunConfig, MetricPoint, RunState, RunStatus, LogFile } from "../types.js";
import { estimateParams } from "./format.js";

export function scanRuns(outputsDir: string): RunState[] {
  if (!fs.existsSync(outputsDir)) return [];

  const entries = fs.readdirSync(outputsDir, { withFileTypes: true });
  const results: RunState[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dirPath = path.join(outputsDir, entry.name);
    const configPath = path.join(dirPath, "config.json");
    if (!fs.existsSync(configPath)) continue;

    let config: RunConfig;
    try {
      config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch {
      continue;
    }

    const metrics = readMetrics(path.join(dirPath, "metrics.jsonl"));
    const checkpoints = findCheckpoints(dirPath);
    const run = buildRunState(entry.name, dirPath, config, metrics, checkpoints);
    results.push(run);
  }

  return results.sort((a, b) => b.mtime - a.mtime);
}

export function readMetrics(metricsPath: string): MetricPoint[] {
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

export function readMetricsFrom(metricsPath: string, byteOffset: number): { points: MetricPoint[]; newOffset: number } {
  if (!fs.existsSync(metricsPath)) return { points: [], newOffset: byteOffset };

  const stat = fs.statSync(metricsPath);
  if (stat.size <= byteOffset) return { points: [], newOffset: byteOffset };

  const fd = fs.openSync(metricsPath, "r");
  const buf = Buffer.alloc(stat.size - byteOffset);
  fs.readSync(fd, buf, 0, buf.length, byteOffset);
  fs.closeSync(fd);

  const text = buf.toString("utf-8").trim();
  if (!text) return { points: [], newOffset: stat.size };

  const points: MetricPoint[] = [];
  for (const line of text.split("\n")) {
    try {
      points.push(JSON.parse(line));
    } catch { /* skip */ }
  }

  return { points, newOffset: stat.size };
}

function findCheckpoints(dirPath: string): string[] {
  const files = fs.readdirSync(dirPath);
  return files
    .filter(f => /^checkpoint-\d+\.json$/.test(f))
    .sort((a, b) => {
      const stepA = parseInt(a.match(/checkpoint-(\d+)\.json/)![1], 10);
      const stepB = parseInt(b.match(/checkpoint-(\d+)\.json/)![1], 10);
      return stepB - stepA;
    });
}

export function buildRunState(
  name: string,
  dirPath: string,
  config: RunConfig,
  metrics: MetricPoint[],
  checkpoints: string[],
): RunState {
  const mc = config.modelConfig;
  const tc = config.trainConfig;
  const params = estimateParams(mc);

  const last = metrics.length > 0 ? metrics[metrics.length - 1] : undefined;
  const latestStep = last?.step ?? 0;

  const valLosses = metrics.filter(m => m.valLoss != null).map(m => m.valLoss!);
  const bestValLoss = valLosses.length > 0 ? Math.min(...valLosses) : undefined;

  const recent = metrics.slice(-10);
  const avgTps = recent.length > 0
    ? recent.reduce((s, m) => s + m.tokens_per_sec, 0) / recent.length
    : 0;

  let etaMs: number | undefined;
  if (avgTps > 0 && latestStep < tc.iters && last) {
    const remainingSteps = tc.iters - latestStep;
    const avgMsPerIter = recent.reduce((s, m) => s + m.ms_per_iter, 0) / recent.length;
    etaMs = remainingSteps * avgMsPerIter;
  }

  // Determine mtime from metrics file or config
  let mtime = 0;
  try {
    const metricsPath = path.join(dirPath, "metrics.jsonl");
    if (fs.existsSync(metricsPath)) {
      mtime = fs.statSync(metricsPath).mtimeMs;
    }
  } catch { /* ignore */ }

  const status: RunStatus =
    latestStep >= tc.iters ? "completed" :
    (Date.now() - mtime < 60_000) ? "active" :
    "stale";

  return {
    name,
    dirPath,
    config,
    domain: config.domain ?? "unknown",
    metrics,
    checkpoints,
    latestStep,
    totalIters: tc.iters,
    lastLoss: last?.loss,
    bestValLoss,
    estimatedParams: params,
    avgTokensPerSec: avgTps,
    etaMs,
    status,
    mtime,
  };
}

// ── Log file scanning ─────────────────────────────────────────────────────

export function scanLogs(outputsDir: string): LogFile[] {
  if (!fs.existsSync(outputsDir)) return [];

  const results: LogFile[] = [];
  const seen = new Set<string>();

  // Scan .log files at top level of outputs/
  const topEntries = fs.readdirSync(outputsDir);
  for (const name of topEntries) {
    if (!name.endsWith(".log")) continue;
    const filePath = path.join(outputsDir, name);
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) continue;
    addLogFile(results, seen, name, filePath, stat);
  }

  // Scan .log files inside run directories
  for (const name of topEntries) {
    const dirPath = path.join(outputsDir, name);
    try {
      const stat = fs.statSync(dirPath);
      if (!stat.isDirectory()) continue;
    } catch { continue; }

    const files = fs.readdirSync(dirPath);
    for (const f of files) {
      if (!f.endsWith(".log")) continue;
      const filePath = path.join(dirPath, f);
      const logName = `${name}/${f}`;
      const stat = fs.statSync(filePath);
      addLogFile(results, seen, logName, filePath, stat);
    }
  }

  return results.sort((a, b) => b.mtime - a.mtime);
}

function addLogFile(results: LogFile[], seen: Set<string>, name: string, filePath: string, stat: fs.Stats): void {
  if (seen.has(filePath)) return;
  seen.add(filePath);

  let lines: string[] = [];
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    lines = content.split("\n");
  } catch { /* skip */ }

  results.push({
    name,
    path: filePath,
    lines,
    size: stat.size,
    mtime: stat.mtimeMs,
  });
}
