import * as fs from "node:fs";
import * as path from "node:path";
import type { MetricPoint } from "../types.js";
import { readMetricsFrom } from "./scanner.js";

export interface WatcherCallbacks {
  onMetrics: (runName: string, newPoints: MetricPoint[]) => void;
  onNewRun: (runName: string) => void;
}

interface FileWatch {
  watcher: fs.FSWatcher;
  byteOffset: number;
  debounceTimer: ReturnType<typeof setTimeout> | null;
}

export class RunWatcher {
  private outputsDir: string;
  private callbacks: WatcherCallbacks;
  private fileWatches = new Map<string, FileWatch>();
  private dirWatcher: fs.FSWatcher | null = null;
  private knownRuns = new Set<string>();

  constructor(outputsDir: string, callbacks: WatcherCallbacks) {
    this.outputsDir = outputsDir;
    this.callbacks = callbacks;
  }

  start(existingRuns: string[]): void {
    for (const name of existingRuns) {
      this.knownRuns.add(name);
      this.watchMetrics(name);
    }
    this.watchDirectory();
  }

  stop(): void {
    for (const [, fw] of this.fileWatches) {
      if (fw.debounceTimer) clearTimeout(fw.debounceTimer);
      fw.watcher.close();
    }
    this.fileWatches.clear();
    this.dirWatcher?.close();
    this.dirWatcher = null;
  }

  private watchMetrics(runName: string): void {
    const metricsPath = path.join(this.outputsDir, runName, "metrics.jsonl");
    if (!fs.existsSync(metricsPath)) return;
    if (this.fileWatches.has(runName)) return;

    // Start at current file size (we already read everything)
    const stat = fs.statSync(metricsPath);
    let byteOffset = stat.size;

    try {
      const watcher = fs.watch(metricsPath, () => {
        const fw = this.fileWatches.get(runName);
        if (!fw) return;
        if (fw.debounceTimer) clearTimeout(fw.debounceTimer);
        fw.debounceTimer = setTimeout(() => {
          const { points, newOffset } = readMetricsFrom(metricsPath, fw.byteOffset);
          if (points.length > 0) {
            fw.byteOffset = newOffset;
            this.callbacks.onMetrics(runName, points);
          }
        }, 100);
      });

      this.fileWatches.set(runName, { watcher, byteOffset, debounceTimer: null });
    } catch { /* file might vanish */ }
  }

  private watchDirectory(): void {
    if (!fs.existsSync(this.outputsDir)) return;

    try {
      this.dirWatcher = fs.watch(this.outputsDir, (_, filename) => {
        if (!filename) return;
        if (this.knownRuns.has(filename)) return;

        const dirPath = path.join(this.outputsDir, filename);
        const configPath = path.join(dirPath, "config.json");

        // Small delay to let config.json be written
        setTimeout(() => {
          if (fs.existsSync(configPath)) {
            this.knownRuns.add(filename);
            this.watchMetrics(filename);
            this.callbacks.onNewRun(filename);
          }
        }, 500);
      });
    } catch { /* dir might not exist yet */ }
  }
}
