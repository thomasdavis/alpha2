import { useState, useEffect, useRef, useCallback } from "react";
import type { RunState, MetricPoint } from "../types.js";
import { scanRuns, readMetrics, buildRunState } from "../lib/scanner.js";
import { RunWatcher } from "../lib/watcher.js";
import * as fs from "node:fs";
import * as path from "node:path";

export function useRuns(outputsDir: string): {
  runs: RunState[];
  refresh: () => void;
} {
  const [runs, setRuns] = useState<RunState[]>(() => scanRuns(outputsDir));
  const watcherRef = useRef<RunWatcher | null>(null);

  const refresh = useCallback(() => {
    const scanned = scanRuns(outputsDir);
    setRuns(scanned);
    return scanned;
  }, [outputsDir]);

  useEffect(() => {
    const initial = refresh();

    const watcher = new RunWatcher(outputsDir, {
      onMetrics: (runName: string, newPoints: MetricPoint[]) => {
        setRuns(prev => prev.map(run => {
          if (run.name !== runName) return run;
          const updated = [...run.metrics, ...newPoints];
          return buildRunState(run.name, run.dirPath, run.config, updated, run.checkpoints);
        }));
      },
      onNewRun: (runName: string) => {
        const dirPath = path.join(outputsDir, runName);
        const configPath = path.join(dirPath, "config.json");
        try {
          const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
          const metrics = readMetrics(path.join(dirPath, "metrics.jsonl"));
          const run = buildRunState(runName, dirPath, config, metrics, []);
          setRuns(prev => {
            if (prev.some(r => r.name === runName)) return prev;
            return [run, ...prev];
          });
        } catch { /* skip */ }
      },
    });

    watcher.start(initial.map(r => r.name));
    watcherRef.current = watcher;

    return () => {
      watcher.stop();
      watcherRef.current = null;
    };
  }, [outputsDir, refresh]);

  return { runs, refresh };
}
