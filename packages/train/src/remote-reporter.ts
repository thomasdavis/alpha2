/**
 * Remote reporter — streams training metrics to a remote server via HTTP.
 *
 * Fire-and-forget: network errors never block training.
 */
import type { StepMetrics } from "./trainer.js";
import type { ModelConfig, TrainConfig } from "@alpha/core";

export interface RemoteReporterConfig {
  /** Base URL of the remote server (e.g. https://alpha.omegaai.dev) */
  url: string;
  /** Bearer token for authentication */
  secret: string;
  /** How many metrics to buffer before flushing (default: 1 = every step) */
  batchSize?: number;
  /** Max time between flushes in ms (default: 5000) */
  flushInterval?: number;
}

export interface RemoteReporter {
  /** Register a new training run with the server */
  registerRun(info: {
    runId: string;
    domain?: string;
    modelConfig: ModelConfig;
    trainConfig: TrainConfig;
    totalParams: number;
    dataPath?: string;
  }): void;
  /** Buffer a step metric and flush if batch is full */
  onStep(metrics: StepMetrics): void;
  /** Mark run as completed and flush remaining metrics */
  complete(finalStep: number): Promise<void>;
  /** Flush buffered metrics immediately */
  flush(): Promise<void>;
  /** Upload a checkpoint to the remote server for inference */
  uploadCheckpoint(info: { step: number; path: string; runId: string }): void;
}

export function createRemoteReporter(config: RemoteReporterConfig): RemoteReporter {
  const { url, secret } = config;
  const batchSize = config.batchSize ?? 1;
  const flushInterval = config.flushInterval ?? 5000;

  let runId = "";
  let domain: string | undefined;
  let storedModelConfig: ModelConfig | undefined;
  let storedTrainConfig: TrainConfig | undefined;
  let storedDataPath: string | undefined;
  let trainingDataSent = false;
  let buffer: StepMetrics[] = [];
  let timer: ReturnType<typeof setInterval> | null = null;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${secret}`,
  };

  async function post(path: string, body: unknown): Promise<void> {
    try {
      await fetch(`${url}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
    } catch {
      // Fire-and-forget — never block training
    }
  }

  async function flushBuffer(): Promise<void> {
    if (buffer.length === 0 || !runId) return;
    const batch = buffer;
    buffer = [];
    await post("/api/ingest", { type: "metrics", runId, metrics: batch });
  }

  function startTimer(): void {
    if (timer) return;
    timer = setInterval(() => {
      flushBuffer();
    }, flushInterval);
  }

  function stopTimer(): void {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  return {
    registerRun(info) {
      runId = info.runId;
      domain = info.domain;
      storedModelConfig = info.modelConfig;
      storedTrainConfig = info.trainConfig;
      storedDataPath = info.dataPath;
      post("/api/ingest", {
        type: "run_start",
        runId: info.runId,
        domain: info.domain,
        modelConfig: info.modelConfig,
        trainConfig: info.trainConfig,
        totalParams: info.totalParams,
      });
      startTimer();
    },

    onStep(metrics: StepMetrics) {
      buffer.push(metrics);
      if (buffer.length >= batchSize) {
        flushBuffer();
      }
    },

    async complete(finalStep: number) {
      stopTimer();
      await flushBuffer();
      await post("/api/ingest/complete", { runId, finalStep });
    },

    async flush() {
      await flushBuffer();
    },

    uploadCheckpoint(info: { step: number; path: string; runId: string }) {
      // Fire-and-forget — never block training
      (async () => {
        try {
          const fs = await import("node:fs/promises");
          const nodePath = await import("node:path");
          const { gzipSync } = await import("node:zlib");

          const runDir = nodePath.dirname(info.path);

          // Read checkpoint, config, and metrics files
          const [checkpointRaw, configRaw, metricsRaw] = await Promise.all([
            fs.readFile(info.path, "utf-8"),
            fs.readFile(nodePath.join(runDir, "config.json"), "utf-8").catch(() => null),
            fs.readFile(nodePath.join(runDir, "metrics.jsonl"), "utf-8").catch(() => null),
          ]);

          const config = configRaw ? JSON.parse(configRaw) : {
            modelConfig: storedModelConfig,
            trainConfig: storedTrainConfig,
            domain,
            runId: info.runId,
          };

          // On the first checkpoint upload, include training data sample
          let trainingData: string | undefined;
          if (!trainingDataSent && storedDataPath) {
            try {
              const raw = await fs.readFile(storedDataPath, "utf-8");
              trainingData = raw.slice(0, 50 * 1024); // Truncate to 50KB
              trainingDataSent = true;
            } catch {
              // Data file not readable — skip silently
            }
          }

          const body = JSON.stringify({
            name: info.runId,
            config,
            checkpoint: JSON.parse(checkpointRaw),
            step: info.step,
            metrics: metricsRaw || undefined,
            trainingData,
          });

          const compressed = gzipSync(body);

          await fetch(`${url}/api/upload`, {
            method: "POST",
            headers: {
              ...headers,
              "Content-Encoding": "gzip",
            },
            body: compressed,
          });

          console.log(`  checkpoint uploaded to ${url} (step ${info.step})`);
        } catch {
          // Fire-and-forget — never block training
        }
      })();
    },
  };
}
