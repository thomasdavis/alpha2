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
  }): Promise<void>;
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
    async registerRun(info) {
      runId = info.runId;
      domain = info.domain;
      storedModelConfig = info.modelConfig;
      storedTrainConfig = info.trainConfig;
      storedDataPath = info.dataPath;
      await post("/api/ingest", {
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

          // Read checkpoint as raw buffer (works for both binary and JSON formats)
          const [checkpointBuf, configRaw, metricsRaw] = await Promise.all([
            fs.readFile(info.path),
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

          // Detect binary checkpoint (starts with "ALPH" magic)
          const isBinary = checkpointBuf.length >= 4 &&
            checkpointBuf[0] === 0x41 && checkpointBuf[1] === 0x4c &&
            checkpointBuf[2] === 0x50 && checkpointBuf[3] === 0x48;

          if (isBinary) {
            // Binary format: send checkpoint as gzipped binary, metadata in headers
            const compressed = gzipSync(checkpointBuf);

            await fetch(`${url}/api/upload`, {
              method: "POST",
              headers: {
                ...headers,
                "Content-Type": "application/octet-stream",
                "Content-Encoding": "gzip",
                "X-Checkpoint-Name": info.runId,
                "X-Checkpoint-Step": String(info.step),
                "X-Checkpoint-Config": Buffer.from(JSON.stringify(config)).toString("base64"),
                "X-Checkpoint-Metrics": metricsRaw ? Buffer.from(metricsRaw).toString("base64") : "",
                "X-Checkpoint-TrainingData": trainingData ? Buffer.from(trainingData).toString("base64") : "",
              },
              body: compressed,
            });
          } else {
            // Legacy JSON format: send as JSON body
            const body = JSON.stringify({
              name: info.runId,
              config,
              checkpoint: JSON.parse(checkpointBuf.toString("utf-8")),
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
          }

          console.log(`  checkpoint uploaded to ${url} (step ${info.step})`);
        } catch {
          // Fire-and-forget — never block training
        }
      })();
    },
  };
}
