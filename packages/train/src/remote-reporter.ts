/**
 * Remote reporter — streams training metrics to a remote server via HTTP.
 *
 * Fire-and-forget: network errors never block training.
 */
import type { StepMetrics } from "./trainer.js";
import type { ModelConfig, TrainConfig } from "@alpha/core";

export interface RemoteReporterConfig {
  /** Base URL of the remote server (e.g. ALPHA_REMOTE_URL) */
  url: string;
  /** Bearer token for authentication */
  secret: string;
  /** How many metrics to buffer before flushing (default: 1 = every step) */
  batchSize?: number;
  /** Max time between flushes in ms (default: 5000) */
  flushInterval?: number;
}

export interface SampleGeneration {
  prompt: string;
  output: string;
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
  /** Send sample generations to the server for display */
  sendSamples(samples: SampleGeneration[]): Promise<void>;
}

/** Max chunk size for chunked uploads (512KB — well under Railway CDN limit) */
const CHUNK_SIZE = 512 * 1024;

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
  const pendingUploads: Promise<void>[] = [];

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

  /**
   * Upload a buffer to the server via chunked upload.
   * Splits data into CHUNK_SIZE pieces, sends each sequentially,
   * then calls assemble to combine them on the server volume.
   */
  async function chunkedUpload(
    data: Buffer,
    meta: {
      name: string;
      step: number;
      config: Record<string, unknown>;
      metrics?: string | null;
      trainingData?: string;
    },
  ): Promise<void> {
    const { gzipSync } = await import("node:zlib");
    const compressed = gzipSync(data);
    const totalChunks = Math.ceil(compressed.length / CHUNK_SIZE);
    const uploadId = `${meta.name}_${meta.step}_${Date.now()}`;

    // Send each chunk
    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, compressed.length);
      const chunk = compressed.subarray(start, end);

      const res = await fetch(`${url}/api/upload/chunk`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${secret}`,
          "Content-Type": "application/octet-stream",
          "X-Upload-Id": uploadId,
          "X-Chunk-Index": String(i),
          "X-Total-Chunks": String(totalChunks),
        },
        body: chunk,
      });
      if (!res.ok) {
        throw new Error(`Chunk ${i}/${totalChunks} failed: ${res.status}`);
      }
    }

    // Assemble chunks on the server
    const res = await fetch(`${url}/api/upload/assemble`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        uploadId,
        name: meta.name,
        step: meta.step,
        totalChunks,
        config: meta.config,
        metrics: meta.metrics || undefined,
        trainingData: meta.trainingData,
      }),
    });
    if (!res.ok) {
      throw new Error(`Assemble failed: ${res.status}`);
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
      // Wait for any in-flight checkpoint uploads before marking complete
      if (pendingUploads.length > 0) {
        console.log(`  waiting for ${pendingUploads.length} checkpoint upload(s)...`);
        await Promise.allSettled(pendingUploads);
      }
      await post("/api/ingest/complete", { runId, finalStep });
    },

    async flush() {
      await flushBuffer();
    },

    async sendSamples(samples: SampleGeneration[]) {
      await post("/api/ingest", { type: "samples", runId, samples });
    },

    uploadCheckpoint(info: { step: number; path: string; runId: string }) {
      // Async but tracked — complete() waits for all pending uploads
      const p = (async () => {
        try {
          const fs = await import("node:fs/promises");
          const nodePath = await import("node:path");

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

          // Use chunked upload to bypass Railway CDN body size limit
          await chunkedUpload(checkpointBuf, {
            name: info.runId,
            step: info.step,
            config,
            metrics: metricsRaw,
            trainingData,
          });

          console.log(`  checkpoint uploaded to ${url} (step ${info.step})`);
        } catch (e) {
          console.error(`  checkpoint upload failed: ${e}`);
        }
      })();
      pendingUploads.push(p);
    },
  };
}
