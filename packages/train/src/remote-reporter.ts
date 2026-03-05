/**
 * Remote reporter — streams training metrics to a remote server via HTTP.
 *
 * Fire-and-forget: network errors never block training.
 */
import type { SampleTrendStatus, StepMetrics, TrainingEvent } from "./trainer.js";
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
  /** Heartbeat interval in ms (default: 60_000) */
  heartbeatInterval?: number;
  /** Discord webhook URL for notifications */
  discordWebhook?: string;
}

export interface SampleGeneration {
  prompt: string;
  output: string;
}

export interface RemoteReporter {
  /** Run ID (set after registerRun) */
  readonly runId: string;
  /** Register a new training run with the server */
  registerRun(info: {
    runId: string;
    domain?: string;
    modelConfig: ModelConfig;
    trainConfig: TrainConfig;
    totalParams: number;
    dataPath?: string;
    infra?: { gpuName: string; gpuVendor: string; gpuVramMb: number; hostname: string; cpuCount: number; ramTotalMb: number; osPlatform: string };
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
  sendSamples(samples: SampleGeneration[], step?: number, trend?: SampleTrendStatus): Promise<void>;
  /** Send a single run event to the server event log */
  sendEvent(event: TrainingEvent): Promise<void>;
  /** Send a batch of run events to the server event log */
  sendEvents(events: TrainingEvent[]): Promise<void>;
}

/** Max chunk size for chunked uploads (1MB — safely under Railway CDN limit) */
const CHUNK_SIZE = 1024 * 1024;

/**
 * Strip optimizer state from a binary checkpoint to reduce upload size.
 * Binary format: [ALPH magic][4-byte header len][header JSON][tensor data...]
 * We keep only param tensors (p.*), dropping optimizer tensors (o.*).
 */
function stripOptimizerState(data: Buffer): Buffer {
  // Check magic
  if (data.length < 8 || data[0] !== 0x41 || data[1] !== 0x4c || data[2] !== 0x50 || data[3] !== 0x48) {
    // Not a binary checkpoint — return as-is (legacy JSON format)
    return data;
  }

  let offset = 4;
  const headerLen = data.readUInt32LE(offset); offset += 4;
  const header = JSON.parse(data.subarray(offset, offset + headerLen).toString("utf-8"));
  const dataStart = offset + headerLen;

  // Separate param tensors from optimizer tensors
  const paramTensors: { name: string; shape: number[]; elements: number }[] = [];
  const paramBuffers: Buffer[] = [];
  let tensorOffset = dataStart;

  for (const t of header.tensors) {
    const byteLen = t.elements * 4;
    if (t.name.startsWith("p.")) {
      paramTensors.push(t);
      paramBuffers.push(data.subarray(tensorOffset, tensorOffset + byteLen));
    }
    tensorOffset += byteLen;
  }

  // Rebuild header without optimizer tensors
  const newHeader = JSON.stringify({
    ...header,
    tensors: paramTensors,
    optimizerStep: 0,
  });
  const newHeaderBuf = Buffer.from(newHeader, "utf-8");

  // Rebuild file
  const dataSize = paramBuffers.reduce((acc, b) => acc + b.length, 0);
  const totalSize = 4 + 4 + newHeaderBuf.length + dataSize;
  const out = Buffer.alloc(totalSize);

  let pos = 0;
  Buffer.from("ALPH").copy(out, pos); pos += 4;
  out.writeUInt32LE(newHeaderBuf.length, pos); pos += 4;
  newHeaderBuf.copy(out, pos); pos += newHeaderBuf.length;
  for (const buf of paramBuffers) {
    buf.copy(out, pos);
    pos += buf.length;
  }

  return out;
}

async function sendDiscord(webhookUrl: string, embeds: Array<{
  title: string;
  color: number;
  description?: string;
  fields?: Array<{ name: string; value: string; inline?: boolean }>;
  timestamp?: string;
}>): Promise<void> {
  try {
    await fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embeds }),
      signal: AbortSignal.timeout(5000),
    });
  } catch { /* fire-and-forget */ }
}

export function createRemoteReporter(config: RemoteReporterConfig): RemoteReporter {
  const { url, secret, discordWebhook } = config;
  const batchSize = config.batchSize ?? 10;
  const flushInterval = config.flushInterval ?? 30_000;
  const heartbeatInterval = config.heartbeatInterval ?? 60_000;
  const MAX_BUFFERED_METRICS = 5_000;

  let runId = "";
  let domain: string | undefined;
  let storedModelConfig: ModelConfig | undefined;
  let storedTrainConfig: TrainConfig | undefined;
  let storedDataPath: string | undefined;
  let trainingDataSent = false;
  let buffer: StepMetrics[] = [];
  let timer: ReturnType<typeof setInterval> | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  let heartbeatInFlight = false;
  let flushInFlight = false;
  let lastStep = 0;
  const pendingUploads: Promise<void>[] = [];

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${secret}`,
  };

  let postErrorCount = 0;

  async function post(path: string, body: unknown): Promise<boolean> {
    try {
      const res = await fetch(`${url}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120_000), // 120s — event loop may be blocked by GPU training steps
      });
      if (!res.ok) {
        postErrorCount++;
        // Log first 5 errors then every 100th
        if (postErrorCount <= 5 || postErrorCount % 100 === 0) {
          const text = await res.text().catch(() => "");
          console.warn(`  [remote] POST ${path} failed: ${res.status} ${text.slice(0, 200)} (error #${postErrorCount})`);
        }
        return false;
      } else {
        // Reset on success
        if (postErrorCount > 0) {
          console.log(`  [remote] POST ${path} recovered after ${postErrorCount} errors`);
          postErrorCount = 0;
        }
        return true;
      }
    } catch (e) {
      postErrorCount++;
      if (postErrorCount <= 5 || postErrorCount % 100 === 0) {
        console.warn(`  [remote] POST ${path} error: ${(e as Error).message} (error #${postErrorCount})`);
      }
      return false;
    }
  }

  async function flushBuffer(): Promise<void> {
    if (flushInFlight || buffer.length === 0 || !runId) return;
    flushInFlight = true;
    const batch = buffer;
    buffer = [];
    try {
      const ok = await post("/api/ingest", { type: "metrics", runId, metrics: batch });
      if (!ok) {
        buffer = batch.concat(buffer);
        if (buffer.length > MAX_BUFFERED_METRICS) {
          const dropCount = buffer.length - MAX_BUFFERED_METRICS;
          buffer = buffer.slice(dropCount);
          if (postErrorCount <= 5 || postErrorCount % 100 === 0) {
            console.warn(`  [remote] dropped ${dropCount} buffered metric(s) after repeated ingest failures`);
          }
        }
      }
    } finally {
      flushInFlight = false;
    }

    // Drain quickly after recovery without waiting for the next timer tick.
    if (buffer.length >= batchSize) {
      void flushBuffer();
    }
  }

  function startTimer(): void {
    if (timer) return;
    timer = setInterval(() => {
      void flushBuffer();
    }, flushInterval);
    if (!heartbeatTimer) {
      heartbeatTimer = setInterval(() => {
        if (!runId || heartbeatInFlight) return;
        heartbeatInFlight = true;
        void post("/api/ingest", { type: "heartbeat", runId, step: lastStep })
          .finally(() => {
            heartbeatInFlight = false;
          });
      }, heartbeatInterval);
    }
  }

  function stopTimer(): void {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
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

    console.log(`  uploading checkpoint: ${(compressed.length / 1024 / 1024).toFixed(1)}MB compressed, ${totalChunks} chunks`);

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
        signal: AbortSignal.timeout(60_000), // 60s timeout per chunk
      });
      if (!res.ok) {
        throw new Error(`Chunk ${i}/${totalChunks} failed: ${res.status}`);
      }
      // Log progress every 50 chunks
      if ((i + 1) % 50 === 0 || i + 1 === totalChunks) {
        console.log(`  upload progress: ${i + 1}/${totalChunks} chunks`);
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
      signal: AbortSignal.timeout(120_000), // 120s — server needs time to reassemble large files
    });
    if (!res.ok) {
      throw new Error(`Assemble failed: ${res.status}`);
    }
  }

  return {
    get runId() { return runId; },
    async registerRun(info) {
      runId = info.runId;
      lastStep = 0;
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
        infra: info.infra,
      });
      startTimer();

      if (discordWebhook) {
        const mc = info.modelConfig;
        const tc = info.trainConfig;
        const paramStr = info.totalParams > 1e6
          ? `${(info.totalParams / 1e6).toFixed(1)}M`
          : `${(info.totalParams / 1e3).toFixed(0)}K`;
        const fields = [
          { name: "Run ID", value: `\`${info.runId}\``, inline: true },
          { name: "Domain", value: info.domain ?? "unknown", inline: true },
          { name: "Params", value: paramStr, inline: true },
          { name: "Model", value: `${mc.nEmbd}d ${mc.nHead}h ${mc.nLayer}L`, inline: true },
          { name: "Training", value: `batch=${tc.batchSize} lr=${tc.lr} iters=${tc.iters}`, inline: true },
          { name: "Backend", value: tc.backend, inline: true },
        ];
        if (info.infra) {
          fields.push({ name: "GPU", value: `${info.infra.gpuName} (${info.infra.gpuVendor})`, inline: true });
        }
        await sendDiscord(discordWebhook, [{
          title: "🚀 Training Started",
          color: 0x00c853,
          fields,
          timestamp: new Date().toISOString(),
        }]);
      }
    },

    onStep(metrics: StepMetrics) {
      if (Number.isFinite(metrics.step)) {
        lastStep = Math.max(lastStep, metrics.step);
      }
      buffer.push(metrics);
      if (buffer.length >= batchSize) {
        void flushBuffer();
      }
    },

    async complete(finalStep: number) {
      stopTimer();
      // Keep flushing until buffer drains, allowing one in-flight request at a time.
      while (buffer.length > 0 || flushInFlight) {
        await flushBuffer();
        if (flushInFlight) {
          await new Promise<void>((resolve) => setTimeout(resolve, 50));
        }
      }
      // Wait for any in-flight checkpoint uploads before marking complete
      if (pendingUploads.length > 0) {
        console.log(`  waiting for ${pendingUploads.length} checkpoint upload(s)...`);
        await Promise.allSettled(pendingUploads);
      }
      await post("/api/ingest/complete", { runId, finalStep });

      if (discordWebhook) {
        await sendDiscord(discordWebhook, [{
          title: "✅ Training Complete",
          color: 0x2196f3,
          fields: [
            { name: "Run ID", value: `\`${runId}\``, inline: true },
            { name: "Final Step", value: `${finalStep}`, inline: true },
            { name: "Domain", value: domain ?? "unknown", inline: true },
          ],
          timestamp: new Date().toISOString(),
        }]);
      }
    },

    async flush() {
      await flushBuffer();
    },

    async sendSamples(samples: SampleGeneration[], step?: number, trend?: SampleTrendStatus) {
      await post("/api/ingest", { type: "samples", runId, samples, step, trend });

      if (discordWebhook && samples.length > 0) {
        const sampleFields = samples.map((s, i) => ({
          name: `Prompt ${i + 1}: "${s.prompt}"`,
          value: `\`\`\`\n${s.output.slice(0, 300)}${s.output.length > 300 ? "..." : ""}\n\`\`\``,
        }));
        if (trend) {
          sampleFields.push({
            name: "Training Status",
            value: `${trend.summary}\nwindow ${trend.windowStartStep}->${trend.windowEndStep}`,
          });
        }
        await sendDiscord(discordWebhook, [{
          title: `📝 Inference Samples${step ? ` (Step ${step})` : ""}`,
          color: 0xff9800,
          description: `Run \`${runId}\``,
          fields: sampleFields,
          timestamp: new Date().toISOString(),
        }]);
      }
    },

    async sendEvent(event: TrainingEvent) {
      if (!runId) return;
      await post("/api/ingest", {
        type: "event",
        runId,
        event: {
          ...event,
          timestamp: event.timestamp ?? new Date().toISOString(),
        },
      });
    },

    async sendEvents(events: TrainingEvent[]) {
      if (!runId || events.length === 0) return;
      await post("/api/ingest", {
        type: "events",
        runId,
        events: events.map((event) => ({
          ...event,
          timestamp: event.timestamp ?? new Date().toISOString(),
        })),
      });
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

          // Strip optimizer state — inference only needs model params
          // This cuts upload size by ~2/3 (e.g. 1.3GB → 440MB for 115M model)
          const inferBuf = stripOptimizerState(checkpointBuf as unknown as Buffer);
          console.log(`  checkpoint: ${(checkpointBuf.length / 1024 / 1024).toFixed(1)}MB full → ${(inferBuf.length / 1024 / 1024).toFixed(1)}MB inference-only`);

          // Use chunked upload to bypass Railway CDN body size limit
          await chunkedUpload(inferBuf, {
            name: info.runId,
            step: info.step,
            config,
            metrics: metricsRaw,
            trainingData,
          });

          console.log(`  checkpoint uploaded to ${url} (step ${info.step})`);
          await post("/api/ingest", {
            type: "event",
            runId,
            event: {
              step: info.step,
              level: "info",
              kind: "checkpoint_uploaded",
              message: `checkpoint uploaded (step ${info.step})`,
              payload: {
                fullBytes: checkpointBuf.length,
                inferenceBytes: inferBuf.length,
              },
              timestamp: new Date().toISOString(),
            },
          });
        } catch (e) {
          console.error(`  checkpoint upload failed: ${e}`);
          await post("/api/ingest", {
            type: "event",
            runId,
            event: {
              step: info.step,
              level: "error",
              kind: "checkpoint_upload_failed",
              message: `checkpoint upload failed at step ${info.step}`,
              payload: { error: (e as Error).message ?? String(e) },
              timestamp: new Date().toISOString(),
            },
          });
        }
      })();
      pendingUploads.push(p);
    },
  };
}
