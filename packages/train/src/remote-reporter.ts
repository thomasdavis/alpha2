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
  }): void;
  /** Buffer a step metric and flush if batch is full */
  onStep(metrics: StepMetrics): void;
  /** Mark run as completed and flush remaining metrics */
  complete(finalStep: number): Promise<void>;
  /** Flush buffered metrics immediately */
  flush(): Promise<void>;
}

export function createRemoteReporter(config: RemoteReporterConfig): RemoteReporter {
  const { url, secret } = config;
  const batchSize = config.batchSize ?? 1;
  const flushInterval = config.flushInterval ?? 5000;

  let runId = "";
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
  };
}
