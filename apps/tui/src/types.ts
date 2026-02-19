// ── Config types (mirrored from @alpha/core to stay decoupled) ────────────

export interface ModelConfig {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly nLayer: number;
  readonly nEmbd: number;
  readonly nHead: number;
  readonly dropout: number;
}

export interface TrainConfig {
  readonly iters: number;
  readonly batchSize: number;
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly gradClip: number;
  readonly evalInterval: number;
  readonly evalIters: number;
  readonly seed: number;
  readonly backend: string;
  readonly tokenizer: string;
  readonly optimizer: string;
  readonly logLevel: string;
  readonly trace: boolean;
}

export interface RunConfig {
  modelConfig: ModelConfig;
  trainConfig: TrainConfig;
  configHash: string;
  runId: string;
  domain?: string;
}

// ── Metrics (matches trainer.ts StepMetrics) ──────────────────────────────

export interface MetricPoint {
  step: number;
  loss: number;
  valLoss?: number;
  lr: number;
  gradNorm: number;
  elapsed_ms: number;
  tokens_per_sec: number;
  ms_per_iter: number;
}

// ── Run state ─────────────────────────────────────────────────────────────

export type RunStatus = "active" | "completed" | "stale";

export interface RunState {
  name: string;
  dirPath: string;
  config: RunConfig;
  domain: string;
  metrics: MetricPoint[];
  checkpoints: string[];
  latestStep: number;
  totalIters: number;
  lastLoss: number | undefined;
  bestValLoss: number | undefined;
  estimatedParams: number;
  avgTokensPerSec: number;
  etaMs: number | undefined;
  status: RunStatus;
  mtime: number;
}

// ── View ──────────────────────────────────────────────────────────────────

export type Tab = "monitor" | "logs" | "runs" | "models";
export const TABS: Tab[] = ["monitor", "logs", "runs", "models"];
export const TAB_LABELS: Record<Tab, string> = {
  monitor: "Monitor",
  logs: "Logs",
  runs: "Runs",
  models: "Models",
};

export type ViewMode = "list" | "detail";

// ── Log file ──────────────────────────────────────────────────────────────

export interface LogFile {
  name: string;
  path: string;
  lines: string[];
  size: number;
  mtime: number;
}
