/**
 * Shared inference engine.
 *
 * Scans the local filesystem (outputs/ directory) for model checkpoints.
 * The loaded model is cached in module scope so warm invocations skip loading.
 *
 * Uses @alpha/inference for fast CPU inference (KV cache, tiled matmul,
 * zero-allocation decode loop) — 10-20× faster than the autograd path.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { Effect } from "effect";
import type { ModelConfig, Tokenizer, CheckpointState } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import {
  type InferenceWeights,
  type InferenceSession,
  type InferenceModel,
  prepareInferenceWeights,
  createSession,
  prepareInferenceModel,
  resetCache,
  prefill,
  decodeStep,
  sampleFromLogits,
  countModelParams,
  SessionPool,
} from "@alpha/inference";
import { FileCheckpoint } from "@alpha/train";
import { BpeTokenizer, CharTokenizer, WordTokenizer } from "@alpha/tokenizers";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  LanguageModelV3FinishReason,
  LanguageModelV3Prompt,
} from "@ai-sdk/provider";

// ── Types ─────────────────────────────────────────────────────────────────

export interface RunInfo {
  id: string;
  name: string;
  /** Filesystem path to checkpoint */
  checkpoint: string;
  step: number;
  mtime: number;
  config: {
    modelConfig: ModelConfig;
    trainConfig: Record<string, unknown>;
    configHash: string;
    runId: string;
  };
  lastLoss?: number;
  source: "local";
  /** Model domain (e.g. "novels", "chords"). Defaults to "novels". */
  domain: string;
}

export interface LoadedModel {
  runId: string;
  config: ModelConfig;
  /** @deprecated Use weights + sessionPool instead. */
  inference: InferenceModel;
  weights: InferenceWeights;
  sessionPool: SessionPool;
  tokenizer: Tokenizer;
  paramCount: number;
}

// ── State (module-scoped, persists across warm invocations) ───────────────

let runs: RunInfo[] = [];
let loaded: LoadedModel | null = null;
let initialized = false;

export function getRuns(): RunInfo[] { return runs; }
export function getLoaded(): LoadedModel | null { return loaded; }
export function isInitialized(): boolean { return initialized; }

/** Clear cached state so the next initEngine() call rescans from disk. */
export function resetEngine(): void {
  runs = [];
  loaded = null;
  initialized = false;
}

// ── Scan local filesystem ─────────────────────────────────────────────────

export function scanLocalRuns(outputsDir: string): RunInfo[] {
  if (!fs.existsSync(outputsDir)) return [];

  const entries = fs.readdirSync(outputsDir, { withFileTypes: true });
  const results: RunInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dirPath = path.join(outputsDir, entry.name);
    const configPath = path.join(dirPath, "config.json");
    if (!fs.existsSync(configPath)) continue;

    const files = fs.readdirSync(dirPath);
    const checkpoints = files
      .filter((f) => /^checkpoint-\d+\.json$/.test(f))
      .map((f) => ({
        file: f,
        step: parseInt(f.match(/checkpoint-(\d+)\.json/)![1], 10),
      }))
      .sort((a, b) => b.step - a.step);

    if (checkpoints.length === 0) continue;

    const best = checkpoints[0];
    const stat = fs.statSync(path.join(dirPath, best.file));

    // Skip tiny/corrupt checkpoints (valid checkpoints are at least 1KB)
    if (stat.size < 1024) {
      console.warn(`Skipping ${entry.name}: checkpoint too small (${stat.size} bytes)`);
      continue;
    }

    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

    let lastLoss: number | undefined;
    const metricsPath = path.join(dirPath, "metrics.jsonl");
    if (fs.existsSync(metricsPath)) {
      const lines = fs.readFileSync(metricsPath, "utf-8").trim().split("\n");
      if (lines.length > 0) {
        try {
          const last = JSON.parse(lines[lines.length - 1]);
          lastLoss = last.valLoss ?? last.loss;
        } catch { /* ignore */ }
      }
    }

    results.push({
      id: entry.name,
      name: entry.name,
      checkpoint: path.join(dirPath, best.file),
      step: best.step,
      mtime: stat.mtimeMs,
      config,
      lastLoss,
      source: "local",
      domain: config.domain ?? "novels",
    });
  }

  return results.sort((a, b) => b.mtime - a.mtime);
}

// ── Initialize ────────────────────────────────────────────────────────────

export async function initEngine(outputsDir?: string): Promise<void> {
  if (initialized) return;

  if (outputsDir) {
    runs = scanLocalRuns(outputsDir);
  }

  console.log(`Found ${runs.length} run(s): ${runs.map((r) => r.name).join(", ")}`);
  initialized = true;
}

// ── Load model from checkpoint ────────────────────────────────────────────

function buildModel(state: CheckpointState, runId: string): LoadedModel {
  const weights = prepareInferenceWeights(state.modelConfig, state.params);
  const inference = prepareInferenceModel(state.modelConfig, state.params);
  const sessionPool = new SessionPool(weights);
  const paramCount = countModelParams(state.params);

  let tokenizer: Tokenizer;
  if (state.tokenizerArtifacts) {
    if (state.tokenizerArtifacts.type === "bpe") {
      const tok = new BpeTokenizer();
      tok.loadArtifacts(state.tokenizerArtifacts);
      tokenizer = tok;
    } else if (state.tokenizerArtifacts.type === "word") {
      const tok = new WordTokenizer();
      tok.loadArtifacts(state.tokenizerArtifacts);
      tokenizer = tok;
    } else {
      const tok = new CharTokenizer();
      tok.loadArtifacts(state.tokenizerArtifacts);
      tokenizer = tok;
    }
  } else {
    throw new Error("Checkpoint has no tokenizer artifacts");
  }

  return { runId, config: state.modelConfig, inference, weights, sessionPool, tokenizer, paramCount };
}

export async function ensureModel(modelId: string): Promise<LoadedModel> {
  if (loaded && loaded.runId === modelId) return loaded;

  const run = runs.find((r) => r.id === modelId || r.config?.runId === modelId);
  if (!run) throw new Error(`Unknown model: ${modelId}`);

  console.log(`Loading model: ${run.name}...`);
  const t0 = Date.now();

  const ckpt = new FileCheckpoint();
  const state = await Effect.runPromise(ckpt.load(run.checkpoint));

  loaded = buildModel(state, run.id);
  console.log(
    `Loaded ${run.name} (${(loaded.paramCount / 1e6).toFixed(2)}M params) in ${Date.now() - t0}ms`,
  );
  return loaded;
}

// ── Token generation ──────────────────────────────────────────────────────

export function* generateTokens(
  model: LoadedModel,
  prompt: string,
  steps: number,
  temperature: number,
  topk: number,
  topp = 1.0,
): Generator<string> {
  const { config, tokenizer, weights, sessionPool } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);
  const session = sessionPool.acquire();

  try {
    const allPromptTokens = tokenizer.encode(prompt);
    const maxPrompt = Math.max(1, config.blockSize - 1);
    const promptTokens = allPromptTokens.length > maxPrompt
      ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
      : allPromptTokens;

    yield tokenizer.decode(new Int32Array(promptTokens));

    let logits = prefill(weights, session, Int32Array.from(promptTokens));
    let currentPos = promptTokens.length;

    for (let i = 0; i < steps && currentPos < config.blockSize; i++) {
      const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
      const raw = tokenizer.decode(new Int32Array([tok]));
      const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";
      yield sep + raw;

      logits = decodeStep(weights, session, tok, currentPos);
      currentPos++;
    }
  } finally {
    sessionPool.release(session);
  }
}

// ── AI SDK provider ───────────────────────────────────────────────────────

function promptToText(prompt: LanguageModelV3Prompt): string {
  const parts: string[] = [];
  for (const msg of prompt) {
    if (msg.role === "system") {
      parts.push(msg.content);
    } else if (msg.role === "user" || msg.role === "assistant") {
      for (const part of msg.content) {
        if (part.type === "text") parts.push(part.text);
      }
    }
  }
  return parts.join("\n");
}

function makeUsage(inputTokens: number, outputTokens: number): LanguageModelV3Usage {
  return {
    inputTokens: { total: inputTokens, noCache: inputTokens, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: { total: outputTokens, text: outputTokens, reasoning: undefined },
  };
}

function makeFinish(reason: "stop" | "length"): LanguageModelV3FinishReason {
  return { unified: reason, raw: reason };
}

export class AlphaLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider = "alpha";
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private _steps: number;
  private _temperature: number;
  private _topk: number;
  private _topp: number;

  constructor(modelId: string, opts?: { steps?: number; temperature?: number; topk?: number; topp?: number }) {
    this.modelId = modelId;
    this._steps = opts?.steps ?? 200;
    this._temperature = opts?.temperature ?? 0.8;
    this._topk = opts?.topk ?? 40;
    this._topp = opts?.topp ?? 1.0;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const model = await ensureModel(this.modelId);
    const promptText = promptToText(options.prompt);
    const maxTokens = options.maxOutputTokens ?? this._steps;
    const temperature = options.temperature ?? this._temperature;
    const topk = options.topK ?? this._topk;
    const topp = (options as any).topP ?? this._topp;

    const { config, tokenizer, weights, sessionPool } = model;
    const rng = new SeededRng(Date.now() & 0xffffffff);
    const session = sessionPool.acquire();

    try {
      const allPromptTokens = tokenizer.encode(promptText);
      const maxPrompt = Math.max(1, config.blockSize - 1);
      const promptTokens = allPromptTokens.length > maxPrompt
        ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
        : allPromptTokens;

      let logits = prefill(weights, session, Int32Array.from(promptTokens));
      let currentPos = promptTokens.length;
      let outputTokenCount = 0;
      const generatedTokens: number[] = [];

      for (let i = 0; i < maxTokens && currentPos < config.blockSize; i++) {
        const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
        generatedTokens.push(tok);
        outputTokenCount++;

        logits = decodeStep(weights, session, tok, currentPos);
        currentPos++;
      }

      const generatedText = tokenizer.decode(new Int32Array(generatedTokens));

      return {
        content: [{ type: "text", text: generatedText }],
        finishReason: makeFinish(outputTokenCount >= maxTokens ? "length" : "stop"),
        usage: makeUsage(promptTokens.length, outputTokenCount),
        warnings: [],
      };
    } finally {
      sessionPool.release(session);
    }
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const model = await ensureModel(this.modelId);
    const promptText = promptToText(options.prompt);
    const maxTokens = options.maxOutputTokens ?? this._steps;
    const temperature = options.temperature ?? this._temperature;
    const topk = options.topK ?? this._topk;
    const topp = (options as any).topP ?? this._topp;
    const signal = options.abortSignal;

    const { config, tokenizer, weights, sessionPool } = model;
    const rng = new SeededRng(Date.now() & 0xffffffff);
    const session = sessionPool.acquire();

    const allPromptTokens = tokenizer.encode(promptText);
    const maxPrompt = Math.max(1, config.blockSize - 1);
    const promptTokens = allPromptTokens.length > maxPrompt
      ? allPromptTokens.slice(allPromptTokens.length - maxPrompt)
      : allPromptTokens;

    let logits = prefill(weights, session, Int32Array.from(promptTokens));
    let currentPos = promptTokens.length;
    const textId = "t0";

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start(controller) {
        controller.enqueue({ type: "stream-start", warnings: [] });
        controller.enqueue({ type: "text-start", id: textId });

        let outputCount = 0;

        function step() {
          if (signal?.aborted || outputCount >= maxTokens || currentPos >= config.blockSize) {
            controller.enqueue({ type: "text-end", id: textId });
            controller.enqueue({
              type: "finish",
              finishReason: makeFinish(outputCount >= maxTokens ? "length" : "stop"),
              usage: makeUsage(promptTokens.length, outputCount),
            });
            controller.close();
            sessionPool.release(session);
            return;
          }

          const tok = sampleFromLogits(session, logits, temperature, topk, rng, topp);
          outputCount++;

          const raw = tokenizer.decode(new Int32Array([tok]));
          const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";
          controller.enqueue({ type: "text-delta", id: textId, delta: sep + raw });

          logits = decodeStep(weights, session, tok, currentPos);
          currentPos++;

          setImmediate(step);
        }

        step();
      },
    });

    return { stream };
  }
}
