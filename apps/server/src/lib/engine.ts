/**
 * Shared inference engine.
 *
 * Supports two storage backends:
 *   1. Local filesystem (dev) — scans outputs/ directory
 *   2. Vercel Blob (production) — fetches checkpoints from blob store
 *
 * The loaded model is cached in module scope so warm function invocations skip loading.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { Effect } from "effect";
import type { ModelConfig, TensorData, Tokenizer, CheckpointState } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import { CpuRefBackend } from "@alpha/tensor";
import { Tape } from "@alpha/autograd";
import type { GPTParams } from "@alpha/model";
import { initGPT, gptForward, countParams } from "@alpha/model";
import { FileCheckpoint, restoreParams } from "@alpha/train";
import { BpeTokenizer, CharTokenizer } from "@alpha/tokenizers";
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
  /** Filesystem path (local) or blob URL (Vercel) */
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
  /** "local" or "blob" */
  source: "local" | "blob";
}

export interface LoadedModel {
  runId: string;
  config: ModelConfig;
  params: GPTParams;
  tokenizer: Tokenizer;
  backend: CpuRefBackend;
  paramCount: number;
}

// ── State (module-scoped, persists across warm invocations) ───────────────

let runs: RunInfo[] = [];
let loaded: LoadedModel | null = null;
let initialized = false;

export function getRuns(): RunInfo[] { return runs; }
export function getLoaded(): LoadedModel | null { return loaded; }
export function isInitialized(): boolean { return initialized; }

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
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    const stat = fs.statSync(path.join(dirPath, best.file));

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
    });
  }

  return results.sort((a, b) => b.mtime - a.mtime);
}

// ── Scan Vercel Blob storage ──────────────────────────────────────────────

export async function scanBlobRuns(): Promise<RunInfo[]> {
  const { list } = await import("@vercel/blob");

  // List all manifests: models/{name}/manifest.json
  const { blobs } = await list({ prefix: "models/", token: process.env.BLOB_READ_WRITE_TOKEN });

  const manifests = blobs.filter((b) => b.pathname.endsWith("/manifest.json"));
  const results: RunInfo[] = [];

  for (const manifest of manifests) {
    try {
      const res = await fetch(manifest.url);
      const data = await res.json() as {
        id: string;
        name: string;
        checkpointUrl: string;
        step: number;
        config: RunInfo["config"];
        lastLoss?: number;
        uploadedAt: string;
      };

      results.push({
        id: data.id,
        name: data.name,
        checkpoint: data.checkpointUrl,
        step: data.step,
        mtime: new Date(data.uploadedAt).getTime(),
        config: data.config,
        lastLoss: data.lastLoss,
        source: "blob",
      });
    } catch (e) {
      console.error(`Failed to read manifest ${manifest.pathname}:`, e);
    }
  }

  return results.sort((a, b) => b.mtime - a.mtime);
}

// ── Initialize (auto-detect local vs Vercel) ─────────────────────────────

export async function initEngine(outputsDir?: string): Promise<void> {
  if (initialized) return;

  if (process.env.BLOB_READ_WRITE_TOKEN) {
    console.log("Vercel Blob token found, scanning blob storage...");
    runs = await scanBlobRuns();
    // Also scan local if available (for hybrid)
    if (outputsDir) {
      const local = scanLocalRuns(outputsDir);
      const blobIds = new Set(runs.map((r) => r.id));
      for (const r of local) {
        if (!blobIds.has(r.id)) runs.push(r);
      }
      runs.sort((a, b) => b.mtime - a.mtime);
    }
  } else if (outputsDir) {
    runs = scanLocalRuns(outputsDir);
  }

  console.log(`Found ${runs.length} run(s): ${runs.map((r) => `${r.name} [${r.source}]`).join(", ")}`);
  initialized = true;
}

// ── Load model from checkpoint ────────────────────────────────────────────

function buildModel(state: CheckpointState, runId: string): LoadedModel {
  const backend = new CpuRefBackend();
  const rng = new SeededRng(state.rngState);
  const params = initGPT(state.modelConfig, backend, rng);
  restoreParams(params, state.params);

  let tokenizer: Tokenizer;
  if (state.tokenizerArtifacts) {
    if (state.tokenizerArtifacts.type === "bpe") {
      const tok = new BpeTokenizer();
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

  const paramCount = countParams(params);
  return { runId, config: state.modelConfig, params, tokenizer, backend, paramCount };
}

async function loadFromBlob(url: string): Promise<CheckpointState> {
  console.log(`Fetching checkpoint from blob: ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Blob fetch failed: ${res.status}`);
  const parsed: any = await res.json();

  // Reconstruct typed arrays (same as FileCheckpoint.load)
  const buffers = new Map();
  if (parsed.optimizerState?.buffers) {
    for (const [k, v] of Object.entries(parsed.optimizerState.buffers) as [string, any][]) {
      buffers.set(k, { shape: v.shape, dtype: "f32", data: new Float32Array(v.data) });
    }
  }

  return {
    modelConfig: parsed.modelConfig,
    params: parsed.params,
    optimizerState: { step: parsed.optimizerState?.step ?? 0, buffers },
    tokenizerArtifacts: parsed.tokenizerArtifacts,
    rngState: parsed.rngState,
    configHash: parsed.configHash,
    step: parsed.step,
  };
}

export async function ensureModel(modelId: string): Promise<LoadedModel> {
  if (loaded && loaded.runId === modelId) return loaded;

  const run = runs.find((r) => r.id === modelId);
  if (!run) throw new Error(`Unknown model: ${modelId}`);

  console.log(`Loading model: ${run.name} (source: ${run.source})...`);
  const t0 = Date.now();

  let state: CheckpointState;
  if (run.source === "blob") {
    state = await loadFromBlob(run.checkpoint);
  } else {
    const ckpt = new FileCheckpoint();
    state = await Effect.runPromise(ckpt.load(run.checkpoint));
  }

  loaded = buildModel(state, run.id);
  console.log(
    `Loaded ${run.name} (${(loaded.paramCount / 1e6).toFixed(2)}M params) in ${Date.now() - t0}ms`,
  );
  return loaded;
}

// ── Token generation ──────────────────────────────────────────────────────

export function sampleNextToken(
  model: LoadedModel,
  tokens: Int32Array,
  currentLen: number,
  temperature: number,
  topk: number,
  rng: SeededRng,
): number {
  const { config, params, backend } = model;
  const ctxStart = Math.max(0, currentLen - config.blockSize);
  const ctxLen = currentLen - ctxStart;
  const ctx = tokens.slice(ctxStart, ctxStart + ctxLen);

  const inputData: TensorData = {
    shape: [1, ctxLen],
    dtype: "i32",
    data: new Int32Array(ctx),
  };

  const tape = new Tape();
  const { logits } = gptForward(config, params, backend, tape, inputData);

  const vocabSize = config.vocabSize;
  const lastLogits = new Float32Array(vocabSize);
  const logitsArr = logits.data.data as Float32Array;
  const offset = (ctxLen - 1) * vocabSize;
  for (let v = 0; v < vocabSize; v++) {
    lastLogits[v] = logitsArr[offset + v] / temperature;
  }

  if (topk > 0 && topk < vocabSize) {
    const indexed = Array.from(lastLogits).map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);
    const threshold = indexed[topk - 1].val;
    for (let v = 0; v < vocabSize; v++) {
      if (lastLogits[v] < threshold) lastLogits[v] = -Infinity;
    }
  }

  let maxVal = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    if (lastLogits[v] > maxVal) maxVal = lastLogits[v];
  }
  let sumExp = 0;
  const probs = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    probs[v] = Math.exp(lastLogits[v] - maxVal);
    sumExp += probs[v];
  }
  for (let v = 0; v < vocabSize; v++) {
    probs[v] /= sumExp;
  }

  const r = rng.next();
  let cumsum = 0;
  let nextToken = 0;
  for (let v = 0; v < vocabSize; v++) {
    cumsum += probs[v];
    if (r < cumsum) { nextToken = v; break; }
  }
  return nextToken;
}

export function* generateTokens(
  model: LoadedModel,
  prompt: string,
  steps: number,
  temperature: number,
  topk: number,
): Generator<string> {
  const { config, tokenizer } = model;
  const rng = new SeededRng(Date.now() & 0xffffffff);

  const promptTokens = tokenizer.encode(prompt);
  const maxLen = Math.min(promptTokens.length + steps, config.blockSize);
  const tokens = new Int32Array(maxLen);
  tokens.set(promptTokens);
  let currentLen = promptTokens.length;

  yield tokenizer.decode(promptTokens);

  for (let i = 0; i < steps && currentLen < config.blockSize; i++) {
    const next = sampleNextToken(model, tokens, currentLen, temperature, topk, rng);
    tokens[currentLen] = next;
    currentLen++;
    yield tokenizer.decode(new Int32Array([next]));
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

  constructor(modelId: string, opts?: { steps?: number; temperature?: number; topk?: number }) {
    this.modelId = modelId;
    this._steps = opts?.steps ?? 200;
    this._temperature = opts?.temperature ?? 0.8;
    this._topk = opts?.topk ?? 40;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const model = await ensureModel(this.modelId);
    const promptText = promptToText(options.prompt);
    const maxTokens = options.maxOutputTokens ?? this._steps;
    const temperature = options.temperature ?? this._temperature;
    const topk = options.topK ?? this._topk;

    const { config, tokenizer } = model;
    const rng = new SeededRng(Date.now() & 0xffffffff);
    const promptTokens = tokenizer.encode(promptText);
    const maxLen = Math.min(promptTokens.length + maxTokens, config.blockSize);
    const tokens = new Int32Array(maxLen);
    tokens.set(promptTokens);
    let currentLen = promptTokens.length;
    let outputTokenCount = 0;

    for (let i = 0; i < maxTokens && currentLen < config.blockSize; i++) {
      const next = sampleNextToken(model, tokens, currentLen, temperature, topk, rng);
      tokens[currentLen] = next;
      currentLen++;
      outputTokenCount++;
    }

    const generatedText = tokenizer.decode(tokens.slice(promptTokens.length, currentLen));

    return {
      content: [{ type: "text", text: generatedText }],
      finishReason: makeFinish(outputTokenCount >= maxTokens ? "length" : "stop"),
      usage: makeUsage(promptTokens.length, outputTokenCount),
      warnings: [],
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const model = await ensureModel(this.modelId);
    const promptText = promptToText(options.prompt);
    const maxTokens = options.maxOutputTokens ?? this._steps;
    const temperature = options.temperature ?? this._temperature;
    const topk = options.topK ?? this._topk;
    const signal = options.abortSignal;

    const { config, tokenizer } = model;
    const rng = new SeededRng(Date.now() & 0xffffffff);
    const promptTokens = tokenizer.encode(promptText);
    const maxLen = Math.min(promptTokens.length + maxTokens, config.blockSize);
    const allTokens = new Int32Array(maxLen);
    allTokens.set(promptTokens);
    let currentLen = promptTokens.length;
    const textId = "t0";

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start(controller) {
        controller.enqueue({ type: "stream-start", warnings: [] });
        controller.enqueue({ type: "text-start", id: textId });

        let outputCount = 0;

        function step() {
          if (signal?.aborted || outputCount >= maxTokens || currentLen >= config.blockSize) {
            controller.enqueue({ type: "text-end", id: textId });
            controller.enqueue({
              type: "finish",
              finishReason: makeFinish(outputCount >= maxTokens ? "length" : "stop"),
              usage: makeUsage(promptTokens.length, outputCount),
            });
            controller.close();
            return;
          }

          const next = sampleNextToken(model, allTokens, currentLen, temperature, topk, rng);
          allTokens[currentLen] = next;
          currentLen++;
          outputCount++;

          const decoded = tokenizer.decode(new Int32Array([next]));
          controller.enqueue({ type: "text-delta", id: textId, delta: decoded });

          setImmediate(step);
        }

        step();
      },
    });

    return { stream };
  }
}
