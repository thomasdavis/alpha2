import { createServer, type Server, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { setImmediate as yieldToLoop } from "node:timers/promises";
import { SeededRng } from "@alpha/core";
import { createSession, decodeStep, prefill, sampleFromLogits } from "@alpha/inference";
import { AlphaLensAdapter } from "./adapter.js";
import { rankLogitRow, tensorReadout } from "./readout.js";
import { readLensSafetensors, type SafeTensorValue } from "./safetensors.js";
import type { ChatMessage } from "./types.js";

/**
 * The reference backend computes in synchronous JavaScript, so the runtime's
 * honesty depends on where it yields. Every unit of work below ends with a
 * checkpoint: the event loop turns, written NDJSON flushes to the socket (real
 * incremental delivery, which conformance now measures), /healthz answers
 * during an analysis, and a client that has gone away is noticed instead of
 * ground for to completion.
 */
class AnalysisCancelledError extends Error {
  constructor() {
    super("client disconnected before the analysis finished");
    this.name = "AnalysisCancelledError";
  }
}

/** Analyses queued beyond this are refused rather than silently piled up. */
const MAX_QUEUED_ANALYSES = 4;

/** One blah-core-http/1 scoring item; see the evals.blah.dev CORE docs. */
interface ScoreExample {
  readonly kind: "choice_set" | "suffix_set" | "greedy_pair";
  readonly prompts: readonly string[];
}

/** Shared token-span length: prefix ('left') or suffix ('right'). */
function commonSpanLength(sequences: readonly number[][], direction: "left" | "right"): number {
  const minLen = Math.min(...sequences.map((s) => s.length));
  for (let i = 0; i < minLen; i++) {
    const idx = direction === "left" ? i : -1 - i;
    const reference = sequences[0].at(idx);
    if (!sequences.every((s) => s.at(idx) === reference)) return i;
  }
  return minLen;
}

/** Cross-entropy of `target` under the logits row: logsumexp - logits[target]. */
function negLogSoftmax(logits: Float32Array, target: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - max);
  return max + Math.log(sumExp) - logits[target];
}

function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

/** Keep intermediaries alive across a long compute chunk. NDJSON framing
 * tolerates blank lines by design, so a heartbeat is protocol-invisible. */
const HEARTBEAT_INTERVAL_MS = 10_000;

interface RuntimeManifest {
  readonly format: "blah-jacobian-lens";
  readonly format_version: 1;
  readonly model: { readonly repo_id: string; readonly revision: string; readonly weights_fingerprint: string; readonly tokenizer_fingerprint: string };
  readonly lens: {
    readonly centering?: { readonly mode: "none" | "affine"; readonly target_mean_key?: string };
  };
  readonly sites: readonly {
    readonly id: string;
    readonly logit_lens_supported: boolean;
    readonly transport: { readonly representation: "dense"; readonly tensor_key: string; readonly source_mean_key?: string };
  }[];
}

interface AnalyzeRequest {
  readonly prompt?: string | null;
  readonly chat?: readonly ChatMessage[] | null;
  readonly input_token_ids?: readonly number[] | null;
  readonly max_new_tokens?: number;
  readonly temperature?: number;
  readonly top_k?: number;
  readonly modes?: readonly ("jacobian" | "logit")[];
  readonly sites?: readonly string[] | null;
  readonly filter_non_word_tokens?: boolean;
  readonly pinned_token_ids?: readonly number[] | null;
}

export interface LensRuntimeOptions {
  readonly checkpoint: string;
  readonly bundle: string;
  readonly backend?: string;
  readonly host?: string;
  readonly port?: number;
}

export class AlphaLensRuntime {
  readonly adapter: AlphaLensAdapter;
  readonly manifest: RuntimeManifest;
  readonly transports: ReadonlyMap<string, SafeTensorValue>;
  readonly sourceMeans: ReadonlyMap<string, SafeTensorValue>;
  readonly targetMean?: SafeTensorValue;

  private constructor(
    adapter: AlphaLensAdapter,
    manifest: RuntimeManifest,
    transports: ReadonlyMap<string, SafeTensorValue>,
    sourceMeans: ReadonlyMap<string, SafeTensorValue>,
    targetMean?: SafeTensorValue,
  ) {
    this.adapter = adapter;
    this.manifest = manifest;
    this.transports = transports;
    this.sourceMeans = sourceMeans;
    this.targetMean = targetMean;
  }

  static async load(options: LensRuntimeOptions): Promise<AlphaLensRuntime> {
    const adapter = await AlphaLensAdapter.load({ checkpoint: options.checkpoint, backend: options.backend, prepareInference: true });
    const manifest = JSON.parse(await readFile(join(options.bundle, "lens-manifest.json"), "utf8")) as RuntimeManifest;
    if (manifest.format !== "blah-jacobian-lens" || manifest.format_version !== 1) throw new Error("bundle is not blah-jacobian-lens v1");
    if (manifest.model.weights_fingerprint !== adapter.description.weightsFingerprint) {
      throw new Error(`fingerprint mismatch: bundle ${manifest.model.weights_fingerprint}, loaded ${adapter.description.weightsFingerprint}`);
    }
    if (manifest.model.tokenizer_fingerprint !== adapter.description.tokenizerFingerprint) {
      throw new Error("tokenizer fingerprint mismatch");
    }
    const stored = await readLensSafetensors(join(options.bundle, "transports.safetensors"));
    const transports = new Map<string, SafeTensorValue>();
    const sourceMeans = new Map<string, SafeTensorValue>();
    const affine = manifest.lens.centering?.mode === "affine";
    const targetMean = affine
      ? requiredStoredTensor(stored.tensors, manifest.lens.centering?.target_mean_key, "target mean")
      : undefined;
    for (const site of manifest.sites) {
      if (site.transport.representation !== "dense") throw new Error(`runtime does not support non-dense site ${site.id}`);
      const tensor = stored.tensors.get(site.transport.tensor_key);
      if (!tensor) throw new Error(`transport ${site.transport.tensor_key} for ${site.id} is missing`);
      transports.set(site.id, tensor);
      if (affine) {
        sourceMeans.set(
          site.id,
          requiredStoredTensor(stored.tensors, site.transport.source_mean_key, `source mean for ${site.id}`),
        );
      }
    }
    return new AlphaLensRuntime(adapter, manifest, transports, sourceMeans, targetMean);
  }

  /** Analyses run one at a time; the chain is the mutex. */
  private analysisChain: Promise<void> = Promise.resolve();
  private queuedAnalyses = 0;

  createServer(): Server {
    return createServer(async (request, response) => {
      try {
        if (request.method === "GET" && request.url === "/healthz") return json(response, 200, {
          status: "ok",
          protocol: "blah-lens-http/1",
          model_revision: this.manifest.model.revision,
        });
        if (request.method === "GET" && request.url === "/v1/lens/manifest") return json(response, 200, this.manifest);
        if (request.method === "POST" && request.url === "/v1/score") {
          let body: unknown;
          try { body = JSON.parse(await readRequest(request)); }
          catch (error) { return json(response, 400, { error: `invalid JSON: ${String(error)}` }); }

          if (this.queuedAnalyses >= MAX_QUEUED_ANALYSES) {
            console.log(`[lens] score refused: ${this.queuedAnalyses} jobs already queued`);
            return json(response, 503, { error: "runtime is saturated; try again shortly" });
          }
          this.queuedAnalyses++;
          const turn = this.analysisChain;
          let release: () => void = () => {};
          this.analysisChain = new Promise<void>((resolve) => { release = resolve; });
          const startedAt = Date.now();
          try {
            await turn;
            const cancelled = () => response.destroyed || response.writableEnded;
            if (cancelled()) {
              console.log("[lens] score skipped: client left before its turn");
              return;
            }
            const checkpoint = async () => {
              if (cancelled()) throw new AnalysisCancelledError();
              await yieldToLoop();
            };
            const examples = (body as { examples?: unknown[] }).examples;
            if (!Array.isArray(examples) || examples.length === 0 || examples.length > 64) {
              return json(response, 400, { error: "examples must be an array of 1..64 items" });
            }
            const results: Record<string, unknown>[] = [];
            for (const example of examples) {
              results.push(await this.scoreExample(example as ScoreExample, checkpoint));
            }
            console.log(`[lens] score done: ${examples.length} examples in ${Date.now() - startedAt}ms`);
            json(response, 200, { results });
          } catch (error) {
            if (error instanceof AnalysisCancelledError) {
              console.log(`[lens] score cancelled by client after ${Date.now() - startedAt}ms`);
              response.destroy();
              return;
            }
            console.log(`[lens] score failed: ${error instanceof Error ? error.message : String(error)}`);
            if (!response.headersSent) json(response, 400, { error: error instanceof Error ? error.message : String(error) });
          } finally {
            this.queuedAnalyses--;
            release();
          }
          return;
        }
        if (request.method === "POST" && request.url === "/v1/lens/analyze") {
          response.writeHead(200, { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store", Connection: "keep-alive" });
          let body: unknown;
          try { body = JSON.parse(await readRequest(request)); }
          catch (error) { return endError(response, "invalid_request", `invalid JSON: ${String(error)}`); }

          if (this.queuedAnalyses >= MAX_QUEUED_ANALYSES) {
            console.log(`[lens] analyze refused: ${this.queuedAnalyses} analyses already queued`);
            return endError(response, "model_unavailable", "runtime is saturated; try again shortly");
          }

          // One analysis at a time. Interleaving two synchronous computations
          // makes both miss every deadline; a queue keeps each one honest.
          this.queuedAnalyses++;
          const turn = this.analysisChain;
          let release: () => void = () => {};
          this.analysisChain = new Promise<void>((resolve) => { release = resolve; });
          const startedAt = Date.now();
          try {
            await turn;
            const cancelled = () => response.destroyed || response.writableEnded;
            if (cancelled()) {
              console.log("[lens] analyze skipped: client left before its turn");
              return;
            }
            let lastWriteAt = Date.now();
            const emit = (message: Record<string, unknown>) => {
              lastWriteAt = Date.now();
              response.write(JSON.stringify(message) + "\n");
            };
            const heartbeat = setInterval(() => {
              if (!cancelled() && Date.now() - lastWriteAt >= HEARTBEAT_INTERVAL_MS) response.write("\n");
            }, HEARTBEAT_INTERVAL_MS);
            try {
              const outcome = await this.analyze(body as AnalyzeRequest, emit, cancelled);
              console.log(`[lens] analyze done: ${outcome.positions} positions in ${Date.now() - startedAt}ms`);
            } catch (error) {
              if (error instanceof AnalysisCancelledError) {
                console.log(`[lens] analyze cancelled by client after ${Date.now() - startedAt}ms`);
                response.destroy();
                return;
              }
              console.log(`[lens] analyze failed: ${error instanceof Error ? error.message : String(error)}`);
              endError(response, classifyError(error), error instanceof Error ? error.message : String(error));
              return;
            } finally {
              clearInterval(heartbeat);
            }
            response.end();
          } finally {
            this.queuedAnalyses--;
            release();
          }
          return;
        }
        json(response, 404, { error: "not_found" });
      } catch (error) {
        if (!response.headersSent) json(response, 500, { error: "internal_error", message: error instanceof Error ? error.message : String(error) });
        else endError(response, "internal_error", error instanceof Error ? error.message : String(error));
      }
    });
  }

  /**
   * blah-core-http/1 scoring (the evals.blah.dev CORE benchmark). Answer spans
   * are found in TOKEN space per the protocol: common prefix across a choice
   * set, common suffix across a schema set, without/with prefix split for a
   * greedy pair. Every prompt is stepped through the native inference path one
   * token at a time with a checkpoint per position, sharing the analyze
   * queue's fairness and cancellation.
   */
  private async scoreExample(
    example: ScoreExample,
    checkpoint: () => Promise<void>,
  ): Promise<Record<string, unknown>> {
    const kind = example?.kind;
    const prompts = example?.prompts;
    if (!Array.isArray(prompts) || prompts.some((p) => typeof p !== "string" || p.length === 0)) {
      throw new Error("each example needs a prompts array of non-empty strings");
    }

    const vocab = this.adapter.tokenizerArtifacts.vocab;
    let bos = vocab.indexOf("<|bos|>");
    if (bos < 0) bos = vocab.indexOf("<|end_of_text|>");
    if (bos < 0) throw new Error("tokenizer has neither <|bos|> nor <|end_of_text|>");
    const tokens = prompts.map((p) => [bos, ...this.adapter.encode(p)]);

    if (kind === "choice_set" || kind === "suffix_set") {
      if (tokens.length < 2) throw new Error(`${kind} needs at least two prompts`);
      let starts: number[];
      const ends = tokens.map((t) => t.length);
      if (kind === "choice_set") {
        const prefix = commonSpanLength(tokens, "left");
        starts = tokens.map(() => prefix);
      } else {
        const suffix = commonSpanLength(tokens, "right");
        starts = ends.map((e) => e - suffix);
      }
      const meanLosses: number[] = [];
      for (let i = 0; i < tokens.length; i++) {
        const { meanLoss } = await this.scoreSequence(tokens[i], starts[i], ends[i], checkpoint);
        meanLosses.push(meanLoss);
      }
      return { kind, mean_losses: meanLosses };
    }

    if (kind === "greedy_pair") {
      if (tokens.length !== 2) throw new Error("greedy_pair needs exactly [without, with]");
      const [without, withCont] = tokens;
      const isPrefix =
        without.length < withCont.length && without.every((t, i) => withCont[i] === t);
      if (!isPrefix) throw new Error("prompt_without must be a strict token prefix of prompt_with");
      const { greedyMatch } = await this.scoreSequence(
        withCont, without.length, withCont.length, checkpoint,
      );
      return { kind, match: greedyMatch };
    }

    throw new Error(`unknown example kind: ${String(kind)}`);
  }

  /**
   * Teacher-forced pass over one sequence (cropped to the native context,
   * keeping the tail), returning the mean loss and greedy-match verdict over
   * the [start, end) continuation span.
   */
  private async scoreSequence(
    sequence: number[],
    startIn: number,
    endIn: number,
    checkpoint: () => Promise<void>,
  ): Promise<{ meanLoss: number; greedyMatch: boolean }> {
    const maxLen = this.adapter.description.blockSize;
    let tokens = sequence;
    let start = startIn;
    let end = endIn;
    if (tokens.length > maxLen) {
      const drop = tokens.length - maxLen;
      if (start - drop < 0) throw new Error("continuation span exceeds the native context length");
      tokens = tokens.slice(-maxLen);
      start = start - drop;
      end = end - drop;
    }

    const weights = this.adapter.inferenceWeights;
    if (!weights) throw new Error("native inference weights were not prepared");
    const session = createSession(weights);

    let lossSum = 0;
    let lossCount = 0;
    let greedyMatch = true;
    // logits after position p predict token p+1; spans start at >= 1 (BOS).
    let logits = prefill(weights, session, Int32Array.of(tokens[0]));
    for (let position = 1; position < tokens.length; position++) {
      const target = tokens[position];
      if (position >= start && position < end) {
        lossSum += negLogSoftmax(logits, target);
        lossCount++;
        if (argmax(logits) !== target) greedyMatch = false;
      }
      if (position < tokens.length - 1) {
        logits = decodeStep(weights, session, target, position);
      }
      await checkpoint();
    }

    return { meanLoss: lossCount > 0 ? lossSum / lossCount : Number.NaN, greedyMatch };
  }

  async analyze(
    request: AnalyzeRequest,
    emit: (message: Record<string, unknown>) => void,
    cancelled?: () => boolean,
  ): Promise<{ positions: number }> {
    // A checkpoint ends every unit of work: the loop turns (so writes flush and
    // /healthz answers) and a vanished client stops the computation.
    const checkpoint = async () => {
      if (cancelled?.()) throw new AnalysisCancelledError();
      await yieldToLoop();
    };
    const supplied = [request.prompt, request.chat, request.input_token_ids].filter((value) => value !== null && value !== undefined);
    if (supplied.length !== 1) throw new Error("supply exactly one of prompt, chat, or input_token_ids");
    const maxNewTokens = boundedInt(request.max_new_tokens ?? 64, 0, 256, "max_new_tokens");
    const topK = boundedInt(request.top_k ?? 8, 1, 64, "top_k");
    const modes = request.modes ?? ["jacobian", "logit"];
    if (modes.length === 0 || modes.some((mode) => mode !== "jacobian" && mode !== "logit")) throw new Error("unsupported readout mode");
    const declared = new Map(this.manifest.sites.map((site) => [site.id, site]));
    const sites = request.sites ?? this.manifest.sites.map((site) => site.id);
    for (const site of sites) if (!declared.has(site)) throw new Error(`unsupported site: ${site}`);
    const pinned = request.pinned_token_ids ?? null;
    if (pinned?.some((id) => !Number.isInteger(id) || id < 0 || id >= this.adapter.description.vocabularySize)) throw new Error("pinned_token_ids contains an out-of-vocabulary ID");

    let promptIds: Int32Array;
    if (request.prompt !== null && request.prompt !== undefined) {
      promptIds = this.adapter.encode(request.prompt);
    } else if (request.chat) {
      const formatted = this.adapter.formatChat(request.chat);
      promptIds = formatted.tokenIds;
    } else {
      promptIds = Int32Array.from(request.input_token_ids!);
    }
    if (promptIds.length === 0) throw new Error("prompt token sequence is empty");
    if (promptIds.length + maxNewTokens > this.adapter.description.blockSize) throw new Error("requested sequence exceeds native context length");

    const weights = this.adapter.inferenceWeights;
    if (!weights) throw new Error("native inference weights were not prepared");
    const session = createSession(weights);
    const requested = new Set(sites);
    const allIds = [...promptIds];
    const eosId = this.adapter.tokenizerArtifacts.vocab.indexOf("<|end_of_text|>");
    const promptLength = promptIds.length;
    emit({
      kind: "meta",
      protocol: "blah-lens-http/1",
      model_repo: this.manifest.model.repo_id,
      model_revision: this.manifest.model.revision,
      modes,
      sites,
      vocab_size: this.adapter.description.vocabularySize,
      prompt_length: promptLength,
      max_new_tokens: maxNewTokens,
    });
    emit({
      kind: "prompt",
      tokens: Array.from(promptIds, (id, position) => ({
        position,
        ...this.adapter.tokenDescriptor(id),
      })),
    });
    await checkpoint();

    // The prompt is processed one position at a time — prefill seeds the
    // session with the first token, decodeStep extends it exactly as
    // generation does — so each position's readout streams as it is computed
    // instead of arriving in one block after a long silence.
    const filterNonWord = request.filter_non_word_tokens ?? false;
    const firstCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
    let nextLogits = prefill(weights, session, promptIds.subarray(0, 1), firstCapture);
    await checkpoint();
    const emitPosition = async (
      capture: Map<string, Float32Array>,
      position: number,
      generated: boolean,
    ) => {
      const results = await this.calculateReadouts(
        capture, 1, sites, modes, declared, topK, pinned, filterNonWord, checkpoint,
      );
      emit({
        kind: "token",
        position,
        ...this.adapter.tokenDescriptor(allIds[position]),
        generated,
        results: results[0],
      });
      await checkpoint();
    };
    await emitPosition(firstCapture.sites, 0, false);
    for (let position = 1; position < promptLength; position++) {
      const capture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
      nextLogits = decodeStep(weights, session, promptIds[position], position, capture);
      await emitPosition(capture.sites, position, false);
    }

    const rng = new SeededRng(0x0b1a4);
    for (let step = 0; step < maxNewTokens; step++) {
      const next = sampleFromLogits(session, nextLogits, request.temperature ?? 0, topK, rng, 1);
      const position = allIds.length;
      allIds.push(next);
      const stepCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
      nextLogits = decodeStep(weights, session, next, position, stepCapture);
      await emitPosition(stepCapture.sites, position, true);
      if (next === eosId) break;
    }
    const generated = allIds.slice(promptLength).filter((id) => id !== eosId);
    emit({
      kind: "done",
      sequence_length: allIds.length,
      prompt_length: promptLength,
      completion: this.adapter.decode(generated),
    });
    return { positions: allIds.length };
  }

  private async calculateReadouts(
    captured: ReadonlyMap<string, Float32Array>,
    time: number,
    sites: readonly string[],
    modes: readonly ("jacobian" | "logit")[],
    declared: ReadonlyMap<string, RuntimeManifest["sites"][number]>,
    topK: number,
    pinned: readonly number[] | null,
    filterNonWordTokens: boolean,
    checkpoint: () => Promise<void>,
  ): Promise<Array<Array<Record<string, unknown>>>> {
    const results: Array<Array<Record<string, unknown>>> = Array.from({ length: time }, () => []);
    const width = this.adapter.description.targetSite.width;
    const vocab = this.adapter.description.vocabularySize;
    for (const siteId of sites) {
      const data = captured.get(siteId);
      if (!data) throw new Error(`native inference did not capture ${siteId}`);
      const tensor = { shape: [1, time, width], dtype: "f32" as const, data };
      const site = declared.get(siteId)!;
      for (const mode of modes) {
        if (mode === "logit" && !site.logit_lens_supported) continue;
        const transport = mode === "jacobian" ? this.transports.get(siteId) : undefined;
        const centering = transport && this.targetMean
          ? { sourceMean: this.sourceMeans.get(siteId)!, targetMean: this.targetMean }
          : undefined;
        const logits = tensorReadout(this.adapter, tensor, transport, centering);
        for (let position = 0; position < time; position++) {
          const ranked = rankLogitRow(logits.data, position * vocab, vocab, this.adapter, topK, pinned, filterNonWordTokens);
          results[position].push({
            mode,
            site: siteId,
            top: ranked.top,
            ...(ranked.pinned ? { pinned: ranked.pinned.map(({ id, logit, rank }) => ({ id, logit, rank })) } : {}),
          });
        }
        // Each site x mode is one dense [time, width] @ [width, vocab] pass —
        // the unit of work worth a turn of the loop.
        await checkpoint();
      }
    }
    return results;
  }
}

export async function serveLensRuntime(options: LensRuntimeOptions): Promise<Server> {
  const runtime = await AlphaLensRuntime.load(options);
  const server = runtime.createServer();
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 8000;
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolve());
  });
  return server;
}

function boundedInt(value: number, min: number, max: number, label: string): number {
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${label} must be an integer from ${min} to ${max}`);
  return value;
}

function requiredStoredTensor(
  tensors: ReadonlyMap<string, SafeTensorValue>,
  key: string | undefined,
  label: string,
): SafeTensorValue {
  if (!key) throw new Error(`manifest does not declare ${label} tensor key`);
  const tensor = tensors.get(key);
  if (!tensor) throw new Error(`${label} tensor ${key} is missing`);
  return tensor;
}

function classifyError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("unsupported site")) return "unsupported_site";
  if (message.includes("fingerprint")) return "fingerprint_mismatch";
  return "invalid_request";
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

function endError(response: ServerResponse, code: string, message: string): void {
  response.write(JSON.stringify({ kind: "error", code, message }) + "\n");
  response.end();
}

async function readRequest(request: NodeJS.ReadableStream): Promise<string> {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 1_000_000) throw new Error("request body is too large");
  }
  return body;
}
