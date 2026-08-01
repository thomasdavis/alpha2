import { createServer, type Server, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { SeededRng } from "@alpha/core";
import { createSession, decodeStep, prefill, sampleFromLogits } from "@alpha/inference";
import { AlphaLensAdapter } from "./adapter.js";
import { rankLogitRow, tensorReadout } from "./readout.js";
import { readLensSafetensors, type SafeTensorValue } from "./safetensors.js";
import type { ChatMessage } from "./types.js";

interface RuntimeManifest {
  readonly format: "blah-jacobian-lens";
  readonly format_version: 1;
  readonly model: { readonly repo_id: string; readonly revision: string; readonly weights_fingerprint: string; readonly tokenizer_fingerprint: string };
  readonly sites: readonly {
    readonly id: string;
    readonly logit_lens_supported: boolean;
    readonly transport: { readonly representation: "dense"; readonly tensor_key: string };
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

  private constructor(adapter: AlphaLensAdapter, manifest: RuntimeManifest, transports: ReadonlyMap<string, SafeTensorValue>) {
    this.adapter = adapter;
    this.manifest = manifest;
    this.transports = transports;
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
    for (const site of manifest.sites) {
      if (site.transport.representation !== "dense") throw new Error(`runtime does not support non-dense site ${site.id}`);
      const tensor = stored.tensors.get(site.transport.tensor_key);
      if (!tensor) throw new Error(`transport ${site.transport.tensor_key} for ${site.id} is missing`);
      transports.set(site.id, tensor);
    }
    return new AlphaLensRuntime(adapter, manifest, transports);
  }

  createServer(): Server {
    return createServer(async (request, response) => {
      try {
        if (request.method === "GET" && request.url === "/healthz") return json(response, 200, {
          status: "ok",
          protocol: "blah-lens-http/1",
          model_revision: this.manifest.model.revision,
        });
        if (request.method === "GET" && request.url === "/v1/lens/manifest") return json(response, 200, this.manifest);
        if (request.method === "POST" && request.url === "/v1/lens/analyze") {
          response.writeHead(200, { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store", Connection: "keep-alive" });
          let body: unknown;
          try { body = JSON.parse(await readRequest(request)); }
          catch (error) { return endError(response, "invalid_request", `invalid JSON: ${String(error)}`); }
          try { await this.analyze(body as AnalyzeRequest, (message) => response.write(JSON.stringify(message) + "\n")); }
          catch (error) { endError(response, classifyError(error), error instanceof Error ? error.message : String(error)); return; }
          response.end();
          return;
        }
        json(response, 404, { error: "not_found" });
      } catch (error) {
        if (!response.headersSent) json(response, 500, { error: "internal_error", message: error instanceof Error ? error.message : String(error) });
        else endError(response, "internal_error", error instanceof Error ? error.message : String(error));
      }
    });
  }

  async analyze(request: AnalyzeRequest, emit: (message: Record<string, unknown>) => void): Promise<void> {
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
    const promptCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
    let nextLogits = prefill(weights, session, promptIds, promptCapture);
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
      tokens: Array.from(promptIds, (id, position) => ({ position, id, text: this.adapter.tokenString(id) })),
    });

    const promptResults = this.calculateReadouts(
      promptCapture.sites,
      promptLength,
      sites,
      modes,
      declared,
      topK,
      pinned,
      request.filter_non_word_tokens ?? false,
    );
    for (let position = 0; position < promptLength; position++) emit({
      kind: "token",
      position,
      id: allIds[position],
      text: this.adapter.tokenString(allIds[position]),
      generated: false,
      results: promptResults[position],
    });

    const rng = new SeededRng(0x0b1a4);
    for (let step = 0; step < maxNewTokens; step++) {
      const next = sampleFromLogits(session, nextLogits, request.temperature ?? 0, topK, rng, 1);
      const position = allIds.length;
      allIds.push(next);
      const stepCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
      nextLogits = decodeStep(weights, session, next, position, stepCapture);
      const stepResults = this.calculateReadouts(
        stepCapture.sites,
        1,
        sites,
        modes,
        declared,
        topK,
        pinned,
        request.filter_non_word_tokens ?? false,
      );
      emit({
        kind: "token",
        position,
        id: next,
        text: this.adapter.tokenString(next),
        generated: true,
        results: stepResults[0],
      });
      if (next === eosId) break;
    }
    const generated = allIds.slice(promptLength).filter((id) => id !== eosId);
    emit({
      kind: "done",
      sequence_length: allIds.length,
      prompt_length: promptLength,
      completion: this.adapter.decode(generated),
    });
  }

  private calculateReadouts(
    captured: ReadonlyMap<string, Float32Array>,
    time: number,
    sites: readonly string[],
    modes: readonly ("jacobian" | "logit")[],
    declared: ReadonlyMap<string, RuntimeManifest["sites"][number]>,
    topK: number,
    pinned: readonly number[] | null,
    filterNonWordTokens: boolean,
  ): Array<Array<Record<string, unknown>>> {
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
        const logits = tensorReadout(this.adapter, tensor, mode === "jacobian" ? this.transports.get(siteId) : undefined);
        for (let position = 0; position < time; position++) {
          const ranked = rankLogitRow(logits.data, position * vocab, vocab, this.adapter, topK, pinned, filterNonWordTokens);
          results[position].push({
            mode,
            site: siteId,
            top: ranked.top,
            ...(ranked.pinned ? { pinned: ranked.pinned.map(({ id, logit, rank }) => ({ id, logit, rank })) } : {}),
          });
        }
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
