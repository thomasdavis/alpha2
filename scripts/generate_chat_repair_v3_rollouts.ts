#!/usr/bin/env npx tsx

/**
 * Generate the immutable train-only RCR-UL rollout ledger.
 *
 * This deliberately uses Alpha's native cached inference path and greedy
 * decoding. It is resumable without regenerating completed rows, records an
 * audit digest and probability summary for every selected token, and refuses
 * any checkpoint, tokenizer, prompt-token, or candidate-order drift.
 */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open, readFile, rename, stat, writeFile } from "node:fs/promises";
import { endianness, platform, arch } from "node:os";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { Effect } from "effect";
import { FileCheckpoint, fourGramRepeatRate } from "@alpha/train";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import { SeededRng } from "@alpha/core";
import {
  prepareInferenceModel,
  resetCache,
  prefill,
  decodeStep,
  sampleFromLogits,
} from "@alpha/inference";

type RolloutCandidate = {
  schema: "alpha-rcr-ul-rollout-candidate-v1";
  stable_id: string;
  source: string;
  source_id: string;
  positive_conversation_sha256: string;
  positive_response: string;
  positive_response_sha256: string;
  prompt: string;
  prompt_sha256: string;
  prompt_tokens: number;
  prompt_token_ids: number[];
};

type TokenAudit = {
  token_id: number;
  chosen_logit: number;
  chosen_probability: number;
  runner_up_token_id: number;
  runner_up_logit: number;
  logsumexp: number;
  logits_f32_sha256: string;
};

type RawRollout = {
  schema: "alpha-rcr-ul-raw-rollout-v1";
  stable_id: string;
  source: string;
  source_id: string;
  positive_conversation_sha256: string;
  prompt_sha256: string;
  checkpoint_sha256: string;
  prompt_token_ids: number[];
  generated_token_ids: number[];
  content_token_ids: number[];
  token_audit: TokenAudit[];
  text: string;
  stop_reason: "learned_eos" | "role_boundary" | "max_tokens" | "block_limit";
  stop_token_id: number | null;
  eos_terminated: boolean;
  four_gram_repeat_rate: number;
  degenerate_loop: boolean;
  output_sha256: string;
};

const cli = new Map<string, string>();
for (const raw of process.argv.slice(2)) {
  const match = raw.match(/^--([^=]+)=(.*)$/s);
  if (!match) throw new Error(`expected --key=value, received ${raw}`);
  cli.set(match[1], match[2]);
}

function required(name: string): string {
  const value = cli.get(name);
  if (!value) throw new Error(`missing --${name}=...`);
  return value;
}

function integerArg(name: string, fallback: number): number {
  const value = cli.get(name);
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`--${name} must be a safe integer: ${value}`);
  return parsed;
}

function boolArg(name: string, fallback: boolean): boolean {
  const value = cli.get(name);
  if (value === undefined) return fallback;
  if (value === "true" || value === "1") return true;
  if (value === "false" || value === "0") return false;
  throw new Error(`--${name} must be true or false: ${value}`);
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, path);
}

async function readJsonl<T>(path: string): Promise<T[]> {
  const rows: T[] = [];
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let lineNumber = 0;
  for await (const line of lines) {
    lineNumber++;
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line) as T);
    } catch (error) {
      throw new Error(`${path}:${lineNumber} is not valid JSON`, { cause: error });
    }
  }
  return rows;
}

function validateHex64(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256`);
  }
}

function validateCandidates(rows: RolloutCandidate[]): void {
  if (rows.length === 0) throw new Error("rollout candidate file is empty");
  const identities = new Set<string>();
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const label = `candidate ${index + 1}`;
    if (row.schema !== "alpha-rcr-ul-rollout-candidate-v1") throw new Error(`${label} has unexpected schema`);
    validateHex64(row.stable_id, `${label} stable_id`);
    validateHex64(row.positive_conversation_sha256, `${label} positive_conversation_sha256`);
    validateHex64(row.prompt_sha256, `${label} prompt_sha256`);
    validateHex64(row.positive_response_sha256, `${label} positive_response_sha256`);
    if (row.stable_id !== row.positive_conversation_sha256) throw new Error(`${label} stable identity drift`);
    if (identities.has(row.stable_id)) throw new Error(`${label} duplicates ${row.stable_id}`);
    identities.add(row.stable_id);
    if (!row.prompt || sha256Text(row.prompt) !== row.prompt_sha256) throw new Error(`${label} prompt hash mismatch`);
    if (sha256Text(row.positive_response) !== row.positive_response_sha256) {
      throw new Error(`${label} positive response hash mismatch`);
    }
    if (!Array.isArray(row.prompt_token_ids) || row.prompt_token_ids.length !== row.prompt_tokens ||
        row.prompt_token_ids.some((token) => !Number.isSafeInteger(token) || token < 0)) {
      throw new Error(`${label} has invalid prompt_token_ids`);
    }
  }
}

function tokenAudit(logits: Float32Array, chosenToken: number): TokenAudit {
  let maximum = -Infinity;
  let runnerUp = -Infinity;
  let maximumId = -1;
  let runnerUpId = -1;
  for (let index = 0; index < logits.length; index++) {
    const value = logits[index];
    if (!Number.isFinite(value)) throw new Error(`non-finite inference logit at vocabulary index ${index}`);
    if (value > maximum) {
      runnerUp = maximum;
      runnerUpId = maximumId;
      maximum = value;
      maximumId = index;
    } else if (value > runnerUp) {
      runnerUp = value;
      runnerUpId = index;
    }
  }
  if (maximumId !== chosenToken) throw new Error(`greedy decoder selected ${chosenToken}, expected argmax ${maximumId}`);
  let expSum = 0;
  for (let index = 0; index < logits.length; index++) expSum += Math.exp(logits[index] - maximum);
  const logsumexp = maximum + Math.log(expSum);
  const bytes = Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength);
  return {
    token_id: chosenToken,
    chosen_logit: logits[chosenToken],
    chosen_probability: Math.exp(logits[chosenToken] - logsumexp),
    runner_up_token_id: runnerUpId,
    runner_up_logit: runnerUp,
    logsumexp,
    logits_f32_sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function validateExistingRows(rows: RawRollout[], candidates: RolloutCandidate[], checkpointSha256: string): void {
  if (rows.length > candidates.length) throw new Error("raw rollout file has more rows than the frozen candidate file");
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const candidate = candidates[index];
    if (row.schema !== "alpha-rcr-ul-raw-rollout-v1") throw new Error(`existing rollout ${index + 1} has unexpected schema`);
    if (row.stable_id !== candidate.stable_id || row.prompt_sha256 !== candidate.prompt_sha256 ||
        row.positive_conversation_sha256 !== candidate.positive_conversation_sha256) {
      throw new Error(`existing rollout ${index + 1} no longer matches frozen candidate order`);
    }
    if (row.checkpoint_sha256 !== checkpointSha256) throw new Error(`existing rollout ${index + 1} checkpoint drift`);
    if (row.token_audit.length !== row.generated_token_ids.length) throw new Error(`existing rollout ${index + 1} audit length mismatch`);
    if (sha256Text(JSON.stringify({
      prompt_token_ids: row.prompt_token_ids,
      generated_token_ids: row.generated_token_ids,
    })) !== row.output_sha256) throw new Error(`existing rollout ${index + 1} output hash mismatch`);
  }
}

const checkpointPath = resolve(required("checkpoint"));
const candidatesPath = resolve(required("candidates"));
const outDir = resolve(required("out-dir"));
const expectedCheckpointSha256 = required("expected-checkpoint-sha256");
validateHex64(expectedCheckpointSha256, "--expected-checkpoint-sha256");
const maxTokens = integerArg("max-tokens", 128);
const stopAfter = integerArg("stop-after", 0);
const resume = boolArg("resume", false);
if (maxTokens <= 0) throw new Error("--max-tokens must be positive");
if (stopAfter < 0) throw new Error("--stop-after must be non-negative");

await mkdir(outDir, { recursive: true });
const outputPath = join(outDir, "raw-rollouts.jsonl");
const progressPath = join(outDir, "progress.json");
const manifestPath = join(outDir, "rollout-manifest.json");
const checkpointSha256 = await sha256File(checkpointPath);
if (checkpointSha256 !== expectedCheckpointSha256) {
  throw new Error(`checkpoint hash mismatch: ${checkpointSha256} != ${expectedCheckpointSha256}`);
}
const candidates = await readJsonl<RolloutCandidate>(candidatesPath);
validateCandidates(candidates);
const candidateSha256 = await sha256File(candidatesPath);
let checkpointAudit: {
  step: number;
  model_config: unknown;
  tokenizer_artifacts_sha256: string;
  control_token_ids: { eos: number; user: number; assistant: number };
} | null = null;

let existing: RawRollout[] = [];
try {
  await stat(outputPath);
  if (!resume) throw new Error(`${outputPath} exists; pass --resume=true to verify and continue it`);
  existing = await readJsonl<RawRollout>(outputPath);
  validateExistingRows(existing, candidates, checkpointSha256);
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
}

if (existing.length === candidates.length) {
  console.log(`Rollout ledger already complete (${existing.length}/${candidates.length}); verifying manifest state.`);
} else {
  const checkpoint = new FileCheckpoint();
  const state = await Effect.runPromise(checkpoint.load(checkpointPath));
  if (!state.tokenizerArtifacts) throw new Error("checkpoint has no tokenizer artifacts");
  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);
  const model = prepareInferenceModel(state.modelConfig, state.params);
  const rng = new SeededRng(0);
  const tokenId = (marker: string): number => {
    const ids = Array.from(tokenizer.encode(marker));
    if (ids.length !== 1) throw new Error(`${marker} is not atomic in the checkpoint tokenizer`);
    return ids[0];
  };
  const eosId = tokenId("<|end_of_text|>");
  const userId = tokenId("<|user|>");
  const assistantId = tokenId("<|assistant|>");
  checkpointAudit = {
    step: state.step,
    model_config: state.modelConfig,
    tokenizer_artifacts_sha256: sha256Text(JSON.stringify(state.tokenizerArtifacts)),
    control_token_ids: { eos: eosId, user: userId, assistant: assistantId },
  };
  const output = await open(outputPath, existing.length > 0 ? "a" : "wx");
  let generatedThisInvocation = 0;
  try {
    for (let index = existing.length; index < candidates.length; index++) {
      if (stopAfter > 0 && generatedThisInvocation >= stopAfter) break;
      const candidate = candidates[index];
      const reencodedPrompt = Array.from(tokenizer.encode(candidate.prompt));
      if (reencodedPrompt.length !== candidate.prompt_token_ids.length ||
          reencodedPrompt.some((token, tokenIndex) => token !== candidate.prompt_token_ids[tokenIndex])) {
        throw new Error(`candidate ${index + 1} prompt tokenization drift`);
      }
      if (reencodedPrompt.length === 0 || reencodedPrompt.length + maxTokens > state.modelConfig.blockSize) {
        throw new Error(`candidate ${index + 1} violates the frozen generation reserve`);
      }
      if (reencodedPrompt.at(-1) !== assistantId) throw new Error(`candidate ${index + 1} does not end on assistant marker`);

      resetCache(model);
      let logits = prefill(model, Int32Array.from(reencodedPrompt));
      let position = reencodedPrompt.length;
      const generatedTokenIds: number[] = [];
      const contentTokenIds: number[] = [];
      const audits: TokenAudit[] = [];
      let stopReason: RawRollout["stop_reason"] = "max_tokens";
      let stopTokenId: number | null = null;
      for (let step = 0; step < maxTokens; step++) {
        if (position >= state.modelConfig.blockSize) {
          stopReason = "block_limit";
          break;
        }
        const next = sampleFromLogits(model, logits, 0, 0, rng, 1);
        audits.push(tokenAudit(logits, next));
        generatedTokenIds.push(next);
        if (next === eosId) {
          stopReason = "learned_eos";
          stopTokenId = next;
          break;
        }
        if (next === userId || next === assistantId) {
          stopReason = "role_boundary";
          stopTokenId = next;
          break;
        }
        contentTokenIds.push(next);
        logits = decodeStep(model, next, position);
        position++;
      }
      const repeatRate = fourGramRepeatRate(contentTokenIds);
      const row: RawRollout = {
        schema: "alpha-rcr-ul-raw-rollout-v1",
        stable_id: candidate.stable_id,
        source: candidate.source,
        source_id: candidate.source_id,
        positive_conversation_sha256: candidate.positive_conversation_sha256,
        prompt_sha256: candidate.prompt_sha256,
        checkpoint_sha256: checkpointSha256,
        prompt_token_ids: reencodedPrompt,
        generated_token_ids: generatedTokenIds,
        content_token_ids: contentTokenIds,
        token_audit: audits,
        text: tokenizer.decode(contentTokenIds),
        stop_reason: stopReason,
        stop_token_id: stopTokenId,
        eos_terminated: stopReason === "learned_eos",
        four_gram_repeat_rate: repeatRate,
        degenerate_loop: repeatRate >= 0.2,
        output_sha256: sha256Text(JSON.stringify({
          prompt_token_ids: reencodedPrompt,
          generated_token_ids: generatedTokenIds,
        })),
      };
      await output.write(`${JSON.stringify(row)}\n`);
      existing.push(row);
      generatedThisInvocation++;
      if (existing.length % 10 === 0 || existing.length === candidates.length) {
        await output.sync();
        await atomicJson(progressPath, {
          schema: "alpha-rcr-ul-rollout-progress-v1",
          status: existing.length === candidates.length ? "complete" : "partial",
          completed_rows: existing.length,
          total_rows: candidates.length,
          checkpoint_sha256: checkpointSha256,
          candidates_sha256: candidateSha256,
        });
        console.log(`rollouts ${existing.length}/${candidates.length}`);
      }
    }
  } finally {
    await output.sync();
    await output.close();
  }
}

const complete = existing.length === candidates.length;
await atomicJson(progressPath, {
  schema: "alpha-rcr-ul-rollout-progress-v1",
  status: complete ? "complete" : "partial",
  completed_rows: existing.length,
  total_rows: candidates.length,
  checkpoint_sha256: checkpointSha256,
  candidates_sha256: candidateSha256,
});

if (complete) {
  if (checkpointAudit === null) {
    const checkpoint = new FileCheckpoint();
    const state = await Effect.runPromise(checkpoint.load(checkpointPath));
    if (!state.tokenizerArtifacts) throw new Error("checkpoint has no tokenizer artifacts");
    const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);
    const tokenId = (marker: string): number => {
      const ids = Array.from(tokenizer.encode(marker));
      if (ids.length !== 1) throw new Error(`${marker} is not atomic in the checkpoint tokenizer`);
      return ids[0];
    };
    checkpointAudit = {
      step: state.step,
      model_config: state.modelConfig,
      tokenizer_artifacts_sha256: sha256Text(JSON.stringify(state.tokenizerArtifacts)),
      control_token_ids: {
        eos: tokenId("<|end_of_text|>"),
        user: tokenId("<|user|>"),
        assistant: tokenId("<|assistant|>"),
      },
    };
  }
  const stopReasons: Record<string, number> = {};
  for (const row of existing) stopReasons[row.stop_reason] = (stopReasons[row.stop_reason] ?? 0) + 1;
  await atomicJson(manifestPath, {
    schema: "alpha-rcr-ul-rollout-manifest-v1",
    status: "complete",
    created_at: new Date().toISOString(),
    runtime: { node: process.version, platform: platform(), architecture: arch(), endian: endianness() },
    checkpoint: {
      path: checkpointPath,
      sha256: checkpointSha256,
      ...checkpointAudit,
    },
    candidates: {
      path: candidatesPath,
      sha256: candidateSha256,
      rows: candidates.length,
    },
    generation: {
      engine: "@alpha/inference cached native path",
      deterministic_greedy: true,
      max_tokens: maxTokens,
      repetition_penalty: null,
      minimum_length: null,
      role_boundary_protection: ["<|user|>", "<|assistant|>"],
      logits_audit: "chosen and runner-up logits, chosen softmax probability, logsumexp, and SHA-256 of every f32 logit vector",
    },
    output: {
      path: outputPath,
      sha256: await sha256File(outputPath),
      rows: existing.length,
    },
    summary: {
      degenerate_loops: existing.filter((row) => row.degenerate_loop).length,
      stop_reasons: stopReasons,
    },
  });
  console.log(`complete: ${manifestPath}`);
} else {
  console.log(`partial: ${existing.length}/${candidates.length}; rerun with --resume=true`);
}
