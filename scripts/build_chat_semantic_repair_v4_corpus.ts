#!/usr/bin/env npx tsx

/** Build the reviewed semantic-chat pilot corpus plus declared natural replay. */

import { createHash } from "node:crypto";
import { readdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import type { TokenizerArtifacts } from "@alpha/core";

const USER = "<|user|>";
const ASSISTANT = "<|assistant|>";
const END = "<|end_of_text|>";

interface Turn {
  readonly role: "user" | "assistant";
  readonly content: string;
}

interface Candidate {
  readonly candidate_id: string;
  readonly category: string;
  readonly intent_summary: string;
  readonly skills: readonly string[];
  readonly turns: readonly Turn[];
  readonly why_useful: string;
}

interface GenerationBatch {
  readonly batch_id: string;
  readonly items: readonly Candidate[];
}

interface Review {
  readonly candidate_id: string;
  readonly decision: "accept" | "reject";
  readonly semantic_correctness: number;
  readonly response_contingency: number;
  readonly naturalness: number;
  readonly compactness: number;
  readonly concern: string | null;
}

interface ReviewBatch {
  readonly review_batch_id: string;
  readonly reviews: readonly Review[];
}

interface Config {
  readonly schema: string;
  readonly seed: string;
  readonly development_fraction: number;
  readonly max_tokens: number;
  readonly tokenizer: string;
  readonly generation_batches: string;
  readonly review_directory: string;
  readonly replay: {
    readonly catalog: string;
    readonly train: string;
    readonly development: string;
    readonly sources: readonly string[];
  };
  readonly exact_holdouts: readonly string[];
  readonly review_thresholds: {
    readonly semantic_correctness: number;
    readonly response_contingency: number;
    readonly naturalness: number;
    readonly compactness: number;
  };
}

interface SourceRow {
  readonly id: string;
  readonly source: string;
  readonly source_id: string;
  readonly line: string;
  readonly turns: number;
  readonly metadata: Record<string, unknown>;
}

interface AcceptedRow extends SourceRow {
  readonly conversation_sha256: string;
  readonly tokens: number;
  readonly split: "train" | "dev";
  readonly order: string;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function parseArgs(argv: readonly string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`invalid argument near ${String(key)}`);
    }
    result[key.slice(2)] = value;
    index += 1;
  }
  return result;
}

async function fileEvidence(
  path: string,
): Promise<{ path: string; bytes: number; sha256: string }> {
  const content = await readFile(path);
  return { path, bytes: content.byteLength, sha256: sha256(content) };
}

function resolveConfigPath(value: string, configPath: string): string {
  return isAbsolute(value) ? value : resolve(dirname(configPath), "..", value);
}

function lines(content: string): readonly string[] {
  const withoutEnd = content.endsWith("\n") ? content.slice(0, -1) : content;
  return withoutEnd.length === 0 ? [] : withoutEnd.split("\n");
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateTurns(turns: readonly Turn[], id: string): void {
  if (turns.length < 2 || turns.length % 2 !== 0)
    throw new Error(`${id}: invalid turn count`);
  turns.forEach((turn, index) => {
    const role = index % 2 === 0 ? "user" : "assistant";
    if (turn.role !== role || !turn.content.trim())
      throw new Error(`${id}: invalid turn ${index}`);
    if (
      turn.content.includes(USER) ||
      turn.content.includes(ASSISTANT) ||
      turn.content.includes(END)
    ) {
      throw new Error(`${id}: model-visible marker leak`);
    }
  });
}

function render(turns: readonly Turn[]): string {
  return `${turns
    .map(
      (turn) =>
        `${turn.role === "user" ? USER : ASSISTANT} ${turn.content.trim().replace(/\s+/g, " ")}`,
    )
    .join(" ")} ${END}`;
}

function parseRendered(line: string, id: string): readonly Turn[] {
  if (!line.endsWith(END))
    throw new Error(`${id}: rendered row lacks terminal marker`);
  const body = line.slice(0, -END.length).trim();
  const marker = /(<\|user\|>|<\|assistant\|>)/g;
  const matches = [...body.matchAll(marker)];
  const result: Turn[] = [];
  for (let index = 0; index < matches.length; index += 1) {
    const current = matches[index]!;
    const next = matches[index + 1];
    const content = body
      .slice(
        (current.index ?? 0) + current[0].length,
        next?.index ?? body.length,
      )
      .trim();
    result.push({ role: current[0] === USER ? "user" : "assistant", content });
  }
  validateTurns(result, id);
  return result;
}

function acceptedReview(review: Review, config: Config): boolean {
  const threshold = config.review_thresholds;
  return (
    review.decision === "accept" &&
    review.semantic_correctness >= threshold.semantic_correctness &&
    review.response_contingency >= threshold.response_contingency &&
    review.naturalness >= threshold.naturalness &&
    review.compactness >= threshold.compactness
  );
}

async function loadReviews(directory: string): Promise<Map<string, Review>> {
  const names = (await readdir(directory))
    .filter((name) => /^review-\d+\.json$/.test(name))
    .sort();
  if (names.length === 0)
    throw new Error(`no completed review batches in ${directory}`);
  const result = new Map<string, Review>();
  for (const name of names) {
    const batch = JSON.parse(
      await readFile(join(directory, name), "utf8"),
    ) as ReviewBatch;
    for (const review of batch.reviews) {
      if (result.has(review.candidate_id))
        throw new Error(`duplicate review ${review.candidate_id}`);
      result.set(review.candidate_id, review);
    }
  }
  return result;
}

async function generatedRows(
  config: Config,
  reviews: Map<string, Review>,
  rejected: unknown[],
): Promise<SourceRow[]> {
  const names = (await readdir(config.generation_batches))
    .filter((name) => name.endsWith(".json") && !name.includes(".tmp-"))
    .sort();
  if (names.length === 0)
    throw new Error(`no generation batches in ${config.generation_batches}`);
  const result: SourceRow[] = [];
  for (const name of names) {
    const batch = JSON.parse(
      await readFile(join(config.generation_batches, name), "utf8"),
    ) as GenerationBatch;
    for (const candidate of batch.items) {
      validateTurns(candidate.turns, candidate.candidate_id);
      const review = reviews.get(candidate.candidate_id);
      if (!review)
        throw new Error(`missing review for ${candidate.candidate_id}`);
      if (!acceptedReview(review, config)) {
        rejected.push({
          candidate_id: candidate.candidate_id,
          source: "gpt-5.4",
          reason: "review",
          review,
          candidate,
        });
        continue;
      }
      result.push({
        id: candidate.candidate_id,
        source: "gpt-5.4",
        source_id: candidate.candidate_id,
        line: render(candidate.turns),
        turns: candidate.turns.length,
        metadata: {
          category: candidate.category,
          intent_summary: candidate.intent_summary,
          skills: candidate.skills,
          why_useful: candidate.why_useful,
          review,
        },
      });
    }
  }
  return result;
}

async function replayRows(config: Config): Promise<SourceRow[]> {
  const catalog = lines(await readFile(config.replay.catalog, "utf8")).map(
    (line) => JSON.parse(line) as Record<string, unknown>,
  );
  const renderedBySplit = {
    dev: lines(await readFile(config.replay.development, "utf8")),
    train: lines(await readFile(config.replay.train, "utf8")),
  };
  const indexes = { dev: 0, train: 0 };
  const allowed = new Set(config.replay.sources);
  const result: SourceRow[] = [];
  for (const row of catalog) {
    const split = row.split;
    if (split !== "dev" && split !== "train")
      throw new Error(`invalid replay split ${String(split)}`);
    const line = renderedBySplit[split][indexes[split]];
    indexes[split] += 1;
    if (line === undefined)
      throw new Error(`replay ${split} catalog exceeds rendered rows`);
    const source = String(row.source);
    if (!allowed.has(source)) continue;
    const sourceId = String(row.source_id);
    const id = `replay-${source}-${sourceId}`;
    const parsed = parseRendered(line, id);
    result.push({
      id,
      source,
      source_id: sourceId,
      line,
      turns: parsed.length,
      metadata: {},
    });
  }
  for (const split of ["dev", "train"] as const) {
    if (indexes[split] !== renderedBySplit[split].length) {
      throw new Error(
        `replay ${split}: catalog ${indexes[split]} != rows ${renderedBySplit[split].length}`,
      );
    }
  }
  return result;
}

async function holdoutUserTurns(
  paths: readonly string[],
): Promise<Set<string>> {
  const result = new Set<string>();
  for (const path of paths) {
    for (const line of lines(await readFile(path, "utf8"))) {
      const row = JSON.parse(line) as unknown;
      if (!isObject(row) || !Array.isArray(row.messages))
        throw new Error(`${path}: malformed holdout row`);
      for (const message of row.messages) {
        if (
          isObject(message) &&
          message.role === "user" &&
          typeof message.content === "string"
        ) {
          result.add(normalize(message.content));
        }
      }
    }
  }
  return result;
}

function quantiles(values: readonly number[]): Record<string, number> {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction: number): number =>
    sorted[Math.floor((sorted.length - 1) * fraction)] ?? 0;
  return {
    min: sorted[0] ?? 0,
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: sorted.at(-1) ?? 0,
  };
}

async function writeAtomic(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, content);
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));
  if (!cli.config || !cli.out) throw new Error("required: --config and --out");
  const configPath = resolve(cli.config);
  const rawConfig = JSON.parse(await readFile(configPath, "utf8")) as Config;
  if (rawConfig.schema !== "alpha-chat-semantic-repair-v4-build-config-v1") {
    throw new Error(`unexpected config schema ${rawConfig.schema}`);
  }
  const config: Config = {
    ...rawConfig,
    tokenizer: resolveConfigPath(rawConfig.tokenizer, configPath),
    generation_batches: resolveConfigPath(
      rawConfig.generation_batches,
      configPath,
    ),
    review_directory: resolveConfigPath(rawConfig.review_directory, configPath),
    replay: {
      ...rawConfig.replay,
      catalog: resolveConfigPath(rawConfig.replay.catalog, configPath),
      train: resolveConfigPath(rawConfig.replay.train, configPath),
      development: resolveConfigPath(rawConfig.replay.development, configPath),
    },
    exact_holdouts: rawConfig.exact_holdouts.map((path) =>
      resolveConfigPath(path, configPath),
    ),
  };
  if (!(config.development_fraction > 0 && config.development_fraction < 0.5)) {
    throw new Error("development_fraction must be between zero and 0.5");
  }
  const outputRoot = resolve(cli.out);
  const reviews = await loadReviews(config.review_directory);
  const rejected: unknown[] = [];
  const candidates = [
    ...(await generatedRows(config, reviews, rejected)),
    ...(await replayRows(config)),
  ];
  const tokenizerArtifacts = JSON.parse(
    await readFile(config.tokenizer, "utf8"),
  ) as TokenizerArtifacts;
  const tokenizer = tokenizerFromArtifacts(tokenizerArtifacts);
  const heldout = await holdoutUserTurns(config.exact_holdouts);
  const unique = new Map<string, SourceRow>();
  for (const candidate of candidates) {
    const parsed = parseRendered(candidate.line, candidate.id);
    // Match the user move that directly conditions the final assistant target.
    // Earlier generic turns such as "Hi there" are common dialogue scaffolding;
    // rejecting an entire multi-turn row for sharing one would erase the natural
    // replay corpus without representing meaningful evaluation contamination.
    const finalUser = parsed.at(-2);
    const exactCollision =
      finalUser?.role === "user" && heldout.has(normalize(finalUser.content));
    if (exactCollision) {
      rejected.push({
        id: candidate.id,
        source: candidate.source,
        reason: "exact_holdout_collision",
      });
      continue;
    }
    const tokens = tokenizer.encode(candidate.line).length;
    if (tokens > config.max_tokens) {
      rejected.push({
        id: candidate.id,
        source: candidate.source,
        reason: "over_token_bound",
        tokens,
      });
      continue;
    }
    const digest = sha256(candidate.line);
    if (unique.has(digest)) {
      rejected.push({
        id: candidate.id,
        source: candidate.source,
        reason: "exact_duplicate",
        duplicate_of: unique.get(digest)?.id,
      });
      continue;
    }
    unique.set(digest, candidate);
  }

  const splitThreshold = Math.floor(config.development_fraction * 2 ** 32);
  const accepted: AcceptedRow[] = [...unique.entries()].map(([digest, row]) => {
    const splitDigest = createHash("sha256")
      .update(`${config.seed}\0split\0${digest}`)
      .digest();
    const split =
      splitDigest.readUInt32BE(0) < splitThreshold ? "dev" : "train";
    return {
      ...row,
      conversation_sha256: digest,
      tokens: tokenizer.encode(row.line).length,
      split,
      order: sha256(`${config.seed}\0order\0${digest}`),
    };
  });
  accepted.sort(
    (a, b) => a.split.localeCompare(b.split) || a.order.localeCompare(b.order),
  );
  const bySplit = {
    dev: accepted.filter((row) => row.split === "dev"),
    train: accepted.filter((row) => row.split === "train"),
  };
  if (bySplit.dev.length === 0 || bySplit.train.length === 0)
    throw new Error("empty train or development split");
  const trainPath = join(outputRoot, "train.txt");
  const devPath = join(outputRoot, "dev.txt");
  const catalogPath = join(outputRoot, "catalog.jsonl");
  const rejectedPath = join(outputRoot, "rejected.jsonl");
  await import("node:fs/promises").then(({ mkdir }) =>
    mkdir(outputRoot, { recursive: true }),
  );
  await writeAtomic(
    trainPath,
    `${bySplit.train.map((row) => row.line).join("\n")}\n`,
  );
  await writeAtomic(
    devPath,
    `${bySplit.dev.map((row) => row.line).join("\n")}\n`,
  );
  await writeAtomic(
    catalogPath,
    `${accepted
      .map((row) =>
        JSON.stringify({
          schema: "alpha-chat-semantic-repair-v4-catalog-v1",
          id: row.id,
          source: row.source,
          source_id: row.source_id,
          conversation_sha256: row.conversation_sha256,
          split: row.split,
          tokens: row.tokens,
          turns: row.turns,
          metadata: row.metadata,
        }),
      )
      .join("\n")}\n`,
  );
  await writeAtomic(
    rejectedPath,
    rejected.length === 0
      ? ""
      : `${rejected.map((row) => JSON.stringify(row)).join("\n")}\n`,
  );

  const sourceCounts = Object.fromEntries(
    [...new Set(accepted.map((row) => row.source))]
      .sort()
      .map((source) => [
        source,
        accepted.filter((row) => row.source === source).length,
      ]),
  );
  const manifest = {
    schema: "alpha-chat-semantic-repair-v4-corpus-manifest-v1",
    built_utc: new Date().toISOString(),
    config: await fileEvidence(configPath),
    inputs: {
      tokenizer: await fileEvidence(config.tokenizer),
      generation_manifest: await fileEvidence(
        join(dirname(config.generation_batches), "generation-manifest.json"),
      ),
      review_manifest: await fileEvidence(
        join(config.review_directory, "review-manifest.json"),
      ),
      replay_catalog: await fileEvidence(config.replay.catalog),
      replay_train: await fileEvidence(config.replay.train),
      replay_development: await fileEvidence(config.replay.development),
      exact_holdouts: await Promise.all(
        config.exact_holdouts.map(fileEvidence),
      ),
    },
    recipe: {
      seed: config.seed,
      development_fraction: config.development_fraction,
      max_tokens: config.max_tokens,
      replay_sources: config.replay.sources,
      review_thresholds: config.review_thresholds,
      split: "sha256(seed, conversation_sha256)",
      order: "sha256(seed, conversation_sha256)",
    },
    rows: {
      train: bySplit.train.length,
      dev: bySplit.dev.length,
      total: accepted.length,
    },
    sources: sourceCounts,
    rejected: rejected.length,
    tokens: {
      train: quantiles(bySplit.train.map((row) => row.tokens)),
      dev: quantiles(bySplit.dev.map((row) => row.tokens)),
    },
    outputs: {
      train: await fileEvidence(trainPath),
      dev: await fileEvidence(devPath),
      catalog: await fileEvidence(catalogPath),
      rejected: await fileEvidence(rejectedPath),
    },
  };
  const manifestPath = join(outputRoot, "manifest.json");
  await writeAtomic(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", manifest: manifestPath, rows: manifest.rows })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
