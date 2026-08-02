#!/usr/bin/env npx tsx

/** Compile the independently reviewed Alpha v8 synthetic conversation corpus. */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import { resolveChatSpecialIds } from "@alpha/train";
import type { TokenizerArtifacts } from "@alpha/core";

const USER = "<|user|>";
const ASSISTANT = "<|assistant|>";
const END = "<|end_of_text|>";
const EXPECTED_SEALED_SHA =
  "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d";

type Turn = { readonly role: "user" | "assistant"; readonly content: string };

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

interface FileEvidence {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface Config {
  readonly schema: string;
  readonly seed: string;
  readonly max_tokens: number;
  readonly tokenizer: string;
  readonly generation_directory: string;
  readonly review_directory: string;
  readonly evaluation_freeze: string;
  readonly blah_baseline: string;
  readonly rendered_holdouts: readonly string[];
  readonly development_batches_per_focus: number;
  readonly review_thresholds: {
    readonly semantic_correctness: number;
    readonly response_contingency: number;
    readonly naturalness: number;
    readonly compactness: number;
  };
}

interface LoadedCandidate {
  readonly batch_id: string;
  readonly focus: string;
  readonly candidate: Candidate;
  readonly review: Review;
  readonly rendered: string;
  readonly conversation_sha256: string;
  readonly tokens: number;
  readonly user_turns: readonly string[];
  readonly normalized_user_turns: readonly string[];
}

interface CatalogRow {
  readonly schema: "alpha-chat-foundations-v8-catalog-v1";
  readonly candidate_id: string;
  readonly batch_id: string;
  readonly focus: string;
  readonly status: "train" | "dev" | "rejected";
  readonly rejection_reasons: readonly Record<string, unknown>[];
  readonly conversation_sha256: string;
  readonly tokens: number;
  readonly turns: number;
  readonly normalized_user_turn_sha256: readonly string[];
  readonly review: Review;
  readonly candidate: Candidate;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(argv: readonly string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`invalid argument near ${String(key)}`);
    result[key.slice(2)] = value;
    index += 1;
  }
  return result;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path))
    hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function evidence(path: string): Promise<FileEvidence> {
  const metadata = await stat(path);
  return { path, bytes: metadata.size, sha256: await sha256File(path) };
}

async function verifyEvidence(
  expected: Record<string, unknown>,
  label: string,
): Promise<void> {
  assert(typeof expected.path === "string", `${label}: path missing`);
  assert(
    expected.bytes === undefined || Number.isSafeInteger(expected.bytes),
    `${label}: invalid byte count`,
  );
  assert(
    typeof expected.sha256 === "string" &&
      /^[0-9a-f]{64}$/.test(expected.sha256),
    `${label}: sha256 missing`,
  );
  const actual = await evidence(expected.path);
  if (expected.bytes !== undefined)
    assert(actual.bytes === expected.bytes, `${label}: byte count drift`);
  assert(actual.sha256 === expected.sha256, `${label}: SHA-256 drift`);
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, content, { encoding: "utf8", flag: "wx" });
  await rename(temporary, path);
}

function configPath(value: string, configFile: string): string {
  return isAbsolute(value) ? value : resolve(dirname(configFile), value);
}

function normalize(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function validateTurns(turns: readonly Turn[], id: string): void {
  assert(
    turns.length === 2 || turns.length === 4,
    `${id}: expected 2 or 4 turns`,
  );
  turns.forEach((turn, index) => {
    const expected = index % 2 === 0 ? "user" : "assistant";
    assert(turn.role === expected, `${id}: turn ${index} role drift`);
    assert(turn.content.trim().length > 0, `${id}: empty turn ${index}`);
    for (const marker of [USER, ASSISTANT, END])
      assert(
        !turn.content.includes(marker),
        `${id}: model-visible marker leak`,
      );
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

function renderedUserTurns(line: string, label: string): readonly string[] {
  assert(
    line.startsWith(`${USER} `),
    `${label}: does not start with user marker`,
  );
  assert(line.endsWith(` ${END}`), `${label}: terminal EOS missing`);
  const result: string[] = [];
  for (const segment of line.split(USER).slice(1)) {
    const end = segment.indexOf(ASSISTANT);
    assert(end >= 0, `${label}: user turn lacks assistant reply`);
    const value = segment.slice(0, end).trim();
    assert(value.length > 0, `${label}: empty rendered user turn`);
    result.push(value);
  }
  return result;
}

function userMessages(row: unknown, label: string): readonly string[] {
  assert(typeof row === "object" && row !== null, `${label}: malformed row`);
  const messages = (row as Record<string, unknown>).messages;
  assert(Array.isArray(messages), `${label}: messages missing`);
  return messages
    .filter(
      (message): message is { role: "user"; content: string } =>
        typeof message === "object" &&
        message !== null &&
        (message as Record<string, unknown>).role === "user" &&
        typeof (message as Record<string, unknown>).content === "string",
    )
    .map((message) => message.content);
}

async function visibleHoldouts(freezePath: string): Promise<{
  prompts: Array<{ normalized: string; original: string; source: string }>;
  freeze: Record<string, any>;
}> {
  const freeze = JSON.parse(await readFile(freezePath, "utf8")) as Record<
    string,
    any
  >;
  assert(
    freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1",
    "unexpected evaluation freeze",
  );
  assert(
    freeze.status === "development-visible; inherited-final-sealed-unexecuted",
    "evaluation freeze is not selection-safe",
  );
  assert(
    freeze.sealed_final?.sha256 === EXPECTED_SEALED_SHA,
    "sealed-final identity drift",
  );
  const prompts: Array<{
    normalized: string;
    original: string;
    source: string;
  }> = [];
  for (const [suite, item] of Object.entries(
    freeze.visible_development as Record<string, Record<string, unknown>>,
  )) {
    await verifyEvidence(item, `visible suite ${suite}`);
    const content = await readFile(String(item.path), "utf8");
    for (const [index, line] of content.split(/\r?\n/).entries()) {
      if (!line) continue;
      for (const original of userMessages(
        JSON.parse(line) as unknown,
        `${suite}:${index + 1}`,
      ))
        prompts.push({
          normalized: normalize(original),
          original,
          source: `visible:${suite}`,
        });
    }
  }
  // Deliberately never open freeze.sealed_final.path.
  return { prompts, freeze };
}

async function blahHoldouts(
  path: string,
): Promise<Array<{ normalized: string; original: string; source: string }>> {
  const baseline = JSON.parse(await readFile(path, "utf8")) as Record<
    string,
    any
  >;
  assert(
    baseline.schema === "alpha-blah-frozen-baseline-v1",
    "unexpected BLAH baseline schema",
  );
  assert(
    Array.isArray(baseline.results) && baseline.results.length === 24,
    "BLAH baseline is not the frozen 24-item population",
  );
  return baseline.results.map((result: any, index: number) => {
    const original = result?.definition?.prompt;
    assert(
      typeof original === "string" && original.length > 0,
      `BLAH ${index}: prompt missing`,
    );
    return {
      normalized: normalize(original),
      original,
      source: "blah:frozen-baseline-24",
    };
  });
}

async function renderedHoldouts(
  paths: readonly string[],
): Promise<Array<{ normalized: string; original: string; source: string }>> {
  const result: Array<{
    normalized: string;
    original: string;
    source: string;
  }> = [];
  for (const path of paths) {
    const content = await readFile(path, "utf8");
    for (const [index, line] of content.split(/\r?\n/).entries()) {
      if (!line) continue;
      for (const original of renderedUserTurns(line, `${path}:${index + 1}`))
        result.push({
          normalized: normalize(original),
          original,
          source: `rendered:${basename(path)}`,
        });
    }
  }
  return result;
}

function acceptedByThreshold(review: Review, config: Config): boolean {
  const threshold = config.review_thresholds;
  return (
    review.decision === "accept" &&
    review.semantic_correctness >= threshold.semantic_correctness &&
    review.response_contingency >= threshold.response_contingency &&
    review.naturalness >= threshold.naturalness &&
    review.compactness >= threshold.compactness
  );
}

function quantiles(values: readonly number[]): Record<string, number> {
  const ordered = [...values].sort((a, b) => a - b);
  const at = (fraction: number): number =>
    ordered[Math.floor(Math.max(0, ordered.length - 1) * fraction)] ?? 0;
  return {
    min: ordered[0] ?? 0,
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: ordered.at(-1) ?? 0,
  };
}

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));
  assert(cli.config && cli.out, "required: --config and --out");
  const repo = resolve(cli.repo ?? process.cwd());
  const configFile = resolve(cli.config);
  const raw = JSON.parse(await readFile(configFile, "utf8")) as Config;
  assert(
    raw.schema === "alpha-chat-foundations-v8-build-config-v1",
    "unexpected config schema",
  );
  assert(
    raw.max_tokens === 512,
    "v8 is frozen to the 512-token training block",
  );
  assert(
    raw.development_batches_per_focus === 1,
    "v8 requires one whole dev batch per focus",
  );
  const config: Config = {
    ...raw,
    tokenizer: configPath(raw.tokenizer, configFile),
    generation_directory: configPath(raw.generation_directory, configFile),
    review_directory: configPath(raw.review_directory, configFile),
    evaluation_freeze: configPath(raw.evaluation_freeze, configFile),
    blah_baseline: configPath(raw.blah_baseline, configFile),
    rendered_holdouts: raw.rendered_holdouts.map((path) =>
      configPath(path, configFile),
    ),
  };
  const outputRoot = resolve(cli.out);
  await mkdir(outputRoot, { recursive: false });

  const generationManifestPath = join(
    dirname(config.generation_directory),
    "generation-manifest.json",
  );
  const reviewManifestPath = join(
    config.review_directory,
    "review-manifest.json",
  );
  const generationManifest = JSON.parse(
    await readFile(generationManifestPath, "utf8"),
  ) as Record<string, any>;
  const reviewManifest = JSON.parse(
    await readFile(reviewManifestPath, "utf8"),
  ) as Record<string, any>;
  assert(
    generationManifest.schema ===
      "alpha-chat-foundations-v8-generation-manifest-v1" &&
      generationManifest.model === "gpt-5.4" &&
      generationManifest.source_tree_dirty === false &&
      generationManifest.requested_batches === 100 &&
      generationManifest.completed?.length === 100,
    "generation manifest is incomplete or has incorrect provenance",
  );
  assert(
    reviewManifest.schema === "alpha-chat-foundations-v8-review-manifest-v1" &&
      reviewManifest.model === "gpt-5.5" &&
      reviewManifest.source_tree_dirty === false &&
      reviewManifest.requested_review_groups === 50 &&
      reviewManifest.completed?.length === 50 &&
      reviewManifest.generation_batches === 100,
    "review manifest is incomplete or has incorrect provenance",
  );
  assert(
    generationManifest.blueprint?.sha256 === reviewManifest.blueprint?.sha256,
    "generation and review blueprint identities differ",
  );

  for (const [index, item] of generationManifest.completed.entries()) {
    await verifyEvidence(item, `generation batch ${index}`);
    await verifyEvidence(
      {
        path: item.events_path,
        bytes: item.events_bytes,
        sha256: item.events_sha256,
      },
      `generation events ${index}`,
    );
  }
  for (const [index, item] of reviewManifest.completed.entries()) {
    await verifyEvidence(item, `review batch ${index}`);
    await verifyEvidence(
      {
        path: item.events_path,
        bytes: item.events_bytes,
        sha256: item.events_sha256,
      },
      `review events ${index}`,
    );
  }
  for (const [family, manifest] of [
    ["generation", generationManifest],
    ["review", reviewManifest],
  ] as const) {
    for (const [index, attempt] of (
      manifest.rejected_attempts ?? []
    ).entries()) {
      await verifyEvidence(
        attempt.output,
        `${family} rejected output ${index}`,
      );
      await verifyEvidence(
        attempt.events,
        `${family} rejected events ${index}`,
      );
    }
  }

  const blueprintPath = String(generationManifest.blueprint.path);
  await verifyEvidence(generationManifest.blueprint, "generation blueprint");
  const blueprint = JSON.parse(await readFile(blueprintPath, "utf8")) as Record<
    string,
    any
  >;
  assert(
    Array.isArray(blueprint.batches) && blueprint.batches.length === 100,
    "blueprint batch count drift",
  );
  const plan = new Map<string, string>();
  for (const item of blueprint.batches) {
    assert(
      typeof item.batch_id === "string" && typeof item.focus === "string",
      "malformed blueprint batch",
    );
    assert(
      !plan.has(item.batch_id),
      `duplicate blueprint batch ${item.batch_id}`,
    );
    plan.set(item.batch_id, item.focus);
  }

  const generationFiles = (await readdir(config.generation_directory))
    .filter((name) => /^v8-\d{3}-.+\.json$/.test(name))
    .sort();
  const reviewFiles = (await readdir(config.review_directory))
    .filter((name) => /^review-\d{3}\.json$/.test(name))
    .sort();
  assert(
    generationFiles.length === 100,
    `generation directory has ${generationFiles.length} batches`,
  );
  assert(
    reviewFiles.length === 50,
    `review directory has ${reviewFiles.length} groups`,
  );

  const batches: GenerationBatch[] = [];
  const candidateById = new Map<
    string,
    { batch_id: string; focus: string; candidate: Candidate }
  >();
  for (const file of generationFiles) {
    const batch = JSON.parse(
      await readFile(join(config.generation_directory, file), "utf8"),
    ) as GenerationBatch;
    const focus = plan.get(batch.batch_id);
    assert(focus, `${batch.batch_id}: absent from blueprint`);
    assert(
      batch.items.length === 64,
      `${batch.batch_id}: expected 64 candidates`,
    );
    assert(
      batch.items.filter((item) => item.turns.length === 2).length === 42,
      `${batch.batch_id}: single-exchange allocation drift`,
    );
    assert(
      batch.items.filter((item) => item.turns.length === 4).length === 22,
      `${batch.batch_id}: two-exchange allocation drift`,
    );
    for (const candidate of batch.items) {
      assert(
        candidate.category.trim().length > 0,
        `${candidate.candidate_id}: empty category`,
      );
      assert(
        candidate.candidate_id.startsWith(`${batch.batch_id}-`),
        `${candidate.candidate_id}: batch identity drift`,
      );
      assert(
        !candidateById.has(candidate.candidate_id),
        `duplicate candidate ${candidate.candidate_id}`,
      );
      validateTurns(candidate.turns, candidate.candidate_id);
      candidateById.set(candidate.candidate_id, {
        batch_id: batch.batch_id,
        focus,
        candidate,
      });
    }
    batches.push(batch);
  }
  assert(
    candidateById.size === 6400,
    `candidate population is ${candidateById.size}, expected 6400`,
  );

  const reviewById = new Map<string, Review>();
  for (const [groupIndex, file] of reviewFiles.entries()) {
    const batch = JSON.parse(
      await readFile(join(config.review_directory, file), "utf8"),
    ) as ReviewBatch;
    assert(
      batch.review_batch_id === `review-${String(groupIndex).padStart(3, "0")}`,
      `${file}: review identity drift`,
    );
    assert(batch.reviews.length === 128, `${file}: expected 128 reviews`);
    const expectedIds = new Set(
      batches
        .slice(groupIndex * 2, groupIndex * 2 + 2)
        .flatMap((item) =>
          item.items.map((candidate) => candidate.candidate_id),
        ),
    );
    for (const review of batch.reviews) {
      assert(
        expectedIds.delete(review.candidate_id),
        `${file}: unexpected or duplicate ${review.candidate_id}`,
      );
      assert(
        !reviewById.has(review.candidate_id),
        `duplicate review ${review.candidate_id}`,
      );
      assert(
        ["accept", "reject"].includes(review.decision),
        `${review.candidate_id}: invalid decision`,
      );
      for (const score of [
        review.semantic_correctness,
        review.response_contingency,
        review.naturalness,
        review.compactness,
      ])
        assert(
          Number.isSafeInteger(score) && score >= 1 && score <= 5,
          `${review.candidate_id}: invalid score`,
        );
      assert(
        review.concern === null ||
          (typeof review.concern === "string" &&
            review.concern.trim().length > 0),
        `${review.candidate_id}: invalid concern`,
      );
      reviewById.set(review.candidate_id, review);
    }
    assert(
      expectedIds.size === 0,
      `${file}: missing ${expectedIds.size} candidate reviews`,
    );
  }
  assert(
    reviewById.size === candidateById.size,
    "candidate/review population differs",
  );

  const tokenizerArtifacts = JSON.parse(
    await readFile(config.tokenizer, "utf8"),
  ) as TokenizerArtifacts;
  const tokenizer = tokenizerFromArtifacts(tokenizerArtifacts);
  const specialIds = resolveChatSpecialIds(tokenizer);
  assert(tokenizer.encode(USER).length === 1, "user marker is not atomic");
  assert(
    tokenizer.encode(ASSISTANT).length === 1,
    "assistant marker is not atomic",
  );
  assert(tokenizer.encode(END).length === 1, "EOS marker is not atomic");

  const { prompts: frozenPrompts, freeze } = await visibleHoldouts(
    config.evaluation_freeze,
  );
  frozenPrompts.push(...(await blahHoldouts(config.blah_baseline)));
  frozenPrompts.push(...(await renderedHoldouts(config.rendered_holdouts)));
  const holdoutsByNormalized = new Map<
    string,
    Array<{ original: string; source: string }>
  >();
  for (const prompt of frozenPrompts) {
    if (!prompt.normalized) continue;
    const group = holdoutsByNormalized.get(prompt.normalized) ?? [];
    group.push({ original: prompt.original, source: prompt.source });
    holdoutsByNormalized.set(prompt.normalized, group);
  }

  const focusBatches = new Map<string, string[]>();
  for (const [batchId, focus] of plan) {
    const group = focusBatches.get(focus) ?? [];
    group.push(batchId);
    focusBatches.set(focus, group);
  }
  assert(
    focusBatches.size === 10,
    `expected ten focuses, found ${focusBatches.size}`,
  );
  const developmentBatches = new Set<string>();
  for (const [focus, ids] of focusBatches) {
    assert(ids.length === 10, `${focus}: expected ten batches`);
    ids.sort((a, b) =>
      sha256(`${config.seed}\0dev-batch\0${a}`).localeCompare(
        sha256(`${config.seed}\0dev-batch\0${b}`),
      ),
    );
    for (const batchId of ids.slice(0, config.development_batches_per_focus))
      developmentBatches.add(batchId);
  }

  const loaded: LoadedCandidate[] = [...candidateById.values()]
    .map(({ batch_id, focus, candidate }) => {
      const review = reviewById.get(candidate.candidate_id)!;
      const rendered = render(candidate.turns);
      return {
        batch_id,
        focus,
        candidate,
        review,
        rendered,
        conversation_sha256: sha256(rendered),
        tokens: tokenizer.encode(rendered).length,
        user_turns: candidate.turns
          .filter((turn) => turn.role === "user")
          .map((turn) => turn.content),
        normalized_user_turns: candidate.turns
          .filter((turn) => turn.role === "user")
          .map((turn) => normalize(turn.content)),
      };
    })
    .sort((a, b) =>
      a.candidate.candidate_id.localeCompare(b.candidate.candidate_id),
    );

  const seenConversation = new Map<string, string>();
  const seenUserTurn = new Map<string, string>();
  const accepted: Array<LoadedCandidate & { split: "train" | "dev" }> = [];
  const catalog: CatalogRow[] = [];
  for (const row of loaded) {
    const rejectionReasons: Record<string, unknown>[] = [];
    if (!acceptedByThreshold(row.review, config))
      rejectionReasons.push({ kind: "independent_review", review: row.review });
    const holdoutMatches = row.normalized_user_turns.flatMap((turn) =>
      (holdoutsByNormalized.get(turn) ?? []).map((match) => ({
        turn,
        ...match,
      })),
    );
    if (holdoutMatches.length > 0)
      rejectionReasons.push({
        kind: "exact_normalized_holdout_collision",
        matches: holdoutMatches,
      });
    if (row.tokens > config.max_tokens)
      rejectionReasons.push({ kind: "over_token_bound", tokens: row.tokens });
    const duplicateConversation = seenConversation.get(row.conversation_sha256);
    if (duplicateConversation)
      rejectionReasons.push({
        kind: "exact_conversation_duplicate",
        duplicate_of: duplicateConversation,
      });
    const duplicateUserTurns = row.normalized_user_turns
      .map((turn) => ({ turn, duplicate_of: seenUserTurn.get(turn) }))
      .filter((item) => item.duplicate_of !== undefined);
    if (duplicateUserTurns.length > 0)
      rejectionReasons.push({
        kind: "normalized_user_turn_duplicate",
        duplicates: duplicateUserTurns,
      });

    let status: CatalogRow["status"] = "rejected";
    if (rejectionReasons.length === 0) {
      const split = developmentBatches.has(row.batch_id) ? "dev" : "train";
      status = split;
      accepted.push({ ...row, split });
      seenConversation.set(row.conversation_sha256, row.candidate.candidate_id);
      for (const turn of row.normalized_user_turns)
        seenUserTurn.set(turn, row.candidate.candidate_id);
    }
    catalog.push({
      schema: "alpha-chat-foundations-v8-catalog-v1",
      candidate_id: row.candidate.candidate_id,
      batch_id: row.batch_id,
      focus: row.focus,
      status,
      rejection_reasons: rejectionReasons,
      conversation_sha256: row.conversation_sha256,
      tokens: row.tokens,
      turns: row.candidate.turns.length,
      normalized_user_turn_sha256: row.normalized_user_turns.map(sha256),
      review: row.review,
      candidate: row.candidate,
    });
  }

  const train = accepted
    .filter((row) => row.split === "train")
    .sort((a, b) =>
      sha256(
        `${config.seed}\0train\0${a.candidate.candidate_id}`,
      ).localeCompare(
        sha256(`${config.seed}\0train\0${b.candidate.candidate_id}`),
      ),
    );
  const dev = accepted
    .filter((row) => row.split === "dev")
    .sort((a, b) =>
      sha256(`${config.seed}\0dev\0${a.candidate.candidate_id}`).localeCompare(
        sha256(`${config.seed}\0dev\0${b.candidate.candidate_id}`),
      ),
    );
  assert(
    train.length > 4000,
    `training population unexpectedly small: ${train.length}`,
  );
  assert(
    dev.length > 400,
    `development population unexpectedly small: ${dev.length}`,
  );
  for (const focus of focusBatches.keys()) {
    assert(
      train.some((row) => row.focus === focus),
      `${focus}: empty training focus`,
    );
    assert(
      dev.some((row) => row.focus === focus),
      `${focus}: empty development focus`,
    );
  }
  assert(
    !train.some((row) => developmentBatches.has(row.batch_id)) &&
      dev.every((row) => developmentBatches.has(row.batch_id)),
    "whole-batch split invariant failed",
  );
  assert(
    new Set(train.map((row) => row.conversation_sha256)).size ===
      train.length &&
      new Set(dev.map((row) => row.conversation_sha256)).size === dev.length,
    "accepted conversation hashes are not unique",
  );
  assert(
    !train.some((row) =>
      dev.some(
        (other) => row.conversation_sha256 === other.conversation_sha256,
      ),
    ),
    "train/development hash overlap",
  );

  const trainPath = join(outputRoot, "train.txt");
  const devPath = join(outputRoot, "dev.txt");
  const catalogPath = join(outputRoot, "catalog.jsonl");
  await atomicWrite(
    trainPath,
    `${train.map((row) => row.rendered).join("\n")}\n`,
  );
  await atomicWrite(devPath, `${dev.map((row) => row.rendered).join("\n")}\n`);
  await atomicWrite(
    catalogPath,
    `${catalog.map((row) => JSON.stringify(row)).join("\n")}\n`,
  );

  const countsByFocus = Object.fromEntries(
    [...focusBatches.keys()].sort().map((focus) => [
      focus,
      {
        generated: catalog.filter((row) => row.focus === focus).length,
        train: catalog.filter(
          (row) => row.focus === focus && row.status === "train",
        ).length,
        dev: catalog.filter(
          (row) => row.focus === focus && row.status === "dev",
        ).length,
        rejected: catalog.filter(
          (row) => row.focus === focus && row.status === "rejected",
        ).length,
      },
    ]),
  );
  const rejectionCounts = new Map<string, number>();
  for (const row of catalog)
    for (const reason of row.rejection_reasons) {
      const kind = String(reason.kind);
      rejectionCounts.set(kind, (rejectionCounts.get(kind) ?? 0) + 1);
    }

  const manifest = {
    schema: "alpha-chat-foundations-v8-corpus-v1",
    createdUtc: new Date().toISOString(),
    purpose:
      "install compact foundational conversational competence from independently reviewed synthetic dialogue",
    sourceCommit: execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repo,
      encoding: "utf8",
    }).trim(),
    sourceTreeDirty:
      execFileSync("git", ["status", "--porcelain"], {
        cwd: repo,
        encoding: "utf8",
      }).trim().length > 0,
    inputs: {
      config: await evidence(configFile),
      tokenizer: await evidence(config.tokenizer),
      blueprint: await evidence(blueprintPath),
      generationManifest: await evidence(generationManifestPath),
      reviewManifest: await evidence(reviewManifestPath),
      evaluationFreeze: await evidence(config.evaluation_freeze),
      blahBaseline: await evidence(config.blah_baseline),
      renderedHoldouts: await Promise.all(
        config.rendered_holdouts.map(evidence),
      ),
    },
    provenance: {
      planner: {
        model: "gpt-5.5",
        blueprintSha256: generationManifest.blueprint.sha256,
      },
      generator: {
        model: generationManifest.model,
        reasoningEffort: generationManifest.reasoning_effort,
        rejectedAttempts: generationManifest.rejected_attempts?.length ?? 0,
      },
      reviewer: {
        model: reviewManifest.model,
        reasoningEffort: reviewManifest.reasoning_effort,
        rejectedAttempts: reviewManifest.rejected_attempts?.length ?? 0,
      },
      candidatePopulation: candidateById.size,
      reviewedExactlyOnce: reviewById.size,
    },
    recipe: {
      seed: config.seed,
      maxTokens: config.max_tokens,
      reviewThresholds: config.review_thresholds,
      splitUnit: "whole generation batch",
      developmentBatchSelection:
        "lowest sha256(seed, dev-batch, batch_id) within each focus",
      developmentBatches: [...developmentBatches].sort(),
      noReplayData: true,
      duplicateRule:
        "first threshold-passing candidate in candidate_id order owns each exact normalized user turn and conversation hash",
    },
    rows: {
      generated: catalog.length,
      train: train.length,
      dev: dev.length,
      accepted: accepted.length,
      rejected: catalog.length - accepted.length,
      byFocus: countsByFocus,
    },
    rejectionReasons: Object.fromEntries(
      [...rejectionCounts.entries()].sort(([a], [b]) => a.localeCompare(b)),
    ),
    tokens: {
      train: quantiles(train.map((row) => row.tokens)),
      dev: quantiles(dev.map((row) => row.tokens)),
    },
    holdouts: {
      records: frozenPrompts.length,
      uniqueNormalizedPrompts: holdoutsByNormalized.size,
      acceptedCollisions: 0,
      sealedFinalIdentity: freeze.sealed_final.sha256,
      sealedFinalInspected: false,
    },
    tokenizer: {
      vocabSize: tokenizer.vocabSize,
      atomicMarkers: true,
      specialIds,
    },
    outputs: {
      train: { ...(await evidence(trainPath)), rows: train.length },
      dev: { ...(await evidence(devPath)), rows: dev.length },
      catalog: { ...(await evidence(catalogPath)), rows: catalog.length },
    },
    invariants: {
      allCandidatesCataloged: catalog.length === 6400,
      allCandidatesReviewedExactlyOnce: reviewById.size === 6400,
      rejectedCandidatesPreserved: true,
      onlyIndependentlyAcceptedSyntheticDataTrains: true,
      wholeBatchDevelopmentHoldout: true,
      exactNormalizedVisiblePromptExclusion: true,
      exactConversationDeduplication: true,
      normalizedUserTurnDeduplication: true,
      modelVisibleDelimitersInjectedAtCompileTime: true,
      assistantOnlyMaskAuditRequiredBeforeTraining: true,
      semanticOrTopicStringFilterApplied: false,
      replayDataIncluded: false,
      sealedFinalInspected: false,
    },
  };
  const manifestPath = join(outputRoot, "manifest.json");
  await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", manifest: manifestPath, rows: manifest.rows, rejectionReasons: manifest.rejectionReasons })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
