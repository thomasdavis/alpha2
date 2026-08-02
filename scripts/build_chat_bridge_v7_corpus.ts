#!/usr/bin/env npx tsx

/**
 * Build the v7 direct-semantic bridge corpus.
 *
 * The bridge combines the reviewed v4 semantic-chat training rows with a
 * deterministic sample of compact, single-exchange rows from the canonical
 * SmolTalk span.  Selection is based on provenance, conversational shape,
 * exact tokenizer length, and a hash order.  It does not use topic-name or
 * answer-key heuristics.
 */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { execFileSync } from "node:child_process";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import type { TokenizerArtifacts } from "@alpha/core";

const USER = "<|user|>";
const ASSISTANT = "<|assistant|>";
const END = "<|end_of_text|>";

interface FileEvidence {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface Turn {
  readonly role: "user" | "assistant";
  readonly content: string;
}

interface Candidate {
  readonly line: string;
  readonly digest: string;
  readonly tokens: number;
  readonly origin: "reviewed-semantic" | "canonical-direct";
  readonly sourceLine: number | null;
  readonly order: string;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index += 1) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
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

function normalize(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function nonemptyLines(value: string): readonly string[] {
  const trimmed = value.endsWith("\n") ? value.slice(0, -1) : value;
  return trimmed ? trimmed.split("\n") : [];
}

function parseRendered(line: string, label: string): readonly Turn[] {
  assert(line.startsWith(`${USER} `), `${label}: does not begin with user`);
  assert(line.endsWith(` ${END}`), `${label}: lacks terminal marker`);
  const body = line.slice(0, -` ${END}`.length);
  const markers = [...body.matchAll(/(<\|user\|>|<\|assistant\|>)/g)];
  const turns: Turn[] = [];
  for (let index = 0; index < markers.length; index += 1) {
    const current = markers[index]!;
    const next = markers[index + 1];
    const content = body
      .slice(
        (current.index ?? 0) + current[0].length,
        next?.index ?? body.length,
      )
      .trim();
    assert(content.length > 0, `${label}: empty turn ${index}`);
    const expected = index % 2 === 0 ? USER : ASSISTANT;
    assert(current[0] === expected, `${label}: non-alternating turn ${index}`);
    turns.push({ role: current[0] === USER ? "user" : "assistant", content });
  }
  assert(
    turns.length >= 2 && turns.length % 2 === 0,
    `${label}: invalid turn count`,
  );
  return turns;
}

function userTurnsFromEvalRow(row: unknown, label: string): readonly string[] {
  assert(typeof row === "object" && row !== null, `${label}: malformed row`);
  const messages = (row as { messages?: unknown }).messages;
  assert(Array.isArray(messages), `${label}: missing messages`);
  return messages
    .filter(
      (message): message is { role: string; content: string } =>
        typeof message === "object" &&
        message !== null &&
        (message as { role?: unknown }).role === "user" &&
        typeof (message as { content?: unknown }).content === "string",
    )
    .map((message) => message.content);
}

async function loadHoldoutPrompts(
  freezePath: string,
  blahPath: string,
  semanticDevPath: string,
): Promise<Set<string>> {
  const result = new Set<string>();
  const freeze = JSON.parse(await readFile(freezePath, "utf8")) as Record<
    string,
    any
  >;
  assert(
    freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1" &&
      freeze.status ===
        "development-visible; inherited-final-sealed-unexecuted",
    "unexpected or unsafe evaluation freeze",
  );
  for (const [suite, file] of Object.entries(
    freeze.visible_development as Record<string, FileEvidence>,
  )) {
    const content = await readFile(file.path, "utf8");
    assert(
      sha256(content) === file.sha256,
      `${suite}: visible suite hash drift`,
    );
    for (const [index, line] of nonemptyLines(content).entries())
      for (const prompt of userTurnsFromEvalRow(
        JSON.parse(line),
        `${suite}:${index + 1}`,
      ))
        result.add(normalize(prompt));
  }
  // Deliberately never read freeze.sealed_final.path.
  const blah = JSON.parse(await readFile(blahPath, "utf8")) as Record<
    string,
    any
  >;
  assert(
    blah.schema === "alpha-blah-frozen-baseline-v1" &&
      Array.isArray(blah.results) &&
      blah.results.length === 24,
    "unexpected BLAH baseline",
  );
  for (const [index, item] of blah.results.entries()) {
    const prompt = item?.definition?.prompt;
    assert(
      typeof prompt === "string" && prompt.length > 0,
      `BLAH ${index}: missing prompt`,
    );
    result.add(normalize(prompt));
  }
  for (const [index, line] of nonemptyLines(
    await readFile(semanticDevPath, "utf8"),
  ).entries()) {
    const turns = parseRendered(line, `semantic-dev:${index + 1}`);
    for (const turn of turns)
      if (turn.role === "user") result.add(normalize(turn.content));
  }
  result.delete("");
  return result;
}

function quantiles(values: readonly number[]): Record<string, number> {
  const sorted = [...values].sort((left, right) => left - right);
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

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, content, { flag: "wx" });
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of [
    "canonical-source",
    "canonical-manifest",
    "semantic-train",
    "semantic-dev",
    "semantic-manifest",
    "evaluation-freeze",
    "blah-baseline",
    "tokenizer",
    "output-dir",
  ])
    if (!args[key]) throw new Error(`required: --${key}`);

  const canonicalPath = resolve(args["canonical-source"]!);
  const canonicalManifestPath = resolve(args["canonical-manifest"]!);
  const semanticTrainPath = resolve(args["semantic-train"]!);
  const semanticDevPath = resolve(args["semantic-dev"]!);
  const semanticManifestPath = resolve(args["semantic-manifest"]!);
  const freezePath = resolve(args["evaluation-freeze"]!);
  const blahPath = resolve(args["blah-baseline"]!);
  const tokenizerPath = resolve(args.tokenizer!);
  const outputDir = resolve(args["output-dir"]!);
  const directTrainCount = Number(args["direct-train"] ?? "40000");
  const directDevCount = Number(args["direct-dev"] ?? "2000");
  const maxDirectTokens = Number(args["max-direct-tokens"] ?? "384");
  const seed = args.seed ?? "alpha-chat-bridge-v7";
  for (const [label, value] of Object.entries({
    directTrainCount,
    directDevCount,
    maxDirectTokens,
  }))
    assert(
      Number.isSafeInteger(value) && value > 0,
      `${label} must be a positive integer`,
    );

  const canonicalManifest = JSON.parse(
    await readFile(canonicalManifestPath, "utf8"),
  ) as Record<string, any>;
  const semanticManifest = JSON.parse(
    await readFile(semanticManifestPath, "utf8"),
  ) as Record<string, any>;
  assert(
    canonicalManifest.schema === "alpha-sft-corpus-v2",
    "unexpected canonical manifest",
  );
  assert(
    semanticManifest.schema ===
      "alpha-chat-semantic-repair-v4-corpus-manifest-v1",
    "unexpected semantic manifest",
  );
  assert(
    (await sha256File(canonicalPath)) === canonicalManifest.output.sha256,
    "canonical corpus hash drift",
  );
  assert(
    (await sha256File(semanticTrainPath)) ===
      semanticManifest.outputs.train.sha256 &&
      (await sha256File(semanticDevPath)) ===
        semanticManifest.outputs.dev.sha256,
    "reviewed semantic corpus hash drift",
  );
  const smolSpan = canonicalManifest.source_spans.find(
    (span: Record<string, unknown>) => span.source === "smol-smoltalk",
  );
  assert(
    smolSpan?.start_line === 1 && Number.isSafeInteger(smolSpan.end_line),
    "canonical SmolTalk source span is unavailable",
  );

  const tokenizerArtifacts = JSON.parse(
    await readFile(tokenizerPath, "utf8"),
  ) as TokenizerArtifacts;
  const tokenizer = tokenizerFromArtifacts(tokenizerArtifacts);
  const heldout = await loadHoldoutPrompts(
    freezePath,
    blahPath,
    semanticDevPath,
  );
  const seen = new Set<string>();
  const semanticTrain: Candidate[] = [];
  let semanticTrainHoldoutRejected = 0;
  for (const [index, line] of nonemptyLines(
    await readFile(semanticTrainPath, "utf8"),
  ).entries()) {
    const turns = parseRendered(line, `semantic-train:${index + 1}`);
    if (
      turns.some(
        (turn) => turn.role === "user" && heldout.has(normalize(turn.content)),
      )
    ) {
      semanticTrainHoldoutRejected += 1;
      continue;
    }
    const digest = sha256(line);
    assert(!seen.has(digest), `semantic-train:${index + 1}: duplicate`);
    seen.add(digest);
    semanticTrain.push({
      line,
      digest,
      tokens: tokenizer.encode(line).length,
      origin: "reviewed-semantic",
      sourceLine: null,
      order: sha256(`${seed}\0train-order\0${digest}`),
    });
  }

  const semanticDev: Candidate[] = [];
  for (const [index, line] of nonemptyLines(
    await readFile(semanticDevPath, "utf8"),
  ).entries()) {
    const digest = sha256(line);
    assert(!seen.has(digest), `semantic-dev:${index + 1}: train collision`);
    seen.add(digest);
    semanticDev.push({
      line,
      digest,
      tokens: tokenizer.encode(line).length,
      origin: "reviewed-semantic",
      sourceLine: null,
      order: sha256(`${seed}\0dev-order\0${digest}`),
    });
  }

  const directPool: Candidate[] = [];
  const rejected = {
    outsideSmolTalkSpan: 0,
    notSingleExchange: 0,
    holdoutPrompt: 0,
    fencedCode: 0,
    overTokenBound: 0,
    duplicate: 0,
  };
  let sourceRows = 0;
  const reader = createInterface({
    input: createReadStream(canonicalPath),
    crlfDelay: Infinity,
  });
  for await (const line of reader) {
    sourceRows += 1;
    if (sourceRows > smolSpan.end_line) {
      rejected.outsideSmolTalkSpan += 1;
      continue;
    }
    const turns = parseRendered(line, `canonical:${sourceRows}`);
    if (turns.length !== 2) {
      rejected.notSingleExchange += 1;
      continue;
    }
    if (heldout.has(normalize(turns[0]!.content))) {
      rejected.holdoutPrompt += 1;
      continue;
    }
    // A fenced code block is an exact serialization feature, not an
    // open-ended topic classifier. The product target does not need code yet,
    // and the rejected population remains fully recoverable from source line.
    if (line.includes("```")) {
      rejected.fencedCode += 1;
      continue;
    }
    const tokens = tokenizer.encode(line).length;
    if (tokens > maxDirectTokens) {
      rejected.overTokenBound += 1;
      continue;
    }
    const digest = sha256(line);
    if (seen.has(digest)) {
      rejected.duplicate += 1;
      continue;
    }
    directPool.push({
      line,
      digest,
      tokens,
      origin: "canonical-direct",
      sourceLine: sourceRows,
      order: sha256(`${seed}\0direct-selection\0${digest}`),
    });
  }
  assert(
    sourceRows === canonicalManifest.total,
    "canonical source row count drift",
  );
  directPool.sort((left, right) => left.order.localeCompare(right.order));
  assert(
    directPool.length >= directTrainCount + directDevCount,
    `only ${directPool.length} eligible direct rows`,
  );
  const directDev = directPool.slice(0, directDevCount).map((row) => ({
    ...row,
    order: sha256(`${seed}\0dev-order\0${row.digest}`),
  }));
  const directTrain = directPool
    .slice(directDevCount, directDevCount + directTrainCount)
    .map((row) => ({
      ...row,
      order: sha256(`${seed}\0train-order\0${row.digest}`),
    }));
  const train = [...semanticTrain, ...directTrain].sort((left, right) =>
    left.order.localeCompare(right.order),
  );
  const dev = [...semanticDev, ...directDev].sort((left, right) =>
    left.order.localeCompare(right.order),
  );
  assert(
    new Set([...train, ...dev].map((row) => row.digest)).size ===
      train.length + dev.length,
    "split overlap",
  );

  await mkdir(outputDir, { recursive: false });
  const trainPath = join(outputDir, "train.txt");
  const devPath = join(outputDir, "dev.txt");
  const catalogPath = join(outputDir, "catalog.jsonl");
  await atomicWrite(trainPath, `${train.map((row) => row.line).join("\n")}\n`);
  await atomicWrite(devPath, `${dev.map((row) => row.line).join("\n")}\n`);
  await atomicWrite(
    catalogPath,
    `${[
      ...train.map((row) => ({ ...row, line: undefined, split: "train" })),
      ...dev.map((row) => ({ ...row, line: undefined, split: "dev" })),
    ]
      .map((row) =>
        JSON.stringify({
          schema: "alpha-chat-bridge-v7-catalog-v1",
          split: row.split,
          origin: row.origin,
          source_line: row.sourceLine,
          conversation_sha256: row.digest,
          tokens: row.tokens,
          order: row.order,
        }),
      )
      .join("\n")}\n`,
  );

  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
  const sourceTreeDirty =
    execFileSync("git", ["status", "--porcelain"], { encoding: "utf8" }).trim()
      .length > 0;
  const manifest = {
    schema: "alpha-chat-bridge-v7-corpus-v1",
    purpose:
      "bridge proven autoregressive chat stability to compact direct and reviewed semantic answers",
    createdUtc: new Date().toISOString(),
    sourceCommit,
    sourceTreeDirty,
    inputs: {
      canonicalCorpus: await evidence(canonicalPath),
      canonicalManifest: await evidence(canonicalManifestPath),
      reviewedSemanticTrain: await evidence(semanticTrainPath),
      reviewedSemanticDev: await evidence(semanticDevPath),
      reviewedSemanticManifest: await evidence(semanticManifestPath),
      evaluationFreeze: await evidence(freezePath),
      blahBaseline: await evidence(blahPath),
      tokenizer: await evidence(tokenizerPath),
    },
    recipe: {
      seed,
      canonicalSourceSpan: smolSpan,
      directEligibility:
        "one user turn, one assistant turn, no fenced-code serialization, exact tokenizer length within bound",
      directSelection:
        "lowest sha256(seed, direct-selection, conversation-sha256)",
      directTrainCount,
      directDevCount,
      maxDirectTokens,
      semanticTopicHeuristicApplied: false,
      exactFencedCodeSyntaxExcluded: true,
      exactVisiblePromptExclusion: true,
      exactBlahPromptExclusion: true,
      semanticDevPromptExclusion: true,
      sealedFinalRead: false,
    },
    census: {
      canonicalRows: sourceRows,
      eligibleDirectPool: directPool.length,
      holdoutNormalizedPrompts: heldout.size,
      semanticTrainHoldoutRejected,
      rejected,
    },
    rows: {
      train: train.length,
      dev: dev.length,
      total: train.length + dev.length,
      bySplitAndOrigin: {
        train: {
          reviewedSemantic: semanticTrain.length,
          canonicalDirect: directTrain.length,
        },
        dev: {
          reviewedSemantic: semanticDev.length,
          canonicalDirect: directDev.length,
        },
      },
    },
    tokens: {
      train: quantiles(train.map((row) => row.tokens)),
      dev: quantiles(dev.map((row) => row.tokens)),
    },
    outputs: {
      train: await evidence(trainPath),
      dev: await evidence(devPath),
      catalog: await evidence(catalogPath),
    },
    invariants: {
      trainDevConversationDisjoint: true,
      wholeConversationHashAddressed: true,
      exactTokenizerMeasured: true,
      sourceRowsNotRewritten: true,
      semanticOrTopicStringFilterApplied: false,
      sealedFinalInspected: false,
    },
  };
  await atomicWrite(
    join(outputDir, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", outputDir, rows: manifest.rows, tokens: manifest.tokens }, null, 2)}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
