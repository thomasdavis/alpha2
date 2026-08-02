#!/usr/bin/env npx tsx

/**
 * Stream Alpha's canonical SFT corpus into a v6 foundation corpus while
 * removing entire conversations whose user turns exactly match visible
 * development or BLAH-baseline prompts.  The source bytes are otherwise
 * preserved verbatim; no topic or quality decision is encoded as a string
 * heuristic.
 */

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";

interface FileEvidence {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

interface HoldoutPrompt {
  readonly normalized: string;
  readonly original: string;
  readonly source: string;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(): Record<string, string> {
  const values: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index += 1) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`invalid argument near ${String(key)}`);
    values[key.slice(2)] = value;
    index += 1;
  }
  return values;
}

function normalize(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function evidence(path: string): Promise<FileEvidence> {
  const metadata = await stat(path);
  return { path, bytes: metadata.size, sha256: await sha256File(path) };
}

function messagesFromRow(row: unknown, label: string): readonly string[] {
  if (typeof row !== "object" || row === null || !Array.isArray((row as any).messages))
    throw new Error(`${label}: missing messages`);
  return (row as any).messages
    .filter((message: any) => message?.role === "user" && typeof message.content === "string")
    .map((message: any) => message.content);
}

async function loadVisibleFreeze(path: string): Promise<HoldoutPrompt[]> {
  const freeze = JSON.parse(await readFile(path, "utf8")) as Record<string, any>;
  assert(
    freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1",
    "unexpected evaluation freeze",
  );
  assert(
    freeze.status === "development-visible; inherited-final-sealed-unexecuted",
    "evaluation freeze is not selection-safe",
  );
  const prompts: HoldoutPrompt[] = [];
  for (const [suite, file] of Object.entries(freeze.visible_development as Record<string, any>)) {
    if (suite === "panel") continue;
    const content = await readFile(file.path, "utf8");
    assert(createHash("sha256").update(content).digest("hex") === file.sha256, `${suite} hash drift`);
    for (const [index, line] of content.split(/\n/).entries()) {
      if (!line) continue;
      const row = JSON.parse(line);
      for (const original of messagesFromRow(row, `${suite}:${index + 1}`))
        prompts.push({ normalized: normalize(original), original, source: `visible:${suite}` });
    }
  }
  // Deliberately do not read freeze.sealed_final.path.
  return prompts;
}

async function loadBlahBaseline(path: string): Promise<HoldoutPrompt[]> {
  const baseline = JSON.parse(await readFile(path, "utf8")) as Record<string, any>;
  assert(baseline.schema === "alpha-blah-frozen-baseline-v1", "unexpected BLAH baseline schema");
  assert(Array.isArray(baseline.results) && baseline.results.length === 24, "BLAH baseline is not frozen 24/24");
  return baseline.results.map((result: any, index: number) => {
    const original = result?.definition?.prompt;
    assert(typeof original === "string" && original.length > 0, `BLAH result ${index} has no prompt`);
    return { normalized: normalize(original), original, source: "blah:frozen-baseline-24" };
  });
}

function userTurns(rendered: string, lineNumber: number): readonly string[] {
  assert(rendered.startsWith("<|user|> "), `line ${lineNumber}: does not start with user marker`);
  assert(rendered.endsWith(" <|end_of_text|>"), `line ${lineNumber}: lacks terminal marker`);
  const turns: string[] = [];
  for (const segment of rendered.split("<|user|>").slice(1)) {
    const end = segment.indexOf("<|assistant|>");
    assert(end >= 0, `line ${lineNumber}: user turn lacks assistant reply`);
    turns.push(segment.slice(0, end).trim());
  }
  return turns;
}

async function closeStream(stream: ReturnType<typeof createWriteStream>): Promise<void> {
  await new Promise<void>((accept, reject) => {
    stream.on("error", reject);
    stream.end(accept);
  });
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of ["source", "source-manifest", "evaluation-freeze", "blah-baseline", "output-dir"])
    if (!args[key]) throw new Error(`required: --${key}`);
  const source = resolve(args.source!);
  const sourceManifestPath = resolve(args["source-manifest"]!);
  const freezePath = resolve(args["evaluation-freeze"]!);
  const blahPath = resolve(args["blah-baseline"]!);
  const outputDir = resolve(args["output-dir"]!);
  await mkdir(outputDir, { recursive: false });

  const sourceManifest = JSON.parse(await readFile(sourceManifestPath, "utf8"));
  assert(sourceManifest.schema === "alpha-sft-corpus-v2", "unexpected source manifest");
  const sourceEvidence = await evidence(source);
  assert(sourceEvidence.sha256 === sourceManifest.output.sha256, "canonical SFT hash drift");

  const holdouts = [
    ...(await loadVisibleFreeze(freezePath)),
    ...(await loadBlahBaseline(blahPath)),
  ].filter((entry) => entry.normalized.length > 0);
  const byNormalized = new Map<string, HoldoutPrompt[]>();
  for (const holdout of holdouts) {
    const group = byNormalized.get(holdout.normalized) ?? [];
    group.push(holdout);
    byNormalized.set(holdout.normalized, group);
  }

  const trainPath = resolve(outputDir, "train.txt");
  const rejectedPath = resolve(outputDir, "excluded-visible-overlap.jsonl");
  const trainTmp = `${trainPath}.tmp`;
  const rejectedTmp = `${rejectedPath}.tmp`;
  const train = createWriteStream(trainTmp, { flags: "wx" });
  const rejected = createWriteStream(rejectedTmp, { flags: "wx" });
  const trainHash = createHash("sha256");
  const rejectedHash = createHash("sha256");
  let sourceRows = 0;
  let acceptedRows = 0;
  let rejectedRows = 0;
  let acceptedBytes = 0;
  let rejectedBytes = 0;
  const excludedByPrompt = new Map<string, number>();

  const lines = createInterface({ input: createReadStream(source), crlfDelay: Infinity });
  for await (const rendered of lines) {
    sourceRows += 1;
    const matches = new Map<string, HoldoutPrompt>();
    for (const turn of userTurns(rendered, sourceRows)) {
      for (const holdout of byNormalized.get(normalize(turn)) ?? [])
        matches.set(`${holdout.source}\u0000${holdout.normalized}`, holdout);
    }
    if (matches.size > 0) {
      rejectedRows += 1;
      for (const holdout of matches.values())
        excludedByPrompt.set(holdout.original, (excludedByPrompt.get(holdout.original) ?? 0) + 1);
      const record = `${JSON.stringify({
        source_line: sourceRows,
        conversation_sha256: createHash("sha256").update(rendered).digest("hex"),
        matches: [...matches.values()],
        rendered,
      })}\n`;
      rejected.write(record);
      rejectedHash.update(record);
      rejectedBytes += Buffer.byteLength(record);
    } else {
      const record = `${rendered}\n`;
      train.write(record);
      trainHash.update(record);
      acceptedBytes += Buffer.byteLength(record);
      acceptedRows += 1;
    }
  }
  await Promise.all([closeStream(train), closeStream(rejected)]);
  assert(sourceRows === sourceManifest.total, "source row count drift");
  assert(acceptedRows + rejectedRows === sourceRows, "row accounting drift");
  await Promise.all([rename(trainTmp, trainPath), rename(rejectedTmp, rejectedPath)]);

  const manifest = {
    schema: "alpha-chat-foundation-v6-corpus-v1",
    purpose: "broad direct instruction and conversation grounding without exact visible-evaluation user turns",
    createdUtc: new Date().toISOString(),
    source: {
      corpus: sourceEvidence,
      manifest: await evidence(sourceManifestPath),
      rows: sourceRows,
      sourceSpans: sourceManifest.source_spans,
    },
    holdouts: {
      evaluationFreeze: await evidence(freezePath),
      blahBaseline: await evidence(blahPath),
      visiblePromptRecords: holdouts.length,
      uniqueNormalizedPrompts: byNormalized.size,
      sealedFinalRead: false,
    },
    outputs: {
      train: { path: trainPath, rows: acceptedRows, bytes: acceptedBytes, sha256: trainHash.digest("hex") },
      excluded: {
        path: rejectedPath,
        rows: rejectedRows,
        bytes: rejectedBytes,
        sha256: rejectedHash.digest("hex"),
      },
    },
    excludedByPrompt: Object.fromEntries([...excludedByPrompt].sort(([a], [b]) => a.localeCompare(b))),
    invariants: {
      sourceBytesOtherwisePreserved: true,
      wholeConversationExcludedOnAnyExactNormalizedUserTurnMatch: true,
      semanticOrTopicFilterApplied: false,
      sealedFinalInspected: false,
    },
  };
  const manifestPath = resolve(outputDir, "manifest.json");
  const temporary = `${manifestPath}.tmp`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { flag: "wx" });
  await rename(temporary, manifestPath);
  process.stdout.write(`${JSON.stringify({ result: "PASS", manifest: manifestPath, acceptedRows, rejectedRows }, null, 2)}\n`);
}

await main();
