#!/usr/bin/env npx tsx

/** Pair an immutable RCR-UL trajectory population with a new positive corpus. */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";

interface NegativeRow {
  readonly schema: "alpha-rcr-ul-example-v1";
  readonly stable_id: string;
  readonly positive_conversation_sha256: string;
  readonly token_ids: readonly number[];
  readonly penalty_target_positions: readonly number[];
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (const raw of process.argv.slice(2)) {
    const match = raw.match(/^--([^=]+)=(.*)$/s);
    if (!match) throw new Error(`expected --key=value, received ${raw}`);
    result[match[1]] = match[2];
  }
  return result;
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path))
    hash.update(chunk as Buffer);
  return hash.digest("hex");
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

async function evidence(
  path: string,
  rows?: number,
): Promise<Record<string, unknown>> {
  const metadata = await stat(path);
  return {
    path,
    bytes: metadata.size,
    sha256: await sha256File(path),
    ...(rows === undefined ? {} : { rows }),
  };
}

async function jsonl<T>(path: string): Promise<T[]> {
  const result: T[] = [];
  const lines = createInterface({
    input: createReadStream(path),
    crlfDelay: Infinity,
  });
  for await (const line of lines) if (line) result.push(JSON.parse(line) as T);
  return result;
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, content, { encoding: "utf8", flag: "wx" });
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of [
    "positive",
    "corpus-manifest",
    "source-negative",
    "source-manifest",
    "seed",
    "out-dir",
  ])
    assert(args[key], `required: --${key}=...`);
  const positivePath = resolve(args.positive);
  const corpusManifestPath = resolve(args["corpus-manifest"]);
  const sourceNegativePath = resolve(args["source-negative"]);
  const sourceManifestPath = resolve(args["source-manifest"]);
  const outDir = resolve(args["out-dir"]);
  const outputPath = resolve(outDir, "negative-cohort.jsonl");
  const manifestPath = resolve(outDir, "manifest.json");
  await mkdir(outDir, { recursive: false });

  const corpus = JSON.parse(
    await readFile(corpusManifestPath, "utf8"),
  ) as Record<string, any>;
  const sourceManifest = JSON.parse(
    await readFile(sourceManifestPath, "utf8"),
  ) as Record<string, any>;
  assert(
    corpus.schema === "alpha-chat-foundations-v8-corpus-v1",
    "unexpected v8 corpus manifest",
  );
  assert(
    sourceManifest.schema === "alpha-rcr-ul-cohort-manifest-v1",
    "unexpected source RCR-UL manifest",
  );
  assert(
    sourceManifest.status === "complete-and-immutable",
    "source RCR-UL cohort is not immutable",
  );
  assert(
    sourceManifest.outputs?.negative_cohort?.sha256 ===
      (await sha256File(sourceNegativePath)),
    "source negative cohort SHA-256 drift",
  );
  const positiveText = await readFile(positivePath, "utf8");
  assert(
    sha256Text(positiveText) === corpus.outputs?.train?.sha256,
    "positive corpus SHA-256 drift",
  );
  const positives = positiveText.split(/\r?\n/).filter(Boolean);
  assert(positives.length === corpus.rows?.train, "positive row count drift");
  const sourceRows = await jsonl<NegativeRow>(sourceNegativePath);
  assert(
    sourceRows.length === sourceManifest.outputs?.negative_cohort?.rows,
    "source negative row count drift",
  );
  assert(sourceRows.length > 0, "source RCR-UL cohort is empty");
  sourceRows.forEach((row, index) => {
    assert(
      row.schema === "alpha-rcr-ul-example-v1",
      `source row ${index + 1}: schema drift`,
    );
    assert(row.token_ids.length > 0, `source row ${index + 1}: empty tokens`);
    assert(
      row.penalty_target_positions.every(
        (position) =>
          Number.isSafeInteger(position) &&
          position >= 0 &&
          position < row.token_ids.length,
      ),
      `source row ${index + 1}: invalid penalty position`,
    );
  });

  const ordered = sourceRows
    .map((row, index) => ({
      row,
      index,
      order: sha256Text(`${args.seed}\0source\0${row.stable_id}\0${index}`),
    }))
    .sort((a, b) => a.order.localeCompare(b.order));
  const outputRows = positives.map((line, index) => {
    const source = ordered[index % ordered.length];
    const positiveHash = sha256Text(line);
    return JSON.stringify({
      schema: "alpha-rcr-ul-example-v1",
      stable_id: sha256Text(`${args.seed}\0paired\0${positiveHash}\0${index}`),
      positive_conversation_sha256: positiveHash,
      token_ids: source.row.token_ids,
      penalty_target_positions: source.row.penalty_target_positions,
      source_stable_id: source.row.stable_id,
      source_row_index: source.index,
      source_cycle: Math.floor(index / ordered.length),
    });
  });
  await atomicWrite(outputPath, `${outputRows.join("\n")}\n`);

  const parsedOutput = await jsonl<NegativeRow>(outputPath);
  assert(parsedOutput.length === positives.length, "output row count drift");
  let totalPenaltyPositions = 0;
  let eligibleRows = 0;
  for (let index = 0; index < parsedOutput.length; index++) {
    const row = parsedOutput[index];
    assert(
      row.positive_conversation_sha256 === sha256Text(positives[index]),
      `output row ${index + 1}: positive pairing drift`,
    );
    totalPenaltyPositions += row.penalty_target_positions.length;
    if (row.penalty_target_positions.length > 0) eligibleRows++;
  }
  assert(
    totalPenaltyPositions > 0,
    "remapped cohort has zero penalty positions",
  );

  const manifest = {
    schema: "alpha-chat-foundations-v8-rcr-ul-remap-v1",
    status: "complete-and-immutable",
    createdUtc: new Date().toISOString(),
    sourceCommit: execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: process.cwd(),
      encoding: "utf8",
    }).trim(),
    sourceTreeDirty:
      execFileSync("git", ["status", "--porcelain"], {
        cwd: process.cwd(),
        encoding: "utf8",
      }).trim().length > 0,
    purpose:
      "retain U1-derived repetition-unlikelihood trajectories while pairing them exactly with the v8 positive corpus",
    seed: args.seed,
    inputs: {
      positive: await evidence(positivePath, positives.length),
      corpusManifest: await evidence(corpusManifestPath),
      sourceNegative: await evidence(sourceNegativePath, sourceRows.length),
      sourceManifest: await evidence(sourceManifestPath),
    },
    rule: {
      sourceOrder: "sha256(seed, source, stable_id, source_row_index)",
      assignment: "ordered source rows cycled over positive rows",
      positiveHashReboundOnly: true,
      negativeTokenIdsChanged: false,
      penaltyPositionsChanged: false,
    },
    summary: {
      rows: parsedOutput.length,
      uniqueSourceRowsUsed: Math.min(sourceRows.length, parsedOutput.length),
      maximumSourceReuse: Math.ceil(parsedOutput.length / sourceRows.length),
      eligibleNegativeRows: eligibleRows,
      totalPenaltyPositions,
    },
    output: await evidence(outputPath, parsedOutput.length),
  };
  await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", manifest: manifestPath, summary: manifest.summary })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
