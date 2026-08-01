#!/usr/bin/env npx tsx

/** Select an immutable, license-safe synthetic fitting slice from an Alpha corpus. */

import { createHash } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { execFileSync } from "node:child_process";

interface CatalogRow {
  readonly id: string;
  readonly source: string;
  readonly split: "train" | "dev";
  readonly conversation_sha256: string;
  readonly tokens: number;
  readonly metadata?: Record<string, unknown>;
}

function args(argv: readonly string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`invalid argument near ${String(key)}`);
    }
    parsed.set(key.slice(2), value);
  }
  return parsed;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function nonemptyLines(value: string): string[] {
  return value.split(/\r?\n/).filter((line) => line.length > 0);
}

async function atomicWrite(path: string, value: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, value, { flag: "wx" });
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const cli = args(process.argv.slice(2));
  const required = (name: string): string => {
    const value = cli.get(name);
    if (!value) throw new Error(`--${name} is required`);
    return value;
  };
  const corpus = resolve(required("corpus"));
  const output = resolve(required("out"));
  const samples = Number(cli.get("samples") ?? "100");
  const maxSeqLen = Number(cli.get("max-seq-len") ?? "128");
  const skipFirst = Number(cli.get("skip-first") ?? "16");
  const source = cli.get("source") ?? "gpt-5.4";
  const seed = cli.get("seed") ?? "alpha-blah-jacobian-lens-fit-v1";
  if (!Number.isSafeInteger(samples) || samples < 1) throw new Error("samples must be a positive integer");
  if (!Number.isSafeInteger(maxSeqLen) || maxSeqLen < 2) throw new Error("max-seq-len must be at least two");
  if (!Number.isSafeInteger(skipFirst) || skipFirst < 0 || skipFirst + 1 >= maxSeqLen) {
    throw new Error("skip-first must leave at least one source/target position");
  }

  const paths = {
    manifest: resolve(corpus, "manifest.json"),
    catalog: resolve(corpus, "catalog.jsonl"),
    train: resolve(corpus, "train.txt"),
    dev: resolve(corpus, "dev.txt"),
  };
  const [manifestBytes, catalogBytes, trainBytes, devBytes] = await Promise.all([
    readFile(paths.manifest), readFile(paths.catalog), readFile(paths.train), readFile(paths.dev),
  ]);
  const corpusManifest = JSON.parse(manifestBytes.toString("utf8")) as { schema?: unknown };
  if (corpusManifest.schema !== "alpha-chat-semantic-repair-v4-corpus-manifest-v1") {
    throw new Error("unexpected corpus manifest schema");
  }
  const catalog = nonemptyLines(catalogBytes.toString("utf8")).map((line) => JSON.parse(line) as CatalogRow);
  const rendered = {
    train: nonemptyLines(trainBytes.toString("utf8")),
    dev: nonemptyLines(devBytes.toString("utf8")),
  };
  const index = { train: 0, dev: 0 };
  const eligible: Array<{ row: CatalogRow; text: string; order: string }> = [];
  for (const row of catalog) {
    if (row.split !== "train" && row.split !== "dev") throw new Error(`${row.id}: invalid split`);
    const text = rendered[row.split][index[row.split]++];
    if (text === undefined) throw new Error(`${row.id}: catalog exceeds ${row.split} rows`);
    if (sha256(text) !== row.conversation_sha256) throw new Error(`${row.id}: rendered hash mismatch`);
    if (row.source !== source || row.tokens > maxSeqLen || row.tokens <= skipFirst + 1) continue;
    eligible.push({ row, text, order: sha256(`${seed}\0${row.conversation_sha256}`) });
  }
  for (const split of ["train", "dev"] as const) {
    if (index[split] !== rendered[split].length) throw new Error(`${split}: catalog/rendered row mismatch`);
  }
  if (eligible.length < samples) {
    throw new Error(`only ${eligible.length} eligible ${source} rows for ${samples} requested samples`);
  }
  eligible.sort((a, b) => a.order.localeCompare(b.order));
  const selected = eligible.slice(0, samples);
  const promptText = `${selected.map(({ row, text }) => JSON.stringify({
    id: row.id,
    text,
    source: row.source,
    conversation_sha256: row.conversation_sha256,
    category: row.metadata?.category ?? null,
  })).join("\n")}\n`;
  await atomicWrite(output, promptText);
  const selectionManifest = {
    schema: "alpha-blah-lens-fit-prompts-v1",
    created_utc: new Date().toISOString(),
    source_commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    selection: { seed, samples, source, max_seq_len: maxSeqLen, skip_first: skipFirst },
    inputs: Object.fromEntries(Object.entries(paths).map(([name, path]) => [name, {
      path,
      sha256: sha256(name === "manifest" ? manifestBytes : name === "catalog" ? catalogBytes : name === "train" ? trainBytes : devBytes),
    }])),
    eligible_rows: eligible.length,
    selected_ids: selected.map(({ row }) => row.id),
    output: { path: output, bytes: Buffer.byteLength(promptText), sha256: sha256(promptText) },
  };
  await atomicWrite(`${output}.manifest.json`, `${JSON.stringify(selectionManifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ result: "PASS", output, samples, sha256: sha256(promptText) })}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`);
  process.exitCode = 1;
});
