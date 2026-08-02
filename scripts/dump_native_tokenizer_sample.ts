#!/usr/bin/env npx tsx
/** Stream a deterministic line sample and encode it with native Alpha artifacts. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, rename, stat, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { Effect } from "effect";
import { loadArtifacts, tokenizerFromArtifacts } from "@alpha/tokenizers";

function value(name: string): string {
  const prefix = `--${name}=`;
  const found = process.argv.find((argument) => argument.startsWith(prefix));
  if (!found) throw new Error(`required: ${prefix}<value>`);
  return found.slice(prefix.length);
}

async function sha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const artifactsPath = resolve(value("artifacts"));
  const dataPath = resolve(value("data"));
  const outputPath = resolve(value("out"));
  const rows = Number(value("rows"));
  const samples = Number(value("samples"));
  if (!Number.isSafeInteger(rows) || rows < 1) throw new Error("rows must be positive");
  if (!Number.isSafeInteger(samples) || samples < 1 || samples > rows)
    throw new Error("samples must be in [1, rows]");
  await Promise.all([stat(artifactsPath), stat(dataPath)]);

  const artifacts = await Effect.runPromise(loadArtifacts(artifactsPath));
  const tokenizer = tokenizerFromArtifacts(artifacts);
  const wanted = new Set<number>();
  if (samples === 1) wanted.add(1);
  else {
    for (let index = 0; index < samples; index += 1)
      wanted.add(1 + Math.floor((index * (rows - 1)) / (samples - 1)));
  }

  const records: string[] = [];
  let lineNumber = 0;
  const reader = createInterface({
    input: createReadStream(dataPath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  for await (const text of reader) {
    if (text.length === 0) continue;
    lineNumber += 1;
    if (wanted.has(lineNumber)) {
      records.push(JSON.stringify({
        line_number: lineNumber,
        text,
        ids: Array.from(tokenizer.encode(text)),
      }));
    }
  }
  if (lineNumber !== rows) throw new Error(`observed ${lineNumber} rows, expected ${rows}`);
  if (records.length !== wanted.size)
    throw new Error(`captured ${records.length} samples, expected ${wanted.size}`);

  const temporary = `${outputPath}.tmp-${process.pid}`;
  await writeFile(temporary, `${records.join("\n")}\n`, { flag: "wx" });
  await rename(temporary, outputPath);
  process.stdout.write(`${JSON.stringify({
    result: "PASS",
    data: { path: dataPath, rows, sha256: await sha256(dataPath) },
    tokenizer: {
      path: artifactsPath,
      sha256: await sha256(artifactsPath),
      type: artifacts.type,
      vocabSize: tokenizer.vocabSize,
    },
    sample: { path: outputPath, rows: records.length, sha256: await sha256(outputPath) },
  }, null, 2)}\n`);
}

await main();
