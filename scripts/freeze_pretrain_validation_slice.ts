#!/usr/bin/env npx tsx
/** Freeze a newline-aligned prefix of a held-out pretraining shard. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, rename, stat, writeFile } from "node:fs/promises";

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    result[arg.slice(2)] = value;
  }
  return result;
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const cli = parseArgs();
  for (const required of ["source", "out", "bytes", "sourceSha256", "manifest"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  if (!/^[0-9a-f]{64}$/.test(cli.sourceSha256)) throw new Error("--sourceSha256 must be a lowercase SHA-256");
  const requestedBytes = Number(cli.bytes);
  if (!Number.isSafeInteger(requestedBytes) || requestedBytes < 1_048_576) {
    throw new Error("--bytes must be an integer of at least 1 MiB");
  }
  const sourceStat = await stat(cli.source);
  if (!sourceStat.isFile() || sourceStat.size <= requestedBytes) {
    throw new Error(`source must be a file larger than ${requestedBytes} bytes`);
  }
  const sourceSha256 = await sha256File(cli.source);
  if (sourceSha256 !== cli.sourceSha256) {
    throw new Error(`source SHA-256 ${sourceSha256} != ${cli.sourceSha256}`);
  }

  const source = await open(cli.source, "r");
  const buffer = Buffer.allocUnsafe(requestedBytes);
  try {
    let offset = 0;
    while (offset < requestedBytes) {
      const read = await source.read(buffer, offset, requestedBytes - offset, offset);
      if (read.bytesRead === 0) throw new Error("source ended before requested validation slice");
      offset += read.bytesRead;
    }
  } finally {
    await source.close();
  }
  const finalNewline = buffer.lastIndexOf(0x0a);
  if (finalNewline < 0) throw new Error("requested prefix contains no newline boundary");
  const frozen = buffer.subarray(0, finalNewline + 1);
  const outputTmp = `${cli.out}.tmp`;
  await writeFile(outputTmp, frozen, { flag: "wx" });
  await rename(outputTmp, cli.out);
  const outputSha256 = await sha256File(cli.out);
  const manifest = {
    schema: "alpha-pretrain-validation-slice-v1",
    source: {
      path: cli.source,
      bytes: sourceStat.size,
      sha256: sourceSha256,
    },
    selection: {
      method: "newline-aligned-prefix",
      requested_bytes: requestedBytes,
      selected_bytes: frozen.length,
    },
    output: {
      path: cli.out,
      bytes: frozen.length,
      sha256: outputSha256,
    },
  };
  await writeFile(cli.manifest, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(manifest, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
