#!/usr/bin/env npx tsx
/** Safely retain only the newest N mirrored training checkpoints. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readdir, realpath, stat, unlink } from "node:fs/promises";
import * as path from "node:path";

interface CheckpointFile {
  basename: string;
  path: string;
  step: number;
}

function parseArgs(): { run: string; keep: number } {
  const values: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    values[arg.slice(2)] = value;
  }
  if (!values.run || !values.keep) throw new Error("required: --run and --keep");
  const keep = Number(values.keep);
  if (!Number.isSafeInteger(keep) || keep < 3) throw new Error("--keep must be an integer >= 3");
  return { run: values.run, keep };
}

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function appendLedger(ledgerPath: string, record: Record<string, unknown>): Promise<void> {
  const handle = await open(ledgerPath, "a", 0o664);
  try {
    await handle.write(`${JSON.stringify(record)}\n`);
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function main(): Promise<void> {
  const cli = parseArgs();
  const runDir = await realpath(cli.run);
  const allowedRoot = "/mnt/donto-data/alpha-runs";
  if (runDir === allowedRoot || !runDir.startsWith(`${allowedRoot}/`)) {
    throw new Error(`run directory must resolve below ${allowedRoot}: ${runDir}`);
  }

  const entries = await readdir(runDir, { withFileTypes: true });
  const checkpoints: CheckpointFile[] = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const match = /^checkpoint-([1-9][0-9]*)\.json$/.exec(entry.name);
    if (!match) continue;
    const step = Number(match[1]);
    if (!Number.isSafeInteger(step)) throw new Error(`checkpoint step is unsafe: ${entry.name}`);
    checkpoints.push({ basename: entry.name, path: path.join(runDir, entry.name), step });
  }
  checkpoints.sort((left, right) => left.step - right.step);
  for (let index = 1; index < checkpoints.length; index++) {
    if (checkpoints[index - 1].step === checkpoints[index].step) {
      throw new Error(`duplicate checkpoint step ${checkpoints[index].step}`);
    }
  }

  const retained = checkpoints.slice(-cli.keep);
  for (const checkpoint of retained) {
    const info = await stat(checkpoint.path);
    if (!info.isFile() || info.size <= 0) throw new Error(`retained checkpoint is empty: ${checkpoint.basename}`);
  }
  const candidates = checkpoints.slice(0, Math.max(0, checkpoints.length - cli.keep));
  const ledgerPath = path.join(runDir, "checkpoint-prune-ledger.jsonl");
  const removed: { basename: string; step: number; bytes: number; sha256: string }[] = [];
  for (const checkpoint of candidates) {
    const before = await stat(checkpoint.path);
    if (!before.isFile() || before.size <= 0) throw new Error(`prune candidate is empty: ${checkpoint.basename}`);
    const sha256 = await sha256File(checkpoint.path);
    const after = await stat(checkpoint.path);
    if (before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size || before.mtimeMs !== after.mtimeMs) {
      throw new Error(`prune candidate changed while hashing: ${checkpoint.basename}`);
    }
    const common = {
      schema: "alpha-local-checkpoint-prune-v1",
      recorded_utc: new Date().toISOString(),
      run_dir: runDir,
      checkpoint: { basename: checkpoint.basename, step: checkpoint.step, bytes: before.size, sha256 },
      retained: retained.map((item) => item.basename),
      keep: cli.keep,
      reason: "bounded generated-checkpoint retention after verified RunPod mirror",
    };
    await appendLedger(ledgerPath, { ...common, state: "delete_committed" });
    await unlink(checkpoint.path);
    await appendLedger(ledgerPath, { ...common, recorded_utc: new Date().toISOString(), state: "deleted" });
    removed.push({ basename: checkpoint.basename, step: checkpoint.step, bytes: before.size, sha256 });
  }

  process.stdout.write(JSON.stringify({
    result: "PASS",
    run_dir: runDir,
    keep: cli.keep,
    retained: retained.map((item) => item.basename),
    removed,
    ledger: ledgerPath,
  }) + "\n");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
