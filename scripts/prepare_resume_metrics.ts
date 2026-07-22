#!/usr/bin/env npx tsx
/** Safely align an append-only metrics stream to the checkpoint used for resume. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { appendFile, copyFile, readFile, realpath, rename, stat, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import * as path from "node:path";

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

async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.run || !cli.checkpoint || !cli.sourceCommit) {
    throw new Error("required: --run, --checkpoint, and --sourceCommit");
  }
  const runDir = await realpath(cli.run);
  const checkpoint = await realpath(cli.checkpoint);
  if (path.dirname(checkpoint) !== runDir) throw new Error("checkpoint must be directly inside the run directory");
  const match = /^checkpoint-(\d+)\.json$/.exec(path.basename(checkpoint));
  if (!match) throw new Error("checkpoint filename must be checkpoint-<step>.json");
  const checkpointStep = Number(match[1]);
  if (!Number.isSafeInteger(checkpointStep) || checkpointStep < 1) throw new Error("invalid checkpoint step");
  const checkpointStat = await stat(checkpoint);
  if (checkpointStat.size === 0) throw new Error("checkpoint is empty");

  const metricsPath = path.join(runDir, "metrics.jsonl");
  const original = await readFile(metricsPath, "utf8");
  const lines = original.trim().split("\n").filter(Boolean);
  const rows = lines.map((line) => JSON.parse(line) as { step?: number });
  for (let index = 0; index < rows.length; index++) {
    if (rows[index].step !== index + 1) {
      throw new Error(`metrics are not sequential at row ${index + 1}: step=${String(rows[index].step)}`);
    }
  }
  const lastMetricStep = rows.at(-1)?.step ?? 0;
  if (lastMetricStep < checkpointStep) {
    throw new Error(`metrics end at ${lastMetricStep}, before checkpoint step ${checkpointStep}`);
  }

  const originalSha256 = createHash("sha256").update(original).digest("hex");
  let archivedTailPath: string | null = null;
  let activeMetricsSha256 = originalSha256;
  if (lastMetricStep > checkpointStep) {
    archivedTailPath = path.join(runDir, `metrics.pre-resume-checkpoint-${checkpointStep}-through-${lastMetricStep}.jsonl`);
    await copyFile(metricsPath, archivedTailPath, constants.COPYFILE_EXCL);
    if (await sha256File(archivedTailPath) !== originalSha256) {
      throw new Error("preserved metrics copy failed SHA-256 verification; active metrics left untouched");
    }
    const prefix = lines.slice(0, checkpointStep).join("\n") + "\n";
    activeMetricsSha256 = createHash("sha256").update(prefix).digest("hex");
    const tmp = `${metricsPath}.resume-${checkpointStep}.tmp`;
    await writeFile(tmp, prefix, { encoding: "utf8", flag: "wx" });
    await rename(tmp, metricsPath);
  }

  const record = {
    schema: "alpha-resume-ledger-v1",
    prepared_utc: new Date().toISOString(),
    source_commit: cli.sourceCommit,
    checkpoint: {
      path: checkpoint,
      step: checkpointStep,
      bytes: checkpointStat.size,
      sha256: await sha256File(checkpoint),
    },
    metrics: {
      last_step_before_prepare: lastMetricStep,
      active_last_step: checkpointStep,
      original_sha256: originalSha256,
      active_sha256: activeMetricsSha256,
      preserved_copy: archivedTailPath,
    },
  };
  await appendFile(path.join(runDir, "resume-ledger.jsonl"), JSON.stringify(record) + "\n", "utf8");
  console.log(JSON.stringify(record, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
