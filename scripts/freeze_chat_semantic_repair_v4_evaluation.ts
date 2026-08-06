#!/usr/bin/env npx tsx

/** Bind v4 corpus and visible development suites without opening the sealed final. */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

interface Evidence {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly rows?: number;
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index]!;
    const value = argv[index + 1];
    if (!name.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`invalid argument near ${name}`);
    values.set(name.slice(2), value);
    index += 1;
  }
  return values;
}

function sha256(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

async function evidence(path: string): Promise<Evidence> {
  const resolved = resolve(path);
  const bytes = await readFile(resolved);
  return { path: resolved, bytes: bytes.byteLength, sha256: sha256(bytes) };
}

async function jsonlEvidence(path: string): Promise<Evidence> {
  const result = await evidence(path);
  const text = await readFile(result.path, "utf8");
  const rows = text.split("\n").filter((line) => line.trim().length > 0).length;
  if (rows === 0) throw new Error(`empty JSONL input: ${result.path}`);
  return { ...result, rows };
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${label} is not an object`);
  return value as Record<string, unknown>;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const required = (name: string): string => {
    const value = args.get(name);
    if (!value) throw new Error(`--${name} is required`);
    return value;
  };
  const out = resolve(required("out"));
  try {
    await readFile(out);
    throw new Error(`output already exists: ${out}`);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  const corpusManifestEvidence = await evidence(required("corpus-manifest"));
  const corpus = object(
    JSON.parse(await readFile(corpusManifestEvidence.path, "utf8")) as unknown,
    "corpus manifest",
  );
  if (corpus.schema !== "alpha-chat-semantic-repair-v4-corpus-manifest-v1")
    throw new Error("unexpected corpus manifest schema");

  const priorFreezeEvidence = await evidence(required("prior-freeze"));
  const priorFreeze = object(
    JSON.parse(await readFile(priorFreezeEvidence.path, "utf8")) as unknown,
    "prior freeze",
  );
  if (priorFreeze.schema !== "alpha-chat-repair-v2-eval-freeze-v1")
    throw new Error("unexpected prior freeze schema");
  if (priorFreeze.status !== "development-visible; final-sealed-unexecuted")
    throw new Error("prior sealed-final state changed");
  const priorOutputs = object(priorFreeze.outputs, "prior freeze outputs");
  const priorSealed = object(priorOutputs.sealedFinal, "prior sealed final");
  const sealedEvidence = await evidence(String(priorSealed.path));
  if (sealedEvidence.sha256 !== priorSealed.sha256)
    throw new Error("sealed-final hash changed");

  const visible = {
    selector: await jsonlEvidence(required("selector")),
    panel: await jsonlEvidence(required("panel")),
    regression: await jsonlEvidence(required("regression")),
    releaseProbes: await jsonlEvidence(required("release-probes")),
  };
  const manifest = {
    schema: "alpha-chat-semantic-repair-v4-evaluation-freeze-v1",
    status: "development-visible; inherited-final-sealed-unexecuted",
    frozen_utc: new Date().toISOString(),
    source_commit: execFileSync("git", ["rev-parse", "HEAD"], {
      encoding: "utf8",
    }).trim(),
    inputs: {
      corpus_manifest: corpusManifestEvidence,
      corpus_outputs: corpus.outputs,
      prior_freeze: priorFreezeEvidence,
    },
    visible_development: visible,
    sealed_final: {
      ...sealedEvidence,
      inherited_from: priorFreezeEvidence.path,
      execution_policy:
        "do not execute until one v4 checkpoint beats the public baseline on visible free generation",
    },
    selection: {
      primary:
        "semantic contingency and conversational correctness on held-out families",
      mechanical_prerequisites: [
        "nonempty response",
        "no immediate EOS regression",
        "no role leak",
        "no repetition-loop regression",
      ],
      validation_loss_can_select: false,
      public_baseline_required: true,
    },
  };
  const temporary = `${out}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, {
    flag: "wx",
  });
  await rename(temporary, out);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", out, sha256: sha256(await readFile(out)) })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
