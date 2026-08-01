#!/usr/bin/env npx tsx

/** Produce one strong-model semantic allocation before cheaper generation. */

import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

interface Focus {
  readonly slug: string;
  readonly batches: number;
  readonly prompt: string;
}

interface BlueprintBatch {
  readonly batch_id: string;
  readonly focus: string;
  readonly semantic_territory: string;
  readonly coverage_targets: readonly string[];
  readonly excluded_cliches: readonly string[];
  readonly variation_notes: readonly string[];
}

interface Blueprint {
  readonly plan_id: string;
  readonly batches: readonly BlueprintBatch[];
}

const FOCUSES: readonly Focus[] = [
  {
    slug: "explanation",
    batches: 10,
    prompt: "plain explanations of stable ordinary ideas and mechanisms",
  },
  {
    slug: "distinction",
    batches: 10,
    prompt: "useful conceptual boundaries, consequences, and counterexamples",
  },
  {
    slug: "pragmatics",
    batches: 8,
    prompt:
      "emotional and pragmatic intent, tact, implication, and conversational policy",
  },
  {
    slug: "language",
    batches: 7,
    prompt:
      "meaning, reference, scope, word senses, presupposition, and context",
  },
  {
    slug: "evidence",
    batches: 6,
    prompt: "observation, report, inference, source, confidence, and unknowns",
  },
  {
    slug: "repair",
    batches: 5,
    prompt: "correction, disagreement, concession, and local revision",
  },
  {
    slug: "common-ground",
    batches: 4,
    prompt:
      "delayed reuse, narrowing, challenge, and shared conversational meaning",
  },
] as const;

const RESERVED =
  "DNA; promises versus predictions; uncertainty about one's life direction; a terrible-day disclosure that explicitly refuses advice; replacing every committee member; the sentence 'I saw her duck'.";

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index]!;
    if (!item.startsWith("--")) throw new Error(`unexpected argument: ${item}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--"))
      throw new Error(`missing value for ${item}`);
    parsed.set(item.slice(2), value);
    index += 1;
  }
  return parsed;
}

function expectedBatches(): Array<{
  batch_id: string;
  focus: string;
  purpose: string;
}> {
  const batches: Array<{ batch_id: string; focus: string; purpose: string }> =
    [];
  let index = 0;
  for (const focus of FOCUSES) {
    for (let within = 0; within < focus.batches; within += 1) {
      batches.push({
        batch_id: `v4-${String(index).padStart(3, "0")}-${focus.slug}`,
        focus: focus.slug,
        purpose: focus.prompt,
      });
      index += 1;
    }
  }
  return batches;
}

function validate(value: unknown): asserts value is Blueprint {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("malformed blueprint envelope");
  const record = value as Record<string, unknown>;
  if (record.plan_id !== "alpha-chat-semantic-repair-v4-planned-v2")
    throw new Error(`unexpected plan_id ${String(record.plan_id)}`);
  if (!Array.isArray(record.batches))
    throw new Error("blueprint batches missing");
  const expected = expectedBatches();
  if (record.batches.length !== expected.length)
    throw new Error(
      `expected ${expected.length} batches, got ${record.batches.length}`,
    );
  record.batches.forEach((raw, index) => {
    if (typeof raw !== "object" || raw === null || Array.isArray(raw))
      throw new Error(`batch ${index} is malformed`);
    const batch = raw as Record<string, unknown>;
    if (
      batch.batch_id !== expected[index]!.batch_id ||
      batch.focus !== expected[index]!.focus
    )
      throw new Error(`batch ${index} does not match declared order`);
    for (const field of [
      "coverage_targets",
      "excluded_cliches",
      "variation_notes",
    ] as const) {
      const values = batch[field];
      if (!Array.isArray(values) || new Set(values).size !== values.length)
        throw new Error(`${String(batch.batch_id)} has invalid ${field}`);
    }
  });
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repo = resolve(args.get("repo") ?? process.cwd());
  const out = resolve(
    args.get("out") ??
      "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/planned-v2/blueprint.json",
  );
  const events = resolve(
    args.get("events") ??
      "/opencode/logs/alpha-chat-semantic-repair-v4/planned-v2/blueprint.events.jsonl",
  );
  const model = args.get("model") ?? "gpt-5.5";
  const reasoning = args.get("reasoning-effort") ?? "high";
  const promptPath = resolve(repo, "prompts/chat-semantic-repair-planner.md");
  const schemaPath = resolve(
    repo,
    "schemas/chat-semantic-repair-blueprint.schema.json",
  );
  const promptTemplate = await readFile(promptPath, "utf8");
  const prompt = `${promptTemplate}\n\n## Required plan\n\nPlan ID: alpha-chat-semantic-repair-v4-planned-v2\nReserved release probes: ${RESERVED}\n\nBatches in required order:\n${expectedBatches()
    .map((batch) => `${batch.batch_id} | ${batch.focus} | ${batch.purpose}`)
    .join("\n")}\n`;
  await mkdir(dirname(out), { recursive: true });
  await mkdir(dirname(events), { recursive: true });
  const temporaryOut = `${out}.tmp-${process.pid}`;
  const temporaryEvents = `${events}.tmp-${process.pid}`;
  await rm(temporaryOut, { force: true });
  await rm(temporaryEvents, { force: true });
  await new Promise<void>((accept, reject) => {
    const child = spawn(
      "nice",
      [
        "-n",
        "10",
        "codex",
        "exec",
        "-m",
        model,
        "-c",
        `model_reasoning_effort=\"${reasoning}\"`,
        "--ephemeral",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-C",
        repo,
        "--output-schema",
        schemaPath,
        "-o",
        temporaryOut,
        "--json",
        "-",
      ],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    const chunks: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.on("error", reject);
    child.on("close", async (code) => {
      await writeFile(temporaryEvents, Buffer.concat(chunks));
      await rename(temporaryEvents, events);
      code === 0
        ? accept()
        : reject(new Error(`planner exited ${String(code)}`));
    });
    child.stdin.end(prompt);
  });
  const parsed: unknown = JSON.parse(await readFile(temporaryOut, "utf8"));
  validate(parsed);
  await rename(temporaryOut, out);
  const manifest = {
    schema: "alpha-chat-semantic-repair-v4-blueprint-manifest-v1",
    created_utc: new Date().toISOString(),
    model,
    reasoning_effort: reasoning,
    codex_version: execFileSync("codex", ["--version"], {
      encoding: "utf8",
    }).trim(),
    blueprint: { path: out, sha256: sha256(await readFile(out)) },
    prompt: { path: promptPath, sha256: sha256(promptTemplate) },
    output_schema: {
      path: schemaPath,
      sha256: sha256(await readFile(schemaPath)),
    },
    events: { path: events, sha256: sha256(await readFile(events)) },
  };
  const manifestPath = `${out}.manifest.json`;
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", out, manifest: manifestPath, batches: parsed.batches.length })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
