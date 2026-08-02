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

const V4_FOCUSES: readonly Focus[] = [
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

const V8_FOCUSES: readonly Focus[] = [
  {
    slug: "foundational-answer",
    batches: 10,
    prompt:
      "short direct answers about stable, broadly useful everyday knowledge",
  },
  {
    slug: "quantitative-reasoning",
    batches: 10,
    prompt:
      "small arithmetic, counting, comparison, and one- or two-step word problems",
  },
  {
    slug: "instruction-control",
    batches: 10,
    prompt:
      "literal compliance with concise constraints on wording, count, order, or layout",
  },
  {
    slug: "context-grounding",
    batches: 10,
    prompt:
      "answering from supplied fictional context instead of prior associations",
  },
  {
    slug: "negation-contrast",
    batches: 10,
    prompt:
      "negation, opposites, exclusions, corrections, and contrastive category boundaries",
  },
  {
    slug: "uncertainty-honesty",
    batches: 10,
    prompt:
      "unknown personal facts, missing evidence, tool limits, and calibrated uncertainty",
  },
  {
    slug: "language-pragmatics",
    batches: 10,
    prompt:
      "idioms, implicature, reference, ambiguity, tone, and intended conversational meaning",
  },
  {
    slug: "multi-turn-update",
    batches: 10,
    prompt:
      "follow-ups that correct, narrow, challenge, or apply earlier common ground",
  },
  {
    slug: "premise-resistance",
    batches: 10,
    prompt:
      "false premises, pressure to abandon a correct answer, and polite factual correction",
  },
  {
    slug: "ordinary-conversation",
    batches: 10,
    prompt:
      "natural greetings, reactions, opinions, creative continuations, and social exchanges",
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

function expectedBatches(
  prefix: string,
  focuses: readonly Focus[],
): Array<{
  batch_id: string;
  focus: string;
  purpose: string;
}> {
  const batches: Array<{ batch_id: string; focus: string; purpose: string }> =
    [];
  let index = 0;
  for (const focus of focuses) {
    for (let within = 0; within < focus.batches; within += 1) {
      batches.push({
        batch_id: `${prefix}-${String(index).padStart(3, "0")}-${focus.slug}`,
        focus: focus.slug,
        purpose: focus.prompt,
      });
      index += 1;
    }
  }
  return batches;
}

function validate(
  value: unknown,
  planId: string,
  expected: readonly { batch_id: string; focus: string; purpose: string }[],
): asserts value is Blueprint {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("malformed blueprint envelope");
  const record = value as Record<string, unknown>;
  if (record.plan_id !== planId)
    throw new Error(`unexpected plan_id ${String(record.plan_id)}`);
  if (!Array.isArray(record.batches))
    throw new Error("blueprint batches missing");
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
  const variant = args.get("variant") ?? "v4";
  if (variant !== "v4" && variant !== "v8")
    throw new Error(`unsupported variant: ${variant}`);
  const isV8 = variant === "v8";
  const focuses = isV8 ? V8_FOCUSES : V4_FOCUSES;
  const planId = isV8
    ? "alpha-chat-foundations-v8-planned-v1"
    : "alpha-chat-semantic-repair-v4-planned-v2";
  const expected = expectedBatches(variant, focuses);
  const out = resolve(
    args.get("out") ??
      (isV8
        ? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v8-20260802/planned/blueprint.json"
        : "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/planned-v2/blueprint.json"),
  );
  const events = resolve(
    args.get("events") ??
      (isV8
        ? "/opencode/logs/alpha-chat-foundations-v8/planned/blueprint.events.jsonl"
        : "/opencode/logs/alpha-chat-semantic-repair-v4/planned-v2/blueprint.events.jsonl"),
  );
  const model = args.get("model") ?? "gpt-5.5";
  const reasoning = args.get("reasoning-effort") ?? "high";
  const promptPath = resolve(
    args.get("prompt") ??
      resolve(
        repo,
        isV8
          ? "prompts/chat-foundations-v8-planner.md"
          : "prompts/chat-semantic-repair-planner.md",
      ),
  );
  const schemaPath = resolve(
    args.get("schema") ??
      resolve(
        repo,
        isV8
          ? "schemas/chat-foundations-v8-blueprint.schema.json"
          : "schemas/chat-semantic-repair-blueprint.schema.json",
      ),
  );
  const promptTemplate = await readFile(promptPath, "utf8");
  const prompt = `${promptTemplate}\n\n## Required plan\n\nPlan ID: ${planId}\nReserved release probes: ${RESERVED}\n\nBatches in required order:\n${expected
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
  validate(parsed, planId, expected);
  await rename(temporaryOut, out);
  const manifest = {
    schema: isV8
      ? "alpha-chat-foundations-v8-blueprint-manifest-v1"
      : "alpha-chat-semantic-repair-v4-blueprint-manifest-v1",
    variant,
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
