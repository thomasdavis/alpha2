#!/usr/bin/env npx tsx

/**
 * Resumable structured generation for Alpha semantic-chat repair v4.
 *
 * The teacher produces researcher-visible metadata plus natural conversations.
 * Every completed batch is independently schema-constrained, validated, hashed,
 * and accompanied by its raw Codex event log under /opencode/logs.
 */

import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

interface Turn {
  readonly role: "user" | "assistant";
  readonly content: string;
}

interface Candidate {
  readonly candidate_id: string;
  readonly category: string;
  readonly intent_summary: string;
  readonly skills: readonly string[];
  readonly turns: readonly Turn[];
  readonly why_useful: string;
}

interface BatchOutput {
  readonly batch_id: string;
  readonly items: readonly Candidate[];
}

interface Focus {
  readonly slug: string;
  readonly batches: number;
  readonly prompt: string;
}

interface BatchPlan {
  readonly index: number;
  readonly batchId: string;
  readonly focus: Focus;
  readonly candidateIds: readonly string[];
  readonly singleCount: number;
}

const FOCUSES: readonly Focus[] = [
  {
    slug: "explanation",
    batches: 10,
    prompt:
      "Give plain, accurate explanations of ordinary stable ideas and mechanisms. Prefer understanding over fact trivia. Cover unrelated domains and let the user sound naturally curious rather than like a benchmark.",
  },
  {
    slug: "distinction",
    batches: 10,
    prompt:
      "Clarify and apply useful conceptual distinctions. State the boundary, consequence, or counterexample that makes the distinction matter; do not merely say the terms are different.",
  },
  {
    slug: "pragmatics",
    batches: 8,
    prompt:
      "Respond to pragmatic and emotional intent: acknowledgement without forced advice, gentle challenge, indirect requests, tact, frustration, uncertainty, and mismatches between literal words and the conversational need.",
  },
  {
    slug: "language",
    batches: 7,
    prompt:
      "Discuss meaning in language through natural examples: reference, implication, presupposition, scope, word senses, category boundaries, paraphrase, and context. Explain the source of a reading rather than invoking vague ambiguity.",
  },
  {
    slug: "evidence",
    batches: 6,
    prompt:
      "Reason carefully from short user-supplied observations or testimony. Separate observation, report, inference, confidence, and what remains unknown without becoming bureaucratic or overcautious.",
  },
  {
    slug: "repair",
    batches: 5,
    prompt:
      "Handle correction and disagreement constructively. Locate the real disagreement, concede or revise locally when warranted, and preserve unaffected points instead of restarting or becoming defensive.",
  },
  {
    slug: "common-ground",
    batches: 4,
    prompt:
      "Build common ground across two exchanges. The follow-up changes, narrows, challenges, or applies the first answer, and the final reply must visibly use both turns without restating everything.",
  },
] as const;

const RESERVED_LIVE_PROBE =
  "Do not generate conversations about DNA, the contrast between promises and predictions, uncertainty about one's life direction, a user reporting a terrible day while explicitly refusing advice, replacement of every committee member, or the sentence 'I saw her duck'. These are held-out release probes.";

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index]!;
    if (!item.startsWith("--")) throw new Error(`unexpected argument: ${item}`);
    const name = item.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--"))
      throw new Error(`missing value for --${name}`);
    parsed.set(name, value);
    index += 1;
  }
  return parsed;
}

function positiveInteger(
  value: string | undefined,
  fallback: number,
  name: string,
): number {
  const resolved = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(resolved) || resolved <= 0)
    throw new Error(`${name} must be a positive integer`);
  return resolved;
}

function plans(
  itemsPerBatch: number,
  maximumBatches: number,
): readonly BatchPlan[] {
  const all: BatchPlan[] = [];
  let index = 0;
  for (const focus of FOCUSES) {
    for (let within = 0; within < focus.batches; within += 1) {
      const batchId = `v4-${String(index).padStart(3, "0")}-${focus.slug}`;
      const candidateIds = Array.from(
        { length: itemsPerBatch },
        (_, item) => `${batchId}-${String(item).padStart(2, "0")}`,
      );
      all.push({
        index,
        batchId,
        focus,
        candidateIds,
        singleCount: Math.ceil(itemsPerBatch * 0.65),
      });
      index += 1;
    }
  }
  return all.slice(0, maximumBatches);
}

function batchPrompt(template: string, plan: BatchPlan): string {
  const singleIds = plan.candidateIds.slice(0, plan.singleCount);
  const doubleIds = plan.candidateIds.slice(plan.singleCount);
  return `${template}\n\n## Batch specification\n\nBatch ID: ${plan.batchId}\nCapability focus: ${plan.focus.prompt}\n\nReturn exactly ${plan.candidateIds.length} items in this exact order:\n${plan.candidateIds.join(", ")}\n\nConversation allocation:\n- The following ${singleIds.length} items contain exactly one user/assistant exchange: ${singleIds.join(", ")}\n- The following ${doubleIds.length} items contain exactly two user/assistant exchanges: ${doubleIds.join(", ")}\n\nEvery item must use a different situation and a materially different user intent. Avoid neighboring paraphrases within the batch. ${RESERVED_LIVE_PROBE}\n`;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateBatch(
  value: unknown,
  plan: BatchPlan,
): asserts value is BatchOutput {
  if (
    !isObject(value) ||
    value.batch_id !== plan.batchId ||
    !Array.isArray(value.items)
  ) {
    throw new Error(`${plan.batchId}: malformed batch envelope`);
  }
  if (value.items.length !== plan.candidateIds.length) {
    throw new Error(
      `${plan.batchId}: expected ${plan.candidateIds.length} items, got ${value.items.length}`,
    );
  }
  const normalizedUsers = new Set<string>();
  value.items.forEach((rawItem, itemIndex) => {
    if (!isObject(rawItem))
      throw new Error(`${plan.batchId}: item ${itemIndex} is not an object`);
    const expectedId = plan.candidateIds[itemIndex]!;
    if (rawItem.candidate_id !== expectedId) {
      throw new Error(
        `${plan.batchId}: item ${itemIndex} id ${String(rawItem.candidate_id)} != ${expectedId}`,
      );
    }
    if (!Array.isArray(rawItem.turns))
      throw new Error(`${expectedId}: turns missing`);
    const expectedTurns = itemIndex < plan.singleCount ? 2 : 4;
    if (rawItem.turns.length !== expectedTurns) {
      throw new Error(
        `${expectedId}: expected ${expectedTurns} turns, got ${rawItem.turns.length}`,
      );
    }
    rawItem.turns.forEach((rawTurn, turnIndex) => {
      if (!isObject(rawTurn))
        throw new Error(`${expectedId}: turn ${turnIndex} is not an object`);
      const expectedRole = turnIndex % 2 === 0 ? "user" : "assistant";
      if (rawTurn.role !== expectedRole) {
        throw new Error(
          `${expectedId}: turn ${turnIndex} role ${String(rawTurn.role)} != ${expectedRole}`,
        );
      }
      if (
        typeof rawTurn.content !== "string" ||
        rawTurn.content.trim().length === 0
      ) {
        throw new Error(`${expectedId}: turn ${turnIndex} has empty content`);
      }
      if (/<\|(user|assistant|end_of_text)\|>/.test(rawTurn.content)) {
        throw new Error(
          `${expectedId}: model-visible marker leaked into turn ${turnIndex}`,
        );
      }
      if (expectedRole === "user") {
        const normalized = rawTurn.content
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, " ")
          .trim();
        if (normalizedUsers.has(normalized))
          throw new Error(`${expectedId}: duplicate normalized user turn`);
        normalizedUsers.add(normalized);
      }
    });
  });
}

async function directoryKiB(path: string): Promise<number> {
  try {
    const output = execFileSync("du", ["-sk", path], {
      encoding: "utf8",
    }).trim();
    return Number(output.split(/\s+/)[0]);
  } catch {
    return 0;
  }
}

async function runCodex(
  plan: BatchPlan,
  prompt: string,
  outputPath: string,
  eventPath: string,
  schemaPath: string,
  repoRoot: string,
  model: string,
  reasoningEffort: string,
): Promise<void> {
  const temporaryOutput = `${outputPath}.tmp-${process.pid}`;
  const temporaryEvents = `${eventPath}.tmp-${process.pid}`;
  await rm(temporaryOutput, { force: true });
  await rm(temporaryEvents, { force: true });

  await new Promise<void>((resolvePromise, rejectPromise) => {
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
        `model_reasoning_effort=\"${reasoningEffort}\"`,
        "--ephemeral",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-C",
        repoRoot,
        "--output-schema",
        schemaPath,
        "-o",
        temporaryOutput,
        "--json",
        "-",
      ],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    const eventChunks: Buffer[] = [];
    const errorChunks: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => eventChunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => errorChunks.push(chunk));
    child.on("error", rejectPromise);
    child.on("close", async (code) => {
      const events = Buffer.concat(eventChunks);
      const errors = Buffer.concat(errorChunks);
      await writeFile(temporaryEvents, Buffer.concat([events, errors]));
      await rename(temporaryEvents, eventPath);
      if (code !== 0) {
        rejectPromise(
          new Error(
            `${plan.batchId}: codex exited ${String(code)}; preserved ${eventPath}`,
          ),
        );
        return;
      }
      resolvePromise();
    });
    child.stdin.end(prompt);
  });

  const parsed: unknown = JSON.parse(await readFile(temporaryOutput, "utf8"));
  validateBatch(parsed, plan);
  await rename(temporaryOutput, outputPath);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = resolve(args.get("repo") ?? process.cwd());
  const outputRoot = resolve(
    args.get("out-dir") ??
      "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/generation",
  );
  const logRoot = resolve(
    args.get("log-dir") ?? "/opencode/logs/alpha-chat-semantic-repair-v4",
  );
  const itemsPerBatch = positiveInteger(
    args.get("items-per-batch"),
    40,
    "items-per-batch",
  );
  if (itemsPerBatch > 64)
    throw new Error("items-per-batch cannot exceed schema maximum 64");
  const maximumBatches = positiveInteger(args.get("batches"), 50, "batches");
  const workers = positiveInteger(args.get("workers"), 2, "workers");
  const model = args.get("model") ?? "gpt-5.4";
  const reasoningEffort = args.get("reasoning-effort") ?? "medium";
  const schemaPath = resolve(
    repoRoot,
    "schemas/chat-semantic-repair-candidates.schema.json",
  );
  const templatePath = resolve(
    repoRoot,
    "prompts/chat-semantic-repair-generator.md",
  );
  const template = await readFile(templatePath, "utf8");
  const selectedPlans = plans(itemsPerBatch, maximumBatches);
  if (selectedPlans.length !== maximumBatches) {
    throw new Error(
      `requested ${maximumBatches} batches but the declared plan contains ${selectedPlans.length}`,
    );
  }

  const batchRoot = join(outputRoot, "batches");
  await mkdir(batchRoot, { recursive: true });
  await mkdir(logRoot, { recursive: true });
  const queue = [...selectedPlans];
  const completed: Array<Record<string, unknown>> = [];

  async function worker(workerIndex: number): Promise<void> {
    while (queue.length > 0) {
      const plan = queue.shift();
      if (!plan) return;
      const outputPath = join(batchRoot, `${plan.batchId}.json`);
      const eventPath = join(logRoot, `${plan.batchId}.events.jsonl`);
      try {
        const existing = JSON.parse(
          await readFile(outputPath, "utf8"),
        ) as unknown;
        validateBatch(existing, plan);
        const content = await readFile(outputPath);
        completed.push({
          batch_id: plan.batchId,
          status: "existing",
          path: outputPath,
          bytes: content.byteLength,
          sha256: sha256(content),
        });
        process.stdout.write(
          `[worker ${workerIndex}] ${plan.batchId}: existing valid batch\n`,
        );
        continue;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
          process.stdout.write(
            `[worker ${workerIndex}] ${plan.batchId}: regenerating invalid batch\n`,
          );
        }
      }

      const usedKiB = await directoryKiB(outputRoot);
      if (usedKiB > 15 * 1024 * 1024) {
        throw new Error(
          `generation output exceeded 15 GiB pause threshold: ${usedKiB} KiB`,
        );
      }
      process.stdout.write(
        `[worker ${workerIndex}] ${plan.batchId}: generating with ${model}\n`,
      );
      const prompt = batchPrompt(template, plan);
      await runCodex(
        plan,
        prompt,
        outputPath,
        eventPath,
        schemaPath,
        repoRoot,
        model,
        reasoningEffort,
      );
      const content = await readFile(outputPath);
      const eventContent = await readFile(eventPath);
      completed.push({
        batch_id: plan.batchId,
        status: "generated",
        path: outputPath,
        bytes: content.byteLength,
        sha256: sha256(content),
        events_path: eventPath,
        events_bytes: eventContent.byteLength,
        events_sha256: sha256(eventContent),
      });
      process.stdout.write(
        `[worker ${workerIndex}] ${plan.batchId}: accepted structured batch\n`,
      );
    }
  }

  await Promise.all(
    Array.from({ length: workers }, (_, index) => worker(index + 1)),
  );
  completed.sort((left, right) =>
    String(left.batch_id).localeCompare(String(right.batch_id)),
  );
  const codexVersion = execFileSync("codex", ["--version"], {
    encoding: "utf8",
  }).trim();
  const manifest = {
    schema: "alpha-chat-semantic-repair-v4-generation-manifest-v1",
    generated_utc: new Date().toISOString(),
    model,
    reasoning_effort: reasoningEffort,
    codex_version: codexVersion,
    prompt: {
      path: templatePath,
      sha256: sha256(template),
    },
    output_schema: {
      path: schemaPath,
      sha256: sha256(await readFile(schemaPath)),
    },
    requested_batches: maximumBatches,
    items_per_batch: itemsPerBatch,
    expected_candidates: maximumBatches * itemsPerBatch,
    workers,
    completed,
  };
  const manifestPath = join(outputRoot, "generation-manifest.json");
  const temporaryManifest = `${manifestPath}.tmp-${process.pid}`;
  await writeFile(temporaryManifest, `${JSON.stringify(manifest, null, 2)}\n`);
  await rename(temporaryManifest, manifestPath);
  const finalStat = await stat(manifestPath);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", manifest: manifestPath, bytes: finalStat.size, batches: completed.length })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
