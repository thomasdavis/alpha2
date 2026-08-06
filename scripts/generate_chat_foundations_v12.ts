#!/usr/bin/env npx tsx

/** Resumable GPT-5.4 generation of Alpha V12 linked contrast families. */

import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { join, resolve } from "node:path";

type Role = "user" | "assistant";

interface Turn {
  readonly role: Role;
  readonly content: string;
}

const SCENE_KINDS = [
  "base",
  "paraphrase",
  "minimal_change",
  "irrelevant_detail",
  "hard_negative",
  "cross_domain_transfer",
  "compare_and_apply",
  "update_and_revise",
] as const;

interface Scene {
  readonly scene_id: string;
  readonly kind: (typeof SCENE_KINDS)[number];
  readonly turns: readonly Turn[];
  readonly relation_to_base: string;
  readonly learning_signal: string;
}

interface Family {
  readonly family_id: string;
  readonly focus: string;
  readonly operation: string;
  readonly invariant: string;
  readonly must_change: string;
  readonly forbidden_shortcut: string;
  readonly scenes: readonly Scene[];
}

interface BatchOutput {
  readonly batch_id: string;
  readonly families: readonly Family[];
}

interface Focus {
  readonly slug: string;
  readonly brief: string;
}

interface Plan {
  readonly index: number;
  readonly batchId: string;
  readonly focus: Focus;
  readonly familyIds: readonly string[];
}

const FOCUSES: readonly Focus[] = [
  {
    slug: "perform-transform",
    brief:
      "Perform small arithmetic, counting, ordering, comparison, and one- or two-step transformations instead of restating the operands. Use varied quantities and non-school settings.",
  },
  {
    slug: "literal-instruction",
    brief:
      "Follow compact natural constraints on word count, ordering, selection, copying, completion, or layout. Use ordinary content, never code or JSON.",
  },
  {
    slug: "negation-category",
    brief:
      "Apply negation, exclusion, antonymy, membership, exception, and category boundaries. Produce a valid instance rather than echoing the requested category.",
  },
  {
    slug: "context-binding",
    brief:
      "Bind invented names, roles, locations, quantities, temporal facts, and relations from supplied context. Test which details matter and which are distractors.",
  },
  {
    slug: "update-revision",
    brief:
      "Revise an answer after a correction, retraction, time change, or new constraint while preserving unaffected commitments and the conversation's prior common ground.",
  },
  {
    slug: "premise-evidence",
    brief:
      "Challenge false or unsupported premises, distinguish report from observation and inference, and remain stable under social pressure without becoming combative.",
  },
  {
    slug: "language-pragmatics",
    brief:
      "Interpret idioms, indirect requests, implication, presupposition, reference, tone, lexical ambiguity, and conversational intent through context rather than keyword matching.",
  },
  {
    slug: "uncertainty-scope",
    brief:
      "Handle missing personal knowledge, limited evidence, scope, modality, and what would resolve uncertainty. Avoid both fabrication and generic disclaimers.",
  },
  {
    slug: "ordinary-presence",
    brief:
      "Sustain warm, direct ordinary conversation: greetings, reactions, opinions, small creative continuations, tactful disagreement, and answer-and-stop behavior without canned questions.",
  },
  {
    slug: "conceptual-structure",
    brief:
      "Reason conversationally about roles and bearers, events and objects, parts and members, groups, identity through change, purpose-relative categories, and related language distinctions.",
  },
] as const;

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`expected --name value, received ${String(key)}`);
    parsed.set(key.slice(2), value);
    index += 1;
  }
  return parsed;
}

function positiveInteger(value: string | undefined, fallback: number, name: string): number {
  const resolved = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(resolved) || resolved <= 0)
    throw new Error(`${name} must be a positive integer`);
  return resolved;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function plans(batchesPerFocus: number, maximumBatches: number): readonly Plan[] {
  const all: Plan[] = [];
  let index = 0;
  for (const focus of FOCUSES) {
    for (let within = 0; within < batchesPerFocus; within += 1) {
      const batchId = `v12-${String(index).padStart(3, "0")}-${focus.slug}`;
      all.push({
        index,
        batchId,
        focus,
        familyIds: Array.from(
          { length: 4 },
          (_, family) => `${batchId}-f${family}`,
        ),
      });
      index += 1;
    }
  }
  return all.slice(0, maximumBatches);
}

function normalizeUser(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function validateBatch(value: unknown, plan: Plan): asserts value is BatchOutput {
  if (!isObject(value) || value.batch_id !== plan.batchId || !Array.isArray(value.families))
    throw new Error(`${plan.batchId}: malformed batch envelope`);
  if (value.families.length !== plan.familyIds.length)
    throw new Error(`${plan.batchId}: expected four families`);
  const users = new Set<string>();
  value.families.forEach((rawFamily, familyIndex) => {
    if (!isObject(rawFamily)) throw new Error(`${plan.batchId}: family is not an object`);
    const familyId = plan.familyIds[familyIndex]!;
    if (rawFamily.family_id !== familyId || rawFamily.focus !== plan.focus.slug)
      throw new Error(`${plan.batchId}: family identity drift at ${familyIndex}`);
    for (const field of ["operation", "invariant", "must_change", "forbidden_shortcut"])
      if (typeof rawFamily[field] !== "string" || !rawFamily[field].trim())
        throw new Error(`${familyId}: empty ${field}`);
    if (!Array.isArray(rawFamily.scenes) || rawFamily.scenes.length !== SCENE_KINDS.length)
      throw new Error(`${familyId}: expected eight scenes`);
    rawFamily.scenes.forEach((rawScene, sceneIndex) => {
      if (!isObject(rawScene)) throw new Error(`${familyId}: invalid scene ${sceneIndex}`);
      const expectedKind = SCENE_KINDS[sceneIndex]!;
      if (rawScene.kind !== expectedKind || rawScene.scene_id !== `${familyId}-${expectedKind}`)
        throw new Error(`${familyId}: scene ${sceneIndex} identity/order drift`);
      for (const field of ["relation_to_base", "learning_signal"])
        if (typeof rawScene[field] !== "string" || !rawScene[field].trim())
          throw new Error(`${rawScene.scene_id}: empty ${field}`);
      const expectedTurns = sceneIndex < 6 ? 2 : 4;
      if (!Array.isArray(rawScene.turns) || rawScene.turns.length !== expectedTurns)
        throw new Error(`${rawScene.scene_id}: expected ${expectedTurns} turns`);
      rawScene.turns.forEach((rawTurn, turnIndex) => {
        if (!isObject(rawTurn)) throw new Error(`${rawScene.scene_id}: invalid turn`);
        const role = turnIndex % 2 === 0 ? "user" : "assistant";
        if (rawTurn.role !== role || typeof rawTurn.content !== "string" || !rawTurn.content.trim())
          throw new Error(`${rawScene.scene_id}: invalid ${role} turn ${turnIndex}`);
        if (rawTurn.content.length > 1_200)
          throw new Error(`${rawScene.scene_id}: turn too long`);
        if (/<\|(user|assistant|end_of_text)\|>/.test(rawTurn.content))
          throw new Error(`${rawScene.scene_id}: marker leakage`);
        if (role === "user") {
          const normalized = normalizeUser(rawTurn.content);
          if (users.has(normalized)) throw new Error(`${rawScene.scene_id}: duplicate user turn`);
          users.add(normalized);
        }
      });
    });
  });
}

function prompt(template: string, plan: Plan): string {
  return `${template}\n\n## Batch specification\n\nBatch ID: ${plan.batchId}\nCapability focus: ${plan.focus.slug}\nFocus brief: ${plan.focus.brief}\n\nReturn exactly four families in this order:\n${plan.familyIds.join(", ")}\n\nFor each family, use the exact scene IDs formed as FAMILY_ID-KIND. The four families must instantiate materially different operations or situations within the focus. Do not make them neighboring paraphrases of each other.\n`;
}

async function runCodex(
  plan: Plan,
  text: string,
  outputPath: string,
  eventPath: string,
  schemaPath: string,
  repo: string,
  model: string,
  effort: string,
): Promise<void> {
  const temporaryOutput = `${outputPath}.tmp-${process.pid}`;
  const temporaryEvents = `${eventPath}.tmp-${process.pid}`;
  await Promise.all([rm(temporaryOutput, { force: true }), rm(temporaryEvents, { force: true })]);
  await new Promise<void>((accept, reject) => {
    const child = spawn("nice", [
      "-n", "10", "codex", "exec", "-m", model,
      "-c", `model_reasoning_effort=\"${effort}\"`,
      "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
      "-C", repo, "--output-schema", schemaPath, "-o", temporaryOutput,
      "--json", "-",
    ], { stdio: ["pipe", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.on("error", reject);
    child.on("close", async (code) => {
      await writeFile(temporaryEvents, Buffer.concat(chunks));
      await rename(temporaryEvents, eventPath);
      code === 0 ? accept() : reject(new Error(`${plan.batchId}: codex exited ${String(code)}`));
    });
    child.stdin.end(text);
  });
  const parsed: unknown = JSON.parse(await readFile(temporaryOutput, "utf8"));
  validateBatch(parsed, plan);
  await rename(temporaryOutput, outputPath);
}

async function directoryKiB(path: string): Promise<number> {
  try {
    return Number(execFileSync("du", ["-sk", path], { encoding: "utf8" }).trim().split(/\s+/)[0]);
  } catch {
    return 0;
  }
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repo = resolve(args.get("repo") ?? process.cwd());
  const root = resolve(args.get("out-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/generation");
  const logRoot = resolve(args.get("log-dir") ?? "/opencode/logs/alpha-chat-foundations-v12/generation");
  const model = args.get("model") ?? "gpt-5.4";
  const effort = args.get("reasoning-effort") ?? "medium";
  const workers = positiveInteger(args.get("workers"), 3, "workers");
  const batchesPerFocus = positiveInteger(args.get("batches-per-focus"), 6, "batches-per-focus");
  const maximumBatches = positiveInteger(args.get("batches"), FOCUSES.length * batchesPerFocus, "batches");
  const maxAttempts = positiveInteger(args.get("max-attempts"), 3, "max-attempts");
  const storageLimitGiB = positiveInteger(args.get("storage-limit-gib"), 15, "storage-limit-gib");
  if (!root.startsWith("/mnt/donto-data/donto-resources/research/"))
    throw new Error("V12 research output must live on the mounted research drive");
  if (model !== "gpt-5.4") throw new Error("V12 generation is bound to gpt-5.4");
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim();
  const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: repo, encoding: "utf8" }).trim();
  if (dirty) throw new Error("generation requires a clean committed source tree");
  const templatePath = resolve(repo, "prompts/chat-foundations-v12-family-generator.md");
  const schemaPath = resolve(repo, "schemas/chat-foundations-v12-families.schema.json");
  const [template, schemaBytes] = await Promise.all([readFile(templatePath, "utf8"), readFile(schemaPath)]);
  const selectedPlans = plans(batchesPerFocus, maximumBatches);
  const batchRoot = join(root, "batches");
  const rejectedRoot = join(root, "rejected-attempts");
  await Promise.all([mkdir(batchRoot, { recursive: true }), mkdir(rejectedRoot, { recursive: true }), mkdir(logRoot, { recursive: true })]);
  let cursor = 0;
  let completed = 0;
  const failures: Array<Record<string, unknown>> = [];
  async function worker(workerId: number): Promise<void> {
    while (true) {
      const index = cursor++;
      if (index >= selectedPlans.length) return;
      const plan = selectedPlans[index]!;
      const outputPath = join(batchRoot, `${plan.batchId}.json`);
      const existing = await stat(outputPath).catch(() => null);
      if (existing?.isFile()) {
        validateBatch(JSON.parse(await readFile(outputPath, "utf8")), plan);
        completed += 1;
        process.stdout.write(`[generator ${workerId}] ${plan.batchId}: existing valid batch\n`);
        continue;
      }
      let lastError = "";
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        if ((await directoryKiB(resolve(root, ".."))) > storageLimitGiB * 1024 * 1024)
          throw new Error(`V12 project artifacts exceeded ${storageLimitGiB} GiB; pausing`);
        const eventPath = join(logRoot, `${plan.batchId}.attempt-${attempt}.events.jsonl`);
        process.stdout.write(`[generator ${workerId}] ${plan.batchId}: gpt-5.4 attempt ${attempt}\n`);
        try {
          await runCodex(plan, prompt(template, plan) + (lastError ? `\n\nPrevious output was rejected: ${lastError}\nRegenerate the complete batch.` : ""), outputPath, eventPath, schemaPath, repo, model, effort);
          completed += 1;
          lastError = "";
          break;
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          const temporaryOutput = `${outputPath}.tmp-${process.pid}`;
          const invalid = join(rejectedRoot, `${plan.batchId}.attempt-${attempt}.invalid.json`);
          if ((await stat(temporaryOutput).catch(() => null))?.isFile()) await rename(temporaryOutput, invalid);
          await atomicJson(join(rejectedRoot, `${plan.batchId}.attempt-${attempt}.failure.json`), {
            schema: "alpha-chat-foundations-v12-generation-failure-v1",
            batch_id: plan.batchId,
            attempt,
            error: lastError,
            rejected_utc: new Date().toISOString(),
          });
        }
      }
      if (lastError) failures.push({ batch_id: plan.batchId, error: lastError });
    }
  }
  await Promise.all(Array.from({ length: workers }, (_, index) => worker(index + 1)));
  const files = (await readdir(batchRoot)).filter((name) => name.endsWith(".json")).sort();
  const evidence = await Promise.all(files.map(async (name) => {
    const bytes = await readFile(join(batchRoot, name));
    return { name, bytes: bytes.length, sha256: sha256(bytes) };
  }));
  const manifest = {
    schema: "alpha-chat-foundations-v12-generation-manifest-v1",
    status: failures.length === 0 && completed === selectedPlans.length ? "complete" : "incomplete",
    source_commit: sourceCommit,
    source_tree_dirty: false,
    teacher: { model, reasoning_effort: effort },
    prompt: { path: templatePath, sha256: sha256(template) },
    output_schema: { path: schemaPath, sha256: sha256(schemaBytes) },
    plan: { focuses: FOCUSES, batches_per_focus: batchesPerFocus, batches: selectedPlans.length, families_per_batch: 4, planned_families: selectedPlans.length * 4 },
    outputs: evidence,
    failures,
    created_utc: new Date().toISOString(),
  };
  await atomicJson(join(root, "generation-manifest.json"), manifest);
  if (manifest.status !== "complete") throw new Error(`generation incomplete: ${failures.length} failed batches`);
  process.stdout.write(`${JSON.stringify({ result: "PASS", batches: files.length, families: files.length * 4 })}\n`);
}

void main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
