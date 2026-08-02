#!/usr/bin/env npx tsx

/** Resumable independent GPT-5.5 review of Alpha V12 linked families. */

import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

interface Family {
  readonly family_id: string;
  readonly [key: string]: unknown;
}

interface GenerationBatch {
  readonly batch_id: string;
  readonly families: readonly Family[];
}

interface Review {
  readonly family_id: string;
  readonly decision: "accept" | "reject";
  readonly semantic_correctness: number;
  readonly response_contingency: number;
  readonly relational_coherence: number;
  readonly naturalness: number;
  readonly shortcut_resistance: number;
  readonly scene_concerns: readonly string[];
  readonly fatal_concerns: readonly string[];
  readonly rationale: string;
}

interface ReviewOutput {
  readonly review_batch_id: string;
  readonly reviews: readonly Review[];
}

interface Plan {
  readonly reviewBatchId: string;
  readonly sourceFiles: readonly string[];
  readonly families: readonly Family[];
}

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

function validateReview(value: unknown, plan: Plan): asserts value is ReviewOutput {
  if (!isObject(value) || value.review_batch_id !== plan.reviewBatchId || !Array.isArray(value.reviews))
    throw new Error(`${plan.reviewBatchId}: malformed review envelope`);
  if (value.reviews.length !== plan.families.length)
    throw new Error(`${plan.reviewBatchId}: expected ${plan.families.length} reviews`);
  value.reviews.forEach((raw, index) => {
    if (!isObject(raw) || raw.family_id !== plan.families[index]!.family_id)
      throw new Error(`${plan.reviewBatchId}: review identity drift at ${index}`);
    if (raw.decision !== "accept" && raw.decision !== "reject")
      throw new Error(`${raw.family_id}: invalid decision`);
    const scores = ["semantic_correctness", "response_contingency", "relational_coherence", "naturalness", "shortcut_resistance"];
    for (const field of scores)
      if (!Number.isInteger(raw[field]) || Number(raw[field]) < 1 || Number(raw[field]) > 5)
        throw new Error(`${raw.family_id}: invalid ${field}`);
    if (!Array.isArray(raw.scene_concerns) || !Array.isArray(raw.fatal_concerns) || typeof raw.rationale !== "string" || !raw.rationale.trim())
      throw new Error(`${raw.family_id}: malformed concerns/rationale`);
    const minimum = Math.min(...scores.map((field) => Number(raw[field])));
    if (raw.decision === "accept" && (minimum < 4 || raw.fatal_concerns.length > 0))
      throw new Error(`${raw.family_id}: acceptance violates review contract`);
    if (raw.decision === "reject" && raw.fatal_concerns.length === 0 && minimum >= 4)
      throw new Error(`${raw.family_id}: rejection lacks a score or fatal basis`);
  });
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

async function runCodex(
  plan: Plan,
  prompt: string,
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
      code === 0 ? accept() : reject(new Error(`${plan.reviewBatchId}: codex exited ${String(code)}`));
    });
    child.stdin.end(prompt);
  });
  const parsed: unknown = JSON.parse(await readFile(temporaryOutput, "utf8"));
  validateReview(parsed, plan);
  await rename(temporaryOutput, outputPath);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repo = resolve(args.get("repo") ?? process.cwd());
  const generationRoot = resolve(args.get("generation-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/generation");
  const root = resolve(args.get("out-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/review");
  const logRoot = resolve(args.get("log-dir") ?? "/opencode/logs/alpha-chat-foundations-v12/review");
  const model = args.get("model") ?? "gpt-5.5";
  const effort = args.get("reasoning-effort") ?? "high";
  const workers = positiveInteger(args.get("workers"), 3, "workers");
  const sourceBatchesPerReview = positiveInteger(args.get("source-batches-per-review"), 2, "source-batches-per-review");
  const maxAttempts = positiveInteger(args.get("max-attempts"), 3, "max-attempts");
  const maximumReviews = positiveInteger(args.get("reviews"), 100_000, "reviews");
  if (model !== "gpt-5.5") throw new Error("V12 independent review is bound to gpt-5.5");
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim();
  if (execFileSync("git", ["status", "--porcelain"], { cwd: repo, encoding: "utf8" }).trim())
    throw new Error("review requires a clean committed source tree");
  const generationManifestPath = join(generationRoot, "generation-manifest.json");
  const generationManifestBytes = await readFile(generationManifestPath);
  const generationManifest = JSON.parse(generationManifestBytes.toString("utf8")) as Record<string, unknown>;
  if (generationManifest.status !== "complete") throw new Error("generation manifest is not complete");
  const batchRoot = join(generationRoot, "batches");
  const sourceFiles = (await readdir(batchRoot)).filter((name) => name.endsWith(".json")).sort();
  const plans: Plan[] = [];
  for (let offset = 0; offset < sourceFiles.length; offset += sourceBatchesPerReview) {
    const group = sourceFiles.slice(offset, offset + sourceBatchesPerReview);
    const batches = await Promise.all(group.map(async (name) => JSON.parse(await readFile(join(batchRoot, name), "utf8")) as GenerationBatch));
    const families = batches.flatMap((batch) => [...batch.families]);
    if (families.length > 8) throw new Error("review schema supports at most eight families");
    plans.push({ reviewBatchId: `v12-review-${String(plans.length).padStart(3, "0")}`, sourceFiles: group, families });
  }
  const selected = plans.slice(0, maximumReviews);
  const templatePath = resolve(repo, "prompts/chat-foundations-v12-family-reviewer.md");
  const schemaPath = resolve(repo, "schemas/chat-foundations-v12-reviews.schema.json");
  const [template, schemaBytes] = await Promise.all([readFile(templatePath, "utf8"), readFile(schemaPath)]);
  const rejectedRoot = join(root, "rejected-attempts");
  await Promise.all([mkdir(root, { recursive: true }), mkdir(rejectedRoot, { recursive: true }), mkdir(logRoot, { recursive: true })]);
  let cursor = 0;
  let completed = 0;
  const failures: Array<Record<string, unknown>> = [];
  async function worker(workerId: number): Promise<void> {
    while (true) {
      const index = cursor++;
      if (index >= selected.length) return;
      const plan = selected[index]!;
      const outputPath = join(root, `${plan.reviewBatchId}.json`);
      if ((await stat(outputPath).catch(() => null))?.isFile()) {
        validateReview(JSON.parse(await readFile(outputPath, "utf8")), plan);
        completed += 1;
        process.stdout.write(`[reviewer ${workerId}] ${plan.reviewBatchId}: existing valid review\n`);
        continue;
      }
      let lastError = "";
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        const eventPath = join(logRoot, `${plan.reviewBatchId}.attempt-${attempt}.events.jsonl`);
        const fullPrompt = `${template}\n\n## Review group\n\nReview batch ID: ${plan.reviewBatchId}\nSource generation batches: ${plan.sourceFiles.join(", ")}\nReturn families in this exact order:\n${plan.families.map((family) => family.family_id).join(", ")}\n\nResearch metadata and model-visible scenes:\n${JSON.stringify(plan.families)}\n${lastError ? `\nPrevious output was rejected: ${lastError}\nReview the complete group again.` : ""}`;
        process.stdout.write(`[reviewer ${workerId}] ${plan.reviewBatchId}: gpt-5.5 attempt ${attempt}\n`);
        try {
          await runCodex(plan, fullPrompt, outputPath, eventPath, schemaPath, repo, model, effort);
          completed += 1;
          lastError = "";
          break;
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          const temporaryOutput = `${outputPath}.tmp-${process.pid}`;
          if ((await stat(temporaryOutput).catch(() => null))?.isFile())
            await rename(temporaryOutput, join(rejectedRoot, `${plan.reviewBatchId}.attempt-${attempt}.invalid.json`));
          await atomicJson(join(rejectedRoot, `${plan.reviewBatchId}.attempt-${attempt}.failure.json`), {
            schema: "alpha-chat-foundations-v12-review-failure-v1",
            review_batch_id: plan.reviewBatchId,
            attempt,
            error: lastError,
            rejected_utc: new Date().toISOString(),
          });
        }
      }
      if (lastError) failures.push({ review_batch_id: plan.reviewBatchId, error: lastError });
    }
  }
  await Promise.all(Array.from({ length: workers }, (_, index) => worker(index + 1)));
  const reviewFiles = (await readdir(root)).filter((name) => /^v12-review-\d+\.json$/.test(name)).sort();
  const outputs = await Promise.all(reviewFiles.map(async (name) => {
    const bytes = await readFile(join(root, name));
    const value = JSON.parse(bytes.toString("utf8")) as ReviewOutput;
    return { name, bytes: bytes.length, sha256: sha256(bytes), accepted: value.reviews.filter((review) => review.decision === "accept").length, rejected: value.reviews.filter((review) => review.decision === "reject").length };
  }));
  const manifest = {
    schema: "alpha-chat-foundations-v12-review-manifest-v1",
    status: failures.length === 0 && completed === selected.length ? "complete" : "incomplete",
    source_commit: sourceCommit,
    source_tree_dirty: false,
    reviewer: { model, reasoning_effort: effort, independent_of_teacher: true },
    generation_manifest: { path: generationManifestPath, sha256: sha256(generationManifestBytes) },
    prompt: { path: templatePath, sha256: sha256(template) },
    output_schema: { path: schemaPath, sha256: sha256(schemaBytes) },
    plan: { source_batches_per_review: sourceBatchesPerReview, review_groups: selected.length, families: selected.reduce((sum, plan) => sum + plan.families.length, 0) },
    outputs,
    summary: { accepted: outputs.reduce((sum, item) => sum + item.accepted, 0), rejected: outputs.reduce((sum, item) => sum + item.rejected, 0) },
    failures,
    created_utc: new Date().toISOString(),
  };
  await atomicJson(join(root, "review-manifest.json"), manifest);
  if (manifest.status !== "complete") throw new Error(`review incomplete: ${failures.length} groups failed`);
  process.stdout.write(`${JSON.stringify({ result: "PASS", ...manifest.summary })}\n`);
}

void main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
