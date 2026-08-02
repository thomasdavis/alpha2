#!/usr/bin/env npx tsx

/** Batch-review structured v4 candidates with a stronger Codex reviewer. */

import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { constants } from "node:fs";
import {
  copyFile,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { join, resolve } from "node:path";

interface Review {
  readonly candidate_id: string;
  readonly decision: "accept" | "reject";
  readonly semantic_correctness: number;
  readonly response_contingency: number;
  readonly naturalness: number;
  readonly compactness: number;
  readonly concern: string | null;
}

interface ReviewOutput {
  readonly review_batch_id: string;
  readonly reviews: readonly Review[];
}

interface ReviewPlan {
  readonly reviewBatchId: string;
  readonly sourcePaths: readonly string[];
  readonly candidateIds: readonly string[];
  readonly sourcePayloads: readonly unknown[];
  readonly blueprintPayloads: readonly unknown[];
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const result = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`invalid argument near ${String(key)}`);
    }
    result.set(key.slice(2), value);
    index += 1;
  }
  return result;
}

function positiveInteger(
  value: string | undefined,
  fallback: number,
  name: string,
): number {
  const parsed = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${name} must be a positive integer`);
  return parsed;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function candidateIds(value: unknown, path: string): readonly string[] {
  if (!isObject(value) || !Array.isArray(value.items))
    throw new Error(`${path}: malformed generation batch`);
  return value.items.map((item, index) => {
    if (!isObject(item) || typeof item.candidate_id !== "string") {
      throw new Error(`${path}: item ${index} has no candidate_id`);
    }
    return item.candidate_id;
  });
}

function validateReview(
  value: unknown,
  plan: ReviewPlan,
): asserts value is ReviewOutput {
  if (
    !isObject(value) ||
    value.review_batch_id !== plan.reviewBatchId ||
    !Array.isArray(value.reviews)
  ) {
    throw new Error(`${plan.reviewBatchId}: malformed review envelope`);
  }
  if (value.reviews.length !== plan.candidateIds.length) {
    throw new Error(
      `${plan.reviewBatchId}: expected ${plan.candidateIds.length} reviews, got ${value.reviews.length}`,
    );
  }
  value.reviews.forEach((rawReview, index) => {
    if (
      !isObject(rawReview) ||
      rawReview.candidate_id !== plan.candidateIds[index]
    ) {
      throw new Error(
        `${plan.reviewBatchId}: review ${index} candidate order mismatch`,
      );
    }
    if (rawReview.decision !== "accept" && rawReview.decision !== "reject") {
      throw new Error(
        `${plan.reviewBatchId}: review ${index} has invalid decision`,
      );
    }
    for (const field of [
      "semantic_correctness",
      "response_contingency",
      "naturalness",
      "compactness",
    ] as const) {
      const score = rawReview[field];
      if (!Number.isInteger(score) || Number(score) < 1 || Number(score) > 5) {
        throw new Error(
          `${plan.reviewBatchId}: review ${index} invalid ${field}`,
        );
      }
    }
    if (
      rawReview.decision === "accept" &&
      (Number(rawReview.semantic_correctness) < 4 ||
        Number(rawReview.response_contingency) < 4)
    ) {
      throw new Error(
        `${plan.reviewBatchId}: review ${index} accepts below required threshold`,
      );
    }
    if (
      rawReview.decision === "reject" &&
      typeof rawReview.concern !== "string"
    ) {
      throw new Error(
        `${plan.reviewBatchId}: rejected review ${index} lacks concern`,
      );
    }
  });
}

function promptFor(template: string, plan: ReviewPlan): string {
  return `${template}\n\n## Review batch\n\nReview batch ID: ${plan.reviewBatchId}\nReturn reviews in this exact candidate order:\n${plan.candidateIds.join(", ")}\n\nResearcher-side semantic blueprints:\n${JSON.stringify(plan.blueprintPayloads)}\n\nCandidate batches:\n${JSON.stringify(plan.sourcePayloads)}\n`;
}

async function runCodex(
  plan: ReviewPlan,
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
    const chunks: Buffer[] = [];
    const errors: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => errors.push(chunk));
    child.on("error", rejectPromise);
    child.on("close", async (code) => {
      await writeFile(temporaryEvents, Buffer.concat([...chunks, ...errors]));
      await rename(temporaryEvents, eventPath);
      if (code !== 0) {
        rejectPromise(
          new Error(`${plan.reviewBatchId}: codex exited ${String(code)}`),
        );
        return;
      }
      resolvePromise();
    });
    child.stdin.end(prompt);
  });
  const value: unknown = JSON.parse(await readFile(temporaryOutput, "utf8"));
  validateReview(value, plan);
  await rename(temporaryOutput, outputPath);
}

async function movePreserving(
  source: string,
  destination: string,
): Promise<void> {
  try {
    await rename(source, destination);
    return;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EXDEV") throw error;
  }
  await copyFile(source, destination, constants.COPYFILE_EXCL);
  const [sourceContent, destinationContent] = await Promise.all([
    readFile(source),
    readFile(destination),
  ]);
  if (sha256(sourceContent) !== sha256(destinationContent)) {
    throw new Error(
      `cross-filesystem preservation hash mismatch: ${source} -> ${destination}`,
    );
  }
  await rm(source);
}

async function preserveRejectedAttempt(
  plan: ReviewPlan,
  attempt: number,
  outputPath: string,
  eventPath: string,
  rejectedRoot: string,
  error: unknown,
): Promise<Record<string, unknown>> {
  await mkdir(rejectedRoot, { recursive: true });
  const stem = `${plan.reviewBatchId}.pid-${process.pid}.attempt-${String(attempt).padStart(2, "0")}`;
  const temporaryOutput = `${outputPath}.tmp-${process.pid}`;
  const rejectedOutput = join(rejectedRoot, `${stem}.invalid.json`);
  const rejectedEvents = join(rejectedRoot, `${stem}.events.jsonl`);
  const rejectedReport = join(rejectedRoot, `${stem}.failure.json`);
  const output = await stat(temporaryOutput).catch(() => null);
  const events = await stat(eventPath).catch(() => null);
  if (output?.isFile()) await movePreserving(temporaryOutput, rejectedOutput);
  if (events?.isFile()) await movePreserving(eventPath, rejectedEvents);
  const report = {
    schema: "alpha-chat-review-rejected-attempt-v1",
    review_batch_id: plan.reviewBatchId,
    attempt,
    error: error instanceof Error ? error.message : String(error),
    output: output?.isFile()
      ? {
          path: rejectedOutput,
          bytes: output.size,
          sha256: sha256(await readFile(rejectedOutput)),
        }
      : null,
    events: events?.isFile()
      ? {
          path: rejectedEvents,
          bytes: events.size,
          sha256: sha256(await readFile(rejectedEvents)),
        }
      : null,
    rejected_utc: new Date().toISOString(),
  };
  await writeFile(rejectedReport, `${JSON.stringify(report, null, 2)}\n`, {
    flag: "wx",
  });
  return report;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = resolve(args.get("repo") ?? process.cwd());
  const generationRoot = resolve(
    args.get("generation-dir") ??
      "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/generation/batches",
  );
  const reviewRoot = resolve(
    args.get("out-dir") ??
      "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/review",
  );
  const logRoot = resolve(
    args.get("log-dir") ??
      "/opencode/logs/alpha-chat-semantic-repair-v4/review",
  );
  const batchesPerReview = positiveInteger(
    args.get("batches-per-review"),
    2,
    "batches-per-review",
  );
  const workers = positiveInteger(args.get("workers"), 2, "workers");
  const maxAttempts = positiveInteger(
    args.get("max-attempts"),
    3,
    "max-attempts",
  );
  const model = args.get("model") ?? "gpt-5.5";
  const reasoningEffort = args.get("reasoning-effort") ?? "medium";
  const declaredSourceHistory = (args.get("source-commit-history") ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (declaredSourceHistory.some((value) => !/^[0-9a-f]{40}$/.test(value))) {
    throw new Error("source-commit-history must contain full Git SHAs");
  }
  const blueprintPath = resolve(
    args.get("blueprint") ??
      "/mnt/donto-data/donto-resources/research/alpha-chat-semantic-repair-v4-20260801/planned-v2/blueprint.json",
  );
  const maximumReviews = positiveInteger(
    args.get("reviews"),
    Number.MAX_SAFE_INTEGER,
    "reviews",
  );
  const templatePath = resolve(
    args.get("prompt") ??
      resolve(repoRoot, "prompts/chat-semantic-repair-reviewer.md"),
  );
  const schemaPath = resolve(
    args.get("schema") ??
      resolve(repoRoot, "schemas/chat-semantic-repair-reviews.schema.json"),
  );
  const template = await readFile(templatePath, "utf8");
  const blueprintBytes = await readFile(blueprintPath);
  const blueprint = JSON.parse(blueprintBytes.toString("utf8")) as unknown;
  if (!isObject(blueprint) || !Array.isArray(blueprint.batches))
    throw new Error("malformed semantic blueprint");
  const blueprintById = new Map<string, unknown>();
  for (const raw of blueprint.batches) {
    if (!isObject(raw) || typeof raw.batch_id !== "string")
      throw new Error("semantic blueprint has malformed batch");
    if (blueprintById.has(raw.batch_id))
      throw new Error(`semantic blueprint duplicates ${raw.batch_id}`);
    blueprintById.set(raw.batch_id, raw);
  }
  const names = (await readdir(generationRoot))
    .filter((name) => name.endsWith(".json") && !name.includes(".tmp-"))
    .sort();
  if (names.length === 0)
    throw new Error(`no generation batches in ${generationRoot}`);
  const plans: ReviewPlan[] = [];
  for (let offset = 0; offset < names.length; offset += batchesPerReview) {
    const selected = names.slice(offset, offset + batchesPerReview);
    const sourcePaths = selected.map((name) => join(generationRoot, name));
    const sourcePayloads = await Promise.all(
      sourcePaths.map(
        async (path) => JSON.parse(await readFile(path, "utf8")) as unknown,
      ),
    );
    const ids = sourcePayloads.flatMap((value, index) =>
      candidateIds(value, sourcePaths[index]!),
    );
    const blueprintPayloads = sourcePayloads.map((value, index) => {
      if (!isObject(value) || typeof value.batch_id !== "string")
        throw new Error(`${sourcePaths[index]} has no batch_id`);
      const allocated = blueprintById.get(value.batch_id);
      if (!allocated)
        throw new Error(
          `semantic blueprint does not contain ${value.batch_id}`,
        );
      return allocated;
    });
    if (ids.length > 128)
      throw new Error(`review group exceeds schema maximum: ${ids.length}`);
    plans.push({
      reviewBatchId: `review-${String(plans.length).padStart(3, "0")}`,
      sourcePaths,
      sourcePayloads,
      blueprintPayloads,
      candidateIds: ids,
    });
  }
  const selectedPlans = plans.slice(0, maximumReviews);
  const requestedReviewGroups = selectedPlans.length;
  const queue = [...selectedPlans];
  await mkdir(reviewRoot, { recursive: true });
  const rejectedRoot = join(reviewRoot, "rejected-attempts");
  await mkdir(rejectedRoot, { recursive: true });
  await mkdir(logRoot, { recursive: true });
  const priorCampaignFailurePath = join(reviewRoot, "review-failures.json");
  const priorCampaignFailure = await stat(priorCampaignFailurePath).catch(
    () => null,
  );
  if (priorCampaignFailure?.isFile()) {
    const content = await readFile(priorCampaignFailurePath);
    await movePreserving(
      priorCampaignFailurePath,
      join(
        rejectedRoot,
        `campaign-failures-${sha256(content).slice(0, 16)}-pid-${process.pid}.json`,
      ),
    );
  }
  const completed: Array<Record<string, unknown>> = [];
  const rejectedAttempts: Array<Record<string, unknown>> = await Promise.all(
    (await readdir(rejectedRoot))
      .filter((name) => name.endsWith(".failure.json"))
      .sort()
      .map(async (name) =>
        JSON.parse(await readFile(join(rejectedRoot, name), "utf8")),
      ),
  );
  const failed: Array<Record<string, unknown>> = [];

  async function worker(index: number): Promise<void> {
    while (queue.length > 0) {
      const plan = queue.shift();
      if (!plan) return;
      const outputPath = join(reviewRoot, `${plan.reviewBatchId}.json`);
      const eventPath = join(logRoot, `${plan.reviewBatchId}.events.jsonl`);
      try {
        const existing: unknown = JSON.parse(
          await readFile(outputPath, "utf8"),
        );
        validateReview(existing, plan);
        const content = await readFile(outputPath);
        const events = await readFile(eventPath);
        completed.push({
          review_batch_id: plan.reviewBatchId,
          status: "existing",
          path: outputPath,
          bytes: content.byteLength,
          sha256: sha256(content),
          events_path: eventPath,
          events_bytes: events.byteLength,
          events_sha256: sha256(events),
        });
        process.stdout.write(
          `[reviewer ${index}] ${plan.reviewBatchId}: existing valid review\n`,
        );
        continue;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
          process.stdout.write(
            `[reviewer ${index}] ${plan.reviewBatchId}: replacing invalid review\n`,
          );
        }
      }
      const prompt = promptFor(template, plan);
      let acceptedAttempt = 0;
      let lastError: unknown = [...rejectedAttempts]
        .reverse()
        .find(
          (attempt) => attempt.review_batch_id === plan.reviewBatchId,
        )?.error;
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        process.stdout.write(
          `[reviewer ${index}] ${plan.reviewBatchId}: reviewing with ${model} (attempt ${attempt}/${maxAttempts})\n`,
        );
        try {
          const validatorFeedback =
            typeof lastError === "string" && lastError.length > 0
              ? `\n\n## Validator feedback from the previous rejected attempt\n\nThe previous review output was rejected for this exact reason: ${lastError}\nReview the complete group again from scratch and correct that structural or scoring-contract failure. Return every candidate in the required order.`
              : "";
          await runCodex(
            plan,
            `${prompt}${validatorFeedback}`,
            outputPath,
            eventPath,
            schemaPath,
            repoRoot,
            model,
            reasoningEffort,
          );
          acceptedAttempt = attempt;
          break;
        } catch (error) {
          lastError = error;
          rejectedAttempts.push(
            await preserveRejectedAttempt(
              plan,
              attempt,
              outputPath,
              eventPath,
              rejectedRoot,
              error,
            ),
          );
          process.stderr.write(
            `[reviewer ${index}] ${plan.reviewBatchId}: rejected attempt ${attempt}/${maxAttempts}: ${error instanceof Error ? error.message : String(error)}\n`,
          );
        }
      }
      if (acceptedAttempt === 0) {
        failed.push({
          review_batch_id: plan.reviewBatchId,
          attempts: maxAttempts,
          error:
            lastError instanceof Error ? lastError.message : String(lastError),
        });
        process.stderr.write(
          `[reviewer ${index}] ${plan.reviewBatchId}: exhausted ${maxAttempts} attempts; continuing remaining queue\n`,
        );
        continue;
      }
      const content = await readFile(outputPath);
      const events = await readFile(eventPath);
      const parsed = JSON.parse(content.toString("utf8")) as ReviewOutput;
      completed.push({
        review_batch_id: plan.reviewBatchId,
        status: "generated",
        path: outputPath,
        bytes: content.byteLength,
        sha256: sha256(content),
        events_path: eventPath,
        events_bytes: events.byteLength,
        events_sha256: sha256(events),
        accepted: parsed.reviews.filter(
          (review) => review.decision === "accept",
        ).length,
        rejected: parsed.reviews.filter(
          (review) => review.decision === "reject",
        ).length,
        accepted_attempt: acceptedAttempt,
      });
      process.stdout.write(
        `[reviewer ${index}] ${plan.reviewBatchId}: accepted structured review\n`,
      );
    }
  }

  await Promise.all(
    Array.from({ length: workers }, (_, index) => worker(index + 1)),
  );
  if (failed.length > 0) {
    const failurePath = join(reviewRoot, "review-failures.json");
    await writeFile(
      failurePath,
      `${JSON.stringify(
        {
          schema: "alpha-chat-review-failures-v1",
          failed,
          rejected_attempts: rejectedAttempts,
        },
        null,
        2,
      )}\n`,
    );
    throw new Error(
      `${failed.length} review groups exhausted ${maxAttempts} attempts; see ${failurePath}`,
    );
  }
  completed.sort((a, b) =>
    String(a.review_batch_id).localeCompare(String(b.review_batch_id)),
  );
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  const sourceCommitHistory = [
    ...new Set([...declaredSourceHistory, sourceCommit]),
  ];
  const manifest = {
    schema:
      args.get("manifest-schema") ??
      "alpha-chat-semantic-repair-v4-review-manifest-v1",
    reviewed_utc: new Date().toISOString(),
    model,
    reasoning_effort: reasoningEffort,
    codex_version: execFileSync("codex", ["--version"], {
      encoding: "utf8",
    }).trim(),
    source_commit: sourceCommit,
    source_commit_history: sourceCommitHistory,
    source_tree_dirty:
      execFileSync("git", ["status", "--porcelain"], {
        cwd: repoRoot,
        encoding: "utf8",
      }).trim().length > 0,
    prompt: { path: templatePath, sha256: sha256(template) },
    output_schema: {
      path: schemaPath,
      sha256: sha256(await readFile(schemaPath)),
    },
    blueprint: { path: blueprintPath, sha256: sha256(blueprintBytes) },
    generation_batches: names.length,
    batches_per_review: batchesPerReview,
    requested_review_groups: requestedReviewGroups,
    max_attempts: maxAttempts,
    rejected_attempts: rejectedAttempts,
    completed,
  };
  const manifestPath = join(reviewRoot, "review-manifest.json");
  const temporary = `${manifestPath}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`);
  await rename(temporary, manifestPath);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", manifest: manifestPath, reviews: completed.length })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
