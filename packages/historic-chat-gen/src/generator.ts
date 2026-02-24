/**
 * Core generation loop: batching, concurrency, atomic commits.
 */
import { writeFileSync, mkdirSync, renameSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { loadFigures } from "./figures.js";
import { loadTopics } from "./topics.js";
import { SeededRng, genId, now } from "./util.js";
import { buildPrompt } from "./prompt.js";
import { chatCompletion } from "./openai.js";
import { validateConversation } from "./validator.js";
import { calculateCost, canAffordBatch, isBudgetExhausted, formatCost } from "./budget.js";
import {
  createRun, getRun, updateRunStatus,
  createBatch, commitBatch, getCompletedAssignmentKeys,
  getActiveRun,
} from "./db.js";
import type {
  ConversationAssignment, GeneratedConversation,
  GenerateOptions, Run, Tone,
} from "./types.js";
import { TONES } from "./types.js";

/**
 * Generate all possible (figureA, figureB, topic, tone) assignments,
 * shuffle with seeded RNG, slice to count.
 */
export function generateAssignments(
  seed: number,
  count: number,
  excludeKeys?: Set<string>
): ConversationAssignment[] {
  const figures = loadFigures();
  const topics = loadTopics();
  const rng = new SeededRng(seed);

  const assignments: ConversationAssignment[] = [];

  for (const figA of figures) {
    for (const figB of figures) {
      if (figA.id === figB.id) continue;
      for (const topic of topics) {
        for (const tone of TONES) {
          const key = `${figA.id}|${figB.id}|${topic.id}|${tone}`;
          if (excludeKeys?.has(key)) continue;

          const turnCount = 6 + rng.int(0, 6) * 2; // 6, 8, 10, 12, 14, 16
          assignments.push({ figureA: figA, figureB: figB, topic, tone, turnCount });
        }
      }
    }
  }

  rng.shuffle(assignments);
  return assignments.slice(0, count);
}

/**
 * Process a single conversation assignment via OpenAI.
 */
async function processAssignment(
  assignment: ConversationAssignment,
  runId: string,
  batchId: string,
): Promise<GeneratedConversation | null> {
  const { system, user } = buildPrompt(assignment);

  try {
    const result = await chatCompletion({
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });

    let parsed: unknown;
    try {
      parsed = JSON.parse(result.content);
    } catch {
      console.error(`  JSON parse error for ${assignment.figureA.name} vs ${assignment.figureB.name}`);
      return null;
    }

    const validation = validateConversation(
      parsed,
      assignment.figureA.name,
      assignment.figureB.name,
      assignment.turnCount
    );

    if (!validation.valid) {
      console.error(`  Validation error: ${validation.error}`);
      return null;
    }

    const cost = calculateCost(result.inputTokens, result.outputTokens);

    return {
      id: genId(),
      runId,
      batchId,
      figureA: assignment.figureA.id,
      figureB: assignment.figureB.id,
      topic: assignment.topic.id,
      tone: assignment.tone,
      turnCount: validation.turns!.length,
      turns: validation.turns!,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
      cost,
      createdAt: now(),
    };
  } catch (err) {
    console.error(`  API error: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

/**
 * Process a batch of assignments with concurrency limit.
 */
async function processBatch(
  assignments: ConversationAssignment[],
  runId: string,
  batchId: string,
  concurrency: number,
): Promise<{ succeeded: GeneratedConversation[]; failed: number }> {
  const succeeded: GeneratedConversation[] = [];
  let failed = 0;

  // Process with concurrency limiter
  let active = 0;
  let idx = 0;
  const results: Promise<void>[] = [];

  const processNext = async (): Promise<void> => {
    while (idx < assignments.length) {
      const i = idx++;
      const assignment = assignments[i]!;

      const conv = await processAssignment(assignment, runId, batchId);
      if (conv) {
        succeeded.push(conv);
      } else {
        failed++;
      }
    }
  };

  const workers = Math.min(concurrency, assignments.length);
  const workerPromises: Promise<void>[] = [];
  for (let w = 0; w < workers; w++) {
    workerPromises.push(processNext());
  }

  await Promise.all(workerPromises);

  return { succeeded, failed };
}

/**
 * Write batch to temp file, then atomic rename.
 */
function atomicWriteBatch(batchId: string, conversations: GeneratedConversation[]): void {
  const baseDir = resolve(process.cwd(), ".histchat");
  const tmpDir = resolve(baseDir, ".tmp");
  const batchDir = resolve(baseDir, "batches");
  mkdirSync(tmpDir, { recursive: true });
  mkdirSync(batchDir, { recursive: true });

  const tmpPath = resolve(tmpDir, `batch-${batchId}.jsonl.tmp`);
  const finalPath = resolve(batchDir, `batch-${batchId}.jsonl`);

  const lines = conversations.map((c) => JSON.stringify(c)).join("\n") + "\n";
  writeFileSync(tmpPath, lines, "utf-8");
  renameSync(tmpPath, finalPath);
}

/**
 * Start a new generation run.
 */
export async function generate(opts: GenerateOptions): Promise<Run> {
  const seed = opts.seed ?? Date.now();

  const run = await createRun({
    targetCount: opts.count,
    budgetLimit: opts.budget,
    concurrency: opts.concurrency,
    batchSize: opts.batchSize,
    seed,
  });

  console.log(`Run ${run.id} started: ${opts.count} conversations, budget ${formatCost(opts.budget)}`);

  const assignments = generateAssignments(seed, opts.count);
  console.log(`Generated ${assignments.length} assignments`);

  await processAssignments(run, assignments, opts);
  return (await getRun(run.id))!;
}

/**
 * Resume a paused or interrupted run.
 */
export async function resume(): Promise<Run | null> {
  const run = await getActiveRun();
  if (!run) {
    console.log("No active or paused run to resume.");
    return null;
  }

  console.log(`Resuming run ${run.id} (${run.completedCount}/${run.targetCount} done, ${formatCost(run.totalCost)} spent)`);

  if (run.status === "paused") {
    await updateRunStatus(run.id, "active");
  }

  // Regenerate assignments excluding completed ones
  const completedKeys = await getCompletedAssignmentKeys(run.id);
  const remaining = run.targetCount - run.completedCount;

  if (remaining <= 0) {
    console.log("Run already complete.");
    await updateRunStatus(run.id, "completed");
    return run;
  }

  const assignments = generateAssignments(run.seed, run.targetCount, completedKeys);
  console.log(`${assignments.length} assignments remaining`);

  await processAssignments(run, assignments, {
    budget: run.budgetLimit,
    concurrency: run.concurrency,
    batchSize: run.batchSize,
  });

  return (await getRun(run.id))!;
}

async function processAssignments(
  run: Run,
  assignments: ConversationAssignment[],
  opts: { budget: number; concurrency: number; batchSize: number },
): Promise<void> {
  const totalBatches = Math.ceil(assignments.length / opts.batchSize);

  for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
    // Re-read run to get latest cost
    const currentRun = (await getRun(run.id))!;

    // Budget check before batch
    if (isBudgetExhausted(currentRun.budgetLimit, currentRun.totalCost)) {
      console.log(`\nBudget exhausted. Pausing run.`);
      await updateRunStatus(run.id, "paused");
      return;
    }

    const avgCost = currentRun.completedCount > 0
      ? currentRun.totalCost / currentRun.completedCount
      : undefined;

    if (!canAffordBatch(currentRun.budgetLimit, currentRun.totalCost, opts.batchSize, avgCost)) {
      console.log(`\nInsufficient budget for next batch. Pausing run.`);
      await updateRunStatus(run.id, "paused");
      return;
    }

    const start = batchIdx * opts.batchSize;
    const batchAssignments = assignments.slice(start, start + opts.batchSize);

    const batch = await createBatch(run.id, batchIdx);
    console.log(`\nBatch ${batchIdx + 1}/${totalBatches} (${batchAssignments.length} conversations)...`);

    const { succeeded, failed } = await processBatch(
      batchAssignments,
      run.id,
      batch.id,
      opts.concurrency
    );

    if (succeeded.length > 0) {
      atomicWriteBatch(batch.id, succeeded);
      await commitBatch(batch.id, succeeded);
    }

    const runNow = (await getRun(run.id))!;
    const totalCost = formatCost(runNow.totalCost);
    console.log(`  ${succeeded.length} succeeded, ${failed} failed | Total: ${runNow.completedCount}/${runNow.targetCount} | Cost: ${totalCost}`);
  }

  const finalRun = (await getRun(run.id))!;
  if (finalRun.status === "active") {
    await updateRunStatus(run.id, "completed");
    console.log(`\nRun completed: ${finalRun.completedCount} conversations, ${formatCost(finalRun.totalCost)} total cost`);
  }
}
