/**
 * CLI dispatcher + arg parsing for historic-chat-gen.
 */
import { createDb, closeDb, getRunStats, getLatestRun, getActiveRun, listRuns, rollbackBatch, updateRunStatus, getPendingBatches } from "./db.js";
import { generate, resume } from "./generator.js";
import { exportTrainingData } from "./export.js";
import { doctor } from "./doctor.js";
import { loadFigures } from "./figures.js";
import { loadTopics } from "./topics.js";
import { hasApiKey } from "./openai.js";
import { estimateTotalCost, estimateCostPerConversation, formatCost, budgetSummary } from "./budget.js";
import { TONES } from "./types.js";

interface KV { [key: string]: string }

function parseKV(args: string[]): KV {
  const result: KV = {};
  for (const arg of args) {
    if (arg.startsWith("--")) {
      const eqIdx = arg.indexOf("=");
      if (eqIdx > 0) {
        result[arg.slice(2, eqIdx)] = arg.slice(eqIdx + 1);
      } else {
        result[arg.slice(2)] = "true";
      }
    }
  }
  return result;
}

function intArg(kv: KV, key: string, def: number): number {
  const v = kv[key];
  return v ? parseInt(v, 10) : def;
}

function floatArg(kv: KV, key: string, def: number): number {
  const v = kv[key];
  return v ? parseFloat(v) : def;
}

function strArg(kv: KV, key: string, def: string): string {
  return kv[key] ?? def;
}

const USAGE = `
historic-chat-gen — Synthetic historical dialogue generator

Commands:
  plan          Preview run: figure/topic counts, estimated cost
  generate      Start new generation run
  resume        Resume a paused/interrupted run
  export        Export committed conversations to training JSONL
  stats         Show progress, cost, distribution stats
  rollback      Revert a batch or entire run
  doctor        Diagnose and repair inconsistent state

Options (generate):
  --count=N       Number of conversations to generate (default: 100)
  --budget=N      Budget limit in dollars (default: 20.00)
  --concurrency=N Concurrent API calls (default: 5)
  --batch=N       Batch size (default: 25)
  --seed=N        RNG seed for reproducibility

Options (export):
  --format=FORMAT chat (default) or jsonl
  --out=PATH      Output path (default: data/historic-chat.txt)
  --run=ID        Specific run ID (default: latest)

Options (doctor):
  --fix           Apply fixes (default: dry run)
`.trim();

export async function chatgenCmd(args: string[]): Promise<void> {
  const command = args[0];

  if (!command || command === "--help" || command === "-h") {
    console.log(USAGE);
    return;
  }

  const subArgs = args.slice(1);
  const kv = parseKV(subArgs);

  // Init DB for all commands
  await createDb();

  try {
    switch (command) {
      case "plan":
        await planCmd(kv);
        break;
      case "generate":
        await generateCmd(kv);
        break;
      case "resume":
        await resumeCmd();
        break;
      case "export":
        await exportCmd(kv);
        break;
      case "stats":
        await statsCmd(kv);
        break;
      case "rollback":
        await rollbackCmd(kv);
        break;
      case "doctor":
        await doctorCmd(kv);
        break;
      default:
        console.error(`Unknown command: ${command}`);
        console.log(USAGE);
        process.exit(1);
    }
  } finally {
    closeDb();
  }
}

async function planCmd(kv: KV): Promise<void> {
  const count = intArg(kv, "count", 100);
  const budget = floatArg(kv, "budget", 20.0);

  const figures = loadFigures();
  const topics = loadTopics();
  const toneCount = TONES.length;
  const uniquePairs = figures.length * (figures.length - 1);
  const uniqueCombos = uniquePairs * topics.length * toneCount;
  const estCost = estimateTotalCost(count);
  const perConv = estimateCostPerConversation();

  console.log("=== Generation Plan ===\n");
  console.log(`Figures:       ${figures.length}`);
  console.log(`Topics:        ${topics.length}`);
  console.log(`Tones:         ${toneCount}`);
  console.log(`Unique combos: ${uniqueCombos.toLocaleString()}`);
  console.log(`Requested:     ${count.toLocaleString()}`);
  console.log();
  console.log(`Est. cost/conv: ${formatCost(perConv)}`);
  console.log(`Est. total:     ${formatCost(estCost)}`);
  console.log(`Budget:         ${formatCost(budget)}`);
  console.log(`Max affordable: ~${Math.floor(budget / perConv).toLocaleString()} conversations`);
  console.log();
  console.log(`API key set:    ${hasApiKey() ? "Yes" : "NO — set OPENAI_API_KEY in .env.local"}`);
}

async function generateCmd(kv: KV): Promise<void> {
  if (!hasApiKey()) {
    console.error("Error: OPENAI_API_KEY not set. Add it to .env.local");
    process.exit(1);
  }

  const count = intArg(kv, "count", 100);
  const budget = floatArg(kv, "budget", 20.0);
  const concurrency = intArg(kv, "concurrency", 5);
  const batchSize = intArg(kv, "batch", 25);
  const seed = kv["seed"] ? parseInt(kv["seed"]!, 10) : undefined;

  await generate({ count, budget, concurrency, batchSize, seed });
}

async function resumeCmd(): Promise<void> {
  if (!hasApiKey()) {
    console.error("Error: OPENAI_API_KEY not set. Add it to .env.local");
    process.exit(1);
  }

  await resume();
}

async function exportCmd(kv: KV): Promise<void> {
  const format = strArg(kv, "format", "chat") as "jsonl" | "chat";
  const defaultOut = format === "chat" ? "data/historic-chat.txt" : "data/historic-chat.jsonl";
  const out = strArg(kv, "out", defaultOut);
  const runId = kv["run"];
  await exportTrainingData({ out, runId, format });
}

async function statsCmd(kv: KV): Promise<void> {
  let runId = kv["run"];

  if (!runId) {
    const latest = await getLatestRun();
    if (!latest) {
      console.log("No runs found.");
      return;
    }
    runId = latest.id;
  }

  const stats = await getRunStats(runId);
  if (!stats) {
    console.log(`Run ${runId} not found.`);
    return;
  }

  console.log("=== Run Stats ===\n");
  console.log(`Run ID:     ${stats.runId}`);
  console.log(`Status:     ${stats.status}`);
  console.log(`Progress:   ${stats.completed}/${stats.target} (${stats.failed} failed)`);
  console.log(`Budget:     ${budgetSummary(stats.budgetLimit, stats.totalCost)}`);
  console.log(`Avg cost:   ${formatCost(stats.avgCostPerConversation)}/conversation`);
  console.log(`Tokens:     ${stats.inputTokens.toLocaleString()} input, ${stats.outputTokens.toLocaleString()} output`);

  if (Object.keys(stats.figureDistribution).length > 0) {
    console.log("\nFigure distribution (top 10):");
    const sorted = Object.entries(stats.figureDistribution)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    for (const [fig, count] of sorted) {
      console.log(`  ${fig}: ${count}`);
    }
  }

  if (Object.keys(stats.topicDistribution).length > 0) {
    console.log("\nTopic distribution (top 10):");
    const sorted = Object.entries(stats.topicDistribution)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    for (const [topic, count] of sorted) {
      console.log(`  ${topic}: ${count}`);
    }
  }
}

async function rollbackCmd(kv: KV): Promise<void> {
  const batchId = kv["batch"];
  const runId = kv["run"];

  if (batchId) {
    await rollbackBatch(batchId);
    console.log(`Rolled back batch ${batchId}`);
  } else if (runId) {
    // Rollback all batches in a run
    const pending = await getPendingBatches(runId);
    for (const b of pending) {
      await rollbackBatch(b.id);
    }
    await updateRunStatus(runId, "failed");
    console.log(`Rolled back run ${runId}`);
  } else {
    console.error("Specify --batch=ID or --run=ID");
    process.exit(1);
  }
}

async function doctorCmd(kv: KV): Promise<void> {
  const fix = kv["fix"] === "true";

  console.log(`Running doctor${fix ? " (fix mode)" : " (dry run)"}...\n`);

  const result = await doctor(fix);

  if (result.orphanedTempFiles.length > 0) {
    console.log(`Orphaned temp files: ${result.orphanedTempFiles.length}`);
    for (const f of result.orphanedTempFiles) console.log(`  ${f}`);
  }

  if (result.pendingBatches.length > 0) {
    console.log(`Pending batches: ${result.pendingBatches.length}`);
    for (const b of result.pendingBatches) console.log(`  ${b}`);
  }

  if (result.staleRuns.length > 0) {
    console.log(`Stale runs: ${result.staleRuns.length}`);
    for (const r of result.staleRuns) console.log(`  ${r}`);
  }

  if (result.actions.length > 0) {
    console.log("\nActions taken:");
    for (const a of result.actions) console.log(`  ${a}`);
  }

  if (result.orphanedTempFiles.length === 0 && result.pendingBatches.length === 0 && result.staleRuns.length === 0) {
    console.log("Everything looks clean.");
  } else if (!fix) {
    console.log("\nRun with --fix to apply repairs.");
  }
}
