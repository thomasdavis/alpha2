#!/usr/bin/env npx tsx
/** Validate the contracted three-way Llama LR sweep and select by held-out loss. */

import { writeFile } from "node:fs/promises";
import { mean, summarizePilot, type PilotRunSummary } from "./pilot_analysis.js";

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

function lastThreeMean(run: PilotRunSummary): number {
  return mean(run.eval.slice(-3).map((point) => point.val_loss));
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.lr1e3 || !cli.lr2e3 || !cli.lr3e3 || !cli.out) {
    throw new Error("required: --lr1e3, --lr2e3, --lr3e3, and --out");
  }
  const expected = [
    { learningRate: 1e-3, dir: cli.lr1e3 },
    { learningRate: 2e-3, dir: cli.lr2e3 },
    { learningRate: 3e-3, dir: cli.lr3e3 },
  ];
  const runs = await Promise.all(expected.map(async ({ learningRate, dir }) => {
    const run = await summarizePilot(dir, "llama");
    if (run.contract.learning_rate !== learningRate) {
      throw new Error(`${dir}: learning rate ${run.contract.learning_rate} != ${learningRate}`);
    }
    if (run.contract.learning_rate_min !== learningRate / 10) {
      throw new Error(`${dir}: minimum learning rate ${run.contract.learning_rate_min} != ${learningRate / 10}`);
    }
    return run;
  }));

  const reference = runs[0];
  for (const run of runs.slice(1)) {
    if (run.contract.source_commit !== reference.contract.source_commit) throw new Error("LR sweep source commits differ");
    if (run.contract.data.sha256 !== reference.contract.data.sha256) throw new Error("LR sweep data hashes differ");
    if (run.contract.tokenizer.sha256 !== reference.contract.tokenizer.sha256) throw new Error("LR sweep tokenizer hashes differ");
    if (run.total_params !== reference.total_params || run.tokens !== reference.tokens) {
      throw new Error("LR sweep parameter/token counts differ");
    }
    const referenceSteps = reference.eval.map((point) => point.step).join(",");
    const runSteps = run.eval.map((point) => point.step).join(",");
    if (runSteps !== referenceSteps) throw new Error("LR sweep validation steps do not align");
  }
  if (runs.some((run) => run.allocator_overflow_max !== 0)) throw new Error("LR sweep has allocator overflow");

  const ranking = runs.map((run) => ({
    learning_rate: run.contract.learning_rate,
    run_dir: run.dir,
    last_three_validation_mean: lastThreeMean(run),
    final_validation_loss: run.eval.at(-1)!.val_loss,
    final_train_loss: run.final_train_loss,
    median_tokens_per_sec_after_warmup: run.median_tokens_per_sec_after_warmup,
  })).sort((left, right) =>
    left.last_three_validation_mean - right.last_three_validation_mean ||
    left.final_validation_loss - right.final_validation_loss ||
    left.learning_rate - right.learning_rate);

  const report = {
    schema: "alpha-lr-sweep-analysis-v1",
    result: "PASS",
    selection_rule: "lowest mean held-out loss over the final three aligned evaluations; final held-out loss then lower LR break ties",
    selected_learning_rate: ranking[0].learning_rate,
    selected_run_dir: ranking[0].run_dir,
    source_commit: reference.contract.source_commit,
    data_sha256: reference.contract.data.sha256,
    tokenizer_sha256: reference.contract.tokenizer.sha256,
    ranking,
    runs,
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
