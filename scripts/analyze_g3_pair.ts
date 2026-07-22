#!/usr/bin/env npx tsx
/** Verify and compare the two equal-token G3 architecture pilots. */

import { writeFile } from "node:fs/promises";
import { mean, summarizePilot } from "./pilot_analysis.js";

function args(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = process.argv[++i];
    if (!value || value.startsWith("--")) throw new Error(`missing value for --${key}`);
    result[key] = value;
  }
  return result;
}

async function main(): Promise<void> {
  const cli = args();
  if (!cli.llama || !cli.gpt2 || !cli.out) throw new Error("required: --llama, --gpt2, and --out");
  const [llama, gpt2] = await Promise.all([summarizePilot(cli.llama, "llama"), summarizePilot(cli.gpt2, "gpt2")]);
  for (const key of ["source_commit", "learning_rate", "learning_rate_min"] as const) {
    if (llama.contract[key] !== gpt2.contract[key]) throw new Error(`contract ${key} differs between pilots`);
  }
  if (llama.contract.data.sha256 !== gpt2.contract.data.sha256) throw new Error("pilot data hashes differ");
  if (llama.contract.tokenizer.sha256 !== gpt2.contract.tokenizer.sha256) throw new Error("pilot tokenizer hashes differ");
  if (llama.tokens !== gpt2.tokens) throw new Error("pilot token counts differ");
  const paramDifferenceFraction = Math.abs(llama.total_params - gpt2.total_params) / gpt2.total_params;
  if (paramDifferenceFraction > 0.01) throw new Error(`parameter difference ${(paramDifferenceFraction * 100).toFixed(3)}% exceeds 1%`);

  const gpt2Eval = new Map(gpt2.eval.map((point) => [point.step, point.val_loss]));
  const aligned = llama.eval.map((point) => ({
    step: point.step,
    llama: point.val_loss,
    gpt2: gpt2Eval.get(point.step),
  }));
  if (aligned.some((point) => point.gpt2 === undefined) || aligned.length !== gpt2.eval.length) {
    throw new Error("validation steps do not align");
  }
  const final = aligned.at(-1)! as { step: number; llama: number; gpt2: number };
  const lastThree = aligned.slice(-3) as { step: number; llama: number; gpt2: number }[];
  const llamaLastThree = mean(lastThree.map((point) => point.llama));
  const gpt2LastThree = mean(lastThree.map((point) => point.gpt2));
  const pass = final.llama <= final.gpt2 && llamaLastThree <= gpt2LastThree &&
    llama.allocator_overflow_max === 0 && gpt2.allocator_overflow_max === 0;
  const report = {
    schema: "alpha-g3-pair-analysis-v1",
    result: pass ? "PASS" : "FAIL",
    gate: "Llama final and last-three mean validation loss must be <= equal-token/equal-parameter GPT-2 control; zero allocator overflow",
    contracts_match: true,
    parameter_difference_fraction: paramDifferenceFraction,
    llama,
    gpt2,
    comparison: {
      aligned_validation: aligned,
      final_validation_delta_llama_minus_gpt2: final.llama - final.gpt2,
      last_three_mean: { llama: llamaLastThree, gpt2: gpt2LastThree, delta: llamaLastThree - gpt2LastThree },
    },
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (!pass) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
