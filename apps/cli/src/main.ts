#!/usr/bin/env node
/**
 * alpha CLI — the main entry point.
 *
 * Commands: tokenizer build, train, sample, eval, bench
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";

// Load .env.local (no dotenv dependency — zero deps philosophy)
try {
  const envContent = readFileSync(".env.local", "utf8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!process.env[key]) process.env[key] = val;
  }
} catch { /* .env.local is optional */ }
import { tokenizerBuildCmd } from "./commands/tokenizer-build.js";
import { trainCmd } from "./commands/train.js";
import { sampleCmd } from "./commands/sample.js";
import { evalCmd } from "./commands/eval.js";
import { benchCmd } from "./commands/bench.js";
import { datagenCmd } from "./commands/datagen.js";
import { fleetCmd } from "./commands/fleet.js";

const USAGE = `
alpha — a tiny, readable GPT training system

Commands:
  tokenizer build  Build tokenizer artifacts from text
  train            Train a GPT model
  sample           Generate text from a checkpoint
  eval             Evaluate a checkpoint on validation data
  bench            Run benchmarks
  datagen          Generate synthetic training data
  fleet            Manage remote training instances

Options:
  --help, -h       Show this help

Examples:
  alpha tokenizer build --type=bpe --input=data/train.txt --vocabSize=2000 --out=artifacts/tokenizer.json
  alpha train --data=data/train.txt --iters=1000 --batch=64 --block=256
  alpha sample --checkpoint=runs/.../checkpoint-1000.json --prompt="ROMEO:" --steps=200
  alpha eval --checkpoint=runs/.../checkpoint-1000.json --data=data/val.txt
  alpha bench --suite=ops --backend=cpu_ref
`.trim();

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(USAGE);
    process.exit(0);
  }

  const command = args[0];

  if (command === "tokenizer" && args[1] === "build") {
    await tokenizerBuildCmd(args.slice(2));
  } else if (command === "train") {
    await trainCmd(args.slice(1));
  } else if (command === "sample") {
    await sampleCmd(args.slice(1));
  } else if (command === "eval") {
    await evalCmd(args.slice(1));
  } else if (command === "bench") {
    await benchCmd(args.slice(1));
  } else if (command === "datagen") {
    await datagenCmd(args.slice(1));
  } else if (command === "fleet") {
    await fleetCmd(args.slice(1));
  } else {
    console.error(`Unknown command: ${args.join(" ")}`);
    console.log(USAGE);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
