#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import {
  compileCoopF16x3BalancePlan,
  parseCoopF16x3CalibrationJsonl,
} from "../packages/helios/dist/coop-balance-plan.js";

function usage(message) {
  if (message) console.error(message);
  console.error(
    "usage: node scripts/compile-helios-coop-balance-plan.mjs " +
      "--input calibration.jsonl --output balance-plan.json " +
      "[--graph-fingerprint SHA256] [--safe-fp16-max 32752]",
  );
  process.exit(2);
}

const options = new Map();
for (let index = 2; index < process.argv.length; index++) {
  const argument = process.argv[index];
  if (!argument.startsWith("--")) usage(`unexpected argument ${argument}`);
  const name = argument.slice(2);
  const value = process.argv[++index];
  if (value === undefined || value.startsWith("--")) usage(`missing value for --${name}`);
  if (options.has(name)) usage(`duplicate option --${name}`);
  options.set(name, value);
}

const input = options.get("input");
const output = options.get("output");
if (!input || !output) usage("--input and --output are required");
for (const name of options.keys()) {
  if (!["input", "output", "graph-fingerprint", "safe-fp16-max"].includes(name)) {
    usage(`unknown option --${name}`);
  }
}

const safeFp16MaxRaw = options.get("safe-fp16-max");
const safeFp16Max = safeFp16MaxRaw === undefined ? undefined : Number(safeFp16MaxRaw);
if (safeFp16MaxRaw !== undefined && !Number.isFinite(safeFp16Max)) {
  usage(`invalid --safe-fp16-max ${safeFp16MaxRaw}`);
}

const records = parseCoopF16x3CalibrationJsonl(readFileSync(input, "utf8"));
const plan = compileCoopF16x3BalancePlan(records, {
  graphFingerprint: options.get("graph-fingerprint"),
  safeFp16Max,
});

// Refuse overwrite. Calibration and plans are immutable research artifacts;
// a new attempt gets a new path rather than silently replacing evidence.
writeFileSync(output, `${JSON.stringify(plan, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
console.log(JSON.stringify({
  output,
  checkpointFingerprint: plan.checkpointFingerprint,
  graphFingerprint: plan.graphFingerprint,
  planFingerprint: plan.planFingerprint,
  operations: plan.entries.length,
  samplesPerOperation: Math.min(...plan.entries.map((entry) => entry.samples)),
  exponentRange: [
    Math.min(...plan.entries.map((entry) => entry.exponent)),
    Math.max(...plan.entries.map((entry) => entry.exponent)),
  ],
}));
