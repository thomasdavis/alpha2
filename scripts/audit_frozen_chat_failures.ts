#!/usr/bin/env npx tsx
/**
 * Stratify a frozen chat run without changing or rejudging its recorded outputs.
 *
 * The audit joins the immutable prompt set to one immutable result set and makes
 * context-capacity failures explicit. A prompt is generation-eligible only when
 * at least one token position remains inside the checkpoint block size.
 */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

interface PromptRow {
  id: string;
  source: string;
  prompt_tokens: number;
  messages: Array<{ role: string; content: string }>;
}

interface ResultRow {
  id: string;
  source: string;
  promptTokens: number;
  generatedIds: number[];
  text: string;
  eosTerminated: boolean;
  hitBlockLimit: boolean;
  roleLeak: boolean;
  nonempty: boolean;
  fourGramRepeatRate: number;
  degenerateLoop: boolean;
  structuralPass: boolean;
}

interface Summary {
  checkpoint: {
    sha256: string;
    step: number;
    modelConfig: { blockSize: number };
  };
  generation: { chatMaxTokens: number; eosId: number; userId: number };
  outputs: { chat: { sha256: string; rows: number } };
}

interface JoinedRow extends ResultRow {
  sourcePromptTokens: number;
  userTurns: number;
  assistantTurns: number;
  generationEligible: boolean;
}

interface Aggregate {
  total: number;
  structuralPass: number;
  nonempty: number;
  empty: number;
  eosTerminated: number;
  roleLeaks: number;
  degenerateLoops: number;
  hitBlockLimit: number;
  meanFourGramRepeatRate: number;
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`expected --key value, received ${key} ${value ?? ""}`.trim());
    }
    result[key.slice(2)] = value;
    index += 1;
  }
  return result;
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function parseJsonl<T>(text: string, label: string): T[] {
  return text.split("\n").filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`${label}:${index + 1} is invalid JSON`, { cause: error });
    }
  });
}

function requireSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} is invalid: ${value}`);
}

function aggregate(rows: readonly JoinedRow[]): Aggregate {
  return {
    total: rows.length,
    structuralPass: rows.filter((row) => row.structuralPass).length,
    nonempty: rows.filter((row) => row.nonempty).length,
    empty: rows.filter((row) => !row.nonempty).length,
    eosTerminated: rows.filter((row) => row.eosTerminated).length,
    roleLeaks: rows.filter((row) => row.roleLeak).length,
    degenerateLoops: rows.filter((row) => row.degenerateLoop).length,
    hitBlockLimit: rows.filter((row) => row.hitBlockLimit).length,
    meanFourGramRepeatRate: rows.length === 0
      ? 0
      : rows.reduce((sum, row) => sum + row.fourGramRepeatRate, 0) / rows.length,
  };
}

function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}

function lengthBin(tokens: number): string {
  if (tokens <= 63) return "000-063";
  if (tokens <= 127) return "064-127";
  if (tokens <= 255) return "128-255";
  if (tokens <= 383) return "256-383";
  if (tokens <= 511) return "384-511";
  return "512+";
}

function turnBin(row: JoinedRow): string {
  const turns = row.userTurns + row.assistantTurns;
  if (turns <= 1) return "1";
  if (turns <= 3) return "2-3";
  if (turns <= 5) return "4-5";
  if (turns <= 7) return "6-7";
  return "8+";
}

function groupBy(rows: readonly JoinedRow[], key: (row: JoinedRow) => string): Record<string, Aggregate> {
  const groups = new Map<string, JoinedRow[]>();
  for (const row of rows) {
    const label = key(row);
    const group = groups.get(label) ?? [];
    group.push(row);
    groups.set(label, group);
  }
  return Object.fromEntries(
    [...groups.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([label, group]) => [label, aggregate(group)]),
  );
}

async function main(): Promise<void> {
  const args = parseArgs();
  if (!args.prompts || !args.results || !args.summary || !args.out) {
    throw new Error("required: --prompts, --results, --summary, and --out");
  }
  const [promptText, resultText, summaryText] = await Promise.all([
    readFile(args.prompts, "utf8"),
    readFile(args.results, "utf8"),
    readFile(args.summary, "utf8"),
  ]);
  const prompts = parseJsonl<PromptRow>(promptText, args.prompts);
  const results = parseJsonl<ResultRow>(resultText, args.results);
  const summary = JSON.parse(summaryText) as Summary;
  requireSafeInteger(summary.checkpoint.modelConfig.blockSize, "block size");
  if (summary.outputs.chat.sha256 !== sha256(resultText)) throw new Error("result SHA-256 does not match summary");
  if (summary.outputs.chat.rows !== results.length) throw new Error("result count does not match summary");
  if (prompts.length !== results.length) throw new Error("prompt/result row counts differ");

  const promptsById = new Map(prompts.map((row) => [row.id, row]));
  if (promptsById.size !== prompts.length) throw new Error("duplicate prompt id");
  const joined: JoinedRow[] = results.map((result) => {
    const prompt = promptsById.get(result.id);
    if (!prompt) throw new Error(`result has no prompt: ${result.id}`);
    if (prompt.source !== result.source) throw new Error(`source mismatch for ${result.id}`);
    requireSafeInteger(prompt.prompt_tokens, `${result.id} source prompt tokens`);
    requireSafeInteger(result.promptTokens, `${result.id} runtime prompt tokens`);
    if (Math.abs(prompt.prompt_tokens - result.promptTokens) > 1) {
      throw new Error(`prompt-token mismatch exceeds boundary allowance for ${result.id}`);
    }
    const generationEligible = result.promptTokens < summary.checkpoint.modelConfig.blockSize;
    if (!generationEligible && !result.hitBlockLimit) {
      throw new Error(`over-context row was not marked hitBlockLimit: ${result.id}`);
    }
    return {
      ...result,
      sourcePromptTokens: prompt.prompt_tokens,
      userTurns: prompt.messages.filter((message) => message.role === "user").length,
      assistantTurns: prompt.messages.filter((message) => message.role === "assistant").length,
      generationEligible,
    };
  });

  const eligible = joined.filter((row) => row.generationEligible);
  const ineligible = joined.filter((row) => !row.generationEligible);
  const empty = joined.filter((row) => !row.nonempty);
  const promptLengths = joined.map((row) => row.promptTokens).sort((left, right) => left - right);
  const audit = {
    schema: "alpha-frozen-chat-failure-audit-v1",
    inputs: {
      prompts: { path: args.prompts, sha256: sha256(promptText), rows: prompts.length },
      results: { path: args.results, sha256: sha256(resultText), rows: results.length },
      summary: { path: args.summary, sha256: sha256(summaryText) },
      checkpoint: summary.checkpoint,
      generation: summary.generation,
    },
    promptLengthTokens: {
      min: promptLengths[0] ?? 0,
      p25: percentile(promptLengths, 0.25),
      median: percentile(promptLengths, 0.5),
      p75: percentile(promptLengths, 0.75),
      max: promptLengths.at(-1) ?? 0,
      mean: promptLengths.reduce((sum, value) => sum + value, 0) / Math.max(1, promptLengths.length),
    },
    all: aggregate(joined),
    generationEligible: aggregate(eligible),
    generationIneligible: aggregate(ineligible),
    emptyAttribution: {
      total: empty.length,
      overContext: empty.filter((row) => !row.generationEligible).length,
      withinContext: empty.filter((row) => row.generationEligible).length,
    },
    eligibleBySource: groupBy(eligible, (row) => row.source),
    eligibleByPromptLength: groupBy(eligible, (row) => lengthBin(row.promptTokens)),
    eligibleByConversationTurns: groupBy(eligible, turnBin),
    rows: joined.map((row) => ({
      id: row.id,
      source: row.source,
      promptTokens: row.promptTokens,
      userTurns: row.userTurns,
      assistantTurns: row.assistantTurns,
      generationEligible: row.generationEligible,
      hitBlockLimit: row.hitBlockLimit,
      nonempty: row.nonempty,
      eosTerminated: row.eosTerminated,
      degenerateLoop: row.degenerateLoop,
      fourGramRepeatRate: row.fourGramRepeatRate,
      structuralPass: row.structuralPass,
    })),
  };
  await writeFile(args.out, `${JSON.stringify(audit, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify({
    out: args.out,
    all: audit.all,
    generationEligible: audit.generationEligible,
    emptyAttribution: audit.emptyAttribution,
  }, null, 2)}\n`);
}

await main();
