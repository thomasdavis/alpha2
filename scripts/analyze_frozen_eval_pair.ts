#!/usr/bin/env npx tsx
/** Recompute and compare the machine-verifiable frozen-eval gates for base and chat checkpoints. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { isDeepStrictEqual } from "node:util";
import {
  answerIsContained,
  answerTokenF1,
  fourGramRepeatRate,
  normalizedAnswer,
} from "@alpha/train";

interface FrozenSummary {
  schema: string;
  checkpoint: { path: string; sha256: string; step: number; modelConfig: Record<string, unknown> };
  inputs: {
    chat: { path: string; sha256: string; rows: number };
    qa: { path: string; sha256: string; rows: number };
  };
  outputs: {
    chat: { filename: string; sha256: string; rows: number };
    qa: { filename: string; sha256: string; rows: number };
  };
  generation: { chatMaxTokens: number; qaMaxTokens: number; eosId: number; userId: number };
  chat: Record<string, number>;
  closedBookQa: Record<string, number>;
}

interface ChatResult {
  id: string;
  generatedIds: number[];
  text: string;
  eosTerminated: boolean;
  roleLeak: boolean;
  nonempty: boolean;
  fourGramRepeatRate: number;
  degenerateLoop: boolean;
  structuralPass: boolean;
}

interface QaResult {
  id: string;
  expected: string;
  text: string;
  normalizedPrediction: string;
  normalizedExpected: string;
  exactMatch: boolean;
  answerContained: boolean;
  tokenF1: number;
}

interface VerifiedRun {
  dir: string;
  summary: FrozenSummary;
  chat: ChatResult[];
  qa: QaResult[];
}

function parseArgs(): Record<string, string> {
  const values: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    values[arg.slice(2)] = value;
  }
  return values;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parseJsonl<T>(text: string, label: string): T[] {
  return text.split("\n").filter((line) => line.length > 0).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`${label}:${index + 1} is invalid JSON`, { cause: error });
    }
  });
}

function assertEqual(actual: unknown, expected: unknown, label: string): void {
  if (actual !== expected) throw new Error(`${label}: ${String(actual)} != ${String(expected)}`);
}

function assertFinite(value: unknown, label: string): asserts value is number {
  if (!Number.isFinite(value)) throw new Error(`${label} is not finite: ${String(value)}`);
}

function assertSha256(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${label} is not a lowercase SHA-256: ${String(value)}`);
  }
}

function assertUniqueIds(rows: readonly { id: string }[], label: string): void {
  const ids = new Set<string>();
  for (const [index, row] of rows.entries()) {
    if (!row.id || ids.has(row.id)) throw new Error(`${label} row ${index + 1} has a missing/duplicate id: ${row.id}`);
    ids.add(row.id);
  }
}

async function verifyRun(dir: string): Promise<VerifiedRun> {
  const [summaryText, chatText, qaText] = await Promise.all([
    readFile(path.join(dir, "summary.json"), "utf8"),
    readFile(path.join(dir, "chat-results.jsonl"), "utf8"),
    readFile(path.join(dir, "qa-results.jsonl"), "utf8"),
  ]);
  const summary = JSON.parse(summaryText) as FrozenSummary;
  const chat = parseJsonl<ChatResult>(chatText, `${dir}/chat-results.jsonl`);
  const qa = parseJsonl<QaResult>(qaText, `${dir}/qa-results.jsonl`);
  assertEqual(summary.schema, "alpha-frozen-eval-results-v2", `${dir} summary schema`);
  assertEqual(summary.outputs?.chat?.filename, "chat-results.jsonl", `${dir} chat output filename`);
  assertEqual(summary.outputs?.qa?.filename, "qa-results.jsonl", `${dir} QA output filename`);
  assertEqual(summary.outputs.chat.sha256, sha256(chatText), `${dir} chat output SHA-256`);
  assertEqual(summary.outputs.qa.sha256, sha256(qaText), `${dir} QA output SHA-256`);
  assertEqual(summary.outputs.chat.rows, chat.length, `${dir} chat output rows`);
  assertEqual(summary.outputs.qa.rows, qa.length, `${dir} QA output rows`);
  assertEqual(chat.length, 100, `${dir} frozen chat rows`);
  assertEqual(qa.length, 200, `${dir} frozen QA rows`);
  assertEqual(summary.inputs.chat.rows, 100, `${dir} frozen chat input rows`);
  assertEqual(summary.inputs.qa.rows, 200, `${dir} frozen QA input rows`);
  assertEqual(summary.generation.chatMaxTokens, 128, `${dir} chat generation limit`);
  assertEqual(summary.generation.qaMaxTokens, 64, `${dir} QA generation limit`);
  if (!Number.isSafeInteger(summary.generation.eosId) || summary.generation.eosId < 0) {
    throw new Error(`${dir} EOS token id is invalid`);
  }
  if (!Number.isSafeInteger(summary.generation.userId) || summary.generation.userId < 0) {
    throw new Error(`${dir} user token id is invalid`);
  }
  for (const [label, value] of [
    ["checkpoint", summary.checkpoint.sha256],
    ["chat input", summary.inputs.chat.sha256],
    ["QA input", summary.inputs.qa.sha256],
    ["chat output", summary.outputs.chat.sha256],
    ["QA output", summary.outputs.qa.sha256],
  ] as const) assertSha256(value, `${dir} ${label} SHA-256`);
  assertUniqueIds(chat, `${dir} chat`);
  assertUniqueIds(qa, `${dir} QA`);

  for (const [index, row] of chat.entries()) {
    if (!Array.isArray(row.generatedIds) || row.generatedIds.some((token) => !Number.isSafeInteger(token) || token < 0)) {
      throw new Error(`${dir} chat row ${index + 1} has invalid generated token ids`);
    }
    const contentIds = row.eosTerminated ? row.generatedIds.slice(0, -1) : row.generatedIds;
    const eosTerminated = row.generatedIds.at(-1) === summary.generation.eosId;
    assertEqual(row.eosTerminated, eosTerminated, `${dir} chat row ${index + 1} EOS flag`);
    assertEqual(
      row.roleLeak,
      contentIds.includes(summary.generation.userId),
      `${dir} chat row ${index + 1} role-leak flag`,
    );
    const repeatRate = fourGramRepeatRate(contentIds);
    assertFinite(row.fourGramRepeatRate, `${dir} chat row ${index + 1} repeat rate`);
    if (Math.abs(row.fourGramRepeatRate - repeatRate) > 1e-12) {
      throw new Error(`${dir} chat row ${index + 1} repeat rate does not recompute`);
    }
    const nonempty = row.text.trim().length > 0;
    assertEqual(row.nonempty, nonempty, `${dir} chat row ${index + 1} nonempty`);
    assertEqual(row.degenerateLoop, repeatRate >= 0.2, `${dir} chat row ${index + 1} loop flag`);
    assertEqual(
      row.structuralPass,
      row.eosTerminated && !row.roleLeak && nonempty,
      `${dir} chat row ${index + 1} structural flag`,
    );
  }
  for (const [index, row] of qa.entries()) {
    const prediction = normalizedAnswer(row.text);
    const expected = normalizedAnswer(row.expected);
    const contained = answerIsContained(row.text, row.expected);
    const tokenF1 = answerTokenF1(row.text, row.expected);
    assertEqual(row.normalizedPrediction, prediction, `${dir} QA row ${index + 1} normalized prediction`);
    assertEqual(row.normalizedExpected, expected, `${dir} QA row ${index + 1} normalized expected`);
    assertEqual(row.exactMatch, prediction === expected, `${dir} QA row ${index + 1} exact match`);
    assertEqual(row.answerContained, contained, `${dir} QA row ${index + 1} containment`);
    assertFinite(row.tokenF1, `${dir} QA row ${index + 1} token F1`);
    if (Math.abs(row.tokenF1 - tokenF1) > 1e-12) throw new Error(`${dir} QA row ${index + 1} F1 does not recompute`);
  }

  const chatMetrics = {
    total: chat.length,
    structuralPass: chat.filter((row) => row.structuralPass).length,
    eosTerminated: chat.filter((row) => row.eosTerminated).length,
    roleLeaks: chat.filter((row) => row.roleLeak).length,
    nonempty: chat.filter((row) => row.nonempty).length,
    degenerateLoops: chat.filter((row) => row.degenerateLoop).length,
    meanFourGramRepeatRate: mean(chat.map((row) => row.fourGramRepeatRate)),
    maxFourGramRepeatRate: Math.max(0, ...chat.map((row) => row.fourGramRepeatRate)),
  };
  const qaMetrics = {
    total: qa.length,
    exactMatch: qa.filter((row) => row.exactMatch).length,
    answerContained: qa.filter((row) => row.answerContained).length,
    meanTokenF1: mean(qa.map((row) => row.tokenF1)),
  };
  for (const [key, value] of Object.entries(chatMetrics)) {
    const recorded = summary.chat[key];
    assertFinite(recorded, `${dir} summary chat.${key}`);
    if (Math.abs(recorded - value) > 1e-12) throw new Error(`${dir} summary chat.${key} does not recompute`);
  }
  for (const [key, value] of Object.entries(qaMetrics)) {
    const recorded = summary.closedBookQa[key];
    assertFinite(recorded, `${dir} summary closedBookQa.${key}`);
    if (Math.abs(recorded - value) > 1e-12) throw new Error(`${dir} summary closedBookQa.${key} does not recompute`);
  }
  return { dir, summary, chat, qa };
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.base || !cli.chat || !cli.manifest || !cli.out) {
    throw new Error("required: --base, --chat, --manifest, and --out");
  }
  const [base, chat, manifestText] = await Promise.all([
    verifyRun(cli.base),
    verifyRun(cli.chat),
    readFile(cli.manifest, "utf8"),
  ]);
  const manifest = JSON.parse(manifestText) as {
    schema?: string;
    status?: string;
    final?: {
      chat?: { rows?: number; sha256?: string };
      closed_book_qa?: { rows?: number; sha256?: string };
    };
  };
  assertEqual(manifest.schema, "alpha-frozen-eval-v1", "frozen manifest schema");
  assertEqual(manifest.status, "final", "frozen manifest status");
  assertEqual(manifest.final?.chat?.rows, 100, "frozen manifest chat rows");
  assertEqual(manifest.final?.closed_book_qa?.rows, 200, "frozen manifest QA rows");
  assertSha256(manifest.final?.chat?.sha256, "frozen manifest chat SHA-256");
  assertSha256(manifest.final?.closed_book_qa?.sha256, "frozen manifest QA SHA-256");
  assertEqual(base.summary.checkpoint.step, 61_036, "base checkpoint step");
  assertEqual(chat.summary.checkpoint.step, 30_322, "chat checkpoint step");
  if (!isDeepStrictEqual(base.summary.checkpoint.modelConfig, chat.summary.checkpoint.modelConfig)) {
    throw new Error("base/chat model config differs");
  }
  for (const suite of ["chat", "qa"] as const) {
    assertEqual(base.summary.inputs[suite].sha256, chat.summary.inputs[suite].sha256, `${suite} input SHA-256`);
    assertEqual(base.summary.inputs[suite].rows, chat.summary.inputs[suite].rows, `${suite} input rows`);
  }
  assertEqual(base.summary.inputs.chat.sha256, manifest.final.chat.sha256, "chat input frozen-manifest SHA-256");
  assertEqual(base.summary.inputs.qa.sha256, manifest.final.closed_book_qa.sha256, "QA input frozen-manifest SHA-256");
  assertEqual(base.chat.map((row) => row.id).join("\n"), chat.chat.map((row) => row.id).join("\n"), "chat case order");
  assertEqual(base.qa.map((row) => row.id).join("\n"), chat.qa.map((row) => row.id).join("\n"), "QA case order");

  const machineGatePass =
    chat.summary.chat.structuralPass >= 95 &&
    chat.summary.chat.degenerateLoops === 0 &&
    chat.summary.chat.maxFourGramRepeatRate < 0.2;
  const report = {
    schema: "alpha-frozen-eval-pair-analysis-v1",
    result: machineGatePass ? "PASS" : "FAIL",
    scope: "machine-verifiable D3 structural/repetition gate only",
    semantic_coherence: "REQUIRES_SEPARATE_REVIEW; this analyzer does not classify open-ended language quality",
    frozen_manifest: { path: cli.manifest, sha256: sha256(manifestText), schema: manifest.schema, status: manifest.status },
    gate: {
      structural_pass_minimum: 95,
      degenerate_loops_maximum: 0,
      per_sample_four_gram_repeat_rate_exclusive_maximum: 0.2,
    },
    inputs_match: true,
    base: base.summary,
    chat: chat.summary,
    deltas_chat_minus_base: {
      structuralPass: chat.summary.chat.structuralPass - base.summary.chat.structuralPass,
      eosTerminated: chat.summary.chat.eosTerminated - base.summary.chat.eosTerminated,
      roleLeaks: chat.summary.chat.roleLeaks - base.summary.chat.roleLeaks,
      nonempty: chat.summary.chat.nonempty - base.summary.chat.nonempty,
      degenerateLoops: chat.summary.chat.degenerateLoops - base.summary.chat.degenerateLoops,
      meanFourGramRepeatRate: chat.summary.chat.meanFourGramRepeatRate - base.summary.chat.meanFourGramRepeatRate,
      qaExactMatch: chat.summary.closedBookQa.exactMatch - base.summary.closedBookQa.exactMatch,
      qaAnswerContained: chat.summary.closedBookQa.answerContained - base.summary.closedBookQa.answerContained,
      qaMeanTokenF1: chat.summary.closedBookQa.meanTokenF1 - base.summary.closedBookQa.meanTokenF1,
    },
  };
  await writeFile(cli.out, `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (!machineGatePass) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
