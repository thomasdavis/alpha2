#!/usr/bin/env npx tsx
/** Build a hash-bound, reference-blinded packet for the separate human D3 review. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";

interface FrozenPrompt {
  id: string;
  source: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  reference: string;
  prompt_tokens: number;
}

interface ChatResult {
  id: string;
  text: string;
  eosTerminated: boolean;
  roleLeak: boolean;
  nonempty: boolean;
  fourGramRepeatRate: number;
  degenerateLoop: boolean;
  structuralPass: boolean;
}

interface FrozenManifest {
  schema?: string;
  status?: string;
  final?: { chat?: { rows?: number; sha256?: string } };
}

interface FrozenSummary {
  schema?: string;
  checkpoint?: { step?: number; sha256?: string };
  inputs?: { chat?: { rows?: number; sha256?: string } };
  outputs?: { chat?: { filename?: string; rows?: number; sha256?: string } };
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

function assertSha256(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(`${label} is not a lowercase SHA-256: ${String(value)}`);
  }
}

function assertUniqueIds(rows: readonly { id: string }[], label: string): void {
  const ids = new Set<string>();
  for (const [index, row] of rows.entries()) {
    if (typeof row.id !== "string" || row.id.length === 0 || ids.has(row.id)) {
      throw new Error(`${label} row ${index + 1} has a missing/duplicate id: ${String(row.id)}`);
    }
    ids.add(row.id);
  }
}

function validatePrompt(row: FrozenPrompt, index: number): void {
  if (typeof row.source !== "string" || row.source.length === 0) {
    throw new Error(`prompt row ${index + 1} has no source`);
  }
  if (!Array.isArray(row.messages) || row.messages.length === 0) {
    throw new Error(`prompt row ${index + 1} has no messages`);
  }
  for (const [messageIndex, message] of row.messages.entries()) {
    const expectedRole = messageIndex % 2 === 0 ? "user" : "assistant";
    assertEqual(message.role, expectedRole, `prompt row ${index + 1} message ${messageIndex + 1} role`);
    if (typeof message.content !== "string" || message.content.trim().length === 0) {
      throw new Error(`prompt row ${index + 1} message ${messageIndex + 1} has no content`);
    }
  }
  assertEqual(row.messages.at(-1)?.role, "user", `prompt row ${index + 1} final role`);
  if (typeof row.reference !== "string" || row.reference.trim().length === 0) {
    throw new Error(`prompt row ${index + 1} has no held-out reference`);
  }
  if (!Number.isSafeInteger(row.prompt_tokens) || row.prompt_tokens < 1 || row.prompt_tokens > 896) {
    throw new Error(`prompt row ${index + 1} has invalid prompt_tokens: ${String(row.prompt_tokens)}`);
  }
}

function validateResult(row: ChatResult, index: number): void {
  if (typeof row.text !== "string") throw new Error(`result row ${index + 1} has invalid text`);
  for (const key of ["eosTerminated", "roleLeak", "nonempty", "degenerateLoop", "structuralPass"] as const) {
    if (typeof row[key] !== "boolean") throw new Error(`result row ${index + 1} has invalid ${key}`);
  }
  if (!Number.isFinite(row.fourGramRepeatRate) || row.fourGramRepeatRate < 0 || row.fourGramRepeatRate > 1) {
    throw new Error(`result row ${index + 1} has invalid fourGramRepeatRate`);
  }
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.prompts || !cli.results || !cli.summary || !cli.manifest || !cli.out) {
    throw new Error("required: --prompts, --results, --summary, --manifest, and --out");
  }
  const [promptText, resultText, summaryText, manifestText] = await Promise.all([
    readFile(cli.prompts, "utf8"),
    readFile(cli.results, "utf8"),
    readFile(cli.summary, "utf8"),
    readFile(cli.manifest, "utf8"),
  ]);
  const prompts = parseJsonl<FrozenPrompt>(promptText, cli.prompts);
  const results = parseJsonl<ChatResult>(resultText, cli.results);
  const summary = JSON.parse(summaryText) as FrozenSummary;
  const manifest = JSON.parse(manifestText) as FrozenManifest;
  assertEqual(manifest.schema, "alpha-frozen-eval-v1", "frozen manifest schema");
  assertEqual(manifest.status, "final", "frozen manifest status");
  assertEqual(manifest.final?.chat?.rows, 100, "frozen manifest chat rows");
  assertSha256(manifest.final?.chat?.sha256, "frozen manifest chat SHA-256");
  assertEqual(summary.schema, "alpha-frozen-eval-results-v2", "frozen result schema");
  assertEqual(summary.checkpoint?.step, 30_322, "chat checkpoint step");
  assertSha256(summary.checkpoint?.sha256, "chat checkpoint SHA-256");
  assertEqual(summary.inputs?.chat?.rows, 100, "summary chat input rows");
  assertEqual(summary.inputs?.chat?.sha256, sha256(promptText), "summary chat input SHA-256");
  assertEqual(manifest.final.chat.sha256, sha256(promptText), "manifest chat input SHA-256");
  assertEqual(summary.outputs?.chat?.filename, path.basename(cli.results), "summary chat output filename");
  assertEqual(summary.outputs?.chat?.rows, 100, "summary chat output rows");
  assertEqual(summary.outputs?.chat?.sha256, sha256(resultText), "summary chat output SHA-256");
  assertEqual(prompts.length, 100, "frozen prompt rows");
  assertEqual(results.length, 100, "frozen result rows");
  assertUniqueIds(prompts, "prompts");
  assertUniqueIds(results, "results");
  prompts.forEach(validatePrompt);
  results.forEach(validateResult);
  assertEqual(prompts.map((row) => row.id).join("\n"), results.map((row) => row.id).join("\n"), "case order");

  const packet = {
    schema: "alpha-frozen-chat-semantic-review-packet-v1",
    status: "PENDING_HUMAN_REVIEW",
    scope: "All 100 frozen chat responses; open-ended conversational coherence only",
    rubric: {
      PASS: "Intelligible assistant response that addresses the latest user turn; simplistic or factually weak is allowed.",
      BORDERLINE: "Understandable assistant-like prose with substantial irrelevance, contradiction, or fragmentation.",
      FAIL: "Gibberish, word salad, role confusion, an empty answer, or degenerate repetition.",
      decision: "Review every case without the held-out reference, record a verdict and rationale, then judge whether the suite is clearly conversational rather than gibberish. Machine structure and factual QA remain separate gates.",
    },
    provenance: {
      manifest: { path: cli.manifest, sha256: sha256(manifestText) },
      prompts: { path: cli.prompts, sha256: sha256(promptText), rows: prompts.length },
      results: { path: cli.results, sha256: sha256(resultText), rows: results.length },
      summary: { path: cli.summary, sha256: sha256(summaryText) },
      checkpoint: summary.checkpoint,
    },
    reference_blinded: true,
    cases: prompts.map((prompt, index) => ({
      index: index + 1,
      id: prompt.id,
      messages: prompt.messages,
      model_response: results[index].text,
      machine: {
        structuralPass: results[index].structuralPass,
        eosTerminated: results[index].eosTerminated,
        roleLeak: results[index].roleLeak,
        degenerateLoop: results[index].degenerateLoop,
        fourGramRepeatRate: results[index].fourGramRepeatRate,
      },
      human_verdict: "PENDING",
      human_rationale: "",
    })),
  };
  await writeFile(cli.out, `${JSON.stringify(packet, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  console.log(`semantic review packet=PASS rows=${prompts.length} reference_blinded=true out=${cli.out}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
