#!/usr/bin/env npx tsx
/** Verify all human annotations and seal the separate semantic portion of D3. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";

type Verdict = "PASS" | "BORDERLINE" | "FAIL";

interface FrozenPrompt {
  id: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
}

interface ChatResult {
  id: string;
  text: string;
  eosTerminated: boolean;
  roleLeak: boolean;
  degenerateLoop: boolean;
  structuralPass: boolean;
  fourGramRepeatRate: number;
}

interface ReviewCase {
  index: number;
  id: string;
  messages: FrozenPrompt["messages"];
  model_response: string;
  machine: Omit<ChatResult, "id" | "text">;
  human_verdict: Verdict | "PENDING";
  human_rationale: string;
}

interface ReviewPacket {
  schema?: string;
  status?: string;
  semantic_gate?: { pass_minimum?: number; fail_maximum?: number };
  provenance?: {
    manifest?: { path?: string; sha256?: string };
    prompts?: { path?: string; sha256?: string; rows?: number };
    results?: { path?: string; sha256?: string; rows?: number };
    summary?: { path?: string; sha256?: string };
    checkpoint?: { step?: number; sha256?: string };
  };
  reference_blinded?: boolean;
  reviewer?: string;
  reviewed_utc?: string;
  overall_rationale?: string;
  cases?: ReviewCase[];
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

function requireString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.trim().length === 0 || value === "PENDING") {
    throw new Error(`${label} is incomplete`);
  }
}

function requirePath(value: unknown, label: string): asserts value is string {
  requireString(value, label);
  if (!path.isAbsolute(value)) throw new Error(`${label} is not absolute: ${value}`);
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.review || !cli.out) throw new Error("required: --review and --out");
  const reviewText = await readFile(cli.review, "utf8");
  const review = JSON.parse(reviewText) as ReviewPacket;
  assertEqual(review.schema, "alpha-frozen-chat-semantic-review-packet-v1", "review schema");
  assertEqual(review.status, "COMPLETE", "review status");
  assertEqual(review.reference_blinded, true, "reference blinding");
  assertEqual(review.semantic_gate?.pass_minimum, 80, "semantic PASS minimum");
  assertEqual(review.semantic_gate?.fail_maximum, 0, "semantic FAIL maximum");
  requireString(review.reviewer, "reviewer");
  requireString(review.reviewed_utc, "reviewed_utc");
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$/.test(review.reviewed_utc) ||
      !Number.isFinite(Date.parse(review.reviewed_utc))) {
    throw new Error(`reviewed_utc is not a UTC ISO timestamp: ${review.reviewed_utc}`);
  }
  requireString(review.overall_rationale, "overall_rationale");
  const provenance = review.provenance;
  requirePath(provenance?.manifest?.path, "manifest path");
  requirePath(provenance?.prompts?.path, "prompts path");
  requirePath(provenance?.results?.path, "results path");
  requirePath(provenance?.summary?.path, "summary path");
  const [manifestText, promptText, resultText, summaryText] = await Promise.all([
    readFile(provenance.manifest.path, "utf8"),
    readFile(provenance.prompts.path, "utf8"),
    readFile(provenance.results.path, "utf8"),
    readFile(provenance.summary.path, "utf8"),
  ]);
  assertEqual(provenance.manifest.sha256, sha256(manifestText), "manifest SHA-256");
  assertEqual(provenance.prompts.sha256, sha256(promptText), "prompts SHA-256");
  assertEqual(provenance.results.sha256, sha256(resultText), "results SHA-256");
  assertEqual(provenance.summary.sha256, sha256(summaryText), "summary SHA-256");
  const manifest = JSON.parse(manifestText) as {
    schema?: string;
    status?: string;
    final?: { chat?: { rows?: number; sha256?: string } };
  };
  const summary = JSON.parse(summaryText) as {
    schema?: string;
    checkpoint?: { step?: number; sha256?: string };
    inputs?: { chat?: { rows?: number; sha256?: string } };
    outputs?: { chat?: { filename?: string; rows?: number; sha256?: string } };
  };
  assertEqual(manifest.schema, "alpha-frozen-eval-v1", "manifest schema");
  assertEqual(manifest.status, "final", "manifest status");
  assertEqual(manifest.final?.chat?.rows, 100, "manifest chat rows");
  assertEqual(manifest.final?.chat?.sha256, sha256(promptText), "manifest chat SHA-256");
  assertEqual(summary.schema, "alpha-frozen-eval-results-v2", "summary schema");
  assertEqual(summary.checkpoint?.step, 30_322, "checkpoint step");
  assertEqual(summary.checkpoint?.sha256, provenance.checkpoint?.sha256, "checkpoint SHA-256");
  assertEqual(provenance.checkpoint?.step, 30_322, "provenance checkpoint step");
  assertEqual(summary.inputs?.chat?.rows, 100, "summary chat input rows");
  assertEqual(summary.inputs?.chat?.sha256, sha256(promptText), "summary chat input SHA-256");
  assertEqual(summary.outputs?.chat?.filename, path.basename(provenance.results.path), "summary result filename");
  assertEqual(summary.outputs?.chat?.rows, 100, "summary chat output rows");
  assertEqual(summary.outputs?.chat?.sha256, sha256(resultText), "summary chat output SHA-256");
  const prompts = parseJsonl<FrozenPrompt>(promptText, provenance.prompts.path);
  const results = parseJsonl<ChatResult>(resultText, provenance.results.path);
  const cases = review.cases;
  if (!Array.isArray(cases)) throw new Error("review cases are missing");
  assertEqual(prompts.length, 100, "prompt rows");
  assertEqual(results.length, 100, "result rows");
  assertEqual(cases.length, 100, "review rows");
  assertEqual(provenance.prompts.rows, 100, "provenance prompt rows");
  assertEqual(provenance.results.rows, 100, "provenance result rows");
  const ids = new Set<string>();
  const counts: Record<Verdict, number> = { PASS: 0, BORDERLINE: 0, FAIL: 0 };
  for (let index = 0; index < cases.length; index++) {
    const row = cases[index];
    const prompt = prompts[index];
    const result = results[index];
    assertEqual(row.index, index + 1, `review row ${index + 1} index`);
    assertEqual(row.id, prompt.id, `review row ${index + 1} prompt id`);
    assertEqual(row.id, result.id, `review row ${index + 1} result id`);
    if (ids.has(row.id)) throw new Error(`duplicate review id: ${row.id}`);
    ids.add(row.id);
    assertEqual(JSON.stringify(row.messages), JSON.stringify(prompt.messages), `review row ${index + 1} messages`);
    assertEqual(row.model_response, result.text, `review row ${index + 1} model response`);
    for (const key of ["eosTerminated", "roleLeak", "degenerateLoop", "structuralPass", "fourGramRepeatRate"] as const) {
      assertEqual(row.machine[key], result[key], `review row ${index + 1} machine ${key}`);
    }
    if (!(["PASS", "BORDERLINE", "FAIL"] as const).includes(row.human_verdict as Verdict)) {
      throw new Error(`review row ${index + 1} has incomplete verdict: ${String(row.human_verdict)}`);
    }
    requireString(row.human_rationale, `review row ${index + 1} rationale`);
    counts[row.human_verdict as Verdict] += 1;
  }
  const result = counts.PASS >= 80 && counts.FAIL === 0 ? "PASS" : "FAIL";
  const report = {
    schema: "alpha-frozen-chat-semantic-review-v1",
    result,
    scope: "Human open-ended conversational-coherence portion of D3 only",
    gate: { pass_minimum: 80, fail_maximum: 0 },
    review: { path: cli.review, sha256: sha256(reviewText), reviewer: review.reviewer, reviewed_utc: review.reviewed_utc },
    provenance,
    reference_blinded: true,
    counts: { total: cases.length, ...counts },
    borderline_ids: cases.filter((row) => row.human_verdict === "BORDERLINE").map((row) => row.id),
    fail_ids: cases.filter((row) => row.human_verdict === "FAIL").map((row) => row.id),
    overall_rationale: review.overall_rationale,
  };
  await writeFile(cli.out, `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
  if (result !== "PASS") process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
