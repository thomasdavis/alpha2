#!/usr/bin/env npx tsx
/** Render the pre-frozen qualitative subset from an immutable development run. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

interface PromptRow {
  id: string;
  source: string;
  prompt_tokens: number;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  reference: string;
}

interface ResultRow {
  id: string;
  source: string;
  promptTokens: number;
  text: string;
  eosTerminated: boolean;
  roleLeak: boolean;
  nonempty: boolean;
  fourGramRepeatRate: number;
  degenerateLoop: boolean;
  structuralPass: boolean;
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`expected --key value, received ${key ?? ""} ${value ?? ""}`.trim());
    }
    result[key.slice(2)] = value;
  }
  return result;
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

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function block(value: string): string {
  return value.split("\n").map((line) => `    ${line}`).join("\n");
}

async function main(): Promise<void> {
  const args = parseArgs();
  if (!args.panel || !args.results || !args.out) {
    throw new Error("required: --panel, --results, and --out");
  }
  const [panelText, resultsText] = await Promise.all([
    readFile(args.panel, "utf8"),
    readFile(args.results, "utf8"),
  ]);
  const panel = parseJsonl<PromptRow>(panelText, args.panel);
  const results = parseJsonl<ResultRow>(resultsText, args.results);
  const byId = new Map(results.map((row) => [row.id, row]));
  if (byId.size !== results.length) throw new Error("result IDs are not unique");
  const sections = panel.map((prompt, index) => {
    const result = byId.get(prompt.id);
    if (!result) throw new Error(`panel prompt has no result: ${prompt.id}`);
    if (result.source !== prompt.source) throw new Error(`source mismatch for ${prompt.id}`);
    const transcript = prompt.messages
      .map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`)
      .join("\n");
    return [
      `## ${index + 1}. ${prompt.id}`,
      "",
      `Source: ${prompt.source}; prompt tokens: ${result.promptTokens}; EOS: ${result.eosTerminated}; ` +
        `loop: ${result.degenerateLoop}; repeat rate: ${result.fourGramRepeatRate.toFixed(4)}.`,
      "",
      "Conversation:",
      "",
      block(transcript),
      "",
      "Model output:",
      "",
      block(result.text || "[EMPTY]"),
      "",
      "Held-out source response (context only, not an exact-match target):",
      "",
      block(prompt.reference),
    ].join("\n");
  });
  const content = [
    `# ${args.title ?? "Alpha chat repair v2 — frozen qualitative panel"}`,
    "",
    `Panel input SHA-256: \`${sha256(panelText)}\`  `,
    `Development results SHA-256: \`${sha256(resultsText)}\`  `,
    `Rows: ${panel.length}`,
    "",
    "The source response is shown only as contextual evidence. Alpha is judged for directness, contingency, " +
      "coherence, naturalness, stopping, and absence of looping—not lexical imitation.",
    "",
    sections.join("\n\n"),
    "",
  ].join("\n");
  await writeFile(args.out, content, { encoding: "utf8", flag: "wx" });
  process.stdout.write(`${args.out}\n`);
}

await main();
