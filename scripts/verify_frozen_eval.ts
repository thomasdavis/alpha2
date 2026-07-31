#!/usr/bin/env npx tsx
/** Verify frozen chat structure and Alpha/Hugging Face tokenizer parity. */

import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import * as readline from "node:readline";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import type { TokenizerArtifacts } from "@alpha/core";

interface ChatRow {
  id: string;
  source: string;
  messages: { role: "user" | "assistant"; content: string }[];
  reference: string;
  prompt_tokens: number;
}

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

function renderPrompt(messages: ChatRow["messages"]): string {
  return messages
    .map((message) => `${message.role === "user" ? "<|user|>" : "<|assistant|>"} ${message.content}`)
    .join(" ") + " <|assistant|>";
}

async function main(): Promise<void> {
  const cli = args();
  if (!cli.chat || !cli.tokenizer) throw new Error("required: --chat and --tokenizer");
  const artifacts = JSON.parse(await readFile(cli.tokenizer, "utf8")) as TokenizerArtifacts;
  const tokenizer = tokenizerFromArtifacts(artifacts);
  const reader = readline.createInterface({
    input: createReadStream(cli.chat, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  const ids = new Set<string>();
  const sources: Record<string, number> = {};
  let rows = 0;
  let maxPromptTokens = 0;
  let multiTurnRows = 0;
  for await (const line of reader) {
    if (!line) continue;
    const row = JSON.parse(line) as ChatRow;
    rows++;
    if (!row.id || ids.has(row.id)) throw new Error(`row ${rows}: missing/duplicate id ${row.id}`);
    ids.add(row.id);
    if (!row.reference?.trim()) throw new Error(`row ${rows}: empty reference`);
    if (row.messages.length === 0 || row.messages.at(-1)?.role !== "user") {
      throw new Error(`row ${rows}: prompt must be non-empty and end in user`);
    }
    for (let i = 0; i < row.messages.length; i++) {
      const expected = i % 2 === 0 ? "user" : "assistant";
      if (row.messages[i].role !== expected || !row.messages[i].content.trim()) {
        throw new Error(`row ${rows}: invalid turn ${i}`);
      }
    }
    const alphaTokens = tokenizer.encode(renderPrompt(row.messages)).length;
    if (alphaTokens !== row.prompt_tokens) {
      throw new Error(`row ${rows}: Alpha tokens ${alphaTokens} != HF-built ${row.prompt_tokens}`);
    }
    maxPromptTokens = Math.max(maxPromptTokens, alphaTokens);
    if (row.messages.length > 1) multiTurnRows++;
    sources[row.source] = (sources[row.source] ?? 0) + 1;
  }
  if (rows !== 100) throw new Error(`expected 100 chat rows, found ${rows}`);
  const report = {
    schema: "alpha-frozen-chat-audit-v1",
    result: "PASS",
    rows,
    unique_ids: ids.size,
    multi_turn_rows: multiTurnRows,
    sources,
    max_prompt_tokens: maxPromptTokens,
    alpha_hf_prompt_length_parity: `${rows}/${rows}`,
    tokenizer: { path: cli.tokenizer, type: artifacts.type, vocab_size: tokenizer.vocabSize },
  };
  if (cli.out) await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
