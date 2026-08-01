import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { formatAlphaChat } from "./chat.js";
import type { ChatMessage } from "./types.js";

export interface LoadedPrompts {
  readonly prompts: string[];
  readonly fingerprint: string;
  readonly rawBytes: number;
}

export async function loadLensPrompts(path: string): Promise<LoadedPrompts> {
  const raw = await readFile(path);
  const text = raw.toString("utf8");
  const prompts: string[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) continue;
    let value: unknown = line;
    if (line.trimStart().startsWith("{") || line.trimStart().startsWith("\"")) {
      try { value = JSON.parse(line); }
      catch (error) { throw new Error(`invalid JSONL at line ${index + 1}: ${String(error)}`); }
    }
    if (typeof value === "string") prompts.push(value);
    else if (value && typeof value === "object") {
      const record = value as { text?: unknown; prompt?: unknown; messages?: unknown };
      if (typeof record.text === "string") prompts.push(record.text);
      else if (typeof record.prompt === "string") prompts.push(record.prompt);
      else if (Array.isArray(record.messages)) prompts.push(formatAlphaChat(record.messages as ChatMessage[]));
      else throw new Error(`JSONL line ${index + 1} has no text, prompt, or messages`);
    } else throw new Error(`unsupported prompt at line ${index + 1}`);
  }
  return {
    prompts,
    fingerprint: `sha256:${createHash("sha256").update(raw).digest("hex")}`,
    rawBytes: raw.byteLength,
  };
}
