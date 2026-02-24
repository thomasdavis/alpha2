/**
 * Export committed conversations to training data.
 *
 * Formats:
 *   jsonl — OpenAI fine-tuning format with messages array
 *   chat  — Flat token format: <|user|> msg <|assistant|> msg ... <|end_of_text|>
 *           No names, no system prompt. Ready to concatenate with other training sets.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { getCommittedConversations, getLatestRun } from "./db.js";
import { loadFigures } from "./figures.js";
import type { ExportOptions, ExportFormat, GeneratedConversation } from "./types.js";

// ── JSONL format (OpenAI fine-tuning) ──

interface TrainingMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

function conversationToJsonl(conv: GeneratedConversation): string {
  const figures = loadFigures();
  const figA = figures.find((f) => f.id === conv.figureA);
  const figB = figures.find((f) => f.id === conv.figureB);
  const figAName = figA?.name ?? conv.figureA;
  const figBName = figB?.name ?? conv.figureB;

  const system = `A conversation between ${figAName} and ${figBName} about ${conv.topic}. ${figAName} and ${figBName} engage in authentic dialogue reflecting their historical perspectives, speech patterns, and worldviews.`;

  const messages: TrainingMessage[] = [{ role: "system", content: system }];
  for (const turn of conv.turns) {
    const role = turn.speaker === figAName ? "user" : "assistant";
    messages.push({ role, content: `${turn.speaker}: ${turn.text}` });
  }

  return JSON.stringify({ messages });
}

// ── Chat format (flat token format for custom training) ──

function conversationToChat(conv: GeneratedConversation): string {
  const figures = loadFigures();
  const figA = figures.find((f) => f.id === conv.figureA);
  const figAName = figA?.name ?? conv.figureA;

  const parts: string[] = [];

  for (const turn of conv.turns) {
    const role = turn.speaker === figAName ? "user" : "assistant";
    // Strip "Name: " prefix if present, just use the raw text
    parts.push(`<|${role}|> ${turn.text}`);
  }

  // Ensure ends with assistant turn
  // (turns alternate starting with figA=user, so even-length = ends on assistant)
  // If last turn is user, drop it so we end on assistant
  if (parts.length > 0 && parts[parts.length - 1]!.startsWith("<|user|>")) {
    parts.pop();
  }

  return parts.join(" ") + " <|end_of_text|>";
}

// ── Export dispatcher ──

export async function exportTrainingData(opts: ExportOptions): Promise<{ count: number; path: string }> {
  let runId = opts.runId;

  if (!runId) {
    const latest = await getLatestRun();
    if (!latest) throw new Error("No runs found. Generate conversations first.");
    runId = latest.id;
  }

  const conversations = await getCommittedConversations(runId);
  if (conversations.length === 0) {
    throw new Error(`No committed conversations found for run ${runId}`);
  }

  const formatter = opts.format === "chat" ? conversationToChat : conversationToJsonl;
  const lines = conversations.map(formatter).join("\n") + "\n";

  mkdirSync(dirname(opts.out), { recursive: true });
  writeFileSync(opts.out, lines, "utf-8");

  console.log(`Exported ${conversations.length} conversations (${opts.format}) to ${opts.out}`);
  return { count: conversations.length, path: opts.out };
}
