/**
 * Export committed conversations to training JSONL.
 *
 * Format: Figure A turns → "user" role, Figure B turns → "assistant" role.
 * System prompt provides context.
 */
import { writeFileSync } from "node:fs";
import { getCommittedConversations, getLatestRun } from "./db.js";
import { loadFigures } from "./figures.js";
import type { ExportOptions, GeneratedConversation } from "./types.js";

interface TrainingMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface TrainingExample {
  messages: TrainingMessage[];
}

function conversationToTraining(conv: GeneratedConversation): TrainingExample {
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

  return { messages };
}

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

  const lines = conversations
    .map((c) => JSON.stringify(conversationToTraining(c)))
    .join("\n") + "\n";

  writeFileSync(opts.out, lines, "utf-8");

  console.log(`Exported ${conversations.length} conversations to ${opts.out}`);
  return { count: conversations.length, path: opts.out };
}
