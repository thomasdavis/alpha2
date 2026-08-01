import type { ChatMessage } from "./types.js";

/** Exact system-folded, conversation-terminal-EOS Alpha chat formatting. */
export function formatAlphaChat(messages: readonly ChatMessage[]): string {
  if (messages.length === 0) throw new Error("messages must be non-empty");
  let cursor = 0;
  let firstUserPrefix = "<|user|> ";
  if (messages[0]?.role === "system") {
    const system = requireContent(messages[0], 0);
    firstUserPrefix += `[Instructions: ${system}]\n\n`;
    cursor = 1;
  }
  if (messages[cursor]?.role !== "user") {
    throw new Error("conversation must begin with a user message after an optional system message");
  }

  const parts: string[] = [];
  let expected: "user" | "assistant" = "user";
  for (; cursor < messages.length; cursor++) {
    const message = messages[cursor];
    if (message.role !== expected) {
      throw new Error(`roles must alternate user/assistant; expected ${expected} at message ${cursor}`);
    }
    const marker = expected === "user"
      ? (parts.length === 0 ? firstUserPrefix : "<|user|> ")
      : "<|assistant|> ";
    parts.push(`${marker}${requireContent(message, cursor)}`);
    expected = expected === "user" ? "assistant" : "user";
  }
  if (expected !== "assistant") throw new Error("conversation must end with a user message");
  return `${parts.join(" ")} <|assistant|>`;
}

function requireContent(message: ChatMessage, index: number): string {
  if (typeof message.content !== "string" || message.content.trim().length === 0) {
    throw new Error(`message ${index} has empty content`);
  }
  return message.content;
}
