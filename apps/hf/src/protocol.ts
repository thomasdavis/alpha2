export const MODEL_ID = process.env.MODEL_ID?.trim() || "ajaxdavis/alpha-60m-chat";
export const END_TOKEN = "<|end_of_text|>";
export const USER_TOKEN = "<|user|>";
export const ASSISTANT_TOKEN = "<|assistant|>";

export interface ChatStopTokenIds {
  readonly eos: number;
  readonly user: number;
  readonly assistant: number;
  readonly all: ReadonlySet<number>;
}

/** Resolve every token that ends the currently generated assistant turn.
 *
 * Alpha's serialized multi-turn chats place the next `<|user|>` marker directly
 * after an assistant response; there is no separate end-of-turn token between
 * them. A live chat runtime must therefore stop before emitting that marker,
 * while `<|end_of_text|>` remains the end of the complete serialized dialogue.
 */
export function resolveChatStopTokenIds(
  encode: (text: string) => ArrayLike<number>,
): ChatStopTokenIds {
  const atomic = (token: string): number => {
    const ids = Array.from(encode(token));
    if (ids.length !== 1) throw new Error(`${token} is not an atomic tokenizer token`);
    return ids[0]!;
  };
  const eos = atomic(END_TOKEN);
  const user = atomic(USER_TOKEN);
  const assistant = atomic(ASSISTANT_TOKEN);
  const all = new Set([eos, user, assistant]);
  if (all.size !== 3) throw new Error("chat stop tokens must have distinct token IDs");
  return { eos, user, assistant, all };
}

export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  readonly role: ChatRole;
  readonly content: string;
}

function requireContent(message: ChatMessage, index: number): string {
  if (typeof message.content !== "string" || !message.content.trim()) {
    throw new Error(`message ${index} has empty content`);
  }
  return message.content;
}

/** Render the same system-folded, conversation-terminal-EOS format used for Alpha SFT. */
export function formatChatPrompt(messages: readonly ChatMessage[]): string {
  if (messages.length === 0) throw new Error("messages must be non-empty");
  const validRoles = new Set<ChatRole>(["system", "user", "assistant"]);
  for (const [index, message] of messages.entries()) {
    if (!message || !validRoles.has(message.role)) throw new Error(`message ${index} has an invalid role`);
    requireContent(message, index);
  }

  let cursor = 0;
  let firstUserPrefix = "<|user|> ";
  if (messages[0]?.role === "system") {
    firstUserPrefix += `[Instructions: ${requireContent(messages[0], 0)}]\n\n`;
    cursor = 1;
  }
  if (messages[cursor]?.role !== "user") {
    throw new Error("conversation must begin with a user message after an optional system message");
  }

  const parts: string[] = [];
  let expected: "user" | "assistant" = "user";
  for (; cursor < messages.length; cursor++) {
    const message = messages[cursor]!;
    if (message.role !== expected) {
      throw new Error(`roles must alternate user/assistant; expected ${expected} at message ${cursor}`);
    }
    const marker = expected === "user" ? (parts.length === 0 ? firstUserPrefix : "<|user|> ") : "<|assistant|> ";
    parts.push(`${marker}${message.content}`);
    expected = expected === "user" ? "assistant" : "user";
  }
  if (expected !== "assistant") throw new Error("conversation must end with a user message");
  // The first generated content token carries its own leading space. Adding
  // one after the marker creates an out-of-distribution standalone token.
  return `${parts.join(" ")} <|assistant|>`;
}

export function parseMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) throw new Error("messages must be an array");
  return value.map((entry, index) => {
    if (typeof entry !== "object" || entry === null) throw new Error(`message ${index} must be an object`);
    const role = (entry as { role?: unknown }).role;
    const content = (entry as { content?: unknown }).content;
    if (role !== "system" && role !== "user" && role !== "assistant") {
      throw new Error(`message ${index} has an invalid role`);
    }
    if (typeof content !== "string") throw new Error(`message ${index} content must be a string`);
    return { role, content };
  });
}

export function finiteNumber(value: unknown, fallback: number, min: number, max: number): number {
  const candidate = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(candidate) || candidate < min || candidate > max) {
    throw new Error(`numeric option must be between ${min} and ${max}`);
  }
  return candidate;
}

export function positiveInteger(value: unknown, fallback: number, max: number): number {
  const candidate = value === undefined ? fallback : Number(value);
  if (!Number.isInteger(candidate) || candidate < 1 || candidate > max) {
    throw new Error(`token limit must be an integer between 1 and ${max}`);
  }
  return candidate;
}
