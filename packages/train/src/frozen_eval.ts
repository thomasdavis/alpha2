/** Pure formatting and scoring helpers for Alpha's frozen greedy eval. */

export interface FrozenChatMessage {
  readonly role: "user" | "assistant";
  readonly content: string;
}

/** Render history exactly like scripts/build_sft_corpus.py, then open an assistant turn. */
export function formatFrozenChatPrompt(messages: readonly FrozenChatMessage[]): string {
  if (messages.length === 0 || messages[messages.length - 1]?.role !== "user") {
    throw new Error("frozen chat history must be non-empty and end with a user turn");
  }
  const parts: string[] = [];
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const expected = i % 2 === 0 ? "user" : "assistant";
    if (message.role !== expected) {
      throw new Error(`frozen chat roles must alternate user/assistant (index ${i})`);
    }
    if (!message.content.trim()) {
      throw new Error(`frozen chat content is empty at index ${i}`);
    }
    parts.push(`${message.role === "user" ? "<|user|>" : "<|assistant|>"} ${message.content}`);
  }
  // Do not append a literal space after the terminal assistant marker. In
  // training, the following content token owns its leading space (for
  // example, ` Hello`). A generation-only trailing space becomes a separate
  // token that never occupies this boundary in the SFT corpus.
  return `${parts.join(" ")} <|assistant|>`;
}

/** Fraction of generated token 4-grams that repeat an earlier generated 4-gram. */
export function fourGramRepeatRate(tokens: readonly number[]): number {
  const total = Math.max(0, tokens.length - 3);
  if (total === 0) return 0;
  const seen = new Set<string>();
  let repeated = 0;
  for (let i = 0; i < total; i++) {
    const gram = `${tokens[i]},${tokens[i + 1]},${tokens[i + 2]},${tokens[i + 3]}`;
    if (seen.has(gram)) repeated++;
    else seen.add(gram);
  }
  return repeated / total;
}

/** Generated-content token indices whose token completes a 4-gram already
 * observed earlier in the same content trajectory. */
export function repeatedFourGramCompletionPositions(tokens: readonly number[]): number[] {
  const positions: number[] = [];
  const seen = new Set<string>();
  for (let start = 0; start + 3 < tokens.length; start++) {
    const gram = `${tokens[start]},${tokens[start + 1]},${tokens[start + 2]},${tokens[start + 3]}`;
    if (seen.has(gram)) positions.push(start + 3);
    else seen.add(gram);
  }
  return positions;
}

export function normalizedAnswerTokens(text: string): string[] {
  return [...text.toLocaleLowerCase("en-US").matchAll(/[\p{L}\p{N}]+/gu)].map((match) => match[0]);
}

export function normalizedAnswer(text: string): string {
  return normalizedAnswerTokens(text).join(" ");
}

export function answerTokenF1(generated: string, expected: string): number {
  const prediction = normalizedAnswerTokens(generated);
  const truth = normalizedAnswerTokens(expected);
  if (prediction.length === 0 || truth.length === 0) return 0;
  const available = new Map<string, number>();
  for (const token of truth) available.set(token, (available.get(token) ?? 0) + 1);
  let common = 0;
  for (const token of prediction) {
    const count = available.get(token) ?? 0;
    if (count > 0) {
      common++;
      available.set(token, count - 1);
    }
  }
  const precision = common / prediction.length;
  const recall = common / truth.length;
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

export function answerIsContained(generated: string, expected: string): boolean {
  const prediction = normalizedAnswer(generated);
  const truth = normalizedAnswer(expected);
  return truth.length > 0 && prediction.includes(truth);
}
