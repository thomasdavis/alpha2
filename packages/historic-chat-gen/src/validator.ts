/**
 * Structural JSON validation for generated conversations.
 */
import type { ConversationTurn } from "./types.js";

export interface ValidationResult {
  valid: boolean;
  error?: string;
  turns?: ConversationTurn[];
}

/**
 * Validate a parsed conversation object.
 * Expected shape: { turns: [{ speaker, text }, ...] }
 */
export function validateConversation(
  parsed: unknown,
  expectedFigureA: string,
  expectedFigureB: string,
  expectedTurnCount: number
): ValidationResult {
  if (!parsed || typeof parsed !== "object") {
    return { valid: false, error: "Response is not an object" };
  }

  const obj = parsed as Record<string, unknown>;

  if (!Array.isArray(obj["turns"])) {
    return { valid: false, error: "Missing or invalid 'turns' array" };
  }

  const turns = obj["turns"] as unknown[];

  if (turns.length < 4) {
    return { valid: false, error: `Too few turns: ${turns.length} (minimum 4)` };
  }

  if (turns.length > 30) {
    return { valid: false, error: `Too many turns: ${turns.length} (maximum 30)` };
  }

  const validSpeakers = new Set([expectedFigureA, expectedFigureB]);
  const validatedTurns: ConversationTurn[] = [];

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (!turn || typeof turn !== "object") {
      return { valid: false, error: `Turn ${i} is not an object` };
    }

    const t = turn as Record<string, unknown>;

    if (typeof t["speaker"] !== "string" || !t["speaker"]) {
      return { valid: false, error: `Turn ${i}: missing or invalid 'speaker'` };
    }

    if (typeof t["text"] !== "string" || !t["text"]) {
      return { valid: false, error: `Turn ${i}: missing or invalid 'text'` };
    }

    if (!validSpeakers.has(t["speaker"])) {
      return { valid: false, error: `Turn ${i}: unknown speaker '${t["speaker"]}' (expected '${expectedFigureA}' or '${expectedFigureB}')` };
    }

    // Check alternating speakers
    if (i > 0) {
      const prevSpeaker = validatedTurns[i - 1]!.speaker;
      if (t["speaker"] === prevSpeaker) {
        return { valid: false, error: `Turn ${i}: same speaker '${t["speaker"]}' as previous turn (must alternate)` };
      }
    }

    if ((t["text"] as string).length < 10) {
      return { valid: false, error: `Turn ${i}: text too short (${(t["text"] as string).length} chars)` };
    }

    validatedTurns.push({ speaker: t["speaker"] as string, text: t["text"] as string });
  }

  return { valid: true, turns: validatedTurns };
}
