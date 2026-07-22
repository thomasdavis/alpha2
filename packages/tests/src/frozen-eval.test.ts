import { describe, expect, it } from "vitest";
import {
  answerIsContained,
  answerTokenF1,
  formatFrozenChatPrompt,
  fourGramRepeatRate,
  normalizedAnswer,
} from "@alpha/train";

describe("frozen eval helpers", () => {
  it("formats multi-turn history exactly like the SFT corpus", () => {
    expect(formatFrozenChatPrompt([
      { role: "user", content: "First question" },
      { role: "assistant", content: "First answer" },
      { role: "user", content: "Follow-up" },
    ])).toBe(
      "<|user|> First question <|assistant|> First answer <|user|> Follow-up <|assistant|> ",
    );
  });

  it("rejects malformed role histories", () => {
    expect(() => formatFrozenChatPrompt([])).toThrow(/non-empty/);
    expect(() => formatFrozenChatPrompt([
      { role: "user", content: "Question" },
      { role: "user", content: "Still user" },
    ])).toThrow(/alternate/);
  });

  it("measures repeated token 4-grams", () => {
    expect(fourGramRepeatRate([1, 2, 3])).toBe(0);
    expect(fourGramRepeatRate([1, 2, 3, 4])).toBe(0);
    // Six windows, of which the final two repeat the first two.
    expect(fourGramRepeatRate([1, 2, 3, 4, 1, 2, 3, 4, 1])).toBeCloseTo(2 / 6);
  });

  it("normalizes and scores closed-book answers", () => {
    expect(normalizedAnswer("  Lomas—Juniors! ")).toBe("lomas juniors");
    expect(answerIsContained("The answer is Lomas Juniors.", "Lomas Juniors")).toBe(true);
    expect(answerIsContained("Lomas", "Lomas Juniors")).toBe(false);
    expect(answerTokenF1("red green", "green blue")).toBeCloseTo(0.5);
  });
});
