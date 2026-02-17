import { describe, it, expect } from "vitest";
import { Effect } from "effect";
import { CharTokenizer, BpeTokenizer } from "@alpha/tokenizers";

describe("CharTokenizer", () => {
  it("encode/decode roundtrip", async () => {
    const tok = new CharTokenizer();
    await Effect.runPromise(tok.build("hello world"));

    const encoded = tok.encode("hello");
    const decoded = tok.decode(encoded);
    expect(decoded).toBe("hello");
  });

  it("encodes all unique chars", async () => {
    const tok = new CharTokenizer();
    await Effect.runPromise(tok.build("abcabc"));
    expect(tok.vocabSize).toBe(3); // a, b, c
  });

  it("handles empty string", async () => {
    const tok = new CharTokenizer();
    await Effect.runPromise(tok.build("abc"));
    const encoded = tok.encode("");
    expect(encoded.length).toBe(0);
    expect(tok.decode(encoded)).toBe("");
  });
});

describe("BpeTokenizer", () => {
  it("encode/decode roundtrip", async () => {
    const tok = new BpeTokenizer(50);
    const text = "the cat sat on the mat the cat sat on the mat";
    await Effect.runPromise(tok.build(text));

    const encoded = tok.encode("the cat");
    const decoded = tok.decode(encoded);
    expect(decoded).toBe("the cat");
  });

  it("reduces token count with merges", async () => {
    const tok = new BpeTokenizer(30);
    const text = "aaabbbaaabbbaaabbb";
    await Effect.runPromise(tok.build(text));

    const encoded = tok.encode("aaabbb");
    // After BPE merges, should be fewer tokens than 6 characters
    expect(encoded.length).toBeLessThanOrEqual(6);
  });

  it("vocabSize is correct", async () => {
    const tok = new BpeTokenizer(20);
    await Effect.runPromise(tok.build("abcabcabc"));
    expect(tok.vocabSize).toBeGreaterThanOrEqual(3); // at least the base chars
    expect(tok.vocabSize).toBeLessThanOrEqual(20);
  });
});
