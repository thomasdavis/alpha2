/**
 * @alpha/tokenizers -- tokenizer implementations for the alpha system.
 *
 * Provides a character-level tokenizer, a byte-pair encoding tokenizer,
 * persistence helpers, and a pre-populated registry so the rest of the
 * system can look up tokenizers by name.
 */
import { Registry, type Tokenizer } from "@alpha/core";
import { CharTokenizer } from "./char.js";
import { BpeTokenizer } from "./bpe.js";
import { WordTokenizer } from "./word.js";

// ── Re-exports ────────────────────────────────────────────────────────────
export { CharTokenizer } from "./char.js";
export { BpeTokenizer } from "./bpe.js";
export { WordTokenizer } from "./word.js";
export { saveArtifacts, loadArtifacts } from "./persist.js";

// ── Tokenizer registry ────────────────────────────────────────────────────

/**
 * Global tokenizer registry.
 *
 * Pre-registered implementations:
 * - `"char"` -- character-level tokenizer
 * - `"bpe"`  -- byte-pair encoding tokenizer (default vocab size 2000)
 * - `"word"` -- word-level tokenizer for discrete symbol domains
 *
 * Usage:
 * ```ts
 * const tok = tokenizerRegistry.get("bpe");
 * ```
 */
export const tokenizerRegistry = new Registry<Tokenizer>("tokenizer");

tokenizerRegistry.register("char", () => new CharTokenizer());
tokenizerRegistry.register("bpe", () => new BpeTokenizer());
tokenizerRegistry.register("bpe-4k", () => new BpeTokenizer(4000));
tokenizerRegistry.register("word", () => new WordTokenizer());
