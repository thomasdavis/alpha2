/**
 * @alpha/tokenizers -- tokenizer implementations for the alpha system.
 *
 * Provides a character-level tokenizer, a char-seeded byte-pair encoding
 * tokenizer, a byte-level BPE tokenizer (GPT-2/SmolLM2 family), persistence
 * helpers, an HF `tokenizer.json` exporter, and a pre-populated registry so the
 * rest of the system can look up tokenizers by name.
 */
import { Registry, type Tokenizer, type TokenizerArtifacts } from "@alpha/core";
import { CharTokenizer } from "./char.js";
import { BpeTokenizer } from "./bpe.js";
import { ByteBpeTokenizer } from "./byte_bpe.js";
import { WordTokenizer } from "./word.js";

const CHAT_SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end_of_text|>"] as const;

// ── Re-exports ────────────────────────────────────────────────────────────
export { CharTokenizer } from "./char.js";
export { BpeTokenizer } from "./bpe.js";
export {
  ByteBpeTokenizer,
  bytesToUnicode,
  gpt2SplitRegex,
  GPT2_SPLIT_PATTERN,
  buildByteTrainingSample,
  CHAT_SPECIAL_TOKENS as BYTE_BPE_CHAT_SPECIAL_TOKENS,
} from "./byte_bpe.js";
export { WordTokenizer } from "./word.js";
export { saveArtifacts, loadArtifacts } from "./persist.js";
export { exportHfTokenizer, writeHfTokenizer, buildChatTemplate } from "./export_hf.js";
export type { HfTokenizerBundle } from "./export_hf.js";

// ── Tokenizer registry ────────────────────────────────────────────────────

/**
 * Global tokenizer registry.
 *
 * Pre-registered implementations:
 * - `"char"`         -- character-level tokenizer
 * - `"bpe"`          -- char-seeded byte-pair encoding tokenizer
 * - `"byte_bpe"`     -- byte-level BPE (256-byte base, chat specials reserved)
 * - `"bpe-byte-12k"` -- byte-level BPE, vocab 12,288 (flagship; chat specials)
 * - `"bpe-byte-4k"`  -- byte-level BPE, vocab 4,096 (pilots; chat specials)
 * - `"word"`         -- word-level tokenizer for discrete symbol domains
 *
 * Usage:
 * ```ts
 * const tok = tokenizerRegistry.get("bpe-byte-12k");
 * ```
 */
export const tokenizerRegistry = new Registry<Tokenizer>("tokenizer");

tokenizerRegistry.register("char", () => new CharTokenizer());
tokenizerRegistry.register("bpe", () => new BpeTokenizer());
tokenizerRegistry.register("bpe-4k", () => new BpeTokenizer(4000));
tokenizerRegistry.register("bpe-8k", () => new BpeTokenizer(8000));
tokenizerRegistry.register("bpe-16k", () => new BpeTokenizer(16000));
tokenizerRegistry.register("bpe-32k", () => new BpeTokenizer(32000));
tokenizerRegistry.register("bpe-64k", () => new BpeTokenizer(64000));
tokenizerRegistry.register("bpe-chat", () => new BpeTokenizer(2000, { reservedTokens: CHAT_SPECIAL_TOKENS }));
tokenizerRegistry.register("bpe-chat-4k", () => new BpeTokenizer(4000, { reservedTokens: CHAT_SPECIAL_TOKENS }));
// Byte-level BPE (GPT-2/SmolLM2 family) — the flagship tokenizer per GOAL Stage 3.
tokenizerRegistry.register("byte_bpe", () => new ByteBpeTokenizer(12288, { specialTokens: CHAT_SPECIAL_TOKENS }));
tokenizerRegistry.register("bpe-byte-12k", () => new ByteBpeTokenizer(12288, { specialTokens: CHAT_SPECIAL_TOKENS }));
tokenizerRegistry.register("bpe-byte-4k", () => new ByteBpeTokenizer(4096, { specialTokens: CHAT_SPECIAL_TOKENS }));
tokenizerRegistry.register("word", () => new WordTokenizer());

/**
 * Reconstruct a tokenizer instance from persisted artifacts, dispatching on
 * `artifacts.type`. Single source of truth for the CLI's train/sample/eval
 * commands so a new tokenizer type is wired in exactly one place.
 */
export function tokenizerFromArtifacts(artifacts: TokenizerArtifacts): Tokenizer {
  switch (artifacts.type) {
    case "byte_bpe": {
      const tok = new ByteBpeTokenizer();
      tok.loadArtifacts(artifacts);
      return tok;
    }
    case "bpe": {
      const tok = new BpeTokenizer();
      tok.loadArtifacts(artifacts);
      return tok;
    }
    case "word": {
      const tok = new WordTokenizer();
      tok.loadArtifacts(artifacts);
      return tok;
    }
    default: {
      const tok = new CharTokenizer();
      tok.loadArtifacts(artifacts);
      return tok;
    }
  }
}
