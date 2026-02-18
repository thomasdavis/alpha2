/**
 * Word-level tokenizer.
 *
 * Designed for data where tokens are discrete space-separated symbols
 * (chords, commands, tags, etc.). Each unique word becomes one token.
 * Line breaks are preserved as an explicit `\n` token.
 */
import { Effect } from "effect";
import {
  TokenizerError,
  type Tokenizer,
  type TokenizerArtifacts,
} from "@alpha/core";

export class WordTokenizer implements Tokenizer {
  readonly name = "word";

  /** Sorted word vocabulary. */
  private _vocab: string[] = [];

  /** word -> token id */
  private _stoi = new Map<string, number>();

  /** token id -> word */
  private _itos = new Map<number, string>();

  // ── Public interface ─────────────────────────────────────────────────────

  /** Number of tokens in the current vocabulary. */
  get vocabSize(): number {
    return this._vocab.length;
  }

  /**
   * Build the vocabulary from raw input text.
   *
   * Splits on whitespace, injects `\n` as a token for line boundaries,
   * and builds vocab from unique words sorted alphabetically.
   */
  build(input: string): Effect.Effect<TokenizerArtifacts, TokenizerError> {
    return Effect.try({
      try: () => {
        if (input.length === 0) {
          throw new Error("Cannot build word tokenizer from empty input");
        }

        const words = new Set<string>();
        const lines = input.split("\n");

        for (const line of lines) {
          const tokens = line.trim().split(/\s+/).filter((w) => w.length > 0);
          for (const token of tokens) {
            words.add(token);
          }
        }

        // Always include \n for line boundaries
        words.add("\n");

        const sorted = [...words].sort();
        this._setVocab(sorted);

        return {
          type: "word",
          vocabSize: sorted.length,
          vocab: sorted,
        } satisfies TokenizerArtifacts;
      },
      catch: (cause) => new TokenizerError({ message: String(cause), cause }),
    });
  }

  /**
   * Encode text into token ids.
   *
   * Splits on spaces, maps words to IDs. Newline characters are mapped
   * to the `\n` token ID. Unknown words are silently skipped.
   */
  encode(text: string): Int32Array {
    const ids: number[] = [];
    const lines = text.split("\n");

    for (let li = 0; li < lines.length; li++) {
      const tokens = lines[li].trim().split(/\s+/).filter((w) => w.length > 0);
      for (const token of tokens) {
        const id = this._stoi.get(token);
        if (id !== undefined) {
          ids.push(id);
        }
      }
      // Add newline token between lines (not after the last one)
      if (li < lines.length - 1) {
        const nlId = this._stoi.get("\n");
        if (nlId !== undefined) {
          ids.push(nlId);
        }
      }
    }

    return new Int32Array(ids);
  }

  /**
   * Decode token ids back into text.
   *
   * Joins token strings with spaces, replacing ` \n ` with actual newlines.
   */
  decode(tokens: ArrayLike<number>): string {
    const parts: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const word = this._itos.get(tokens[i]);
      if (word !== undefined) {
        parts.push(word);
      }
    }
    // Join with spaces, then fix newline tokens
    return parts.join(" ").replace(/ \n /g, "\n").replace(/ \n/g, "\n").replace(/\n /g, "\n");
  }

  // ── Restore from persisted artifacts ─────────────────────────────────────

  /**
   * Re-initialise the tokenizer from previously saved artifacts.
   */
  loadArtifacts(artifacts: TokenizerArtifacts): void {
    this._setVocab([...artifacts.vocab]);
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  /** Set the vocabulary and rebuild lookup tables. */
  private _setVocab(words: string[]): void {
    this._vocab = words;
    this._stoi.clear();
    this._itos.clear();
    for (let i = 0; i < words.length; i++) {
      this._stoi.set(words[i], i);
      this._itos.set(i, words[i]);
    }
  }
}
