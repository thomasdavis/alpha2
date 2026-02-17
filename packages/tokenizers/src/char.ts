/**
 * Character-level tokenizer.
 *
 * Builds a vocabulary from the unique characters in the input text (sorted),
 * then maps each character to its index. This is the simplest possible
 * tokenizer -- identical to the approach used in microgpt.py.
 */
import { Effect } from "effect";
import {
  TokenizerError,
  type Tokenizer,
  type TokenizerArtifacts,
} from "@alpha/core";

export class CharTokenizer implements Tokenizer {
  readonly name = "char";

  /** Sorted character vocabulary. */
  private _vocab: string[] = [];

  /** char -> token id */
  private _stoi = new Map<string, number>();

  /** token id -> char */
  private _itos = new Map<number, string>();

  // ── Public interface ─────────────────────────────────────────────────────

  /** Number of tokens in the current vocabulary. */
  get vocabSize(): number {
    return this._vocab.length;
  }

  /**
   * Build the vocabulary from raw input text.
   *
   * Extracts every unique character, sorts them, and assigns indices.
   * Returns the resulting artifacts so they can be persisted.
   */
  build(input: string): Effect.Effect<TokenizerArtifacts, TokenizerError> {
    return Effect.try({
      try: () => {
        if (input.length === 0) {
          throw new Error("Cannot build char tokenizer from empty input");
        }

        const chars = [...new Set(input)].sort();
        this._setVocab(chars);

        return {
          type: "char",
          vocabSize: chars.length,
          vocab: chars,
        } satisfies TokenizerArtifacts;
      },
      catch: (cause) => new TokenizerError({ message: String(cause), cause }),
    });
  }

  /**
   * Encode a string into a token id array.
   *
   * Unknown characters are silently skipped -- this keeps behaviour
   * predictable when the text contains characters not seen during build().
   */
  encode(text: string): Int32Array {
    const ids: number[] = [];
    for (const ch of text) {
      const id = this._stoi.get(ch);
      if (id !== undefined) {
        ids.push(id);
      }
    }
    return new Int32Array(ids);
  }

  /**
   * Decode token ids back into a string.
   *
   * Unknown ids map to the empty string (ignored).
   */
  decode(tokens: ArrayLike<number>): string {
    const parts: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const ch = this._itos.get(tokens[i]);
      if (ch !== undefined) {
        parts.push(ch);
      }
    }
    return parts.join("");
  }

  // ── Restore from persisted artifacts ─────────────────────────────────────

  /**
   * Re-initialise the tokenizer from previously saved artifacts.
   *
   * Useful after loading a JSON file via `persist.ts`.
   */
  loadArtifacts(artifacts: TokenizerArtifacts): void {
    this._setVocab([...artifacts.vocab]);
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  /** Set the vocabulary and rebuild lookup tables. */
  private _setVocab(chars: string[]): void {
    this._vocab = chars;
    this._stoi.clear();
    this._itos.clear();
    for (let i = 0; i < chars.length; i++) {
      this._stoi.set(chars[i], i);
      this._itos.set(i, chars[i]);
    }
  }
}
