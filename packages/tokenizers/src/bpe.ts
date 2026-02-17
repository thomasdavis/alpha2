/**
 * Byte-pair encoding tokenizer.
 *
 * Starts from a character-level vocabulary and iteratively merges the most
 * frequent adjacent pair until the target vocab size is reached. The learned
 * merges are applied greedily at encode-time in the same order they were
 * discovered during training.
 */
import { Effect } from "effect";
import {
  TokenizerError,
  type Tokenizer,
  type TokenizerArtifacts,
} from "@alpha/core";

/** A single learned merge: (left id, right id) -> new id. */
interface Merge {
  readonly left: number;
  readonly right: number;
  readonly newId: number;
}

export class BpeTokenizer implements Tokenizer {
  readonly name = "bpe";

  /** Target vocabulary size (base chars + merges). */
  private readonly _targetVocabSize: number;

  /** id -> string token */
  private _vocab: string[] = [];

  /** string token -> id */
  private _stoi = new Map<string, number>();

  /** Ordered list of learned merges. */
  private _merges: Merge[] = [];

  constructor(vocabSize = 2000) {
    this._targetVocabSize = vocabSize;
  }

  // ── Public interface ─────────────────────────────────────────────────────

  get vocabSize(): number {
    return this._vocab.length;
  }

  /**
   * Train BPE on raw text.
   *
   * 1. Initialise vocab with sorted unique characters.
   * 2. Tokenise the entire input at character level.
   * 3. Repeatedly find the most frequent adjacent pair, merge it into a new
   *    token, and record the merge -- until we hit the target vocab size.
   */
  build(input: string): Effect.Effect<TokenizerArtifacts, TokenizerError> {
    return Effect.try({
      try: () => {
        if (input.length === 0) {
          throw new Error("Cannot build BPE tokenizer from empty input");
        }

        // Base vocabulary: sorted unique characters from full input.
        const baseChars = [...new Set(input)].sort();
        this._vocab = [...baseChars];
        this._rebuildStoi();
        this._merges = [];

        // Cap training corpus for efficiency -- pair statistics stabilise
        // well before 500K chars, so there's no quality loss.
        const maxTrainChars = 500_000;
        const trainText = input.length > maxTrainChars ? input.slice(0, maxTrainChars) : input;

        // Working corpus as Int32Array for cache-friendly access.
        let len = trainText.length;
        let corpus = new Int32Array(len);
        for (let i = 0; i < len; i++) {
          corpus[i] = this._stoi.get(trainText[i])!;
        }

        // Encode pair as single number: left * 65536 + right
        const pairKey = (a: number, b: number) => a * 65536 + b;

        // Initial pair count.
        const pairCounts = new Map<number, number>();
        for (let i = 0; i < len - 1; i++) {
          const key = pairKey(corpus[i], corpus[i + 1]);
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
        }

        const numMerges = this._targetVocabSize - baseChars.length;
        for (let m = 0; m < numMerges; m++) {
          if (pairCounts.size === 0) break;

          // Find the most frequent pair.
          let bestKey = 0;
          let bestCount = 0;
          for (const [key, count] of pairCounts) {
            if (count > bestCount) {
              bestCount = count;
              bestKey = key;
            }
          }

          if (bestCount < 2) break;

          const leftId = (bestKey / 65536) | 0;
          const rightId = bestKey % 65536;
          const newToken = this._vocab[leftId] + this._vocab[rightId];
          const newId = this._vocab.length;

          this._vocab.push(newToken);
          this._stoi.set(newToken, newId);
          this._merges.push({ left: leftId, right: rightId, newId });

          // Apply merge in-place and update pair counts incrementally.
          const mergeKey = pairKey(leftId, rightId);
          pairCounts.delete(mergeKey);

          let write = 0;
          for (let i = 0; i < len; i++) {
            if (
              i < len - 1 &&
              corpus[i] === leftId &&
              corpus[i + 1] === rightId
            ) {
              // Before merge: ..., prev, LEFT, RIGHT, next, ...
              // Decrement old pairs around this site.
              if (write > 0) {
                const prev = corpus[write - 1];
                const oldPairL = pairKey(prev, leftId);
                const c = pairCounts.get(oldPairL);
                if (c !== undefined) {
                  if (c <= 1) pairCounts.delete(oldPairL);
                  else pairCounts.set(oldPairL, c - 1);
                }
              }
              const next = i + 2 < len ? corpus[i + 2] : -1;
              if (next >= 0) {
                const oldPairR = pairKey(rightId, next);
                const c = pairCounts.get(oldPairR);
                if (c !== undefined) {
                  if (c <= 1) pairCounts.delete(oldPairR);
                  else pairCounts.set(oldPairR, c - 1);
                }
              }

              // Write merged token.
              corpus[write] = newId;

              // Add new pairs around merged token.
              if (write > 0) {
                const prev = corpus[write - 1];
                const np = pairKey(prev, newId);
                pairCounts.set(np, (pairCounts.get(np) ?? 0) + 1);
              }
              if (next >= 0) {
                const np = pairKey(newId, next);
                pairCounts.set(np, (pairCounts.get(np) ?? 0) + 1);
              }

              write++;
              i++; // skip right token
            } else {
              corpus[write] = corpus[i];
              write++;
            }
          }
          len = write;

          // Clean up zero-count entries periodically.
          if (m % 100 === 0) {
            for (const [k, v] of pairCounts) {
              if (v <= 0) pairCounts.delete(k);
            }
          }
        }

        return {
          type: "bpe",
          vocabSize: this._vocab.length,
          vocab: this._vocab,
          merges: this._merges.map((m) => [m.left, m.right] as [number, number]),
        } satisfies TokenizerArtifacts;
      },
      catch: (cause) => new TokenizerError({ message: String(cause), cause }),
    });
  }

  /**
   * Encode text by greedily applying learned merges.
   *
   * Start with the character-level token ids, then replay every merge in
   * training order. This guarantees deterministic encoding.
   */
  encode(text: string): Int32Array {
    // Start with character-level ids.
    let ids: number[] = [];
    for (const ch of text) {
      const id = this._stoi.get(ch);
      if (id !== undefined) {
        ids.push(id);
      }
    }

    // Replay merges in order.
    for (const merge of this._merges) {
      ids = this._applyMerge(ids, merge.left, merge.right, merge.newId);
    }

    return new Int32Array(ids);
  }

  /**
   * Decode token ids back into a string.
   */
  decode(tokens: ArrayLike<number>): string {
    const parts: string[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const tok = this._vocab[tokens[i]];
      if (tok !== undefined) {
        parts.push(tok);
      }
    }
    return parts.join("");
  }

  // ── Restore from persisted artifacts ─────────────────────────────────────

  /**
   * Re-initialise from previously saved artifacts.
   */
  loadArtifacts(artifacts: TokenizerArtifacts): void {
    this._vocab = [...artifacts.vocab];
    this._rebuildStoi();

    if (artifacts.merges) {
      const baseSize = this._vocab.length - artifacts.merges.length;
      this._merges = artifacts.merges.map(([left, right], i) => ({
        left,
        right,
        newId: baseSize + i,
      }));
    } else {
      this._merges = [];
    }
  }

  // ── Internal helpers ─────────────────────────────────────────────────────

  /** Rebuild the string-to-id lookup from the current vocab. */
  private _rebuildStoi(): void {
    this._stoi.clear();
    for (let i = 0; i < this._vocab.length; i++) {
      this._stoi.set(this._vocab[i], i);
    }
  }

  /**
   * Scan `ids` and replace every adjacent (left, right) with `newId`.
   *
   * Returns a new array -- the corpus is never mutated in place so that
   * earlier indices remain valid during the scan.
   */
  private _applyMerge(
    ids: number[],
    left: number,
    right: number,
    newId: number,
  ): number[] {
    const out: number[] = [];
    let i = 0;
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === left && ids[i + 1] === right) {
        out.push(newId);
        i += 2;
      } else {
        out.push(ids[i]);
        i += 1;
      }
    }
    return out;
  }
}
