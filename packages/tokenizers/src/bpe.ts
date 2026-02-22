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

  /** Cached merge rank lookup for fast encoding. */
  private _mergeRank: Map<number, { rank: number; newId: number }> | null = null;

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
   * Uses a min-heap keyed by merge rank over a doubly-linked list of tokens.
   * Complexity: O(N log N) amortized instead of O(M × N) where M = merge count.
   * Each token position is visited a constant number of times.
   */
  encode(text: string): Int32Array {
    // Start with character-level ids.
    const charIds: number[] = [];
    for (const ch of text) {
      const id = this._stoi.get(ch);
      if (id !== undefined) {
        charIds.push(id);
      }
    }

    const n = charIds.length;
    if (this._merges.length === 0 || n <= 1) {
      return new Int32Array(charIds);
    }

    // For small vocabs (< 1000 merges), use the simpler O(M×N) approach
    // which avoids typed-array allocation overhead.
    if (this._merges.length < 1000) {
      return this._encodeSimple(charIds);
    }

    // Build merge rank lookup (cached across encode calls)
    if (!this._mergeRank) {
      this._mergeRank = new Map();
      for (let i = 0; i < this._merges.length; i++) {
        const m = this._merges[i];
        this._mergeRank.set(m.left * 65536 + m.right, { rank: i, newId: m.newId });
      }
    }
    const mr = this._mergeRank;
    const pk = (a: number, b: number) => a * 65536 + b;

    // ── Doubly-linked list of tokens ──
    const nodeId = new Int32Array(n);
    const nodeNext = new Int32Array(n);
    const nodePrev = new Int32Array(n);
    const deleted = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      nodeId[i] = charIds[i];
      nodePrev[i] = i - 1;
      nodeNext[i] = i + 1;
    }
    nodeNext[n - 1] = -1;

    // ── Binary min-heap keyed by merge rank ──
    // Each entry: (rank, position) where position is the LEFT node of the pair.
    // Stale entries are detected and skipped at pop time.
    // Capacity: initial pairs (≤ n-1) + at most 2 new pairs per merge, merges < n → max 3n.
    let heapSize = 0;
    const heapCap = Math.min(n * 3, n + 1_000_000); // cap growth for huge texts
    const heapRank = new Int32Array(heapCap);
    const heapPos = new Int32Array(heapCap);

    const heapPush = (rank: number, pos: number): void => {
      if (heapSize >= heapCap) return; // safety: skip if full (stale entries will be popped)
      let i = heapSize++;
      heapRank[i] = rank;
      heapPos[i] = pos;
      // Bubble up
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heapRank[p] <= heapRank[i]) break;
        // Swap rank
        const tr = heapRank[p]; heapRank[p] = heapRank[i]; heapRank[i] = tr;
        // Swap pos
        const tp = heapPos[p]; heapPos[p] = heapPos[i]; heapPos[i] = tp;
        i = p;
      }
    };

    const heapPop = (): [number, number] => {
      const rank = heapRank[0];
      const pos = heapPos[0];
      heapSize--;
      if (heapSize > 0) {
        heapRank[0] = heapRank[heapSize];
        heapPos[0] = heapPos[heapSize];
        // Sink down
        let i = 0;
        while (true) {
          let s = i;
          const l = 2 * i + 1;
          const r = 2 * i + 2;
          if (l < heapSize && heapRank[l] < heapRank[s]) s = l;
          if (r < heapSize && heapRank[r] < heapRank[s]) s = r;
          if (s === i) break;
          const tr = heapRank[s]; heapRank[s] = heapRank[i]; heapRank[i] = tr;
          const tp = heapPos[s]; heapPos[s] = heapPos[i]; heapPos[i] = tp;
          i = s;
        }
      }
      return [rank, pos];
    };

    // Initialize heap with all mergeable adjacent pairs
    for (let i = 0; i < n - 1; i++) {
      const info = mr.get(pk(charIds[i], charIds[i + 1]));
      if (info) heapPush(info.rank, i);
    }

    // ── Process merges in rank order ──
    while (heapSize > 0) {
      const [rank, pos] = heapPop();

      // Skip stale entries: node deleted, no next, or pair changed
      if (deleted[pos]) continue;
      const nxt = nodeNext[pos];
      if (nxt === -1 || deleted[nxt]) continue;
      const info = mr.get(pk(nodeId[pos], nodeId[nxt]));
      if (!info || info.rank !== rank) continue;

      // Apply merge: replace pos's token, remove nxt from linked list
      nodeId[pos] = info.newId;
      deleted[nxt] = 1;
      nodeNext[pos] = nodeNext[nxt];
      if (nodeNext[nxt] !== -1) nodePrev[nodeNext[nxt]] = pos;

      // New pair with left neighbor → push to heap
      const prv = nodePrev[pos];
      if (prv !== -1 && !deleted[prv]) {
        const pInfo = mr.get(pk(nodeId[prv], nodeId[pos]));
        if (pInfo) heapPush(pInfo.rank, prv);
      }

      // New pair with right neighbor → push to heap
      const nxtNext = nodeNext[pos];
      if (nxtNext !== -1 && !deleted[nxtNext]) {
        const pInfo = mr.get(pk(nodeId[pos], nodeId[nxtNext]));
        if (pInfo) heapPush(pInfo.rank, pos);
      }
    }

    // ── Collect surviving tokens ──
    const result: number[] = [];
    let cur = 0;
    while (cur < n && deleted[cur]) cur++;
    while (cur !== -1 && cur < n) {
      result.push(nodeId[cur]);
      cur = nodeNext[cur];
    }

    return new Int32Array(result);
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
    this._mergeRank = null; // invalidate cache

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
   * Simple O(M×N) encode for small vocabs (< 1000 merges).
   * Avoids typed-array allocation overhead of the heap-based encoder.
   */
  private _encodeSimple(ids: number[]): Int32Array {
    let current = ids;
    for (const merge of this._merges) {
      current = this._applyMerge(current, merge.left, merge.right, merge.newId);
      if (current.length <= 1) break;
    }
    return new Int32Array(current);
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
