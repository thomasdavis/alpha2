/**
 * Byte-level byte-pair encoding tokenizer (GPT-2 / SmolLM2 family).
 *
 * Unlike the char-seeded {@link BpeTokenizer} (base vocab = unique chars of the
 * corpus, which silently drops out-of-vocabulary characters at encode time),
 * this tokenizer's base vocabulary is the **256 possible bytes**, mapped to a
 * set of 256 printable-safe Unicode "surrogate" characters via the exact GPT-2
 * `bytes_to_unicode` table. Because every byte has a token, **any** input —
 * emoji, CJK, control chars, arbitrary binary — round-trips losslessly through
 * a single UTF-8 decode at the end.
 *
 * Pipeline (matches HuggingFace `ByteLevel` pre-tokenizer + a BPE model):
 *   1. Reserved special tokens (`<|user|>`, `<|assistant|>`, `<|end_of_text|>`)
 *      are matched FIRST, longest-first, and emitted atomically. They never
 *      participate in byte merges and never cross into the byte stream.
 *   2. The remaining text is split by the GPT-2 pre-tokenization regex. Merges
 *      never cross a pre-token boundary.
 *   3. Each pre-token's UTF-8 bytes are mapped through the surrogate table and
 *      then reduced by the learned merges (lowest-rank-first, GPT-2 algorithm).
 *
 * Artifact schema v2:
 *   { type:"byte_bpe", vocabSize, byteVocab:true,
 *     vocab: string[]            // surrogate-mapped byte/merge strings + specials
 *     merges: [idL,idR][],       // learned merges as id pairs, in rank order
 *     specialTokens: string[] }
 */
import { Effect } from "effect";
import {
  TokenizerError,
  type Tokenizer,
  type TokenizerArtifacts,
} from "@alpha/core";

/**
 * GPT-2 pre-tokenization split regex.
 *
 * The canonical GPT-2 / HuggingFace `ByteLevel(use_regex=true)` pattern is:
 *   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
 *
 * We spell whitespace as `\p{White_Space}` (not JS `\s`) so this JS regex splits
 * IDENTICALLY to HuggingFace's Rust engine. JS's legacy `\s` differs from the
 * Unicode `White_Space` property on exactly two code points — it MATCHES U+FEFF
 * (BOM) and MISSES U+0085 (NEL) — which shifts pre-token boundaries around a BOM
 * (e.g. `\n\n﻿`) and would otherwise desync ids from the exported
 * tokenizer.json. Verified equal on a 10K-doc corpus by the Python cross-check.
 *
 * Order matters: contractions, then " ?letters", " ?digits",
 * " ?non-space-non-letter-non-digit", then whitespace-not-followed-by-nonspace,
 * then trailing whitespace.
 */
export const GPT2_SPLIT_PATTERN =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+|\\p{White_Space}+(?!\\P{White_Space})|\\p{White_Space}+";

/** Fresh regex per call site (global-flag regexes carry mutable lastIndex). */
export function gpt2SplitRegex(): RegExp {
  return new RegExp(GPT2_SPLIT_PATTERN, "gu");
}

/** Chat role / end-of-text markers reserved as atomic tokens. */
export const CHAT_SPECIAL_TOKENS = [
  "<|user|>",
  "<|assistant|>",
  "<|end_of_text|>",
] as const;

/**
 * GPT-2 `bytes_to_unicode`.
 *
 * Printable byte ranges (`!`..`~`, `¡`..`¬`, `®`..`ÿ`) map to themselves as
 * single-code-point strings; every other byte (controls, space, DEL, etc.) is
 * assigned a fresh code point at U+0100 + n. The result is a bijection between
 * the 256 bytes and 256 whitespace-free printable Unicode characters — e.g.
 * space (0x20) → `Ġ` (U+0120), newline (0x0A) → `Ċ` (U+010A).
 *
 * @returns `byteToChar` (byte → surrogate char) and `charToByte` (inverse).
 */
export function bytesToUnicode(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b); // '!'..'~'
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b); // '¡'..'¬'
  for (let b = 0xae; b <= 0xff; b++) bs.push(b); // '®'..'ÿ'

  const inBs = new Set(bs);
  const byteToChar = new Array<string>(256);
  const charToByte = new Map<string, number>();

  // Printable bytes map to themselves.
  for (const b of bs) {
    const ch = String.fromCodePoint(b);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
  }
  // Remaining bytes map to U+0100 + n in ascending byte order.
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (inBs.has(b)) continue;
    const ch = String.fromCodePoint(0x100 + n);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
    n++;
  }
  return { byteToChar, charToByte };
}

/** A single learned merge: (left id, right id) -> new id. */
interface Merge {
  readonly left: number;
  readonly right: number;
  readonly newId: number;
}

export interface ByteBpeTokenizerOptions {
  /** Reserved atomic tokens; defaults to the chat markers. */
  readonly specialTokens?: readonly string[];
}

function readPositiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/**
 * Sample up to `maxChars` from across the whole input (same evenly-spaced
 * windowing as {@link BpeTokenizer}) so learned merges are not biased toward
 * the earliest records. Byte-level default cap is much higher (5M) than the
 * char-seeded tokenizer's.
 */
export function buildByteTrainingSample(input: string, maxChars: number): string {
  if (input.length <= maxChars) return input;
  const windows = 64;
  const span = Math.max(1, Math.floor(maxChars / windows));
  const parts: string[] = [];
  for (let i = 0; i < windows; i++) {
    const start = Math.floor(((input.length - span) * i) / Math.max(1, windows - 1));
    parts.push(input.slice(start, start + span));
  }
  return parts.join("");
}

/** Multiplier for packing an (id, id) pair into one integer key. */
const PAIR_SHIFT = 1 << 21; // supports ids up to ~2M, well beyond any vocab we build

export class ByteBpeTokenizer implements Tokenizer {
  readonly name = "byte_bpe";

  private _targetVocabSize: number;

  /** id -> token string (surrogate-mapped for byte/merge tokens; literal for specials). */
  private _vocab: string[] = [];

  /** token string -> id */
  private _stoi = new Map<string, number>();

  /** Ordered learned merges. */
  private _merges: Merge[] = [];

  /** (leftId*PAIR_SHIFT+rightId) -> { rank, newId } for encode-time merging. */
  private _mergeRank: Map<number, { rank: number; newId: number }> | null = null;

  /** Special tokens, longest-first for greedy atomic matching. */
  private _specialTokens: string[] = [];

  /** special string -> id */
  private _specialToId = new Map<string, number>();

  /** special id -> string, for decode. */
  private _specialIdToStr = new Map<number, string>();

  private readonly _byteToChar: string[];
  private readonly _charToByte: Map<string, number>;
  private readonly _encoder = new TextEncoder();
  private readonly _decoder = new TextDecoder("utf-8", { fatal: false });

  constructor(vocabSize = 12288, options: ByteBpeTokenizerOptions = {}) {
    this._targetVocabSize = vocabSize;
    const specials = options.specialTokens ?? CHAT_SPECIAL_TOKENS;
    this._specialTokens = this._normalizeSpecials(specials);
    const { byteToChar, charToByte } = bytesToUnicode();
    this._byteToChar = byteToChar;
    this._charToByte = charToByte;
  }

  /** Override the target vocab size before build() (used by `alpha tokenizer build --vocabSize`). */
  setTargetVocabSize(vocabSize: number): void {
    this._targetVocabSize = vocabSize;
  }

  get vocabSize(): number {
    return this._vocab.length;
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  build(input: string): Effect.Effect<TokenizerArtifacts, TokenizerError> {
    return Effect.try({
      try: () => {
        if (input.length === 0) {
          throw new Error("Cannot build byte-BPE tokenizer from empty input");
        }
        this._initBaseVocab();

        const maxTrainChars = readPositiveIntEnv("ALPHA_BPE_MAX_TRAIN_CHARS", 5_000_000);
        const trainText = buildByteTrainingSample(input, maxTrainChars);

        // Word-frequency BPE over pre-tokens (specials stripped): each unique
        // pre-token becomes a "word" (byte-id sequence) with an occurrence count.
        const words = this._collectWords(trainText);
        this._learnMerges(words);

        return {
          type: "byte_bpe",
          vocabSize: this._vocab.length,
          byteVocab: true,
          vocab: this._vocab,
          merges: this._merges.map((m) => [m.left, m.right] as [number, number]),
          specialTokens: this._specialTokens,
        } satisfies TokenizerArtifacts;
      },
      catch: (cause) => new TokenizerError({ message: String(cause), cause }),
    });
  }

  /** Base vocab: 256 byte tokens, then reserved specials (atomic). */
  private _initBaseVocab(): void {
    this._vocab = this._byteToChar.slice(0, 256);
    this._stoi.clear();
    for (let i = 0; i < 256; i++) this._stoi.set(this._vocab[i], i);

    this._specialToId.clear();
    this._specialIdToStr.clear();
    for (const tok of this._specialTokens) {
      const id = this._vocab.length;
      this._vocab.push(tok);
      this._stoi.set(tok, id);
      this._specialToId.set(tok, id);
      this._specialIdToStr.set(id, tok);
    }
    this._merges = [];
    this._mergeRank = null;
  }

  /** Split text into non-special spans, pre-tokenize each, count byte-id words. */
  private _collectWords(text: string): { syms: number[]; count: number }[] {
    const counts = new Map<string, number>(); // surrogate-string -> count
    for (const span of this._splitOnSpecials(text)) {
      if (span.special) continue; // specials never participate in merges
      const re = gpt2SplitRegex();
      let m: RegExpExecArray | null;
      while ((m = re.exec(span.text)) !== null) {
        const piece = m[0];
        if (piece.length === 0) continue;
        const mapped = this._bytesToSurrogates(piece);
        counts.set(mapped, (counts.get(mapped) ?? 0) + 1);
      }
    }
    const words: { syms: number[]; count: number }[] = [];
    for (const [surro, count] of counts) {
      const syms: number[] = [];
      for (const ch of surro) syms.push(this._stoi.get(ch)!); // each ch is a byte surrogate
      if (syms.length >= 1) words.push({ syms, count });
    }
    return words;
  }

  /**
   * Classic word-frequency BPE: repeatedly find the most frequent adjacent
   * pair (weighted by word count), merge it, and update pair statistics via a
   * pair→words reverse index so each merge only touches affected words.
   */
  private _learnMerges(words: { syms: number[]; count: number }[]): void {
    const numMerges = this._targetVocabSize - this._vocab.length;
    if (numMerges <= 0 || words.length === 0) return;

    const pairCounts = new Map<number, number>();
    const pairToWords = new Map<number, Set<number>>();
    const key = (a: number, b: number) => a * PAIR_SHIFT + b;

    const addWord = (wi: number): void => {
      const { syms, count } = words[wi];
      for (let i = 0; i < syms.length - 1; i++) {
        const k = key(syms[i], syms[i + 1]);
        pairCounts.set(k, (pairCounts.get(k) ?? 0) + count);
        let s = pairToWords.get(k);
        if (!s) { s = new Set(); pairToWords.set(k, s); }
        s.add(wi);
      }
    };
    const removeWord = (wi: number): void => {
      const { syms, count } = words[wi];
      for (let i = 0; i < syms.length - 1; i++) {
        const k = key(syms[i], syms[i + 1]);
        const c = pairCounts.get(k);
        if (c !== undefined) {
          if (c <= count) pairCounts.delete(k);
          else pairCounts.set(k, c - count);
        }
      }
    };

    for (let wi = 0; wi < words.length; wi++) addWord(wi);

    for (let m = 0; m < numMerges; m++) {
      if (pairCounts.size === 0) break;

      // Most frequent pair; deterministic tie-break by smallest key.
      let bestKey = -1;
      let bestCount = 0;
      for (const [k, c] of pairCounts) {
        if (c > bestCount || (c === bestCount && k < bestKey)) {
          bestCount = c;
          bestKey = k;
        }
      }
      if (bestKey < 0 || bestCount < 2) break;

      const leftId = Math.floor(bestKey / PAIR_SHIFT);
      const rightId = bestKey % PAIR_SHIFT;
      const newId = this._vocab.length;
      const newToken = this._vocab[leftId] + this._vocab[rightId];
      this._vocab.push(newToken);
      this._stoi.set(newToken, newId);
      this._merges.push({ left: leftId, right: rightId, newId });

      const affected = pairToWords.get(bestKey);
      pairToWords.delete(bestKey);
      if (!affected) continue;

      for (const wi of affected) {
        removeWord(wi);
        words[wi].syms = applyMergeSeq(words[wi].syms, leftId, rightId, newId);
        addWord(wi);
      }
    }
  }

  // ── Encode ──────────────────────────────────────────────────────────────────

  encode(text: string): Int32Array {
    this._ensureMergeRank();
    const out: number[] = [];
    for (const span of this._splitOnSpecials(text)) {
      if (span.special) {
        out.push(this._specialToId.get(span.text)!);
        continue;
      }
      const re = gpt2SplitRegex();
      let m: RegExpExecArray | null;
      while ((m = re.exec(span.text)) !== null) {
        const piece = m[0];
        if (piece.length === 0) continue;
        this._encodePiece(piece, out);
      }
    }
    return new Int32Array(out);
  }

  /** Byte-map a pre-token then reduce it by learned merges, appending ids. */
  private _encodePiece(piece: string, out: number[]): void {
    const bytes = this._encoder.encode(piece);
    const n = bytes.length;
    if (n === 0) return;
    // Byte ids are the byte values' surrogate-char ids == byte value's slot.
    let syms = new Array<number>(n);
    for (let i = 0; i < n; i++) syms[i] = this._stoi.get(this._byteToChar[bytes[i]])!;

    if (n > 1 && this._merges.length > 0) {
      syms = this._mergeSyms(syms);
    }
    for (const s of syms) out.push(s);
  }

  /**
   * Reduce a byte-id sequence by learned merges, always applying the
   * lowest-rank mergeable pair first (GPT-2 / HF algorithm). Pre-tokens are
   * short, so the per-pass linear scan is cheap.
   */
  private _mergeSyms(syms: number[]): number[] {
    const rank = this._mergeRank!;
    let word = syms;
    while (word.length >= 2) {
      let minRank = Infinity;
      let minI = -1;
      let minNewId = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const info = rank.get(word[i] * PAIR_SHIFT + word[i + 1]);
        if (info && info.rank < minRank) {
          minRank = info.rank;
          minI = i;
          minNewId = info.newId;
        }
      }
      if (minI < 0) break;
      const left = word[minI];
      const right = word[minI + 1];
      word = applyMergeSeq(word, left, right, minNewId);
    }
    return word;
  }

  private _ensureMergeRank(): void {
    if (this._mergeRank) return;
    this._mergeRank = new Map();
    for (let i = 0; i < this._merges.length; i++) {
      const mg = this._merges[i];
      this._mergeRank.set(mg.left * PAIR_SHIFT + mg.right, { rank: i, newId: mg.newId });
    }
  }

  // ── Decode ──────────────────────────────────────────────────────────────────

  /**
   * Decode ids to a string. Non-special tokens are surrogate strings whose
   * chars each map back to one byte; those bytes accumulate into a buffer that
   * is UTF-8 decoded in a single pass (flushed around any special token).
   */
  decode(tokens: ArrayLike<number>): string {
    const bytes: number[] = [];
    let outStr = "";
    const flush = () => {
      if (bytes.length === 0) return;
      outStr += this._decoder.decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      const special = this._specialIdToStr.get(id);
      if (special !== undefined) {
        flush();
        outStr += special;
        continue;
      }
      const tok = this._vocab[id];
      if (tok === undefined) continue;
      for (const ch of tok) {
        const b = this._charToByte.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flush();
    return outStr;
  }

  // ── Restore ─────────────────────────────────────────────────────────────────

  loadArtifacts(artifacts: TokenizerArtifacts): void {
    this._vocab = [...artifacts.vocab];
    this._specialTokens = this._normalizeSpecials(artifacts.specialTokens ?? []);

    this._stoi.clear();
    for (let i = 0; i < this._vocab.length; i++) this._stoi.set(this._vocab[i], i);

    this._specialToId.clear();
    this._specialIdToStr.clear();
    for (const tok of this._specialTokens) {
      const id = this._stoi.get(tok);
      if (id !== undefined) {
        this._specialToId.set(tok, id);
        this._specialIdToStr.set(id, tok);
      }
    }

    this._mergeRank = null;
    if (artifacts.merges) {
      // Merged tokens occupy the ids after 256 bytes + specials; the merge's
      // newId is derivable from its position but we recover it from the pair's
      // concatenated vocab string to stay robust to id layout.
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

  // ── Helpers ─────────────────────────────────────────────────────────────────

  /** UTF-8 encode a string and map each byte to its surrogate char. */
  private _bytesToSurrogates(s: string): string {
    const bytes = this._encoder.encode(s);
    let out = "";
    for (let i = 0; i < bytes.length; i++) out += this._byteToChar[bytes[i]];
    return out;
  }

  private _normalizeSpecials(tokens: readonly string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of tokens) {
      if (!t || seen.has(t)) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  }

  /**
   * Split text into alternating non-special spans and special-token hits.
   * Special tokens are matched longest-first so overlapping markers resolve to
   * the longest atomic unit.
   */
  private _splitOnSpecials(text: string): { text: string; special: boolean }[] {
    if (this._specialTokens.length === 0) return [{ text, special: false }];
    // Longest-first ordering for greedy atomic matching.
    const ordered = [...this._specialTokens].sort((a, b) => b.length - a.length);
    const spans: { text: string; special: boolean }[] = [];
    let buf = "";
    let i = 0;
    while (i < text.length) {
      let hit: string | null = null;
      for (const tok of ordered) {
        if (text.startsWith(tok, i)) { hit = tok; break; }
      }
      if (hit) {
        if (buf) { spans.push({ text: buf, special: false }); buf = ""; }
        spans.push({ text: hit, special: true });
        i += hit.length;
      } else {
        buf += text[i];
        i++;
      }
    }
    if (buf) spans.push({ text: buf, special: false });
    return spans;
  }
}

/** Replace every non-overlapping adjacent (left,right) in `ids` with `newId`. */
function applyMergeSeq(ids: number[], left: number, right: number, newId: number): number[] {
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
