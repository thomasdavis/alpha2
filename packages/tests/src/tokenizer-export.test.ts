/**
 * Round-trip proof for the HF tokenizer.json exporter (network-free).
 *
 * Builds a small byte-level BPE, exports the tokenizer.json, then applies the
 * exported spec MANUALLY (an independent re-implementation of the HuggingFace
 * ByteLevel + BPE pipeline that reads ONLY the exported file's vocab/merges and
 * a locally-recomputed GPT-2 byte map) and asserts it produces the exact same
 * ids as our ByteBpeTokenizer.encode() on a battery of tricky strings.
 *
 * This is the in-repo equivalent of the Python `tokenizers` cross-check in
 * scripts/verify_tokenizer_export.py.
 */
import { describe, it, expect } from "vitest";
import { Effect } from "effect";
import { ByteBpeTokenizer, exportHfTokenizer } from "@alpha/tokenizers";

const CHAT_SPECIALS = ["<|user|>", "<|assistant|>", "<|end_of_text|>"];

/** Local, independent GPT-2 bytes_to_unicode (does not import the tokenizer's copy). */
function gpt2ByteMap(): string[] {
  const bs: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
  for (let b = 0xae; b <= 0xff; b++) bs.push(b);
  const inBs = new Set(bs);
  const map = new Array<string>(256);
  for (const b of bs) map[b] = String.fromCodePoint(b);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (inBs.has(b)) continue;
    map[b] = String.fromCodePoint(0x100 + n);
    n++;
  }
  return map;
}

// Whitespace spelled as \p{White_Space} to mirror HF's Rust ByteLevel engine
// (see byte_bpe.ts GPT2_SPLIT_PATTERN — JS \s wrongly includes U+FEFF).
const GPT2_RE =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+|\\p{White_Space}+(?!\\P{White_Space})|\\p{White_Space}+";

interface HfTokenizerJson {
  added_tokens: { id: number; content: string; special: boolean }[];
  model: { vocab: Record<string, number>; merges: string[] };
}

/**
 * Apply the exported tokenizer.json to `text` exactly as HF would:
 *  1. isolate special (added) tokens, longest-first;
 *  2. GPT-2 regex pre-tokenize the rest;
 *  3. UTF-8 → surrogate chars;
 *  4. BPE-merge by the merges list (lowest rank first);
 *  5. map surrogate token strings → ids via the vocab.
 */
function applyExportedSpec(tok: HfTokenizerJson, text: string): number[] {
  const byteMap = gpt2ByteMap();
  const enc = new TextEncoder();
  const vocab = tok.model.vocab;
  const specials = tok.added_tokens
    .filter((t) => t.special)
    .map((t) => t.content)
    .sort((a, b) => b.length - a.length); // longest-first

  // Merge rank: "left right" -> rank; merged token string = left+right.
  const rank = new Map<string, number>();
  for (let i = 0; i < tok.model.merges.length; i++) rank.set(tok.model.merges[i], i);

  const out: number[] = [];

  // Split on specials.
  const segments: { text: string; special: boolean }[] = [];
  let buf = "";
  let i = 0;
  while (i < text.length) {
    let hit: string | null = null;
    for (const sp of specials) {
      if (text.startsWith(sp, i)) { hit = sp; break; }
    }
    if (hit) {
      if (buf) { segments.push({ text: buf, special: false }); buf = ""; }
      segments.push({ text: hit, special: true });
      i += hit.length;
    } else {
      buf += text[i];
      i++;
    }
  }
  if (buf) segments.push({ text: buf, special: false });

  for (const seg of segments) {
    if (seg.special) {
      out.push(vocab[seg.text]);
      continue;
    }
    const re = new RegExp(GPT2_RE, "gu");
    let m: RegExpExecArray | null;
    while ((m = re.exec(seg.text)) !== null) {
      const piece = m[0];
      if (!piece) continue;
      // bytes -> surrogate chars
      const bytes = enc.encode(piece);
      let symbols: string[] = [];
      for (const b of bytes) symbols.push(byteMap[b]);
      // BPE merge by rank
      while (symbols.length >= 2) {
        let minRank = Infinity;
        let minI = -1;
        for (let k = 0; k < symbols.length - 1; k++) {
          const r = rank.get(symbols[k] + " " + symbols[k + 1]);
          if (r !== undefined && r < minRank) { minRank = r; minI = k; }
        }
        if (minI < 0) break;
        const merged = symbols[minI] + symbols[minI + 1];
        symbols = [...symbols.slice(0, minI), merged, ...symbols.slice(minI + 2)];
      }
      for (const s of symbols) out.push(vocab[s]);
    }
  }
  return out;
}

const TRICKY_STRINGS = [
  "hello world",
  "The quick brown fox jumps over the lazy dog.",
  "  leading and   internal    whitespace   runs  ",
  "tabs\tand\nnewlines\r\nmixed",
  "café résumé naïve — em—dash",
  "日本語のテキスト 中文 한국어",
  "emoji party 🎉🚀 and family 👨‍👩‍👧‍👦",
  "'contractions' don't can't won't it's I'm we'll they'd you're",
  "function add(a, b) { return a + b; } // code snippet",
  "numbers 0 1 2 3 42 1000000 3.14159 -7",
  "PUNCTUATION!!! ??? ...  @#$%^&*()_+-=[]{}|;:,.<>/?",
  "MixedCASE camelCase snake_case kebab-case",
  "<|user|> hi there <|assistant|> hello! <|end_of_text|>",
  "text before <|end_of_text|> text after",
  "back-to-back<|user|><|assistant|><|end_of_text|>specials",
  "русский текст и ελληνικά",
  "trailing spaces at end     ",
  "     leading spaces at start",
  "BOM﻿then text and \n\n﻿newlines-before-bom", // U+FEFF: JS \s vs Rust \s divergence
  "NELseparatedlines", // U+0085 (NEL): the other \s divergence
  "a",
  "",
];

describe("HF tokenizer.json exporter", () => {
  it("throws on non-byte-level artifacts", async () => {
    const bad: any = { type: "bpe", vocabSize: 3, vocab: ["a", "b", "c"] };
    expect(() => exportHfTokenizer(bad)).toThrow(/byte-level/);
  });

  it("emits a well-formed BPE model + ByteLevel pre_tokenizer/decoder", async () => {
    const tok = new ByteBpeTokenizer(600, { specialTokens: CHAT_SPECIALS });
    const art = await Effect.runPromise(
      tok.build("the cat sat on the mat. <|user|> hi <|assistant|> yo <|end_of_text|> ".repeat(150)),
    );
    const { tokenizerJson, tokenizerConfig } = exportHfTokenizer(art) as any;

    expect(tokenizerJson.model.type).toBe("BPE");
    expect(tokenizerJson.model.ignore_merges).toBe(false);
    expect(tokenizerJson.model.byte_fallback).toBe(false);
    expect(tokenizerJson.pre_tokenizer.type).toBe("ByteLevel");
    expect(tokenizerJson.pre_tokenizer.add_prefix_space).toBe(false);
    expect(tokenizerJson.pre_tokenizer.use_regex).toBe(true);
    expect(tokenizerJson.decoder.type).toBe("ByteLevel");

    // 256 byte tokens are all present in vocab.
    expect(Object.keys(tokenizerJson.model.vocab).length).toBe(art.vocabSize);
    // Every merge string has exactly two space-separated parts (surrogate map
    // guarantees no literal spaces inside tokens).
    for (const mg of tokenizerJson.model.merges) {
      expect(mg.split(" ").length).toBe(2);
    }
    // 3 specials as added_tokens with special:true.
    expect(tokenizerJson.added_tokens.map((t: any) => t.content).sort()).toEqual(
      [...CHAT_SPECIALS].sort(),
    );
    for (const t of tokenizerJson.added_tokens) expect(t.special).toBe(true);

    // config
    expect(tokenizerConfig.tokenizer_class).toBe("PreTrainedTokenizerFast");
    expect(tokenizerConfig.eos_token).toBe("<|end_of_text|>");
    expect(tokenizerConfig.bos_token).toBe("<|end_of_text|>");
    expect(tokenizerConfig.chat_template).toContain("{% generation %}");
    expect(tokenizerConfig.chat_template).toContain("<|user|>");
    expect(tokenizerConfig.chat_template).toContain("<|assistant|>");
    expect(tokenizerConfig.chat_template).toContain("{% if loop.last %}");
    expect(tokenizerConfig.chat_template).toContain("[Instructions: ");
    expect(tokenizerConfig.chat_template).toContain(
      "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}",
    );
    expect(tokenizerConfig.chat_template).not.toContain(
      "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}",
    );
  });

  it("exported spec reproduces ByteBpeTokenizer.encode() on tricky strings", async () => {
    const tok = new ByteBpeTokenizer(1200, { specialTokens: CHAT_SPECIALS });
    const art = await Effect.runPromise(
      tok.build(
        (
          "the cat sat on the mat. the quick brown fox jumps over the lazy dog. " +
          "café résumé 日本語 counting 123 456. don't won't it's. code: fn(a,b){return a+b;} " +
          "<|user|> hello there friend <|assistant|> hi, how can I help? <|end_of_text|> "
        ).repeat(200),
      ),
    );
    const { tokenizerJson } = exportHfTokenizer(art) as { tokenizerJson: HfTokenizerJson };

    for (const s of TRICKY_STRINGS) {
      const ours = [...tok.encode(s)];
      const theirs = applyExportedSpec(tokenizerJson, s);
      expect(theirs, `mismatch on ${JSON.stringify(s)}`).toEqual(ours);
    }
  });

  it("exported vocab/merges ids agree with our decode round-trip", async () => {
    const tok = new ByteBpeTokenizer(800, { specialTokens: CHAT_SPECIALS });
    const art = await Effect.runPromise(
      tok.build("mixed content 🎉 <|user|> q <|assistant|> a <|end_of_text|> ".repeat(200)),
    );
    const { tokenizerJson } = exportHfTokenizer(art) as { tokenizerJson: HfTokenizerJson };
    const probe = "round 🎉 trip <|user|> hi <|end_of_text|>";
    const theirs = applyExportedSpec(tokenizerJson, probe);
    expect([...tok.encode(probe)]).toEqual(theirs);
    expect(tok.decode(theirs)).toBe(probe);
  });
});
