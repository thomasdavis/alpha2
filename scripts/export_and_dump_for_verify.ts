/**
 * Driver for the Python tokenizer-export cross-check (scripts/verify_tokenizer_export.py).
 *
 * Builds a byte-level BPE on a real corpus slice, exports the HF tokenizer.json
 * bundle, then encodes a sample of documents with OUR ByteBpeTokenizer and dumps
 * {text, ids} as JSONL so the Python `tokenizers` library can assert it produces
 * identical ids from the exported tokenizer.json.
 *
 * Usage:
 *   npx tsx scripts/export_and_dump_for_verify.ts \
 *     --out=/tmp/alpha-tok-verify --docs=10000 --vocab=12288
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { Effect } from "effect";
import { ByteBpeTokenizer, writeHfTokenizer, saveArtifacts } from "@alpha/tokenizers";

function argVal(name: string, def: string): string {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : def;
}

const PRETRAIN = "/mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt";
const SFT = "/mnt/donto-data/alpha-corpora/sft-text/sft.txt";
const CHAT_SPECIALS = ["<|user|>", "<|assistant|>", "<|end_of_text|>"];

async function main() {
  const outDir = argVal("out", "/tmp/alpha-tok-verify");
  const nDocs = Number.parseInt(argVal("docs", "10000"), 10);
  const vocab = Number.parseInt(argVal("vocab", "12288"), 10);
  await mkdir(outDir, { recursive: true });

  // ── Load corpus slices ────────────────────────────────────────────────────
  // ~40MB of pretrain (broad unicode text) is plenty to train + sample from.
  const pretrainBuf = existsSync(PRETRAIN)
    ? (await readFile(PRETRAIN)).subarray(0, 40 * 1024 * 1024).toString("utf-8")
    : "the quick brown fox jumps over the lazy dog. café 日本語 🎉 ".repeat(50000);
  const sftBuf = existsSync(SFT)
    ? (await readFile(SFT)).subarray(0, 8 * 1024 * 1024).toString("utf-8")
    : "<|user|> hi <|assistant|> hello <|end_of_text|>\n".repeat(5000);

  // ── Build tokenizer ───────────────────────────────────────────────────────
  console.log(`Building byte-BPE (vocab=${vocab}) on ~${(pretrainBuf.length / 1e6).toFixed(1)}M chars...`);
  const t0 = performance.now();
  const tok = new ByteBpeTokenizer(vocab, { specialTokens: CHAT_SPECIALS });
  const artifacts = await Effect.runPromise(tok.build(pretrainBuf));
  console.log(
    `  built vocab=${artifacts.vocabSize} merges=${artifacts.merges?.length} in ${((performance.now() - t0) / 1000).toFixed(1)}s`,
  );

  await saveArtifacts(join(outDir, "artifacts.json"), artifacts).pipe(Effect.runPromise);
  const written = await writeHfTokenizer(outDir, artifacts);
  console.log("Exported:", written.join(", "));

  // ── Assemble a doc sample (pretrain docs + SFT convos + special-heavy) ─────
  const docs: string[] = [];
  const pretrainDocs = pretrainBuf.split("<|end_of_text|>").map((d) => d.trim()).filter((d) => d.length > 20);
  const sftLines = sftBuf.split("\n").map((l) => l.trim()).filter((l) => l.includes("<|user|>"));

  // 60% pretrain, 25% SFT (role markers), 15% special-wrapped pretrain.
  const nP = Math.floor(nDocs * 0.6);
  const nS = Math.floor(nDocs * 0.25);
  const nW = nDocs - nP - nS;
  for (let i = 0; i < nP && i < pretrainDocs.length; i++) docs.push(pretrainDocs[i].slice(0, 4000));
  for (let i = 0; i < nS && i < sftLines.length; i++) docs.push(sftLines[i].slice(0, 4000));
  for (let i = 0; i < nW && i < pretrainDocs.length; i++) {
    const d = pretrainDocs[pretrainDocs.length - 1 - i].slice(0, 800);
    docs.push(`<|user|> ${d} <|assistant|> ${d} <|end_of_text|>`);
  }

  // ── Encode with OUR tokenizer + dump JSONL ────────────────────────────────
  const lines: string[] = [];
  let maxLen = 0;
  for (const text of docs) {
    const ids = Array.from(tok.encode(text));
    maxLen = Math.max(maxLen, ids.length);
    lines.push(JSON.stringify({ text, ids }));
  }
  await writeFile(join(outDir, "samples.jsonl"), lines.join("\n"), "utf-8");
  console.log(`Wrote ${docs.length} sample docs (max ${maxLen} tokens) to ${join(outDir, "samples.jsonl")}`);
  console.log(`\nNext: python3 scripts/verify_tokenizer_export.py --dir=${outDir}`);
}

main().catch((e) => {
  console.error("driver failed:", e);
  process.exit(1);
});
