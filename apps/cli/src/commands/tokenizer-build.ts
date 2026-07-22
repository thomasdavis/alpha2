/**
 * Command: alpha tokenizer build
 */
import { Effect } from "effect";
import { parseKV, requireArg, intArg, strArg } from "../parse.js";
import { resolveTokenizer } from "../resolve.js";
import { saveArtifacts } from "@alpha/tokenizers";

export async function tokenizerBuildCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const type = strArg(kv, "type", "bpe");
  const inputPath = requireArg(kv, "input", "path to training text");
  // Only override the tokenizer's target vocab when --vocabSize is EXPLICITLY
  // given; otherwise honor the registry preset (e.g. bpe-byte-12k → 12288,
  // bpe-byte-4k → 4096). Passing the 2000 default unconditionally used to
  // clobber every preset back to 2000.
  const explicitVocabSize = kv["vocabSize"] !== undefined ? intArg(kv, "vocabSize", 2000) : undefined;
  const outPath = requireArg(kv, "out", "output path for artifacts");

  const tokenizer = resolveTokenizer(type);

  if (explicitVocabSize !== undefined) {
    if (typeof (tokenizer as any).setTargetVocabSize === "function") {
      (tokenizer as any).setTargetVocabSize(explicitVocabSize);
    } else {
      console.warn(`--vocabSize=${explicitVocabSize} ignored: tokenizer "${type}" has no target vocab size`);
    }
  }

  console.log(
    `Building ${type} tokenizer from ${inputPath}` +
      (explicitVocabSize !== undefined ? ` (vocabSize=${explicitVocabSize})` : " (preset vocab)"),
  );

  const fs = await import("node:fs/promises");
  const text = await fs.readFile(inputPath, "utf-8");

  const artifacts = await Effect.runPromise(tokenizer.build(text));

  await Effect.runPromise(saveArtifacts(outPath, artifacts));

  console.log(`Tokenizer built: vocab_size=${artifacts.vocabSize}`);
  console.log(`Artifacts saved to ${outPath}`);
}
