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
  const vocabSize = intArg(kv, "vocabSize", 2000);
  const outPath = requireArg(kv, "out", "output path for artifacts");

  console.log(`Building ${type} tokenizer from ${inputPath} (vocabSize=${vocabSize})`);

  const fs = await import("node:fs/promises");
  const text = await fs.readFile(inputPath, "utf-8");

  const tokenizer = resolveTokenizer(type);

  // For BPE, set vocab size if supported.
  // (The old `"targetVocabSize" in tokenizer` guard never matched the private field,
  // so --vocabSize was silently ignored and every `--type=bpe` build produced vocab 2000.)
  if (typeof (tokenizer as any).setTargetVocabSize === "function") {
    (tokenizer as any).setTargetVocabSize(vocabSize);
  } else if (vocabSize) {
    console.warn(`--vocabSize=${vocabSize} ignored: tokenizer "${type}" has no target vocab size`);
  }

  const artifacts = await Effect.runPromise(tokenizer.build(text));

  await Effect.runPromise(saveArtifacts(outPath, artifacts));

  console.log(`Tokenizer built: vocab_size=${artifacts.vocabSize}`);
  console.log(`Artifacts saved to ${outPath}`);
}
