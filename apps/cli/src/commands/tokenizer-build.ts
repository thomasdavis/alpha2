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

  // For BPE, set vocab size if supported
  if ("targetVocabSize" in tokenizer) {
    (tokenizer as any).targetVocabSize = vocabSize;
  }

  const artifacts = await Effect.runPromise(tokenizer.build(text));

  await Effect.runPromise(saveArtifacts(outPath, artifacts));

  console.log(`Tokenizer built: vocab_size=${artifacts.vocabSize}`);
  console.log(`Artifacts saved to ${outPath}`);
}
