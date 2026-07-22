/**
 * Command: alpha tokenizer export-hf
 *
 * Convert byte-level BPE artifacts (schema v2) into a HuggingFace fast
 * tokenizer directory: tokenizer.json + tokenizer_config.json + chat_template.jinja.
 *
 * Usage:
 *   alpha tokenizer export-hf --artifacts=artifacts/tokenizer.json --out=hf/tokenizer
 */
import { Effect } from "effect";
import { parseKV, requireArg } from "../parse.js";
import { loadArtifacts, writeHfTokenizer } from "@alpha/tokenizers";

export async function tokenizerExportHfCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const artifactsPath = requireArg(kv, "artifacts", "path to byte-BPE artifacts JSON");
  const outDir = requireArg(kv, "out", "output directory for HF tokenizer files");

  const artifacts = await Effect.runPromise(loadArtifacts(artifactsPath));
  if (artifacts.type !== "byte_bpe" || artifacts.byteVocab !== true) {
    throw new Error(
      `export-hf requires byte-level BPE artifacts (type:"byte_bpe"); got type:"${artifacts.type}". ` +
        `Build one with: alpha tokenizer build --type=bpe-byte-12k ...`,
    );
  }

  const written = await writeHfTokenizer(outDir, artifacts);
  console.log(`Exported HF tokenizer (vocab=${artifacts.vocabSize}) to ${outDir}:`);
  for (const path of written) console.log(`  ${path}`);
}
