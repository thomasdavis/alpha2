#!/usr/bin/env npx tsx
/** Build the exact token cache consumed by Alpha training without initializing a model or GPU. */

import { createHash } from "node:crypto";
import { stat } from "node:fs/promises";
import { Effect } from "effect";
import { resolveTokenizer } from "../apps/cli/src/resolve.js";
import { loadArtifacts } from "@alpha/tokenizers";
import { loadOrCacheTokens } from "@alpha/train";

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    result[arg.slice(2)] = value;
  }
  return result;
}

async function main(): Promise<void> {
  const cli = parseArgs();
  for (const required of ["data", "tokenizerArtifacts"]) {
    if (!cli[required]) throw new Error(`missing --${required}`);
  }
  const tokenizerName = cli.tokenizer ?? "bpe-byte-12k";
  const [dataStat, artifacts] = await Promise.all([
    stat(cli.data),
    Effect.runPromise(loadArtifacts(cli.tokenizerArtifacts)),
  ]);
  if (!dataStat.isFile() || dataStat.size === 0) throw new Error("--data must be a non-empty file");
  const tokenizer = resolveTokenizer(tokenizerName);
  const tokenizerWithArtifacts = tokenizer as typeof tokenizer & {
    loadArtifacts?: (artifacts: typeof artifacts) => void;
  };
  if (typeof tokenizerWithArtifacts.loadArtifacts !== "function") {
    throw new Error(`tokenizer ${tokenizerName} does not support loading frozen artifacts`);
  }
  tokenizerWithArtifacts.loadArtifacts(artifacts);
  if (tokenizer.vocabSize !== artifacts.vocabSize) {
    throw new Error(`tokenizer vocabulary ${tokenizer.vocabSize} != artifact ${artifacts.vocabSize}`);
  }
  const cacheIdentity = createHash("sha256").update(JSON.stringify(artifacts)).digest("hex");
  const started = performance.now();
  const tokens = await loadOrCacheTokens(cli.data, tokenizer, undefined, cacheIdentity);
  const report = {
    schema: "alpha-pretokenize-pretrain-shard-v1",
    data: cli.data,
    data_bytes: dataStat.size,
    tokenizer_artifacts: cli.tokenizerArtifacts,
    tokenizer_name: tokenizer.name,
    tokenizer_vocab_size: tokenizer.vocabSize,
    tokenizer_cache_identity: cacheIdentity,
    tokens: tokens.length,
    elapsed_seconds: (performance.now() - started) / 1000,
  };
  console.log(JSON.stringify(report));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
