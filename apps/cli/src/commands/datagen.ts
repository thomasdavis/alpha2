/**
 * Command: alpha datagen
 *
 * Generate a training corpus from SimpleWiki, Wiktionary, Gutenberg, and English Wikipedia.
 */
import { parseKV, strArg, intArg } from "../parse.js";
import { generate, generateConcordance, defaultDatagenConfig } from "@alpha/datagen";

export async function datagenCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const outPath = strArg(kv, "out", "data/chaos.txt");
  const cacheDir = strArg(kv, "cache", defaultDatagenConfig.cacheDir);
  const filterStr = kv["scowlFilter"];
  const scowlFilter = filterStr ? new RegExp(filterStr) : defaultDatagenConfig.scowlFilter;

  const sourcesStr = strArg(kv, "sources", "simplewiki,wiktionary,gutenberg,enwiki");
  const sources = sourcesStr.split(",") as Array<"simplewiki" | "wiktionary" | "gutenberg" | "enwiki">;
  const gutenbergBooks = intArg(kv, "gutenbergBooks", defaultDatagenConfig.gutenbergBooks!);
  const mode = strArg(kv, "mode", "corpus") as "corpus" | "concordance";

  console.log(`Generating ${mode} → ${outPath} (sources: ${sources.join(", ")})`);

  const genFn = mode === "concordance" ? generateConcordance : generate;
  const result = await genFn({
    cacheDir,
    outPath,
    scowlFilter,
    sources,
    gutenbergBooks,
    mode,
    onProgress: (phase: string, count: number) => console.log(`[${phase}] ${count}`),
  });

  const sizeMB = (result.sizeBytes / (1024 * 1024)).toFixed(1);
  console.log(`\nDone: ${result.totalLines} lines → ${result.outPath} (${sizeMB} MB)`);
  console.log(`Unique words: ${result.uniqueWords}`);
  console.log(`Covered by SimpleWiki: ${result.coveredBySimpleWiki}`);
  console.log(`Covered by Wiktionary: +${result.coveredByWiktionary}`);
  console.log(`Covered by Gutenberg: +${result.coveredByGutenberg}`);
  console.log(`Covered by EnWiki: +${result.coveredByEnWiki}`);
  console.log(`Uncovered: ${result.uncovered}`);
}
