export interface DatagenConfig {
  /** Directory for caching downloaded sources. */
  cacheDir: string;
  /** Output file path for the generated corpus. */
  outPath: string;
  /** Regex filter for SCOWL file selection. Default matches all final/* files. */
  scowlFilter: RegExp;
  /** Which sources to use. Default: all. */
  sources?: Array<"simplewiki" | "wiktionary" | "gutenberg" | "enwiki">;
  /** How many Gutenberg books to download. Default: 5000. */
  gutenbergBooks?: number;
  /** Generation mode. "corpus" dumps full text, "concordance" extracts context windows. */
  mode?: "corpus" | "concordance";
  /** How many distinct context windows each word should appear in. Default: 1. */
  contextsPerWord?: number;
  /** Progress callback. */
  onProgress?: (phase: string, count: number) => void;
}

export const defaultDatagenConfig: Omit<DatagenConfig, "outPath"> = {
  cacheDir: "data/.cache",
  scowlFilter: /./,
  gutenbergBooks: 5000,
};

export interface GenerateResult {
  outPath: string;
  totalLines: number;
  sizeBytes: number;
  uniqueWords: number;
  coveredBySimpleWiki: number;
  coveredByWiktionary: number;
  coveredByGutenberg: number;
  coveredByEnWiki: number;
  uncovered: number;
}
