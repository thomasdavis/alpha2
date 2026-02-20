import * as fs from "node:fs";
import * as path from "node:path";
import type { DatagenConfig, GenerateResult } from "./types.js";
import {
  fetchWordList,
  downloadSimpleWiki,
  downloadWiktionary,
  downloadGutenberg,
  streamGutenberg,
  downloadEnWiki,
  streamBz2Dump,
} from "./sources.js";

/**
 * Generate a training corpus from multiple real-text sources:
 * 1. SimpleWiki articles
 * 2. English Wiktionary (dictionary definitions)
 * 3. English Wikipedia (if still uncovered words remain)
 *
 * Tracks vocabulary coverage at each stage.
 */
export async function generate(config: DatagenConfig): Promise<GenerateResult> {
  const { cacheDir, outPath, scowlFilter, onProgress } = config;
  const sources = config.sources ?? ["simplewiki", "wiktionary", "gutenberg", "enwiki"];

  // Phase 1: Build target word list from SCOWL + wiki titles
  const allWords = fetchWordList(cacheDir, scowlFilter, onProgress);
  const targetWords = new Set(allWords);
  onProgress?.("target vocab", targetWords.size);

  // Setup output file
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const fd = fs.openSync(outPath, "w");
  let totalLines = 0;

  function writeLine(line: string): void {
    fs.writeSync(fd, line + "\n");
    totalLines++;
  }

  function writeArticle(text: string): void {
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (trimmed.length > 0) {
        writeLine(trimmed);
      } else {
        writeLine("");
      }
    }
    writeLine("");
  }

  const covered = new Set<string>();

  // Phase 2: SimpleWiki
  let coveredBySimpleWiki = 0;
  if (sources.includes("simplewiki")) {
    const simpleWikiBz2 = downloadSimpleWiki(cacheDir, onProgress);
    onProgress?.("streaming SimpleWiki", 0);
    const swCovered = await streamBz2Dump(simpleWikiBz2, targetWords, writeArticle, onProgress, "simplewiki");
    for (const w of swCovered) covered.add(w);
    coveredBySimpleWiki = swCovered.size;
    onProgress?.("simplewiki coverage", covered.size);
    onProgress?.("uncovered after simplewiki", targetWords.size - covered.size);
  }

  // Phase 3: Wiktionary
  let coveredByWiktionary = 0;
  if (sources.includes("wiktionary") && covered.size < targetWords.size) {
    const wiktBz2 = downloadWiktionary(cacheDir, onProgress);
    onProgress?.("streaming Wiktionary", 0);
    const before = covered.size;
    const wiktCovered = await streamBz2Dump(wiktBz2, targetWords, writeArticle, onProgress, "wiktionary");
    for (const w of wiktCovered) covered.add(w);
    coveredByWiktionary = covered.size - before;
    onProgress?.("wiktionary new coverage", coveredByWiktionary);
    onProgress?.("uncovered after wiktionary", targetWords.size - covered.size);
  }

  // Phase 4: Project Gutenberg books
  let coveredByGutenberg = 0;
  if (sources.includes("gutenberg") && covered.size < targetWords.size) {
    const gutenbergDir = await downloadGutenberg(cacheDir, config.gutenbergBooks ?? 5000, onProgress);
    onProgress?.("processing Gutenberg books", 0);
    const before = covered.size;
    const gutenbergCovered = streamGutenberg(gutenbergDir, targetWords, writeArticle, onProgress);
    for (const w of gutenbergCovered) covered.add(w);
    coveredByGutenberg = covered.size - before;
    onProgress?.("gutenberg new coverage", coveredByGutenberg);
    onProgress?.("uncovered after gutenberg", targetWords.size - covered.size);
  }

  // Phase 5: English Wikipedia (only if uncovered words remain)
  let coveredByEnWiki = 0;
  if (sources.includes("enwiki") && covered.size < targetWords.size) {
    const enwikiBz2 = downloadEnWiki(cacheDir, onProgress);
    onProgress?.("streaming English Wikipedia", 0);
    const before = covered.size;
    const enwikiCovered = await streamBz2Dump(enwikiBz2, targetWords, writeArticle, onProgress, "enwiki");
    for (const w of enwikiCovered) covered.add(w);
    coveredByEnWiki = covered.size - before;
    onProgress?.("enwiki new coverage", coveredByEnWiki);
    onProgress?.("uncovered after enwiki", targetWords.size - covered.size);
  }

  fs.closeSync(fd);

  const stat = fs.statSync(outPath);
  return {
    outPath,
    totalLines,
    sizeBytes: stat.size,
    uniqueWords: targetWords.size,
    coveredBySimpleWiki,
    coveredByWiktionary,
    coveredByGutenberg,
    coveredByEnWiki,
    uncovered: targetWords.size - covered.size,
  };
}
