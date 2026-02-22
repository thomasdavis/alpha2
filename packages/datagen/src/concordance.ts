import * as fs from "node:fs";
import * as path from "node:path";
import { SeededRng } from "@alpha/core";
import type { DatagenConfig, GenerateResult } from "./types.js";
import { shuffle } from "./templates.js";
import {
  fetchWordList,
  downloadSimpleWiki,
  downloadWiktionary,
  downloadGutenberg,
  downloadEnWiki,
  streamBz2Dump,
  streamGutenberg,
} from "./sources.js";

const WORD_RE = /[^a-zA-Z]+/;
const MIN_PARAGRAPH_LENGTH = 40;

/** Split text into paragraphs on blank lines. */
function splitParagraphs(text: string): string[] {
  const paragraphs: string[] = [];
  for (const para of text.split(/\n\s*\n/)) {
    const trimmed = para.trim().replace(/\s+/g, " ");
    if (trimmed.length > 0) paragraphs.push(trimmed);
  }
  return paragraphs;
}

/** Extract words from text as lowercase tokens. */
function extractWords(text: string): string[] {
  return text.split(WORD_RE).filter((w) => w.length >= 2).map((w) => w.toLowerCase());
}

/**
 * Generate a concordance corpus: for each under-covered target word, find real
 * paragraphs containing it and extract a context window (paragraph + 1 neighbor
 * on each side). Produces a smaller, denser corpus than full-text dumping.
 *
 * Improvements over v1:
 * - Paragraph-based windowing (no fragile sentence splitter)
 * - Better dedup (skip trigger paragraphs already included in prior windows)
 * - Minimum length filter (skip short degenerate paragraphs)
 * - Multiple contexts per word (configurable via contextsPerWord)
 * - Shuffled output (deterministic with SeededRng)
 */
export async function generateConcordance(config: DatagenConfig): Promise<GenerateResult> {
  const { cacheDir, outPath, scowlFilter, onProgress } = config;
  const sources = config.sources ?? ["simplewiki", "wiktionary", "gutenberg", "enwiki"];
  const contextsPerWord = config.contextsPerWord ?? 1;

  // Phase 1: Build target word set
  const allWords = fetchWordList(cacheDir, scowlFilter, onProgress);
  const targetWords = new Set(allWords);
  onProgress?.("target vocab", targetWords.size);

  // Track how many contexts each word has been seen in
  const wordContextCount = new Map<string, number>();

  /** Check if a word still needs more contexts. */
  function needsCoverage(word: string): boolean {
    return (wordContextCount.get(word) ?? 0) < contextsPerWord;
  }

  /** Count how many words still need coverage. */
  function uncoveredCount(): number {
    let count = 0;
    for (const w of targetWords) {
      if (needsCoverage(w)) count++;
    }
    return count;
  }

  // Collect all windows, then shuffle at the end
  const windows: string[] = [];

  /** Process a single article/book text through the concordance extractor. */
  function processText(text: string): void {
    const paragraphs = splitParagraphs(text);
    if (paragraphs.length === 0) return;

    // Track which paragraph indices we've already emitted for this article
    const emitted = new Set<number>();

    for (let i = 0; i < paragraphs.length; i++) {
      // Skip short trigger paragraphs ("Section 3.", "ARTICLE FIVE", etc.)
      if (paragraphs[i].length < MIN_PARAGRAPH_LENGTH) continue;

      // Skip if this paragraph was already included in a prior window
      if (emitted.has(i)) continue;

      const words = extractWords(paragraphs[i]);
      const hasNeeded = words.some((w) => targetWords.has(w) && needsCoverage(w));
      if (!hasNeeded) continue;

      // Extract context window: trigger paragraph + 1 neighbor on each side
      const start = Math.max(0, i - 1);
      const end = Math.min(paragraphs.length - 1, i + 1);

      // Build the context window
      const windowParts: string[] = [];
      for (let j = start; j <= end; j++) {
        windowParts.push(paragraphs[j]);
        emitted.add(j);
      }

      const windowText = windowParts.join("\n\n");
      windows.push(windowText);

      // Increment context count for ALL target words in the window
      const windowWords = extractWords(windowText);
      for (const w of windowWords) {
        if (targetWords.has(w)) {
          wordContextCount.set(w, (wordContextCount.get(w) ?? 0) + 1);
        }
      }
    }
  }

  const covered = new Set<string>();
  function snapshotCoverage(): number {
    const prevSize = covered.size;
    for (const w of targetWords) {
      if (!needsCoverage(w)) covered.add(w);
    }
    return covered.size - prevSize;
  }

  let coveredBySimpleWiki = 0;
  let coveredByWiktionary = 0;
  let coveredByGutenberg = 0;
  let coveredByEnWiki = 0;

  // Process sources in the order specified by the config
  for (const source of sources) {
    switch (source) {
      case "simplewiki": {
        const bz2 = downloadSimpleWiki(cacheDir, onProgress);
        onProgress?.("concordance SimpleWiki", 0);
        await streamBz2Dump(bz2, targetWords, processText, onProgress, "simplewiki");
        coveredBySimpleWiki = snapshotCoverage();
        onProgress?.("simplewiki concordance coverage", coveredBySimpleWiki);
        onProgress?.("uncovered remaining", uncoveredCount());
        break;
      }
      case "wiktionary": {
        const bz2 = downloadWiktionary(cacheDir, onProgress);
        onProgress?.("concordance Wiktionary", 0);
        await streamBz2Dump(bz2, targetWords, processText, onProgress, "wiktionary");
        coveredByWiktionary = snapshotCoverage();
        onProgress?.("wiktionary concordance coverage", coveredByWiktionary);
        onProgress?.("uncovered remaining", uncoveredCount());
        break;
      }
      case "gutenberg": {
        const dir = await downloadGutenberg(cacheDir, config.gutenbergBooks ?? 5000, onProgress);
        onProgress?.("concordance Gutenberg", 0);
        streamGutenberg(dir, targetWords, processText, onProgress);
        coveredByGutenberg = snapshotCoverage();
        onProgress?.("gutenberg concordance coverage", coveredByGutenberg);
        onProgress?.("uncovered remaining", uncoveredCount());
        break;
      }
      case "enwiki": {
        const bz2 = downloadEnWiki(cacheDir, onProgress);
        onProgress?.("concordance EnWiki", 0);
        await streamBz2Dump(bz2, targetWords, processText, onProgress, "enwiki");
        coveredByEnWiki = snapshotCoverage();
        onProgress?.("enwiki concordance coverage", coveredByEnWiki);
        onProgress?.("uncovered remaining", uncoveredCount());
        break;
      }
    }
  }

  // Shuffle all collected windows deterministically
  const rng = new SeededRng(42);
  shuffle(windows, rng);

  // Write shuffled output
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const fd = fs.openSync(outPath, "w");
  let totalLines = 0;

  for (const window of windows) {
    fs.writeSync(fd, window + "\n\n");
    // Count lines in this window (paragraph lines + blank separator)
    totalLines += window.split("\n").length + 1;
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
    uncovered: uncoveredCount(),
  };
}
