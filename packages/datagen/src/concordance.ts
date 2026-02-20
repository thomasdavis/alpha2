import * as fs from "node:fs";
import * as path from "node:path";
import type { DatagenConfig, GenerateResult } from "./types.js";
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
const SENTENCE_RE = /(?<=[.!?])\s+(?=[A-Z])/;
const CONTEXT_RADIUS = 2;

/** Split text into sentences. Falls back to one sentence per paragraph if no splits found. */
function splitSentences(text: string): string[] {
  const sentences: string[] = [];
  for (const para of text.split(/\n\s*\n/)) {
    const trimmed = para.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(SENTENCE_RE);
    for (const s of parts) {
      const st = s.trim();
      if (st.length > 0) sentences.push(st);
    }
  }
  return sentences;
}

/** Extract words from text as lowercase tokens. */
function extractWords(text: string): string[] {
  return text.split(WORD_RE).filter((w) => w.length >= 2).map((w) => w.toLowerCase());
}

/**
 * Generate a concordance corpus: for each uncovered target word, find one real
 * sentence containing it and extract a context window (sentence +/- 2 surrounding
 * sentences). Produces a smaller, denser corpus than full-text dumping.
 */
export async function generateConcordance(config: DatagenConfig): Promise<GenerateResult> {
  const { cacheDir, outPath, scowlFilter, onProgress } = config;
  const sources = config.sources ?? ["simplewiki", "wiktionary", "gutenberg", "enwiki"];

  // Phase 1: Build target word set
  const allWords = fetchWordList(cacheDir, scowlFilter, onProgress);
  const targetWords = new Set(allWords);
  const uncovered = new Set(allWords);
  onProgress?.("target vocab", targetWords.size);

  // Setup output
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const fd = fs.openSync(outPath, "w");
  let totalLines = 0;

  function writeLine(line: string): void {
    fs.writeSync(fd, line + "\n");
    totalLines++;
  }

  /** Process a single article/book text through the concordance extractor. */
  function processText(text: string): void {
    if (uncovered.size === 0) return;

    const sentences = splitSentences(text);
    if (sentences.length === 0) return;

    // Track which sentence indices we've already emitted to avoid duplicates
    const emitted = new Set<number>();

    for (let i = 0; i < sentences.length; i++) {
      if (uncovered.size === 0) break;

      const words = extractWords(sentences[i]);
      const hasUncovered = words.some((w) => uncovered.has(w));
      if (!hasUncovered) continue;

      // Extract context window: [i-2 .. i+2] clamped to bounds
      const start = Math.max(0, i - CONTEXT_RADIUS);
      const end = Math.min(sentences.length - 1, i + CONTEXT_RADIUS);

      // Skip if we already emitted this window's trigger sentence
      if (emitted.has(i)) continue;

      // Build the context window
      const window: string[] = [];
      for (let j = start; j <= end; j++) {
        window.push(sentences[j]);
        emitted.add(j);
      }

      const windowText = window.join(" ");
      writeLine(windowText);
      writeLine("");

      // Mark ALL target words in the window as covered
      const windowWords = extractWords(windowText);
      for (const w of windowWords) {
        if (uncovered.has(w)) uncovered.delete(w);
      }
    }
  }

  const covered = new Set<string>();
  function snapshotCoverage(): number {
    const nowCovered = targetWords.size - uncovered.size;
    const delta = nowCovered - covered.size;
    for (const w of targetWords) {
      if (!uncovered.has(w)) covered.add(w);
    }
    return delta;
  }

  // Phase 2: SimpleWiki
  let coveredBySimpleWiki = 0;
  if (sources.includes("simplewiki") && uncovered.size > 0) {
    const bz2 = downloadSimpleWiki(cacheDir, onProgress);
    onProgress?.("concordance SimpleWiki", 0);
    await streamBz2Dump(bz2, targetWords, processText, onProgress, "simplewiki");
    coveredBySimpleWiki = snapshotCoverage();
    onProgress?.("simplewiki concordance coverage", coveredBySimpleWiki);
    onProgress?.("uncovered remaining", uncovered.size);
  }

  // Phase 3: Wiktionary
  let coveredByWiktionary = 0;
  if (sources.includes("wiktionary") && uncovered.size > 0) {
    const bz2 = downloadWiktionary(cacheDir, onProgress);
    onProgress?.("concordance Wiktionary", 0);
    await streamBz2Dump(bz2, targetWords, processText, onProgress, "wiktionary");
    coveredByWiktionary = snapshotCoverage();
    onProgress?.("wiktionary concordance coverage", coveredByWiktionary);
    onProgress?.("uncovered remaining", uncovered.size);
  }

  // Phase 4: Gutenberg
  let coveredByGutenberg = 0;
  if (sources.includes("gutenberg") && uncovered.size > 0) {
    const dir = await downloadGutenberg(cacheDir, config.gutenbergBooks ?? 5000, onProgress);
    onProgress?.("concordance Gutenberg", 0);
    streamGutenberg(dir, targetWords, processText, onProgress);
    coveredByGutenberg = snapshotCoverage();
    onProgress?.("gutenberg concordance coverage", coveredByGutenberg);
    onProgress?.("uncovered remaining", uncovered.size);
  }

  // Phase 5: English Wikipedia
  let coveredByEnWiki = 0;
  if (sources.includes("enwiki") && uncovered.size > 0) {
    const bz2 = downloadEnWiki(cacheDir, onProgress);
    onProgress?.("concordance EnWiki", 0);
    await streamBz2Dump(bz2, targetWords, processText, onProgress, "enwiki");
    coveredByEnWiki = snapshotCoverage();
    onProgress?.("enwiki concordance coverage", coveredByEnWiki);
    onProgress?.("uncovered remaining", uncovered.size);
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
    uncovered: uncovered.size,
  };
}
