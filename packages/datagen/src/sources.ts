import * as fs from "node:fs";
import * as path from "node:path";
import { execSync, spawn } from "node:child_process";
import { gunzipSync } from "node:zlib";
import * as readline from "node:readline";

const SCOWL_URL =
  "https://sourceforge.net/projects/wordlist/files/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz/download";
const WIKI_TITLES_URL =
  "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-all-titles-in-ns0.gz";
const SIMPLEWIKI_URL =
  "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2";
const WIKTIONARY_URL =
  "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2";
const ENWIKI_URL =
  "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2";

const WORD_RE = /^[a-zA-Z]{2,30}$/;

type ProgressFn = (phase: string, count: number) => void;

function download(url: string, dest: string, label: string, onProgress?: ProgressFn): void {
  if (fs.existsSync(dest)) {
    onProgress?.("cache hit " + label, 0);
    return;
  }
  onProgress?.("download " + label, 0);
  execSync(`curl -fSL -o "${dest}" "${url}"`, { stdio: "inherit" });
}

/** Download SCOWL + SimpleWiki titles and return a deduplicated word list. */
export function fetchWordList(
  cacheDir: string,
  scowlFilter: RegExp,
  onProgress?: ProgressFn,
): string[] {
  fs.mkdirSync(cacheDir, { recursive: true });

  const scowlTarball = path.join(cacheDir, "scowl.tar.gz");
  const scowlDir = path.join(cacheDir, "scowl-2020.12.07");
  const wikiGz = path.join(cacheDir, "simplewiki-titles.gz");

  download(SCOWL_URL, scowlTarball, "SCOWL wordlists", onProgress);
  download(WIKI_TITLES_URL, wikiGz, "SimpleWiki titles", onProgress);

  if (!fs.existsSync(scowlDir)) {
    onProgress?.("extract SCOWL", 0);
    execSync(`tar xzf "${scowlTarball}" -C "${cacheDir}"`, { stdio: "inherit" });
  }

  const wordSet = new Set<string>();
  const scowlFinal = path.join(scowlDir, "final");
  if (fs.existsSync(scowlFinal)) {
    const files = fs.readdirSync(scowlFinal).filter((f) => scowlFilter.test(f));
    for (const f of files) {
      const content = fs.readFileSync(path.join(scowlFinal, f), "utf-8");
      for (const line of content.split("\n")) {
        const w = line.trim().toLowerCase();
        if (WORD_RE.test(w)) wordSet.add(w);
      }
    }
  }
  onProgress?.("scowl words", wordSet.size);

  const wikiGzBuf = fs.readFileSync(wikiGz);
  const wikiText = gunzipSync(wikiGzBuf).toString("utf-8");
  for (const line of wikiText.split("\n")) {
    for (const part of line.trim().split("_")) {
      const w = part.toLowerCase();
      if (WORD_RE.test(w)) wordSet.add(w);
    }
  }
  onProgress?.("total words", wordSet.size);

  return Array.from(wordSet);
}

/** Download a bz2 dump file. Returns path to the cached bz2. */
export function downloadDump(
  cacheDir: string,
  url: string,
  filename: string,
  label: string,
  onProgress?: ProgressFn,
): string {
  fs.mkdirSync(cacheDir, { recursive: true });
  const bz2Path = path.join(cacheDir, filename);
  download(url, bz2Path, label, onProgress);
  return bz2Path;
}

export function downloadSimpleWiki(cacheDir: string, onProgress?: ProgressFn): string {
  return downloadDump(cacheDir, SIMPLEWIKI_URL, "simplewiki-articles.xml.bz2", "SimpleWiki articles (~330MB)", onProgress);
}

export function downloadWiktionary(cacheDir: string, onProgress?: ProgressFn): string {
  return downloadDump(cacheDir, WIKTIONARY_URL, "enwiktionary-articles.xml.bz2", "Wiktionary (~800MB)", onProgress);
}

export function downloadEnWiki(cacheDir: string, onProgress?: ProgressFn): string {
  return downloadDump(cacheDir, ENWIKI_URL, "enwiki-articles.xml.bz2", "English Wikipedia (~22GB)", onProgress);
}

/** Strip MediaWiki markup from raw wikitext, returning clean plain text. */
export function stripWikitext(raw: string): string {
  let text = raw;
  text = text.replace(/<!--[\s\S]*?-->/g, "");
  for (let i = 0; i < 5; i++) {
    text = text.replace(/\{\{[^{}]*\}\}/g, "");
  }
  text = text.replace(/\{\|[\s\S]*?\|\}/g, "");
  text = text.replace(/\[\[(Category|File|Image|Media):[^\]]*\]\]/gi, "");
  text = text.replace(/\[\[[^\]|]*\|([^\]]*)\]\]/g, "$1");
  text = text.replace(/\[\[([^\]]*)\]\]/g, "$1");
  text = text.replace(/\[https?:\/\/[^\s\]]*\s*([^\]]*)\]/g, "$1");
  text = text.replace(/\[https?:\/\/[^\]]*\]/g, "");
  text = text.replace(/<ref[^>]*\/>/gi, "");
  text = text.replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi, "");
  text = text.replace(/<[^>]+>/g, "");
  text = text.replace(/^=+\s*(.*?)\s*=+$/gm, "$1");
  text = text.replace(/'{2,5}/g, "");
  text = text.replace(/^[*#:;]+\s*/gm, "");
  text = text.replace(/__[A-Z]+__/g, "");
  text = text.replace(/\n{3,}/g, "\n\n");
  text = text.replace(/[ \t]+/g, " ");
  return text.trim();
}

/**
 * Core MediaWiki XML streaming parser. Reads lines from a readline interface,
 * extracts article text, strips wikitext markup, and calls onArticle.
 * Returns the set of target words found in the text.
 */
async function parseMediaWikiStream(
  rl: readline.Interface,
  targetWords: Set<string>,
  onArticle: (text: string) => void,
  onProgress: ProgressFn | undefined,
  label: string,
): Promise<Set<string>> {
  const covered = new Set<string>();
  let articleCount = 0;
  let inText = false;
  let textLines: string[] = [];

  for await (const line of rl) {
    const textStart = line.match(/<text[^>]*>([\s\S]*)/);
    if (textStart) {
      const rest = textStart[1];
      const endIdx = rest.indexOf("</text>");
      if (endIdx >= 0) {
        processArticle(rest.substring(0, endIdx));
      } else {
        inText = true;
        textLines = [rest];
      }
      continue;
    }

    if (inText) {
      const endIdx = line.indexOf("</text>");
      if (endIdx >= 0) {
        textLines.push(line.substring(0, endIdx));
        processArticle(textLines.join("\n"));
        textLines = [];
        inText = false;
      } else {
        textLines.push(line);
      }
    }
  }

  function processArticle(raw: string): void {
    if (raw.startsWith("#REDIRECT") || raw.startsWith("#redirect")) return;

    const clean = stripWikitext(raw);
    if (clean.length < 80) return;

    onArticle(clean);
    articleCount++;

    for (const w of clean.split(/[^a-zA-Z]+/)) {
      const lower = w.toLowerCase();
      if (lower.length >= 2 && targetWords.has(lower)) {
        covered.add(lower);
      }
    }

    if (articleCount % 10000 === 0) {
      onProgress?.(label, articleCount);
    }
  }

  onProgress?.(`${label} done`, articleCount);
  return covered;
}

// ── Project Gutenberg ──

const GUTENBERG_URL = "https://www.gutenberg.org/cache/epub";

/** Download Project Gutenberg books in parallel. Returns path to cache dir. */
export async function downloadGutenberg(
  cacheDir: string,
  maxBooks: number,
  onProgress?: ProgressFn,
): Promise<string> {
  const dir = path.join(cacheDir, "gutenberg");
  fs.mkdirSync(dir, { recursive: true });

  const cached = fs.readdirSync(dir).filter((f) => f.endsWith(".txt"));
  if (cached.length >= maxBooks) {
    onProgress?.("cache hit Gutenberg", cached.length);
    return dir;
  }

  const cachedIds = new Set(cached.map((f) => parseInt(f.replace(".txt", ""), 10)));
  const needed = maxBooks - cached.length;

  // Build candidate IDs (try 2x what we need to account for missing books)
  const candidates: number[] = [];
  for (let id = 1; id < 70000 && candidates.length < needed * 2; id++) {
    if (!cachedIds.has(id)) candidates.push(id);
  }

  onProgress?.(`downloading ~${needed} Gutenberg books`, cached.length);

  // Write candidate IDs and download 10 in parallel via xargs
  const idFile = path.join(dir, ".fetch-ids");
  fs.writeFileSync(idFile, candidates.join("\n"));

  const script =
    `cat "${idFile}" | xargs -P 10 -I{} sh -c '` +
    `curl -fsSL --max-time 20 -o "${dir}/{}.txt" ` +
    `"${GUTENBERG_URL}/{}/pg{}.txt" 2>/dev/null || rm -f "${dir}/{}.txt"'`;

  const proc = spawn("sh", ["-c", script], { stdio: "inherit" });
  await new Promise<void>((resolve) => {
    if (proc.exitCode !== null) resolve();
    else proc.on("exit", () => resolve());
  });

  // Clean up tiny/empty files (failed downloads, non-English, etc.)
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith(".txt")) continue;
    const fp = path.join(dir, f);
    if (fs.statSync(fp).size < 1000) fs.unlinkSync(fp);
  }

  try { fs.unlinkSync(idFile); } catch {}
  const total = fs.readdirSync(dir).filter((f) => f.endsWith(".txt")).length;
  onProgress?.("gutenberg books downloaded", total);
  return dir;
}

/** Strip Project Gutenberg header/footer boilerplate from a book. */
export function stripGutenbergBoilerplate(raw: string): string {
  const startMarkers = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
  ];
  const endMarkers = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
    "End of the Project Gutenberg EBook",
    "End of Project Gutenberg's",
  ];

  let startIdx = -1;
  for (const marker of startMarkers) {
    const idx = raw.indexOf(marker);
    if (idx >= 0) {
      startIdx = raw.indexOf("\n", idx);
      break;
    }
  }

  let endIdx = raw.length;
  for (const marker of endMarkers) {
    const idx = raw.indexOf(marker);
    if (idx >= 0) {
      endIdx = idx;
      break;
    }
  }

  let text = raw.substring(startIdx >= 0 ? startIdx + 1 : 0, endIdx);
  text = text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+/g, " ");

  // Strip residual boilerplate that slips past the header/footer markers
  const lines = text.split("\n");
  const filtered = lines.filter((line) => {
    const trimmed = line.trim();
    // Triple-star annotations (*** Transcribers' Notes, etc.)
    if (/^\*\*\*\s/.test(trimmed)) return false;
    // Standalone bracketed editorial notes like [RO, Aug 2025: ...]
    if (/^\[.{0,60}\]$/.test(trimmed)) return false;
    // Editorial brackets at line start: [Transcriber, [Editor, [Illustration, [Pg, [Page, [XX,
    if (/^\[([A-Z]{1,3},|Transcriber|Editor|Illustration|Pg |Page )/.test(trimmed)) return false;
    return true;
  });

  return filtered.join("\n").trim();
}

/**
 * Process cached Gutenberg books for vocabulary coverage.
 * Reads plain text files, strips boilerplate, writes clean text.
 */
export function streamGutenberg(
  gutenbergDir: string,
  targetWords: Set<string>,
  onArticle: (text: string) => void,
  onProgress?: ProgressFn,
): Set<string> {
  const covered = new Set<string>();
  const files = fs
    .readdirSync(gutenbergDir)
    .filter((f) => f.endsWith(".txt"))
    .sort((a, b) => parseInt(a) - parseInt(b));

  let count = 0;
  for (const f of files) {
    const raw = fs.readFileSync(path.join(gutenbergDir, f), "utf-8");
    const clean = stripGutenbergBoilerplate(raw);
    if (clean.length < 200) continue;

    onArticle(clean);
    count++;

    for (const w of clean.split(/[^a-zA-Z]+/)) {
      const lower = w.toLowerCase();
      if (lower.length >= 2 && targetWords.has(lower)) {
        covered.add(lower);
      }
    }

    if (count % 500 === 0) {
      onProgress?.("gutenberg", count);
    }
  }

  onProgress?.("gutenberg done", count);
  return covered;
}

/**
 * Stream a bz2-compressed MediaWiki dump via bzcat pipe.
 * Parses articles, writes clean text via onArticle, tracks word coverage.
 */
export async function streamBz2Dump(
  bz2Path: string,
  targetWords: Set<string>,
  onArticle: (text: string) => void,
  onProgress?: ProgressFn,
  label: string = "dump",
): Promise<Set<string>> {
  const proc = spawn("bzcat", [bz2Path], { stdio: ["ignore", "pipe", "inherit"] });
  proc.stdout!.setEncoding("utf-8");
  const rl = readline.createInterface({ input: proc.stdout!, crlfDelay: Infinity });

  const covered = await parseMediaWikiStream(rl, targetWords, onArticle, onProgress, label);

  // Wait for bzcat to finish
  const exitCode = await new Promise<number | null>((resolve) => {
    if (proc.exitCode !== null) resolve(proc.exitCode);
    else proc.on("exit", (code) => resolve(code));
  });

  if (exitCode !== 0 && exitCode !== null) {
    throw new Error(`bzcat exited with code ${exitCode}`);
  }

  return covered;
}
