/**
 * Generate the "chaos" domain dataset — maximize unique English word coverage.
 *
 * Downloads two small sources:
 * - SCOWL wordlists (~3MB tarball) from SourceForge
 * - SimpleWiki titles (~2MB gzipped) from Wikimedia
 *
 * Then generates a synthetic corpus where every word appears in short sentence
 * templates, producing a ~15-25MB training file.
 *
 * Usage: npx tsx scripts/gen-chaos.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import { gunzipSync } from "node:zlib";

// ── Seeded RNG (deterministic output) ─────────────────────────────────────

let seed = 42;
function rand(): number {
  seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function randInt(min: number, max: number): number {
  return Math.floor(rand() * (max - min + 1)) + min;
}
function pick<T>(arr: T[]): T {
  return arr[randInt(0, arr.length - 1)];
}

// Fisher-Yates shuffle (in-place, seeded)
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = randInt(0, i);
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

// ── Paths ──────────────────────────────────────────────────────────────────

const CACHE_DIR = "data/.cache";
const OUT_PATH = "data/chaos.txt";

const SCOWL_URL =
  "https://sourceforge.net/projects/wordlist/files/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz/download";
const SCOWL_TARBALL = path.join(CACHE_DIR, "scowl.tar.gz");
const SCOWL_DIR = path.join(CACHE_DIR, "scowl-2020.12.07");

const WIKI_URL =
  "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-all-titles-in-ns0.gz";
const WIKI_GZ = path.join(CACHE_DIR, "simplewiki-titles.gz");

// ── Phase 1: Download ──────────────────────────────────────────────────────

fs.mkdirSync(CACHE_DIR, { recursive: true });

function download(url: string, dest: string, label: string): void {
  if (fs.existsSync(dest)) {
    console.log(`[cache hit] ${label} → ${dest}`);
    return;
  }
  console.log(`[download] ${label} → ${dest}`);
  execSync(`curl -fSL -o "${dest}" "${url}"`, { stdio: "inherit" });
}

download(SCOWL_URL, SCOWL_TARBALL, "SCOWL wordlists");
download(WIKI_URL, WIKI_GZ, "SimpleWiki titles");

// Extract SCOWL if needed
if (!fs.existsSync(SCOWL_DIR)) {
  console.log("[extract] SCOWL tarball...");
  execSync(`tar xzf "${SCOWL_TARBALL}" -C "${CACHE_DIR}"`, { stdio: "inherit" });
}

// ── Phase 2: Extract & dedupe words ────────────────────────────────────────

const wordSet = new Set<string>();
const WORD_RE = /^[a-zA-Z]{2,30}$/;

// Parse SCOWL final/* files
const scowlFinal = path.join(SCOWL_DIR, "final");
if (fs.existsSync(scowlFinal)) {
  const files = fs.readdirSync(scowlFinal);
  for (const f of files) {
    const content = fs.readFileSync(path.join(scowlFinal, f), "utf-8");
    for (const line of content.split("\n")) {
      const w = line.trim().toLowerCase();
      if (WORD_RE.test(w)) wordSet.add(w);
    }
  }
}
console.log(`[scowl] ${wordSet.size} words after SCOWL`);

// Parse SimpleWiki titles
const wikiGzBuf = fs.readFileSync(WIKI_GZ);
const wikiText = gunzipSync(wikiGzBuf).toString("utf-8");
for (const line of wikiText.split("\n")) {
  const title = line.trim();
  // Split on underscores (wiki title format)
  for (const part of title.split("_")) {
    const w = part.toLowerCase();
    if (WORD_RE.test(w)) wordSet.add(w);
  }
}
console.log(`[wiki] ${wordSet.size} words after SimpleWiki titles`);

const words = shuffle(Array.from(wordSet));
console.log(`[total] ${words.length} unique words`);

// ── Phase 3: Generate synthetic corpus ─────────────────────────────────────

// Sentence templates — {w} and {w2} are replaced with random words
const TEMPLATES = [
  "The {w} was quite remarkable.",
  "She found a {w} near the {w2}.",
  "Unlike {w}, the {w2} is different.",
  "Every {w} has a hidden {w2} inside.",
  "They discovered that {w} changes everything.",
  "A {w} appeared beside the old {w2}.",
  "Nothing compares to a genuine {w}.",
  "The concept of {w} puzzled the {w2} expert.",
  "Between {w} and {w2}, he chose neither.",
  "Without {w}, the {w2} would be lost.",
  "People often confuse {w} with {w2}.",
  "The ancient {w} revealed a secret {w2}.",
  "Somewhere beyond the {w}, a {w2} waited.",
  "He called it {w} but she preferred {w2}.",
  "In the realm of {w}, only {w2} matters.",
];

function fillTemplate(template: string, w: string, w2: string): string {
  return template.replace("{w}", w).replace("{w2}", w2);
}

fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
const fd = fs.openSync(OUT_PATH, "w");

let totalLines = 0;
let linesSinceParagraph = 0;

function writeLine(line: string): void {
  fs.writeSync(fd, line + "\n");
  totalLines++;
  linesSinceParagraph++;
}

// Pass 1: Every word appears at least once as {w}
console.log("[gen] Pass 1: primary word coverage...");
for (let i = 0; i < words.length; i++) {
  const w = words[i];
  const w2 = words[(i + 1) % words.length];
  const template = TEMPLATES[i % TEMPLATES.length];
  writeLine(fillTemplate(template, w, w2));

  // Paragraph breaks
  if (linesSinceParagraph >= randInt(5, 10)) {
    writeLine("");
    linesSinceParagraph = 0;
  }
}

// Pass 2: Every word appears at least once more in a different template
console.log("[gen] Pass 2: secondary word coverage...");
const shifted = [...words];
// Rotate by a large prime-ish offset so pairings differ from pass 1
const offset = Math.floor(words.length * 0.37);
for (let i = 0; i < shifted.length; i++) {
  const w = shifted[i];
  const w2 = shifted[(i + offset) % shifted.length];
  // Use a different template set offset
  const template = TEMPLATES[(i + 7) % TEMPLATES.length];
  writeLine(fillTemplate(template, w, w2));

  if (linesSinceParagraph >= randInt(5, 10)) {
    writeLine("");
    linesSinceParagraph = 0;
  }
}

fs.closeSync(fd);

const stat = fs.statSync(OUT_PATH);
const sizeMB = (stat.size / (1024 * 1024)).toFixed(1);
console.log(`\nGenerated ${totalLines} lines → ${OUT_PATH} (${sizeMB} MB)`);
console.log(`Unique words: ${words.length}`);
