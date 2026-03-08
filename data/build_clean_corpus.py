#!/usr/bin/env python3
"""
Build a clean English prose training corpus from available text files.
Filters out non-prose content (HTML, markup, references, short lines,
non-English text, data tables, etc.)
"""

import os
import re
import sys
from pathlib import Path

DATA_DIR = Path("/home/ajax/repos/models/alpha/data")
OUTPUT_FILE = DATA_DIR / "fine_corpus_clean.txt"

# Precompile all regex patterns
RE_HTML_TAG = re.compile(r'<[^>]{2,}>')
RE_HTML_ENTITY = re.compile(r'&[a-z]+;|&#\d+;|&lt;|&gt;|&amp;|&quot;')
RE_WIKI_MARKUP = re.compile(r'\|\s*class\s*=|colspan|rowspan|\{\{|\}\}|\[\[|\]\]|\{\||\|\}')
RE_URL = re.compile(r'https?://\S+|www\.\S+|\.org\b|\.com\b|\.net\b|\.edu\b')
RE_REFERENCE = re.compile(r'<ref|</ref>|DOI:|doi:|ISBN:|ISSN:|PMID:|arXiv:')
RE_SPECIAL_START = re.compile(r'^[@#\|!{}\[\]*=+]')
RE_CHESS_NOTATION = re.compile(r'^[0-9rnbqkpRNBQKP/]+\s+[bw]\s+-\s+-')
RE_CODE_LIKE = re.compile(r'[{}();=].*[{}();=]')
RE_NON_LATIN = re.compile(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u0600-\u06FF\u0900-\u097F\u00C0-\u024F]{3,}')
RE_MOSTLY_NON_ASCII = re.compile(r'[^\x00-\x7F]')
RE_ALL_CAPS_LINE = re.compile(r'^[A-Z\s\d\-:,.]{40,}$')
RE_LIST_ENTRY = re.compile(r'^\s*[-*]\s+\S+\s*$')  # single-word list items
RE_CITATION = re.compile(r'^\s*(Baum|Wikipedia|World Conservation|Available online|Access on|Retrieved|Archived)', re.IGNORECASE)
RE_SEPARATOR = re.compile(r'^[\s\-=_*#~]{3,}$')
RE_DATA_LINE = re.compile(r'^\s*\{.*\}\s*,?\s*$')  # JSON-like lines
RE_TABLE_LINE = re.compile(r'\|\s*\w+\s*\|')  # pipe-delimited tables


def is_clean_prose(line: str) -> bool:
    """Return True if line looks like clean English prose."""
    stripped = line.strip()

    # Skip empty/very short lines (but keep them as paragraph breaks)
    if len(stripped) == 0:
        return True  # Keep blank lines for paragraph structure

    if len(stripped) < 20:
        return False

    # Skip lines with HTML tags or entities
    if RE_HTML_TAG.search(stripped):
        return False
    if RE_HTML_ENTITY.search(stripped):
        return False

    # Skip wiki markup
    if RE_WIKI_MARKUP.search(stripped):
        return False

    # Skip URLs
    if RE_URL.search(stripped):
        return False

    # Skip references
    if RE_REFERENCE.search(stripped):
        return False

    # Skip lines starting with special characters
    if RE_SPECIAL_START.match(stripped):
        return False

    # Skip chess notation
    if RE_CHESS_NOTATION.match(stripped):
        return False

    # Skip code-like lines
    if RE_CODE_LIKE.search(stripped) and len(stripped) < 100:
        return False

    # Skip lines with significant non-Latin characters
    non_ascii_count = len(RE_MOSTLY_NON_ASCII.findall(stripped))
    if non_ascii_count > len(stripped) * 0.3:
        return False

    # Skip non-Latin script clusters (German umlauts are fine, but CJK/Cyrillic blocks aren't)
    if RE_NON_LATIN.search(stripped):
        return False

    # Skip all-caps lines (headers/titles > 40 chars)
    if RE_ALL_CAPS_LINE.match(stripped):
        return False

    # Skip single-word list entries
    if RE_LIST_ENTRY.match(stripped):
        return False

    # Skip citation/reference lines
    if RE_CITATION.match(stripped):
        return False

    # Skip separator lines
    if RE_SEPARATOR.match(stripped):
        return False

    # Skip data/JSON lines
    if RE_DATA_LINE.match(stripped):
        return False

    # Skip pipe-delimited table lines
    if RE_TABLE_LINE.search(stripped):
        return False

    # Skip lines that are mostly numbers/punctuation
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count < len(stripped) * 0.4:
        return False

    # Must have at least some spaces (prose has words)
    if stripped.count(' ') < 2:
        return False

    return True


def clean_line(line: str) -> str:
    """Light cleanup of a prose line."""
    # Remove leading/trailing whitespace but preserve content
    return line.strip()


def process_file(filepath: str, max_lines: int = None, skip_lines: int = 0) -> list:
    """Process a single file and return clean lines."""
    clean = []
    consecutive_blanks = 0
    lines_read = 0

    print(f"  Processing {filepath}...", flush=True)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if max_lines and lines_read >= max_lines:
                break
            lines_read += 1

            if is_clean_prose(line):
                cleaned = clean_line(line)
                if cleaned == '':
                    consecutive_blanks += 1
                    if consecutive_blanks <= 2:  # Max 2 consecutive blank lines
                        clean.append('')
                else:
                    consecutive_blanks = 0
                    clean.append(cleaned)

    # Remove trailing blanks
    while clean and clean[-1] == '':
        clean.pop()

    print(f"    -> {len(clean)} clean lines from {lines_read} total", flush=True)
    return clean


def process_books_novels_dir(dirpath: str) -> list:
    """Process all .txt files in a directory."""
    clean = []
    if not os.path.isdir(dirpath):
        return clean

    for fname in sorted(os.listdir(dirpath)):
        if fname.endswith('.txt'):
            filepath = os.path.join(dirpath, fname)
            lines = process_file(filepath)
            if lines:
                clean.extend(lines)
                clean.append('')  # Separator between files

    return clean


def main():
    all_clean = []

    print("=== Building clean English prose corpus ===\n", flush=True)

    # 1. fine_corpus.txt - high quality, light cleaning needed
    print("[1/7] fine_corpus.txt (6.2MB - clean prose articles)", flush=True)
    lines = process_file(str(DATA_DIR / "fine_corpus.txt"))
    all_clean.extend(lines)
    all_clean.append('')

    # 2. concordance-v2.txt - best concordance, mixed content
    print("[2/6] concordance-v2.txt (788MB - encyclopedia/prose mix)", flush=True)
    lines = process_file(str(DATA_DIR / "concordance-v2.txt"))
    all_clean.extend(lines)
    all_clean.append('')

    # 3. test-gutenberg.txt - large Gutenberg collection (superset of concordance-gutenberg-first.txt)
    print("[3/6] test-gutenberg.txt (781MB - Gutenberg prose)", flush=True)
    lines = process_file(str(DATA_DIR / "test-gutenberg.txt"))
    all_clean.extend(lines)
    all_clean.append('')

    # 4. books_all.txt
    print("[4/6] books_all.txt (5.2MB - book-style content)", flush=True)
    lines = process_file(str(DATA_DIR / "books_all.txt"))
    all_clean.extend(lines)
    all_clean.append('')

    # 5. novels_all.txt
    print("[5/6] novels_all.txt (1.7MB - novel-style content)", flush=True)
    lines = process_file(str(DATA_DIR / "novels_all.txt"))
    all_clean.extend(lines)
    all_clean.append('')

    # 6. Individual books and novels from subdirectories
    print("[6/6] books/ and novels/ directories", flush=True)
    lines = process_books_novels_dir(str(DATA_DIR / "books"))
    all_clean.extend(lines)
    lines = process_books_novels_dir(str(DATA_DIR / "novels"))
    all_clean.extend(lines)

    # Write output
    print(f"\nWriting {len(all_clean)} lines to {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in all_clean:
            f.write(line + '\n')

    # Stats
    file_size = os.path.getsize(OUTPUT_FILE)
    size_mb = file_size / (1024 * 1024)

    # Rough token estimate: ~4 chars per token for English text
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        total_chars = sum(len(line) for line in f)
    estimated_tokens = total_chars / 4

    print(f"\n=== RESULTS ===", flush=True)
    print(f"Output file: {OUTPUT_FILE}", flush=True)
    print(f"File size: {size_mb:.1f} MB", flush=True)
    print(f"Total lines: {len(all_clean):,}", flush=True)
    print(f"Total characters: {total_chars:,}", flush=True)
    print(f"Estimated tokens: {estimated_tokens:,.0f} (~{estimated_tokens/1e6:.1f}M)", flush=True)


if __name__ == '__main__':
    main()
