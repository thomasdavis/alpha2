#!/usr/bin/env python3
"""Convert HF premix parquet shards to Alpha's flat-text pretraining format.

Input:  parquet shards of HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
        (schema: text, id, url, dataset), downloaded via `hf download`.
Output: sharded UTF-8 .txt files where each document ends with <|end_of_text|>\n —
        the delimiter Alpha's trainer uses for doc-aware train/val splitting and
        the tokenizer treats as an atomic reserved token.

Alpha's loader constraints honored here:
  - loadAndTokenize chunks files >30MB at 10MB boundaries cut at newlines, so we
    normalize away bare newlines inside documents (paragraphs joined with a single
    \n is fine — we just avoid \r and NUL) and keep shards <= --shard-mb.
  - Output is plain text, no JSON, one document then the delimiter.

Usage:
  python scripts/build_pretrain_corpus.py \
    --src /mnt/donto-data/alpha-corpora/premix-shuffled/data \
    --out /mnt/donto-data/alpha-corpora/pretrain-text \
    --target-tokens 1.5e9 [--shard-mb 1900] [--min-chars 200] [--max-chars 100000]

Token accounting uses the ~3.9 chars/token heuristic for byte-BPE at 12k vocab;
the real count is measured after tokenizer build (GOAL.md Stage 4 gate).
"""
import argparse
import glob
import os
import sys

import pyarrow.parquet as pq

END = "<|end_of_text|>\n"
CHARS_PER_TOKEN = 3.9


def clean(text: str) -> str:
    # Strip NUL and carriage returns; collapse >2 consecutive newlines.
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing train-*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=float, required=True)
    ap.add_argument("--shard-mb", type=int, default=1900, help="max shard size (stay < 2GB for safety)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=100_000)
    args = ap.parse_args()

    target_chars = int(args.target_tokens * CHARS_PER_TOKEN)
    shard_bytes_cap = args.shard_mb * 1024 * 1024
    os.makedirs(args.out, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.src, "train-*.parquet")))
    if not shards:
        print(f"no parquet shards in {args.src}", file=sys.stderr)
        return 1

    total_chars = 0
    n_docs = 0
    n_skipped = 0
    shard_idx = 0
    shard_written = 0
    out_f = open(os.path.join(args.out, f"pretrain-{shard_idx:03d}.txt"), "w", encoding="utf-8")
    manifest = open(os.path.join(args.out, "MANIFEST.txt"), "w", encoding="utf-8")
    manifest.write(f"source_shards: {[os.path.basename(s) for s in shards]}\n")

    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for v in batch.column("text"):
                text = clean(v.as_py() or "")
                if len(text) < args.min_chars or len(text) > args.max_chars:
                    n_skipped += 1
                    continue
                doc = text + "\n" + END
                out_f.write(doc)
                shard_written += len(doc.encode("utf-8", "ignore"))
                total_chars += len(doc)
                n_docs += 1
                if shard_written >= shard_bytes_cap:
                    out_f.close()
                    print(f"shard {shard_idx:03d} done ({shard_written/1e9:.2f} GB), "
                          f"{n_docs} docs, ~{total_chars/CHARS_PER_TOKEN/1e9:.2f}B tokens so far", flush=True)
                    shard_idx += 1
                    shard_written = 0
                    out_f = open(os.path.join(args.out, f"pretrain-{shard_idx:03d}.txt"), "w", encoding="utf-8")
                if total_chars >= target_chars:
                    break
            if total_chars >= target_chars:
                break
        if total_chars >= target_chars:
            break

    out_f.close()
    est_tokens = total_chars / CHARS_PER_TOKEN
    summary = (f"docs={n_docs} skipped={n_skipped} chars={total_chars} "
               f"est_tokens={est_tokens/1e9:.3f}B shards={shard_idx + 1}")
    manifest.write(summary + "\n")
    manifest.close()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
