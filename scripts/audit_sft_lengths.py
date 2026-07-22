#!/usr/bin/env python3
"""Exact batched tokenizer-length audit for an Alpha SFT corpus."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer


def percentile(sorted_values, fraction):
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def stats(values):
    ordered = sorted(values)
    return {
        "rows": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")

    manifest = json.loads(args.manifest.read_text())
    max_tokens = manifest["max_tokens"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    lengths = []
    digest = hashlib.sha256()
    total_bytes = 0
    batch = []

    def flush():
        if batch:
            lengths.extend(len(encoding.ids) for encoding in tokenizer.encode_batch(batch))
            batch.clear()

    with args.data.open("rb") as source:
        for raw_line in source:
            digest.update(raw_line)
            total_bytes += len(raw_line)
            batch.append(raw_line.removesuffix(b"\n").decode("utf-8"))
            if len(batch) == args.batch_size:
                flush()
    flush()

    output = manifest["output"]
    sha256 = digest.hexdigest()
    if len(lengths) != manifest["total"]:
        raise RuntimeError(f"rows {len(lengths)} != manifest {manifest['total']}")
    if total_bytes != output["bytes"]:
        raise RuntimeError(f"bytes {total_bytes} != manifest {output['bytes']}")
    if sha256 != output["sha256"]:
        raise RuntimeError(f"sha256 {sha256} != manifest {output['sha256']}")
    if not lengths or max(lengths) > max_tokens:
        raise RuntimeError(f"token bound failed: max={max(lengths, default=0)}, bound={max_tokens}")

    source_stats = []
    for span in manifest["source_spans"]:
        subset = lengths[span["start_line"] - 1:span["end_line"]]
        source_stats.append({"source": span["source"], **stats(subset)})
    report = {
        "schema": "alpha-sft-length-audit-v1",
        "result": "PASS",
        "corpus": {"path": str(args.data), "rows": len(lengths), "bytes": total_bytes, "sha256": sha256},
        "tokenizer": {"path": str(args.tokenizer), "vocab_size": tokenizer.get_vocab_size()},
        "token_bound": max_tokens,
        "rows_over_bound": sum(length > max_tokens for length in lengths),
        "rows_at_bound": sum(length == max_tokens for length in lengths),
        "overall": stats(lengths),
        "sources": source_stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
