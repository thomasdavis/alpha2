#!/usr/bin/env python3
"""Build immutable Smol-SmolTalk train/test text for Alpha chat recipe V12.

This reuses Alpha's established chat renderer while limiting the source to the
exact public Smol-SmolTalk train and test parquets. Training order is preserved
because the upstream rows are already highly source-interleaved.
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from build_sft_corpus import convo_from_messages, file_record, render_batch


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_split(
    *,
    name: str,
    paths: list[str],
    output: Path,
    tokenizer: Tokenizer,
    max_chars: int,
    max_tokens: int,
    excluded_hashes: set[str] | None = None,
) -> tuple[dict, set[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    seen: set[str] = set()
    accepted_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    rendered_token_total = 0

    with tmp.open("w", encoding="utf-8") as sink:
        for parquet_path in paths:
            parquet = pq.ParquetFile(parquet_path)
            available = set(parquet.schema.names)
            columns = ["messages"] + (["source"] if "source" in available else [])
            for batch in parquet.iter_batches(batch_size=1024, columns=columns):
                rows = batch.to_pylist()
                counters["raw_rows"] += len(rows)
                conversations = [convo_from_messages(row["messages"]) for row in rows]
                rendered = render_batch(conversations, max_chars, tokenizer, max_tokens)
                valid_lines = [item[0] for item in rendered if item is not None]
                encoded = tokenizer.encode_batch(valid_lines) if valid_lines else []
                encoded_index = 0
                for row, conversation, item in zip(rows, conversations, rendered):
                    source = str(row.get("source") or "unknown")
                    if conversation is None or item is None:
                        counters["invalid_or_overlong"] += 1
                        continue
                    line, trimmed = item
                    token_count = len(encoded[encoded_index].ids)
                    encoded_index += 1
                    digest = sha256_text(line)
                    if digest in seen:
                        counters["within_split_duplicates"] += 1
                        continue
                    seen.add(digest)
                    if excluded_hashes is not None and digest in excluded_hashes:
                        counters["train_overlap_excluded"] += 1
                        continue
                    sink.write(line + "\n")
                    accepted_hashes.add(digest)
                    source_counts[source] += 1
                    counters["accepted"] += 1
                    counters["prefix_trimmed"] += int(trimmed)
                    rendered_token_total += token_count

    os.replace(tmp, output)
    report = {
        "name": name,
        "output": file_record(str(output)),
        "inputs": [file_record(path) for path in paths],
        "counts": dict(sorted(counters.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "unique_rendering_hashes": len(accepted_hashes),
        "sum_per_conversation_tokens": rendered_token_total,
        "max_chars": max_chars,
        "max_tokens": max_tokens,
    }
    return report, accepted_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--test-glob", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")
    train_paths = sorted(glob.glob(args.train_glob))
    test_paths = sorted(glob.glob(args.test_glob))
    if not train_paths:
        parser.error(f"no train parquets matched {args.train_glob}")
    if not test_paths:
        parser.error(f"no test parquets matched {args.test_glob}")
    overlap = set(train_paths) & set(test_paths)
    if overlap:
        parser.error(f"train/test path overlap: {sorted(overlap)}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "smol-smoltalk-train.txt"
    test_path = args.output_dir / "smol-smoltalk-test.txt"
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    for path in (train_path, test_path, manifest_path):
        if path.exists():
            parser.error(f"refusing to overwrite existing artifact: {path}")

    train, train_hashes = build_split(
        name="train",
        paths=train_paths,
        output=train_path,
        tokenizer=tokenizer,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
    )
    test, test_hashes = build_split(
        name="test",
        paths=test_paths,
        output=test_path,
        tokenizer=tokenizer,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        excluded_hashes=train_hashes,
    )
    remaining_overlap = train_hashes & test_hashes
    if remaining_overlap:
        raise RuntimeError(f"post-build train/test overlap: {len(remaining_overlap)}")

    manifest = {
        "schema": "alpha-chat-recipe-v12-corpus-v1",
        "result": "PASS",
        "rendering": {
            "format": "<|user|> ... <|assistant|> ... <|end_of_text|>",
            "system_policy": "fold_into_first_user",
            "order": "preserve_upstream",
            "deduplication": "sha256_exact_rendered_conversation",
            "test_overlap_policy": "exclude_exact_train_rendering",
        },
        "tokenizer": file_record(str(args.tokenizer)),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train": train,
        "test": test,
        "post_build_train_test_overlap": len(remaining_overlap),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
