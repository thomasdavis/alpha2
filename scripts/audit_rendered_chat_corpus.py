#!/usr/bin/env python3
"""Audit rendered Alpha chat data by source, answer length, and repetition.

The catalog is expected to use the deterministic layout produced by
build_chat_repair_corpus.py: catalog rows are grouped by split and retain the
same order as the corresponding rendered split file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from tokenizers import Tokenizer


USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end_of_text|>"
ROLE = re.compile(r"(<\|user\|>|<\|assistant\|>)")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def percentile(values: list[int] | list[float], fraction: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return float(ordered[int((len(ordered) - 1) * fraction)])


def repeat_rate(tokens: list[int], n: int = 4) -> float:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for index in range(total):
        gram = tuple(tokens[index : index + n])
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / total


def parse_assistant_turns(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.endswith(END):
        raise ValueError("rendered conversation does not end in EOS")
    parts = ROLE.split(stripped[: -len(END)].strip())
    assistants: list[str] = []
    for index in range(1, len(parts), 2):
        if parts[index] == ASSISTANT:
            content = parts[index + 1].strip() if index + 1 < len(parts) else ""
            if not content:
                raise ValueError("empty assistant turn")
            assistants.append(content)
    if not assistants:
        raise ValueError("conversation has no assistant turn")
    return assistants


def summary(records: list[dict[str, object]], duplicate_counts: Counter[str]) -> dict[str, object]:
    answer_lengths = [int(record["answer_tokens"]) for record in records]
    repeat_rates = [float(record["repeat_rate"]) for record in records]
    conversations = {str(record["conversation_sha256"]) for record in records}
    exact_duplicate_answers = sum(max(0, count - 1) for count in duplicate_counts.values())
    prefix_counts = Counter(str(record["first_four_tokens"]) for record in records)
    return {
        "conversations": len(conversations),
        "assistant_turns": len(records),
        "assistant_tokens": sum(answer_lengths),
        "answer_tokens": {
            "min": min(answer_lengths, default=0),
            "p25": percentile(answer_lengths, 0.25),
            "median": percentile(answer_lengths, 0.5),
            "p75": percentile(answer_lengths, 0.75),
            "p95": percentile(answer_lengths, 0.95),
            "max": max(answer_lengths, default=0),
            "mean": mean(answer_lengths) if answer_lengths else 0,
        },
        "answer_repetition": {
            "mean_four_gram_repeat_rate": mean(repeat_rates) if repeat_rates else 0,
            "p95_four_gram_repeat_rate": percentile(repeat_rates, 0.95),
            "answers_at_or_above_0p2": sum(rate >= 0.2 for rate in repeat_rates),
            "answers_at_or_above_0p2_fraction": sum(rate >= 0.2 for rate in repeat_rates) / max(1, len(records)),
        },
        "exact_duplicate_assistant_turns_after_first": exact_duplicate_answers,
        "exact_duplicate_assistant_turn_fraction": exact_duplicate_answers / max(1, len(records)),
        "most_common_first_four_token_signature_count": max(prefix_counts.values(), default=0),
        "most_common_first_four_token_signature_fraction": max(prefix_counts.values(), default=0)
        / max(1, len(records)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    split_lines = {
        "train": Path(args.train).read_text(encoding="utf-8").splitlines(),
        "dev": Path(args.dev).read_text(encoding="utf-8").splitlines(),
    }
    catalog = [
        json.loads(line)
        for line in Path(args.catalog).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in catalog:
        by_split[str(row["split"])].append(row)
    for split, lines in split_lines.items():
        if len(lines) != len(by_split[split]):
            raise ValueError(f"{split} line/catalog count differs")

    records: list[dict[str, object]] = []
    duplicates_all: Counter[str] = Counter()
    duplicates_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("dev", "train"):
        for line, catalog_row in zip(split_lines[split], by_split[split]):
            digest = sha256_text(line)
            if digest != catalog_row["conversation_sha256"]:
                raise ValueError(f"{split} conversation hash mismatch: {digest}")
            source = str(catalog_row["source"])
            for assistant_text in parse_assistant_turns(line):
                tokens = tokenizer.encode(assistant_text).ids
                answer_digest = sha256_text(" ".join(assistant_text.split()).casefold())
                duplicates_all[answer_digest] += 1
                duplicates_by_source[source][answer_digest] += 1
                records.append(
                    {
                        "conversation_sha256": digest,
                        "split": split,
                        "source": source,
                        "answer_tokens": len(tokens),
                        "repeat_rate": repeat_rate(tokens),
                        "first_four_tokens": ",".join(map(str, tokens[:4])),
                    }
                )

    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source"])].append(record)
    artifact = {
        "schema": "alpha-rendered-chat-corpus-audit-v1",
        "inputs": {
            "train": str(Path(args.train).resolve()),
            "dev": str(Path(args.dev).resolve()),
            "catalog": str(Path(args.catalog).resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()),
        },
        "all": summary(records, duplicates_all),
        "by_source": {
            source: summary(source_records, duplicates_by_source[source])
            for source, source_records in sorted(by_source.items())
        },
    }
    out = Path(args.out).resolve()
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "all": artifact["all"], "by_source": artifact["by_source"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
