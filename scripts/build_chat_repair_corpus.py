#!/usr/bin/env python3
"""Build a compact, conversation-first SFT corpus for Alpha's response repair.

This is intentionally not a miniature encyclopedia. It favors short natural
dialogue, delayed follow-ups, and concise assistant turns. Every accepted row is
hash-addressed, source-attributed, token-bounded, deterministically assigned to
train/development, and interleaved by a seed-derived order key.

Runtime dependencies are deliberately external to the repository::

  uv run --with duckdb --with tokenizers python3 scripts/build_chat_repair_corpus.py \
    --out-dir /mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import os
import platform
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import tokenizers
from tokenizers import Tokenizer

U = "<|user|>"
A = "<|assistant|>"
END = "<|end_of_text|>"
MARKERS = (U, A, END)
ROLE_SPLIT = re.compile(r"(<\|user\|>|<\|assistant\|>)")


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    line: str
    tokens: int
    turns: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def normalize(value: object) -> str | None:
    text = " ".join(str(value or "").split())
    if not text or any(marker in text for marker in MARKERS):
        return None
    return text


def normalize_turns(raw: Sequence[tuple[str, object]]) -> list[tuple[str, str]] | None:
    turns: list[tuple[str, str]] = []
    for role, raw_content in raw:
        if role not in ("user", "assistant"):
            return None
        content = normalize(raw_content)
        if content is None:
            return None
        turns.append((role, content))
    if turns and turns[-1][0] == "user":
        turns.pop()
    if len(turns) < 2 or turns[0][0] != "user":
        return None
    if any(role != ("user" if index % 2 == 0 else "assistant") for index, (role, _) in enumerate(turns)):
        return None
    return turns


def render(turns: Sequence[tuple[str, str]], tokenizer: Tokenizer, max_tokens: int) -> tuple[str, int] | None:
    line = " ".join(f"{U if role == 'user' else A} {content}" for role, content in turns) + f" {END}"
    token_count = len(tokenizer.encode(line).ids)
    if token_count > max_tokens:
        return None
    return line, token_count


def parse_rendered(line: str) -> list[tuple[str, str]] | None:
    line = line.strip()
    if not line.endswith(END):
        return None
    body = line[: -len(END)].strip()
    parts = ROLE_SPLIT.split(body)
    turns: list[tuple[str, object]] = []
    for index in range(1, len(parts), 2):
        marker = parts[index]
        content = parts[index + 1] if index + 1 < len(parts) else ""
        turns.append(("user" if marker == U else "assistant", content))
    return normalize_turns(turns)


def score(seed: str, namespace: str, identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{namespace}\0{identifier}".encode()).digest(), "big")


def keep_lowest(
    heap: list[tuple[int, str, object]],
    limit: int,
    item_score: int,
    identifier: str,
    payload: object,
) -> None:
    item = (-item_score, identifier, payload)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def accepted_candidate(
    source: str,
    source_id: str,
    turns: list[tuple[str, str]] | None,
    tokenizer: Tokenizer,
    max_tokens: int,
    rejected: Counter[str],
) -> Candidate | None:
    if turns is None:
        rejected[f"{source}:invalid_structure"] += 1
        return None
    rendered = render(turns, tokenizer, max_tokens)
    if rendered is None:
        rejected[f"{source}:over_token_bound"] += 1
        return None
    line, token_count = rendered
    return Candidate(source, source_id, line, token_count, len(turns))


def everyday_candidates(
    paths: Sequence[str], tokenizer: Tokenizer, max_tokens: int, rejected: Counter[str]
) -> list[Candidate]:
    connection = duckdb.connect()
    accepted: list[Candidate] = []
    row_number = 0
    for path in paths:
        cursor = connection.execute("SELECT messages FROM read_parquet(?)", [path])
        while rows := cursor.fetchmany(2048):
            for (messages,) in rows:
                row_number += 1
                raw = [(message.get("role", ""), message.get("content")) for message in (messages or [])]
                candidate = accepted_candidate(
                    "everyday", str(row_number), normalize_turns(raw), tokenizer, max_tokens, rejected
                )
                if candidate:
                    accepted.append(candidate)
    connection.close()
    return accepted


def soda_candidates(
    path: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    target: int,
    seed: str,
    rejected: Counter[str],
) -> list[Candidate]:
    # Select by hash only after inexpensive surface checks, then tokenize a
    # reserve larger than the target. This avoids tokenizing the million-row
    # source merely to choose a deterministic small sample.
    reserve = max(target * 2, target + 5000)
    heap: list[tuple[int, str, object]] = []
    connection = duckdb.connect()
    cursor = connection.execute("SELECT original_index, dialogue FROM read_parquet(?)", [path])
    while rows := cursor.fetchmany(4096):
        for original_index, dialogue in rows:
            source_id = str(original_index)
            utterances = [normalize(value) for value in (dialogue or [])[:8]]
            if len(utterances) < 4 or any(value is None for value in utterances):
                rejected["soda:surface_filter"] += 1
                continue
            if len(utterances) % 2 == 1:
                utterances.pop()
            assert all(value is not None for value in utterances)
            text_values = [value for value in utterances if value is not None]
            if max(map(len, text_values), default=0) > 600 or sum(map(len, text_values)) > 2400:
                rejected["soda:surface_filter"] += 1
                continue
            turns = [("user" if index % 2 == 0 else "assistant", value) for index, value in enumerate(text_values)]
            keep_lowest(heap, reserve, score(seed, "soda", source_id), source_id, turns)
    connection.close()

    accepted: list[Candidate] = []
    for _, source_id, payload in sorted(heap, key=lambda item: -item[0]):
        candidate = accepted_candidate("soda", source_id, payload, tokenizer, max_tokens, rejected)
        if candidate:
            accepted.append(candidate)
        if len(accepted) >= target:
            break
    return accepted


def rendered_candidates(
    path: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    smol_target: int,
    seed: str,
    rejected: Counter[str],
) -> list[Candidate]:
    smol_reserve = max(smol_target * 2, smol_target + 2500)
    smol_heap: list[tuple[int, str, object]] = []
    oasst: list[tuple[str, list[tuple[str, str]]]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number > 486_738:
                break
            if line_number <= 450_402:
                source = "smol_concise"
            elif line_number >= 483_300:
                source = "oasst2"
            else:
                continue
            turns = parse_rendered(raw_line)
            if turns is None:
                rejected[f"{source}:invalid_structure"] += 1
                continue
            contents = [content for _, content in turns]
            assistants = [content for role, content in turns if role == "assistant"]
            if source == "smol_concise":
                if (
                    len(turns) < 4
                    or len(turns) > 8
                    or "[Instructions:" in raw_line
                    or max(map(len, contents), default=0) > 600
                    or (sum(map(len, assistants)) / max(1, len(assistants))) > 350
                    or sum(map(len, contents)) > 2600
                ):
                    rejected["smol_concise:surface_filter"] += 1
                    continue
                source_id = str(line_number)
                keep_lowest(smol_heap, smol_reserve, score(seed, source, source_id), source_id, turns)
            else:
                if max(map(len, contents), default=0) <= 900 and sum(map(len, contents)) <= 3200:
                    oasst.append((str(line_number), turns))
                else:
                    rejected["oasst2:surface_filter"] += 1

    accepted: list[Candidate] = []
    smol_count = 0
    for _, source_id, payload in sorted(smol_heap, key=lambda item: -item[0]):
        candidate = accepted_candidate("smol_concise", source_id, payload, tokenizer, max_tokens, rejected)
        if candidate:
            accepted.append(candidate)
            smol_count += 1
        if smol_count >= smol_target:
            break
    for source_id, turns in oasst:
        candidate = accepted_candidate("oasst2", source_id, turns, tokenizer, max_tokens, rejected)
        if candidate:
            accepted.append(candidate)
    return accepted


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="alpha-chat-repair-v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--soda-target", type=int, default=30_000)
    parser.add_argument("--smol-target", type=int, default=5_000)
    parser.add_argument(
        "--tokenizer",
        default="/mnt/donto-data/alpha-corpora/tokenizers/hf-bpe-byte-12k-20260722/tokenizer.json",
    )
    parser.add_argument(
        "--old-rendered",
        default="/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt",
    )
    parser.add_argument(
        "--everyday-glob",
        default="/mnt/donto-data/alpha-corpora/sft/smoltalk2/SFT/*everyday*_no_think-*.parquet",
    )
    parser.add_argument("--soda", default="/mnt/donto-data/alpha-corpora/sft/soda/train.parquet")
    args = parser.parse_args()
    if args.max_tokens < 32:
        parser.error("--max-tokens must be at least 32")
    if not 0 < args.dev_fraction < 0.5:
        parser.error("--dev-fraction must be between 0 and 0.5")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    everyday_paths = sorted(glob.glob(args.everyday_glob))
    if not everyday_paths:
        raise RuntimeError(f"no everyday parquet matched {args.everyday_glob}")

    rejected: Counter[str] = Counter()
    candidates: list[Candidate] = []
    candidates.extend(everyday_candidates(everyday_paths, tokenizer, args.max_tokens, rejected))
    candidates.extend(
        soda_candidates(args.soda, tokenizer, args.max_tokens, args.soda_target, args.seed, rejected)
    )
    candidates.extend(
        rendered_candidates(
            args.old_rendered,
            tokenizer,
            args.max_tokens,
            args.smol_target,
            args.seed,
            rejected,
        )
    )

    unique: dict[str, Candidate] = {}
    duplicates = 0
    for candidate in candidates:
        digest = sha256_bytes(candidate.line.encode())
        if digest in unique:
            duplicates += 1
            continue
        unique[digest] = candidate

    rows: list[tuple[str, str, Candidate]] = []
    dev_threshold = int(args.dev_fraction * (1 << 64))
    for digest, candidate in unique.items():
        split_value = int.from_bytes(hashlib.sha256(f"{args.seed}\0split\0{digest}".encode()).digest()[:8], "big")
        split = "dev" if split_value < dev_threshold else "train"
        order = sha256_bytes(f"{args.seed}\0order\0{digest}".encode())
        rows.append((split, order, candidate))
    rows.sort(key=lambda item: (item[0], item[1]))

    output_lines: dict[str, list[str]] = {"train": [], "dev": []}
    catalog_lines: list[str] = []
    selected_counts: Counter[str] = Counter()
    token_counts: dict[str, list[int]] = {"train": [], "dev": []}
    for split, _, candidate in rows:
        digest = sha256_bytes(candidate.line.encode())
        output_lines[split].append(candidate.line)
        selected_counts[candidate.source] += 1
        token_counts[split].append(candidate.tokens)
        catalog_lines.append(
            json.dumps(
                {
                    "schema": "alpha-chat-repair-catalog-v1",
                    "conversation_sha256": digest,
                    "source": candidate.source,
                    "source_id": candidate.source_id,
                    "split": split,
                    "tokens": candidate.tokens,
                    "turns": candidate.turns,
                },
                sort_keys=True,
            )
        )

    train_path = out_dir / "train.txt"
    dev_path = out_dir / "dev.txt"
    catalog_path = out_dir / "catalog.jsonl"
    write_atomic(train_path, "\n".join(output_lines["train"]) + "\n")
    write_atomic(dev_path, "\n".join(output_lines["dev"]) + "\n")
    write_atomic(catalog_path, "\n".join(catalog_lines) + "\n")

    source_paths = [args.tokenizer, args.old_rendered, args.soda, *everyday_paths]
    manifest = {
        "schema": "alpha-chat-repair-corpus-v1",
        "purpose": "compact conversation-first corrective SFT; not a knowledge corpus",
        "seed": args.seed,
        "recipe": {
            "maxTokens": args.max_tokens,
            "devFraction": args.dev_fraction,
            "sodaTarget": args.soda_target,
            "smolConciseTarget": args.smol_target,
            "sourceOrder": "sha256(seed, conversation_sha256)",
            "split": "sha256(seed, conversation_sha256)",
            "systemPrompts": "excluded from the recovered everyday split; pre-rendered instruction rows filtered",
        },
        "selected": dict(sorted(selected_counts.items())),
        "rows": {split: len(lines) for split, lines in output_lines.items()},
        "tokens": {
            split: {
                "min": min(values, default=0),
                "max": max(values, default=0),
                "mean": (sum(values) / len(values)) if values else 0,
            }
            for split, values in token_counts.items()
        },
        "exactDuplicatesRemoved": duplicates,
        "rejected": dict(sorted(rejected.items())),
        "dependencies": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "sources": [sha256_file(path) for path in source_paths],
        "outputs": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "catalog": sha256_file(catalog_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
