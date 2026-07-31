#!/usr/bin/env python3
"""Build Alpha's second, de-templated conversational repair corpus.

The intervention is deliberately narrow:

* restore the pretrained model's 1,024-token conversation contract;
* reduce synthetic roleplay dominance;
* mix direct assistant responses with a minority of natural social dialogue;
* cap repeated answer signatures and exact assistant turns dynamically;
* reject target answers that already contain strong token-level loops;
* preserve source identity, rejections, hashes, and deterministic split/order.

No generated or hand-written AlphaCorpus material is used here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import os
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer


USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end_of_text|>"
MARKERS = (USER, ASSISTANT, END)


@dataclass(frozen=True)
class RawCandidate:
    source: str
    source_id: str
    turns: tuple[tuple[str, str], ...]
    selection_score: int


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    line: str
    tokens: int
    turns: int
    assistant_turns: int
    max_assistant_tokens: int
    answer_signatures: tuple[str, ...]
    answer_hashes: tuple[str, ...]
    selection_score: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def score(seed: str, namespace: str, identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{namespace}\0{identifier}".encode()).digest(), "big")


def normalize(value: object) -> str | None:
    text = " ".join(str(value or "").split())
    if not text or any(marker in text for marker in MARKERS):
        return None
    return text


def normalize_turns(raw: Sequence[tuple[str, object]]) -> tuple[tuple[str, str], ...] | None:
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
    return tuple(turns)


def repeat_rate(tokens: list[int], n: int = 4) -> float:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for index in range(total):
        gram = tuple(tokens[index : index + n])
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / total


def keep_lowest(
    heap: list[tuple[int, str, RawCandidate]], limit: int, candidate: RawCandidate
) -> None:
    item = (-candidate.selection_score, candidate.source_id, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def strip_dominant_leading_pair(
    candidates: list[RawCandidate], minimum_count: int = 32, minimum_fraction: float = 0.5
) -> tuple[list[RawCandidate], dict[str, object] | None]:
    # Greeting prompts vary ("Hi", "Hello", "Hi there"), while this source's
    # canned first assistant response is nearly invariant. Detect that response
    # from the distribution instead of matching greeting words or a literal
    # phrase, then remove the entire opening exchange only where it occurs.
    # Requiring another user-assistant exchange preserves a valid conversation.
    replies = Counter(candidate.turns[1] for candidate in candidates if len(candidate.turns) >= 4)
    reply, count = replies.most_common(1)[0] if replies else ((), 0)
    if count < minimum_count or count / max(1, len(candidates)) < minimum_fraction:
        return candidates, None
    stripped = [
        replace(candidate, turns=candidate.turns[2:])
        if len(candidate.turns) >= 4 and candidate.turns[1] == reply
        else candidate
        for candidate in candidates
    ]
    return stripped, {
        "count": count,
        "fraction": count / len(candidates),
        "assistant_response_sha256": sha256_bytes(json.dumps(reply, ensure_ascii=False).encode()),
        "policy": "strip the leading user-assistant exchange only when one exact first assistant response occupies at least half the source",
    }


def render_candidate(
    raw: RawCandidate,
    tokenizer: Tokenizer,
    max_tokens: int,
    max_answer_tokens: int,
    max_answer_repeat: float,
    rejected: Counter[str],
) -> Candidate | None:
    line = " ".join(f"{USER if role == 'user' else ASSISTANT} {content}" for role, content in raw.turns) + f" {END}"
    encoded = tokenizer.encode(line).ids
    if len(encoded) > max_tokens:
        rejected[f"{raw.source}:over_token_bound"] += 1
        return None
    answers = [content for role, content in raw.turns if role == "assistant"]
    answer_ids = [tokenizer.encode(answer).ids for answer in answers]
    if max(map(len, answer_ids), default=0) > max_answer_tokens:
        rejected[f"{raw.source}:answer_over_token_bound"] += 1
        return None
    if any(repeat_rate(ids) > max_answer_repeat for ids in answer_ids):
        rejected[f"{raw.source}:repetitive_target"] += 1
        return None
    return Candidate(
        source=raw.source,
        source_id=raw.source_id,
        line=line,
        tokens=len(encoded),
        turns=len(raw.turns),
        assistant_turns=len(answers),
        max_assistant_tokens=max(map(len, answer_ids), default=0),
        answer_signatures=tuple(",".join(map(str, ids[:4])) for ids in answer_ids),
        answer_hashes=tuple(sha256_bytes(" ".join(answer.split()).casefold().encode()) for answer in answers),
        selection_score=raw.selection_score,
    )


def select_diverse(
    raw_candidates: Iterable[RawCandidate],
    target: int,
    tokenizer: Tokenizer,
    max_tokens: int,
    max_answer_tokens: int,
    max_answer_repeat: float,
    signature_cap: int,
    exact_answer_cap: int,
    rejected: Counter[str],
) -> list[Candidate]:
    signature_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    seen_conversations: set[str] = set()
    selected: list[Candidate] = []
    for raw in sorted(raw_candidates, key=lambda candidate: (candidate.selection_score, candidate.source_id)):
        candidate = render_candidate(
            raw, tokenizer, max_tokens, max_answer_tokens, max_answer_repeat, rejected
        )
        if candidate is None:
            continue
        digest = sha256_bytes(candidate.line.encode())
        if digest in seen_conversations:
            rejected[f"{candidate.source}:exact_conversation_duplicate"] += 1
            continue
        if any(signature_counts[signature] >= signature_cap for signature in candidate.answer_signatures):
            rejected[f"{candidate.source}:answer_signature_cap"] += 1
            continue
        if any(answer_counts[answer_hash] >= exact_answer_cap for answer_hash in candidate.answer_hashes):
            rejected[f"{candidate.source}:exact_answer_cap"] += 1
            continue
        selected.append(candidate)
        seen_conversations.add(digest)
        signature_counts.update(candidate.answer_signatures)
        answer_counts.update(candidate.answer_hashes)
        if len(selected) >= target:
            break
    return selected


def smol_magpie_reserve(paths: Sequence[str], target: int, seed: str, rejected: Counter[str]) -> list[RawCandidate]:
    heap: list[tuple[int, str, RawCandidate]] = []
    reserve = max(target * 4, target + 20_000)
    row_number = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["source", "messages"]):
            for row in batch.to_pylist():
                row_number += 1
                if row["source"] != "smol-magpie-ultra-short":
                    continue
                messages = row["messages"] or []
                turns = normalize_turns([(message.get("role", ""), message.get("content")) for message in messages])
                if turns is None or len(turns) > 8:
                    rejected["smol_magpie:surface_or_structure"] += 1
                    continue
                contents = [content for _, content in turns]
                if max(map(len, contents), default=0) > 1_600 or sum(map(len, contents)) > 5_000:
                    rejected["smol_magpie:surface_or_structure"] += 1
                    continue
                identifier = str(row_number)
                keep_lowest(
                    heap,
                    reserve,
                    RawCandidate("smol_magpie", identifier, turns, score(seed, "smol_magpie", identifier)),
                )
    return [item[2] for item in heap]


def everyday_candidates(paths: Sequence[str], seed: str, rejected: Counter[str]) -> tuple[list[RawCandidate], dict[str, object] | None]:
    candidates: list[RawCandidate] = []
    row_number = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["messages"]):
            for row in batch.to_pylist():
                row_number += 1
                turns = normalize_turns(
                    [(message.get("role", ""), message.get("content")) for message in (row["messages"] or [])]
                )
                if turns is None:
                    rejected["everyday:structure"] += 1
                    continue
                identifier = str(row_number)
                candidates.append(
                    RawCandidate("everyday", identifier, turns, score(seed, "everyday", identifier))
                )
    return strip_dominant_leading_pair(candidates)


def oasst2_candidates(paths: Sequence[str], seed: str, rejected: Counter[str]) -> list[RawCandidate]:
    rows: list[dict[str, object]] = []
    columns = [
        "message_id", "parent_id", "text", "role", "lang", "rank", "review_result", "review_count", "deleted"
    ]
    for path in paths:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    children: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["parent_id"]:
            children[str(row["parent_id"])].append(row)

    def best_child(node_id: str, role: str) -> dict[str, object] | None:
        candidates = [
            row
            for row in children.get(node_id, [])
            if row["role"] == role
            and row["lang"] == "en"
            and not row.get("deleted")
            and row.get("review_result") is not False
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda row: (
                row.get("rank") if row.get("rank") is not None else 99,
                -(row.get("review_count") or 0),
            )
        )
        return candidates[0]

    result: list[RawCandidate] = []
    for root in rows:
        if root["parent_id"] is not None or root["lang"] != "en" or root["role"] != "prompter":
            continue
        raw_turns: list[tuple[str, object]] = [("user", root["text"])]
        node = root
        while True:
            assistant = best_child(str(node["message_id"]), "assistant")
            if assistant is None:
                break
            raw_turns.append(("assistant", assistant["text"]))
            following = best_child(str(assistant["message_id"]), "prompter")
            if following is None:
                break
            raw_turns.append(("user", following["text"]))
            node = following
        turns = normalize_turns(raw_turns)
        if turns is None:
            rejected["oasst2:structure"] += 1
            continue
        identifier = str(root["message_id"])
        result.append(RawCandidate("oasst2", identifier, turns, score(seed, "oasst2", identifier)))
    return result


def soda_reserve(path: str, target: int, seed: str, rejected: Counter[str]) -> list[RawCandidate]:
    heap: list[tuple[int, str, RawCandidate]] = []
    reserve = max(target * 5, target + 20_000)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=["original_index", "dialogue"]):
        for row in batch.to_pylist():
            turns = normalize_turns(
                [("user" if index % 2 == 0 else "assistant", value) for index, value in enumerate((row["dialogue"] or [])[:8])]
            )
            if turns is None:
                rejected["soda:structure"] += 1
                continue
            contents = [content for _, content in turns]
            if max(map(len, contents), default=0) > 1_200 or sum(map(len, contents)) > 4_000:
                rejected["soda:surface"] += 1
                continue
            identifier = str(row["original_index"])
            keep_lowest(
                heap,
                reserve,
                RawCandidate("soda", identifier, turns, score(seed, "soda", identifier)),
            )
    return [item[2] for item in heap]


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="alpha-chat-repair-v2")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-answer-repeat", type=float, default=0.15)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--magpie-target", type=int, default=30_000)
    parser.add_argument("--soda-target", type=int, default=8_000)
    parser.add_argument("--signature-cap", type=int, default=96)
    parser.add_argument("--exact-answer-cap", type=int, default=4)
    parser.add_argument(
        "--tokenizer",
        default="/mnt/donto-data/alpha-corpora/tokenizers/hf-bpe-byte-12k-20260722/tokenizer.json",
    )
    parser.add_argument(
        "--smol-glob",
        default="/mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data/train-*.parquet",
    )
    parser.add_argument(
        "--everyday-glob",
        default="/mnt/donto-data/alpha-corpora/sft/smoltalk2/SFT/*everyday*_no_think-*.parquet",
    )
    parser.add_argument(
        "--oasst2-glob",
        default="/mnt/donto-data/alpha-corpora/sft/oasst2/data/train-*.parquet",
    )
    parser.add_argument("--soda", default="/mnt/donto-data/alpha-corpora/sft/soda/train.parquet")
    args = parser.parse_args()
    if args.max_tokens < 32 or args.max_answer_tokens < 1:
        parser.error("token bounds are invalid")
    if not 0 < args.dev_fraction < 0.5 or not 0 <= args.max_answer_repeat <= 1:
        parser.error("fraction or repetition bound is invalid")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    smol_paths = sorted(glob.glob(args.smol_glob))
    everyday_paths = sorted(glob.glob(args.everyday_glob))
    oasst2_paths = sorted(glob.glob(args.oasst2_glob))
    if not smol_paths or not everyday_paths or not oasst2_paths:
        raise RuntimeError("one or more source globs matched no files")
    rejected: Counter[str] = Counter()
    everyday_raw, greeting_transform = everyday_candidates(everyday_paths, args.seed, rejected)
    source_raw: dict[str, list[RawCandidate]] = {
        "smol_magpie": smol_magpie_reserve(smol_paths, args.magpie_target, args.seed, rejected),
        "everyday": everyday_raw,
        "oasst2": oasst2_candidates(oasst2_paths, args.seed, rejected),
        "soda": soda_reserve(args.soda, args.soda_target, args.seed, rejected),
    }
    targets = {
        "smol_magpie": args.magpie_target,
        "everyday": len(source_raw["everyday"]),
        "oasst2": len(source_raw["oasst2"]),
        "soda": args.soda_target,
    }
    selected: list[Candidate] = []
    selected_by_source: Counter[str] = Counter()
    for source, raw_candidates in source_raw.items():
        rows = select_diverse(
            raw_candidates,
            targets[source],
            tokenizer,
            args.max_tokens,
            args.max_answer_tokens,
            args.max_answer_repeat,
            args.signature_cap,
            args.exact_answer_cap,
            rejected,
        )
        selected.extend(rows)
        selected_by_source[source] = len(rows)

    unique: dict[str, Candidate] = {}
    cross_source_duplicates = 0
    for candidate in selected:
        digest = sha256_bytes(candidate.line.encode())
        if digest in unique:
            cross_source_duplicates += 1
            continue
        unique[digest] = candidate

    split_rows: list[tuple[str, str, Candidate]] = []
    dev_threshold = int(args.dev_fraction * (1 << 64))
    for digest, candidate in unique.items():
        split_value = int.from_bytes(hashlib.sha256(f"{args.seed}\0split\0{digest}".encode()).digest()[:8], "big")
        split = "dev" if split_value < dev_threshold else "train"
        order = sha256_bytes(f"{args.seed}\0order\0{digest}".encode())
        split_rows.append((split, order, candidate))
    split_rows.sort(key=lambda row: (row[0], row[1]))

    output_lines: dict[str, list[str]] = {"train": [], "dev": []}
    token_counts: dict[str, list[int]] = {"train": [], "dev": []}
    catalog_lines: list[str] = []
    for split, _, candidate in split_rows:
        digest = sha256_bytes(candidate.line.encode())
        output_lines[split].append(candidate.line)
        token_counts[split].append(candidate.tokens)
        catalog_lines.append(
            json.dumps(
                {
                    "schema": "alpha-chat-repair-catalog-v2",
                    "conversation_sha256": digest,
                    "source": candidate.source,
                    "source_id": candidate.source_id,
                    "split": split,
                    "tokens": candidate.tokens,
                    "turns": candidate.turns,
                    "assistant_turns": candidate.assistant_turns,
                    "max_assistant_tokens": candidate.max_assistant_tokens,
                },
                sort_keys=True,
            )
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    dev_path = out_dir / "dev.txt"
    catalog_path = out_dir / "catalog.jsonl"
    write_atomic(train_path, "\n".join(output_lines["train"]) + "\n")
    write_atomic(dev_path, "\n".join(output_lines["dev"]) + "\n")
    write_atomic(catalog_path, "\n".join(catalog_lines) + "\n")
    source_paths = [args.tokenizer, args.soda, *smol_paths, *everyday_paths, *oasst2_paths]
    manifest = {
        "schema": "alpha-chat-repair-corpus-v2",
        "purpose": "direct-assistant-dominant, de-templated conversational repair; no AlphaCorpus data",
        "seed": args.seed,
        "recipe": {
            "maxTokens": args.max_tokens,
            "maxAnswerTokens": args.max_answer_tokens,
            "maxAnswerFourGramRepeatRate": args.max_answer_repeat,
            "devFraction": args.dev_fraction,
            "targets": targets,
            "answerSignatureCapPerSource": args.signature_cap,
            "exactAssistantTurnCapPerSource": args.exact_answer_cap,
            "dominantLeadingPairTransform": greeting_transform,
            "selection": "lowest sha256(seed, source, source_id) subject to content-independent diversity caps",
            "split": "sha256(seed, conversation_sha256)",
            "order": "sha256(seed, conversation_sha256)",
        },
        "selectedBeforeCrossSourceDedup": dict(sorted(selected_by_source.items())),
        "crossSourceExactConversationDuplicatesRemoved": cross_source_duplicates,
        "rows": {split: len(rows) for split, rows in output_lines.items()},
        "tokens": {
            split: {
                "min": min(values, default=0),
                "max": max(values, default=0),
                "mean": sum(values) / max(1, len(values)),
            }
            for split, values in token_counts.items()
        },
        "rejected": dict(sorted(rejected.items())),
        "dependencies": {
            "python": platform.python_version(),
            "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "package-version-unavailable",
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
