#!/usr/bin/env python3
"""Build Alpha's deterministic, pre-flagship frozen evaluation suite.

The builder deliberately separates candidate generation from finalization:

1. Build oversized, deterministic candidate pools and two audit streams.
2. Run ``scripts/audit_13gram.rs`` against pretrain and SFT training text.
3. Re-run this command with both overlap reports. Only zero-overlap candidates
   are admitted to ``final/``.

No benchmark dataset is used. Chat prompts come from the smol-smoltalk test
split that Alpha's SFT builder deliberately excludes. Closed-book questions
come from structured FineWiki infobox facts; the full source page is audited
against Alpha's pretraining slice before the question is frozen. Per-source
validation documents come from a premix parquet shard outside the four shards
used to build the training slice.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from tokenizers import Tokenizer


SEED = "alpha2-frozen-eval-v1"
END = "<|end_of_text|>\n"
AUDIT_START = "@@ALPHA_EVAL_DOC\t"
AUDIT_END = "@@END_ALPHA_EVAL_DOC"


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def normalized_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def clean_document(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or stable_hash(value)[:12]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def source_record(path: Path, known_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": known_sha256 or file_sha256(path),
    }


def fold_and_validate_chat(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str] | None:
    system_parts: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content") or "").strip()
        if not content:
            return None
        if role == "system" and not turns:
            system_parts.append(content)
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": content})
        else:
            return None

    if len(turns) < 4 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return None
    for i, turn in enumerate(turns):
        if turn["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None
    if system_parts:
        instructions = "\n\n".join(system_parts)
        turns[0]["content"] = f"[Instructions: {instructions}]\n\n{turns[0]['content']}"

    prompt = turns[:-1]
    reference = turns[-1]["content"]
    if prompt[-1]["role"] != "user":
        return None
    prompt_chars = sum(len(turn["content"]) for turn in prompt)
    if not (80 <= prompt_chars <= 3_000) or not (1 <= len(reference) <= 1_200):
        return None
    return prompt, reference


def balanced_select(rows: list[dict[str, Any]], count: int, group_key: str) -> list[dict[str, Any]]:
    """Deterministic round-robin selection across every dynamic source group."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda value: value["selection_hash"])
    group_names = sorted(grouped, key=lambda name: stable_hash(SEED, "group", name))

    selected: list[dict[str, Any]] = []
    offsets: Counter[str] = Counter()
    while len(selected) < count:
        progressed = False
        for group in group_names:
            offset = offsets[group]
            if offset >= len(grouped[group]):
                continue
            selected.append(grouped[group][offset])
            offsets[group] += 1
            progressed = True
            if len(selected) == count:
                return selected
        if not progressed:
            break
    return selected


def render_chat_prompt(messages: list[dict[str, str]]) -> str:
    parts = [
        f"{'<|user|>' if message['role'] == 'user' else '<|assistant|>'} {message['content']}"
        for message in messages
    ]
    return " ".join(parts) + " <|assistant|> "


def build_chat_candidates(
    path: Path,
    count: int,
    tokenizer: Tokenizer,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024, columns=["messages", "source"]):
        for row in batch.to_pylist():
            parsed = fold_and_validate_chat(row["messages"])
            if parsed is None:
                continue
            prompt, reference = parsed
            prompt_tokens = len(tokenizer.encode(render_chat_prompt(prompt)).ids)
            if prompt_tokens > max_prompt_tokens:
                continue
            source = normalized_space(row["source"])
            content_key = json.dumps([prompt, reference], ensure_ascii=False, sort_keys=True)
            digest = stable_hash(SEED, "chat", source, content_key)
            candidates.append({
                "id": f"chat-{digest[:16]}",
                "selection_hash": digest,
                "source": source,
                "messages": prompt,
                "reference": reference,
                "prompt_tokens": prompt_tokens,
                "audit_text": "\n".join(turn["content"] for turn in prompt) + "\n" + reference,
            })
    selected = balanced_select(candidates, count, "source")
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} eligible chat candidates; need {count}")
    return selected


def eligible_facts(raw: str | None) -> list[tuple[str, str]]:
    try:
        boxes = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    facts: list[tuple[str, str]] = []
    for box in boxes if isinstance(boxes, list) else []:
        data = box.get("data") if isinstance(box, dict) else None
        if not isinstance(data, dict):
            continue
        for raw_field, raw_answer in data.items():
            field = normalized_space(raw_field)
            answer = normalized_space(raw_answer)
            if not re.match(r"^[^\W\d_]", field, re.UNICODE):
                continue
            if not (1 <= len(field.split()) <= 6 and 2 <= len(field) <= 60):
                continue
            if not (1 <= len(answer.split()) <= 12 and 1 <= len(answer) <= 120):
                continue
            if any(marker in answer for marker in ("http://", "https://", "[", "]", "{", "}", "|", "·")):
                continue
            # Navigation-style infobox values (for example a row of five
            # decades/centuries) are technically facts but poor closed-book
            # questions. Reject any long answer dominated by numeric tokens;
            # this is content-shape validation, not a field-name allow/deny map.
            answer_words = answer.split()
            numeric_words = sum(bool(re.match(r"^[\W_]*\d", word)) for word in answer_words)
            if len(answer_words) >= 4 and numeric_words / len(answer_words) > 0.6:
                continue
            facts.append((field, answer))
    return facts


def build_qa_candidates(path: Path, count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    columns = ["id", "title", "url", "wikidata_id", "infoboxes", "text"]
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            source_id = normalized_space(row["id"])
            title = normalized_space(row["title"])
            text = clean_document(str(row["text"] or ""))
            if not source_id or not (2 <= len(title) <= 100) or len(text) < 200:
                continue
            facts = eligible_facts(row["infoboxes"])
            if not facts:
                continue
            field, answer = min(facts, key=lambda fact: stable_hash(SEED, "fact", source_id, fact[0], fact[1]))
            digest = stable_hash(SEED, "qa", source_id, field, answer)
            candidates.append({
                "id": f"qa-{digest[:16]}",
                "selection_hash": digest,
                "source_id": source_id,
                "title": title,
                "question": f"On the topic of {title}, what is the value of “{field}”?",
                "answer": answer,
                "field": field,
                "url": normalized_space(row["url"]),
                "wikidata_id": normalized_space(row["wikidata_id"]),
                "audit_text": text,
            })
    selected = sorted(candidates, key=lambda value: value["selection_hash"])[:count]
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} eligible FineWiki candidates; need {count}")
    return selected


def heap_keep_smallest(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    group: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    # Python's heap is a min-heap. Negated SHA-256 integers put the worst
    # (largest selection hash) at the root for bounded deterministic sampling.
    score = int(row["selection_hash"], 16)
    item = (-score, row["id"], row)
    heap = heaps[group]
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def build_val_candidates(path: Path, per_source: int) -> dict[str, list[dict[str, Any]]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=["id", "dataset", "url", "text"]):
        for raw in batch.to_pylist():
            text = clean_document(str(raw["text"] or ""))
            if not (200 <= len(text) <= 100_000):
                continue
            dataset = normalized_space(raw["dataset"])
            source_id = normalized_space(raw["id"]) or stable_hash(text)[:24]
            if not dataset:
                continue
            digest = stable_hash(SEED, "val", dataset, source_id)
            row = {
                "id": f"val-{safe_slug(dataset)}-{digest[:16]}",
                "selection_hash": digest,
                "dataset": dataset,
                "source_id": source_id,
                "url": normalized_space(raw["url"]),
                "text": text,
            }
            heap_keep_smallest(heaps, dataset, row, per_source)
    result: dict[str, list[dict[str, Any]]] = {}
    for source, heap in heaps.items():
        result[source] = sorted((item[2] for item in heap), key=lambda row: row["selection_hash"])
        if len(result[source]) < per_source:
            raise RuntimeError(f"only {len(result[source])} validation candidates for {source}; need {per_source}")
    if not result:
        raise RuntimeError("held-out premix shard produced no validation source groups")
    return dict(sorted(result.items()))


def audit_safe_text(text: str) -> str:
    return text.replace(AUDIT_START, "").replace(AUDIT_END, "")


def write_audit_stream(path: Path, docs: Iterable[tuple[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as out:
        for doc_id, text in docs:
            out.write(f"{AUDIT_START}{doc_id}\n")
            out.write(audit_safe_text(text).rstrip() + "\n")
            out.write(AUDIT_END + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def read_overlap_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    overlaps: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("eval_id\t"):
                continue
            overlaps.add(line.split("\t", 1)[0])
    return overlaps


def public_chat(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in ("selection_hash", "audit_text")}


def public_qa(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in ("selection_hash", "audit_text")}


def output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoltalk-test", type=Path, required=True)
    ap.add_argument("--finewiki", type=Path, required=True)
    ap.add_argument("--premix-heldout", type=Path, required=True)
    ap.add_argument("--hf-tokenizer-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pretrain-overlap-report", type=Path)
    ap.add_argument("--sft-overlap-report", type=Path)
    ap.add_argument("--chat-count", type=int, default=100)
    ap.add_argument("--qa-count", type=int, default=200)
    ap.add_argument("--val-per-source", type=int, default=500)
    ap.add_argument("--candidate-multiplier", type=int, default=3)
    ap.add_argument("--chat-candidate-multiplier", type=int, default=8)
    ap.add_argument("--max-chat-prompt-tokens", type=int, default=896)
    args = ap.parse_args()

    if args.candidate_multiplier < 2:
        ap.error("--candidate-multiplier must be at least 2")
    if args.chat_candidate_multiplier < 2:
        ap.error("--chat-candidate-multiplier must be at least 2")
    for path in (args.smoltalk_test, args.finewiki, args.premix_heldout, args.hf_tokenizer_json):
        if not path.is_file():
            ap.error(f"source does not exist: {path}")
    if (args.pretrain_overlap_report is None) != (args.sft_overlap_report is None):
        ap.error("provide both overlap reports to finalize, or neither to build candidates")

    hf_tokenizer = Tokenizer.from_file(str(args.hf_tokenizer_json))
    chat_candidates = build_chat_candidates(
        args.smoltalk_test,
        args.chat_count * args.chat_candidate_multiplier,
        hf_tokenizer,
        args.max_chat_prompt_tokens,
    )
    qa_candidates = build_qa_candidates(args.finewiki, args.qa_count * args.candidate_multiplier)
    val_candidates = build_val_candidates(args.premix_heldout, args.val_per_source * args.candidate_multiplier)

    candidates_dir = args.out / "candidates"
    audit_dir = args.out / "audit"
    candidate_chat_path = candidates_dir / "chat-prompts.jsonl"
    candidate_qa_path = candidates_dir / "closed-book-qa.jsonl"
    write_jsonl(candidate_chat_path, (public_chat(row) for row in chat_candidates))
    write_jsonl(candidate_qa_path, (public_qa(row) for row in qa_candidates))

    pretrain_docs = [(row["id"], row["audit_text"]) for row in qa_candidates]
    for rows in val_candidates.values():
        pretrain_docs.extend((row["id"], row["text"]) for row in rows)
    pretrain_audit_path = audit_dir / "pretrain-eval-docs.txt"
    sft_audit_path = audit_dir / "sft-eval-docs.txt"
    pretrain_audit_count = write_audit_stream(pretrain_audit_path, pretrain_docs)
    sft_audit_count = write_audit_stream(
        sft_audit_path,
        ((row["id"], row["audit_text"]) for row in chat_candidates),
    )

    manifest: dict[str, Any] = {
        "schema": "alpha-frozen-eval-v1",
        "seed": SEED,
        "status": "candidates",
        "licenses": {
            "chat": "Apache-2.0 (HuggingFaceTB/smol-smoltalk)",
            "closed_book_qa": "CC-BY-SA-4.0 (FineWiki sample)",
            "validation": "ODC-BY (HuggingFaceFW premix)",
        },
        "sources": {
            "smoltalk_test": source_record(args.smoltalk_test),
            "finewiki": source_record(args.finewiki),
            "premix_heldout": source_record(args.premix_heldout),
            "hf_tokenizer_json": source_record(args.hf_tokenizer_json),
        },
        "max_chat_prompt_tokens": args.max_chat_prompt_tokens,
        "candidates": {
            "chat": len(chat_candidates),
            "closed_book_qa": len(qa_candidates),
            "validation_by_source": {key: len(value) for key, value in val_candidates.items()},
            "pretrain_audit_docs": pretrain_audit_count,
            "sft_audit_docs": sft_audit_count,
        },
    }

    if args.pretrain_overlap_report is not None and args.sft_overlap_report is not None:
        pretrain_overlaps = read_overlap_ids(args.pretrain_overlap_report)
        sft_overlaps = read_overlap_ids(args.sft_overlap_report)
        clean_chat = [row for row in chat_candidates if row["id"] not in sft_overlaps]
        clean_qa = [row for row in qa_candidates if row["id"] not in pretrain_overlaps]
        final_chat = balanced_select(clean_chat, args.chat_count, "source")
        final_qa = sorted(clean_qa, key=lambda value: value["selection_hash"])[: args.qa_count]
        if len(final_chat) < args.chat_count:
            raise RuntimeError(f"only {len(final_chat)} clean chat prompts; need {args.chat_count}")
        if len(final_qa) < args.qa_count:
            raise RuntimeError(f"only {len(final_qa)} clean QA pages; need {args.qa_count}")

        final_dir = args.out / "final"
        final_chat_path = final_dir / "chat-prompts.jsonl"
        final_qa_path = final_dir / "closed-book-qa.jsonl"
        write_jsonl(final_chat_path, (public_chat(row) for row in final_chat))
        write_jsonl(final_qa_path, (public_qa(row) for row in final_qa))

        val_outputs: dict[str, dict[str, Any]] = {}
        for source, rows in val_candidates.items():
            clean = [row for row in rows if row["id"] not in pretrain_overlaps][: args.val_per_source]
            if len(clean) < args.val_per_source:
                raise RuntimeError(f"only {len(clean)} clean validation documents for {source}; need {args.val_per_source}")
            val_path = final_dir / "val" / f"{safe_slug(source)}.txt"
            atomic_write_text(val_path, "".join(row["text"] + "\n" + END for row in clean))
            val_outputs[source] = {"documents": len(clean), **output_record(val_path)}

        manifest.update({
            "status": "final",
            "overlap_reports": {
                "pretrain": source_record(args.pretrain_overlap_report),
                "sft": source_record(args.sft_overlap_report),
            },
            "excluded_for_13gram_overlap": {
                "pretrain_candidates": len(pretrain_overlaps),
                "sft_candidates": len(sft_overlaps),
            },
            "final": {
                "chat": {"rows": len(final_chat), **output_record(final_chat_path)},
                "closed_book_qa": {"rows": len(final_qa), **output_record(final_qa_path)},
                "validation": val_outputs,
            },
        })

    manifest_path = args.out / "MANIFEST.json"
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "chat_candidates": len(chat_candidates),
        "qa_candidates": len(qa_candidates),
        "val_candidate_sources": {key: len(value) for key, value in val_candidates.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
