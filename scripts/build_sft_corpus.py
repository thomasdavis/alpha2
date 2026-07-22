#!/usr/bin/env python3
"""Build Alpha's deduplicated chat SFT corpus with hashed source provenance.

Output format (one conversation per line-block, validated by scripts/validate-chat-data.ts):
  <|user|> ... <|assistant|> ... [<|user|> ... <|assistant|> ...] <|end_of_text|>\n

System messages: Alpha's tokenizer reserves only <|user|>/<|assistant|>/<|end_of_text|>,
so a system prompt is folded into the first user turn as a bracketed instruction
("[Instructions: ...]\n\n"). Conversations must start with user and strictly alternate;
anything else is skipped and counted.

Usage:
  python scripts/build_sft_corpus.py \
    --smoltalk /mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data \
    --oasst2  /mnt/donto-data/alpha-corpora/sft/oasst2 \
    --smoltalk2 /mnt/donto-data/alpha-corpora/sft/smoltalk2 \
    --soda /mnt/donto-data/alpha-corpora/sft/soda/train.parquet \
    --hf-tokenizer-json /mnt/donto-data/alpha-corpora/tokenizers/hf-bpe-byte-12k-20260722/tokenizer.json \
    --out /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt [--max-chars 8000]
"""
import argparse
import glob
import hashlib
import heapq
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

U, A, END = "<|user|>", "<|assistant|>", "<|end_of_text|>"
MARKERS = (U, A, END)


def render_batch(conversations, max_chars, tokenizer, max_tokens):
    """Render/token-bound conversations in one Rust-tokenizer batch per trim pass."""
    results = [None] * len(conversations)
    pending = []
    for index, turns in enumerate(conversations):
        if not turns:
            continue
        parts = []
        invalid = False
        for role, content in turns:
            content = " ".join(content.split())  # single line; no marker collisions
            if any(marker in content for marker in MARKERS):
                invalid = True
                break
            parts.append(f"{U if role == 'user' else A} {content}")
        if invalid or len(parts) < 2:
            continue
        original_turns = len(parts)
        while len(parts) >= 2 and len(" ".join(parts)) + 1 + len(END) > max_chars:
            parts = parts[:-2]
        if len(parts) >= 2:
            pending.append((index, parts, original_turns))

    while pending:
        lines = [" ".join(parts) + f" {END}" for _, parts, _ in pending]
        encodings = tokenizer.encode_batch(lines)
        next_pending = []
        for item, line, encoding in zip(pending, lines, encodings):
            index, parts, original_turns = item
            if len(encoding.ids) <= max_tokens:
                results[index] = (line, len(parts) < original_turns)
            elif len(parts) > 2:
                next_pending.append((index, parts[:-2], original_turns))
        pending = next_pending
    return results


def convo_from_messages(msgs, external_system=None):
    """smol-smoltalk row -> alternating user-first turns (system folded into first user)."""
    systems = []
    if external_system and external_system.strip():
        systems.append(external_system.strip())
    turns = []
    for m in msgs:
        role, content = m["role"], (m["content"] or "").strip()
        if not content:
            return None
        if role == "system":
            if turns:
                return None
            systems.append(content)
        elif role in ("user", "assistant"):
            turns.append((role, content))
        else:
            return None
    if not turns or turns[0][0] != "user":
        return None
    for i, (role, _) in enumerate(turns):
        if role != ("user" if i % 2 == 0 else "assistant"):
            return None
    if turns[-1][0] == "user":  # drop a trailing unanswered user turn (validator: ends_with_user)
        turns = turns[:-1]
    if len(turns) < 2:
        return None
    if systems:
        turns[0] = ("user", f"[Instructions: {' '.join(systems)}]\n\n{turns[0][1]}")
    return turns


def file_record(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return {"path": str(Path(path).resolve()), "bytes": os.path.getsize(path), "sha256": h.hexdigest()}


def keep_soda_sample(path, limit, seed):
    """Return the deterministic lowest-hash SODA rows without loading the full dataset."""
    if limit <= 0:
        return []
    heap = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048, columns=["original_index", "dialogue"]):
        for row in batch.to_pylist():
            dialogue = row["dialogue"] or []
            if len(dialogue) < 2:
                continue
            key = f"{seed}\0{row['original_index']}".encode()
            score = int.from_bytes(hashlib.sha256(key).digest(), "big")
            item = (-score, str(row["original_index"]), dialogue)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda value: -value[0])]


def oasst2_paths(rows):
    """Reconstruct trees; yield the best-ranked English root->leaf path per tree."""
    children = defaultdict(list)
    for r in rows:
        if r["parent_id"]:
            children[r["parent_id"]].append(r)

    def best_child(node_id, role):
        cands = [c for c in children.get(node_id, [])
                 if c["role"] == role and c["lang"] == "en"
                 and not c.get("deleted") and (c.get("review_result") is not False)]
        if not cands:
            return None
        # rank 0 = best human ranking; fall back to review_count
        cands.sort(key=lambda c: (c.get("rank") if c.get("rank") is not None else 99,
                                  -(c.get("review_count") or 0)))
        return cands[0]

    for r in rows:
        if r["parent_id"] is None and r["lang"] == "en" and r["role"] == "prompter":
            turns = [("user", r["text"])]
            node = r
            while True:
                asst = best_child(node["message_id"], "assistant")
                if not asst:
                    break
                turns.append(("assistant", asst["text"]))
                nxt = best_child(asst["message_id"], "prompter")
                if not nxt:
                    break
                turns.append(("user", nxt["text"]))
                node = nxt
            if len(turns) >= 2 and turns[-1][0] == "assistant":
                yield turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoltalk", required=True)
    ap.add_argument("--oasst2", required=True)
    ap.add_argument("--smoltalk2", help="dir containing selected smoltalk2 SFT no_think parquets")
    ap.add_argument("--soda", help="allenai/soda train parquet")
    ap.add_argument("--soda-max", type=int, default=25_000)
    ap.add_argument("--soda-seed", default="alpha2-sft-v2-soda")
    ap.add_argument("--hf-tokenizer-json", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    if args.max_tokens < 2:
        ap.error("--max-tokens must be at least 2")
    tokenizer = Tokenizer.from_file(args.hf_tokenizer_json)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest_path = args.manifest or args.out + ".manifest.json"
    tmp_out = args.out + ".tmp"
    written = skipped = duplicates = prefix_trimmed = 0
    source_counts = defaultdict(int)
    source_spans = []
    seen = set()

    def emit(out, line, source):
        nonlocal written, duplicates
        digest = hashlib.sha256(line.encode("utf-8")).digest()
        if digest in seen:
            duplicates += 1
            return False
        seen.add(digest)
        out.write(line + "\n")
        written += 1
        source_counts[source] += 1
        if not source_spans or source_spans[-1]["source"] != source:
            source_spans.append({"source": source, "start_line": written, "end_line": written})
        else:
            source_spans[-1]["end_line"] = written
        return True

    def emit_conversations(out, conversations, sources):
        nonlocal skipped, prefix_trimmed
        rendered = render_batch(conversations, args.max_chars, tokenizer, args.max_tokens)
        for result, source in zip(rendered, sources):
            if result is None:
                skipped += 1
                continue
            line, trimmed = result
            if emit(out, line, source):
                prefix_trimmed += int(trimmed)

    source_files = []
    with open(tmp_out, "w", encoding="utf-8") as out:
        # smol-smoltalk (train split only; test kept out for eval)
        for path in sorted(glob.glob(os.path.join(args.smoltalk, "train-*.parquet"))):
            source_files.append(path)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1024, columns=["messages"]):
                conversations = [convo_from_messages(msgs) for msgs in batch.column("messages").to_pylist()]
                emit_conversations(out, conversations, ["smol-smoltalk"] * len(conversations))
        print(f"smol-smoltalk: {source_counts['smol-smoltalk']} unique conversations", flush=True)

        # Two selected smoltalk2 no_think splits. System-chat instructions are
        # stored in chat_template_kwargs rather than as message rows.
        if args.smoltalk2:
            paths = sorted(glob.glob(os.path.join(args.smoltalk2, "**", "*_no_think-*.parquet"), recursive=True))
            if not paths:
                raise RuntimeError(f"no smoltalk2 no_think parquets under {args.smoltalk2}")
            for path in paths:
                source_files.append(path)
                pf = pq.ParquetFile(path)
                columns = ["messages", "chat_template_kwargs", "source"]
                for batch in pf.iter_batches(batch_size=1024, columns=columns):
                    conversations = []
                    sources = []
                    for row in batch.to_pylist():
                        kwargs = row["chat_template_kwargs"] or {}
                        conversations.append(convo_from_messages(row["messages"], kwargs.get("custom_instructions")))
                        sources.append(f"smoltalk2:{row['source']}")
                    emit_conversations(out, conversations, sources)
            print(
                "smoltalk2: " + ", ".join(
                    f"{key}={value}" for key, value in sorted(source_counts.items()) if key.startswith("smoltalk2:")
                ),
                flush=True,
            )

        # oasst2 English best-ranked paths (train split)
        o_files = [p for p in glob.glob(os.path.join(args.oasst2, "**/*.parquet"), recursive=True)
                   if "train" in os.path.basename(p)]
        rows = []
        cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "review_result",
                "review_count", "deleted"]
        for path in o_files:
            source_files.append(path)
            rows.extend(pq.read_table(path, columns=cols).to_pylist())
        oasst_conversations = list(oasst2_paths(rows))
        for start in range(0, len(oasst_conversations), 1024):
            conversations = oasst_conversations[start:start + 1024]
            emit_conversations(out, conversations, ["oasst2"] * len(conversations))
        print(f"oasst2: {source_counts['oasst2']} unique conversations", flush=True)

        # Natural-dialogue seasoning, deterministically sampled well below 5%
        # of the final row count. Dialogue speakers alternate; Alpha learns the
        # next speaker as assistant without importing narrative metadata.
        if args.soda:
            source_files.append(args.soda)
            soda_conversations = []
            for dialogue in keep_soda_sample(args.soda, args.soda_max, args.soda_seed):
                contents = [str(content).strip() for content in dialogue]
                if any(not content for content in contents):
                    skipped += 1
                    continue
                turns = [("user" if i % 2 == 0 else "assistant", content)
                         for i, content in enumerate(contents)]
                if turns and turns[-1][0] == "user":
                    turns = turns[:-1]
                soda_conversations.append(turns if len(turns) >= 2 else None)
                if len(soda_conversations) == 1024:
                    emit_conversations(out, soda_conversations, ["soda"] * len(soda_conversations))
                    soda_conversations = []
            if soda_conversations:
                emit_conversations(out, soda_conversations, ["soda"] * len(soda_conversations))
            print(f"soda: {source_counts['soda']} unique conversations", flush=True)

    os.replace(tmp_out, args.out)

    size = os.path.getsize(args.out)
    soda_share = source_counts["soda"] / written if written else 0
    manifest = {
        "schema": "alpha-sft-corpus-v2",
        "output": file_record(args.out),
        "sources": [file_record(path) for path in source_files],
        "tokenizer": file_record(args.hf_tokenizer_json),
        "counts": dict(sorted(source_counts.items())),
        "source_spans": source_spans,
        "total": written,
        "skipped": skipped,
        "exact_duplicates_removed": duplicates,
        "conversation_prefixes_trimmed": prefix_trimmed,
        "soda_seed": args.soda_seed,
        "soda_share": soda_share,
        "max_chars": args.max_chars,
        "max_tokens": args.max_tokens,
        "format": "<|user|> ... <|assistant|> ... <|end_of_text|>",
    }
    manifest_tmp = manifest_path + ".tmp"
    Path(manifest_tmp).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(
        f"TOTAL: {written} conversations, {skipped} skipped, {duplicates} duplicates removed, "
        f"SODA={soda_share:.3%}, {size/1e6:.1f} MB -> {args.out}",
    )
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
