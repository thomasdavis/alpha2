#!/usr/bin/env python3
"""Build Alpha's chat SFT corpus from smol-smoltalk + oasst2 (English, best-ranked paths).

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
    --out /mnt/donto-data/alpha-corpora/sft-text/sft.txt [--max-chars 8000]
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import pyarrow.parquet as pq

U, A, END = "<|user|>", "<|assistant|>", "<|end_of_text|>"
MARKERS = (U, A, END)


def render(turns, max_chars):
    """turns: list of (role, content) with roles 'user'/'assistant', alternating, user-first."""
    parts = []
    for role, content in turns:
        content = " ".join(content.split())  # single line; no marker collisions
        if any(m in content for m in MARKERS):
            return None
        parts.append(f"{U if role == 'user' else A} {content}")
    line = " ".join(parts) + f" {END}"
    if len(line) > max_chars or len(turns) < 2:
        return None
    return line


def convo_from_messages(msgs):
    """smol-smoltalk row -> alternating user-first turns (system folded into first user)."""
    system = None
    turns = []
    for m in msgs:
        role, content = m["role"], (m["content"] or "").strip()
        if not content:
            return None
        if role == "system":
            if turns:
                return None
            system = content
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
    if system:
        turns[0] = ("user", f"[Instructions: {system}]\n\n{turns[0][1]}")
    return turns


def oasst2_paths(rows):
    """Reconstruct trees; yield the best-ranked English root->leaf path per tree."""
    by_id = {r["message_id"]: r for r in rows}
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        # smol-smoltalk (train split only; test kept out for eval)
        for path in sorted(glob.glob(os.path.join(args.smoltalk, "train-*.parquet"))):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1024, columns=["messages"]):
                for msgs in batch.column("messages").to_pylist():
                    turns = convo_from_messages(msgs)
                    line = render(turns, args.max_chars) if turns else None
                    if line:
                        out.write(line + "\n")
                        written += 1
                    else:
                        skipped += 1
        smoltalk_written = written
        print(f"smol-smoltalk: {smoltalk_written} conversations ({skipped} skipped)", flush=True)

        # oasst2 English best-ranked paths (train split)
        o_files = [p for p in glob.glob(os.path.join(args.oasst2, "**/*.parquet"), recursive=True)
                   if "train" in os.path.basename(p)]
        rows = []
        cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "review_result",
                "review_count", "deleted"]
        for path in o_files:
            rows.extend(pq.read_table(path, columns=cols).to_pylist())
        for turns in oasst2_paths(rows):
            line = render(turns, args.max_chars)
            if line:
                out.write(line + "\n")
                written += 1
            else:
                skipped += 1
        print(f"oasst2: {written - smoltalk_written} conversations", flush=True)

    size = os.path.getsize(args.out)
    print(f"TOTAL: {written} conversations, {skipped} skipped, {size/1e6:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
