#!/usr/bin/env python3
"""Freeze disjoint development and final chat suites for repair-v2.

The source pool was already held out from SFT construction. This builder also
excludes every prompt used in the first published frozen evaluation, verifies
1,024-token generation eligibility, rejects exact full-conversation overlap
with the new training corpus, and samples deterministically across source,
prompt-length, and turn-count strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer


USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end_of_text|>"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def parse_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"roles do not alternate at {index}")
        marker = USER if expected == "user" else ASSISTANT
        parts.append(f"{marker} {message['content']}")
    if not parts or messages[-1]["role"] != "user":
        raise ValueError("prompt does not end with user")
    return " ".join(parts) + f" {ASSISTANT}"


def render_full(messages: list[dict[str, str]], reference: str) -> str:
    prompt = render_prompt(messages)
    return f"{prompt} {reference} {END}"


def length_bin(tokens: int) -> str:
    if tokens <= 127:
        return "000-127"
    if tokens <= 255:
        return "128-255"
    if tokens <= 511:
        return "256-511"
    return "512+"


def turn_bin(messages: list[dict[str, str]]) -> str:
    turns = len(messages)
    if turns <= 1:
        return "1"
    if turns <= 3:
        return "2-3"
    if turns <= 5:
        return "4-5"
    return "6+"


def rank(seed: str, suite: str, identifier: str) -> str:
    return sha256_bytes(f"{seed}\0{suite}\0{identifier}".encode())


def stratified_select(
    candidates: list[dict[str, object]], quotas: dict[str, int], seed: str, suite: str
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for source, quota in quotas.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in candidates:
            if row["source"] != source:
                continue
            key = f"{length_bin(int(row['prompt_tokens']))}:{turn_bin(row['messages'])}"
            groups[key].append(row)
        for group in groups.values():
            group.sort(key=lambda row: rank(seed, suite, str(row["id"])))
        ordered_keys = sorted(groups)
        source_selected: list[dict[str, object]] = []
        while len(source_selected) < quota:
            advanced = False
            for key in ordered_keys:
                if groups[key]:
                    source_selected.append(groups[key].pop(0))
                    advanced = True
                    if len(source_selected) >= quota:
                        break
            if not advanced:
                break
        if len(source_selected) != quota:
            raise ValueError(f"{suite} source {source} supplied {len(source_selected)} of required {quota}")
        selected.extend(source_selected)
    selected.sort(key=lambda row: rank(seed, f"{suite}:order", str(row["id"])))
    return selected


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prior-final", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev-corpus", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="alpha-chat-repair-v2-eval")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--generation-reserve", type=int, default=128)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    candidates_path = Path(args.candidates).resolve()
    prior_final_path = Path(args.prior_final).resolve()
    train_path = Path(args.train).resolve()
    dev_corpus_path = Path(args.dev_corpus).resolve()
    candidates = parse_jsonl(candidates_path)
    prior_ids = {str(row["id"]) for row in parse_jsonl(prior_final_path)}
    training_hashes = {
        sha256_bytes(line.encode())
        for path in (train_path, dev_corpus_path)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }
    maximum_prompt_tokens = args.block_size - args.generation_reserve
    eligible: list[dict[str, object]] = []
    exclusions = defaultdict(int)
    for row in candidates:
        if str(row["id"]) in prior_ids:
            exclusions["prior_frozen_final"] += 1
            continue
        rendered_prompt = render_prompt(row["messages"])
        prompt_tokens = len(tokenizer.encode(rendered_prompt).ids)
        if prompt_tokens > maximum_prompt_tokens:
            exclusions["insufficient_generation_reserve"] += 1
            continue
        full = render_full(row["messages"], str(row["reference"]))
        if sha256_bytes(full.encode()) in training_hashes:
            exclusions["exact_training_conversation_overlap"] += 1
            continue
        eligible.append({**row, "prompt_tokens": prompt_tokens})

    dev_quotas = {
        "everyday-conversations": 16,
        "oasst2-validation": 24,
        "smol-magpie-ultra-short": 56,
    }
    final_quotas = {
        "everyday-conversations": 25,
        "oasst2-validation": 25,
        "smol-magpie-ultra-short": 100,
    }
    panel_quotas = {
        "everyday-conversations": 2,
        "oasst2-validation": 4,
        "smol-magpie-ultra-short": 6,
    }
    development = stratified_select(eligible, dev_quotas, args.seed, "development")
    qualitative_panel = stratified_select(development, panel_quotas, args.seed, "qualitative-panel")
    development_ids = {str(row["id"]) for row in development}
    final_pool = [row for row in eligible if str(row["id"]) not in development_ids]
    final = stratified_select(final_pool, final_quotas, args.seed, "sealed-final")
    if development_ids & {str(row["id"]) for row in final}:
        raise AssertionError("development/final overlap")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    development_path = out_dir / "development-chat-prompts.jsonl"
    panel_path = out_dir / "development-qualitative-panel.jsonl"
    final_path = out_dir / "sealed-final-chat-prompts.jsonl"
    write_atomic(development_path, "\n".join(json.dumps(row, sort_keys=True) for row in development) + "\n")
    write_atomic(panel_path, "\n".join(json.dumps(row, sort_keys=True) for row in qualitative_panel) + "\n")
    write_atomic(final_path, "\n".join(json.dumps(row, sort_keys=True) for row in final) + "\n")
    manifest = {
        "schema": "alpha-chat-repair-v2-eval-freeze-v1",
        "status": "development-visible; final-sealed-unexecuted",
        "seed": args.seed,
        "contract": {
            "blockSize": args.block_size,
            "minimumGenerationPositions": args.generation_reserve,
            "priorFinalIdsExcluded": len(prior_ids),
            "exactFullConversationTrainingOverlapAllowed": 0,
            "developmentQuotas": dev_quotas,
            "qualitativePanelQuotas": panel_quotas,
            "sealedFinalQuotas": final_quotas,
            "statisticalUnit": "prompt conversation",
            "checkpointSelection": "development only",
            "sealedFinalExecution": "exactly once after one checkpoint is selected",
        },
        "exclusions": dict(sorted(exclusions.items())),
        "eligibleCandidateRows": len(eligible),
        "inputs": {
            "candidates": sha256_file(candidates_path),
            "priorFinal": sha256_file(prior_final_path),
            "train": sha256_file(train_path),
            "devCorpus": sha256_file(dev_corpus_path),
            "tokenizer": sha256_file(Path(args.tokenizer).resolve()),
        },
        "outputs": {
            "development": sha256_file(development_path),
            "qualitativePanel": sha256_file(panel_path),
            "sealedFinal": sha256_file(final_path),
        },
    }
    manifest_path = out_dir / "eval-freeze-manifest.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
