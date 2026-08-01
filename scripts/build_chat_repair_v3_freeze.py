#!/usr/bin/env python3
"""Freeze Alpha chat-repair-v3 rollout candidates and development prompts.

This is a data-boundary builder, not a generator or evaluator. It uses only the
immutable v2 train/development corpus, dynamically balances discovered source
groups, records every selected and excluded identity, and writes hash-bound
artifacts before either v3 arm exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import tokenizers
from tokenizers import Tokenizer


USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end_of_text|>"
ROLE_PATTERN = re.compile(r"<\|(user|assistant)\|> ")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalized_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def rank(seed: str, namespace: str, identifier: str) -> str:
    return sha256_bytes(f"{seed}\0{namespace}\0{identifier}".encode())


def parse_rendered_conversation(line: str) -> tuple[list[dict[str, str]], str, str]:
    suffix = f" {END}"
    if not line.endswith(suffix):
        raise ValueError("rendered conversation does not end with atomic EOT")
    body = line[: -len(suffix)]
    matches = list(ROLE_PATTERN.finditer(body))
    if len(matches) < 2:
        raise ValueError("rendered conversation has fewer than two role turns")
    turns: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = match.group(1)
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"role order fails at turn {index}: {role} != {expected}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[match.end() : end].strip()
        if not content:
            raise ValueError(f"empty {role} content at turn {index}")
        turns.append({"role": role, "content": content})
    if turns[-1]["role"] != "assistant" or len(turns) < 2:
        raise ValueError("rendered conversation does not end with an assistant target")
    prompt_messages = turns[:-1]
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        raise ValueError("derived prompt does not end with a user turn")
    prompt = " ".join(
        f"{USER if message['role'] == 'user' else ASSISTANT} {message['content']}"
        for message in prompt_messages
    ) + f" {ASSISTANT}"
    # The generation boundary ends on the atomic assistant marker, with no
    # standalone trailing space that never appeared at inference training time.
    if prompt.endswith(f"{ASSISTANT} "):
        raise AssertionError("assistant generation boundary contains trailing space")
    return prompt_messages, prompt, turns[-1]["content"]


def load_split_rows(
    corpus_path: Path,
    catalog_path: Path,
    split: str,
) -> list[dict[str, object]]:
    lines = [line for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
    catalog = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_catalog = [row for row in catalog if row.get("split") == split]
    if len(lines) != len(split_catalog):
        raise ValueError(f"{split} corpus/catalog rows differ: {len(lines)} != {len(split_catalog)}")
    result: list[dict[str, object]] = []
    for line_number, (line, metadata) in enumerate(zip(lines, split_catalog), start=1):
        digest = sha256_bytes(line.encode())
        if digest != metadata.get("conversation_sha256"):
            raise ValueError(f"{split} line {line_number} does not match catalog hash")
        messages, prompt, reference = parse_rendered_conversation(line)
        result.append(
            {
                **metadata,
                "line_number": line_number,
                "line": line,
                "messages": messages,
                "prompt": prompt,
                "reference": reference,
            }
        )
    return result


def balanced_select(
    rows: list[dict[str, object]],
    total: int,
    seed: str,
    namespace: str,
    group_key: str = "source",
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: rank(seed, namespace, str(row["conversation_sha256"])))
    selected: list[dict[str, object]] = []
    group_names = sorted(groups)
    while len(selected) < total:
        advanced = False
        for group_name in group_names:
            if groups[group_name]:
                selected.append(groups[group_name].pop(0))
                advanced = True
                if len(selected) == total:
                    break
        if not advanced:
            break
    if len(selected) != total:
        raise ValueError(f"balanced selector supplied {len(selected)} of required {total}")
    return selected


def prompt_from_external_row(row: dict[str, object]) -> str | None:
    raw_prompt = row.get("prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not isinstance(message.get("content"), str):
            return None
        rendered.append(f"{USER if expected == 'user' else ASSISTANT} {message['content']}")
    if messages[-1].get("role") != "user":
        return None
    return " ".join(rendered) + f" {ASSISTANT}"


def load_exclusion_sets(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    prompts: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"exclusion {path}:{line_number} is not an object")
            for key in ("id", "stable_id", "conversation_sha256"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    ids.add(value)
            prompt = prompt_from_external_row(row)
            if prompt:
                prompts.add(normalized_prompt(prompt))
    return ids, prompts


def quantile_thresholds(values: list[int]) -> tuple[int, int, int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute quantiles from an empty list")
    return tuple(ordered[round((len(ordered) - 1) * q)] for q in (0.25, 0.5, 0.75))  # type: ignore[return-value]


def quantile_bin(value: int, thresholds: tuple[int, int, int]) -> str:
    if value <= thresholds[0]:
        return "q1"
    if value <= thresholds[1]:
        return "q2"
    if value <= thresholds[2]:
        return "q3"
    return "q4"


def panel_select(
    rows: list[dict[str, object]],
    total: int,
    thresholds: tuple[int, int, int],
    seed: str,
) -> list[dict[str, object]]:
    """Balance the panel by source first, then prompt-length quantile within source."""
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    source_names = sorted(by_source)
    quotas = Counter[str]()
    remaining_capacity = {source: len(by_source[source]) for source in source_names}
    while sum(quotas.values()) < total:
        advanced = False
        for source in source_names:
            if quotas[source] < remaining_capacity[source]:
                quotas[source] += 1
                advanced = True
                if sum(quotas.values()) == total:
                    break
        if not advanced:
            break
    if sum(quotas.values()) != total:
        raise ValueError(f"panel source allocation supplied {sum(quotas.values())} of {total}")

    selected: list[dict[str, object]] = []
    for source in source_names:
        quantile_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in by_source[source]:
            quantile_groups[quantile_bin(int(row["prompt_tokens"]), thresholds)].append(row)
        for group in quantile_groups.values():
            group.sort(key=lambda row: rank(seed, "development-panel", str(row["conversation_sha256"])))
        group_names = sorted(quantile_groups)
        source_selected: list[dict[str, object]] = []
        while len(source_selected) < quotas[source]:
            advanced = False
            for group_name in group_names:
                if quantile_groups[group_name]:
                    source_selected.append(quantile_groups[group_name].pop(0))
                    advanced = True
                    if len(source_selected) == quotas[source]:
                        break
            if not advanced:
                break
        if len(source_selected) != quotas[source]:
            raise ValueError(f"panel source {source} supplied {len(source_selected)} of {quotas[source]}")
        selected.extend(source_selected)
    selected.sort(key=lambda row: rank(seed, "development-panel-order", str(row["conversation_sha256"])))
    return selected


def output_record(row: dict[str, object], schema: str) -> dict[str, object]:
    return {
        "schema": schema,
        "id": f"v3-{str(row['conversation_sha256'])[:20]}",
        "stable_id": row["conversation_sha256"],
        "source": row["source"],
        "source_id": row["source_id"],
        "messages": row["messages"],
        "prompt": row["prompt"],
        "prompt_sha256": sha256_bytes(str(row["prompt"]).encode()),
        "normalized_prompt_sha256": sha256_bytes(normalized_prompt(str(row["prompt"])).encode()),
        "prompt_tokens": row["prompt_tokens"],
        "reference": row["reference"],
        "reference_sha256": sha256_bytes(str(row["reference"]).encode()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="alpha-chat-repair-v3-freeze")
    parser.add_argument("--rollout-count", type=int, default=4096)
    parser.add_argument("--development-count", type=int, default=96)
    parser.add_argument("--panel-count", type=int, default=24)
    # The selected native checkpoint was trained and serialized with a 512-token
    # context. Keep that as the default so a cohort cannot silently authorize a
    # context migration merely because a later experiment used 1,024 tokens.
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--generation-reserve", type=int, default=128)
    args = parser.parse_args()
    if args.rollout_count <= 0 or args.development_count <= 0 or args.panel_count <= 0:
        parser.error("selection counts must be positive")
    if args.panel_count > args.development_count:
        parser.error("panel count cannot exceed development count")
    if args.generation_reserve <= 0 or args.generation_reserve >= args.block_size:
        parser.error("generation reserve must be in (0, block-size)")

    train_path = Path(args.train).resolve()
    dev_path = Path(args.dev).resolve()
    catalog_path = Path(args.catalog).resolve()
    manifest_path = Path(args.corpus_manifest).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    exclusion_paths = [Path(path).resolve() for path in args.exclude_jsonl]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    maximum_prompt_tokens = args.block_size - args.generation_reserve

    train_rows = load_split_rows(train_path, catalog_path, "train")
    dev_rows = load_split_rows(dev_path, catalog_path, "dev")

    rollout_eligible: list[dict[str, object]] = []
    rollout_exclusions: list[dict[str, object]] = []
    for row in train_rows:
        prompt_ids = tokenizer.encode(str(row["prompt"])).ids
        reason: str | None = None
        if len(prompt_ids) > maximum_prompt_tokens:
            reason = "insufficient_generation_reserve"
        elif not str(row["reference"]).strip():
            reason = "empty_positive_target"
        if reason:
            rollout_exclusions.append(
                {
                    "schema": "alpha-rcr-ul-rollout-exclusion-v1",
                    "stable_id": row["conversation_sha256"],
                    "source": row["source"],
                    "line_sha256": row["conversation_sha256"],
                    "prompt_sha256": sha256_bytes(str(row["prompt"]).encode()),
                    "reason": reason,
                }
            )
            continue
        rollout_eligible.append({**row, "prompt_tokens": len(prompt_ids), "prompt_token_ids": prompt_ids})

    rollout_selected = balanced_select(
        rollout_eligible, args.rollout_count, args.seed, "rollout-candidate"
    )
    rollout_selected_ids = {str(row["conversation_sha256"]) for row in rollout_selected}
    for row in rollout_eligible:
        if str(row["conversation_sha256"]) not in rollout_selected_ids:
            rollout_exclusions.append(
                {
                    "schema": "alpha-rcr-ul-rollout-exclusion-v1",
                    "stable_id": row["conversation_sha256"],
                    "source": row["source"],
                    "line_sha256": row["conversation_sha256"],
                    "prompt_sha256": sha256_bytes(str(row["prompt"]).encode()),
                    "reason": "not_selected_by_balanced_rank",
                }
            )

    rollout_records = [
        {
            "schema": "alpha-rcr-ul-rollout-candidate-v1",
            "stable_id": row["conversation_sha256"],
            "source": row["source"],
            "source_id": row["source_id"],
            "positive_line_number": row["line_number"],
            "positive_conversation_sha256": row["conversation_sha256"],
            "prompt": row["prompt"],
            "prompt_sha256": sha256_bytes(str(row["prompt"]).encode()),
            "prompt_token_ids": row["prompt_token_ids"],
            "prompt_tokens": row["prompt_tokens"],
            "positive_response": row["reference"],
            "positive_response_sha256": sha256_bytes(str(row["reference"]).encode()),
        }
        for row in rollout_selected
    ]

    prior_ids, prior_prompts = load_exclusion_sets(exclusion_paths)
    # The rollout cohort is also excluded from selector by identity and prompt.
    prior_ids.update(rollout_selected_ids)
    prior_prompts.update(normalized_prompt(str(row["prompt"])) for row in rollout_selected)

    development_eligible: list[dict[str, object]] = []
    development_exclusions: list[dict[str, object]] = []
    for row in dev_rows:
        prompt = str(row["prompt"])
        prompt_ids = tokenizer.encode(prompt).ids
        stable_id = str(row["conversation_sha256"])
        reason: str | None = None
        if stable_id in prior_ids:
            reason = "previously_exposed_identity"
        elif normalized_prompt(prompt) in prior_prompts:
            reason = "previously_exposed_normalized_prompt"
        elif len(prompt_ids) > maximum_prompt_tokens:
            reason = "insufficient_generation_reserve"
        elif not str(row["reference"]).strip():
            reason = "empty_source_assistant_response"
        if reason:
            development_exclusions.append(
                {
                    "schema": "alpha-chat-repair-v3-development-exclusion-v1",
                    "stable_id": stable_id,
                    "source": row["source"],
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "reason": reason,
                }
            )
        else:
            development_eligible.append({**row, "prompt_tokens": len(prompt_ids)})

    development = balanced_select(
        development_eligible, args.development_count, args.seed, "development-selector"
    )
    thresholds = quantile_thresholds([int(row["prompt_tokens"]) for row in development])
    panel = panel_select(development, args.panel_count, thresholds, args.seed)
    development_records = [output_record(row, "alpha-chat-repair-v3-development-v1") for row in development]
    panel_records = [
        {
            **output_record(row, "alpha-chat-repair-v3-panel-v1"),
            "prompt_length_quantile": quantile_bin(int(row["prompt_tokens"]), thresholds),
        }
        for row in panel
    ]

    if rollout_selected_ids & {str(row["conversation_sha256"]) for row in development}:
        raise AssertionError("rollout/development identity overlap")
    if {normalized_prompt(str(row["prompt"])) for row in rollout_selected} & {
        normalized_prompt(str(row["prompt"])) for row in development
    }:
        raise AssertionError("rollout/development normalized prompt overlap")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = out_dir / "rollout-candidates.jsonl"
    positive_path = out_dir / "positive-cohort.txt"
    rollout_exclusions_path = out_dir / "rollout-exclusions.jsonl"
    development_path = out_dir / "development-selector.jsonl"
    panel_path = out_dir / "development-panel.jsonl"
    development_exclusions_path = out_dir / "development-exclusions.jsonl"
    write_atomic(rollout_path, "\n".join(json.dumps(row, sort_keys=True) for row in rollout_records) + "\n")
    write_atomic(positive_path, "\n".join(str(row["line"]) for row in rollout_selected) + "\n")
    write_atomic(
        rollout_exclusions_path,
        "\n".join(json.dumps(row, sort_keys=True) for row in rollout_exclusions) + "\n",
    )
    write_atomic(
        development_path,
        "\n".join(json.dumps(row, sort_keys=True) for row in development_records) + "\n",
    )
    write_atomic(panel_path, "\n".join(json.dumps(row, sort_keys=True) for row in panel_records) + "\n")
    write_atomic(
        development_exclusions_path,
        "\n".join(json.dumps(row, sort_keys=True) for row in development_exclusions) + "\n",
    )

    selected_source_counts = Counter(str(row["source"]) for row in rollout_selected)
    development_source_counts = Counter(str(row["source"]) for row in development)
    panel_group_counts = Counter(
        f"{row['source']}:{quantile_bin(int(row['prompt_tokens']), thresholds)}" for row in panel
    )
    manifest = {
        "schema": "alpha-chat-repair-v3-freeze-v1",
        "status": "rollout-candidates-and-development-frozen; no-rollouts-generated; final-sealed",
        "seed": args.seed,
        "contract": {
            "block_size": args.block_size,
            "generation_reserve": args.generation_reserve,
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "rollout_count": args.rollout_count,
            "development_count": args.development_count,
            "panel_count": args.panel_count,
            "source_allocation": "round-robin over dynamically discovered source groups after per-group SHA256 ranking",
            "selection_rank": "SHA256(seed, namespace, conversation_sha256)",
            "panel_length_bins": "quartiles computed from the frozen 96-selector prompt-token distribution",
            "normalization": "Unicode NFKC, whitespace collapse, casefold",
            "assistant_boundary": "atomic assistant marker with no trailing standalone space",
            "statistical_unit": "conversation identity",
        },
        "counts": {
            "train_rows": len(train_rows),
            "rollout_eligible": len(rollout_eligible),
            "rollout_selected": len(rollout_selected),
            "rollout_excluded": len(rollout_exclusions),
            "rollout_selected_by_source": dict(sorted(selected_source_counts.items())),
            "dev_rows": len(dev_rows),
            "development_eligible": len(development_eligible),
            "development_selected": len(development),
            "development_excluded": len(development_exclusions),
            "development_selected_by_source": dict(sorted(development_source_counts.items())),
            "panel_groups": dict(sorted(panel_group_counts.items())),
        },
        "contamination": {
            "rollout_development_identity_overlap": 0,
            "rollout_development_normalized_prompt_overlap": 0,
            "external_exclusion_ids": len(prior_ids - rollout_selected_ids),
            "external_exclusion_normalized_prompts": len(prior_prompts - {
                normalized_prompt(str(row["prompt"])) for row in rollout_selected
            }),
        },
        "dependencies": {
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
        },
        "inputs": {
            "train": sha256_file(train_path),
            "dev": sha256_file(dev_path),
            "catalog": sha256_file(catalog_path),
            "corpus_manifest": sha256_file(manifest_path),
            "tokenizer": sha256_file(tokenizer_path),
            "exclusions": [sha256_file(path) for path in exclusion_paths],
        },
        "outputs": {
            "rollout_candidates": sha256_file(rollout_path),
            "positive_cohort": sha256_file(positive_path),
            "rollout_exclusions": sha256_file(rollout_exclusions_path),
            "development_selector": sha256_file(development_path),
            "development_panel": sha256_file(panel_path),
            "development_exclusions": sha256_file(development_exclusions_path),
        },
    }
    freeze_manifest_path = out_dir / "freeze-manifest.json"
    write_atomic(freeze_manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
