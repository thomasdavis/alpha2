#!/usr/bin/env python3
"""Audit v4 prompt diversity and semantic proximity to declared holdouts.

This is a diagnostic, not an automatic quality judge. It never modifies the
corpus and emits no exclusion list. A human or experiment owner must inspect
the nearest pairs before deciding whether a similarity is contamination or a
legitimate shared topic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding


ROLE = re.compile(r"(<\|user\|>|<\|assistant\|>)")
END = "<|end_of_text|>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def final_user(rendered: str) -> str:
    body = rendered.removesuffix(END).strip()
    parts = ROLE.split(body)
    turns: list[tuple[str, str]] = []
    for index in range(1, len(parts), 2):
        turns.append((parts[index], parts[index + 1].strip()))
    if len(turns) < 2 or turns[-2][0] != "<|user|>" or turns[-1][0] != "<|assistant|>":
        raise ValueError("rendered row does not end in user/assistant")
    return turns[-2][1]


def final_holdout_user(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("holdout row lacks messages")
    users = [message["content"] for message in messages if message.get("role") == "user"]
    if not users:
        raise ValueError("holdout row has no user message")
    return str(users[-1])


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed(model: TextEmbedding, texts: list[str]) -> np.ndarray:
    return normalize(np.asarray(list(model.embed(texts)), dtype=np.float32))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def grouped_summary(
    values: np.ndarray, metadata: list[dict[str, str]], thresholds: list[float]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sorted({row["source"] for row in metadata}):
        selected = np.asarray(
            [value for value, row in zip(values, metadata, strict=True) if row["source"] == source],
            dtype=np.float32,
        )
        result[source] = {
            "count": int(len(selected)),
            "quantiles": quantiles(selected),
            "counts_at_or_above": {
                str(threshold): int(np.count_nonzero(selected >= threshold))
                for threshold in thresholds
            },
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True, type=Path)
parser.add_argument("--holdout", required=True, type=Path, action="append")
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--block", type=int, default=256)
options = parser.parse_args()

manifest_path = options.corpus / "manifest.json"
catalog_path = options.corpus / "catalog.jsonl"
dev_path = options.corpus / "dev.txt"
train_path = options.corpus / "train.txt"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
catalog = jsonl(catalog_path)
dev_lines = text_lines(dev_path)
train_lines = text_lines(train_path)
if len(catalog) != len(dev_lines) + len(train_lines):
    raise ValueError("catalog and rendered row counts differ")

rendered_by_split = {"dev": iter(dev_lines), "train": iter(train_lines)}
candidate_texts: list[str] = []
candidate_meta: list[dict[str, str]] = []
for row in catalog:
    split = row["split"]
    if split not in rendered_by_split:
        raise ValueError(f"unknown split {split}")
    rendered = next(rendered_by_split[split])
    candidate_texts.append(final_user(rendered))
    candidate_meta.append(
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "split": str(split),
        }
    )

holdout_rows: list[dict[str, Any]] = []
for path in options.holdout:
    for row in jsonl(path):
        holdout_rows.append(
            {
                "id": str(row["id"]),
                "path": str(path.resolve()),
                "text": final_holdout_user(row),
            }
        )
if not holdout_rows:
    raise ValueError("no holdout rows")

model = TextEmbedding(model_name=options.model, threads=options.threads)
candidate_vectors = embed(model, candidate_texts)
holdout_vectors = embed(model, [row["text"] for row in holdout_rows])

holdout_similarity = candidate_vectors @ holdout_vectors.T
holdout_best_index = np.argmax(holdout_similarity, axis=1)
holdout_best = np.max(holdout_similarity, axis=1)
holdout_top_indices = np.argsort(-holdout_best)[:100]

nearest_score = np.full(len(candidate_texts), -1.0, dtype=np.float32)
nearest_index = np.full(len(candidate_texts), -1, dtype=np.int64)
for start in range(0, len(candidate_texts), options.block):
    end = min(len(candidate_texts), start + options.block)
    scores = candidate_vectors[start:end] @ candidate_vectors.T
    local = np.arange(end - start)
    scores[local, np.arange(start, end)] = -1.0
    nearest_index[start:end] = np.argmax(scores, axis=1)
    nearest_score[start:end] = np.max(scores, axis=1)
nearest_top_indices = np.argsort(-nearest_score)[:100]

thresholds = [0.80, 0.85, 0.90, 0.95]
report = {
    "schema": "alpha-chat-semantic-repair-v4-embedding-audit-v1",
    "configuration": {
        "model": options.model,
        "fastembed_version": importlib.metadata.version("fastembed"),
        "numpy_version": np.__version__,
        "threads": options.threads,
        "block": options.block,
        "policy": "diagnostic-only; no automatic exclusions",
    },
    "inputs": {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "catalog": {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)},
        "development": {"path": str(dev_path.resolve()), "sha256": sha256_file(dev_path)},
        "train": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
        "holdouts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in options.holdout
        ],
    },
    "population": {
        "candidates": len(candidate_texts),
        "holdouts": len(holdout_rows),
        "manifest_rows": manifest.get("rows"),
    },
    "candidate_to_holdout": {
        "best_similarity_quantiles": quantiles(holdout_best),
        "counts_at_or_above": {
            str(threshold): int(np.count_nonzero(holdout_best >= threshold)) for threshold in thresholds
        },
        "by_source": grouped_summary(holdout_best, candidate_meta, thresholds),
        "top_pairs": [
            {
                "candidate": {**candidate_meta[index], "text": candidate_texts[index]},
                "holdout": holdout_rows[int(holdout_best_index[index])],
                "cosine": float(holdout_best[index]),
            }
            for index in holdout_top_indices
        ],
    },
    "candidate_nearest_neighbor": {
        "similarity_quantiles": quantiles(nearest_score),
        "counts_at_or_above": {
            str(threshold): int(np.count_nonzero(nearest_score >= threshold)) for threshold in thresholds
        },
        "by_source": grouped_summary(nearest_score, candidate_meta, thresholds),
        "top_pairs": [
            {
                "candidate": {**candidate_meta[index], "text": candidate_texts[index]},
                "nearest": {
                    **candidate_meta[int(nearest_index[index])],
                    "text": candidate_texts[int(nearest_index[index])],
                },
                "cosine": float(nearest_score[index]),
            }
            for index in nearest_top_indices
        ],
    },
}

options.out.parent.mkdir(parents=True, exist_ok=True)
temporary = options.out.with_suffix(options.out.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(options.out)
print(
    json.dumps(
        {
            "result": "PASS",
            "out": str(options.out),
            "candidates": len(candidate_texts),
            "holdouts": len(holdout_rows),
        }
    )
)
