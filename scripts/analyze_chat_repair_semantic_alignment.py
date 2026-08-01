#!/usr/bin/env python3
"""Measure prompt/output semantic contingency over preserved Alpha generations.

This is a read-only evaluator. It embeds frozen prompts, frozen source responses,
and already-generated Alpha outputs. It does not invoke Alpha or mutate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed(model: TextEmbedding, texts: list[str]) -> np.ndarray:
    return normalize(np.asarray(list(model.embed(texts)), dtype=np.float32))


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def retrieval_metrics(
    prompt_vectors: np.ndarray,
    response_vectors: np.ndarray,
    sources: list[str],
) -> dict[str, Any]:
    similarities = response_vectors @ prompt_vectors.T
    diagonal = np.diag(similarities)
    ranks = np.asarray([
        1 + int(np.count_nonzero(similarities[index] > diagonal[index]))
        for index in range(len(diagonal))
    ])
    unrelated_means = np.asarray([
        (float(np.sum(row)) - float(row[index])) / max(1, len(row) - 1)
        for index, row in enumerate(similarities)
    ])
    unrelated_max = np.asarray([
        float(np.max(np.delete(row, index))) if len(row) > 1 else float(row[index])
        for index, row in enumerate(similarities)
    ])

    within_source_ranks: list[int] = []
    for index, source in enumerate(sources):
        candidates = [position for position, value in enumerate(sources) if value == source]
        local = similarities[index, candidates]
        correct = candidates.index(index)
        within_source_ranks.append(1 + int(np.count_nonzero(local > local[correct])))
    within = np.asarray(within_source_ranks)

    return {
        "count": len(diagonal),
        "pairedCosine": {
            "mean": float(np.mean(diagonal)),
            "q25": quantile(diagonal, 0.25),
            "median": quantile(diagonal, 0.5),
            "q75": quantile(diagonal, 0.75),
        },
        "unrelatedCosineMean": float(np.mean(unrelated_means)),
        "pairedMinusMeanUnrelated": float(np.mean(diagonal - unrelated_means)),
        "pairedMinusBestUnrelated": float(np.mean(diagonal - unrelated_max)),
        "promptRetrievalTop1": int(np.count_nonzero(ranks == 1)),
        "promptRetrievalMrr": float(np.mean(1.0 / ranks)),
        "promptRetrievalMedianRank": float(np.median(ranks)),
        "withinSourceTop1": int(np.count_nonzero(within == 1)),
        "withinSourceMrr": float(np.mean(1.0 / within)),
        "withinSourceMedianRank": float(np.median(within)),
        "pairedCosines": diagonal.tolist(),
        "ranks": ranks.tolist(),
    }


def bootstrap_mean_delta(values: np.ndarray, seed: int, samples: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[indices], axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95Low": quantile(means, 0.025),
        "ci95High": quantile(means, 0.975),
    }


def source_metrics(
    prompt_vectors: np.ndarray,
    response_vectors: np.ndarray,
    reference_vectors: np.ndarray,
    sources: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sorted(set(sources)):
        indices = np.asarray([index for index, value in enumerate(sources) if value == source])
        prompt_cosines = np.sum(prompt_vectors[indices] * response_vectors[indices], axis=1)
        reference_cosines = np.sum(reference_vectors[indices] * response_vectors[indices], axis=1)
        result[source] = {
            "count": len(indices),
            "promptOutputCosineMean": float(np.mean(prompt_cosines)),
            "outputReferenceCosineMean": float(np.mean(reference_cosines)),
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--transition-analysis", required=True, type=Path)
parser.add_argument("--suite", required=True, type=Path)
parser.add_argument("--out-json", required=True, type=Path)
parser.add_argument("--out-markdown", required=True, type=Path)
parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--bootstrap-samples", type=int, default=10_000)
parser.add_argument("--seed", type=int, default=42)
options = parser.parse_args()

transition = json.loads(options.transition_analysis.read_text(encoding="utf-8"))
suite_rows = read_jsonl(options.suite)
suite = {row["id"]: row for row in suite_rows}
prompt_rows = transition["promptTransitions"]
run_labels = list(prompt_rows[0]["outputs"].keys())

# Use one fixed population for every comparison. The baseline has one empty
# response, so the common-nonempty population has 68 of the 69 eligible IDs.
common_rows = [
    row for row in prompt_rows
    if all(str(row["outputs"][label]["text"]).strip() for label in run_labels)
]
ids = [row["id"] for row in common_rows]
sources = [row["source"] for row in common_rows]
prompts = [row["lastUser"] for row in common_rows]
references = [str(suite[row_id].get("reference", "")) for row_id in ids]

model = TextEmbedding(model_name=options.model, threads=options.threads)
prompt_vectors = embed(model, prompts)
reference_vectors = embed(model, references)

reference_retrieval = retrieval_metrics(prompt_vectors, reference_vectors, sources)
run_vectors: dict[str, np.ndarray] = {}
run_metrics: dict[str, Any] = {}
for label in run_labels:
    outputs = [row["outputs"][label]["text"] for row in common_rows]
    vectors = embed(model, outputs)
    run_vectors[label] = vectors
    retrieval = retrieval_metrics(prompt_vectors, vectors, sources)
    output_reference = np.sum(vectors * reference_vectors, axis=1)
    run_metrics[label] = {
        **{key: value for key, value in retrieval.items() if key not in {"pairedCosines", "ranks"}},
        "outputReferenceCosine": {
            "mean": float(np.mean(output_reference)),
            "q25": quantile(output_reference, 0.25),
            "median": quantile(output_reference, 0.5),
            "q75": quantile(output_reference, 0.75),
        },
        "bySource": source_metrics(prompt_vectors, vectors, reference_vectors, sources),
        "perPrompt": [
            {
                "id": row_id,
                "promptOutputCosine": retrieval["pairedCosines"][index],
                "promptRetrievalRank": retrieval["ranks"][index],
                "outputReferenceCosine": float(output_reference[index]),
            }
            for index, row_id in enumerate(ids)
        ],
    }

baseline_label = run_labels[0]
baseline_prompt_cosines = np.sum(run_vectors[baseline_label] * prompt_vectors, axis=1)
baseline_reference_cosines = np.sum(run_vectors[baseline_label] * reference_vectors, axis=1)
paired_deltas: dict[str, Any] = {}
for label in run_labels[1:]:
    current_prompt = np.sum(run_vectors[label] * prompt_vectors, axis=1)
    current_reference = np.sum(run_vectors[label] * reference_vectors, axis=1)
    paired_deltas[label] = {
        "promptOutputCosine": bootstrap_mean_delta(
            current_prompt - baseline_prompt_cosines,
            options.seed,
            options.bootstrap_samples,
        ),
        "outputReferenceCosine": bootstrap_mean_delta(
            current_reference - baseline_reference_cosines,
            options.seed + 1,
            options.bootstrap_samples,
        ),
    }

report = {
    "schema": "alpha-chat-repair-semantic-alignment-v2",
    "configuration": {
        "embeddingModel": options.model,
        "fastembedVersion": importlib.metadata.version("fastembed"),
        "numpyVersion": np.__version__,
        "threads": options.threads,
        "seed": options.seed,
        "bootstrapSamples": options.bootstrap_samples,
        "population": "baseline-generation-eligible and nonempty in every run",
    },
    "inputs": {
        "transitionAnalysis": {
            "path": str(options.transition_analysis.resolve()),
            "sha256": sha256_file(options.transition_analysis),
        },
        "suite": {
            "path": str(options.suite.resolve()),
            "sha256": sha256_file(options.suite),
        },
    },
    "population": {"ids": ids, "count": len(ids), "sources": sources},
    "referenceCalibration": {
        key: value for key, value in reference_retrieval.items()
        if key not in {"pairedCosines", "ranks"}
    },
    "runs": run_metrics,
    "pairedDeltasFromBaseline": paired_deltas,
}

lines = [
    "# Alpha chat repair semantic-alignment analysis",
    "",
    f"Embedding model: `{options.model}`. Common nonempty eligible prompts: **{len(ids)}**.",
    "",
    "This is a supporting semantic-contingency diagnostic, not a quality judge. It compares each preserved",
    "response with its actual last user turn, all unrelated held-out user turns, and the held-out source response.",
    "",
    "## Calibration",
    "",
    f"Held-out source responses retrieve their own prompts top-1 in "
    f"{reference_retrieval['promptRetrievalTop1']}/{len(ids)} cases; MRR "
    f"{reference_retrieval['promptRetrievalMrr']:.4f}.",
    "",
    "## Run comparison",
    "",
    "| Run | Prompt cosine | Prompt minus unrelated | Prompt top-1 | MRR | Output/reference cosine |",
    "|---|---:|---:|---:|---:|---:|",
]
for label in run_labels:
    metrics = run_metrics[label]
    lines.append(
        f"| {label} | {metrics['pairedCosine']['mean']:.4f} | "
        f"{metrics['pairedMinusMeanUnrelated']:.4f} | "
        f"{metrics['promptRetrievalTop1']}/{len(ids)} | "
        f"{metrics['promptRetrievalMrr']:.4f} | "
        f"{metrics['outputReferenceCosine']['mean']:.4f} |"
    )

lines.extend([
    "",
    "## Paired mean delta from baseline with bootstrap 95% interval",
    "",
    "| Run | Prompt/output delta | 95% interval | Output/reference delta | 95% interval |",
    "|---|---:|---:|---:|---:|",
])
for label in run_labels[1:]:
    prompt_delta = paired_deltas[label]["promptOutputCosine"]
    reference_delta = paired_deltas[label]["outputReferenceCosine"]
    lines.append(
        f"| {label} | {prompt_delta['mean']:.4f} | "
        f"[{prompt_delta['ci95Low']:.4f}, {prompt_delta['ci95High']:.4f}] | "
        f"{reference_delta['mean']:.4f} | "
        f"[{reference_delta['ci95Low']:.4f}, {reference_delta['ci95High']:.4f}] |"
    )

lines.extend([
    "",
    "Embedding similarity cannot determine truth, valid reasoning, or conversational naturalness. Treat it only",
    "as evidence about whether a response remains semantically coupled to the prompt and source response.",
    "",
])

options.out_json.parent.mkdir(parents=True, exist_ok=True)
options.out_markdown.parent.mkdir(parents=True, exist_ok=True)
options.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
options.out_markdown.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({
    "schema": report["schema"],
    "population": len(ids),
    "runs": len(run_labels),
    "outJson": str(options.out_json),
    "outMarkdown": str(options.out_markdown),
}))
