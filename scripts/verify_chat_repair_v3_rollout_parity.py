#!/usr/bin/env python3
"""Fail closed unless accelerated v3 rollouts match native token trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--accelerated", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--logit-tolerance", type=float, default=1e-3)
    parser.add_argument("--probability-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.expected_rows <= 0 or args.logit_tolerance <= 0 or args.probability_tolerance <= 0:
        parser.error("expected rows and tolerances must be positive")

    native_path = Path(args.native).resolve()
    accelerated_path = Path(args.accelerated).resolve()
    output_path = Path(args.out).resolve()
    native = read_jsonl(native_path)
    accelerated = read_jsonl(accelerated_path)
    if len(native) != args.expected_rows or len(accelerated) != args.expected_rows:
        raise ValueError(
            f"parity population mismatch: native={len(native)} accelerated={len(accelerated)} expected={args.expected_rows}"
        )

    checkpoint_ids = {row.get("checkpoint_sha256") for row in native + accelerated}
    hf_model_ids = {row.get("hf_model_sha256") for row in accelerated}
    if len(checkpoint_ids) != 1 or None in checkpoint_ids:
        raise ValueError("native/accelerated parity rows do not bind one checkpoint SHA-256")
    if len(hf_model_ids) != 1 or None in hf_model_ids:
        raise ValueError("accelerated parity rows do not bind one HF model SHA-256")

    compared_tokens = 0
    max_chosen_logit_delta = 0.0
    max_runner_logit_delta = 0.0
    max_logsumexp_delta = 0.0
    max_probability_delta = 0.0
    for index, (left, right) in enumerate(zip(native, accelerated), start=1):
        label = f"row {index} ({left.get('stable_id')})"
        for key in (
            "stable_id", "positive_conversation_sha256", "prompt_sha256", "checkpoint_sha256",
            "prompt_token_ids", "generated_token_ids", "content_token_ids", "text", "stop_reason",
            "stop_token_id", "eos_terminated", "degenerate_loop", "output_sha256",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(f"{label} differs at {key}")
        if abs(float(left["four_gram_repeat_rate"]) - float(right["four_gram_repeat_rate"])) > 1e-15:
            raise ValueError(f"{label} differs at four_gram_repeat_rate")
        left_audits = left.get("token_audit")
        right_audits = right.get("token_audit")
        if not isinstance(left_audits, list) or not isinstance(right_audits, list) or len(left_audits) != len(right_audits):
            raise ValueError(f"{label} token audit length mismatch")
        for token_index, (native_audit, accelerated_audit) in enumerate(zip(left_audits, right_audits)):
            if native_audit.get("token_id") != accelerated_audit.get("token_id"):
                raise ValueError(f"{label} token {token_index} chosen-token mismatch")
            if native_audit.get("runner_up_token_id") != accelerated_audit.get("runner_up_token_id"):
                raise ValueError(f"{label} token {token_index} runner-up mismatch")
            chosen_delta = abs(float(native_audit["chosen_logit"]) - float(accelerated_audit["chosen_logit"]))
            runner_delta = abs(float(native_audit["runner_up_logit"]) - float(accelerated_audit["runner_up_logit"]))
            lse_delta = abs(float(native_audit["logsumexp"]) - float(accelerated_audit["logsumexp"]))
            probability_delta = abs(
                float(native_audit["chosen_probability"]) - float(accelerated_audit["chosen_probability"])
            )
            max_chosen_logit_delta = max(max_chosen_logit_delta, chosen_delta)
            max_runner_logit_delta = max(max_runner_logit_delta, runner_delta)
            max_logsumexp_delta = max(max_logsumexp_delta, lse_delta)
            max_probability_delta = max(max_probability_delta, probability_delta)
            if chosen_delta >= args.logit_tolerance or runner_delta >= args.logit_tolerance or lse_delta >= args.logit_tolerance:
                raise ValueError(f"{label} token {token_index} logit/logsumexp drift exceeds tolerance")
            if probability_delta >= args.probability_tolerance:
                raise ValueError(f"{label} token {token_index} probability drift exceeds tolerance")
            compared_tokens += 1

    report = {
        "schema": "alpha-rcr-ul-rollout-parity-v1",
        "status": "PASS",
        "native": {"path": str(native_path), "sha256": sha256_file(native_path), "rows": len(native)},
        "accelerated": {
            "path": str(accelerated_path),
            "sha256": sha256_file(accelerated_path),
            "rows": len(accelerated),
        },
        "compared_tokens": compared_tokens,
        "checkpoint_sha256": next(iter(checkpoint_ids)),
        "hf_model_sha256": next(iter(hf_model_ids)),
        "stable_ids_sha256": hashlib.sha256(
            json.dumps([row["stable_id"] for row in native], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "requirements": {
            "exact_token_trajectories": True,
            "exact_runner_up_ids": True,
            "logit_tolerance": args.logit_tolerance,
            "probability_tolerance": args.probability_tolerance,
        },
        "observed_maxima": {
            "chosen_logit_abs_delta": max_chosen_logit_delta,
            "runner_up_logit_abs_delta": max_runner_logit_delta,
            "logsumexp_abs_delta": max_logsumexp_delta,
            "chosen_probability_abs_delta": max_probability_delta,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
