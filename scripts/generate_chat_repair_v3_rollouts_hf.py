#!/usr/bin/env python3
"""Batch Alpha repair-v3 rollouts with the parity-proven Transformers export.

This is an acceleration path for the immutable train-only rollout ledger. It
does not alter the training objective. Admission requires token-trajectory
parity against the native decoder before its output may become canonical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def four_gram_repeat_rate(tokens: list[int]) -> float:
    total = max(0, len(tokens) - 3)
    if total == 0:
        return 0.0
    seen: set[tuple[int, int, int, int]] = set()
    repeated = 0
    for index in range(total):
        gram = tuple(tokens[index : index + 4])
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / total


def validate_candidates(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rollout candidate file is empty")
    identities: set[str] = set()
    for index, row in enumerate(rows, start=1):
        label = f"candidate {index}"
        if row.get("schema") != "alpha-rcr-ul-rollout-candidate-v1":
            raise ValueError(f"{label} has unexpected schema")
        stable_id = row.get("stable_id")
        positive_id = row.get("positive_conversation_sha256")
        if not isinstance(stable_id, str) or not isinstance(positive_id, str):
            raise ValueError(f"{label} lacks stable identity")
        validate_sha256(stable_id, f"{label} stable_id")
        if stable_id != positive_id:
            raise ValueError(f"{label} stable identity drift")
        if stable_id in identities:
            raise ValueError(f"{label} duplicates {stable_id}")
        identities.add(stable_id)
        prompt = row.get("prompt")
        prompt_sha256 = row.get("prompt_sha256")
        token_ids = row.get("prompt_token_ids")
        if not isinstance(prompt, str) or sha256_text(prompt) != prompt_sha256:
            raise ValueError(f"{label} prompt hash mismatch")
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token, int) or token < 0 for token in token_ids
        ):
            raise ValueError(f"{label} has invalid prompt tokens")
        if len(token_ids) != row.get("prompt_tokens"):
            raise ValueError(f"{label} prompt token count drift")


def validate_existing(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    native_sha256: str,
    model_sha256: str,
) -> None:
    if len(rows) > len(candidates):
        raise ValueError("raw rollout file has more rows than candidates")
    for index, row in enumerate(rows):
        candidate = candidates[index]
        label = f"existing rollout {index + 1}"
        if row.get("schema") != "alpha-rcr-ul-raw-rollout-v1":
            raise ValueError(f"{label} has unexpected schema")
        if row.get("stable_id") != candidate["stable_id"] or row.get("prompt_sha256") != candidate["prompt_sha256"]:
            raise ValueError(f"{label} no longer matches candidate order")
        if row.get("checkpoint_sha256") != native_sha256 or row.get("hf_model_sha256") != model_sha256:
            raise ValueError(f"{label} checkpoint/export drift")
        generated = row.get("generated_token_ids")
        audits = row.get("token_audit")
        if not isinstance(generated, list) or not isinstance(audits, list) or len(generated) != len(audits):
            raise ValueError(f"{label} token audit length mismatch")
        expected_hash = sha256_text(json.dumps({
            "prompt_token_ids": row.get("prompt_token_ids"),
            "generated_token_ids": generated,
        }, separators=(",", ":")))
        # TypeScript JSON.stringify has the same compact object encoding.
        if row.get("output_sha256") != expected_hash:
            raise ValueError(f"{label} output hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--expected-native-checkpoint-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_tokens <= 0 or args.stop_after < 0:
        parser.error("batch-size/max-tokens must be positive and stop-after non-negative")
    validate_sha256(args.expected_native_checkpoint_sha256, "expected native checkpoint SHA-256")
    validate_sha256(args.expected_model_sha256, "expected model SHA-256")

    export_dir = Path(args.export_dir).resolve()
    native_checkpoint = Path(args.native_checkpoint).resolve()
    candidates_path = Path(args.candidates).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_path = export_dir / "model.safetensors"
    tokenizer_path = export_dir / "tokenizer.json"
    config_path = export_dir / "config.json"
    for path in (native_checkpoint, candidates_path, model_path, tokenizer_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing or empty: {path}")
    native_sha256 = sha256_file(native_checkpoint)
    model_sha256 = sha256_file(model_path)
    if native_sha256 != args.expected_native_checkpoint_sha256:
        raise ValueError(f"native checkpoint hash mismatch: {native_sha256}")
    if model_sha256 != args.expected_model_sha256:
        raise ValueError(f"HF model hash mismatch: {model_sha256}")

    candidates = read_jsonl(candidates_path)
    validate_candidates(candidates)
    candidate_sha256 = sha256_file(candidates_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "raw-rollouts.jsonl"
    progress_path = out_dir / "progress.json"
    manifest_path = out_dir / "rollout-manifest.json"
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        if not args.resume:
            raise ValueError(f"{output_path} exists; pass --resume to verify and continue")
        existing = read_jsonl(output_path)
        validate_existing(existing, candidates, native_sha256, model_sha256)

    # Import the expensive ML stack only after every immutable file identity
    # and any resumable prefix has passed its cheap fail-closed checks.
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("torch, numpy, and transformers are required") from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu_smoke:
        raise RuntimeError("CUDA is required; --allow-cpu-smoke is only for bounded parity diagnostics")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    try:
        model = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    tokenizer.padding_side = "left"

    def special_id(marker: str) -> int:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{marker} is not atomic in the exported tokenizer")
        return int(ids[0])

    eos_id = special_id("<|end_of_text|>")
    user_id = special_id("<|user|>")
    assistant_id = special_id("<|assistant|>")
    stop_ids = [eos_id, user_id, assistant_id]
    tokenizer.pad_token_id = eos_id
    block_size = int(model.config.max_position_embeddings)
    if block_size != 512:
        raise ValueError(f"HF export context is {block_size}, expected native 512")

    target_end = len(candidates) if args.stop_after == 0 else min(len(candidates), len(existing) + args.stop_after)
    mode = "a" if existing else "x"
    with output_path.open(mode, encoding="utf-8") as output:
        for batch_start in range(len(existing), target_end, args.batch_size):
            batch_candidates = candidates[batch_start : min(target_end, batch_start + args.batch_size)]
            prompt_ids: list[list[int]] = []
            for local_index, candidate in enumerate(batch_candidates):
                reencoded = tokenizer.encode(candidate["prompt"], add_special_tokens=False)
                expected = candidate["prompt_token_ids"]
                if reencoded != expected:
                    raise ValueError(f"candidate {batch_start + local_index + 1} tokenizer drift")
                if expected[-1] != assistant_id:
                    raise ValueError(f"candidate {batch_start + local_index + 1} lacks assistant boundary")
                if len(expected) + args.max_tokens > block_size:
                    raise ValueError(f"candidate {batch_start + local_index + 1} violates generation reserve")
                prompt_ids.append(expected)

            width = max(map(len, prompt_ids))
            padded = [[eos_id] * (width - len(ids)) + ids for ids in prompt_ids]
            masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompt_ids]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=args.max_tokens,
                    eos_token_id=stop_ids,
                    pad_token_id=eos_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            completion_matrix = generated.sequences[:, width:].detach().cpu().tolist()
            scores = generated.scores
            for local_index, (candidate, completion) in enumerate(zip(batch_candidates, completion_matrix)):
                generated_ids: list[int] = []
                content_ids: list[int] = []
                audits: list[dict[str, Any]] = []
                stop_reason = "max_tokens"
                stop_token_id: int | None = None
                for step, token in enumerate(completion):
                    if step >= len(scores):
                        break
                    row_logits = scores[step][local_index].detach().to(torch.float32).cpu().numpy()
                    row_logits = np.asarray(row_logits, dtype="<f4")
                    maximum_id = int(np.argmax(row_logits))
                    if maximum_id != int(token):
                        raise ValueError(
                            f"candidate {batch_start + local_index + 1} step {step} did not preserve raw greedy argmax"
                        )
                    maximum = float(row_logits[maximum_id])
                    runner_copy = row_logits.copy()
                    runner_copy[maximum_id] = -np.inf
                    runner_id = int(np.argmax(runner_copy))
                    runner_logit = float(row_logits[runner_id])
                    shifted = row_logits.astype(np.float64) - maximum
                    logsumexp = maximum + float(np.log(np.exp(shifted).sum()))
                    audits.append({
                        "token_id": int(token),
                        "chosen_logit": maximum,
                        "chosen_probability": float(np.exp(maximum - logsumexp)),
                        "runner_up_token_id": runner_id,
                        "runner_up_logit": runner_logit,
                        "logsumexp": logsumexp,
                        "logits_f32_sha256": sha256_bytes(row_logits.tobytes(order="C")),
                    })
                    generated_ids.append(int(token))
                    if token in stop_ids:
                        stop_token_id = int(token)
                        stop_reason = "learned_eos" if token == eos_id else "role_boundary"
                        break
                    content_ids.append(int(token))
                repeat_rate = four_gram_repeat_rate(content_ids)
                compact_identity = json.dumps({
                    "prompt_token_ids": prompt_ids[local_index],
                    "generated_token_ids": generated_ids,
                }, separators=(",", ":"))
                row = {
                    "schema": "alpha-rcr-ul-raw-rollout-v1",
                    "stable_id": candidate["stable_id"],
                    "source": candidate["source"],
                    "source_id": candidate["source_id"],
                    "positive_conversation_sha256": candidate["positive_conversation_sha256"],
                    "prompt_sha256": candidate["prompt_sha256"],
                    "checkpoint_sha256": native_sha256,
                    "hf_model_sha256": model_sha256,
                    "generation_backend": f"transformers-{device}-fp32",
                    "prompt_token_ids": prompt_ids[local_index],
                    "generated_token_ids": generated_ids,
                    "content_token_ids": content_ids,
                    "token_audit": audits,
                    "text": tokenizer.decode(
                        content_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "stop_reason": stop_reason,
                    "stop_token_id": stop_token_id,
                    "eos_terminated": stop_reason == "learned_eos",
                    "four_gram_repeat_rate": repeat_rate,
                    "degenerate_loop": repeat_rate >= 0.2,
                    "output_sha256": sha256_text(compact_identity),
                }
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                existing.append(row)
            output.flush()
            os.fsync(output.fileno())
            atomic_json(progress_path, {
                "schema": "alpha-rcr-ul-rollout-progress-v1",
                "status": "complete" if len(existing) == len(candidates) else "partial",
                "completed_rows": len(existing),
                "total_rows": len(candidates),
                "checkpoint_sha256": native_sha256,
                "hf_model_sha256": model_sha256,
                "candidates_sha256": candidate_sha256,
            })
            print(f"rollouts {len(existing)}/{len(candidates)}", flush=True)

    complete = len(existing) == len(candidates)
    if complete:
        stop_reasons = Counter(row["stop_reason"] for row in existing)
        atomic_json(manifest_path, {
            "schema": "alpha-rcr-ul-rollout-manifest-v1",
            "status": "complete",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
                "tf32": False,
                "deterministic_algorithms": True,
            },
            "checkpoint": {
                "path": str(native_checkpoint),
                "sha256": native_sha256,
                "control_token_ids": {"eos": eos_id, "user": user_id, "assistant": assistant_id},
            },
            "export": {
                "path": str(export_dir),
                "model_sha256": model_sha256,
                "tokenizer_sha256": sha256_file(tokenizer_path),
                "config_sha256": sha256_file(config_path),
            },
            "candidates": {"path": str(candidates_path), "sha256": candidate_sha256, "rows": len(candidates)},
            "generation": {
                "engine": "stock Transformers LlamaForCausalLM batched cached generation",
                "deterministic_greedy": True,
                "dtype": "float32",
                "max_tokens": args.max_tokens,
                "repetition_penalty": None,
                "minimum_length": None,
                "role_boundary_protection": ["<|user|>", "<|assistant|>"],
                "native_parity_required_before_admission": True,
            },
            "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": len(existing)},
            "summary": {
                "degenerate_loops": sum(bool(row["degenerate_loop"]) for row in existing),
                "stop_reasons": dict(sorted(stop_reasons.items())),
            },
        })
        print(f"complete: {manifest_path}")
    else:
        print(f"partial: {len(existing)}/{len(candidates)}; rerun with --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
