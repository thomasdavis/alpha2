#!/usr/bin/env python3
"""Batched deterministic chat-repair-v3 development evaluation.

The evaluator consumes only frozen dialogue histories. Reference responses are
validated as corpus metadata but are never passed to the model or copied into
the raw result ledger.
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


def validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def render_prompt(messages: object, label: str) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label} has no dialogue history")
    parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{label} message {index + 1} is not an object")
        expected = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        content = message.get("content")
        if role != expected:
            raise ValueError(f"{label} message {index + 1} role is {role}, expected {expected}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{label} message {index + 1} has empty content")
        marker = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{marker} {content}")
    if messages[-1].get("role") != "user":
        raise ValueError(f"{label} does not end with a user turn")
    return " ".join(parts) + " <|assistant|>"


def four_gram_repeat_rate(tokens: list[int]) -> float:
    total = max(0, len(tokens) - 3)
    if total == 0:
        return 0.0
    seen: set[tuple[int, int, int, int]] = set()
    repeats = 0
    for index in range(total):
        gram = tuple(tokens[index : index + 4])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / total


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_existing(
    existing: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    checkpoint_sha256: str,
    model_sha256: str,
) -> None:
    if len(existing) > len(prompts):
        raise ValueError("result prefix is longer than the frozen prompt suite")
    for index, row in enumerate(existing):
        prompt = prompts[index]
        label = f"existing result {index + 1}"
        if row.get("schema") != "alpha-chat-repair-v3-eval-row-v1":
            raise ValueError(f"{label} has unexpected schema")
        if row.get("id") != prompt.get("id") or row.get("source") != prompt.get("source"):
            raise ValueError(f"{label} no longer matches frozen prompt order")
        if row.get("checkpointSha256") != checkpoint_sha256 or row.get("hfModelSha256") != model_sha256:
            raise ValueError(f"{label} checkpoint/export identity drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--expected-prompts-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.expected_rows <= 0 or args.checkpoint_step < 0 or args.batch_size <= 0 or args.max_tokens <= 0 or args.stop_after < 0:
        parser.error("expected-rows/batch-size/max-tokens must be positive; stop-after must be non-negative")
    validate_sha256(args.expected_checkpoint_sha256, "expected checkpoint SHA-256")
    validate_sha256(args.expected_prompts_sha256, "expected prompts SHA-256")

    export_dir = Path(args.export_dir).resolve()
    checkpoint_path = Path(args.native_checkpoint).resolve()
    prompts_path = Path(args.prompts).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_path = export_dir / "model.safetensors"
    tokenizer_path = export_dir / "tokenizer.json"
    config_path = export_dir / "config.json"
    for path in (checkpoint_path, prompts_path, model_path, tokenizer_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input missing or empty: {path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    prompts_sha256 = sha256_file(prompts_path)
    model_sha256 = sha256_file(model_path)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_sha256}")
    if prompts_sha256 != args.expected_prompts_sha256:
        raise ValueError(f"prompt-suite hash mismatch: {prompts_sha256}")
    prompts = read_jsonl(prompts_path)
    if len(prompts) != args.expected_rows:
        raise ValueError(f"prompt count is {len(prompts)}, expected {args.expected_rows}")
    identities: set[str] = set()
    rendered_prompts: list[str] = []
    for index, row in enumerate(prompts, start=1):
        label = f"prompt {index}"
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError(f"{label} has missing or duplicate ID")
        identities.add(identity)
        if not isinstance(row.get("source"), str) or not row.get("source"):
            raise ValueError(f"{label} has no source")
        if not isinstance(row.get("reference"), str) or not row.get("reference").strip():
            raise ValueError(f"{label} has no held-out reference metadata")
        rendered = render_prompt(row.get("messages"), label)
        if row.get("prompt") is not None and row.get("prompt") != rendered:
            raise ValueError(f"{label} stored prompt differs from rendered dialogue")
        if row.get("prompt_sha256") is not None and row.get("prompt_sha256") != sha256_text(rendered):
            raise ValueError(f"{label} stored prompt hash differs from rendered dialogue")
        rendered_prompts.append(rendered)

    out_dir.mkdir(parents=True, exist_ok=args.resume)
    results_path = out_dir / "chat-results.jsonl"
    progress_path = out_dir / "progress.json"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"completed summary already exists: {summary_path}")
    existing: list[dict[str, Any]] = []
    if results_path.exists():
        if not args.resume:
            raise ValueError(f"{results_path} exists; pass --resume to verify and continue")
        existing = read_jsonl(results_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("torch and transformers are required") from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu_smoke:
        raise RuntimeError("CUDA is required; --allow-cpu-smoke is only for bounded local diagnostics")
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
    tokenizer.pad_token_id = eos_id
    block_size = int(model.config.max_position_embeddings)
    if block_size != 512:
        raise ValueError(f"export context is {block_size}, expected 512")
    prompt_ids: list[list[int]] = []
    for index, rendered in enumerate(rendered_prompts, start=1):
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not ids or ids[-1] != assistant_id:
            raise ValueError(f"prompt {index} has invalid assistant boundary")
        recorded_count = prompts[index - 1].get("prompt_tokens")
        if recorded_count is not None and abs(int(recorded_count) - len(ids)) > 1:
            raise ValueError(f"prompt {index} token count drift: stored={recorded_count} runtime={len(ids)}")
        if len(ids) >= block_size:
            raise ValueError(f"prompt {index} leaves no generation position in the 512-token context")
        prompt_ids.append([int(token) for token in ids])
    validate_existing(existing, prompts, checkpoint_sha256, model_sha256)

    target_end = len(prompts) if args.stop_after == 0 else min(len(prompts), len(existing) + args.stop_after)
    mode = "a" if existing else "x"
    with results_path.open(mode, encoding="utf-8") as output:
        batch_start = len(existing)
        while batch_start < target_end:
            # Rows with the full 128-token reserve may share a padded batch.
            # Longer legacy-regression prompts must share an exact prompt
            # length; otherwise left padding would consume another row's
            # remaining context and silently shorten its native trajectory.
            first_length = len(prompt_ids[batch_start])
            batch_end = batch_start + 1
            while batch_end < target_end and batch_end - batch_start < args.batch_size:
                candidate_length = len(prompt_ids[batch_end])
                if first_length <= block_size - args.max_tokens:
                    if candidate_length > block_size - args.max_tokens:
                        break
                elif candidate_length != first_length:
                    break
                batch_end += 1
            batch_ids = prompt_ids[batch_start:batch_end]
            width = max(map(len, batch_ids))
            batch_max_tokens = min(args.max_tokens, block_size - width)
            if batch_max_tokens <= 0:
                raise ValueError(f"batch beginning at prompt {batch_start + 1} leaves no generation position")
            padded = [[eos_id] * (width - len(ids)) + ids for ids in batch_ids]
            masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch_ids]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=batch_max_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    use_cache=True,
                )
            completions = generated[:, width:].detach().cpu().tolist()
            for local_index, completion in enumerate(completions):
                row_index = batch_start + local_index
                generated_ids: list[int] = []
                eos_terminated = False
                for token in completion:
                    generated_ids.append(int(token))
                    if token == eos_id:
                        eos_terminated = True
                        break
                content_ids = generated_ids[:-1] if eos_terminated else generated_ids
                text = tokenizer.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                repeat_rate = four_gram_repeat_rate(content_ids)
                role_leak = user_id in content_ids
                nonempty = bool(text.strip())
                hit_block_limit = not eos_terminated and len(prompt_ids[row_index]) + len(generated_ids) >= block_size
                result = {
                    "schema": "alpha-chat-repair-v3-eval-row-v1",
                    "id": prompts[row_index]["id"],
                    "source": prompts[row_index]["source"],
                    "checkpointSha256": checkpoint_sha256,
                    "hfModelSha256": model_sha256,
                    "generationBackend": f"transformers-{device}-fp32",
                    "promptTokens": len(prompt_ids[row_index]),
                    "generatedIds": generated_ids,
                    "text": text,
                    "eosTerminated": eos_terminated,
                    "hitBlockLimit": hit_block_limit,
                    "roleLeak": role_leak,
                    "nonempty": nonempty,
                    "fourGramRepeatRate": repeat_rate,
                    "degenerateLoop": repeat_rate >= 0.2,
                    "structuralPass": eos_terminated and not role_leak and nonempty,
                    "outputSha256": sha256_text(json.dumps({
                        "prompt_token_ids": prompt_ids[row_index],
                        "generated_token_ids": generated_ids,
                    }, separators=(",", ":"))),
                }
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
                existing.append(result)
            output.flush()
            os.fsync(output.fileno())
            atomic_json(progress_path, {
                "schema": "alpha-chat-repair-v3-eval-progress-v1",
                "status": "complete" if len(existing) == len(prompts) else "partial",
                "completed_rows": len(existing),
                "total_rows": len(prompts),
                "checkpoint_sha256": checkpoint_sha256,
                "hf_model_sha256": model_sha256,
                "prompts_sha256": prompts_sha256,
            })
            print(f"evaluation {len(existing)}/{len(prompts)}", flush=True)
            batch_start = batch_end

    if len(existing) != len(prompts):
        print(f"partial: {len(existing)}/{len(prompts)}; rerun with --resume")
        return 0
    result_text = results_path.read_text(encoding="utf-8")
    repeat_rates = [float(row["fourGramRepeatRate"]) for row in existing]
    summary = {
        "schema": "alpha-chat-repair-v3-hf-eval-results-v1",
        "status": "complete",
        "completedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": device,
            "deviceName": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "tf32": False,
            "deterministicAlgorithms": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "step": args.checkpoint_step,
            "modelConfig": {
                "blockSize": block_size,
                "nLayer": int(model.config.num_hidden_layers),
                "nEmbd": int(model.config.hidden_size),
                "nHead": int(model.config.num_attention_heads),
                "vocabSize": int(model.config.vocab_size),
            },
        },
        "export": {
            "path": str(export_dir),
            "modelSha256": model_sha256,
            "tokenizerSha256": sha256_file(tokenizer_path),
            "configSha256": sha256_file(config_path),
        },
        "inputs": {"chat": {"path": str(prompts_path), "sha256": prompts_sha256, "rows": len(prompts)}},
        "outputs": {"chat": {"filename": results_path.name, "sha256": sha256_text(result_text), "rows": len(existing)}},
        "generation": {
            "chatMaxTokens": args.max_tokens,
            "eosId": eos_id,
            "userId": user_id,
            "assistantId": assistant_id,
            "deterministicGreedy": True,
            "referenceModelVisible": False,
        },
        "chat": {
            "total": len(existing),
            "structuralPass": sum(bool(row["structuralPass"]) for row in existing),
            "eosTerminated": sum(bool(row["eosTerminated"]) for row in existing),
            "roleLeaks": sum(bool(row["roleLeak"]) for row in existing),
            "nonempty": sum(bool(row["nonempty"]) for row in existing),
            "degenerateLoops": sum(bool(row["degenerateLoop"]) for row in existing),
            "hitBlockLimit": sum(bool(row["hitBlockLimit"]) for row in existing),
            "meanFourGramRepeatRate": mean(repeat_rates),
            "maxFourGramRepeatRate": max(repeat_rates, default=0.0),
            "stopReasons": dict(sorted(Counter(
                "eos" if row["eosTerminated"] else "block_limit" if row["hitBlockLimit"] else "max_tokens"
                for row in existing
            ).items())),
        },
    }
    atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "chat": summary["chat"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
