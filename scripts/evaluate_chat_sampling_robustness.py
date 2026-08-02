#!/usr/bin/env python3
"""Measure chat robustness under Alpha's public sampled-decoding profile.

The scored BLAH evaluations use greedy decoding, but the public chatroom,
health check, and normal chat path use temperature 0.7.  A checkpoint can look
acceptable under one greedy continuation while assigning substantial
probability to incoherent or repetitive continuations.  This evaluator repeats
the same frozen prompt population under deterministic sampling seeds and
records every trajectory.  It is a separate selection lane; it never replaces
the greedy semantic evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def render_prompt(messages: object, label: str) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label}: messages are absent")
    parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{label}: message {index + 1} is not an object")
        expected = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        content = message.get("content")
        if role != expected:
            raise ValueError(
                f"{label}: message {index + 1} role {role!r}, expected {expected!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{label}: message {index + 1} is empty")
        marker = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{marker} {content}")
    if messages[-1].get("role") != "user":
        raise ValueError(f"{label}: dialogue does not end with a user turn")
    return " ".join(parts) + " <|assistant|>"


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-prompts-sha256")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()

    if len(args.checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.checkpoint_sha256
    ):
        parser.error("checkpoint-sha256 must be a lowercase SHA-256")
    if args.repetitions <= 0 or args.batch_size <= 0 or args.max_tokens <= 0:
        parser.error("repetitions, batch-size, and max-tokens must be positive")
    if args.temperature <= 0 or args.top_k <= 0 or not 0 < args.top_p <= 1:
        parser.error("sampled robustness requires temperature/top-k > 0 and top-p in (0,1]")

    export_dir = Path(args.export_dir).resolve()
    prompts_path = Path(args.prompts).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_path = export_dir / "model.safetensors"
    tokenizer_path = export_dir / "tokenizer.json"
    config_path = export_dir / "config.json"
    for path in (model_path, tokenizer_path, config_path, prompts_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing or empty: {path}")
    prompts_sha256 = sha256_file(prompts_path)
    if (
        args.expected_prompts_sha256 is not None
        and prompts_sha256 != args.expected_prompts_sha256
    ):
        raise ValueError(
            f"prompt SHA-256 drift: {prompts_sha256} != {args.expected_prompts_sha256}"
        )

    prompts = read_jsonl(prompts_path)
    if not prompts:
        raise ValueError("prompt suite is empty")
    rendered: list[str] = []
    identities: set[str] = set()
    for index, row in enumerate(prompts, start=1):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError(f"prompt {index}: missing or duplicate ID")
        identities.add(identity)
        rendered.append(render_prompt(row.get("messages"), f"prompt {index}"))

    # Required by deterministic CUDA matmul. This must be set before torch first
    # initializes cuBLAS in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("torch and transformers are required") from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu_smoke:
        raise RuntimeError("CUDA is required; --allow-cpu-smoke is diagnostic only")
    if device == "cuda":
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
            raise ValueError(f"{marker} is not atomic")
        return int(ids[0])

    eos_id = special_id("<|end_of_text|>")
    user_id = special_id("<|user|>")
    assistant_id = special_id("<|assistant|>")
    tokenizer.pad_token_id = eos_id
    block_size = int(model.config.max_position_embeddings)
    prompt_ids: list[list[int]] = []
    for index, text in enumerate(rendered, start=1):
        ids = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
        if not ids or ids[-1] != assistant_id or len(ids) >= block_size:
            raise ValueError(f"prompt {index}: invalid or overlength chat rendering")
        prompt_ids.append(ids)

    out_dir.mkdir(parents=True, exist_ok=False)
    results_path = out_dir / "sampling-results.jsonl"
    rows: list[dict[str, Any]] = []
    with results_path.open("x", encoding="utf-8") as output:
        with torch.inference_mode():
            for repetition in range(args.repetitions):
                repetition_seed = args.seed + repetition
                torch.manual_seed(repetition_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(repetition_seed)
                for batch_start in range(0, len(prompt_ids), args.batch_size):
                    batch_end = min(len(prompt_ids), batch_start + args.batch_size)
                    batch_ids = prompt_ids[batch_start:batch_end]
                    width = max(map(len, batch_ids))
                    batch_max_tokens = min(args.max_tokens, block_size - width)
                    if batch_max_tokens <= 0:
                        raise ValueError(
                            f"batch at prompt {batch_start + 1} leaves no generation position"
                        )
                    padded = [[eos_id] * (width - len(ids)) + ids for ids in batch_ids]
                    masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch_ids]
                    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
                    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        max_new_tokens=batch_max_tokens,
                        eos_token_id=eos_id,
                        pad_token_id=eos_id,
                        use_cache=True,
                    )
                    completions = generated[:, width:].detach().cpu().tolist()
                    for local_index, completion in enumerate(completions):
                        prompt_index = batch_start + local_index
                        generated_ids: list[int] = []
                        eos_terminated = False
                        for token in completion:
                            generated_ids.append(int(token))
                            if token == eos_id:
                                eos_terminated = True
                                break
                        content_ids = generated_ids[:-1] if eos_terminated else generated_ids
                        text = tokenizer.decode(
                            content_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        repeat_rate = four_gram_repeat_rate(content_ids)
                        role_leak = user_id in content_ids or assistant_id in content_ids
                        nonempty = bool(text.strip())
                        hit_block_limit = (
                            not eos_terminated
                            and len(prompt_ids[prompt_index]) + len(generated_ids) >= block_size
                        )
                        row = {
                            "schema": "alpha-chat-sampling-robustness-row-v1",
                            "id": prompts[prompt_index]["id"],
                            "source": prompts[prompt_index].get("source"),
                            "repetition": repetition,
                            "seed": repetition_seed,
                            "checkpointSha256": args.checkpoint_sha256,
                            "promptTokens": len(prompt_ids[prompt_index]),
                            "generatedIds": generated_ids,
                            "text": text,
                            "eosTerminated": eos_terminated,
                            "hitBlockLimit": hit_block_limit,
                            "roleLeak": role_leak,
                            "nonempty": nonempty,
                            "fourGramRepeatRate": repeat_rate,
                            "degenerateLoop": repeat_rate >= 0.2,
                            "structuralPass": eos_terminated and nonempty and not role_leak,
                        }
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        rows.append(row)
                    output.flush()
                    os.fsync(output.fileno())
                print(
                    f"sampling repetition {repetition + 1}/{args.repetitions}: "
                    f"{len(rows)}/{len(prompts) * args.repetitions}",
                    flush=True,
                )

    repeat_rates = [float(row["fourGramRepeatRate"]) for row in rows]
    per_prompt = []
    for identity in sorted(identities):
        selected = [row for row in rows if row["id"] == identity]
        per_prompt.append(
            {
                "id": identity,
                "runs": len(selected),
                "structuralPass": sum(bool(row["structuralPass"]) for row in selected),
                "nonempty": sum(bool(row["nonempty"]) for row in selected),
                "eosTerminated": sum(bool(row["eosTerminated"]) for row in selected),
                "degenerateLoops": sum(bool(row["degenerateLoop"]) for row in selected),
                "uniqueOutputs": len({str(row["text"]) for row in selected}),
            }
        )
    summary = {
        "schema": "alpha-chat-sampling-robustness-summary-v1",
        "status": "complete",
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"sha256": args.checkpoint_sha256},
        "inputs": {
            "prompts": {
                "path": str(prompts_path),
                "sha256": prompts_sha256,
                "rows": len(prompts),
            },
            "export": {
                "path": str(export_dir),
                "modelSha256": sha256_file(model_path),
                "tokenizerSha256": sha256_file(tokenizer_path),
                "configSha256": sha256_file(config_path),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": device,
            "deviceName": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "deterministicAlgorithms": True,
        },
        "generation": {
            "profile": "blah-public-chat-temperature-0.7-top-k-40",
            "temperature": args.temperature,
            "topK": args.top_k,
            "topP": args.top_p,
            "maxTokens": args.max_tokens,
            "batchSize": args.batch_size,
            "repetitions": args.repetitions,
            "baseSeed": args.seed,
        },
        "aggregate": {
            "runs": len(rows),
            "structuralPass": sum(bool(row["structuralPass"]) for row in rows),
            "nonempty": sum(bool(row["nonempty"]) for row in rows),
            "eosTerminated": sum(bool(row["eosTerminated"]) for row in rows),
            "roleLeaks": sum(bool(row["roleLeak"]) for row in rows),
            "degenerateLoops": sum(bool(row["degenerateLoop"]) for row in rows),
            "hitBlockLimit": sum(bool(row["hitBlockLimit"]) for row in rows),
            "meanFourGramRepeatRate": mean(repeat_rates),
            "maxFourGramRepeatRate": max(repeat_rates, default=0.0),
            "stopReasons": dict(
                sorted(
                    Counter(
                        "eos"
                        if row["eosTerminated"]
                        else "block_limit"
                        if row["hitBlockLimit"]
                        else "max_tokens"
                        for row in rows
                    ).items()
                )
            ),
        },
        "perPrompt": per_prompt,
        "outputs": {
            "results": {
                "path": str(results_path),
                "sha256": sha256_file(results_path),
                "rows": len(rows),
            }
        },
    }
    atomic_json(out_dir / "summary.json", summary)
    print(json.dumps({"out": str(out_dir), "aggregate": summary["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
