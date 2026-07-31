#!/usr/bin/env python3
"""Compare decoding policies on a deterministic, context-eligible frozen subset.

This is exploratory evidence only. It never modifies the frozen suite or its
recorded greedy result, and it is not a checkpoint-selection gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"non-alternating message at index {index}")
        marker = "<|user|>" if expected == "user" else "<|assistant|>"
        parts.append(f"{marker} {message['content']}")
    if not parts or messages[-1]["role"] != "user":
        raise ValueError("prompt must end in a user message")
    return " ".join(parts) + " <|assistant|>"


def repeat_rate(tokens: list[int], n: int = 4) -> float:
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for index in range(total):
        gram = tuple(tokens[index : index + n])
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / total


def select_rows(
    prompts: list[dict[str, object]], tokenizer: AutoTokenizer, block_size: int, count: int, seed: str
) -> list[dict[str, object]]:
    eligible: list[tuple[str, dict[str, object], str, int]] = []
    for row in prompts:
        rendered = prompt_text(row["messages"])
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) >= block_size:
            continue
        digest = hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest()
        eligible.append((digest, row, rendered, len(prompt_ids)))
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible prompts for requested sample of {count}")
    return [
        {**row, "rendered_prompt": rendered, "runtime_prompt_tokens": token_count}
        for _, row, rendered, token_count in eligible[:count]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", default="alpha-chat-decoding-v1")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model_path = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).eval()
    eos_id = int(model.config.eos_token_id)
    block_size = int(model.config.max_position_embeddings)
    prompts = parse_jsonl(Path(args.prompts).resolve())
    selected = select_rows(prompts, tokenizer, block_size, args.count, args.seed)

    policies: list[tuple[str, dict[str, object]]] = [
        ("greedy", {"do_sample": False}),
        ("greedy_no_repeat_4", {"do_sample": False, "no_repeat_ngram_size": 4}),
        ("nucleus_0p7_0p9", {"do_sample": True, "temperature": 0.7, "top_p": 0.9}),
        (
            "nucleus_0p7_0p9_repeat_1p1",
            {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1},
        ),
    ]
    rows: list[dict[str, object]] = []
    torch.manual_seed(20260731)
    with torch.inference_mode():
        for selected_row in selected:
            encoded = tokenizer(
                selected_row["rendered_prompt"], add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            remaining = block_size - int(encoded.shape[1])
            max_new_tokens = min(args.max_new_tokens, remaining)
            for policy_name, policy in policies:
                output = model.generate(
                    encoded,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    **policy,
                )
                generated = output[0, encoded.shape[1] :].tolist()
                eos_terminated = bool(generated and generated[-1] == eos_id)
                content = generated[:-1] if eos_terminated else generated
                rows.append(
                    {
                        "id": selected_row["id"],
                        "source": selected_row["source"],
                        "prompt_tokens": selected_row["runtime_prompt_tokens"],
                        "policy": policy_name,
                        "generated_ids": generated,
                        "text": tokenizer.decode(content, skip_special_tokens=False),
                        "eos_terminated": eos_terminated,
                        "four_gram_repeat_rate": repeat_rate(content),
                        "degenerate_loop": repeat_rate(content) >= 0.2,
                    }
                )

    aggregates: dict[str, dict[str, object]] = {}
    for policy_name, _ in policies:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        aggregates[policy_name] = {
            "total": len(policy_rows),
            "nonempty": sum(bool(str(row["text"]).strip()) for row in policy_rows),
            "eos_terminated": sum(bool(row["eos_terminated"]) for row in policy_rows),
            "degenerate_loops": sum(bool(row["degenerate_loop"]) for row in policy_rows),
            "mean_four_gram_repeat_rate": sum(float(row["four_gram_repeat_rate"]) for row in policy_rows)
            / max(1, len(policy_rows)),
        }
    artifact = {
        "schema": "alpha-chat-decoding-exploration-v1",
        "status": "exploratory-not-a-selection-gate",
        "model": str(model_path),
        "prompts": str(Path(args.prompts).resolve()),
        "seed": args.seed,
        "sample_count": args.count,
        "max_new_tokens": args.max_new_tokens,
        "aggregates": aggregates,
        "rows": rows,
    }
    out = Path(args.out).resolve()
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
