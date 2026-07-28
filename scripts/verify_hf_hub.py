#!/usr/bin/env python3
"""Cold-load a public Alpha export from Hugging Face with stock Transformers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="Immutable 40-character Hub commit")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-params", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--plain-prompt", default="Hello")
    parser.add_argument("--chat-prompt", default="Hello")
    parser.add_argument("--plain-max-new-tokens", type=int, default=8)
    parser.add_argument("--chat-max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(ch not in "0123456789abcdef" for ch in args.revision):
        parser.error("--revision must be a lowercase 40-character Hub commit")
    if len(args.expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in args.expected_sha256):
        parser.error("--expected-sha256 must be a lowercase SHA-256")
    if args.expected_params <= 0:
        parser.error("--expected-params must be positive")
    for name in ("plain_max_new_tokens", "chat_max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise RuntimeError(f"cold-load cache is not empty: {args.cache_dir}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Hide CUDA before importing torch. This verifier must never contend with a live training GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    info = HfApi(token=False).model_info(args.repo, revision=args.revision, token=False)
    if info.private:
        raise RuntimeError(f"repository is private: {args.repo}")
    if info.sha != args.revision:
        raise RuntimeError(f"Hub resolved revision {info.sha} != {args.revision}")

    weights_path = Path(
        hf_hub_download(
            repo_id=args.repo,
            filename="model.safetensors",
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=False,
        )
    )
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != args.expected_sha256:
        raise RuntimeError(f"weight SHA-256 {weights_sha256} != {args.expected_sha256}")

    common: dict[str, Any] = {
        "pretrained_model_name_or_path": args.repo,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "token": False,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(**common)
    model = AutoModelForCausalLM.from_pretrained(**common, dtype=torch.float32)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != args.expected_params:
        raise RuntimeError(f"parameter count {parameters} != {args.expected_params}")
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise RuntimeError(f"unexpected model class: {model.__class__.__name__}")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    plain_output = generator(
        args.plain_prompt,
        max_new_tokens=args.plain_max_new_tokens,
        do_sample=False,
    )[0]["generated_text"]
    chat_output = generator(
        [{"role": "user", "content": args.chat_prompt}],
        max_new_tokens=args.chat_max_new_tokens,
        do_sample=False,
    )[0]["generated_text"]

    print(
        json.dumps(
            {
                "schema": "alpha-hf-hub-cold-load-v1",
                "result": "PASS",
                "repo": args.repo,
                "revision": args.revision,
                "public": True,
                "anonymous": True,
                "empty_cache": True,
                "trust_remote_code": False,
                "device": "cpu",
                "class_name": model.__class__.__name__,
                "parameters": parameters,
                "weights_sha256": weights_sha256,
                "plain_prompt": args.plain_prompt,
                "plain_output": plain_output,
                "chat_prompt": args.chat_prompt,
                "chat_output": chat_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
