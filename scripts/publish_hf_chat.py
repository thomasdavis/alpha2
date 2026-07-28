#!/usr/bin/env python3
"""Fail-closed preflight and explicit publication for Alpha 60M Chat."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO = "ajaxdavis/alpha-60m-chat"
SOURCE_COMMIT = "c333bf247fbe87b85d01f3d34789b46615dd1034"
REQUIRED_EXPORT_FILES = {
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}
PUBLISHED_FILES = REQUIRED_EXPORT_FILES | {"README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--terminal-status", type=Path, required=True)
    parser.add_argument("--sft-analysis", type=Path, required=True)
    parser.add_argument("--pair-analysis", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--parity-log", type=Path, required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="Create/update the public Hub repository")
    parser.add_argument("--commit-message", default="Publish validated Alpha 60M Chat")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256: {value!r}")
    return value


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    require_equal(args.repo, REPO, "Hub repository")
    if not args.out.resolve().is_relative_to(Path("/mnt/donto-data/alpha-runs")):
        raise RuntimeError("--out must live under /mnt/donto-data/alpha-runs/")
    for path, label in (
        (args.model_card, "model card"),
        (args.terminal_status, "terminal status"),
        (args.sft_analysis, "SFT analysis"),
        (args.pair_analysis, "pair analysis"),
        (args.semantic_review, "semantic review"),
        (args.parity_log, "parity log"),
    ):
        require_file(path, label)
    if not args.export_dir.is_dir():
        raise RuntimeError(f"export directory is missing: {args.export_dir}")
    actual_export_files = {
        entry.relative_to(args.export_dir).as_posix()
        for entry in args.export_dir.rglob("*")
        if entry.is_file()
    }
    require_equal(actual_export_files, REQUIRED_EXPORT_FILES, "exact export file set")

    export_manifest = {
        name: {"bytes": (args.export_dir / name).stat().st_size, "sha256": sha256_file(args.export_dir / name)}
        for name in sorted(REQUIRED_EXPORT_FILES)
    }
    weights_sha256 = export_manifest["model.safetensors"]["sha256"]
    config = read_json(args.export_dir / "config.json", "export config")
    require_equal(config.get("architectures"), ["LlamaForCausalLM"], "config architectures")
    require_equal(config.get("model_type"), "llama", "config model_type")
    require_equal(config.get("vocab_size"), 12_288, "config vocab_size")
    require_equal(config.get("hidden_size"), 512, "config hidden_size")
    require_equal(config.get("num_hidden_layers"), 16, "config num_hidden_layers")
    require_equal(config.get("num_attention_heads"), 8, "config num_attention_heads")
    require_equal(config.get("tie_word_embeddings"), True, "config tie_word_embeddings")

    terminal = read_json(args.terminal_status, "terminal status")
    sft = read_json(args.sft_analysis, "SFT analysis")
    pair = read_json(args.pair_analysis, "pair analysis")
    semantic = read_json(args.semantic_review, "semantic review")
    require_equal(terminal.get("schema"), "alpha-sft-terminal-finalizer-v1", "terminal schema")
    require_equal(terminal.get("result"), "PASS", "terminal operational result")
    require_equal(terminal.get("source_commit"), SOURCE_COMMIT, "terminal source commit")
    require_equal(terminal.get("machine_d3", {}).get("result"), "PASS", "terminal machine D3")
    terminal_checkpoint_sha = require_sha256(terminal.get("checkpoint", {}).get("sha256"), "terminal checkpoint SHA-256")
    require_equal(sft.get("schema"), "alpha-flagship-sft-analysis-v1", "SFT analysis schema")
    require_equal(sft.get("result"), "PASS", "SFT analysis result")
    require_equal(sft.get("source_commit"), SOURCE_COMMIT, "SFT source commit")
    require_equal(sft.get("rows"), 30_322, "SFT metric rows")
    require_equal(sft.get("checkpoint", {}).get("parameter_elements"), 57_688_576, "SFT parameter elements")
    require_equal(sft.get("checkpoint", {}).get("finite_parameter_elements"), 57_688_576, "SFT finite elements")
    require_equal(sft.get("checkpoint", {}).get("sha256"), terminal_checkpoint_sha, "SFT/terminal checkpoint SHA-256")
    require_equal(pair.get("schema"), "alpha-frozen-eval-pair-analysis-v1", "pair-analysis schema")
    require_equal(pair.get("result"), "PASS", "machine D3 pair result")
    require_equal(pair.get("inputs_match"), True, "base/chat frozen inputs")
    require_equal(pair.get("chat", {}).get("checkpoint", {}).get("sha256"), terminal_checkpoint_sha, "pair/terminal checkpoint SHA-256")
    require_equal(semantic.get("schema"), "alpha-frozen-chat-semantic-review-v1", "semantic-review schema")
    require_equal(semantic.get("result"), "PASS", "semantic-review result")
    require_equal(semantic.get("reference_blinded"), True, "semantic reference blinding")
    require_equal(semantic.get("counts", {}).get("total"), 100, "semantic reviewed rows")
    if semantic.get("counts", {}).get("PASS", 0) < 80 or semantic.get("counts", {}).get("FAIL") != 0:
        raise RuntimeError("semantic-review counts do not satisfy the predeclared gate")
    require_equal(
        semantic.get("provenance", {}).get("checkpoint", {}).get("sha256"),
        terminal_checkpoint_sha,
        "semantic/terminal checkpoint SHA-256",
    )

    parity_text = args.parity_log.read_text(encoding="utf-8")
    if "RESULT               : PASS" not in parity_text:
        raise RuntimeError("Alpha/Transformers export parity did not pass")
    card_text = args.model_card.read_text(encoding="utf-8")
    for marker in ("PENDING", "DRAFT ONLY", "Replace this table"):
        if marker.casefold() in card_text.casefold():
            raise RuntimeError(f"model card still contains release blocker: {marker}")
    for required in (
        "license: apache-2.0",
        "library_name: transformers",
        "# Alpha 60M Chat",
        terminal_checkpoint_sha,
        weights_sha256,
    ):
        if required not in card_text:
            raise RuntimeError(f"model card is missing exact release evidence: {required}")

    return {
        "repo": args.repo,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": terminal_checkpoint_sha,
        "weights_sha256": weights_sha256,
        "parameter_elements": 57_688_576,
        "export_manifest": export_manifest,
        "inputs": {
            "model_card": {"path": str(args.model_card), "sha256": sha256_file(args.model_card)},
            "terminal_status": {"path": str(args.terminal_status), "sha256": sha256_file(args.terminal_status)},
            "sft_analysis": {"path": str(args.sft_analysis), "sha256": sha256_file(args.sft_analysis)},
            "pair_analysis": {"path": str(args.pair_analysis), "sha256": sha256_file(args.pair_analysis)},
            "semantic_review": {"path": str(args.semantic_review), "sha256": sha256_file(args.semantic_review)},
            "parity_log": {"path": str(args.parity_log), "sha256": sha256_file(args.parity_log)},
        },
    }


def publish(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    if identity.get("name") != "ajaxdavis":
        raise RuntimeError(f"authenticated Hub identity is not ajaxdavis: {identity.get('name')!r}")
    api.create_repo(repo_id=args.repo, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alpha-hf-chat-") as temporary:
        staging = Path(temporary)
        for name in REQUIRED_EXPORT_FILES:
            shutil.copy2(args.export_dir / name, staging / name)
        shutil.copy2(args.model_card, staging / "README.md")
        commit = api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=staging,
            commit_message=args.commit_message,
        )
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"Hub upload returned no immutable commit: {revision!r}")
    api.update_repo_settings(repo_id=args.repo, repo_type="model", private=False, gated=False)
    info = HfApi(token=False).model_info(args.repo, revision=revision, token=False)
    require_equal(info.sha, revision, "anonymous Hub revision")
    require_equal(info.private, False, "Hub visibility")
    siblings = {sibling.rfilename for sibling in info.siblings}
    missing = PUBLISHED_FILES - siblings
    if missing:
        raise RuntimeError(f"Hub revision is missing files: {sorted(missing)}")
    if any(name.endswith(".py") for name in siblings):
        raise RuntimeError("Hub revision unexpectedly contains custom Python code")
    unexpected = siblings - PUBLISHED_FILES - {".gitattributes"}
    if unexpected:
        raise RuntimeError(f"Hub revision contains unexpected files: {sorted(unexpected)}")
    return {
        **preflight,
        "mode": "publish",
        "hub": {"revision": revision, "public": True, "anonymous": True, "siblings": sorted(siblings)},
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise RuntimeError(f"output already exists: {args.out}")
    preflight = validate_inputs(args)
    report = {
        "schema": "alpha-hf-chat-publication-v1",
        "result": "PASS",
        "mode": "preflight",
        **preflight,
    }
    if args.publish:
        report = {"schema": "alpha-hf-chat-publication-v1", "result": "PASS", **publish(args, preflight)}
    write_exclusive(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
