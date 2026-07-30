#!/usr/bin/env python3
"""Fail-closed preflight and explicit publication for Alpha native recovery checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


REPO = "ajaxdavis/alpha-60m-training-checkpoints"
SOURCE_COMMIT = "c333bf247fbe87b85d01f3d34789b46615dd1034"
EXPECTED_CHECKPOINTS = {
    "checkpoints/base-pretrain-step-61036.alph": (
        61_036,
        "08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049",
    ),
    "checkpoints/sft-best-retained-step-29000.alph": (
        29_000,
        "03eaac3e7be06e8fb5720415a334b36d7ef5019fcff72ca9227636b84011a7f3",
    ),
    "checkpoints/sft-terminal-step-30322.alph": (
        30_322,
        "6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8",
    ),
}
EXPECTED_FILES = {
    "README.md",
    "CHECKSUMS.sha256",
    *EXPECTED_CHECKPOINTS,
    "inputs/g2-bpe-byte-12k.json",
    "inputs/length-audit.json",
    "inputs/mask-audit.json",
    "inputs/sft-v2.txt.manifest.json",
    "reports/base/checkpoint-61036-native-audit.json",
    "reports/base/flagship-contract.json",
    "reports/base/flagship-pretrain-analysis.json",
    "reports/sft/flagship-sft-analysis.json",
    "reports/sft/frozen-chat-semantic-review-report.json",
    "reports/sft/frozen-eval-pair-analysis.json",
    "reports/sft/metrics.jsonl",
    "reports/sft/sft-contract.json",
    "reports/sft/sft-lr-sweep-analysis.json",
    "reports/sft/terminal-finalizer-status.json",
    "reports/sft/terminal-sft-verification.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        if match is None:
            raise RuntimeError(f"invalid checksum line {line_number}: {line!r}")
        digest, relative = match.groups()
        if relative in entries:
            raise RuntimeError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest
    return entries


def read_native_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if handle.read(4) != b"ALPH":
            raise RuntimeError(f"checkpoint has no ALPH magic: {path}")
        header_length = int.from_bytes(handle.read(4), "little")
        if header_length <= 0 or header_length > 16 * 1024 * 1024:
            raise RuntimeError(f"invalid ALPH header length {header_length}: {path}")
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise RuntimeError(f"ALPH header is not an object: {path}")
    return header


def validate(args: argparse.Namespace) -> dict[str, Any]:
    require_equal(args.repo, REPO, "Hub repository")
    archive = args.archive_dir.resolve()
    if not archive.is_relative_to(Path("/mnt/donto-data/alpha-runs")):
        raise RuntimeError("--archive-dir must live under /mnt/donto-data/alpha-runs/")
    if not args.out.resolve().is_relative_to(Path("/mnt/donto-data/alpha-runs")):
        raise RuntimeError("--out must live under /mnt/donto-data/alpha-runs/")
    if not archive.is_dir():
        raise RuntimeError(f"archive directory is missing: {archive}")

    actual_files = {
        path.relative_to(archive).as_posix()
        for path in archive.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(archive).parts
    }
    require_equal(actual_files, EXPECTED_FILES, "exact archive file set")
    checksums = parse_checksums(archive / "CHECKSUMS.sha256")
    require_equal(set(checksums), EXPECTED_FILES - {"CHECKSUMS.sha256"}, "checksum file set")
    for relative, expected in sorted(checksums.items()):
        require_equal(sha256_file(archive / relative), expected, f"SHA-256 {relative}")

    headers: dict[str, Any] = {}
    for relative, (step, expected_sha) in EXPECTED_CHECKPOINTS.items():
        require_equal(checksums[relative], expected_sha, f"pinned checkpoint SHA-256 {relative}")
        header = read_native_header(archive / relative)
        require_equal(header.get("step"), step, f"checkpoint step {relative}")
        require_equal(header.get("optimizerStep"), step, f"optimizer step {relative}")
        tensors = header.get("tensors")
        if not isinstance(tensors, list):
            raise RuntimeError(f"checkpoint tensors are missing: {relative}")
        require_equal(len(tensors), 342, f"tensor count {relative}")
        prefixes = {str(tensor.get("name", "")).split(".", 1)[0] for tensor in tensors}
        require_equal(prefixes, {"p", "o"}, f"parameter/optimizer tensor prefixes {relative}")
        require_equal(sum(str(tensor.get("name", "")).startswith("p.") for tensor in tensors), 114, f"parameter tensor count {relative}")
        if not isinstance(header.get("rngState"), (int, dict)):
            raise RuntimeError(f"checkpoint RNG state is missing: {relative}")
        if not isinstance(header.get("tokenizerArtifacts"), dict):
            raise RuntimeError(f"checkpoint tokenizer artifacts are missing: {relative}")
        headers[relative] = {"step": step, "optimizer_step": step, "tensors": len(tensors), "sha256": expected_sha}

    terminal = read_json(archive / "reports/sft/terminal-finalizer-status.json")
    pair = read_json(archive / "reports/sft/frozen-eval-pair-analysis.json")
    semantic = read_json(archive / "reports/sft/frozen-chat-semantic-review-report.json")
    analysis = read_json(archive / "reports/sft/flagship-sft-analysis.json")
    require_equal(terminal.get("result"), "PASS", "terminal operational result")
    require_equal(terminal.get("source_commit"), SOURCE_COMMIT, "terminal source commit")
    require_equal(terminal.get("machine_d3", {}).get("result"), "FAIL", "terminal quality result")
    require_equal(pair.get("result"), "FAIL", "machine D3 result")
    require_equal(semantic.get("result"), "FAIL", "semantic review result")
    require_equal(semantic.get("counts"), {"total": 100, "PASS": 0, "BORDERLINE": 0, "FAIL": 100}, "semantic counts")
    require_equal(analysis.get("rows"), 30_322, "terminal metric rows")
    require_equal(
        analysis.get("checkpoint", {}).get("sha256"),
        EXPECTED_CHECKPOINTS["checkpoints/sft-terminal-step-30322.alph"][1],
        "terminal analysis checkpoint SHA-256",
    )
    readme = (archive / "README.md").read_text(encoding="utf-8")
    for marker in (
        "failed the predeclared chat-quality gates",
        "No continuation run is currently authorized or running.",
        SOURCE_COMMIT,
        *[expected_sha for _, expected_sha in EXPECTED_CHECKPOINTS.values()],
    ):
        if marker not in readme:
            raise RuntimeError(f"README is missing required evidence: {marker}")

    return {
        "repo": args.repo,
        "source_commit": SOURCE_COMMIT,
        "files": len(EXPECTED_FILES),
        "bytes": sum((archive / relative).stat().st_size for relative in EXPECTED_FILES),
        "checkpoints": headers,
        "quality_gate_result": "FAIL",
        "checksums_sha256": sha256_file(archive / "CHECKSUMS.sha256"),
    }


def publish(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    require_equal(identity.get("name"), "ajaxdavis", "authenticated Hub identity")
    api.create_repo(repo_id=args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.archive_dir,
        private=False,
        ignore_patterns=[".cache/**"],
        num_workers=4,
        print_report=True,
        print_report_every=30,
    )
    api.update_repo_settings(repo_id=args.repo, repo_type="model", private=False, gated=False)

    expected_hub = EXPECTED_FILES | {".gitattributes"}
    deadline = time.monotonic() + 300
    info = None
    while time.monotonic() < deadline:
        info = HfApi(token=False).model_info(args.repo, token=False)
        siblings = {sibling.rfilename for sibling in info.siblings}
        if EXPECTED_FILES.issubset(siblings):
            break
        time.sleep(5)
    if info is None:
        raise RuntimeError("anonymous Hub verification returned no repository info")
    siblings = {sibling.rfilename for sibling in info.siblings}
    require_equal(info.private, False, "Hub visibility")
    require_equal(siblings, expected_hub, "anonymous Hub file set")
    revision = info.sha
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"Hub returned no immutable revision: {revision!r}")
    return {**preflight, "mode": "publish", "hub": {"revision": revision, "public": True, "anonymous": True, "siblings": sorted(siblings)}}


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise RuntimeError(f"output already exists: {args.out}")
    preflight = validate(args)
    report: dict[str, Any] = {
        "schema": "alpha-hf-checkpoint-publication-v1",
        "result": "PASS",
        "mode": "preflight",
        **preflight,
    }
    if args.publish:
        report = {"schema": "alpha-hf-checkpoint-publication-v1", "result": "PASS", **publish(args, preflight)}
    write_exclusive(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
