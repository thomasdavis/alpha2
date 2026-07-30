#!/usr/bin/env python3
"""Build a pruned, provenance-bound Docker Space source tree and publish it explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO = "ajaxdavis/alpha-60m-chat"
CHECKPOINT_REPO = "ajaxdavis/alpha-60m-training-checkpoints"
CHECKPOINT_REVISION = "7198d1a1f094ffe88d06399ea99fecbd78fa8b66"
CHECKPOINT_SHA256 = "6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8"
MODEL_REVISION = "b481f46924b7a4777a029de1ffb44c06cc925d4c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_staging(destination: Path) -> dict[str, Any]:
    root = repo_root()
    readme = root / "apps/hf/README.md"
    dockerfile = root / "apps/hf/Dockerfile.space"
    for path in (readme, dockerfile):
        if not path.is_file():
            raise RuntimeError(f"Space source is missing: {path}")
    readme_text = readme.read_text(encoding="utf-8")
    docker_text = dockerfile.read_text(encoding="utf-8")
    for marker in (
        "The checkpoint failed its predeclared chat-quality gates.",
        CHECKPOINT_REPO,
        CHECKPOINT_SHA256,
    ):
        if marker not in readme_text:
            raise RuntimeError(f"Space README is missing required evidence: {marker}")
    for marker in (CHECKPOINT_REPO, CHECKPOINT_REVISION, CHECKPOINT_SHA256):
        if marker not in docker_text:
            raise RuntimeError(f"Space Dockerfile is missing immutable checkpoint evidence: {marker}")

    pruned = destination / "pruned"
    subprocess.run(
        ["npx", "turbo", "prune", "@alpha/hf", "--docker", f"--out-dir={pruned}"],
        cwd=root,
        check=True,
    )
    shutil.copytree(pruned / "json", destination / "json")
    shutil.copytree(pruned / "full", destination / "full")
    shutil.copy2(readme, destination / "README.md")
    shutil.copy2(dockerfile, destination / "Dockerfile")
    (destination / ".dockerignore").write_text("pruned\n.cache\n**/dist\n**/.turbo\n**/*.tsbuildinfo\n", encoding="utf-8")
    shutil.rmtree(pruned)

    files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    )
    if not files or any("node_modules" in Path(relative).parts for relative in files):
        raise RuntimeError("invalid pruned Space source tree")
    required = {
        "README.md",
        "Dockerfile",
        ".dockerignore",
        "json/package.json",
        "json/package-lock.json",
        "full/apps/hf/src/index.ts",
        "full/apps/hf/src/protocol.ts",
        "full/apps/hf/src/ui.ts",
        "full/packages/core/src/index.ts",
        "full/packages/inference/src/index.ts",
        "full/packages/tokenizers/src/index.ts",
    }
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"pruned Space source is missing files: {sorted(missing)}")
    manifest = {
        relative: {"bytes": (destination / relative).stat().st_size, "sha256": sha256_file(destination / relative)}
        for relative in files
    }
    return {
        "repo": REPO,
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
        "files": len(files),
        "bytes": sum(entry["bytes"] for entry in manifest.values()),
        "manifest": manifest,
    }


def publish(staging: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    require_equal(api.whoami().get("name"), "ajaxdavis", "authenticated Hub identity")
    api.create_repo(repo_id=REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="space",
        folder_path=staging,
        commit_message="Publish terminal Alpha 60M research Space",
        delete_patterns="*",
    )
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"Space upload returned no immutable commit: {revision!r}")
    api.update_repo_settings(repo_id=REPO, repo_type="space", private=False)
    info = HfApi(token=False).space_info(REPO, revision=revision, token=False)
    require_equal(info.sha, revision, "anonymous Space revision")
    require_equal(info.private, False, "Space visibility")
    siblings = {sibling.rfilename for sibling in info.siblings}
    expected = set(preflight["manifest"]) | {".gitattributes"}
    require_equal(siblings, expected, "anonymous Space file set")
    return {**preflight, "mode": "publish", "hub": {"revision": revision, "public": True, "anonymous": True, "siblings": sorted(siblings)}}


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    require_equal(args.repo, REPO, "Hub Space repository")
    if not args.out.resolve().is_relative_to(Path("/mnt/donto-data/alpha-runs")):
        raise RuntimeError("--out must live under /mnt/donto-data/alpha-runs/")
    if args.out.exists():
        raise RuntimeError(f"output already exists: {args.out}")
    with tempfile.TemporaryDirectory(prefix="alpha-hf-space-", dir="/mnt/donto-data/alpha-runs") as temporary:
        staging = Path(temporary)
        preflight = build_staging(staging)
        report: dict[str, Any] = {
            "schema": "alpha-hf-space-publication-v1",
            "result": "PASS",
            "mode": "preflight",
            **preflight,
        }
        if args.publish:
            report = {"schema": "alpha-hf-space-publication-v1", "result": "PASS", **publish(staging, preflight)}
    write_exclusive(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
