#!/usr/bin/env python3
"""Fail-closed publication for Alpha's free static Hugging Face Space."""

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
BACKEND = "https://donto.org/alpha-60m"
CHECKPOINT_REPO = "ajaxdavis/alpha-60m-training-checkpoints"
CHECKPOINT_REVISION = "7198d1a1f094ffe88d06399ea99fecbd78fa8b66"
CHECKPOINT_SHA256 = "6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8"
MODEL_REVISION = "b481f46924b7a4777a029de1ffb44c06cc925d4c"
EXPECTED_FILES = {"README.md", "index.html"}


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
    builder = root / "scripts/build_hf_static_space.ts"
    for path in (readme, builder):
        if not path.is_file():
            raise RuntimeError(f"Space source is missing: {path}")
    shutil.copy2(readme, destination / "README.md")
    subprocess.run(
        ["npx", "tsx", str(builder), "--out", str(destination / "index.html")],
        cwd=root,
        check=True,
    )
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    require_equal(actual, EXPECTED_FILES, "exact static Space file set")
    readme_text = (destination / "README.md").read_text(encoding="utf-8")
    html = (destination / "index.html").read_text(encoding="utf-8")
    for marker in (
        "sdk: static",
        "The checkpoint failed its predeclared chat-quality gates.",
        BACKEND,
        CHECKPOINT_REPO,
        CHECKPOINT_REVISION,
        MODEL_REVISION,
        CHECKPOINT_SHA256,
    ):
        if marker not in readme_text:
            raise RuntimeError(f"Space README is missing required evidence: {marker}")
    for marker in (
        "A finished experiment, shown without spin.",
        "Archived · D3 quality gate failed",
        f'{BACKEND}/v1/chat/completions',
        "92 / 100",
        "0 / 200",
    ):
        if marker not in html:
            raise RuntimeError(f"Space HTML is missing required evidence or endpoint: {marker}")
    manifest = {
        name: {"bytes": (destination / name).stat().st_size, "sha256": sha256_file(destination / name)}
        for name in sorted(EXPECTED_FILES)
    }
    return {
        "repo": REPO,
        "sdk": "static",
        "backend": BACKEND,
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_revision": MODEL_REVISION,
        "files": len(EXPECTED_FILES),
        "bytes": sum(entry["bytes"] for entry in manifest.values()),
        "manifest": manifest,
    }


def publish(staging: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    require_equal(api.whoami().get("name"), "ajaxdavis", "authenticated Hub identity")
    api.create_repo(repo_id=REPO, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="space",
        folder_path=staging,
        commit_message="Publish terminal Alpha 60M static research Space",
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
    require_equal(siblings, EXPECTED_FILES | {".gitattributes"}, "anonymous Space file set")
    return {
        **preflight,
        "mode": "publish",
        "hub": {"revision": revision, "public": True, "anonymous": True, "siblings": sorted(siblings)},
    }


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
    with tempfile.TemporaryDirectory(prefix="alpha-hf-static-space-", dir="/mnt/donto-data/alpha-runs") as temporary:
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
