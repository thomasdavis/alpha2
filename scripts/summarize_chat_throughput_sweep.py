#!/usr/bin/env python3
"""Summarize an immutable Alpha end-to-end throughput sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


STEP_RE = re.compile(
    r"step (?P<step>\d+)/(?P<total>\d+) \| loss=(?P<loss>[^ ]+).*?"
    r"\| (?P<ms>\d+)ms/it \| (?P<tps>\d+) tok/s \| (?P<ops>\d+) gpu_ops"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--exclude-final", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for log_path in sorted(args.root.glob("*.log")):
        text = log_path.read_text(errors="replace")
        matches = [m.groupdict() for m in STEP_RE.finditer(text)]
        accepted = []
        for match in matches:
            step = int(match["step"])
            total = int(match["total"])
            if step <= args.skip_steps:
                continue
            if args.exclude_final and step == total:
                continue
            accepted.append(match)

        tps = [int(m["tps"]) for m in accepted]
        ms = [int(m["ms"]) for m in accepted]
        ops = [int(m["ops"]) for m in accepted]
        losses = [float(m["loss"]) for m in accepted if math.isfinite(float(m["loss"]))]
        exit_path = args.root / log_path.stem / "exit-code.txt"
        exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
        nonfinite = bool(re.search(r"(?:loss|grad(?:ient)?)[^\n]*(?:NaN|Inf)|non-finite", text, re.I))
        row = {
            "name": log_path.stem,
            "exit_code": exit_code,
            "samples": len(tps),
            "median_tps": statistics.median(tps) if tps else None,
            "mean_tps": statistics.mean(tps) if tps else None,
            "median_ms": statistics.median(ms) if ms else None,
            "median_gpu_ops": statistics.median(ops) if ops else None,
            "last_loss": losses[-1] if losses else None,
            "nonfinite": nonfinite,
        }
        rows.append(row)

    print("# Alpha chat throughput sweep summary\n")
    print(f"Root: `{args.root}`\n")
    print("| Row | Exit | Samples | Median tok/s | Mean tok/s | Median ms | GPU ops | Last loss | Nonfinite |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        def number(key: str, digits: int = 1) -> str:
            value = row[key]
            return "n/a" if value is None else f"{float(value):.{digits}f}"

        print(
            f"| {row['name']} | {row['exit_code'] if row['exit_code'] is not None else 'n/a'} "
            f"| {row['samples']} | {number('median_tps', 0)} | {number('mean_tps', 1)} "
            f"| {number('median_ms', 0)} | {number('median_gpu_ops', 0)} "
            f"| {number('last_loss', 4)} | {'yes' if row['nonfinite'] else 'no'} |"
        )

    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
