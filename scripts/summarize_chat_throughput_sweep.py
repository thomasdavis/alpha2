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
TRACE_RE = re.compile(
    r"\[trace\] data=(?P<data>\d+)ms fwd=(?P<fwd>\d+)ms bwd=(?P<bwd>\d+)ms "
    r"gradnorm=(?P<gradnorm>\d+)ms clip=(?P<clip>\d+)ms "
    r"optim=(?P<optim>\d+)ms flush=(?P<flush>\d+)ms"
)
GPU_OP_RE = re.compile(
    r"\[gpu_ops\] flushes=(?P<flushes>\d+) waited=(?P<waited>\d+) "
    r"dgc=(?P<dgc>\d+) ops_per_flush=(?P<ops_per_flush>[\d.]+) "
    r"kinds=(?P<kinds>[^ ]*) kernels=(?P<kernels>.*)"
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
        traces = [m.groupdict() for m in TRACE_RE.finditer(text)]
        gpu_profiles = [m.groupdict() for m in GPU_OP_RE.finditer(text)]
        accepted = []
        accepted_indices = []
        for index, match in enumerate(matches):
            step = int(match["step"])
            total = int(match["total"])
            if step <= args.skip_steps:
                continue
            if args.exclude_final and step == total:
                continue
            accepted.append(match)
            accepted_indices.append(index)

        accepted_traces = [traces[index] for index in accepted_indices if index < len(traces)]
        accepted_profiles = [gpu_profiles[index] for index in accepted_indices if index < len(gpu_profiles)]

        tps = [int(m["tps"]) for m in accepted]
        ms = [int(m["ms"]) for m in accepted]
        ops = [int(m["ops"]) for m in accepted]
        losses = [float(m["loss"]) for m in accepted if math.isfinite(float(m["loss"]))]
        phase_medians = {
            phase: statistics.median(int(trace[phase]) for trace in accepted_traces)
            if accepted_traces else None
            for phase in ("data", "fwd", "bwd", "gradnorm", "clip", "optim", "flush")
        }
        profile_medians = {
            field: statistics.median(float(profile[field]) for profile in accepted_profiles)
            if accepted_profiles else None
            for field in ("flushes", "waited", "dgc", "ops_per_flush")
        }
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
            "phase_median_ms": phase_medians,
            "gpu_profile_median": profile_medians,
            "top_kinds": accepted_profiles[-1]["kinds"] if accepted_profiles else None,
            "top_kernels": accepted_profiles[-1]["kernels"] if accepted_profiles else None,
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

    print("\n## Median step phases and dispatch shape\n")
    print("| Row | Data ms | Fwd ms | Bwd ms | Grad ms | Optim ms | Flushes | Ops/flush | DGC flushes |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        phases = row["phase_median_ms"]
        profile = row["gpu_profile_median"]

        def nested_number(group: dict[str, object], key: str, digits: int = 0) -> str:
            value = group[key]
            return "n/a" if value is None else f"{float(value):.{digits}f}"

        print(
            f"| {row['name']} | {nested_number(phases, 'data')} | {nested_number(phases, 'fwd')} "
            f"| {nested_number(phases, 'bwd')} | {nested_number(phases, 'gradnorm')} "
            f"| {nested_number(phases, 'optim')} | {nested_number(profile, 'flushes')} "
            f"| {nested_number(profile, 'ops_per_flush', 1)} | {nested_number(profile, 'dgc')} |"
        )

    print("\n## Last accepted operation mix\n")
    for row in rows:
        print(f"- `{row['name']}` kinds: `{row['top_kinds'] or 'n/a'}`")
        print(f"  kernels: `{row['top_kernels'] or 'n/a'}`")

    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
