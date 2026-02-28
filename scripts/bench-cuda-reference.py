#!/usr/bin/env python3
"""
CUDA reference benchmark for Alpha comparisons.

Outputs JSON to stdout so TS tooling can compare Helios vs CUDA.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple


def parse_shapes(raw: str) -> List[Tuple[int, int, int]]:
  shapes: List[Tuple[int, int, int]] = []
  for entry in raw.split(","):
    entry = entry.strip()
    if not entry:
      continue
    parts = entry.split("x")
    if len(parts) != 3:
      raise ValueError(f"invalid shape '{entry}', expected MxKxN")
    m, k, n = (int(parts[0]), int(parts[1]), int(parts[2]))
    if m <= 0 or k <= 0 or n <= 0:
      raise ValueError(f"invalid shape '{entry}', dimensions must be > 0")
    shapes.append((m, k, n))
  if not shapes:
    raise ValueError("at least one shape must be provided")
  return shapes


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--shapes", required=True, help="comma-separated MxKxN list")
  parser.add_argument("--iters", type=int, default=30)
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"])
  args = parser.parse_args()

  try:
    import torch
  except Exception as exc:
    print(json.dumps({
      "ok": False,
      "error": f"PyTorch import failed: {exc}",
      "hint": "Install CUDA PyTorch first, e.g. pip install --index-url https://download.pytorch.org/whl/cu128 torch",
    }))
    return 0

  if not torch.cuda.is_available():
    print(json.dumps({
      "ok": False,
      "error": "CUDA is not available in PyTorch.",
      "hint": "Use an NVIDIA GPU with CUDA driver/toolkit and install a CUDA-enabled torch wheel.",
    }))
    return 0

  dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
  }
  dtype = dtype_map[args.dtype]
  torch_dtype_name = str(dtype).replace("torch.", "")

  try:
    shapes = parse_shapes(args.shapes)
  except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    return 0

  device = torch.device("cuda")
  torch.backends.cuda.matmul.allow_tf32 = True
  if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

  rows = []
  for (m, k, n) in shapes:
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    for _ in range(max(0, args.warmup)):
      _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(1, args.iters)):
      c = a @ b
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    avg_ms = (elapsed_s * 1000.0) / max(1, args.iters)
    flops_per_iter = 2.0 * float(m) * float(k) * float(n)
    tflops = (flops_per_iter / (avg_ms / 1000.0)) / 1e12
    rows.append({
      "shape": f"{m}x{k}x{n}",
      "avg_ms": avg_ms,
      "tflops": tflops,
    })

    # Keep tensor live through timing loop.
    del c

  result = {
    "ok": True,
    "framework": "pytorch_cuda",
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0),
    "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "dtype": torch_dtype_name,
    "iters": int(args.iters),
    "warmup": int(args.warmup),
    "rows": rows,
  }
  print(json.dumps(result))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
