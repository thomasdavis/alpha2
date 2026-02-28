# CUDA Benchmarking (Helios vs PyTorch CUDA)

This project now includes a local side-by-side benchmark:

- Helios matmul benchmark (`@alpha/helios`, Vulkan path)
- CUDA reference benchmark (PyTorch on `cuda:0`)

Run:

```bash
npm run bench:cuda -- --iters=12 --warmup=4 \
  --shapes=1024x1024x1024,2048x2048x2048,3072x3072x3072
```

## Fleet + GCloud One-Shot Cycle (L4)

This repo now includes an automated cycle script that:

1. creates or starts an L4 VM,
2. injects a temporary fleet entry,
3. deploys compiled Alpha binary + prebuilt native addon via Fleet,
4. runs `alpha bench --suite=cuda` remotely,
5. downloads artifacts locally,
6. stops/deletes the VM (configurable).

Run:

```bash
npm run fleet:bench:cuda -- \
  --zone=us-central1-b \
  --machine=g2-standard-4 \
  --iters=12 \
  --warmup=6 \
  --shapes=1024x1024x1024,2048x2048x2048,3072x3072x3072 \
  --shutdown=delete
```

Notes:

- The cycle now skips `fleet setup` by default for speed. Add `--setup` if you explicitly want Nix shell setup/warmup.
- Remote Vulkan is forced to a headless NVIDIA ICD at `/etc/vulkan/icd.d/nvidia_icd_headless.json` to avoid `llvmpipe` fallback.
- Artifacts include `vulkan-summary.txt`, `timings.csv` (remote bench step), and `cycle-timings.csv` (local full-cycle stage timing).

Artifacts are saved to:

- `runs/<instance>-<run_id>/`
- `perf/fleet-cuda/<instance>-<run_id>/`

Script path:

- `scripts/fleet-cuda-bench-cycle.sh`

## Architecture Check

CUDA benchmarking requires an NVIDIA GPU.

Quick checks:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
  print("device", torch.cuda.get_device_name(0))
PY
```

If your laptop GPU is Intel/AMD-only, CUDA comparison is not possible on that machine.  
Helios still benchmarks there (Vulkan path), but no CUDA baseline can be produced.

## PyTorch CUDA Setup

Example install (CUDA 12.8 wheels):

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

Then rerun `npm run bench:cuda`.

## Options

- `--iters=<n>`: timed iterations
- `--warmup=<n>`: warmup iterations
- `--shapes=<MxKxN,...>`: comma-separated matrix shapes
- `--dtype=float16|float32|bfloat16`: CUDA reference dtype
- `--python=<path>`: Python interpreter for CUDA reference script
- `--shutdown=delete|stop|none`: post-run instance handling (fleet cycle script)
- `--setup`: run `fleet setup` before deploy (off by default)

### Cooperative Matmul Controls

- Coop matmul is enabled by default when Helios reports support on the device.
- Use `HELIOS_DISABLE_COOP_MAT=1` to force-disable coop kernels for diagnostics.

## Output

The script prints:

- Helios device info
- CUDA device + Torch/CUDA versions
- per-shape latency and TFLOP/s
- `h_vs_cuda` ratio (`>1` means Helios faster on that shape)
