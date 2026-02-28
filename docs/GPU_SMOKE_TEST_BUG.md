# GPU Smoke Test Bug — Helios + Large BPE Data Hang

## Status: OPEN (2026-02-28)

## Problem

Training hangs indefinitely when using the helios (Vulkan) backend with BPE tokenizer and datasets >= ~33MB. The process gets stuck after `smoke_test: FAIL` with 0% GPU utilization. No training steps are ever produced.

The smoke test does `add(1, 2)` on the GPU and checks the result equals 3. When it fails, it means the GPU is producing wrong results, but training attempts to continue anyway and hangs.

## Reproduction

SSH into either GCP instance (cognitive or train) and run:

```bash
cd ~/alpha
source .env.local

# WORKS — small data + BPE + helios
./alpha train --data=data/small.txt --backend=helios --tokenizer=bpe-64k \
  --domain=concordance --iters=5 --batch=1 --dim=64 --heads=2 --layers=2 --block=64

# WORKS — large data + BPE + CPU
./alpha train --data=data/concordance-v2.txt --backend=cpu_ref --tokenizer=bpe-64k \
  --domain=concordance --iters=5 --batch=1 --dim=64 --heads=2 --layers=2 --block=64

# WORKS — large data + char tokenizer + helios
./alpha train --data=data/concordance-v2.txt --backend=helios --tokenizer=char \
  --domain=concordance --iters=5 --batch=1 --dim=64 --heads=2 --layers=2 --block=64

# HANGS — large data + BPE + helios
./alpha train --data=data/concordance-v2.txt --backend=helios --tokenizer=bpe-64k \
  --domain=concordance --iters=5 --batch=1 --dim=64 --heads=2 --layers=2 --block=64
# Shows "smoke_test: FAIL", then hangs at 0% GPU
```

## Isolation Matrix

| Backend | Tokenizer | Data Size | Result |
|---------|-----------|-----------|--------|
| cpu_ref | bpe-4k | 33MB | WORKS |
| helios | char | 33MB | WORKS |
| helios | bpe-4k | 1KB | WORKS |
| helios | bpe-4k | 1MB | WORKS |
| helios | bpe-4k | 10MB | WORKS |
| helios | bpe-4k | 20MB | WORKS |
| helios | bpe-4k | 30MB | WORKS |
| helios | bpe-4k | 33MB | **HANGS** (smoke_test: FAIL) |

The threshold is somewhere around 30-33MB of input data with BPE tokenizer + helios.

## Symptoms

1. Training output shows `smoke_test: FAIL` (gpu add(1,2) != 3)
2. `nvidia-smi` shows 0% GPU utilization
3. Process state is `do_poll` (blocked in Vulkan timeline semaphore wait)
4. No training steps are ever logged
5. Process must be killed manually

## Key Files

- `packages/helios/src/backend.ts` — smoke test at ~line 945, syncGpu at ~line 832
- `packages/helios/native/helios_vk.c` — `waitTimelineValue` at line 837 uses infinite timeout (`~0ULL`), `submitCmdBufAsync` at line 811 uses timeline semaphores
- `packages/train/src/trainer.ts` — training loop

## What's NOT the Cause

- **Not Nix-related** — hangs identically with and without Nix shell
- **Not env vars** — same behavior with identical env on both instances
- **Not Xvfb/DISPLAY** — Vulkan works fine (vulkaninfo succeeds, small data trains fine)
- **Not the model config** — hangs even with tiny model (dim=64, 2 layers)
- **Not the training loop** — never reaches the first training step

## Hypothesis

BPE tokenization of large files is CPU-intensive and runs synchronously in the Node.js event loop before the training loop starts. GPU initialization (device, buffers, pipelines) happens before tokenization, so there's a long gap between GPU init and first GPU use. Possible causes:

- Vulkan device/queue timeout during long CPU-bound tokenization
- Memory pressure from tokenization displacing GPU-mapped buffers
- Node.js event loop starvation causing missed Vulkan fence signals
- Async staging ring or timeline semaphore state corruption during long idle period

## New Findings (2026-02-28, later)

1. A correctness regression in Helios upload path was identified and fixed:
   - In `packages/helios/native/helios_vk.c`, async small uploads reused `transferCmdBuf` without waiting for its prior submit to complete.
   - This could overwrite earlier in-flight upload recordings and produce wrong results (e.g., `add` returning one input instead of sum).
   - Fix: track `lastUploadTimeline` and wait before resetting/re-recording `transferCmdBuf`.

2. Post-fix validation:
   - Local/compiled `bench --suite=gpu --backend=helios` now reports PASS for add/mul/scale/exp/neg correctness checks.
   - This strongly suggests at least one root cause of smoke failures was upload command-buffer reuse.

3. Separate unresolved issue:
   - Bun standalone + cooperative matmul on L4 can still segfault.
   - Runtime mitigation now disables coop matmul on Bun unless explicitly forced with:
     - `HELIOS_ENABLE_COOP_MAT=1`
     - `HELIOS_FORCE_UNSAFE_COOP_MAT=1`
