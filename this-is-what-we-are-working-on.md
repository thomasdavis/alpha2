# This Is What We Are Working On

## Mission
- Make Helios consistently faster than CUDA on benchmark suites, with priority on flash-attention forward (`b1_h16_t512_d64`).
- Keep L4 training runs stable (no allocator-cap/OOM crashes), with reliable remote metrics/events/samples and Discord reporting.
- Push Super Chat quality toward meaningful text with training loss trending down and validation behavior monitored for plateau/overfit.

## Current Repo State (What Changed)

### Major Stability & Memory Pass (2026-03-06)
- **Solved "Too Many Objects" OOM**: Implemented a native slab allocator in `helios_vk.c` to consolidate thousands of small buffers into large slabs. This prevents hitting the L4 driver's hard limit (~8500 allocations).
- **Fixed Gradient & Intermediate Leaks**: Identified and corrected memory leaks in `crossEntropy`, `sliceQkv`, and `reduceBroadcast` backward closures. Also fixed leaks in padded coop matmuls.
- **Optimized Autograd Tape**: `Tape.backward` now takes direct ownership of gradients when use-count is 1, halving the number of `TensorData` objects created during backward passes.
- **Fixed Attention softCap**: Resolved a critical bug where `softCap` was hardcoded to 30.0 in `gpt.ts`. It now correctly respects model/domain configuration, preventing numerical divergence.
- **Improved L4 Telemetry**: Added `getGpuStats()` to the native addon to track true driver-level allocation counts and bytes.

### Flash / Helios / Perf
- Coop2 flash-attention path has been heavily optimized and instrumented (kernel variants, probes, qt variants, no-lse variants, scope/runtime controls).
- Additional allocator/memory-pressure controls and telemetry exist in Helios backend/native path.
- Fleet L4 runtime handling was updated, including node-runtime resolution and L4 env bootstrapping.

### Training Stability + Observability
- Trainer now includes stronger grad-norm safety checks (GPU/CPU suspicious mismatch recheck + fallback modes).
- Trainer now emits richer sample-quality/plateau signals and event logs.
- API/event/samples pipeline was expanded so remote runs can be tailed and correlated by step.
- Tokenizer artifact loading for chat tokenizers includes stale-artifact checks for required chat special tokens.

### Critical Fixes Applied In This Pass
1. `crossEntropy` captured-target cleanup added so temporary target tensors are deterministically releasable via tape cleanup.
2. Training loop now explicitly releases dataloader batch `inputs/targets` tensors after microstep/eval usage.
   - This is required because dataloader reuses TensorData objects in a ring; without invalidation, GPU object-identity caching can keep stale buffers and/or leak handles.
3. Standalone eval path (`packages/train/src/eval.ts`) now clears tape + releases batch/loss tensors + periodic flush.
4. Fleet L4 ICD preference changed to prefer headless ICD path first.

## Known L4 Runtime Caveat
- On the current benchmark L4 host, compiled binary mode (`./alpha`) is currently failing Vulkan init (`vkCreateInstance failed`).
- Node runtime in `/home/ajax/alpha-repo` has been the reliable path for Helios/Vulkan execution.
- Practical rule for now:
  - Use `--runtime=node` for L4 training loops.
  - Keep compiled-binary perf loops for local/known-good environments only.

## Canonical L4 Loop (What to Run)

### 1) Build local CLI before Fleet commands
```bash
npm run build -w @alpha/cli
```

### 2) Deploy binary/addon payload to Fleet instance
```bash
npm run fleet:deploy -- alpha-bench-l4-coopdbg-20260228084511
```

### 3) If training with `--runtime=node`, sync changed source files to `/home/ajax/alpha-repo`
```bash
rsync -az --relative -e "ssh -i ~/.ssh/google_compute_engine" \
  packages/autograd/src/ops.ts \
  packages/train/src/trainer.ts \
  packages/train/src/eval.ts \
  ajax@136.113.161.152:/home/ajax/alpha-repo/

npm run fleet:run -- alpha-bench-l4-coopdbg-20260228084511 -- \
  "cd /home/ajax/alpha-repo && npm run -s build -w @alpha/autograd && npm run -s build -w @alpha/train && npm run -s build -w @alpha/cli"
```

### 4) Start training on L4 (recommended runtime)
```bash
npm run fleet:train -- alpha-bench-l4-coopdbg-20260228084511 \
  --runtime=node \
  --data=/home/ajax/alpha-repo/data/super_chat.txt \
  --domain=super_chat \
  --steps=12000 \
  --tokenizerArtifacts=/home/ajax/alpha-repo/runs/tokenizer-artifacts-super-chat-bpe4k-v3.json
```

### 5) Monitor
```bash
npm run fleet:status -- alpha-bench-l4-coopdbg-20260228084511
npm run fleet:run -- alpha-bench-l4-coopdbg-20260228084511 -- "tail -n 200 /home/ajax/alpha-repo/train.log"
```

### 6) Stop / Resume
```bash
npm run fleet:stop -- alpha-bench-l4-coopdbg-20260228084511
npm run fleet:resume -- alpha-bench-l4-coopdbg-20260228084511 --runtime=node
```

## Benchmark Loop (Helios vs CUDA)

### Local focused CUDA comparison
```bash
npm run bench:cuda -- --iters=12
```

### Fleet L4 benchmark cycle
```bash
npm run fleet:bench:cuda -- --shutdown=delete
```

### Flash-focused probe loop pattern
```bash
HELIOS_WG_SIZE=256 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  npx tsx scripts/bench-ops.ts --iters=20 --warmup=6 \
  --only=flash_attn_fwd_b1_h16_t512_d64,flash_attn_coop2_fwd_sc_b1_h16_t512_d64,flash_attn_coop2_probe
```

## Remote Reporting / Events / Samples
- Remote reporter + Discord hooks come from `.env.local` (`ALPHA_REMOTE_URL`, `ALPHA_REMOTE_SECRET`, `DISCORD_WEBHOOK_URL`).
- Samples are generated on `sampleInterval` and should be correlated with step/event data.
- Use events tailing to catch spikes, plateau, sample degeneracy, checkpoint writes, and failure signatures.

## Working Agreement For Next Agent
1. Keep iterating in a loop: patch -> build -> deploy/sync -> run -> monitor -> analyze -> repeat.
2. Prioritize root-cause fixes over auto-restart hacks.
3. Treat data quality + tokenizer behavior + allocator stability as first-class blockers for loss/sample quality.
4. Do not trust single-run best numbers; use repeat medians and monitor p90 variance.
5. Keep this file updated after each substantial pass.
