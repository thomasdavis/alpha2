---
name: bench-cuda
description: Benchmark Helios matmul vs PyTorch CUDA on L4 GPU, optimize in a loop until Helios wins
disable-model-invocation: true
---

# Helios vs CUDA Benchmark Loop

You are an optimization agent. Your job is to make Helios matmul outperform PyTorch CUDA on an NVIDIA L4 GPU. You will benchmark, analyze, optimize, and repeat until Helios wins.

## Target instance

Use the `train` fleet instance (L4 24GB GPU). If $ARGUMENTS is provided, use that as the instance name instead.

## Setup

Before starting, ensure the instance is running and reachable:

```bash
node apps/cli/dist/main.js fleet status
```

If the instance is stopped, start it:

```bash
gcloud compute instances start alpha-train --project=$GCP_PROJECT --zone=us-central1-b
```

Verify GPU identity:

```bash
npm run fleet -- run train -- "bash -lc 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'"
```

Expected: `NVIDIA L4`.

## The Loop

Repeat this cycle until Helios beats CUDA on **both** 2048x2048x2048 and 3072x3072x3072 shapes:

### Step 1: Build and deploy

```bash
npm run build -w @alpha/helios && npx turbo build --filter=@alpha/cli --force
npm run fleet -- deploy train
```

### Step 2: Run the benchmark (3x median protocol)

Run 3 repetitions to get stable medians:

```bash
npm run fleet -- run train -- "bash -lc '
set -euo pipefail
cd /home/ajax/alpha
for i in 1 2 3; do
  ts=\$(date -u +%Y%m%dT%H%M%SZ)
  out=perf/l4-repeat-\${i}-\${ts}.json
  ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --heliosVariant=matmul --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384 --out=\$out
done
'"
```

### Step 3: Parse results

Download and parse the JSON results:

```bash
npm run fleet -- run train -- "bash -lc 'cd /home/ajax/alpha && for f in perf/l4-repeat-*.json; do echo \"=== \$f ===\"; jq -r \".rows[] | \\\"\(.shape) helios=\(.heliosMs)ms cuda=\(.cudaMs)ms ratio=\(.cudaMs / .heliosMs)x\\\"\" \$f; done'"
```

### Step 4: Analyze

For each shape, take the **median** heliosMs across the 3 runs. Compare to median cudaMs.

- `ratio > 1.0` means Helios is faster than CUDA
- `ratio < 1.0` means CUDA is still faster
- Ignore wins smaller than ~2% — that's noise

### Step 5: Decide

- **If Helios wins on both 2048 and 3072:** Stop. Commit the result with:
  ```
  perf: <change summary> (L4: 2048 <old>-><new> <pct>%, 3072 <old>-><new> <pct>%)
  ```

- **If CUDA still wins:** Analyze _why_ and make ONE focused change. Then go back to Step 1.

### Step 6: Clean up old results before next cycle

```bash
npm run fleet -- run train -- "bash -lc 'rm -f /home/ajax/alpha/perf/l4-repeat-*.json'"
```

## Where to optimize

The key files for Helios matmul performance:

- `packages/helios/src/kernels/matmul.ts` — SIMT matmul kernel (SPIR-V codegen)
- `packages/helios/src/kernels/matmul-coop.ts` — cooperative matmul kernel
- `packages/helios/src/kernels/helpers.ts` — shared kernel helpers
- `packages/helios/src/backend.ts` — dispatch routing, GPU profiles, buffer management
- `packages/helios/native/helios_vk.c` — Vulkan C runtime (command buffers, semaphores, memory)

Common optimization targets:

- **Tile sizes** — try different workgroup dimensions and tile shapes
- **Memory access patterns** — coalesced loads, shared memory bank conflicts
- **Dispatch overhead** — reduce command buffer recording cost
- **Pipeline barriers** — minimize unnecessary synchronization
- **Buffer reuse** — reduce allocation churn between dispatches
- **Specialization constants** — use Vulkan spec constants for compile-time tile config

## Variant experiments

Test experimental kernel paths with environment variables:

```bash
# Direct load (skip staging)
npm run fleet -- run train -- "bash -lc 'cd /home/ajax/alpha && HELIOS_COOP_DIRECT_LOAD=1 ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384'"

# F16 accumulation (faster but less precise)
npm run fleet -- run train -- "bash -lc 'cd /home/ajax/alpha && HELIOS_COOP_F16_ACCUM=1 ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384'"

# Subgroup tiling variants
npm run fleet -- run train -- "bash -lc 'cd /home/ajax/alpha && HELIOS_COOP_F16IN_SUBGROUP_TILES=2x2 ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384'"
```

## Rules

1. **Correctness is mandatory.** Every bench run uses `--check=true`. If correctness fails (`fails > 0`), the result is invalid. Fix correctness before optimizing further.
2. **Never claim improvement from one run.** Always use 3x median protocol.
3. **One change at a time.** Make one focused optimization, measure, keep/revert.
4. **Commit with numbers.** Every commit message includes before/after ms and percentage delta.
5. **Don't break training.** After kernel changes, verify `npm run fleet -- run train -- "bash -lc 'cd /home/ajax/alpha && ./alpha bench --suite=cuda --iters=5 --warmup=2 --dtype=float16 --check=true --checkShape=384x384x384'"` still passes correctness.

## Health checks

If results look noisy or both CUDA and Helios are unusually slow:

```bash
# Check for competing GPU processes
npm run fleet -- run train -- "bash -lc 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'"

# Check GPU clocks/thermal
npm run fleet -- run train -- "bash -lc 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader'"
```

If the GPU is thermal throttling or another process is using it, kill the competing process or wait for cooldown before benchmarking.
