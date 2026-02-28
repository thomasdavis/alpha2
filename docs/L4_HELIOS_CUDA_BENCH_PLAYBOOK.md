# L4 Helios vs CUDA Benchmark Playbook

This document is the exact process used to benchmark Helios against CUDA on GCP L4 using Fleet.

Use this as a handoff for another coding agent.

## Goal

- Measure **Helios matmul performance vs PyTorch CUDA** on the **same NVIDIA L4 instance**.
- Keep all runs correctness-gated before accepting a perf result.
- Produce reproducible artifacts (JSON summaries) that can be compared across commits.

## Scope

- Backend under test: `@alpha/helios` (Vulkan path).
- CUDA reference: PyTorch CUDA (`torch` on the same VM).
- Primary shapes:
  - `1024x1024x1024`
  - `2048x2048x2048`
  - `3072x3072x3072`
- Required correctness check shape:
  - `384x384x384`

## Preconditions

1. Fleet instance exists and is reachable.
2. Instance has NVIDIA L4 and CUDA runtime available.
3. CLI binary can be deployed via Fleet.

Check Fleet status:

```bash
node apps/cli/dist/main.js fleet status
```

Check GPU identity on target:

```bash
npm run fleet -- run <instance> -- "bash -lc 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'"
```

Expected: `NVIDIA L4`.

## Canonical Baseline Command

Deploy latest local binary:

```bash
npm run fleet -- deploy <instance>
```

Run canonical benchmark:

```bash
npm run fleet -- run <instance> -- "bash -lc 'cd /home/ajax/alpha && ./alpha bench --suite=cuda --iters=100 --warmup=10 --dtype=float16 --heliosVariant=matmul --check=true --checkShape=384x384x384 --out=perf/l4-baseline-$(date -u +%Y%m%dT%H%M%SZ).json'"
```

Notes:

- `--check=true` is mandatory for trusted results.
- `--heliosVariant=matmul` is the default/primary path.
- Output file is JSON summary (not CSV).

## Repeat/Median Protocol (Required)

Single runs can be noisy. Always do at least 3 runs for comparison:

```bash
npm run fleet -- run <instance> -- "bash -lc '
set -euo pipefail
cd /home/ajax/alpha
for i in 1 2 3; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  out=perf/l4-repeat-${i}-${ts}.json
  ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --heliosVariant=matmul --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384 --out=$out
done
'"
```

Acceptance rule:

- Do not claim improvement from one run.
- Use median `heliosMs` for `2048` and `3072`.
- Treat wins smaller than ~2% as noise unless repeated across multiple cycles.

## Output Format

Each `--out` file contains:

- run metadata (`timestamp`, `iters`, `warmup`, `dtype`)
- Helios device/capability info
- CUDA reference metadata (`torch_version`, `cuda_runtime`, GPU)
- per-shape rows:
  - `shape`
  - `heliosMs`
  - `heliosTflops`
  - `cudaMs`
  - `cudaTflops`

Quick parse:

```bash
jq -r '.rows[] | "\(.shape),helios_ms=\(.heliosMs),cuda_ms=\(.cudaMs),ratio=\(.cudaMs / .heliosMs)"' perf/<file>.json
```

## Correctness Gate Details

Helios run is rejected if check fails:

- `fails > 0` for the configured `checkShape`
- tolerance:
  - `atol=0.02`
  - `rtol=0.01`

Canonical check line should show `fails=0/...`.

## Variant Sweeps (Optional)

Used when investigating regressions or experimental paths:

- direct load:
  - `HELIOS_COOP_DIRECT_LOAD=1`
- f16 accum experiment:
  - `HELIOS_COOP_F16_ACCUM=1`
- subgroup tiling:
  - `HELIOS_COOP_F16IN_SUBGROUP_TILES=2x2` (or `2x1`, `1x2`, `4x1`)
- coop tile override:
  - `HELIOS_COOP_TILE=16x8x16` (or other supported mode)

Example:

```bash
npm run fleet -- run <instance> -- "bash -lc 'cd /home/ajax/alpha && HELIOS_COOP_DIRECT_LOAD=1 ./alpha bench --suite=cuda --iters=40 --warmup=8 --dtype=float16 --heliosVariant=matmul --shapes=2048x2048x2048,3072x3072x3072 --check=true --checkShape=384x384x384 --out=perf/l4-direct-$(date -u +%Y%m%dT%H%M%SZ).json'"
```

## Health Checks for Bad/Noisy Runs

If CUDA and Helios both suddenly slow down:

1. Check active compute processes:

```bash
npm run fleet -- run <instance> -- "bash -lc 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'"
```

2. Check current clocks/power/temp:

```bash
npm run fleet -- run <instance> -- "bash -lc 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader'"
```

3. Re-run 3x median protocol before accepting any conclusion.

## Comparison Rules Across Commits

For each candidate commit:

1. Deploy commit to same L4 instance.
2. Run canonical benchmark command.
3. Run 3x repeat protocol.
4. Compare median `heliosMs` vs previous commit medians (same shapes/iters/warmup).
5. Require:
   - correctness pass (`fails=0`)
   - improvement on both 2048 and 3072, or clear justification if only one improves.

## Commit Message Template

When there is a real win, include percentage and absolute values:

```text
perf: <change summary> (L4: 2048 <old_ms>-><new_ms> <pct>%, 3072 <old_ms>-><new_ms> <pct>%)
```

If no win, still document outcome explicitly:

```text
perf: <change summary> (L4 net ~0%, correctness preserved)
```

## Fleet One-Shot Alternative

You can use the automated cycle script if you want full ephemeral VM lifecycle:

```bash
npm run fleet:bench:cuda -- --shutdown=delete
```

This is slower for tight optimization loops. For rapid iteration, reuse one warm L4 instance and use `fleet deploy` + `fleet run`.

## Suggested Agent Loop

1. Make one focused change.
2. Build locally.
3. `fleet deploy`.
4. Run canonical benchmark + 3 repeats.
5. Record medians.
6. Keep/revert based on data.
7. Commit with explicit numeric delta.

