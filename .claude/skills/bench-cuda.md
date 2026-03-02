---
name: bench-cuda
description: Benchmark all Helios training ops vs PyTorch CUDA on L4 GPU, optimize in a loop to maximize Helios wins
disable-model-invocation: true
---

# Helios vs CUDA — Full Training Op Benchmark Loop

You are an optimization agent. Your job is to make Helios (Vulkan compute shaders) beat PyTorch CUDA across ALL training operations on an NVIDIA L4 GPU. You benchmark, analyze, optimize, and repeat — one focused change per cycle.

## Target Machine

SSH directly to the benchmark instance:
```
ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152
```
- Instance: alpha-bench-l4-coopdbg (NVIDIA L4, 24GB VRAM, 58 SMs, 48MB L2)
- VK ICD: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json`
- Node: `/tmp/node-v22.14.0-linux-x64/bin`
- Repo: `~/alpha-repo`

## The Loop

Repeat this cycle. Each iteration: benchmark → analyze → pick ONE target → implement → benchmark again.

### Step 1: Build and Sync

Build locally then rsync ALL THREE directories:
```bash
# Local build
npx tsc -b packages/helios

# Sync to remote (ALL THREE are required)
rsync -az --delete -e "ssh -i ~/.ssh/google_compute_engine" packages/helios/dist/ ajax@136.113.161.152:~/alpha-repo/packages/helios/dist/
rsync -az --delete -e "ssh -i ~/.ssh/google_compute_engine" packages/helios/src/ ajax@136.113.161.152:~/alpha-repo/packages/helios/src/
rsync -az --delete -e "ssh -i ~/.ssh/google_compute_engine" packages/helios/build/ ajax@136.113.161.152:~/alpha-repo/packages/helios/build/
```

### Step 2: Run the Benchmark

Lock GPU clocks and run with pre-recorded CUDA reference:
```bash
ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152 'sudo nvidia-smi -lgc 2040 2>/dev/null; cd ~/alpha-repo && sleep 10 && VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json PATH=/tmp/node-v22.14.0-linux-x64/bin:$PATH node --no-warnings --loader ts-node/esm scripts/bench-ops.ts --iters=50 --warmup=8 --cuda-json=/tmp/cuda-ref-stable.json 2>&1'
```

**Critical**: The `sleep 10` cooldown prevents thermal throttling artifacts from previous runs. Without it, matmul times can inflate 20-30%.

For quick single-op testing during development (faster iteration):
```bash
ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152 'cd ~/alpha-repo && VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json PATH=/tmp/node-v22.14.0-linux-x64/bin:$PATH node --no-warnings --loader ts-node/esm scripts/bench-ops.ts --iters=50 --warmup=8 --cuda-json=/tmp/cuda-ref-stable.json --filter=<op_name> 2>&1'
```

### Step 3: Analyze Results

The benchmark outputs a score: **H wins / C wins / ties** (5% threshold).

Sort CUDA wins by gap ratio. Focus on the **largest actionable gaps** first:

**Categorize each CUDA win:**
1. **Actionable** — kernel-level optimization possible (matmul tile tuning, attention block sizes, kernel fusion, vectorization)
2. **Structural** — gap caused by fundamental Vulkan vs CUDA differences (cannot fix from SPIR-V):
   - Dispatch overhead (~50µs vs ~20µs) — affects all small tensors
   - L2 cache efficiency — CUDA softmax_attn gets ~970 GB/s (L2), Vulkan gets ~232 GB/s (DRAM)
   - SPIR-V→PTX→SASS compilation quality — ~2× instruction efficiency gap vs hand-tuned cuBLAS
   - 8% bandwidth gap on large elementwise ops — structural to Vulkan code generation

### Step 4: Pick ONE Target

Choose the CUDA win with the largest gap that is **actionable**. Skip structural gaps.

Before implementing, check the experiment log at `~/.claude/projects/-home-ajax-repos-models-alpha/memory/bench-experiments.md` to avoid repeating failed experiments.

### Step 5: Implement and Test

Make ONE focused change. Test with `--filter=<op>` first, then run the full benchmark.

If the change **improves** the target op without regressing others: keep it.
If it **regresses** anything: revert immediately and document why in the experiment log.

### Step 6: Update Experiment Log

After each experiment (success or failure), update the experiment log:
`~/.claude/projects/-home-ajax-repos-models-alpha/memory/bench-experiments.md`

Include: what was changed, the result (ms before/after, CUDA gap), and root cause analysis.

## Key Source Files

- `packages/helios/src/backend.ts` — ALL dispatch logic, tile selection, kernel routing
- `packages/helios/src/kernels/attention.ts` — Flash attention forward/backward kernels
- `packages/helios/src/kernels/matmul-coop.ts` — Cooperative matrix matmul
- `packages/helios/src/kernels/elementwise.ts` — Scale, add, mul, relu, gelu kernels
- `packages/helios/src/kernels/nn.ts` — Softmax, layernorm, silu, embedding kernels
- `packages/helios/src/kernels/index.ts` — Kernel registry (regex matching)
- `packages/helios/src/kernels/helpers.ts` — SPIR-V builder helpers
- `scripts/bench-ops.ts` — Benchmark script (shapes, CUDA ref loading)

## Benchmark Shapes (Training Config)

B=1, T=512, D=1024, H=16, Dh=64, FFN=2752, V=64000

Key matmul shapes (forward, M=512):
- matmul_qkv: M=512, N=3072, K=1024
- matmul_attn_out: M=512, N=1024, K=1024
- matmul_swiglu_g/u: M=512, N=2752, K=1024
- matmul_swiglu_p: M=512, N=1024, K=2752
- matmul_lm_head: M=512, N=64000, K=1024

## Known Structural Gaps (DO NOT attempt to fix)

These have been exhaustively tested across 10+ sessions and are confirmed unfixable from SPIR-V:

| Category | Ops | Gap | Root Cause |
|----------|-----|-----|-----------|
| Dispatch overhead | launch_overhead, small elementwise (512x1024) | 1.2-26× | Vulkan dispatch ~50µs vs CUDA ~20µs |
| L2 cache | softmax_attn (16MB fits in 48MB L2) | ~4.4× | CUDA ~970 GB/s L2, Vulkan ~232 GB/s DRAM |
| SPIR-V code quality | flash_attn_fwd/bwd | 2.3-3.4× | cuBLAS hand-tuned SASS vs SPIR-V→PTX→SASS |
| Bandwidth gap | large elementwise (4096sq) | 1.06-1.11× | 8% Vulkan vs CUDA, structural |
| silu_512x2752 | silu small | ~2.5× | 5.6MB tensor L2-cached by CUDA, DRAM by Vulkan |

## Exhausted Optimization Approaches (DO NOT retry)

These have been tested and confirmed to provide no benefit or regression on L4:

- **Subgroup ops for reduction** (LN, softmax): Neutral on L4 — warp shuffles ≈ shared memory speed
- **Register-resident softmax** (Function-scope array): 25% regression — L1-scratch overhead
- **ILP vec4x2 for large elementwise**: Marginal — GPU already has 32 warps/SM saturating memory pipeline
- **wgSize=128 for elementwise**: Worse — fewer WGs/SM, disrupts matmul tiling
- **r1x1 matmul tiles (32x32)**: Removed — 8 FLOP/byte arithmetic intensity too low
- **Split-K matmul**: 1-2ms overhead from separate reduction kernel
- **Double buffering**: No benefit
- **GROUP_M swizzle**: ±1%, noise
- **shmemPad=4**: -7%
- **Flash attention V2 (compile-time j-unrolling)**: 40% regression from icache pressure
- **kMulti tuning (1,2,4)**: All within 2%
- **Occupancy threshold < 96 WGs**: r4x4 at 64 WGs worse than r2x2 at 256 WGs

## Optimization Techniques That Worked

- **Bc=16 for flash_attn_fwd** (halved shmem → 2× occupancy): +32%
- **BrDKV=16, BcDKV=16 for flash_attn_bwd**: +5%
- **NonReadable decoration on write-only buffers**: +1-2% elementwise, +5-16% layernorm
- **Stride addressing for matmul**: +3.4%
- **r4x2 occupancy fallback** (intermediate between r4x4 and r2x2): helps medium shapes
- **Adaptive kMulti=2** for <512 WGs: reduces shmem, +occupancy
- **wgSize=64 for LayerNorm forward**: 8% improvement (sweet spot for dim=1024)
- **Online softmax as default**: +23% for small dims
- **vec4x2 for binary ops** (add/sub/mul ≥1M elements): +2-3% on large tensors

## Rules

1. **One change at a time.** Implement ONE optimization, measure, keep/revert.
2. **Always lock GPU clocks** before benchmarking: `sudo nvidia-smi -lgc 2040`
3. **10s cooldown** between benchmark runs to avoid thermal artifacts.
4. **Use --filter for quick iteration** — only run the full benchmark for final validation.
5. **Document every experiment** in bench-experiments.md, whether success or failure.
6. **Never chase structural gaps** — check the "Known Structural Gaps" table first.
7. **Check experiment log before implementing** — don't repeat failed experiments.
8. **Thermal variance**: Score can vary ±3 across runs. Only count improvements ≥5%.
9. **Don't over-optimize**: If most CUDA wins are structural, the loop is complete.
