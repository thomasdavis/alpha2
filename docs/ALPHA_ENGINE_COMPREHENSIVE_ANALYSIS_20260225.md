# Alpha Training Performance Analysis (Comprehensive)

Date: February 25, 2026
Repo: `/home/ajax/repos/models/alpha`

## Scope

This report is intentionally limited to **training performance**.

It covers:

- training-path architecture (TS autograd/model + Helios GPU backend)
- current bottlenecks and where step time is spent
- realistic 10x performance strategy (prioritized)
- training-focused algorithm upgrades
- measurement and validation plan

It does **not** cover inference/web serving performance.

## Executive Summary

Alpha already has a credible high-performance training foundation for a from-scratch TS engine:

- custom autograd and GPT forward/backward in TypeScript
- graph-batched GPU execution (Helios) to reduce dispatch overhead
- fused FlashAttention and GPU AdamW
- mixed precision support for activation storage (f16 casts)
- strong runtime instrumentation in the trainer

The biggest reason current training throughput is not higher is **not** a missing framework. It is a combination of:

- underfilled GPU workloads (batch/model size too small for device)
- conservative synchronization in the trainer (`gc()` + `syncGpu()` every step)
- hot-loop I/O waits (awaited metrics writes)
- missing mixed-precision **compute** kernels (f16 storage exists, f16/bf16 GEMM/attention compute paths do not)

A practical 10x training speedup is realistic as a staged plan, but it will come from multiple layers:

- occupancy/packing (2x-4x)
- sync/GC and I/O policy improvements (1.3x-2x)
- mixed-precision compute kernels (2x-5x hardware-dependent)
- more fusion + memory lifetime optimization (1.2x-2x)

## Training Architecture (Performance-Relevant)

## 1) Trainer Runtime (`packages/train/src/trainer.ts`)

The trainer is already operationally mature and highly instrumented. Important performance-relevant features that exist today:

- gradient accumulation
- dynamic loss scaling for mixed precision
- grad norm and clipping telemetry
- per-step timing breakdowns
- optional GPU memory diagnostics
- checkpointing and eval integration

This is good because it means the codebase can support serious optimization work without flying blind.

Key hot-loop behaviors to optimize:

- forced `gc()` in the training step loop (`packages/train/src/trainer.ts:525`)
- forced `syncGpu()` in the training step loop (`packages/train/src/trainer.ts:533`)
- awaited metrics file writes in the hot loop (`packages/train/src/trainer.ts:637`)

These decisions improve stability but reduce throughput once memory reuse is stable.

## 2) Model Forward/Backward (`packages/model/src/gpt.ts` + `packages/autograd/src/ops.ts`)

Training path characteristics:

- GPT decoder-only transformer
- optional FlashAttention path when backend supports it
- activation checkpointing
- mixed precision activation storage via f16 cast between layers

Good design choices already implemented:

- cached position indices (`posIndicesCache`) to avoid repeated allocations/uploads
- cached causal masks (`causalMaskCache`)
- deterministic dropout replay support through `DropoutRng` for checkpointing correctness

This means many easy correctness/perf footguns have already been addressed.

## 3) Helios GPU Backend (`packages/helios/src/backend.ts`)

Helios is the core training accelerator and already includes several high-value mechanisms:

- graph batching (`ComputeGraph`) to collapse many ops into fewer GPU submissions
- lazy readback to avoid unnecessary synchronization
- workgroup auto-tuning
- fused FlashAttention forward/backward
- GPU AdamW optimizer step

Key references:

- `packages/helios/src/backend.ts:141` (`autoTuneWgSize`)
- `packages/helios/src/backend.ts:444` (`MAX_PENDING_OPS`)
- `packages/helios/src/backend.ts:467` (`ComputeGraph`)
- `packages/helios/src/backend.ts:1945` (`flashAttention`)
- `packages/helios/src/backend.ts:2800` (`adamwStep`)

Important limitation today:

- there is f16 storage/cast support, but **not** high-throughput f16/bf16 GEMM/attention compute kernels
- current matmul kernels are tiled f32 kernels in `packages/helios/src/kernels/matmul.ts`

That is the single biggest kernel-level upside remaining.

## Current Performance Signals (What the Repo Data Already Shows)

The existing L4 diagnostic is highly informative for training optimization direction:

- low average GPU utilization (`6-39%`) (`L4_RUN_DIAGNOSTIC_20260225.md:97`)
- low VRAM usage relative to 24GB L4 (`~12-17%`, noted as underutilized) (`L4_RUN_DIAGNOSTIC_20260225.md:103`)
- step time mostly forward + backward (healthy split, but GPU underfilled) (`L4_RUN_DIAGNOSTIC_20260225.md:63`, `L4_RUN_DIAGNOSTIC_20260225.md:64`)

Interpretation:

- the engine is already doing substantial work correctly
- the current training configuration is leaving a lot of hardware on the table
- system-level tuning should be prioritized before deep kernel rewrites

## Bottleneck Breakdown (Training Only)

## A) Under-Occupancy (Highest Immediate ROI)

Symptoms:

- low GPU utilization and VRAM usage on L4
- relatively small model for the device (17.4M params in the diagnostic run)

Impact:

- kernels run correctly but do not saturate the GPU
- dispatch overhead and synchronization cost become a larger fraction of step time

What to do first:

- increase batch size until VRAM is materially used
- tune grad accumulation vs true batch size based on whether kernels are compute- or memory-bound
- run larger width/depth configurations on L4 if training objective allows it

## B) Trainer Synchronization Policy (Large Throughput Tax)

Current behavior (safety-first):

- explicit `gc()` in-step
- explicit `syncGpu()` in-step

Why it costs throughput:

- kills overlap between CPU orchestration and queued GPU work
- forces step-level waits even when memory pools are stable
- turns transient memory-management issues into permanent per-step overhead

Recommended change:

- make sync/GC adaptive and policy-driven instead of every step

Suggested controls:

- `syncEvery` (e.g. every 5-20 steps)
- `gcEvery` (e.g. every 20-100 steps)
- `syncOnPoolGrowthMB`
- `syncOnDeferredReleaseCount`

## C) Hot-Loop I/O Waiting (Easy Win)

Current behavior:

- `await metricsHandle.write(...)` each step

Why it matters:

- storage backpressure can stall the training loop
- effect grows with remote disks, noisy hosts, or long runs

Recommended fix:

- buffered metrics writer queue
- flush on checkpoint/eval/end, not every step

## D) Missing Mixed-Precision Compute Kernels (Largest Kernel-Level Upside)

What exists:

- f16 storage and cast kernels (`packages/helios/src/backend.ts:1207`)
- f16 helper/kernel generation infrastructure (`packages/helios/src/kernels/f16.ts`)

What is missing:

- f16/bf16 GEMM compute kernels (with fp32 accumulation)
- f16/bf16 attention compute paths

Consequence:

- training still leaves major math throughput on the table, especially on modern NVIDIA GPUs
- current mixed precision mostly saves memory, not enough compute time

## E) Incomplete Fusion on Training Path

Already fused (good):

- FlashAttention
- AdamW

Still expensive when unfused:

- separate Q/K/V projections (three GEMMs)
- MLP `fc1 -> GELU -> fc2` with intermediate allocations
- extra reduction passes for grad norm / clipping diagnostics

Next fusion targets should focus on reducing graph ops and intermediate tensors, not just raw kernel count.

## 10x Training Performance Roadmap (Prioritized)

## Phase 1: Fastest Practical Gains (1-3 days, low risk)

### 1. Fill the GPU (Batch / Model Sizing)

Expected gain: **2x-4x** depending on current run config and hardware.

Actions:

- run batch sweep to VRAM target (e.g. 50-80% used, not 15%)
- measure tokens/sec, step time, GPU util, OOM threshold
- compare true batch increase vs grad accumulation increase

Why first:

- the diagnostic already shows underutilization on L4
- no kernel work required

### 2. Adaptive Sync/GC Policy in Trainer

Expected gain: **1.2x-1.8x** (sometimes more if current steps are small).

Actions:

- replace unconditional every-step `gc()` / `syncGpu()` with configurable cadence and pressure triggers
- keep current behavior as a debug/safe mode

Metrics to watch:

- step time variance
- deferred release counts
- pool growth trends
- OOM incidence

### 3. Buffered Metrics Writer

Expected gain: **1.05x-1.2x** (depends on filesystem and run frequency).

Actions:

- write metrics into in-memory buffer
- flush every N steps / on checkpoint / on eval / on exit

This is an easy engineering win and helps isolate compute bottlenecks more accurately.

## Phase 2: Kernel/Graph Upgrades (1-3 weeks)

### 4. Mixed-Precision GEMM and Attention Compute

Expected gain: **2x-5x** on supported GPUs (hardware and implementation dependent).

Priority order:

1. matmul (largest training cost center)
2. attention kernels (forward then backward improvements)
3. layernorm/reductions if profiles show bandwidth pressure

Implementation direction (consistent with Alpha philosophy):

- TS-generated SPIR-V kernels, not external shader compilers
- f16/bf16 inputs with fp32 accumulation
- hardware-specific variants selected at runtime by device capabilities

### 5. Grouped QKV Projection (One GEMM Instead of Three)

Expected gain: **1.2x-1.5x** on transformer block forward/backward segments.

Concept:

- combine `wq`, `wk`, `wv` into a single packed weight
- compute one projection and split result

Benefits:

- fewer GEMM launches
- better cache and memory locality
- fewer intermediate tensors

Tradeoff:

- parameter packing/serialization changes (manageable)

### 6. MLP Fusion Improvements

Expected gain: **1.1x-1.4x** depending on hidden sizes.

Targets:

- fuse bias + GELU where possible
- reduce temporary tensor churn around `fc1 -> activation -> fc2`

### 7. Fused or Cheaper Grad-Norm/Clip Diagnostics

Expected gain: modest for large models, but useful once kernels are faster.

Reason:

- as forward/backward speed improves, diagnostics become a bigger fraction of step time
- trainer already computes detailed per-param norms for telemetry; make that path cheaper or configurable

## Phase 3: Memory Lifetime and Scheduling (2-6 weeks)

### 8. Explicit Temporary Lifetime Management / Step Arena

Expected gain: **1.2x-2x** and better stability.

Goal:

- reduce dependence on FinalizationRegistry timing and explicit `gc()`
- keep VRAM reuse deterministic enough to avoid per-step global syncs

Approach:

- temporary tensor arena (step-scoped or microbatch-scoped)
- tape-assisted explicit release scheduling
- pool pressure heuristics driven by actual graph structure

### 9. Sequence Packing (Training Data Efficiency)

Expected gain: **1.2x-2x effective tok/s** for datasets with many short examples.

Why it matters:

- improves token utilization within fixed `blockSize`
- reduces wasted compute on padding/fragmentation
- increases effective throughput without changing kernels

This is one of the best algorithmic systems improvements for training throughput.

## Training-Focused Algorithm Upgrades (Quality/Compute ROI)

These are algorithm changes that affect training efficiency or quality-per-FLOP.

## 1) RoPE (Rotary Positional Embeddings)

Why training-relevant:

- better extrapolation and context behavior than learned absolute embeddings in many settings
- often better quality for the same parameter count and training budget
- supports future longer-context training work cleanly

Engineering fit:

- implementable in TS model code + Helios attention kernels

## 2) RMSNorm + SwiGLU (vs LayerNorm + GELU)

Why training-relevant:

- strong modern baseline for quality-per-compute
- can improve convergence quality at similar parameter budgets
- may simplify/accelerate some kernel paths relative to LN+GELU stacks (depends on implementation)

Recommended rollout:

- add as architecture option, not immediate replacement
- benchmark quality and throughput on same token budget

## 3) Sequence Packing and Curriculum Batching

Why training-relevant:

- direct throughput gain on real corpora with variable lengths
- can improve early-stage stability if curriculum/bucketed batching is used carefully

This should be considered a first-class training optimization, not just a data-loader feature.

## 4) GQA/MQA (If Inference Is Not a Concern Yet, Still Consider for Training Cost)

Even though GQA/MQA are often discussed for inference, they also reduce training memory bandwidth and KV-related costs.

When to prioritize:

- if targeting longer contexts or larger batch sizes on constrained VRAM
- if training throughput is limited by attention memory bandwidth rather than GEMM math

## What Will Not Give 10x Soon (Low Priority for Training Throughput)

- minor TypeScript loop micro-optimizations in already non-hot paths
- optimizer swaps (AdamW -> Lion/SGD/etc.) before kernel and occupancy improvements
- more logging/telemetry unless it directly replaces slower diagnostics

These can matter later, but they are not the current throughput bottleneck.

## Concrete Training Benchmark Plan (Recommended)

Use a reproducible benchmark matrix before and after each optimization phase.

## Metrics to Record Per Run

- tokens/sec (mean, p50, p95)
- ms/step
- GPU utilization
- VRAM used
- forward/backward/gradnorm/optim/flush timings
- OOM incidence / pool growth warnings
- loss curve parity over fixed token budget

## Benchmark Matrix

For each target GPU (e.g. L4, A100, 4090):

- batch size sweep
- grad accumulation sweep
- model size sweep (`nLayer`, `nEmbd`, `blockSize`)
- sync policy sweep (`syncEvery`, `gcEvery`)

This should come before kernel rewrites so the optimization order is data-driven.

## Suggested Implementation Order (Training Performance Only)

### Immediate

1. Add adaptive trainer sync/GC policy
2. Buffer metrics writes
3. Run occupancy sweeps (batch/model sizing) on target GPU
4. Capture baseline traces using existing trainer timings

### Near-term

1. Grouped QKV projection path
2. MLP fusion improvements
3. Sequence packing
4. Cheaper/configurable grad diagnostics

### Medium-term

1. f16/bf16 GEMM compute kernels (fp32 accumulation)
2. f16/bf16 attention compute kernels
3. Memory lifetime planner / step arena
4. RoPE + RMSNorm/SwiGLU variants with throughput/quality comparison

## Notes From This Review Session

- This report intentionally excludes inference and web performance topics.
- I reverted the code changes I previously made in this session (web init race fix and Helios matmul fallback edits), per request.
- No new code changes are included in this report.

## Key Files for Training Performance Work

- `packages/train/src/trainer.ts`
- `packages/model/src/gpt.ts`
- `packages/autograd/src/ops.ts`
- `packages/helios/src/backend.ts`
- `packages/helios/src/kernels/matmul.ts`
- `packages/helios/src/kernels/attention.ts`
- `packages/helios/src/kernels/optimizer.ts`
- `packages/helios/src/kernels/f16.ts`

