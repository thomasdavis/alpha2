# CUDA Parity Research Plan (Helios Vulkan vs CUDA on L4)

Status: Draft for execution
Date: 2026-02-28
Owner: Alpha core runtime

## 1) Objective

Reach practical parity with CUDA for Alpha’s real workloads while preserving the project constraints:

- TS-first control plane and kernel generation
- C only where required (Vulkan native bridge / critical runtime paths)
- Generic code paths first (vendor-neutral by default), with GPU-specific tuning as opt-in overlays

This document is the execution blueprint to move Helios from "works + reasonably fast" to "near-CUDA-class throughput" on NVIDIA L4 and then generalize.

## 2) What "Parity" Means

We need explicit targets. "Beat CUDA on every microbench" is not realistic short-term and not the right KPI.

### 2.1 Primary parity target (L4)

On the canonical matmul shape set:

- `1024x1024x1024`
- `2048x2048x2048`
- `3072x3072x3072`

Target:

- `helios_ms <= 1.25 * cuda_ms` on at least 2/3 shapes
- no shape worse than `1.5x` CUDA
- correctness checks pass under enforced tolerances

### 2.2 Training parity target (L4)

On fixed training workload (same model, data, steps, seed):

- Helios tokens/sec >= 80% of CUDA backend equivalent (if/when CUDA training path exists)
- numerical behavior remains stable (loss curve class, finite gradients, no divergence artifacts)

### 2.3 Engineering parity target

- Compile reliability: `bun:compile` success > 99% in loop runs
- Benchmark reproducibility: <= 5% run-to-run variance for stable configurations
- Every perf toggle has correctness gate and can be disabled by env flag

## 3) Current Baseline (Observed)

Recent L4 results from `alpha bench --suite=cuda --dtype=float16`:

- Helios coop hit rate: 100% (good; tensor-core path is active)
- Typical range:
  - `1024`: ~`0.48-0.50 ms` vs CUDA ~`0.045-0.047 ms` (about `10-11x` slower)
  - `2048`: ~`2.10-2.19 ms` vs CUDA ~`0.246-0.249 ms` (about `8.6-8.9x` slower)
  - `3072`: ~`7.05-7.15 ms` vs CUDA ~`0.731-0.733 ms` (about `9.6-9.8x` slower)

Key findings from recent loops:

- `HELIOS_COOP_DIRECT_LOAD=1`: slower
- `HELIOS_COOP_F16_ACCUM=1`: roughly neutral / noisy, no clear win
- subgroup tiling variants (`1x2`, `2x1`, `2x2`): now correct, but no net throughput win vs baseline on tested shapes
- correctness gate in CUDA bench was necessary; it caught a major invalid fast path earlier

Conclusion: the gap is not from "coop matmul disabled"; it is from deeper kernel/runtime inefficiency versus CUDA’s highly optimized MMA pipelines and launch/runtime stack.

## 4) Root-Cause Hypothesis Tree

### A. Kernel-level inefficiency despite using cooperative matrix

Potential issues:

- non-optimal fragment load/store scheduling around `OpCooperativeMatrixLoadKHR` / `StoreKHR`
- shared-memory traffic and barriers reducing effective tensor-core utilization
- occupancy/register pressure mismatch
- poor overlap of memory fetch and MMA loop

Evidence:

- coop active but still ~9x behind CUDA
- direct-load and subgroup variants did not materially close gap

### B. Runtime/dispatch overhead still too high in tight loops

Potential issues:

- command recording overhead per op
- residual barrier conservatism
- descriptor/binding overhead
- CPU-side orchestration bubbles

Evidence:

- historical bottlenecks were dispatch infrastructure-heavy
- likely still paying non-trivial per-dispatch tax vs CUDA driver stack

### C. Layout and prepack mismatch

Potential issues:

- source tensors not in the ideal memory layout for selected coop tile shape
- repeated runtime transforms instead of one-time prepacking
- transpose/rewrite paths introducing hidden overhead

Evidence:

- enabling alternate tile modes rarely helped
- suggests memory feeding strategy is bottlenecking MMA

### D. Missing graph-level fusion around GEMM envelopes

Potential issues:

- CUDA benefits from heavily fused epilogues/prologues
- Helios may still do extra passes for cast/add/bias/activation/norm

Evidence:

- model-level throughput is sensitive to dispatch count and memory round-trips

### E. Measurement blind spots

Potential issues:

- insufficient in-kernel telemetry
- tuning against noisy signal
- no per-stage stall accounting

Evidence:

- some contradictory one-off wins that disappear under repetition

## 5) Research Program: Phased Plan

## Phase 0: Measurement Hardening (mandatory before major rewrites)

Goal: make optimization signal trustworthy.

Deliverables:

- keep correctness gate in CUDA bench as default for risky modes
- add benchmark mode that emits:
  - per-shape median/p95 over repeated runs
  - coefficient of variation
  - selected kernel variant string (tile, accum type, subgroup tiling, direct/shared path)
- persist run metadata for reproducible comparison

Exit criteria:

- repeated benchmark runs show stable ordering of variants
- no "fast but wrong" mode can pass silently

## Phase 1: Coop GEMM Microarchitecture Deep Dive

Goal: close most of the 9x gap from kernel-side improvements.

Research tracks:

1. Tile-shape search with full runtime implications
- not just `MxNxK` shape selection; include load strategy and stride pattern
- benchmark `16x16x16`, `16x8x16`, `16x8x8` under identical harness

2. Shared-memory path redesign
- explicit double-buffered k-tile pipeline
- reduce full workgroup barriers where safe
- ensure per-subgroup local data regions to avoid alias and bank pressure

3. Direct-load path redesign
- current direct path regresses; profile why:
  - coalescing quality
  - alignment constraints
  - stride handling for transposed variants

4. Accumulation strategy
- compare f16 accumulation with f32 output conversion only under strict accuracy envelope
- gate by task sensitivity, not microbench only

Exit criteria:

- >= 2x Helios improvement on at least one large shape without correctness regressions

## Phase 2: Layout/Prepack Strategy

Goal: feed cooperative kernels data in optimal layout with minimal runtime transforms.

Actions:

- introduce persistent prepacked weight format keyed by kernel profile
- avoid recurring transpose/scatter patterns in hot training loops
- add layout planner in TS backend for matmul call sites

Exit criteria:

- measurable drop in pre/post-GEMM helper dispatch count
- additional 20-40% shape-level gain on 2048/3072

## Phase 3: Dispatch Pipeline Minimization

Goal: reduce CPU/runtime tax per training step.

Actions:

- continue reducing per-op host overhead (batch packing, descriptor strategy, submission cadence)
- verify barrier placement with precise dependency analysis
- ensure async upload/copy paths are fully overlapped where legal

Exit criteria:

- lower CPU wall time share in step profile
- lower variance in tokens/sec under identical runs

## Phase 4: Fusion Around Matmul

Goal: reduce memory traffic and launch count around the core GEMM.

Priority fusions:

- matmul + bias
- matmul + bias + activation (where numerically safe)
- selective norm/epilogue fusion

Rules:

- each fused kernel must pass parity gates
- keep non-fused fallback path intact

Exit criteria:

- significant dispatch reduction in representative training steps
- 10-25% end-to-end tok/s improvement

## Phase 5: Auto-Tuning as Product Feature

Goal: stop hardcoding global assumptions.

Actions:

- startup microtuner for GPU-profiled kernel variant selection
- cache profile by `(gpu, driver, coop properties, model regime)`
- runtime fallback when instability detected

Exit criteria:

- parity-sensitive settings chosen automatically on L4
- performance no longer dependent on hand-set env tuning

## 6) Concrete Experiment Matrix

For each candidate change:

1. Correctness
- `alpha bench --suite=cuda --check=1 --checkShape=384x384x384 ...`
- fail on tolerance breach

2. Shape-level throughput
- compare `1024/2048/3072` latency and TFLOP/s
- report median over N repeated runs

3. Training loop
- run `scripts/run-compiled-benchmark.sh 100`
- run 3-prompt inference smoke

4. Regression policy
- do not merge default-on change if it regresses primary shapes or 100-step tok/s

## 7) Prioritized Backlog (High ROI First)

1. Add per-kernel telemetry tags into benchmark output
2. Implement double-buffered cooperative load pipeline in matmul-coop kernel
3. Add persistent prepack for heavy-reuse weight tensors
4. Reduce conservative barriers in native dispatch path (with dependency proof)
5. Add matmul epilogue fusion prototypes
6. Roll in auto-tuned variant picker

## 8) Risk Register

- Risk: "fast but wrong" kernels
  - Mitigation: correctness gate mandatory for perf modes
- Risk: overfitting to L4
  - Mitigation: keep generic baseline + profile-specific overlays
- Risk: code complexity explosion
  - Mitigation: strict variant naming, telemetry, and kill-switch env flags
- Risk: benchmark noise causing bad decisions
  - Mitigation: repeated-run medians, metadata capture, controlled harness

## 9) What Success Looks Like

Near-term (2-4 weeks):

- Helios matmul gap reduced from ~9x to <=4x on L4 benchmark set
- no correctness regressions
- stable loop pipeline (compile + benchmark + smoke)

Mid-term (4-8 weeks):

- <=2x gap on all canonical shapes
- training tok/s materially closer to CUDA-equivalent throughput envelope

Long-term:

- practical parity envelope (`<=1.25x`) on core shapes and stable training workloads

## 10) Immediate Next Steps

1. Implement Phase 0 benchmark output hardening (median/p95 + variant metadata).
2. Run structured microarchitecture search for coop load/compute scheduling.
3. Add and test persistent prepack for reused matmul weights.
4. Keep subgroup tile controls as opt-in until they win in repeated L4 tests.
5. Maintain default path conservative until it wins both correctness and throughput gates.

