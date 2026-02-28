# Alpha: Strategy for the World's Best TypeScript Model Framework

## What We Have

Alpha is a ~88K LOC, fully custom ML training stack written in TypeScript and C. No PyTorch, no JAX, no TensorFlow, no CUDA, no cuBLAS, no NCCL. Everything from SPIR-V code generation to autograd to BPE tokenization is hand-written.

### Current Stack (Feb 2026)

| Layer | Implementation | LOC |
|-------|---------------|-----|
| GPU backend (Helios) | Vulkan compute via C native addon + TypeScript SPIR-V codegen | ~14K |
| SPIR-V kernels | 17 kernel families: matmul, flash attention, layernorm, softmax, adamw, etc. | ~10K |
| Autograd | Tape-based reverse-mode AD with activation checkpointing | ~2.5K |
| Model | GPT with SwiGLU, RoPE, flash attention, BPE-64K | ~1.5K |
| Training | Full loop: gradient accumulation, mixed precision, loss spike detection, remote metrics | ~5K |
| Tokenizers | BPE (train + encode/decode), char, word | ~825 |
| Inference | KV-cached autoregressive decode, zero-alloc tight loop | ~800 |
| Symbiogenesis | Neural architecture search via evolutionary loss | ~130 |
| CPU tensor | Reference backend for correctness testing | ~930 |
| Web dashboard | Next.js with real-time training curves, loss analysis, inference UI | ~8K |
| CLI + Fleet | Build, deploy, train, resume across GCP instances with Nix | ~5K |
| TUI | Terminal dashboard (Ink/React) | ~2K |
| HF Spaces | Standalone inference server, ~50ms/token on free CPU | ~1.5K |
| DB layer | Turso/libsql for run tracking, metrics, checkpoints | ~1.5K |

### What Actually Works Today

- Single-GPU training on L4 (24GB) and A100 (80GB)
- 300M parameter models (dim=1024, 21 layers, 16 heads)
- ~10K+ tok/s on L4 with cooperative matmul + flash attention
- Mixed precision (FP16 forward, FP32 accumulation)
- Activation checkpointing (trade compute for 2x memory)
- Gradient accumulation (effective batch scaling)
- Live metrics streaming to web dashboard
- BPE-64K tokenizer (custom trained)
- Fleet management: deploy, train, resume, monitor across instances
- Reproducible environments via Nix flakes
- OpenAI-compatible API for inference

### What's Genuinely Impressive

1. **SPIR-V from TypeScript** — kernel code is generated programmatically from TS, not hand-written assembly. This is a compiler approach that no other framework uses for Vulkan compute.
2. **Cooperative matmul** — workgroup-level tiling with shared memory, auto-tuned tile sizes per GPU profile. This is the kind of kernel work that takes months in other frameworks.
3. **Flash attention in SPIR-V** — tiled, memory-efficient attention with online softmax. No one else has this in Vulkan.
4. **End-to-end ownership** — there is no layer we don't understand. Every gradient, every kernel, every byte.

---

## Gap Analysis: Alpha vs The World

### Competitive Landscape

| Framework | Language | Strength | Weakness |
|-----------|----------|----------|----------|
| PyTorch | Python/C++ | Ecosystem, research adoption, CUDA maturity | Bloated, slow iteration on core, Python overhead |
| JAX | Python/C++ | Functional purity, XLA compilation, TPU | Google-dependent, steep learning curve |
| MLX | Swift/C++ | Apple Silicon native, clean API | Apple-only, no multi-GPU |
| tinygrad | Python | Minimal, multi-backend, George Hotz energy | Small team, limited scale testing |
| llm.c | C/CUDA | Raw speed, Karpathy credibility | CUDA-locked, training-only, no ecosystem |
| burn | Rust | Type safety, multi-backend | Young, small community |

### Where Alpha Loses Today

**Performance gap is the only gap that matters.** Everything else (API polish, ecosystem, docs) follows from proving we can train competitive models fast.

1. **Single-GPU throughput** — our MFU is low. The dispatch overhaul helped (+13% tok/s) but we're still well below what other frameworks achieve on equivalent hardware. Root cause: Vulkan dispatch overhead, lack of kernel fusion, no graph capture/replay.

2. **No multi-GPU** — we can't train anything that doesn't fit on one GPU. This is the single biggest capability gap. Data parallelism is table stakes.

3. **No Tensor Core path implemented yet** — H100/L4/A100 have matrix acceleration available via Vulkan cooperative matrix (`VK_KHR_cooperative_matrix`). We currently run SIMT matmul kernels and leave matrix hardware underutilized until we add cooperative matrix kernels + tuning.

4. **No data pipeline** — we load entire datasets into memory as text files. No sharding, no streaming, no pre-tokenization. This blocks scale.

5. **No distributed checkpointing** — checkpoints are single-file JSON. No sharding, no async writes, no resume across different GPU counts.

6. **Numerical validation is informal** — we check correctness by eyeballing loss curves. No systematic gradient checking, no CPU↔GPU parity test suite.

---

## Two Load-Bearing Invariants

These are not features. They are structural constraints that shape everything else. Every phase, every PR, every optimization must respect them.

### Invariant A: The Compiled Step

Treat *one optimizer step* as a compiled artifact — a `StepProgram`. This is a stable ABI you can version, diff, cache on disk, and replay across runs.

* **Inputs:** token batch buffers, RNG seed(s), lr/scalars, pointers to param shards
* **Outputs:** loss scalar, updated param buffers, metrics
* **Artifact:** `StepProgram` — the compiled step, described below

If we do this, graph replay + memory planning + dispatch minimization fall out structurally instead of being separate projects. The training loop becomes: compile the step once, then replay it with patched bindings every iteration.

**StepProgram v0 — minimum fields (don't overbuild):**

* **Resource plan**
  * `slabs[]` — (size, usage flags, memory type). Allocated once, never recreated in steady state
  * `views[]` — (slabId, offset, size, stride/meta). How kernels address tensors
* **Execution plan**
  * `cmdBuffers[]` — (1 primary + N secondary, one secondary per transformer block)
  * `barriers[]` — (barrier regions per cmd buffer, execution + memory dependencies)
  * `timelinePoints[]` — (signal/wait values and semantic meaning)
* **Patch plan**
  * `patchPushConstants[]` — small scalars (lr, beta1/beta2, loss scale, RNG seed)
  * `patchDescriptorWrites[]` — precomputed `VkWriteDescriptorSet` templates
  * `patchUniformWrites[]` — per-step batch base pointers
* **Validation metadata**
  * hash of "shape signature" (model config + batch + seq + dtype)
  * hash of "kernel IR signature" (codegen version + specializations)
  * deterministic seed mapping rules

**Key win:** StepPrograms can be stored on disk keyed by `(model config, seq_len, batch, dtype, gpu profile, codegen version)`. On startup, instantly know if we can replay or must recompile.

**Descriptor stability trap:**

If descriptor sets change at runtime, replay collapses. The fix:

* Allocate one **mega descriptor set per StepProgram** (or per module) with a fixed layout
* Bind views through either dynamic offsets (SSBO dynamic offset style) or a small **indirection buffer** (array of base pointers/offsets that kernels index into)
* Patching becomes "update a few 64-bit offsets + buffer handles" instead of "rebuild descriptor sets"

**Rule: replay is only valid in steady state. Any resource creation invalidates the compiled step.**

**Barrier and synchronization plan:**

Static steps still need synchronization correctness. The compiled StepProgram must capture:

* Pipeline barriers (execution + memory dependencies between dispatches)
* Queue ownership transfers (if any)
* Timeline semaphore signal/wait points

Replay re-signals and re-waits the same synchronization points. Without this, replay is "fast but flaky."

**Slow path policy:**

Not every step is identical. Sequence length can vary, loss scale can change, eval steps have different structure.

* **If shapes or layout change → recompile a new StepProgram**
* **If only pointers/scalars change → patch + replay**

That line saves months of debugging. Dynamic cases (different seq_len, occasional eval) go through the compile path. The common case (steady-state training) stays on the fast replay path.

### Invariant B: No Optimization Without a Numerical Gate

Every performance PR must pass before merge:

* CPU reference forward/backward (small shapes)
* GPU forward/backward parity within tolerance (1e-3 for FP16, 1e-6 for FP32)
* Finite-difference gradient checks on a rotating subset of ops
* Determinism checks (within expected FP noise) for fixed seeds

This lets us fuse aggressively without fear. A fused kernel that produces wrong gradients is worse than a slow correct one.

**Sentinel tests that massively reduce time-to-diagnosis:**

* **Gradient fingerprint:** pick a deterministic subset of params (e.g., 1024 floats spread across layers), compute `mean`, `std`, `max`, `l2`, and a 64-bit hash of bitcasted fp32 values. Store as artifact per CI run. Drift shows up immediately across commits.
* **Op-level NaN provenance:** in debug mode, after each op (or each fused region), run a cheap `isfinite` reduction. Record the first failing dispatch ID. When NaNs happen, you know which op produced them without bisecting.
* **Per-op tolerance budget:** instead of one tolerance for everything, maintain a per-op tolerance table. Layernorm, softmax, and attention often need different tolerances under FP16. This prevents over-tightening from blocking useful fusions while still catching real bugs.

---

## The Path: Priorities in Order

The ordering below is load-bearing. Each phase unlocks the next. Skip nothing.

### Phase 1: Fix What's Broken (weeks, not months)

These are bugs and gaps that actively block progress.

**1a. Fix the Helios smoke test hang — as a runtime discipline issue**

The GPU smoke test fails when BPE tokenization of large files blocks the Node.js event loop for too long, corrupting Vulkan timeline semaphore state. Fix with worker_threads, but also enforce a harder invariant: GPU init occurs once and is never interleaved with heavy JS CPU work. Timeline semaphore usage must be isolated to a single "GPU scheduler" thread/context. Don't let Node scheduling jitter be in the same domain as Vulkan scheduling.

**1b. Pre-tokenized binary data format — future-proofed**

Stop loading raw text files. But don't just do `uint16` token IDs — we'll regret it at 64K+ vocab and multi-dataset mixes.

Container spec:

* **Header:** magic, version, dtype (`u16` or `u32`), vocab_size, seq_len, shard_count
* **Shard table:** offsets + token_count per shard
* **Payload:** flat token arrays per shard, memory-mappable layout
* **Optional:** zstd-compressed blocks (for storage, decompress on load)

Per-shard metadata for dataset mixing + provenance:

* dataset ID
* sampling weight
* document boundaries (optional)
* domain tag

This gives us reproducible sampling, curriculum learning later, and per-domain eval later — without redesigning the format.

Then the "data pipeline" later is just: shard selection + windowing + pinned staging. No redesign needed.

**1c. Numerical correctness test suite**

Add a `packages/tests/` test runner that:
- Runs every autograd op through forward + backward on both CPU and GPU
- Compares results within tolerance (1e-3 for FP16, 1e-6 for FP32)
- Runs finite-difference gradient checks on a rotating subset of ops
- Determinism check: same seed → same loss (within FP noise)
- Gates CI: if parity breaks, nothing ships

This is Invariant B made concrete. It must exist before Phase 2 starts.

### Phase 2: Single-GPU Performance (the main event)

This is where Alpha becomes competitive. The ordering within this phase matters — StepProgram first, then memory planner, then fusion. Fusion becomes "edit one compiled op sequence" rather than "rewrite runtime."

**2a. Instrumentation and profiling**

Before optimizing, measure. Wrap every training phase (forward, backward, optimizer, data load, host↔device transfer) with high-resolution timers. Add Vulkan timestamp queries for per-kernel GPU timing. Calculate MFU every step. Export Chrome trace format for visualization. Without this, optimization is guesswork.

**2b. StepProgram compilation + patch table + replay**

This is not "graph replay" — it's step compilation. Treat the optimizer step as a compilable unit:

1. **Trace:** run one step, recording every dispatch, buffer binding, push constant, barrier
2. **Compile:** produce a `StepProgram` — the recorded command buffers + a patch table listing every per-step variable
3. **Replay:** each subsequent step applies patches (new batch pointers, updated lr, RNG seed) and resubmits the pre-recorded command buffers

This is the single biggest performance win available. All CPU-side dispatch overhead amortizes to near zero.

**Two concrete upgrades that pair with StepProgram:**

* **Secondary command buffers per transformer block:** build once, execute from one primary, patch via push constants / indirection buffers. This is the natural granularity for replay — one secondary per layer.
* **Barrier coalescing pass:** during compilation, analyze dependencies and merge barriers. Remove redundant pipeline stage masks. This often gives "free %" without touching kernels.

**2c. Memory planner with slab allocation**

Don't fight Vulkan's allocator — use big slabs + offsets:

* Allocate a small number of large device buffers ("slabs") at init
* Planner assigns tensor lifetimes to `(slab, offset, size, alignment)` at compile time
* Kernels take `(baseBuffer, offset)` not "tensor objects"
* Zero runtime allocation, minimal VRAM fragmentation, predictable peak usage

**Constraints the planner must handle up front:**

* dtype alignment rules (fp16, bf16, fp32, vectorized loads)
* cooperative matrix tile alignment constraints (future-proofing for Phase 4)
* scratch buffers (attention softmax stats, partial sums, reduction temporaries)
* in-place legality (some ops can reuse input buffer, some cannot — planner must know)

**Planner output is SSA-ish.** Every tensor gets:

* `defId` — dispatch that creates it
* `lastUseId` — last dispatch that reads it
* `sizeBytes`, `alignment`
* `slabId`, `offset` — physical placement
* `reusedFrom` (optional) — which earlier tensor this aliases, for debugging

The planner builds lifetimes and computes **alias classes**. Tensors with overlapping lifetimes cannot share memory; tensors with disjoint lifetimes can. The planner is a bin packer over alias classes — maximizing reuse without violating liveness. Correctness is enforced by the numerical gate suite (Invariant B).

The trace viewer should highlight **alias groups** so we can visually see memory reuse and catch accidental overlap bugs.

This integrates tightly with StepProgram — "tensor identity" becomes "offset in slab," which makes replay trivial because buffer handles don't change.

**2d. Kernel fusion — bandwidth fusions first**

Focus on fusions that eliminate whole read/write passes, not "fancy mega-block fusions":

* **(residual + layernorm)** — top-tier, eliminates a full memory round-trip
* **(bias + activation + dropout)** chains
* **AdamW fused per param shard** — huge win, one kernel per parameter group instead of N

Save "mega transformer block" fusion for after we have replay + memory plan + stable parity harness. Otherwise debugging fused kernels is brutal.

Each fusion reduces dispatch count and memory bandwidth. Target: reduce total dispatches per step by 60%+.

**2e. Persistent pipeline cache + specialization strategy**

The TS → SPIR-V compiler moat becomes lethal with:

* Stable kernel IR hash → pipeline cache key (so Vulkan doesn't recompile every launch)
* Per-GPU tuning profiles stored and reused across runs
* Specialization constants (tile sizes, unroll factors, dtype) instead of generating N shader variants

**Kernel IR contract (the difference between "cool codegen" and "industrial compiler"):**

The SPIR-V codegen generates kernels from TypeScript. That's the moat — but it becomes a liability if we can't stably hash kernels across refactors. Define a minimal IR we can serialize, hash, diff, and replay:

* Op graph (what the kernel computes)
* Types/shapes (input/output tensor metadata)
* Memory access pattern metadata (reads, writes, scratch)
* Specialization constants (tile sizes, unroll factors, dtype)

Then: `pipeline_cache_key = hash(IR + specializations + gpuProfile + codegenVersion)`

This makes pipeline caching correct, auto-tuning reproducible, and kernel regressions detectable.

This is how auto-tuning becomes a product, not a one-off benchmark.

**2f. Async data pipeline**

Double-buffered data loading: while the GPU processes batch N, the CPU prepares batch N+1 in a staging buffer. Transfer overlaps with compute via timeline semaphores. The GPU should never wait for data.

Execution model:

1. `mmap()` shard file (from pre-tokenized binary format)
2. Choose shard + offset (deterministic sampler)
3. Window into token IDs (`u16`/`u32`)
4. Pack into contiguous batch buffer if needed
5. Async H→D copy into a **pinned staging ring**
6. GPU reads from staging ring into model input buffers

**Async staging ring (design detail):**

* One pinned ring per GPU
* Fixed number of slots (4–8)
* Each slot: `(hostPinnedPtr, deviceStagingPtr, tokenCount)`
* Timeline semaphore value per slot for synchronization

This makes overlap measurable and debuggable — the profiler can report exactly how long the GPU waited for data per step.

### Phase 3: Multi-GPU (the unlock)

This is where Alpha goes from "impressive hobby project" to "serious training framework."

**Critical decision: host-driven collectives first (DP v0)**

Two approaches to multi-GPU communication:

*Option 1 — Host-driven collectives (ship fast):*
GPU writes grad chunk → shared/pinned host memory (or BAR if available) → CPU does ring copy/reduce orchestration with multiple threads → GPU reads reduced chunk back. Not peak performance, but correct scaling quickly on 2–8 GPUs. Lets us validate the entire distributed stack.

*Option 2 — GPU-driven collectives (the endgame):*
External memory import/export between Vulkan devices → GPU kernels perform reduce-scatter + all-gather directly → timeline semaphore synchronization across devices. Harder, will stall us if we try it first.

**Ship Option 1 as DP v0. Iterate to Option 2 once training is stable and profiled.**

**3a. Data parallelism v0 — host-driven**
- Each GPU gets the full model, different data shard
- After backward: host-driven ring all-reduce of gradients
- Start with 2 GPUs on one node, scale to 8
- Build **reduce-scatter + all-gather** early — ring all-reduce is fine but RS/AG is the gateway to overlap and later ZeRO-style paths
- No NCCL. We build the ring, the reduction, the synchronization

**3b. Custom collectives package**
```
packages/comm/
  src/topology.ts    — detect PCIe/NVLink layout
  src/transport.ts   — shared memory + pinned host memory abstraction
  src/ring.ts        — ring reduce-scatter + all-gather
  src/tree.ts        — tree all-reduce (lower latency for small messages)
  src/collectives.ts — allReduce, allGather, reduceScatter, broadcast, barrier
```

All custom. Test against CPU reference. Profile bandwidth vs theoretical peak.

**3c. Distributed correctness protocol**

Host-driven DP v0 will silently diverge without explicit checks. Three cheap validations that prevent "it trains but it's worse and we don't know why":

* **Step checksum:** after all-reduce, compute a small hash of a few gradient slices on each rank — must match
* **Loss agreement:** same global batch + same seed must produce matching loss within epsilon across ranks
* **Periodic single-rank replay:** run 1 rank with all grads, compare to DP result on a tiny model

These are cheap per-step and will catch desync before it costs real compute.

**Rank-local determinism contract (pin this down before writing any distributed code):**

* RNG streams: separate, deterministic stream per rank per step
* Dropout seeding: derived from `(global_seed, rank, step, layer)` — must be reproducible
* Data sharding: rank N gets shard N of the global batch, deterministic assignment

If this isn't pinned down, distributed debugging becomes hell.

**3d. Communication overlap**
Overlap the all-reduce of layer N's gradients with the backward pass of layer N+1. This hides communication latency behind compute. Requires careful scheduling but is standard practice and we have full control of the autograd engine.

**3e. GPU-driven collectives (v1)**
Once DP v0 is stable and profiled, upgrade to Vulkan external memory sharing for direct GPU-to-GPU transfers. Timeline semaphore synchronization across devices. This is the real performance path.

**3f. Tensor parallelism**
For models too large for one GPU's memory:
- Shard attention heads across GPUs (each GPU does N/P heads)
- Column-parallel first FFN linear, row-parallel second
- All-reduce after each TP communication point
- Custom sharded SPIR-V kernels

### Phase 4: Matrix Acceleration via Cooperative Matrix

We will never use CUDA. Matrix acceleration hardware (Tensor Cores on NVIDIA H100/L4/A100, Matrix Cores on AMD RDNA/CDNA) is accessible from Vulkan via `VK_KHR_cooperative_matrix` — the same SPIR-V codegen pipeline we already have, extended with cooperative matrix instructions. One backend, one stack, no vendor SDK dependency.

The Helios backend interface must already be compatible with StepProgram + memory planner by this point:

* `allocSlab(size, alignment)`
* `compileKernel(irHash, specializations)`
* `record(program, ops...)`
* `replay(program, patches)`
* `timestampQueries`
* `events/semaphores`

Cooperative matrix support is then "new kernel variants that consume the same StepProgram," not a fork of the world.

**Gating criteria — cooperative matrix kernels are opt-in per GPU profile:**

* Detect extension support at runtime (`VK_KHR_cooperative_matrix` + required features)
* Benchmark cooperative matmul vs SIMT matmul for the model's actual GEMM shapes
* Only enable if speedup exceeds threshold and Invariant B parity passes
* Fall back to SIMT kernels on any driver issue or correctness failure

We don't assume cooperative matrix is always faster. We measure, gate, and ship only what's proven.

**Shape bucketing:** matrix hardware shines on certain shapes; for weird shapes it can lose to SIMT. Add shape buckets so matmuls choose kernel variant by `(M,N,K)` class — small-batch attention projections vs large FFN GEMMs get different kernels.

**Mixed accumulation policy:** make accumulation dtype explicit per kernel. BF16/FP16 input with FP32 accumulate is the default. Validate that the accumulation path matches CPU reference within the per-op tolerance budget (Invariant B).

**4a. Cooperative matrix matmul kernels**
`VK_KHR_cooperative_matrix` exposes hardware matrix multiply-accumulate (MMA) operations through SPIR-V's `OpCooperativeMatrixMulAddKHR`. This maps to Tensor Cores on NVIDIA and Matrix Cores on AMD. We generate these instructions from our existing TypeScript SPIR-V codegen — same pipeline, new instructions. Target: BF16/FP16 inputs with FP32 accumulation, tiled for each GPU's L2 and shared memory hierarchy.

**4b. Per-GPU kernel profiles**
Add GPU profiles for H100, A100, and future hardware alongside the existing L4 profile. Auto-tune tile sizes, workgroup dimensions, and cooperative matrix block sizes (16x16x16, 16x8x16, etc.) per GPU's SM count, shared memory capacity, and register file. The SPIR-V codegen already parameterizes these — we just need the right constants per target.

**4c. Fused transformer kernels via SPIR-V**
With cooperative matrix instructions available, fuse entire transformer sublayers into single SPIR-V dispatches: residual + layernorm + linear (via coop matmul) + activation. Each fusion eliminates global memory round-trips. The SPIR-V codegen approach makes this tractable — we're generating code, not hand-writing assembly.

**4d. FP8 training (when Vulkan exposes it)**
Hopper supports FP8 Tensor Core ops. Vulkan's cooperative matrix extension is being extended to cover 8-bit types. When drivers ship support, our codegen pipeline picks it up with minimal changes — new type annotations in the SPIR-V emitter, dynamic scaling logic in TypeScript, same kernel structure. Until then, BF16 cooperative matrix already delivers the majority of the Tensor Core throughput gain.

### Phase 5: Frontier Scale

This is the endgame described in scale.md. Pipeline parallelism, ZeRO-style optimizer sharding, sequence parallelism, MoE, distributed checkpointing, fault tolerance. Each of these is a significant engineering effort. They become relevant when we're training 70B+ parameter models across 64+ GPUs.

---

## Performance Scorecard

Pick canonical benchmarks. Set explicit targets. Measure every PR.

### Target Numbers

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| Dispatches per step | ~300+ | <120 | StepProgram + fusion |
| CPU overhead per step | ~40%+ | <5% | StepProgram replay |
| L4 MFU (300M model) | ~2-5% | 15-30% | Replay + fusion + memory plan |
| Matmul bandwidth utilization | unmeasured | >60% | Profiler + coop matmul tuning |
| Flash attention bandwidth | unmeasured | >50% | Profiler + tiling |
| 8x L4 scaling efficiency | N/A (no multi-GPU) | >80% | DP v0 + comm overlap |
| Step time breakdown | not instrumented | every ms accounted | Profiler + Vulkan timestamps |
| **GPU Busy %** | unmeasured | >90% | Vulkan timestamp queries: kernel_time / step_wall_time |

GPU Busy % is the single most honest metric. It's computed from Vulkan timestamp queries: total GPU time spent in kernels divided by total wall time per step. It exposes dispatch overhead, host stalls, data bubbles, and synchronization gaps immediately. MFU can be gamed or misunderstood — GPU Busy % cannot.

### Canonical Workload

Define one workload that never changes so regressions are obvious:

* **Model:** GPT, dim=768, heads=12, layers=12, seq=512, SwiGLU, BPE-64K
* **Batch:** 8, accumSteps=1
* **Backend:** helios, FP16

All performance claims, benchmark comparisons, and regression tests use this workload. If it gets faster, Alpha got faster. If it gets slower, something broke.

Make the canonical workload executable in one command:

* `alpha bench canonical --json` — run canonical workload, output structured metrics
* `alpha trace canonical --chrome` — run canonical workload, output Chrome trace

**Required outputs per run (the golden trace format):**

* Wall step time (ms)
* GPU kernel time (Vulkan timestamp sum)
* GPU Busy %
* Dispatch count
* Top 10 kernels by time
* Top 10 barriers / waits by time
* H→D and D→H bytes transferred
* MFU

If we produce this on every PR, performance becomes a regression-tested feature.

### Canonical Benchmarks

* **GPT block forward/backward** at fixed shapes (dim=768, heads=12, seq=512) — time + bandwidth
* **End-to-end step time breakdown** — forward, backward, optimizer, data, host↔device, idle
* **Dispatch count per step** — total and per-phase
* **Achieved bandwidth %** for key kernels (matmul, attention, layernorm)
* **Achieved TFLOP/s %** for matmul and attention
* **MFU + dispatch utilization** (CPU overhead ratio)

---

## Risk Register

These are predictable. Plan for them.

| Risk | Impact | Mitigation |
|------|--------|------------|
| StepProgram replay complexity explodes because descriptor/pipeline layouts aren't stable | Replay becomes fragile or slower than expected | Design patch table to avoid descriptor set rebuilds from the start. Buffer handles don't change (slab model). Push constants for scalars. |
| Kernel fusion breaks gradients silently | Wrong training results, hard to debug | Invariant B — every fused kernel must pass the numerical parity suite before merge. No exceptions. |
| Data pipeline becomes bottleneck once GPU is fast | GPU idle waiting for data even with fast kernels | Pre-tokenized binary format + double-buffered staging from Phase 1. Don't defer this. |
| Distributed correctness (gradient desync across ranks) | Silent divergence, wasted compute | Determinism tooling: same seed → same loss across ranks. Periodic gradient checksum validation. |
| Vulkan `VK_KHR_cooperative_matrix` driver bugs or missing features | Can't access Tensor Cores as planned | L4/A100 training works fine without coop matrix. Phase 4 is an acceleration, not a blocker. Test on latest NVIDIA Vulkan beta drivers early. |
| Memory planner fragmentation under dynamic shapes | VRAM waste, OOM on edge cases | Training shapes are static. Enforce fixed shapes at StepProgram compile time. Dynamic shapes go through a slow path. |

---

## Architecture Decisions

### TypeScript stays as the orchestration layer

TypeScript is the right choice for:
- Autograd (tape operations, graph construction)
- Training loop orchestration
- Configuration, CLI, fleet management
- Web dashboard, TUI, inference server
- SPIR-V code generation (this is Alpha's secret weapon)

TypeScript is the wrong choice for:
- Hot-path kernel dispatch (already in C via native addon)
- GPU kernels (already in SPIR-V/C)
- Collective operations inner loops (should be C)
- Memory-mapped data loading (should be C)

**Rule: TypeScript for control flow, C for data flow.** The boundary is the N-API native addon. This is already the pattern with Helios and it works.

### SPIR-V codegen is a competitive advantage, invest in it

No other framework generates GPU kernels from a high-level language targeting Vulkan SPIR-V. This approach gives us:
- **Portability** — runs on any Vulkan GPU (NVIDIA, AMD, Intel, Apple via MoltenVK)
- **Metaprogramming** — kernel specialization at compile time (tile sizes, unroll factors, data types) without template madness
- **Fusion** — generating fused kernels is "just" generating more SPIR-V instructions in the same function

Double down on this. Build a proper kernel compiler pipeline:
```
TypeScript kernel DSL → IR (shapes, types, memory) → SPIR-V binary → cached pipeline
```

This is the moat. PyTorch has Triton (Python → PTX), locked to NVIDIA. We have TypeScript → SPIR-V, targeting every GPU on earth. And because SPIR-V is an open standard with cooperative matrix extensions, we get Tensor Core access without ever touching CUDA. Make ours the best GPU kernel compiler in any language.

### One backend: Vulkan. No CUDA. Ever.

Every other framework in this space ends up locked to NVIDIA's proprietary toolchain. We don't. Vulkan is an open standard that runs on every GPU vendor's hardware — NVIDIA, AMD, Intel, Apple (via MoltenVK). And crucially, Vulkan's `VK_KHR_cooperative_matrix` extension gives us access to the same Tensor Core hardware that CUDA users access through WMMA/MMA intrinsics.

This is not a compromise. It's a strategic advantage:
- **One codebase** — no second backend to maintain, no divergent kernel implementations, no "works on CUDA but not Vulkan" bugs
- **One codegen pipeline** — TypeScript → SPIR-V for everything, from elementwise ops to Tensor Core matmuls
- **Vendor independence** — when AMD's MI300X or Intel's Gaudi offer better price/performance, we run there day one. CUDA frameworks can't
- **Simpler stack** — no CUDA toolkit install, no nvcc, no PTX. Just a Vulkan driver, which ships with every GPU

The bet: Vulkan's compute capabilities will continue to converge with CUDA's. `VK_KHR_cooperative_matrix` already covers the critical matrix acceleration path. As the Vulkan ecosystem matures (FP8 types, async copy, hardware ray-tracing repurposed for scatter/gather), the gap narrows further. We ride that wave instead of building on a proprietary island.

If cooperative matrix support stalls on some vendors, Alpha still remains competitive through StepProgram replay, memory planning, and kernel fusion — matrix acceleration is an upside, not a dependency. The performance wins from Phases 1–3 are real regardless of driver politics.

---

## What Makes Alpha Win

Alpha won't win by being a better PyTorch. PyTorch has 1000+ engineers and a decade of momentum. Alpha wins by being something PyTorch cannot be.

### 1. Total transparency

Every line of code is readable, understandable, modifiable. There are no black boxes. A researcher can read the matmul kernel, understand the tiling, change the accumulation precision, and see the result immediately. PyTorch buries this under 15 layers of abstraction and vendor libraries.

**Action:** Write exceptional documentation for every kernel. Not just what it does, but why it's shaped that way, what the performance characteristics are, and how to modify it.

### 2. TypeScript as the interface language

Python's dominance in ML is an accident of history, not a technical necessity. TypeScript offers:
- Type safety that catches shape mismatches at compile time
- First-class async/await for overlapping compute and I/O
- Native web integration (dashboard, API, inference server — all the same language)
- npm ecosystem for everything that isn't ML (networking, CLI, UI)
- 10M+ developers who already know it

**Action:** Build the best TypeScript ML developer experience. Types for tensor shapes. IntelliSense for model configuration. Zero-config training that works from `npx`.

### 3. Vertical integration

Alpha is a training framework AND an inference server AND a web dashboard AND a fleet manager AND a data pipeline. In PyTorch-land, this requires PyTorch + vLLM + Weights & Biases + Slurm + a custom data pipeline. Alpha is one codebase.

**Action:** Lean into this. The dashboard should show GPU kernel timelines. The CLI should one-command deploy and train. The inference server should hot-load checkpoints from active training runs. Make the seams invisible.

### 4. The SPIR-V codegen approach

This is genuinely novel. No other project generates Vulkan compute kernels from TypeScript. If the kernel compiler matures, it could become a general-purpose GPU programming tool that others build on.

**Action:** Extract the SPIR-V codegen into a standalone package that others can use independently. This builds community around Alpha's core technology without requiring adoption of the full framework.

### 5. No vendor lock-in

PyTorch is NVIDIA-first. JAX is Google-first. MLX is Apple-only. Alpha runs on Vulkan — any GPU from any vendor. We access Tensor Cores through `VK_KHR_cooperative_matrix`, not through CUDA. When the best price/performance GPU isn't NVIDIA, we don't care. We're already there.

---

## Developer Ergonomics (the community moat)

Performance makes Alpha credible. Ergonomics makes it adopted.

### Kernel docs + perf notes — autogenerated per kernel

Every compiled kernel should emit metadata: op name, shapes, tile config, specialization constants, estimated bandwidth, estimated FLOP/s. Render this into browsable docs. When someone asks "why is my matmul slow?", the answer is one click away.

### `alpha bench kernel <name> --shape <M,N,K>`

A single command that benchmarks any kernel at any shape and produces a report: throughput, bandwidth utilization, dispatch overhead, comparison to theoretical peak. This is how users (and we) find performance regressions.

### `alpha trace step`

One command that runs a single training step, captures a Chrome trace with Vulkan timestamps, and opens it annotated with kernel names, tensor shapes, memory traffic, and phase boundaries. Every millisecond explained.

---

## What to Build Next (Concrete 90-Day Plan)

### Month 1: Foundation

| Week | Deliverable |
|------|-------------|
| 1 | Fix smoke test hang (runtime discipline: GPU scheduler isolation). Pre-tokenized binary format (u16/u32, shards, mmap-ready) |
| 2 | Numerical parity test suite (Invariant B). CPU vs GPU, forward + backward, finite-diff gradients, determinism checks |
| 3 | Per-step instrumentation (timing breakdown, MFU calculation, dispatch count) |
| 4 | Vulkan timestamp queries, Chrome trace export, `alpha trace step` command |

### Month 2: Performance (StepProgram-first — minimizes rework)

| Week | Deliverable |
|------|-------------|
| 5 | StepProgram trace + compile with **stable slab IDs** (even if offsets are naive/sequential). Descriptor stability via mega descriptor set + indirection buffer. Patch table schema defined |
| 6 | StepProgram replay: secondary cmd buffers per transformer block, barrier coalescing pass, patch bindings + push constants per step. Measure dispatch overhead reduction |
| 7 | Full memory planner: alias class analysis, liveness, bin packing over alias classes, tensor lifetime → (slab, offset, size). Integrate with StepProgram so tensor identity = stable slab offset. Trace viewer highlights alias groups |
| 8 | Bandwidth fusions: residual+layernorm, fused AdamW per param group. Pipeline cache with kernel IR hash keys. Fusion debugging is now tractable because tensor identity is stable offsets. Measure dispatches/step reduction |

### Month 3: Distribution

| Week | Deliverable |
|------|-------------|
| 9 | Topology detection, shared memory transport, ring reduce-scatter + all-gather prototype |
| 10 | DP v0: 2-GPU host-driven data parallelism. Gradient all-reduce, synchronized optimizer step. Correctness validated against single-GPU |
| 11 | Async data pipeline: double-buffered loading from pre-tokenized binary, overlap with compute |
| 12 | 8-GPU scaling test, communication overlap with backward, scaling efficiency measurement |

### Success Metric

At the end of 90 days: training a 300M parameter model on 8x L4 GPUs with >80% scaling efficiency, MFU measured and improving, CPU overhead <5% of step time via StepProgram replay, and a profiling dashboard that explains every millisecond of every training step.

---

## Killer Demos (the pitch made undeniable)

### Demo A: "Click a loss spike → see the exact kernel + tensor stats"

The dashboard shows a loss curve. Click any point. It opens the Chrome trace for that step + per-op stats. If the loss spiked, it shows the first divergent op vs CPU reference. No other framework does this. It demonstrates total vertical integration — training, profiling, debugging, visualization, all in one stack.

### Demo B: "Kernel compiler as a product"

`alpha kernel playground` — write a kernel in the TypeScript DSL, compile to SPIR-V, benchmark on your GPU, visualize memory reads/writes. This builds community around the SPIR-V codegen moat without requiring adoption of the full training stack. It's the gateway drug.

---

## Performance as Culture

GPU Busy % on the canonical workload is the North Star metric. Make it a hard gate:

* **PR cannot reduce GPU Busy %** on canonical workload unless it's a correctness fix
* **PR cannot increase dispatches/step** unless it's a feature with documented justification
* **Every optimization PR** includes before/after `alpha bench canonical --json` output

That turns performance from "thing we hope for" into "thing we enforce."

---

## Long-Term Vision

Alpha is not trying to replace PyTorch for the existing ML research community. It's building the future of ML infrastructure for the TypeScript/JavaScript ecosystem — the largest developer community in the world.

The endgame:

1. **`npx alpha train`** — one command to train a model from a dataset, on any GPU, with live metrics in a browser
2. **`npx alpha serve`** — one command to serve inference with an OpenAI-compatible API
3. **`npx alpha scale`** — one command to distribute training across a GPU cluster
4. **Every layer inspectable** — click any point on the loss curve and see the exact kernel execution, memory layout, and gradient flow at that step
5. **Every kernel ours** — when something is slow, we fix it in our code, not in a vendor's closed-source library
6. **Zero NVIDIA SDK dependency** — Tensor Cores via Vulkan cooperative matrix, not CUDA. Run on any GPU, any vendor, any cloud

The world doesn't need another Python ML framework. It needs a TypeScript ML framework that's fast enough to be taken seriously, built from scratch on open standards, locked to no one. That's Alpha.
