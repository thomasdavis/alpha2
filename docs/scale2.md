Perfect. Then the rewrite should **fully embrace that constraint** and stop sounding like it might lean on industry shortcuts.

Below is a **clean rewritten version** of your roadmap with the premise made explicit everywhere:

---

# Alpha Scaling Roadmap (All Custom, From Scratch)

## Mission

Train state-of-the-art models on **100+ H100s** at **90%+ utilization** using a **fully custom training stack**.

**Constraint (non-negotiable):**

* No NCCL
* No DeepSpeed
* No Megatron
* No PyTorch / JAX / TensorFlow
* No cuBLAS / cuDNN for core training ops
* No MPI
* No external ML frameworks

We write **everything**:

* kernels
* autograd
* optimizer logic
* graph execution
* communication collectives
* transport protocols
* checkpointing
* profiling
* scheduling

Every kernel is ours. Every byte over the wire is our protocol.

---

## Current State (Feb 2025)

* Single-GPU training working (L4 23GB / A100 80GB)
* ~10K tok/s on L4 with 6.8M param model
* Custom Vulkan/SPIR-V compute backend (**Helios**)
* Custom autograd, tokenizer, training loop — all TypeScript
* Peak VRAM utilization: ~28% on L4 (activation-memory constrained, not compute-bound)
* Current bottleneck: **Vulkan dispatch / launch overhead (~1ms per dispatch)**, not raw arithmetic throughput

**Interpretation:**
We are not compute-limited yet. We are runtime-overhead limited.
The fastest path forward is **instrumentation + launch amortization + memory efficiency** before distributed scale.

---

## Success Criteria

We define “90% utilization” explicitly so it is measurable.

### Utilization Targets (separate, explicit)

1. **Kernel-active GPU time** (single GPU) > 90%
2. **Single-GPU MFU/HFU** reaches competitive levels for our workload size
3. **Cluster end-to-end utilization** (including comm, checkpointing, data stalls, failures) trends toward 90% on large runs

We will track these independently. “Utilization” is not a vague dashboard number.

---

## Phase 0 — Instrumentation + Performance Model (Next)

Before optimizing anything, we measure everything.

### 0.1 Per-Step Timing Breakdown (trainer-level)

Every training step must be decomposed into:

| Component       | What to measure                                        |
| --------------- | ------------------------------------------------------ |
| **forward**     | First op start → loss value ready                      |
| **backward**    | `loss.backward()` start → gradients complete           |
| **optimizer**   | Parameter update time (AdamW/custom optimizer)         |
| **grad_clip**   | Norm compute + clipping                                |
| **data_load**   | Fetch next batch (disk/memory/cache)                   |
| **host→device** | Upload batch tensors to GPU                            |
| **device→host** | Read back loss/metrics                                 |
| **comm**        | Gradient synchronization / parameter exchange (future) |
| **checkpoint**  | Serialize + write checkpoint                           |
| **sample_gen**  | Inference sample generation during training            |
| **idle/bubble** | Dead time between phases                               |

### 0.2 Per-Op GPU Profiling (backend-level)

For every Vulkan dispatch:

* Host enqueue / command recording time
* Device execution time (GPU timestamps)
* Sync wait time (fence/semaphore blocking)
* Dispatch count per step
* Estimated bytes moved
* Arithmetic intensity estimate
* Bandwidth achieved vs theoretical peak

### 0.3 Metrics to Track

| Metric                           | Definition                                              | Target                                |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------- |
| **Achieved FLOPS**               | `model_flops_per_step / step_time`                      | maximize                              |
| **HFU**                          | `achieved_flops / hardware_peak_flops`                  | >40% eventually                       |
| **MFU**                          | model-useful FLOPS / hardware peak                      | >50% good, >80% excellent (long-term) |
| **Tokens/sec/GPU**               | `(global_batch_tokens_per_step / num_gpus) / step_time` | maximize                              |
| **Memory BW utilization**        | `bytes_moved / (peak_bw × time)`                        | >60% where BW-bound                   |
| **Kernel launch overhead share** | `(dispatch_overhead_time / step_time)`                  | minimize                              |
| **Pipeline bubble ratio**        | `idle_time / total_step_time`                           | <10%                                  |
| **Comm overhead share**          | `comm_time / step_time`                                 | <15% (scaled runs)                    |

### 0.4 FLOPS Model (documented, versioned)

We maintain a versioned FLOPS accounting model per architecture (transformer variants), including:

* attention FLOPS
* FFN FLOPS
* embedding/logits FLOPS
* backward FLOPS
* recompute FLOPS (checkpointing)
* optimizer FLOPS

All performance claims must state the FLOPS model version used.

### 0.5 Implementation Deliverables

1. **Trainer instrumentation**

   * Phase timers using `performance.now()` (host-side)
   * Step metrics emitted every N steps
   * Structured logging + persistent storage

2. **GPU timeline tracing**

   * Vulkan timestamp queries (`vkCmdWriteTimestamp`)
   * Export trace as Chrome trace / Perfetto JSON
   * Visualize host enqueue, device execution, sync waits, bubbles

3. **Op-level profiler**

   * Each backend op logs shape, dtype, dispatch count, time, bytes moved
   * Aggregation by op-type and by step

4. **Storage schema**

   * `step_timing`
   * `op_profile`
   * `gpu_timeline`
   * `run_metadata` (commit hash, backend version, kernel hashes, tokenizer hash, dataset shard)

5. **Dashboard (`/profiling`)**

   * Step waterfall chart
   * MFU/HFU over time
   * Top-N slowest ops
   * Dispatch count trend
   * Memory bandwidth utilization
   * Timeline viewer

### 0.6 Exit Criteria (must pass before optimization work)

* We can explain where every millisecond in a step went
* We can identify top dispatch-overhead offenders
* We can reproduce profiles across runs
* MFU/HFU calculation is consistent and documented

---

## Phase 1 — Single-GPU Efficiency (Runtime + Memory + Kernel Work)

Goal: eliminate runtime overhead, increase arithmetic intensity, and raise per-GPU utilization before multi-GPU complexity.

### 1.1 Runtime Execution Engine (launch amortization first)

Current bottleneck is dispatch overhead, so runtime execution changes come first.

#### 1.1.1 Static Graph Capture / Replay

* Capture stable training-step execution graph
* Reuse execution plans across steps
* Minimize per-step command construction work

#### 1.1.2 Persistent Command Buffers

* Pre-record command buffers where possible
* Replay with updated buffer bindings / offsets
* Reduce CPU overhead per dispatch

#### 1.1.3 Timeline Semaphores (replace fence-heavy sync)

* Move from fence-blocking to timeline semaphore progression
* Overlap compute and transfers
* Reduce host stalls

#### 1.1.4 Double-buffered / staged transfers

* Upload next batch while current batch computes
* Reuse staging buffers
* Avoid unnecessary synchronization points

### 1.2 Memory Efficiency (unlock larger batches)

#### 1.2.1 Activation Checkpointing (custom)

Trade compute for memory by recomputing activations during backward.

* Define checkpoint boundaries in custom tape/autograd graph (e.g., per transformer block)
* Re-run forward subgraphs during backward to reconstruct intermediates
* Increase effective batch size / sequence length
* Accept extra recompute cost to improve utilization and throughput

Expected effect:

* Higher memory headroom
* Larger batches
* Better launch amortization (more useful work per step)

#### 1.2.2 Gradient Accumulation

* Support micro-batch accumulation on a single GPU
* Keep optimizer semantics deterministic
* Increase effective global batch before distributed training

### 1.3 Kernel Fusion (custom compiler/runtime, fully ours)

We fuse high-frequency op sequences into single kernels to reduce dispatch count and memory traffic.

#### Initial fusion targets (profiler-driven)

* `add + layernorm`
* `layernorm + linear (+ bias)`
* `linear + gelu`
* attention path fragments (`scale + mask + softmax`, later more)
* optimizer update chains (multi-parameter fused updates)

#### Fusion approach

* Build a custom op-graph fusion pass (no external compiler frameworks)
* Detect fusible patterns
* Generate fused SPIR-V kernels
* Cache compiled variants by shape/dtype/layout signature

### 1.4 Memory Planner / Buffer Reuse

* Lifetime analysis for tensors
* Reuse temporary buffers across ops
* Reduce allocation churn
* Minimize peak VRAM usage
* Support graph replay with stable buffer plans

### 1.5 Phase 1 Exit Criteria

* Dispatch overhead share reduced substantially (quantified in profiler)
* Larger batch / seq length feasible through checkpointing + reuse
* Step traces show overlap (transfer + compute)
* Single-GPU MFU/HFU materially improved on target benchmark configs
* Numerical parity passes for fused vs unfused execution

---

## Phase C — Correctness & Reproducibility Harness (parallel, mandatory)

This phase runs in parallel with all optimization/scaling work.

Scaling broken numerics is wasted effort.
Every feature must pass correctness gates before being used in production runs.

### 2.1 Numerical Correctness

* Forward parity tests (CPU reference vs GPU backend)
* Backward parity tests
* Finite-difference gradient checks (small tensors)
* Mixed-precision tolerance tests

### 2.2 Determinism / Reproducibility

* Seeded determinism on single GPU
* Deterministic replay mode (debug builds)
* Reproducible checkpoint resume
* Stable kernel selection / fusion hashing per run

### 2.3 Distributed Correctness (as comm features arrive)

* Collective sum/avg correctness checks
* Cross-rank parameter parity after optimizer step
* Shard/gather correctness for TP/ZeRO-style sharding

### 2.4 Failure Testing

* Timeout injection
* Rank hang simulation
* Partial checkpoint corruption detection
* Resume validation from previous checkpoint

### 2.5 Exit Criteria

* A feature cannot be considered “done” unless parity + reproducibility tests pass
* Profiling improvements that break numerical correctness are rejected

---

## Phase 2 — Multi-GPU Single Node (2–8 GPUs, All Custom Collectives)

Goal: build our own collectives and achieve efficient single-node scaling without NCCL.

### 2.1 Custom Collective API (ours only)

```typescript
// packages/comm/src/collectives.ts
interface CollectiveOps {
  allReduce(tensor: TensorData, op: 'sum' | 'avg'): Promise<TensorData>;
  allGather(tensor: TensorData): Promise<TensorData>;
  reduceScatter(tensor: TensorData, op: 'sum'): Promise<TensorData>;
  broadcast(tensor: TensorData, root: number): Promise<TensorData>;
  barrier(): Promise<void>;
}
```

No NCCL. No MPI. No vendor collective libraries.
All protocols, scheduling, and synchronization are custom.

### 2.2 Collective Implementation Layers (single-node)

1. **Shared-memory transport**

   * POSIX shared memory / OS primitives for host-coordinated transfers
   * Deterministic correctness baseline

2. **GPU buffer sharing / P2P path**

   * Vulkan external memory / device-sharing mechanisms
   * Direct GPU-to-GPU transfer path where supported

3. **Topology detection**

   * Detect PCIe/NVLink/NVSwitch layout
   * Build topology map used by collective scheduler

4. **Ring all-reduce (first production collective)**

   * Bandwidth-efficient baseline
   * Deterministic, instrumented, debuggable

5. **Tree variants (later)**

   * Latency-oriented options for smaller messages / different topologies

### 2.3 Data Parallelism (first distributed training mode)

* Full model replica on each GPU
* Different mini-batch shard per GPU
* Gradient all-reduce after backward
* Deterministic optimizer step on every rank
* Overlap all-reduce with backward where possible

### 2.4 Tensor Parallelism (after DP is stable)

Split large matrices across GPUs:

* Attention head sharding
* FFN linear sharding (column/row parallel)
* Custom sharded kernels + custom synchronization

TP will be introduced only after:

* DP correctness is stable
* DP scaling is profiled and understood
* collective stack is robust

### 2.5 Phase 2 Exit Criteria

* 2-GPU and 8-GPU DP training runs stable for long-duration tests
* No deadlocks / hangs in soak tests
* Scaling efficiency quantified and visible in profiler
* Comm overlap visible in traces
* TP training validated on smaller models before scaling up

---

## Phase 3 — Multi-Node (8–100+ GPUs, Custom Transport + Scheduling)

Goal: extend custom collectives and training runtime to cluster scale.

### 3.1 Custom Network Transport (ours only)

No MPI. No NCCL. No third-party distributed runtime.

We build:

* Custom message framing protocol
* Custom rank coordination
* Custom transport abstraction
* Custom error handling / retry semantics
* Custom topology-aware scheduling

#### Transport layers (staged)

1. **Baseline transport** (debuggable, portable path)
2. **High-performance transport** (RDMA / InfiniBand verbs)
3. **GPUDirect RDMA path** (when correctness + stability are proven)

### 3.2 Topology-Aware Placement

* Detect node topology (GPU↔GPU, GPU↔NIC, NUMA)
* Place DP/TP/PP groups accordingly
* Minimize cross-socket and cross-switch penalties
* Optimize communication paths based on measured bandwidth/latency

### 3.3 Pipeline Parallelism (PP)

* Assign layer ranges to ranks/stages
* 1F1B micro-batch schedule (custom scheduler)
* Overlap stage comm with compute
* Minimize pipeline bubbles
* Trace bubble ratio and stage idle time explicitly

### 3.4 Sharded Optimizer (ZeRO-style, custom implementation)

We implement memory sharding ourselves:

* **Stage 1**: shard optimizer states (`m`, `v`)
* **Stage 2**: shard gradients
* **Stage 3**: shard parameters

All gather/reduce schedules are custom and integrated into our runtime.

### 3.5 Gradient Compression (custom)

To reduce comm volume:

* FP16/BF16 gradient transport
* 1-bit quantization + error feedback
* Top-K sparsification + sparse reduction
* Low-rank gradient approximation (PowerSGD-style, custom)

Each technique must be evaluated for:

* convergence impact
* bandwidth reduction
* overlap compatibility
* implementation complexity

### 3.6 Distributed Checkpointing (sharded, async, custom format)

* Non-blocking checkpoint writes
* Per-rank shard save in parallel
* Custom checkpoint manifest
* Supports resharding when GPU count changes
* Resume-safe and corruption-detectable

### 3.7 Phase 3 Exit Criteria

* Stable multi-node training on real workloads
* Transport failures are detectable/recoverable
* Checkpoint/restart works across node count changes
* Bubble ratio and comm overhead are bounded and measured
* Cluster traces explain performance regressions

---

## Phase 4 — H100 Frontier Performance Path (Custom CUDA Backend + Hopper-Specific Work)

Goal: approach H100-class performance ceilings with a backend designed for Hopper/Tensor Cores.

**Reason:** Vulkan/portable kernels are valuable, but H100 peak utilization requires architecture-specific kernel work.
We will write a **second backend** (custom CUDA) while keeping the same high-level trainer/autograd/runtime abstractions.

### 4.1 Custom CUDA Backend (no cuBLAS/cuDNN for core ops)

* Custom CUDA kernel runtime
* Custom kernel codegen/toolchain integration
* Custom memory management + stream scheduling
* Custom profiling hooks integrated with `packages/profile`

### 4.2 Tensor Core Matmuls (custom)

* Implement matmul kernels targeting Tensor Cores
* Use low-level CUDA/PTX/Tensor Core instructions directly
* Tile/block scheduling tuned for Hopper
* Support training dtypes (BF16/FP16, later FP8)

### 4.3 Fused Transformer Kernels

* Fused attention kernels (Flash-style memory-efficient implementation, custom)
* Fused norm/residual kernels
* Fused MLP blocks
* Fused optimizer updates

### 4.4 FP8 Training Support (Hopper)

* FP8 kernel paths
* Dynamic scaling / metadata tracking
* Amax collection
* Numerics validation
* Fallback modes for debugging

### 4.5 Sequence Parallelism (long context)

For long sequences (8K–128K):

* Shard sequence dimension across GPUs
* Ring communication for KV/cache or partial activations
* Custom scheduling and communication overlap

### 4.6 Mixture of Experts (MoE)

* Custom router
* Custom all-to-all token dispatch
* Expert parallel placement
* Load balancing loss
* Fault/imbalance diagnostics

### 4.7 Phase 4 Exit Criteria

* Core H100 kernels benchmarked and profiled against theoretical ceilings
* End-to-end training on Hopper is no longer launch-bound
* Frontier features (FP8, sequence parallel, MoE) are correctness-validated and production-stable

---

## Data Pipeline (First-Class Subsystem, All Custom)

Data must not become the hidden bottleneck as compute improves.

### Requirements

* Custom dataset shard format (streamable, resumable)
* Pre-tokenized binary storage
* Deterministic shuffling across ranks
* Resume-safe sampler state
* Async prefetch and decode pipeline
* Packing/bucketing for sequence efficiency
* Checksums + provenance metadata
* Per-rank throughput and stall metrics in profiler

### Exit Criteria

* Data loader never starves GPUs on target workloads (verified via traces)
* Resume reproduces sample order deterministically (when configured)

---

## Reliability & Operations (Required for 100+ GPUs)

At cluster scale, faults are normal. Reliability is part of the training stack.

### Features

* Rank heartbeat and liveness detection
* Timeout and deadlock diagnostics
* Run manifests (code version, kernel hashes, config hashes)
* Checkpoint atomicity and integrity verification
* Rank failure detection + controlled abort/restart semantics
* Debug modes (deterministic, verbose tracing, comm checksums)

### Rule

A system that is fast but cannot survive long runs is not “scaled.”

---

## Scaling Math (Planning Model)

### 100× H100 SXM (80GB each) — reference planning envelope

| Resource                             | Total                                                    |
| ------------------------------------ | -------------------------------------------------------- |
| VRAM                                 | 8 TB                                                     |
| FP16 compute                         | ~98.9 PFLOPS                                             |
| FP8 compute                          | ~197.9 PFLOPS                                            |
| NVLink/NVSwitch intra-node bandwidth | topology dependent; profile actual                       |
| Inter-node fabric                    | depends on NIC/fabric config (e.g., 400Gb/s class links) |

We treat these as **planning ceilings**, not guaranteed achieved throughput.

---

## Model Size Planning (Order-of-Magnitude, depends on precision + strategy)

| GPUs | Approx model scale                  | Strategy                           |
| ---- | ----------------------------------- | ---------------------------------- |
| 1    | up to ~7B (context/batch dependent) | single GPU + checkpointing         |
| 8    | tens of billions                    | DP + TP                            |
| 64   | hundreds of billions                | DP + TP + PP + sharding            |
| 100+ | frontier-scale / MoE / sparse       | full stack (DP/TP/PP/sharding/MoE) |

These are estimates; actual limits depend on:

* precision
* activation recompute policy
* optimizer state sharding stage
* sequence length
* batch size
* fragmentation / memory planner quality

---

## Package Architecture (All Custom)

```txt
packages/helios      — Vulkan/SPIR-V backend runtime + kernels
packages/cuda        — CUDA backend runtime + kernels (custom, Hopper-focused)
packages/autograd    — custom tape, backward engine, checkpointing integration
packages/runtime     — graph execution, memory planner, capture/replay, scheduling primitives
packages/comm        — collectives, topology detection, transport abstraction, protocols
packages/schedule    — DP/TP/PP placement and distributed execution scheduling
packages/profile     — instrumentation, tracing, metrics, MFU/HFU accounting
packages/checkpoint  — sharded checkpoint format, manifests, resharding support
packages/data        — shard format, sampler, prefetch, deterministic loading
packages/validate    — numerical parity, determinism, distributed correctness tests
```

No external ML runtime dependencies.
Only system-level APIs and GPU driver interfaces where required.

---

## Order of Operations (Rewritten, All-Custom Reality)

1. **Phase 0** — Instrumentation + performance model + trace system
2. **Phase 1.1** — Runtime execution engine (capture/replay, semaphores, overlap)
3. **Phase 1.2** — Activation checkpointing + gradient accumulation + memory planner
4. **Phase 1.3** — Profiler-driven kernel fusion (custom)
5. **Phase C** — Correctness/reproducibility harness (continuous gate across all phases)
6. **Phase 2.1** — Custom collectives (2-GPU correctness + soak tests)
7. **Phase 2.3** — Data parallelism on 2–8 GPUs (single node)
8. **Phase 2.4** — Tensor parallelism (after DP stability/perf)
9. **Phase 3.1** — Custom multi-node transport + protocol
10. **Phase 3.3/3.4** — Pipeline parallelism + sharded optimizer + distributed checkpointing
11. **Phase 4.1–4.4** — Custom CUDA backend + Tensor Core kernels + fused transformer kernels + FP8
12. **Phase 4.5–4.6** — Sequence parallelism + MoE

---

## Engineering Rules (to preserve the mission)

1. **No hidden dependencies**

   * If a component performs ML compute/communication/runtime orchestration, it must be ours.

2. **Profile before optimizing**

   * No optimization lands without before/after traces and metric deltas.

3. **Correctness before scale**

   * A feature is incomplete until parity/determinism tests pass.

4. **Performance claims must be reproducible**

   * Every benchmark must log hardware, kernel hashes, config, and FLOPS model version.

5. **Distributed code must be debuggable**

   * Checksums, trace markers, rank-local logs, and deterministic debug mode are mandatory.

6. **All abstractions exist to support custom implementations**

   * Architecture should enable swapping implementations while preserving total ownership.

---

## End State

A fully custom training stack capable of:

* single-GPU efficiency
* custom collectives and transport
* multi-node distributed training
* Hopper-optimized kernels
* frontier-scale features (PP, sharding, sequence parallelism, MoE)

No borrowed core runtime.
No outsourced collectives.
No framework lock-in.

**Alpha trains on infrastructure we own end-to-end.**

---

If you want, I can also rewrite this next into a **hardcore execution PRD** with:

* milestone dates,
* acceptance tests per phase,
* risk register,
* and a “do-not-build-yet” list (to protect focus while staying fully custom).
