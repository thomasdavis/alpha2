# Research Report: Building an Operation Universe for Alpha

## Executive summary

The useful deliverable is not a flat list of every function anyone has ever named. That list would be both incomplete and actively harmful: arbitrary fusion, user-defined semirings, custom layouts, numerical policies, and schedule choices create an unbounded operation space.

The correct abstraction is a **generative operation ontology** with three parts:

1. **Canonical semantic operations** — distinct mathematical or systems meanings such as GEMM, selective scan, sparse sampled-dense multiplication, quiescence detection, or GPU VA mapping.
2. **Orthogonal variant dimensions** — dtype, layout, sparsity, algebra, precision, determinism, execution scope, epilogue, and schedule.
3. **Generic composition primitives** — Prometheus regions, loops, contractions, reductions, scans, memory transfers, barriers, and actor events that can express operations not yet named.

The generated scaffold contains 2,644 canonical stubs. That number is intentionally large enough to stimulate implementation search but small enough to remain searchable. It spans mathematical operations, compiler IR, `sm_86` instruction/macro families, synchronization, launch transport, memory management, and the NVIDIA RM boundary.

## 1. Why a canonical registry is better than thousands of handwritten APIs

A conventional API often mistakes implementation variants for mathematical operations. For example, all of these may compute the same contraction:

- SIMT FP32 GEMM;
- tensor-core BF16 GEMM with FP32 accumulation;
- INT4 weight-only GEMM with fused dequantization;
- split-K GEMM;
- Stream-K GEMM;
- persistent GEMM;
- grouped MoE GEMM;
- block-sparse GEMM;
- GEMM with fused bias, GELU, residual, and quantization.

Conversely, two operations with identical loops may have genuinely different algebraic meaning. Arithmetic matrix multiplication and min-plus matrix multiplication share an index pattern but solve different problems.

MLIR's Linalg and Vector dialects explicitly separate high-level structured operations from hardware-vector lowerings. Linalg models iteration and indexing semantics; Vector provides retargetable contractions, transfers, reductions, and scans before target-specific lowering. CUTLASS similarly decomposes GEMM into layouts, tiled movement, collective mainloops, and epilogues rather than exposing one monolithic implementation. These are strong precedents for Prometheus to model meaning separately from schedule.

Primary references:

- MLIR Linalg: https://mlir.llvm.org/docs/Dialects/Linalg/
- MLIR Vector: https://mlir.llvm.org/docs/Dialects/Vector/
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass

## 2. The three meanings of “operation”

### 2.1 Semantic operation

A semantic operation specifies mathematical behavior and observable side effects. Examples:

- `alpha.gemm.gemm-semiring`;
- `alpha.scan.selective-scan`;
- `alpha.memory_retrieval.typed-binding-supersede`;
- `alpha.autodiff.hessian-vector-product`.

It must define dtype promotion, shape rules, edge cases, determinism, gradients, and numerical tolerance.

### 2.2 Compiler operation

A compiler operation represents a reusable transformation or executable region. Examples:

- `prometheus.structured.generic-contraction`;
- `prometheus.vector_tile.transfer-read`;
- `prometheus.parallel.warp-reduce`;
- `prometheus.async_actor.mailbox-send`.

Several semantic operations may lower to the same compiler primitives, and one semantic operation may lower differently for different shapes.

### 2.3 Machine/runtime operation

A machine/runtime operation changes physical state or emits target instructions. Examples:

- `hephaestus.sass_tensor.hmma-bf16`;
- `chronos.timeline.signal-timeline-gpu`;
- `hermes.qmd.set-qmd-grid-dimensions`;
- `gaia.virtual_address.map-gpu-va`;
- `aether.rm_channel.allocate-channel`.

These operations are versioned against an exact architecture and driver compatibility profile. They must never be inferred from a high-level name alone.

## 3. Status classes

### Standard

The semantics are established by a specification, mature library, or well-known algorithm. “Standard” does not mean implemented or fast in Alpha.

### Research

The operation is mathematically coherent or supported by nearby work, but the exact Alpha construction requires an experiment. Examples include anytime bitplane GEMM, residue-corrected GEMM, event-driven mechanism execution, and optimizer-consumed products.

### Speculative

The operation is retained as a prompt for future work but should not enter production planning without a formal proposal. Examples include interval or dual-number variants where their benefit to LLM training is unproven.

Codex must treat these classes differently. A research stub is a request for an experiment, not an instruction to quietly ship an approximation.

## 4. Layer-by-layer scope

### Alpha

Alpha owns observable tensor, model, autograd, optimizer, memory, and distributed semantics. It should be broad because new model architectures are easiest to explore when the operation vocabulary already exists.

Key families include:

- creation, shape, indexing, elementwise and reduction operations;
- BLAS, decompositions, iterative solvers, structured matrices and tensor networks;
- sparse formats and generalized sparse computation;
- FFT, wavelet, convolution and signal-processing operations;
- attention, recurrence, state-space scans, routing and memory;
- quantization, sampling, losses, optimizers and differentiation;
- collective communication.

### Helios

Helios owns policy and selection rather than mathematical invention. It compiles, caches, fuses, plans memory, selects kernels, applies budgets, profiles, and explains decisions.

A critical design requirement is `explainKernelChoice`: Codex and human researchers must be able to inspect why a particular operation lowered to one implementation and not another.

### Prometheus

Prometheus must be expressive enough that most new operations are compositions, not new compiler hard-coding. The essential primitives are:

- typed tensors, memrefs, sparse and quantized values;
- structured indexing maps and contractions;
- vector/tile transfers and reductions;
- explicit memory spaces and asynchronous copies;
- loops, predicates, software pipelines and parallel mapping;
- matrix fragments and semiring combination/reduction;
- actor mailboxes, event credits, waves and state commits;
- cost and error models;
- transform passes.

### Hephaestus

Hephaestus is both an assembler and a target-specific macro-assembler. Its operation surface therefore includes:

- exact instruction families;
- control/scheduling fields;
- physical register classes and no-spill constraints;
- validated macro sequences such as online softmax, Welford reduction, packed ternary decode, and persistent work loops.

The macro layer is important: Prometheus should not reproduce delicate SASS scheduling sequences at every call site.

### Chronos

Chronos owns all time and completion semantics:

- timelines, fences and semaphores;
- dependency graphs and hazard classes;
- deterministic waves and epochs;
- task budgets, deadlines and priorities;
- event-credit quiescence;
- watchdogs, rollback and replay;
- correlated host/GPU traces.

A mathematical operation should never improvise its own synchronization protocol.

### Hermes

Hermes transports executable work:

- channels and channel groups;
- GPFIFO rings and pushbuffers;
- QMD construction;
- parameter/constant-bank binding;
- ordinary, indirect, batched, persistent and actor-grid launch;
- device-side work packets and fault telemetry.

### Gaia

Gaia owns memory as a virtualized hierarchy:

- physical allocations and GPU VA mappings;
- host/BAR staging;
- arenas and pools;
- residency, installation and migration;
- model/session/episodic images;
- integrity checks, poisoning and leak detection;
- placement, prefetch and eviction policies.

### Aether

Aether owns the entire raw transport boundary:

- device nodes and ioctl calls;
- RM clients and object graphs;
- device, subdevice, VA-space, memory and channel objects;
- UVM interactions;
- GSP compatibility and RPC evidence;
- capability manifests and crash bundles.

NVIDIA's open kernel-module source confirms that the module must still be paired with matching user-space components and GSP firmware. Alpha's from-scratch boundary is therefore the project-owned userspace stack above the unmodified kernel/GSP boundary, not absolute control of every GPU subsystem.

Primary reference: https://github.com/NVIDIA/open-gpu-kernel-modules

## 5. The matrix multiplication universe

### 5.1 Generalized contraction

The most useful mathematical form is:

```
C[o] = REDUCE_r COMBINE(A[indexA(o,r)], B[indexB(o,r)])
```

Ordinary GEMM chooses multiplication for `COMBINE` and addition for `REDUCE`. GraphBLAS standardizes generalized matrix/vector multiplication over semiring structures, enabling Boolean reachability, shortest paths, Viterbi-like dynamic programs, and other computations with the same broad contraction geometry.

Primary reference: https://graphblas.org/graphblas-api-cpp/

### 5.2 Independent axes

A GEMM implementation should be identified by a tuple rather than a name:

```
geometry × algebra × structure × dtype × accumulator × layout ×
sparsity × schedule × data movement × prologue × epilogue ×
determinism × stopping rule × consumer
```

Examples of independent choices:

- geometry: GEMV, GEMM, grouped, batched, broadcast, tensor contraction;
- algebra: arithmetic, Boolean, min-plus, max-plus, log-sum-exp-plus;
- structure: dense, triangular, low-rank, Toeplitz, butterfly, N:M sparse;
- schedule: split-K, Stream-K, persistent, grouped, countercurrent;
- consumer: activation, input adjoint, weight gradient, optimizer statistic, top-k decision.

### 5.3 Epilogue trees

cuBLASLt and CUTLASS show that epilogues are a first-class source of performance. Existing implementations fuse bias, activations, auxiliary outputs, gradient reductions, amax, scaling, and other elementwise work. Prometheus should represent an epilogue as a small expression tree rather than a fixed enum.

Primary references:

- https://docs.nvidia.com/cuda/cublas/
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md

### 5.4 Uncommon operations worth keeping as stubs

#### Semiring GEMM

Useful for graph algorithms, dynamic programming, routing, probabilistic state transitions, and structured memory operations.

#### Anytime bitplane GEMM

A product is refined from the most significant components. It may stop when the uncomputed remainder cannot alter the consumer's decision. The correctness contract is consumer-specific, not a global Frobenius tolerance.

#### Residue-corrected GEMM

A cheap product is supplemented with an exact correction in selected input/output subspaces. This is a candidate for low-bit training when important error is concentrated.

#### Deferred weight-gradient GEMM

Microbatch activation and adjoint factors are banked and combined into a larger weight-gradient product. It is exact before factor compression, but memory cost and loss of immediate scheduling flexibility must be measured.

#### Optimizer-consumed GEMM

The kernel returns sufficient statistics or a transformed update needed by the optimizer, avoiding a full intermediate gradient when possible.

#### Countercurrent GEMM

Forward and backward tile streams share stationary weights. It requires a scheduling simulator before implementation because dependency and on-chip-capacity constraints may eliminate the theoretical reuse.

#### Conservation-projected or checksum-verified GEMM

Approximate arithmetic preserves selected moments/projections or carries randomized checks. These operations need explicit failure and fallback contracts.

## 6. Operations beyond CUDA/Vulkan's usual surface

The catalog intentionally includes fields that general GPU APIs often leave to libraries or application code:

- generalized semiring contractions;
- structured matrix products and transforms;
- tensor-network contractions;
- recurrent and selective scans;
- explicit optimizer transforms;
- low-bit packing and error-feedback state;
- typed conversational memory operations;
- actor/event semantics;
- quiescence and event-credit accounting;
- model image installation and resident-weight refresh;
- causal or verified approximate stopping rules.

This does not mean every item deserves a kernel. It means Codex can reason about the possibility without inventing inconsistent names or bypassing the correct layer.

## 7. LLM-specific operation clusters

### Attention

The registry spans dense scaled-dot-product attention, FlashAttention-style IO-aware tiling, paged KV caches, grouped-query and multi-query attention, local/block-sparse patterns, recurrent/linear attention, and exact cache lifecycle operations.

FlashAttention is a central design lesson: fewer FLOPs are not sufficient; IO-aware tiling and avoiding intermediate HBM materialization can produce the real gain.

Primary reference: https://arxiv.org/abs/2205.14135

### State-space and recurrent models

Selective scan, affine recurrence scans, delta-rule updates, RWKV-style state evolution, xLSTM cells, and other recurrent primitives are canonical operations because they can lower to parallel scans during training and recurrent updates during inference.

Mamba explicitly introduced a hardware-aware parallel selective-scan algorithm after making state-space parameters input-dependent. This supports treating scan as a first-class compiler and kernel primitive rather than implementing it as an opaque model loop.

Primary reference: https://arxiv.org/abs/2312.00752

### Sparse experts

Routing, dispatch, grouped GEMM, capacity management, combination, replication, prefetch, and eviction are separate operations. A monolithic `moe()` function would prevent useful scheduling and observability.

### Quantization

Packing, calibration, scale computation, dequantization, outlier handling, stochastic rounding, residual feedback, and fused low-bit matrix operations require separate semantics. A dtype alone does not describe a quantized computation.

### Memory

Typed bindings, episodic raw-span retention, associative memory, KV pages, and conversation passports are different storage semantics. They must not be collapsed into one generic `memoryRead` function.

## 8. `sm_86` target reality

The RTX 3070 target is Ampere compute capability 8.6. The catalog keeps future operations for design continuity, but Hephaestus must reject unsupported native paths.

Important `sm_86`-relevant families include:

- FP16, BF16, TF32, INT8, INT4, binary matrix operations where supported;
- asynchronous global-to-shared staging and barrier coordination;
- warp shuffles, votes and reductions;
- shared-memory matrix loads;
- conventional thread/block/grid memory and atomic scopes.

Do not assume native FP8, Hopper TMA, WGMMA, thread-block clusters, or Blackwell block-scaled instructions. Those remain future/emulation targets. CUTLASS's architecture table and PTX target notes are useful references, but Hephaestus must validate actual `sm_86` encodings and behavior through independent execution tests.

Primary references:

- https://docs.nvidia.com/cuda/ampere-tuning-guide/
- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://github.com/NVIDIA/cutlass

## 9. Sparse and tensor operations

cuSPARSE's generic API covers formats, mixed data/index types, SpMV, SpMM, SpGEMM and related operations. cuTENSOR covers arbitrary-layout contractions, reductions, elementwise operations and permutations. cuSOLVER covers factorizations and linear solves. These libraries provide a lower bound on the canonical vocabulary Alpha should understand, even though Alpha will not depend on them in production.

Primary references:

- https://docs.nvidia.com/cuda/cusparse/
- https://docs.nvidia.com/cuda/cutensor/
- https://docs.nvidia.com/cuda/cusolver/

## 10. Parallel primitives

CUB identifies reusable primitives at warp, block and device scope: load/store, reduction, scan, sort and histogram. These belong below model semantics and above raw instructions. Prometheus should expose them, and Hephaestus should contain proven target-specific macro sequences.

Primary reference: https://docs.nvidia.com/cuda/cub/

## 11. FFT and transform operations

cuFFT's plan model illustrates why execution planning is itself an operation family: the same mathematical transform may require different factorizations, workspaces, layouts and multi-stage execution. cuFFTDx shows the value of embedding transforms inside a larger kernel to eliminate global-memory round trips.

Primary references:

- https://docs.nvidia.com/cuda/cufft/
- https://docs.nvidia.com/cuda/cufftdx/

## 12. Codex implementation workflow

For any chosen stub, Codex should produce these artifacts before code is considered complete:

1. semantic specification;
2. supported shape/dtype/layout region;
3. reference implementation;
4. gradient or side-effect specification;
5. Prometheus lowering;
6. schedule and memory plan;
7. generated SASS/resource manifest;
8. correctness tests, including edge cases;
9. deterministic and fault tests;
10. matched end-to-end benchmark;
11. fallback operation;
12. result entered into the experiment ledger.

A function name is only discoverability scaffolding.

## 13. Priority tiers

### Tier 0 — Bring-up and correctness

- scalar arithmetic and conversion;
- loads/stores, views and copies;
- reductions and scans;
- GEMV/GEMM reference paths;
- normalization and activation;
- timelines, QMD launch and memory mapping;
- tracing and fault bundles.

### Tier 1 — Small language-model path

- embedding, RMSNorm, linear projections, recurrent scans;
- attention or recurrent-core operations selected by the model;
- output softmax and sampling;
- FP16/BF16 and INT8/INT4 support;
- optimizer and autograd operations;
- static graph/program replay.

### Tier 2 — Serious performance

- tensor-core tiled GEMM;
- fused epilogues;
- FlashAttention-style kernels;
- grouped expert GEMM;
- selective scan;
- packed low-bit weights;
- static arenas and persistent programs;
- autotuning and cost models.

### Tier 3 — Research kernels

- residue corrections;
- adaptive/anytime precision;
- optimizer-consumed products;
- event-driven actor execution;
- structured/semiring matrix operations relevant to new architectures.

### Tier 4 — Future hardware or speculative mathematics

Retain stubs, but do not let them distract from the validated RTX 3070 path.

## 14. Main risks of a very broad stub universe

### API hallucination

Codex may treat a plausible name as an established algorithm. Status, source, and target metadata reduce this risk.

### False completeness

The registry can create the impression that all possibilities are covered. Generic composition and an explicit extension process are therefore mandatory.

### Layer leakage

A high-level operation may tempt Codex to issue a lower-level ioctl or SASS instruction directly. Strict package dependency checks must prevent this.

### Premature implementation

A large catalog can cause agents to implement novelty instead of the critical path. Priority tiers and experiment gates must govern work selection.

### Variant explosion

Do not create one API for every dtype/layout combination. Use the variant dimensions and only create a new canonical operation when behavior or algorithm changes materially.

## 15. Conclusion

The registry's purpose is not to pre-implement the future. It gives Alpha a coherent language for discussing the future.

The most important architectural choice is that **Alpha expresses meaning, Prometheus expresses computation, Hephaestus expresses the machine, and the lower runtime layers express ownership, time, transport and memory**. Novel optimization emerges by exploring alternative lowerings and compositions while preserving that separation.
