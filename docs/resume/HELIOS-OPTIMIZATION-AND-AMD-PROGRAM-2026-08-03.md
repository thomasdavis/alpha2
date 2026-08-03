# Helios optimization and AMD compatibility program

**Status:** active execution plan; exact profiler, portable IO-aware GEMM portfolio, and gradient-ownership forwarding validated, 2026-08-03
**Product goal:** finish Alpha as a genuinely chatty conversational model on one affordable GPU
**Engine goal:** make Helios a fast, numerically trustworthy, capability-driven training engine across NVIDIA and AMD hardware
**Immediate decision:** token caches are verified and five measured optimizations are selected; continue through the time-ranked graph and accelerator decision before beginning the multi-day Alpha foundation run

**First implementation result:** the new 16 × 16-workgroup, 2 × 2-per-thread scalar-FP32 GEMM reduced exact graph dispatch time by 36.9% and raised matched steady median training throughput from 3,579 to 4,513 tokens/s (+26.1%) with an identical six-step printed trajectory. Full evidence, rejected measurements, and AMD limitations are in `HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md`.

**Second implementation result:** physical-kernel attribution exposed hundreds of identity-scale gradient clones.
The autograd tape now moves a single-owner gradient buffer and clones only real aliases. It removed 728 operations
from the measured graph. A same-source trace-on ablation improved from 4,121.0 to 6,123.2 tokens/s (+48.6%); the
longer selected trace-off path measured p10/median/p90 of 6,432.6 / 6,567.7 / 6,666.5 tokens/s across 18 warm
steps. Matched losses and validation loss were exact, maximum gradient-norm difference was `6.913e-7`, and the
RTX 4090 suite passed 29 files / 283 tests. A separate fixed-order embedding-gradient kernel closed an intermittent
one-ulp replay failure without slowing the production scatter path. The mounted evidence is
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-gradient-ownership-20260803/`.

**Third implementation result:** a portable 16 x 8-workgroup, 4 x 2-output kernel was tested as a replacement for
the 2 x 2 GEMM family. It was not uniformly superior: R4x2 won for ordinary and transposed-A multiplication, while
R2 remained faster for transposed-B. The selected layout portfolio raised the 18-warm-step median from 6,567.7 to
6,836.8 tokens/s (+4.10%), with p10/p90 6,638.4 / 6,970.6. Maximum loss and gradient-norm differences from the
prior selected trajectory were `9.537e-7` and `4.308e-8`; learning rate, clipping coefficient, and terminal held-out
loss were exact. The RTX 4090 suite passed 29 files / 283 tests. Evidence is preserved at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-r42-portfolio-20260803/`.

**Fourth implementation result:** transposed-B profiling showed that adjacent invocations traversed physical B
columns at K-sized strides. An R42C variant makes adjacent X invocations load contiguous reduction elements and
transposes only into shared memory. The paired R2C control was correct but neutral; R42C reduced transposed-B GPU
time from 570,078.2 to 467,672.1 us (-17.96%) and exact full-graph dispatch time from 1,759,004.2 to 1,640,182.0
us (-6.75%). Across 18 warm production steps it raised median throughput from 6,836.8 to 7,048.9 tokens/s
(+3.10%), with p10/p90 6,844.8 / 7,200.8. The full suite passed 29 files / 283 tests. Evidence is preserved at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-coalesced-20260803/`.

**Fifth implementation result:** transposed-A profiling exposed the symmetric physical-layout problem in A.
R42C-A coalesces adjacent M loads from physical `[K,M]` A and transposes into the unchanged shared tile. Three
exact candidates measured 290,239.8-292,475.6 us across 91 calls versus controls at 336,395.8 and 338,954.0 us.
In candidate-first/control-second production runs, warm median improved from 7,085.0 to 7,253.8 tokens/s
(+2.38%), with exact loss and maximum gradient-norm drift `2.154e-8`. The full suite passed 29 files / 283 tests.
Evidence is preserved at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-a-coalesced-20260803/`.

**Experimental, not selected:** R42CK32 keeps the transposed-B coalesced mapping but doubles the reduction tile
to 32. It uses 8 KiB total shared memory and should halve load/barrier rounds, with a possible countervailing
occupancy and shader-size cost. The exact source `2ca869249da901763b7f4a69db939226753b198f` passed the awkward
Mesa numerical smoke and the complete local suite, but has no physical timing. It remains behind
`HELIOS_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32=1`, which the selected foundation launcher does not set. Evidence
and the physical promotion contract are at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-r42ck32-local-preflight-20260803/`.

**Rejected follow-up:** a portable vec4 RMSNorm column-sum kernel preserved the exact one-step trajectory but took
`64,568.7 us` across 37 calls versus `59,631.3 us` for the scalar reference, about 8.3% slower. It was reverted.
The failed candidate and an earlier invalid cooperative-path-confounded run are retained at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-column-sum-vec4-rejected-20260803/`. The result redirects
this line of work toward eliminating or hierarchically shrinking the full-sized RMSNorm partial tensor rather than
merely widening its second traversal.

**New unselected discriminator:** the failure mechanism of the vec4 attempt is now separated from the reduction
itself. Vec4 reduced an already small active population; the selected production reduction exposes only one
thread per roughly 512 columns while each thread walks 24,576 rows. The new `column_sum_row_lanes` candidate uses
32 adjacent columns x 8 row lanes, keeping global reads coalesced and combining row partials through 1 KiB of
workgroup memory without atomics or a subgroup-size assumption. It compiled and executed on llvmpipe subgroup 8,
matching an awkward 257 x 96 RMSNorm weight-gradient reference to `4.2915e-6` maximum absolute error. The local
suite is 233 pass / 55 physical-GPU-gated / 0 fail. It remains opt-in and unmeasured on a physical accelerator;
selection requires exact per-kernel profiling plus an alternating sustained trajectory comparison.

**Rejected attention follow-up:** the repository's previously unwired batched-unroll FlashAttention dKV V2 kernel
passed 29/29 physical-GPU parity tests and reproduced the exact one-step trajectory, but took `449,023.0 us`
across 18 calls versus `261,738.5 us` for the selected runtime-loop kernel, about 71.6% slower. The diagnostic
selector was reverted. Evidence is retained at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-flash-dkv-v2-rejected-20260803/`. Future dKV work must
change IO or work partitioning and explicitly measure register-pressure/occupancy effects; blind inner-loop
unrolling is ruled out for this exact head-dimension-64 graph.

## 1. Why this program exists

The selected Alpha foundation candidate is scientifically credible but currently expensive to train. The frozen candidate has 97,098,880 parameters, a 1,024-token training window, batch size 24, 79,020 optimizer steps, and 1,941,995,520 planned training tokens. The selected learning rate is `0.002`, chosen from three equal-token, equal-seed arms. The pre-optimization RTX 4090 Vulkan path sustained about 3,410-3,579 tokens/s after warmup, implying roughly 158 hours and USD 109 at USD 0.69/hour. The currently selected IO-aware-layout-plus-ownership recipe measures a sustained 18-warm-step median of 7,253.8 tokens/s, implying about 74.37 device-hours and USD 51.31 before validation/checkpoint overhead. These remain bounded estimates, not a completed full-run result.

That is a baseline, not an accepted final runtime. A prior synchronized Helios profile showed that backward computation consumes about 84% of forward-plus-backward wall time and that one ordinary step launches more than two thousand GPU operations. Optimizer and host overhead are small. The current opportunity is therefore algorithmic: fewer materialized intermediates, fewer reductions and transposes, fewer dispatches, more arithmetic intensity, better device-specific tiling, and correct reduced-precision matrix paths.

AMD compatibility is required even if the first Alpha run ultimately remains on NVIDIA. Helios already uses a cross-vendor API, SPIR-V, and a mostly portable kernel generator, but the current trainer rejects every non-NVIDIA device and several kernels silently assume a 32-lane subgroup. Those assumptions must become explicit capabilities, portable algorithms, or selected device-specialized variants.

This document is not permission to trade away mathematical correctness for attractive throughput. Every optimized path retains a portable reference path and must be checked against it. There is deliberately no arbitrary performance threshold: end-to-end measurements and tokens per dollar decide which path is useful.

All prior-art research and experimental evidence is part of the deliverable. Canonical research lives under
`/mnt/donto-data/donto-resources/research/alpha-helios/`; raw benchmark families live under
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-*`. The preservation contract is
`PRESERVATION-POLICY.md` in that research directory. Every success, null result, failure, invalid run, exact
command, device/driver identity, source commit, test result, and digest manifest must reach the mounted drive
before temporary files or remote machines are removed.

## 2. Boundaries and non-goals

This program serves the Alpha model; it does not replace the model objective with an engine benchmark.

- Alpha must become conversationally effective, not merely nonempty or low-loss.
- The foundation run, distillation, chat post-training, and untouched behavioral selection remain required after engine work.
- Helios remains the native tensor/autograd engine. It is not replaced with PyTorch, CUDA, or ROCm libraries merely to report a familiar benchmark.
- Vendor-specific fast paths are welcome, but every one is behind capability discovery and has a correct portable fallback.
- AMD support means more than deleting the trainer guard. Unsupported subgroup, matrix, device-generated-command, memory, or precision behavior must be detected before training.
- The current full run remains paused. Accelerator-independent tokenization may finish because it is reusable on every device.
- Discord receives model samples only after a genuine same-prompt behavioral improvement. Kernel speedups are recorded in repository evidence, not presented as model improvement.
- New checkpoints are published as new versions on Hugging Face and blah.dev only after honest behavioral gains.

## 3. Frozen evidence at the start of optimization

### 3.1 Foundation candidate

| Field | Value |
|---|---:|
| Parameters | 97,098,880 |
| Planned tokens | 1,941,995,520 |
| Optimizer steps | 79,020 |
| Block size | 1,024 |
| Batch size | 24 |
| Selected peak learning rate | 0.002 |
| Minimum learning rate | 0.0002 |
| Warmup steps | 790 |
| Evaluation interval | 500 |
| Checkpoint interval | 1,000 |
| Symbiogenesis | disabled |
| Contract source revision | `f394159d1259f4b1447c411a17afdea481bcdce2` |

The learning-rate pilot completed 384 steps and 9,437,184 tokens per arm. All three arms passed checkpoint, allocator, source, and metric-contract validation. The selected `0.002` arm had a final-three validation-loss mean of `6.010144432385762`, compared with `6.045725544293721` for `0.001` and `6.30454417069753` for `0.003`.

### 3.2 Measured engine profile

The synchronized 60M audit is the best current operation-level evidence:

- baseline throughput: about 5,334 tokens/s for that smaller benchmark;
- forward: about 502 ms;
- backward: about 2,683 ms;
- gradient norm: about 15 ms;
- AdamW: about 2 ms;
- ordinary step: 2,162 GPU operations;
- dominant operation families: unary 938, reductions 459, matmuls 259, binary 161, backward nodes 116, optimizer 114, in-place operations 81, layer normalization 33, softmax 1;
- dominant named kernels: scale 680, sum-reduce 231, strided sum-of-squares 162, transpose 128, AdamW step 114, add 113.

The evidence says to optimize the backward graph and memory traffic first. Optimizer micro-optimization cannot materially rescue this run.

### 3.3 Existing failed shortcuts

Several tempting settings were already tested and must not be rediscovered as folklore:

- larger workgroups were slower on the measured NVIDIA workload;
- block size 512 produced only a small throughput change and would alter the product/training contract;
- an earlier cooperative forward path was numerically wrong or non-finite;
- earlier mixed-precision paths were wrong, slower, or out of memory;
- increasing allocator-pool size alone did not materially improve throughput.

These are diagnoses of specific implementations, not proof that cooperative matrix or mixed-precision algorithms are inherently unsuitable. Their numerical and liveness defects should be repaired under controlled tests.

## 4. AMD means two backend tracks

### 4.1 Vulkan AMD: Radeon and Radeon Pro

This is the nearest compatibility target because it preserves the present native backend. AMD documents current Linux Radeon families with Mesa Vulkan support. AMD's architecture tables describe RDNA2/RDNA3 Radeon devices as 32-lane wavefront machines, which aligns with some existing Helios kernels, but Helios must query rather than infer that property from a vendor ID.

The Vulkan AMD track will:

1. enumerate Vulkan version, device type, memory heaps, subgroup size and supported subgroup stages/operations;
2. query `VK_EXT_subgroup_size_control`, its minimum/maximum subgroup sizes, and required-subgroup-size stages;
3. expose cooperative-matrix shapes, scalar types, scopes, and accumulation types rather than a single boolean;
4. expose buffer-device-address, push-descriptor, timeline-semaphore, device-generated-command, memory-budget, and timestamp capabilities;
5. admit a device on capability and smoke-test evidence, not vendor name;
6. select subgroup-agnostic kernels unless a required size has been requested and validated;
7. use Mesa RADV as the primary Linux Vulkan target, while recording driver ID and version in every benchmark;
8. add AMD telemetry through sysfs/DRM and, where available, `rocm-smi` or `amd-smi`, without making telemetry a training dependency.

### 4.2 ROCm/HIP AMD: Instinct accelerators

AMD Instinct MI300X is a CDNA3 compute accelerator with a documented 64-lane wavefront. AMD's official support and tuning material is centered on ROCm/HIP. It must not be assumed to expose a production Vulkan device merely because Radeon does.

If an available Instinct rental cannot initialize the Vulkan backend, Helios will gain a second native lowering target rather than pretending the device is unsupported forever:

- preserve the Helios tensor, operation, autograd, graph, allocator, checkpoint, and model APIs;
- introduce a backend-neutral operation/kernel intermediate representation where current SPIR-V construction is too tightly coupled;
- lower the same operation contracts to SPIR-V/Vulkan and HIP/ROCm;
- use HIP libraries only as optional leaf implementations behind Helios operation semantics, not as a replacement model runtime;
- share golden operation fixtures and model-step parity tests across both lowerings;
- specialize wave64 reductions and matrix tiles for CDNA while retaining the portable algorithm;
- fingerprint compiler, ROCm, driver, firmware, device architecture, and kernel variant in benchmark artifacts.

This is a larger engineering program than Vulkan-on-Radeon. It is nevertheless part of the compatibility goal because it opens the AMD data-center market instead of limiting “AMD support” to consumer graphics devices.

## 5. Device capability model

The current `GpuDeviceInfo` is too shallow. It reports a device name, vendor ID, a few booleans, and one cooperative-matrix shape. Replace it with a versioned capability record containing at least:

- API/backend and API version;
- vendor ID, device ID, device type, driver ID, driver name, driver version;
- total and budgeted device-local memory;
- maximum workgroup invocations and per-axis dimensions;
- maximum shared/workgroup memory;
- subgroup size, supported stages, supported operations, quad operations, and subgroup-size-control range;
- shader float16/int8/float64 capabilities and storage/buffer precision features;
- timestamp support and period;
- buffer device address, push descriptors, descriptor limits, timeline semaphores, synchronization version;
- cooperative-matrix feature set and every usable matrix shape/type/scope tuple;
- device-generated-command feature/property set;
- memory-budget and memory-priority support;
- a stable capability fingerprint.

Training admission becomes a report with three outcomes:

- **supported:** required operation suite and model-step smoke tests pass;
- **degraded:** correct portable path works, but one or more optional accelerators are unavailable;
- **unsupported:** a named required capability or correctness test fails.

No device is rejected merely for being AMD, Intel, or unknown. No device is accepted merely because its vendor is NVIDIA.

## 6. Operation-by-operation research program

Helios needs an inventory that maps every model operation to its forward kernel, backward construction, materialized intermediates, dispatch count, bytes moved, arithmetic work, supported dtypes, device requirements, and parity coverage. “Optimize all operations” means every row is measured and assigned one of: retain, fuse, specialize, replace, or retire.

### 6.1 Elementwise and broadcast operations

Current scale/add/unary dispatch counts suggest excess graph fragmentation.

Research and implementation candidates:

- lazy expression fusion over pure elementwise subgraphs;
- generated multi-output epilogues so residual writes, bias, activation, dropout-disabled identity, and scaling can share a traversal;
- vector-width selection from alignment and device capabilities;
- destination aliasing when liveness proves the input dead;
- direct gradient accumulation into an existing buffer where dependencies allow;
- specialization constants rather than compiling nearly identical shaders;
- fusion profitability based on measured memory traffic and register pressure, not operation count alone.

### 6.2 Reductions

Reductions are a major source of dispatch and memory traffic.

Research and implementation candidates:

- subgroup-size-agnostic reductions using SPIR-V subgroup built-ins;
- hierarchical reductions parameterized by the queried subgroup size;
- online/one-pass algorithms for mean, variance, max, sum, sum-of-squares, and log-sum-exp;
- fused reduction plus normalization/writeback;
- persistent multi-row reductions where workload shape permits;
- two-stage reduction only when a single workgroup cannot cover the logical row efficiently;
- wave32 and wave64 variants selected by capabilities and benchmark evidence;
- deterministic modes where atomic ordering would materially change validation.

### 6.3 LayerNorm and RMSNorm

- use numerically stable one-pass statistics where validated;
- fuse residual addition and normalization when the graph exposes that pattern;
- fuse backward statistics and input-gradient writeback;
- avoid materializing repeated scale, centered input, and inverse-standard-deviation tensors when a compact saved state is sufficient;
- compare recomputation with saved activations according to bytes, flops, and live-memory pressure.

### 6.4 Activations and gated MLPs

- fuse SwiGLU/GeGLU forward halves and output multiplication;
- implement a single backward kernel that consumes the saved or recomputed gate state and writes both input gradients;
- fuse bias and residual epilogues where present;
- benchmark approximation variants only if the exact model contract permits them—never silently change the activation.

### 6.5 Matrix multiplication

Matmul is computationally central even though graph fragmentation dominates dispatch count.

- maintain a portable tiled f32 implementation;
- enumerate device-reported cooperative-matrix shapes rather than assuming NVIDIA tiles;
- support f16/bf16 input with f32 accumulation and f32 master weights;
- repair overflow/underflow through explicit scaling and validate full model-step trajectories;
- autotune tile shape, register tiling, split-K, workgroup shape, double buffering, transposition strategy, and epilogues by exact matrix-shape families;
- cache autotuning by capability fingerprint and kernel revision;
- remove redundant physical transposes through layout-aware matmul variants;
- fuse gradient accumulation and bias reduction into backward matmul epilogues where safe;
- compare dense, cooperative, and vendor-specialized paths end to end, not only on square microbenchmarks.

### 6.6 Attention

The current portable attention path and cooperative experimental paths need separate correctness and performance treatment.

- retain an exact reference implementation;
- implement IO-aware tiled forward and backward paths based on the principles of FlashAttention and FlashAttention-2;
- specialize work partitioning for head dimension, sequence length, subgroup size, and on-chip memory;
- avoid storing the full attention matrix when exact recomputation is cheaper;
- combine causal masking, scale, row max, exponential sum, normalization, and value accumulation in the tiled algorithm;
- validate layout/token parity aggressively because a prior flash-attention layout scramble existed in this repository;
- keep NVIDIA `VK_NV_cooperative_matrix2` as an optional NVIDIA-only experiment, never the portable definition;
- explore KHR cooperative matrices on devices that report suitable shapes;
- build native wave32 and wave64 scalar/subgroup variants when cooperative matrices are absent.

### 6.7 Embedding, gather, scatter, and cross-entropy

- coalesce embedding gathers and tied-output access;
- fuse masked cross-entropy log-sum-exp, target-logit extraction, loss reduction, and gradient production;
- preserve the proven binding parity that caught the earlier swapped masked-cross-entropy kernels;
- use sparse/indexed gradient accumulation only where it is faster and numerically equivalent for the vocabulary shape;
- benchmark atomics against sorted/segmented reductions per device.

### 6.8 Optimizer and gradient norm

These are currently small portions of wall time, so work follows higher-yield graph changes.

- combine finite check, norm accumulation, clipping scale, and optimizer update where dependency structure permits;
- batch parameter tensors into fewer launches using stable metadata tables;
- preserve f32 optimizer state and checkpoint exactness;
- avoid optimizing AdamW in isolation unless a new profile shows it has become material after other fusions.

### 6.9 Graph execution and device-generated commands

The Khronos device-generated-command specification warns that trivial single-command uses may be slower than ordinary indirect dispatch. Helios will therefore measure DGC rather than treating it as automatically beneficial.

- timestamp and count every dispatch and barrier;
- construct repeatable static training-step command rails;
- use DGC for sequences where device-side selection or reduced host work is measurable;
- compare preprocessed and implicit preprocessing modes;
- retain ordinary command-buffer and indirect-dispatch paths;
- collapse barriers using precise read/write dependency information;
- consider persistent graph executors only after watchdog, fairness, and portability implications are understood.

### 6.10 Memory allocation and liveness

- replace opportunistic pool growth with a static or semi-static liveness plan for the repeated training graph;
- reuse memory across non-overlapping activations and gradients;
- separate long-lived parameters/optimizer state from step arenas and temporary workspaces;
- exploit buffer device address only when available;
- record peak live bytes by operation and graph phase;
- validate aliasing with poisoned-buffer and delayed-use tests;
- overlap checkpoint transfer/compression only after compute correctness and memory headroom are proven.

### 6.11 Token and data path

Pretokenized immutable shards remove tokenizer work from the training loop. Remaining work is lower priority because synchronized measurements show GPU computation dominates.

- asynchronous shard reads and pinned/staged batches;
- deterministic shuffle/replay metadata;
- double-buffered upload where host-device transfer is visible;
- checkpoint writes outside the critical path when memory and durability permit;
- exact token accounting and cache fingerprints independent of accelerator.

### 6.12 State-of-the-art adoption lane

For every material operation family, the ledger must identify the strongest relevant published technique and explain whether Helios adopts it, adapts it, or rejects it on measured grounds. The initial adoption map is:

| Research idea | Helios adaptation question |
|---|---|
| FlashAttention 1/2 IO-aware exact tiling and work partitioning | Can the exact forward/backward algorithm be expressed efficiently in portable SPIR-V for both wave32 and wave64 without reproducing the old layout bug? |
| FlashAttention-3 asynchronous pipelines, warp specialization, GEMM/softmax interleaving, block-scaled low precision | Which principles transfer to Vulkan cooperative matrices and AMD asynchronous facilities, and which are inseparable from Hopper TMA/WGMMA? |
| FlashAttention-4 asymmetric pipeline co-design, software exponentials, larger tiles, and backward shared-memory/atomic reduction | Which bottleneck-shift ideas survive on scalar Vulkan, cooperative Vulkan, Radeon wave32, and Instinct wave64 when tensor throughput grows faster than non-matmul hardware? |
| CODA GEMM-plus-epilogue programs | Can Helios retain the selected GEMM mainloop while keeping residual, RMSNorm, SwiGLU, RoPE, cross-entropy partial reductions, and backward accumulation on chip instead of materializing hundreds of memory-bound operations? |
| ThunderKittens / HipKittens tile, block, and grid-level abstractions | Can Helios introduce a small backend-neutral tile algebra that emits SPIR-V and HIP while keeping tensor layouts explicit and inspectable across NVIDIA and AMD? |
| cuTile Rust ownership-safe tile kernels and asynchronous launch contracts | Can the ownership proof that removed gradient clones become a compile-time disjoint-tile and buffer-lifetime discipline for the future Helios IR rather than a collection of runtime conventions? |
| Mirage multi-level superoptimization | Can Helios search algebraic, workgroup, and kernel-fusion transformations jointly over its own typed operation graph, with equivalence tests as the verifier? |
| Cut Cross-Entropy on-the-fly classifier/loss computation | Does fusing the tied output projection, log-sum-exp, target logit, and backward remove a material logits allocation or HBM round-trip for Alpha's 12,288-token vocabulary? |
| BF16 stochastic rounding and muNit/FP8 scaling | Can operation-specific unbiased rounding and principled scaling rescue Helios's previously incorrect broad mixed-precision path while preserving a fixed-token trajectory? |
| AMD AITER and Composable Kernel portfolios | Which shape-specialized GEMM, attention, RMSNorm, and fusion policies should inform the HIP lowering without importing an opaque runtime as Helios itself? |
| GEAK v4, Kernel-Smith, and harness-governed LLM kernel search | Can a retained population of compiled candidates plus profiler/correctness/trajectory feedback turn Codex-assisted optimization into a reproducible local improver, while certificates and physical measurements prevent reward hacking? |
| Atrex trace-weighted kernel evaluation | Can Helios weight candidate work by measured share of full Alpha device time and a roofline ceiling, while rejecting fallback implementations that pass correctness without executing the proposed kernel? |
| Kernel Forge MCTS and whole-model reintegration | Can branching search escape locally attractive schedules while every retained candidate is re-integrated into the unchanged native Alpha graph rather than winning only on isolated random tensors? |
| Kerncap AMD kernel capture and replay | Can a future HIP backend snapshot exact Helios AMD operands and launch state so candidate edit-compile-validate loops run cheaply without weakening whole-step validation? |
| hipBLASLt offline tuning | Can exact Alpha GEMM shapes use library results as a device-and-release-specific performance ceiling and proposal source while Helios retains its native execution and fingerprinted portable path? |
| Deep Kernel Fusion / megakernels | Which complete transformer forward/backward slices become faster when intermediates remain on-chip, and where do register pressure and lost occupancy make fusion harmful? |
| Persistent kernels and on-device schedulers | Can repeated training-step subgraphs remain resident or device-scheduled without losing fairness, watchdog safety, checkpoint observability, or cross-vendor support? |
| Communication-avoiding and recomputation algorithms | At one-GPU scale, which saved tensors should be recomputed to reduce HBM traffic and peak liveness rather than retained by conventional autograd? |
| Automated shape-specific scheduling | Can a capability-fingerprinted portfolio beat one global tile/workgroup choice across Alpha's actual matrix and reduction shape ecology? |

“State of the art” is established by current primary literature plus reproducible comparison. A CUDA-specific paper is a source of algorithms, not permission to call an unmeasured Vulkan imitation equivalent.

### 6.13 Original Helios research hypotheses

Helios should also try ideas that are not merely ports. These are hypotheses, not pre-announced successes. Each receives a reference implementation, counterfactual control, and a retirement record if it loses.

#### H1 — Kernel ecology and cross-device evolutionary search

Represent a kernel family as a typed genome containing algorithm, tile geometry, subgroup policy, vector width, staging depth, memory layout, fusion boundary, precision policy, and dispatch strategy. Generate legal mutations from device capabilities; reject candidates through operation and trajectory parity; select on a Pareto frontier of latency, throughput, memory, compilation cost, dollars/token, and portability.

Unlike ordinary autotuning over a fixed template, mutations may change the algorithm and fusion boundary. A winning Radeon wave32 genome can seed—but never dictate—the search on NVIDIA or CDNA wave64. The ledger preserves every candidate and failure so the system learns which transformations transfer between architectures.

#### H2 — Backward quotient compiler

Canonical autograd expands a compact forward expression into many repeated scales, reductions, broadcasts, transposes, and additions. Construct an algebraic quotient graph that identifies gradient expressions equivalent under associativity, distributivity, broadcasting, and layout transforms. Emit one fused producer for shared subexpressions and accumulate gradients at the latest safe point.

The falsifiable claim is not “fusion helps.” It is that quotienting the backward graph eliminates a measurable fraction of the 680 scale, 459 reduction, and 128 transpose operations without changing the deterministic update trajectory beyond declared numerical tolerance.

#### H3 — Sensitivity-budgeted precision cartography

Instead of one global mixed-precision switch, estimate each operation family's output/gradient sensitivity using directional derivatives, calibration trajectories, dynamic range, and downstream amplification. Allocate a bounded numerical-error budget across the graph. Select f32, f16, bf16, block-scaled low precision, or recomputation per operation and shape, with f32 master state and automatic fallback when the observed budget is exceeded.

This is especially relevant to the earlier Helios result where a broad mixed-precision mode was wrong. The hypothesis is that precision should follow measured sensitivity topology, not tensor category alone.

#### H4 — Temporal memory coloring with recomputation edges

Treat storage planning as a weighted interval-coloring problem over the repeated forward/backward graph, extended with optional recomputation edges. Jointly choose aliasing, layout, workspace, and recomputation so that peak live bytes and HBM traffic are optimized together. Feed actual timestamp and byte measurements back into the plan after each compiled graph revision.

The control is the current allocator plus conventional activation retention. Poisoned-region tests and exact delayed-use fixtures detect illegal aliases.

#### H5 — Subgroup-polyvariant kernel algebra

Generate the same semantic kernel from subgroup-neutral primitives, then specialize into wave32, wave64, and explicitly required subgroup variants. Use subgroup built-ins rather than deriving lanes from local IDs. Make cross-subgroup combination depend on the runtime-reported subgroup count. Cooperative-matrix tiles are separate capability-selected leaves, not assumptions embedded in semantic code.

The research question is whether one inspectable algebra can approach vendor-specific performance across NVIDIA and AMD without a forked library of unrelated kernels.

#### H6 — Semantic training megakernels

Fuse at model-semantic boundaries rather than arbitrary adjacent operators: residual-plus-norm-plus-projection, gated-MLP backward, attention backward, masked-loss backward, or gradient-check-plus-clip-plus-update. A semantic megakernel knows which intermediates are needed for exact backward and can keep them in registers/shared memory or recompute them deliberately.

Compare against both unfused Helios and surface-level elementwise fusion. This tests whether knowledge of transformer/autograd structure produces benefits beyond generic compiler fusion.

#### H7 — Counterfactual roofline profiler

Augment measured timestamps with executable counterfactuals: estimate and then test what happens if one materialization, transpose, barrier, or dispatch boundary disappears. Rank changes by expected end-to-end gain, uncertainty, and implementation cost. Update the estimator from every landed or rejected optimization.

This creates a closed research loop where profiling proposes falsifiable interventions rather than producing a static flame graph.

#### H8 — Verified on-device graph evolution

Use device-generated commands or a persistent scheduler to choose among prevalidated kernel variants from runtime shape, memory-pressure, and numerical-state signals. The device may select only from an immutable signed/hashed portfolio whose members already passed parity. It cannot synthesize unverified shader code during a training run.

The experiment compares host-selected static graphs, host autotuning, and device-selected portfolios at identical semantics. This is particularly interesting where Vulkan dispatch overhead and repeated optimizer/autograd structure dominate.

#### H9 — Cross-vendor performance transfer model

Learn which kernel and graph features predict performance from the accumulated benchmark ledger, while retaining active exploration on each new capability fingerprint. The model recommends the first benchmark population for an unseen device; direct measurement remains authority.

The useful result would be faster convergence to good AMD configurations from NVIDIA history without assuming equal subgroup, cache, register, or matrix hardware.

#### H10 — Trajectory equivalence as a compiler objective

Most kernel systems validate isolated outputs. Helios will make short deterministic training-trajectory agreement a first-class compiler/search constraint. Candidate transforms are ranked not only by local error but by their effect on loss, gradient direction, parameter update, and checkpoint continuation across several steps.

This may reject locally plausible fast kernels that systematically bias training, and may identify operation-specific error patterns that ordinary max-absolute-error tests miss.

#### H11 — Certificate-carrying kernel evolution

Every generated kernel mutation carries a machine-readable claim about the semantic transformation it applies,
the tensor/layout preconditions it assumes, the capabilities it requires, its estimated resource bounds, and the
metamorphic identities that should remain true. Promotion requires compiling the certificate into adversarial
fixtures and a short trajectory gate. This combines search with explicit failure obligations: the optimizer
cannot merely discover a fast shader and rely on a broad end-to-end tolerance to conceal why it works.

The control is ordinary black-box evolutionary/autotuning search over the same candidate budget. The hypothesis
is that certificates improve cross-shape correctness and reduce wasted physical-GPU measurements without
preventing discovery of non-obvious schedules.

#### H12 — Dual-lowering differential evolution

Apply the same typed kernel mutation to both SPIR-V/Vulkan and HIP/ROCm lowerings. Compare outputs, gradients,
compiler diagnostics, resource usage, and trajectory behavior across NVIDIA, Radeon, and Instinct. Agreement
across independently compiled backends becomes a differential correctness oracle; disagreement becomes a
localized compiler/undefined-behavior investigation rather than an unexplained model failure.

This is not an assumption that performance transfers. The cross-vendor performance divergence becomes training
data for H9, while semantics must remain equal. The control mutates and validates each backend independently.

#### H13 — Epilogue residency synthesis

Treat each high-cost GEMM not as a terminal operation but as an on-chip residency opportunity. Starting from the
selected register-blocked mainloop, synthesize typed epilogue programs that may consume auxiliary tensors,
transform accumulators, emit compact reduction state, and hand off a still-resident tile to the next semantic
operation. Search the joint boundary between GEMM schedule, epilogue program, saved backward state, and later
recomputation rather than greedily fusing whichever operations are adjacent.

CODA is the nearest direct prior art and therefore the mandatory control. The proposed Helios difference is a
cross-vendor SPIR-V/HIP lowering, integration with the backward quotient and liveness planner, and trajectory
equivalence as a promotion constraint. The first falsifiable slice is selected from the exact Alpha graph—not a
square toy GEMM—and must beat both unfused Helios and a direct CODA-style fixed epilogue at identical semantics.

#### H14 — Evidence-distilled kernel scientist

Every physical-device experiment becomes a structured episode: bottleneck evidence, proposed mechanism, source
patch, compiler diagnostics, adversarial parity, resource measurements, trajectory result, and promotion or
rejection. An agent retrieves the most relevant successful and failed episodes for a new shape/device, proposes
a bounded candidate population, and updates a reusable transformation skill only from verified deltas.

Kernel-Smith, GEAK, and current harness-engineering work already establish agentic/evolutionary kernel search;
that is not the novelty claim. The Helios hypothesis is that preserved negative evidence, cross-backend
differential execution, certificate-generated tests, and training-trajectory equivalence produce better useful
candidates per physical-GPU minute than speed-only evolutionary search. A non-agentic autotuner and an agent
without the retained evidence ledger are required controls.

### 6.14 Novelty discipline

Every original hypothesis receives:

- a dated claim statement before implementation;
- nearest known prior art and the exact proposed difference;
- a control implementation using identical model/data tokens;
- operation, block, model-step, and trajectory correctness evidence as applicable;
- synchronized end-to-end measurements;
- negative and null results preserved in the ledger;
- no “novel” label in public claims until a bounded prior-art audit and empirical result support it.

This keeps creative work ambitious without turning Alpha's compute budget into undiagnosable experimentation.

## 7. Profiler and evidence format

Optimization without device timestamps is guesswork. Add an opt-in profiler that records:

- CPU record/submit/wait time;
- GPU timestamps for every dispatch or bounded dispatch group;
- kernel stable ID, source revision, specialization constants, workgroup shape, grid shape, dtype, input/output shapes;
- bytes read/written and estimated arithmetic operations where derivable;
- barriers and dependency reason;
- allocations, aliases, temporary workspace, and peak live memory;
- forward/backward/optimizer phase and autograd node;
- device capability fingerprint, driver, price snapshot, and environment;
- warmup-excluded p10, median, p90, and dispersion;
- operation count and percentage of end-to-end step wall time.

The profiler must be bounded and optionally sample only selected steps so it does not become the default runtime. Its JSON output will feed a generated Markdown/CSV operation ledger. Every optimization proposal must point at a measured row; every completed change must attach before/after parity and performance evidence.

## 8. Validation ladder

Each new kernel or graph rewrite moves through the same ladder:

1. **Shader/build validation:** SPIR-V validation and extension/capability declaration.
2. **Operation fixtures:** adversarial shapes, non-contiguous/layout variants, short/long reductions, masked positions, special values, and multiple dtypes.
3. **Finite difference:** forward/backward agreement where autograd is involved.
4. **Reference parity:** outputs and gradients against the retained portable implementation.
5. **Composed block parity:** complete transformer block forward and backward.
6. **Model-step parity:** loss, gradients, norm, update, and checkpoint after one and several deterministic steps.
7. **Short trajectory:** fixed-token training trajectory, allocator health, non-finite checks, and loss comparison.
8. **End-to-end throughput:** only after correctness, with warmup and synchronized timestamps.
9. **Cross-device replay:** identical fixture corpus and model contract on NVIDIA and AMD.

Tolerance is operation- and dtype-specific and reported, not hidden behind a single permissive epsilon. Fast reduced-precision paths must also report checkpoint trajectory behavior because a one-step close result can still diverge during training.

## 9. Accelerator bake-off

### 9.1 Workload

Every candidate device runs the same frozen sequence:

- Helios operation conformance suite;
- complete transformer-block forward/backward fixtures;
- 97M candidate memory preflight;
- synchronized fixed-step training benchmark on the same pretokenized shard windows;
- a bounded trajectory long enough to expose compilation, allocator, thermal, and stability behavior;
- checkpoint save/load and continuation parity.

The comparison records tokens/s, tokens/USD, peak memory, power/thermal telemetry when available, failures, and which fast paths were actually active.

### 9.2 Current RunPod price snapshot

The live price snapshot observed on 2026-08-03 includes the following single-GPU rates. Prices and availability can change, so the benchmark artifact records the price actually offered at provisioning time.

| GPU | USD/hour | Throughput needed to match the current 4090 dollars/token |
|---|---:|---:|
| RTX 4090 | 0.69 | 3,564 tok/s baseline |
| RTX 5090 | 0.99 | 5,113 tok/s |
| L40S | 0.99 | 5,113 tok/s |
| RTX 6000 Ada | 0.84 | 4,339 tok/s |
| A40 | 0.44 | 2,272 tok/s |
| A6000 | 0.53 | 2,738 tok/s |
| A100 PCIe | 1.39 | 7,181 tok/s |
| A100 SXM | 1.49 | 7,698 tok/s |
| H100 SXM | 2.99 | 15,447 tok/s |
| MI300X, last official RunPod price found | 3.99 | 20,614 tok/s |

The break-even values are arithmetic, not forecasts. A cheaper A40 or A6000 could win dollars/token while losing raw throughput; a 5090 could win both if the implementation exploits it; an H100 or MI300X is poor value unless Helios reaches substantially higher utilization. Live `runpodctl gpu list` currently exposes no AMD type for this account, so Radeon/Instinct benchmarking may require another provider or a temporarily attached test machine.

### 9.3 Decision record

The selected device/backend contract reports:

- fastest correct end-to-end configuration;
- cheapest correct tokens per dollar;
- expected full-run time and cost with confidence interval;
- memory headroom and checkpoint footprint;
- active kernel variants and capabilities;
- exact code, driver, model, tokenizer, and data fingerprints;
- any portability or numerical limitations.

The operator may select raw time, cost, or broader Helios research value. The evidence remains even when a device loses.

## 10. Execution order

### Phase O0 — preserve reusable work

1. Finish and verify the three active pretokenization caches.
2. Record token counts, cache hashes, elapsed time, and tokenizer fingerprint.
3. Keep the complete Alpha full-run contract paused.

### Phase O1 — capability truth

1. Expand native Vulkan physical-device/property queries.
2. Replace the NVIDIA trainer rejection with capability-based admission.
3. Add a human-readable and JSON device report.
4. Add subgroup-agnostic implementations or explicit required-size selection for every existing hard-coded subgroup kernel.
5. Disable NVIDIA-only cooperative-matrix2 automatically outside its declared capability.
6. Prove the ordinary NVIDIA path is unchanged.

### Phase O2 — profiler and operation ledger

1. Add bounded per-dispatch GPU timestamps and graph metadata.
2. Reproduce the 97M benchmark on the current 4090.
3. Generate a complete operation ranking.
4. Select optimization work from measured cumulative wall time.

### Phase O3 — first high-yield portable rewrite

Expected starting targets, subject to the new profile:

1. prototype one CODA-controlled GEMM epilogue on an exact Alpha shape;
2. elementwise expression/gradient fusion;
3. subgroup-portable hierarchical reductions;
4. fused norm and gated-MLP backward paths;
5. layout-aware backward matmuls with fewer transposes;
6. exact fused masked cross-entropy backward.

Each lands separately with parity and end-to-end evidence.

### Phase O4 — matrix and attention research paths

1. repair f16/bf16 cooperative matmul with f32 accumulation;
2. enumerate/autotune cooperative shapes per device;
3. implement and validate portable IO-aware attention forward/backward;
4. retain device-specialized NVIDIA and AMD variants behind capability checks.

### Phase O5 — AMD proof

1. run build/device/conformance tests on a Linux RDNA2/RDNA3 Vulkan device;
2. run a complete deterministic training step and short trajectory;
3. benchmark and record the Radeon result;
4. provision an Instinct device when available and test Vulkan truthfully;
5. if Vulkan is absent or unsuitable, begin the shared-IR HIP/ROCm lowering and prove the first operation slice before expanding it.

### Phase O6 — hardware bake-off and frozen launch

1. compare the optimized 4090 against credible NVIDIA rentals and any available AMD device;
2. choose the one-GPU contract from correct end-to-end evidence;
3. update the immutable full-run contract with backend/device fingerprints;
4. launch the foundation run;
5. monitor real token progress, VRAM, host pressure, checkpoint validity, and training health.

### Phase O7 — finish Alpha

1. select foundation checkpoints without opening sealed chat evaluation;
2. distill and post-train for response initiation, semantic contingency, stable stopping, length control, and natural dialogue;
3. evaluate on the baseline-eligible frozen prompt set and untouched final suite only under its selection contract;
4. publish only a behavioral winner to a new Hugging Face and blah.dev version;
5. generate and validate the matching Jacobian Lens artifacts after the exact published checkpoint is immutable.

## 11. Storage, recoverability, and cost control

The operator asked for a pause if this work creates more than roughly 15 GiB of new retained project data. Track new profiler traces, binaries, comparison checkpoints, and benchmark outputs explicitly. Raw transient build caches do not become research claims, but evidence required to reproduce a decision is retained on `/mnt/donto-data`.

- preserve source and artifact hashes;
- compress losslessly where practical;
- delete a remote checkpoint only after local recovery verification;
- keep benchmark traces bounded by sampled steps;
- avoid duplicating the same immutable caches per device;
- terminate idle rental devices promptly after evidence is copied and verified;
- never sacrifice the selected Alpha checkpoint or tokenizer lineage to save space.

## 12. Research sources and implementation implications

Primary sources guiding the first implementation pass:

- [Vulkan subgroup-size control](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_subgroup_size_control.html): query and, where supported, request subgroup size instead of assuming 32.
- [Vulkan subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html): record native size, stages, and operations.
- [Vulkan device-generated commands](https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html): use DGC only where sequence-level measurement justifies its preprocessing and execution overhead.
- [AMD accelerator and GPU architecture specifications](https://rocm.docs.amd.com/en/docs-6.1.5/reference/gpu-arch-specs.html): RDNA wave32 and CDNA wave64 require genuinely different reduction and tiling assumptions.
- [AMD Radeon Software for Linux 25.10.2 notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-25-10-2.html): Mesa Vulkan is the forward Linux Radeon target.
- [AMD ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.5/reference/system-requirements.html): Instinct and Radeon compute support is a ROCm compatibility question separate from Vulkan graphics support.
- [FlashAttention](https://arxiv.org/abs/2205.14135): make attention exact and IO-aware rather than materializing avoidable HBM traffic.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): improve work partitioning and sequence parallelism rather than assuming the first tiled mapping is optimal.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): study asynchronous overlap, GEMM/softmax interleaving, and block-scaled low precision while separating Hopper-specific mechanisms from portable principles.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451): account for asymmetric hardware scaling, non-matmul bottlenecks, software exponential/rescaling, larger asynchronous tiles, and reduced shared-memory/atomic traffic in backward.
- [CODA](https://arxiv.org/abs/2605.19269): use GEMM-plus-epilogue programs as the direct control for keeping Transformer residual, normalization, activation, RoPE, loss, and backward work on chip.
- [ThunderKittens](https://arxiv.org/abs/2410.20399): study a compact hierarchy of tile, block, and grid primitives as inspiration for a cross-backend Helios kernel algebra.
- [Fearless Concurrency on the GPU](https://arxiv.org/abs/2606.15991): study ownership-safe disjoint tiles and asynchronous launch contracts as prior art for a future Rust-first Helios kernel IR.
- [Mirage](https://arxiv.org/abs/2405.05751): study multi-level algebra/schedule/kernel superoptimization with explicit verification.
- [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009): test an on-the-fly tied classifier/loss path rather than materializing the complete token-by-vocabulary logits matrix.
- [Stochastic Rounding for LLM Training](https://arxiv.org/abs/2502.20566): evaluate unbiased low-precision rounding as a trajectory-level intervention, not a local cast benchmark.
- [muNit Scaling](https://arxiv.org/abs/2502.05967): study principled FP8 scaling and hyperparameter transfer while keeping Helios's existing f32 path as authority.
- [Deep Kernel Fusion for Transformers](https://arxiv.org/abs/2602.11808): investigate deep fusion as a research direction, but validate its numerical and register-pressure tradeoffs in Helios rather than copying headline results.
- [KernelFoundry](https://arxiv.org/abs/2603.12440): use quality-diversity and hardware-aware evolutionary search as the nearest control for H1 rather than claiming evolutionary search itself as novel.
- [Dr. Kernel](https://arxiv.org/abs/2602.05885): use profiling-based rejection and reward-hacking controls when an LLM proposes kernels.
- [GPU Forecasters](https://arxiv.org/abs/2605.31464): treat selective performance prediction as prior-art for H9; the physical device remains the final authority.
- [ROCm AITER optimization guide](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html): inventory AMD's current fused attention, GEMM, RMSNorm, and online-tuning portfolio for the HIP backend.
- [AMD GEAK v4](https://www.amd.com/en/developer/resources/technical-articles/2026/geak-v4.html): treat current agentic AMD kernel optimization as an engineering baseline, including its real-device feedback loop.
- [Kernel-Smith](https://arxiv.org/abs/2603.28342): use retained executable populations and structured execution feedback as the nearest evolutionary-agent control for H14.
- [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979): separate the correctness/timing/archive harness from the agent controller, and prefer expert/evidence-assisted proposals over unconstrained full-agent search.
- [Atrex-Bench and Atrex-Kernel-Agent](https://arxiv.org/abs/2607.14541): weight work by production-trace importance and roofline headroom, detect framework fallbacks, and use optimization dropout only inside a correctness-governed measure-revise loop.
- [Kernel Forge](https://arxiv.org/abs/2607.24762): treat MCTS and whole-model in-place integration as direct controls for the Helios kernel ecology rather than claiming branching agent search itself as novel.
- [Kerncap](https://arxiv.org/abs/2605.03208): use captured AMD kernel state to shorten future HIP edit-recompile-validate cycles while retaining whole-step parity as the promotion gate.
- [ROCm AITER](https://github.com/ROCm/aiter): use AMD's multi-backend, shape-tuned attention/GEMM/RMSNorm portfolio as a moving performance reference for the HIP path, including training operators and accuracy fixes.
- [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.2/how-to/how-to-use-hipblaslt-offline-tuning.html): record tuned solution indices with both device architecture and library release because AMD explicitly states that results are not portable across either boundary.
- [ROCm graph-safe library status](https://rocmdocs.amd.com/en/develop/reference/graph-safe-support.html): do not assume Composable Kernel or every ROCm library can be embedded safely in a captured/replayed training graph.
- [Composable Kernel MI300 block GEMM](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/gemm_optimization.html): use LDS, MFMA, occupancy, and shape-tuning guidance as the Instinct baseline.
- [Vulkan cooperative matrices](https://github.khronos.org/Vulkan-Site/tutorial/latest/Advanced_Vulkan_Compute/09_Specialized_Math/02_cooperative_matrices.html): enumerate implementation-supported shapes; Vulkan 1.4 exposure does not make any fixed tile universal.
- [Vulkan maximal reconvergence](https://github.khronos.org/Vulkan-Site/features/latest/features/proposals/VK_KHR_shader_maximal_reconvergence.html): make divergent subgroup behavior explicit before relying on portable tangled operations.
- [RunPod pricing](https://www.runpod.io/pricing) and [RunPod GPU types](https://docs.runpod.io/references/gpu-types): record volatile price/availability at benchmark time.

## 13. Evidence locations

Existing evidence:

- `docs/resume/HELIOS-CHAT-THROUGHPUT-AUDIT-2026-08-02.md`
- `docs/resume/HELIOS-CHAT-THROUGHPUT-SWEEP-OUTCOME-2026-08-02.md`
- `docs/resume/FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md`
- `docs/resume/SAME-DATASET-RECIPE-AUDIT-2026-08-02.md`
- `/mnt/donto-data/alpha-runs/alpha-foundation-lr-pilot-20260803/`
- `/mnt/donto-data/donto-resources/research/alpha-rejected-foundation-probes-v9-v11-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-gradient-ownership-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-column-sum-vec4-rejected-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-flash-dkv-v2-rejected-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-r42-portfolio-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-coalesced-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-a-coalesced-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-production-oracle-preflight-20260803/`
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-forward-contract-audit-20260803/`
- `/mnt/donto-data/donto-resources/research/alpha-helios/CURRENT-BOTTLENECK-LEDGER-2026-08-03.md`
- `/mnt/donto-data/donto-resources/research/alpha-helios/PERFORMANCE-PRIOR-ART-AND-OPPORTUNITY-AUDIT-2026-08-03.md`

New optimization evidence should be stored under:

- repository summaries: `docs/resume/`;
- bounded machine-readable benchmarks: `perf/helios-optimization/`;
- larger research artifacts: `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-optimization-20260803/`.

## 14. Completion definition

This program is not complete when AMD is allowed past one `if` statement, when one microkernel is faster, or when a model starts training.

It is complete when:

1. Helios describes and admits devices by capabilities rather than NVIDIA identity;
2. the full portable operation suite has no hidden subgroup-32 dependency;
3. a real AMD device passes operation, autograd, checkpoint, and bounded training-trajectory validation;
4. an Instinct path is either proven through Vulkan or supported through a native HIP/ROCm lowering with the same semantics;
5. every material training operation is present in the measured operation ledger;
6. the highest-cost operation families have validated portable or device-specialized optimization decisions;
7. the Alpha accelerator contract is chosen from end-to-end correctness, time, and cost evidence;
8. the full Alpha training, conversational post-training, untouched behavioral selection, and versioned publication are completed.

The engine work is successful only if it helps finish the model—and leaves behind a faster, more portable Helios rather than a one-off benchmark branch.
