# Standing goal — maximise training throughput on the foundation shape

**Set:** 2026-08-03 · **Target revised:** 2026-08-04 · **Status:** ACTIVE
**Parent goal:** [`GOAL-EXTREME-PERFORMANCE-2026-08-03.md`](GOAL-EXTREME-PERFORMANCE-2026-08-03.md) (10× cost reduction)
**Derivation:** `donto-resources/research/alpha-helios-reimagined/experiments/x16_throughput_ladder.py`

---

## The goal

> Raise sustained **complete-step** training throughput on the exact frozen
> foundation shape to **64,000 tokens/s on one RTX 3090**. **50,000 tokens/s is the
> first acceptance gate; 70,000 tokens/s is the stretch.** Every rung is gated on
> numerical parity, and no rung is earned by changing what is computed unless that
> change has already been shown loss- and behaviour-neutral.

| Target | tokens/s | vs current 3090 evidence | GPU-hours for the 1.942B-token run | Cost at $0.22/h |
|---|---:|---:|---:|---:|
| Current quality-bearing RTX 3090 warm median | 7,762 | 1.0× | 69.5 | $15.29 |
| **First acceptance gate** | **50,000** | **6.44×** | **10.8** | **$2.37** |
| **Primary goal** | **64,000** | **8.25×** | **8.4** | **$1.85** |
| **Stretch** | **70,000** | **9.02×** | **7.7** | **$1.70** |
| Hard BF16 roofline | 252,909 | 32.6× | 2.1 | $0.47 |

Shape is fixed throughout: 18 layers, d=640, 10 heads × 64, FFN 1,728, vocab
12,288, and sequence length 1,024. “Complete-step” includes forward, backward,
parameter gradients, gradient norm/clipping, optimizer update, required mixed-
precision bookkeeping, and synchronization before the next batch. Batch changes
must be accompanied by tokens-to-target evidence.

The revised primary target is anchored by a public `llm.c` report of approximately
64,285 tok/s and 72.5–72.7% BF16 MFU for GPT-2 124M at sequence length 1,024 on a
single power-limited RTX 3090. It is an external comparator, not an Alpha result:
the architecture, stack, batch, optimizer path, and corpus differ, and the report
does not name an immutable code revision. Alpha will pin and study the official
source, map its relevant mechanisms onto Helios, and implement them in Helios.
The operator explicitly declined a separate external control run on 2026-08-04;
no GPU reproduction or competing CUDA training engine is part of this plan.
Source: https://github.com/karpathy/llm.c/discussions/481

## Why these numbers and not rounder ones

They are the output of the measured ladder, not an aspiration. The step decomposes as
`step = host_build + gpu_execute` (unoverlapped — the X8 diagnosis), and today that is
1,717 ms of host against 1,671 ms of GPU kernels.

| Rung | change | step ms | tokens/s | vs today |
|---|---|---:|---:|---:|
| — | baseline, measured | 3,388 | 7,254 | 1.00× |
| **R1** | record the static graph once and replay; overlap host with GPU | 1,842 | 13,339 | 1.84× |
| **R2** | GEMM tiling to cuBLAS-class FP32 — 128×128 macro-tiles, 8×8 register blocking, double-buffered shared memory, 128-bit loads | 1,119 | 21,965 | 3.03× |
| **R3** | flash-attention dK/dV work-partition redesign | 943 | 26,052 | 3.59× |
| **R4** | fuse elementwise, layout, rotary and reduction into GEMM epilogues | 737 | **33,338** | 4.60× |
| **R5** | BF16 tensor cores — *die rating* | 627 | 39,191 | 5.40× |
| R5′ | BF16 tensor cores — *pessimistic (half-rate FP32 accumulate)* | 765 | 32,139 | 4.43× |
| **R6** | int8 weight-gradient GEMM on top of R5 | 605 | 40,600 | 5.60× |

The older ladder reached its former 30,000-token/s rung at R4. It does **not**
reach the revised 50,000 first gate: even its optimistic R6 estimate is 40,600.
The source-guided exact path must therefore remove more memory/control overhead
and establish a complete mixed-precision contract; the ladder is retained as a
decomposition, not presented as completion.

## The cooperative-accumulation gate is now measured

The physical RTX 4090 result rejects the ladder's binary full-rate/half-rate
model. FP32 accumulation is shape-dependent: 0.901x the F16-accumulate rate on
the foundation FFN-up shape, 0.613x on square 4096, and 0.791x on the foundation
LM head. The useful engineering comparison is nevertheless decisive: the
FP32-accumulate cooperative path delivered 101.6–118.7 TFLOP/s versus
20.4–20.8 TFLOP/s for selected portable FP32, or 4.99–5.81x per GEMM. Including
the current F32-to-F16 cast still delivered 84.7–101.0 TFLOP/s.

All four modes passed exact production-pattern oracles with maximum error zero.
Mixed precision therefore remains open and worth engineering; it is not
promoted to training until whole-step and trajectory parity clear. Evidence:
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-accum-physical-20260803-r3/`.

## Order of work

| # | Action | Cost | Why first |
|---:|---|---|---|
| 1 | Preserve and close the completed sentinel evidence | local | Do not rerun a quality-losing or mislabeled branch |
| 2 | Pin and audit the official `llm.c` source | local/source | Extract mechanisms without paying for a control run or replacing Helios |
| 3 | Implement the useful mechanisms in Helios | local | Start with same-mathematics changes to large tensors and the 84.59% critical path |
| 4 | Prove local correctness and preserve future physical gates | local | An implementation is not a measured throughput gain |
| 5 | Revisit approximate/multirate methods | later | Only after the exact path's remaining bottleneck is understood |

Current source map and implementation record:

- `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/LLMC-SOURCE-MECHANISM-MAP-2026-08-04.md`;
- `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X25-LLMC-DERIVED-EXACT-PATH-IMPLEMENTATION.md`;
- `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X26-DEVICE-RESIDENT-LAZY-TRAINING-LOSS.md`;
- `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X27-RESIDUAL-ADD-RMSNORM-FUSION.md`.

The first pass now implements four native exact-path mechanisms: combined
ordinary/masked training classifiers that avoid a full probability tensor, and
selective SwiGLU product rematerialization. The latter removes 1,215 MiB of
logical tape retention at the batch-10 shape for one extra elementwise pass
whose preserved 3090 price is about 4.418 ms. The classifier's scalar loss now
stays device-resident through backward graph construction, and tiny loss
scaling/addition no longer falls through to CPU solely because the tensor has
one element. Eligible intra- and cross-block residuals and following RMSNorms
now share exact two-output dispatches, eliminating 36 operations and 900 MiB
of logical forward traffic at the selected batch-10/18-layer shape. These are local implementations,
not new physical throughput results; Vulkan submission-boundary reuse and
complete-step performance remain explicit future gates.

### Current execution state

The direct split is now implemented without weakening the physical-device
guard. Helios measures synchronous completion wall time and the trainer emits
`host_build_ms`, `gpu_blocking_ms`, and `core_step_ms` beside
`dispatch_gpu_us`. Historical profile logs remain parseable. Local validation
is 109 suites / 233 executed tests passed / 55 physical-gated / 0 failed.

The paid discriminator has run. It required all production-pattern cooperative
GEMM oracles to execute with no skips, then compared resident-F16 FP32
accumulation, resident-F16 F16 accumulation, cast-inclusive cooperative
execution, and selected tiled FP32 on exact foundation matrix shapes. Evidence
and the runner live at:

- `scripts/bench-helios-coop-accum.mjs`;
- `scripts/run_helios_coop_accum_sweep.sh`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-accum-physical-20260803-r3/`.

The host split is physically complete on an NVIDIA L40S without changing the
shape or arithmetic. Excluding the first warm step, the exact foundation graph
spent **3,216.2 ms (68.49%)** in host-side build/lifecycle work and **1,479.4 ms
(31.51%)** blocked on the GPU. Timestamped dispatch was **1,470.6 ms**, so the
independent GPU clocks and the wall-clock partition agree. The measured
zero-host-overhead ceiling on this host/device is therefore about **3.19x**.
This confirms the ladder's host-bound mechanism, although the raw L40S rate is
not a substitute for the 4090 baseline.

A ten-step Node CPU profile localizes the removable part of host time. `flush`
is dominated by the genuine synchronous GPU wait and must not be counted as
free speedup. The large non-GPU self-time is instead native buffer lifecycle:
`createFreshBuffer` accounts for 15.230 sampled seconds and
`processPendingDestroys` for 13.629 seconds. Over ten steps the exact-size,
individual-allocation policy created 8,981 buffers and destroyed 8,026. The
next discriminator is therefore a 2x2 physical allocator experiment (coarse
versus exact size classes; temp slabs versus individual allocations), before
attempting full static graph replay.

The first allocator factorial has now banked an end-to-end gain. On the exact
L40S graph, native temporary slabs raised the warm median from **5,234 to 6,509
tokens/s (1.244x)** relative to exact-size individual allocations. Host build
fell from 3,210.9 to 2,307.9 ms while GPU blocking stayed flat at roughly 1,480
ms. With slabs enabled, coarse and exact size classes were effectively tied
(6,509 versus 6,497 tokens/s), so the gain is not padding or arithmetic—it is
driver-memory lifecycle avoidance. The remaining run still created/destroyed
roughly eight to ten thousand `VkBuffer` objects, and 6,471 temporary requests
fell back outside the 8 GiB slab arena. The next bounded sweep increases arena
coverage and retained output capacity before implementing graph replay.

The 50-step stability sweep narrowed the safe candidate. **12 GiB temp slabs +
48 retained large outputs per size class** completed cleanly at a **9,585
tokens/s warm median** on the L40S, about **1.83x** the original 5,234-token/s
exact/individual policy on that host. The 8 GiB/64-output arm also completed at
9,325 tokens/s. The faster 16 GiB policy is rejected for now: it segfaulted
after completing all steps in two separate runs. That failure is evidence, not
an acceptable teardown quirk. No policy becomes a default until it reproduces
on the cheaper RTX 3090 target and passes a longer parity run.

The fundamental track is now explicit in
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X17-ONE-HUNDRED-WAYS-TO-REIMAGINE-TRAINING.md`.
It maps 100 bodies of knowledge to falsifiable Alpha experiments and proposes a
Behavioral Constraint Compiler: closed-loop, counterexample-driven selection
of the smallest synthetic curriculum that changes desired behavior. This
attacks tokens and updates while graph compilation attacks milliseconds.

Evidence:

- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-host-split-physical-l40s-20260803-r1/`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-host-cpu-profile-l40s-20260803-r1/`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-host-cpu-profile-l40s-20260803-r2/`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-allocator-factorial-l40s-20260803-r1/`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-allocator-capacity-l40s-20260803-r1/`;
- `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-allocator-stability-l40s-20260803-r1/`.

## Parity gates — how a rung is earned

R1–R5 preserve exact arithmetic and must clear the project's existing promotion rules:

- exact train loss at every aligned step;
- held-out validation parity;
- gradient-norm and clipping-coefficient parity;
- finiteness and zero allocator overflow;
- kernel telemetry proving the candidate actually dispatched;
- sustained warm end-to-end rate, not a single step;
- the full physical suite passing.

R6 changes the arithmetic, so its gate is **held-out loss parity** rather than bit parity. That is permitted only because it has already been measured closed-loop: int4 block-floating-point weight gradients trained to within 0.0005 nats of FP32 over a full proxy run, with all four reduced-precision arms inside 0.007 nats.

## The rule that keeps this honest

**Tokens/s is gameable.** Lower precision, a smaller batch, a shorter context, or a cheaper approximation can all raise it while making the model worse or the run longer. So:

> A tokens/s number obtained by changing *what* is computed is not a throughput result. It belongs to the parent cost-to-behaviour goal and must be argued there, on held-out loss.

Throughput is a means. The parent goal — dollars to a fixed behavioural target — is the end, and the two can diverge.

## Not on this ladder, and why

| Excluded | Reason |
|---|---|
| Any attention replacement | Attention is 10.8% of arithmetic at S=1,024; ceiling 1.12× even if free |
| Low-rank or structured weights | Measured at 92.8% of full rank; factorisation costs **1.34× more** than dense |
| Weight-gradient token subsampling | Dominated by simply halving the batch (retracted 2026-08-03) |
| Gauge-quotient projection | Gradients are *exactly* orthogonal to gauge orbits by invariance — a theorem, not a measurement |
| Strassen / fast matrix multiplication | 1.31× fewer FLOPs at worse arithmetic intensity; FLOPs are not the binding constraint at 5.74% of peak |

Each of these was closed by measurement or proof, and each re-opens only if its stated mechanism changes.

## Definition of done

The first gate is sustained warm median >= 50,000 complete training tok/s. The goal
is done only at >= 64,000 complete training tok/s on one physical RTX 3090 for the
exact foundation computation, with parity or matched-loss evidence preserved and
checksum-verified on the mounted drive, reproduced from the selected source commit,
and the pod removed after recovery unless it is actively running the accepted job.
