# Standing goal — extreme performance by rethinking the problem

**Set:** 2026-08-03 · **Throughput target revised:** 2026-08-04 · **Status:** ACTIVE · **Owner:** ajax + agent
**Research home:** `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/`
**Primary document:** `REIMAGINING-ALPHA-PERFORMANCE-2026-08-03.md`

This goal runs alongside, and does not replace, the product goal in `GOAL.md`
(finish Alpha as a genuinely chatty, effective conversational model). Its purpose
is to make that product goal cheap enough to iterate on.

---

## The goal

> Reduce the GPU-dollar cost of reaching a fixed held-out loss on the frozen Alpha
> corpus by at least **10x**, by treating cost as a product of four independent
> factors and attacking all four with correctness-gated, physically measured
> experiments. Never by relaxing the quality contract, and never by promoting a
> candidate on theory alone.

```
                (FLOPs per token) x (tokens to target loss)
  cost($)  =    -------------------------------------------  x  ($ per second)
                        (FLOP/s actually achieved)
```

| Factor | Current | Set by | Status |
|---|---:|---|---|
| F1 FLOPs/token | 653.2 MFLOP | model shape | never varied |
| F2 tokens to target | 1,941,995,520 | schedule, optimizer, data, batch | never varied |
| F3 achieved FLOP/s | 4.74 T = **5.74% of FP32 peak** | kernels, precision, dispatch | varied within its bottom decile |
| F4 $/second | $0.000192 | rental market | never varied |

## Why this goal exists

The engine is **12x to 22x** away from what the RTX 4090 can deliver, the workload
is compute-bound by 3.6x so bandwidth is not an excuse, and **51% of the production
step is unattributed to any measured kernel**. Against that, the campaign has been
contesting 2-4% kernel candidates. The unit of progress was wrong.

## Sub-goal: throughput

A dedicated throughput goal is at
[GOAL-THROUGHPUT-2026-08-03.md](GOAL-THROUGHPUT-2026-08-03.md): raise complete
training throughput on one RTX 3090 from the current quality-bearing **7,762
tok/s** to a **50,000 tok/s first gate**, a **64,000 tok/s primary target**, and a
**70,000 tok/s stretch**. At $0.22/hour the primary target would execute the
1.942B-token contract in about 8.4 GPU-hours / $1.85 before evaluation and
checkpoint overhead. The 64k target is anchored by a public `llm.c` 124M BF16
report, but remains unproven for Alpha until Helios reaches it with matched
full-step accounting.

The cooperative-matrix discriminator is now physically measured. GeForce Ada's
FP32-accumulate rate is shape-dependent (0.61–0.90x the F16-accumulate rate),
but the current path is already 4.99–5.81x selected portable FP32 per GEMM and
4.16–4.94x including casts. Mixed precision therefore remains an active path,
subject to whole-step and training-trajectory parity.

## Gates

| Gate | Definition | State |
|---|---|---|
| **G0** | Physical baseline: exact FLOP accounting, roofline position, four-factor decomposition, reproducible from scripts. | **met 2026-08-03** |
| **G1a** | `host_build_ms` printed beside `dispatch_gpu_us` in the trainer, confirming or refuting the host-bound model of X8. **Run this first** — it changes how every candidate is scored. | **met 2026-08-03 on L40S: 68.49% host / 31.51% GPU blocking steady-state; GPU wall agrees with timestamped dispatch** |
| **G1** | Pin and audit the official high-throughput reference source, map its mechanisms to the exact Alpha graph, and implement compatible parts in Helios. No external control run is authorized. | active |
| **G2** | The 51% unattributed step interval explained and either eliminated or accounted for in a corrected ledger. **Diagnosed 2026-08-03 (X8): host-bound, unoverlapped, static graph rebuilt every step.** | diagnosed, unfixed |
| **G3** | >= 50,000 complete tok/s on one RTX 3090 for the exact foundation computation, with exact-loss or matched-loss parity under existing promotion rules. | open |
| **G3b** | >= 64,000 complete tok/s on one RTX 3090, with the full timed boundary and Helios source/mechanism provenance recorded. | open |
| **G4** | >= 10x reduction in GPU-dollars to a fixed held-out loss, demonstrated on a bounded pilot with matched tokens, not extrapolated. | open |
| **G5** | Foundation run executed under the improved recipe with complete mounted evidence. | open |

**Order.** G1a first (free, local, ~1 h), then G1 (~$0.70) — the two cheapest
experiments in the program, and together they price every Tier-1 item. Kernel work
resumes only after both, and then against a known target instead of against the
previous kernel.

G1a and the cooperative-accumulation discriminator are both physically
complete. The cooperative test ran on an RTX 4090; the unchanged exact
foundation graph required a higher-memory L40S for the host/GPU split. After
the first warm step, host build/lifecycle averaged 3,216.2 ms and synchronous
GPU blocking averaged 1,479.4 ms. The independent timestamped dispatch sum was
1,470.6 ms, which closes the accounting and confirms that the missing interval
is host-side rather than hidden GPU execution. A ten-step CPU profile then
localized the actionable self-time to repeated Vulkan buffer creation and
destruction. The immediate experiment is allocator-policy elimination; static
graph replay remains the structural R1 destination.

The first 2x2 allocator experiment recovered **1.244x end to end** without
changing computation: 5,234 to 6,509 warm-median tokens/s on the exact L40S
graph. Native temp slabs removed about 903 ms of host time per step, while GPU
blocking remained unchanged. Size-class rounding had no material effect once
slabs were active. This validates allocator lifecycle as a causal part of G2
and opens a narrower capacity sweep: the 8 GiB arena still fell back 6,471
temporary requests and the static graph still created/destroyed thousands of
buffer handles per run.

A 50-step follow-up selected the bounded stable policy rather than the fastest
short arm. The 12 GiB arena / 48-large-output policy exited cleanly at 9,585
warm-median tokens/s on L40S; the 8 GiB / 64-output policy exited cleanly at
9,325. The 16 GiB policy segfaulted after completing in two runs and is not
eligible. The L40S pod was terminated. Further algorithm tests should use an
RTX 3090 when one is available; raw rates remain device-labelled.

**RTX 3090 reproduction, batch-10 proxy (2026-08-03).** A community 3090 at
$0.22/hour became available and was bound to a three-hour auto-termination.
The same 18-layer, d=640, FFN=1728, context-1024 graph was run at batch 10 so it
fit the card's 24 GiB; this is not labelled as the exact batch-24 foundation
shape. Exact individual allocation achieved 3,711 warm-median tok/s. An 8 GiB
temporary slab plus 64 large outputs per size class achieved **5,807 tok/s
(1.565x)**. GPU blocking stayed nearly fixed (1,185.3 vs 1,205.9 ms) while host
build/lifecycle fell from 1,573.3 to 562.7 ms, reproducing the allocator's causal
mechanism on the operator-selected cheap device. All four arms exited cleanly,
but twelve steps are not a stability promotion. Evidence is mounted at
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-3090-portfolio-20260803/`.

**AOT prerequisite refined (RTX 3090 r3/r4).** The r3 ordered signature correctly
showed that complete event streams differed, but r4 localized every first mismatch
to a flush boundary. After removing flush events and their shifted order counters,
all four traced steps contain the exact same ordered 1,703-operation sequence.
The operation topology is static in this bounded batch-10 test; the flush schedule
is dynamic because allocation pressure inserts waited submissions at different
points. Replay therefore remains viable, but it must separate a compiled operation
blueprint from allocator-dependent submission partitioning, or first replace the
dynamic lifetime system with a deterministic arena plan.

The cost model has also been widened beyond optimizing SGD as given. X17 in the
mounted research tree reduces the task to behavioral construction, surveys 100
external mechanisms, and makes the closed-loop Behavioral Constraint Compiler
the primary L2/L3 research candidate alongside the L5 static graph/arena work.

**Source-guided exact-path implementation (2026-08-04).** Official `llm.c`
source was pinned at `f1e2ace651495b74ae22d45d1723443fd00ecd3a` and used as a
mechanism oracle without launching a control or GPU run. Helios now has combined
ordinary/masked training classifiers and selective SwiGLU product
rematerialization. The classifier loss also stays device-resident until after
backward graph construction, removing a forced forward submit-and-wait boundary;
microbatch loss scaling and accumulation now remain on-device as intended.
Residual additions and their immediately following RMSNorms now share exact
two-output dispatches at all eligible intra- and cross-block boundaries,
removing 36 graph operations and 900 MiB of logical forward traffic at the
selected batch-10/18-layer shape.
The grouped QKV-to-flash boundary now unpacks Q/K/V, writes head-major layout,
and applies Q/K RoPE in one exact dispatch with paired inverse-layout gradient
kernels. Static accounting removes another 180 complete-step dispatches and
8.789 GiB of logical activation traffic at that same selected shape.
That boundary and Flash Attention are now one autograd operation: its backward
consumes dQ/dK/dV together and writes one complete grouped gradient, eliminating
three padded branch tensors and two additions. This removes another 72
dispatches and 10.547 GiB versus X28; cumulatively the pair removes 252
dispatches and 19.336 GiB versus the pre-X28 boundary.
Closed-form savings are 1,440 MiB of classifier traffic per
training call and 1,215 MiB of logical activation retention across 18 layers at
batch 10. Local gradients and release/rematerialization behavior pass; physical
VRAM and complete-step effects remain unmeasured. Canonical record:
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X25-LLMC-DERIVED-EXACT-PATH-IMPLEMENTATION.md`,
with the two-output fusion recorded separately in
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X27-RESIDUAL-ADD-RMSNORM-FUSION.md`
and the grouped layout fusion in
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X28-QKV-HEAD-LAYOUT-ROPE-FUSION.md`,
with the one-tape combined backward in
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/X29-COMBINED-QKV-FLASH-BACKWARD.md`.

**Portfolio obligation added 2026-08-03.** The operator wants every one of X17's
100 directions to receive a faithful attempt, not only the agent's preferred
candidates. X18 defines the evidence ladder and machine-auditable SQLite ledger.
Every direction receives a mechanism/prior-art audit and a direction-appropriate
cheap discriminator; only survivors advance to bounded RTX 3090 tests. A proof,
trace, offline replay, or controlled proxy may faithfully close a direction when
it reaches the mechanism's risky prediction. Discussion alone never counts as an
attempt. The operator-supplied X19 atlas adds a separate 100-item contract and
state namespace for concrete Autonomic Dataflow mechanisms. X17 and X19 therefore
form 200 traceable research objects: they may reuse an instrument or physical run,
but each retains its own evidence and verdict. Current generated state is at
**X17: 96 queued / 4 cheap tests complete; X19: 98 queued / 1 designed /
1 cheap test complete** at
`/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/PORTFOLIO-STATUS.md`.

**Third result already banked** — from preserved logs, no new runs
(`X8-THE-MISSING-HALF-OF-THE-STEP.md`):

3. **Helios appears host-bound and unoverlapped.** Only 5–7 command submissions
   and 1–3 waits per step, so the missing half is *not* submission overhead. The
   step's operation topology is now directly observed as stable over the four r4
   traces, although memory-pressure flush placement remains dynamic. The graph is
   still rebuilt in TypeScript every iteration, with 687 allocator slab fallbacks
   per step. Step time looks like `host_build + gpu_execute`, not `max(...)`.
   **This caps kernel-only work at roughly 2x** and explains why kernel swaps win
   2–4% while gradient-ownership forwarding — the one change that removed
   *operations* rather than kernel time — won 48.6%.

## Results already banked (measured, not predicted)

Two changes to the **training contract** — no kernel work, no new mathematics:

1. **Batch size 24 is 2.3x larger than the measured gradient noise scale**
   (B_simple = 10,674 tokens). Batch 10 reaches the same loss with **1.65x less
   total arithmetic**: ~29 GPU-hours and ~$20 off the run. Batch 24 was chosen
   because 32 exhausted the allocator; the allocator has no opinion about
   convergence.
2. ~~Weight gradients from 12.5% of batch tokens → **1.35x**~~ **RETRACTED
   2026-08-03 (X12).** X5 measured the noise floor with a *symmetric* relative
   error while X9/X11 measured theirs *asymmetrically* against the exact
   gradient, and the two were compared. A noisy estimate has the larger norm, so
   the symmetric denominator flattered every approximation. On one consistent
   metric the floor is **0.844** — the error from simply halving the batch — and
   **no sampling rate stays under it** (uniform @0.5 = 1.41x the floor;
   importance-sampled @0.5 = 1.13x). Sampling `dW` is dominated by using a
   smaller batch, which also saves the forward pass and `dX`.
   *General lesson:* before proposing a cheaper estimator, check it against the
   trivial estimator that uses less data.

5. **Low precision survives that correction and is strengthened (X9/X12).** On
   the corrected metric, stochastic-rounded **int4 block floating point adds
   0.269x the error of halving the batch** (bf16 0.007x, int8 0.016x, fp8-e4m3
   0.106x, int3 0.584x). And the key objection — that quantisation bias may not
   average away over 79,020 steps where sampling noise does — was measured:
   across all nine formats, the residual after averaging 16 stochastic-rounding
   draws divided by the single-draw error is **0.2479–0.2502** against a
   theoretical 1/sqrt(16) = 0.2500. **Stochastic rounding leaves no detectable
   systematic component, down to int2.** Round-to-nearest is smaller per step but
   has no such decay — which is exactly the trap.

   **CLOSED-LOOP CONFIRMED 2026-08-03 (X13).** Trained at proxy scale with
   forward and `dX` exact and only `dW` from quantised operands, matched tokens
   and data order: fp32 7.0567 · bf16 +0.0031 · int8 +0.0003 · **int4 -0.0005**
   · int3 +0.0066. **All four arms within 0.007 nats of FP32**; int4 is
   indistinguishable. Per-step error did not compound. The FP32 control
   reproduced X7's independent run to four decimals, validating the harness.
   Remaining risk is horizon: 250 steps against a 79,020-step contract (316x).

7. **Curvature-aware precision allocation beats uniform (X14).** Testing an
   external proposal that damaging error is `e^T H e` not `||e||_2`, via the
   K-FAC factorisation `H ~ A (x) B` (so `||E||^2_H = tr(B E A E^T)` is exact):
   asymmetric 8/3 bits on the top 12.5% of coefficients by curvature gives
   curvature error 0.0092 at 3.62 avg bits, versus uniform int4's 0.0254 at 4.00
   bits — better, at fewer bits. Wins both valid matched-budget comparisons.
   The quantisation error is itself roughly isotropic (1.124), so the case rests
   on the *gradient's* anisotropy, which the stable-rank measurement confirms.

6. **Norm-importance sampling of `dW` confirmed (X11).** `p_t` proportional to
   `||delta_t|| * ||x_t||` (the Frobenius norm of an outer product factorises)
   gives a clean, rate-independent **1.57x variance reduction**. Correct and
   useful; does not rescue result 2.

4. **Muon reaches the same held-out loss with 2.24x fewer tokens** (X7/X7b).
   Matched-token, matched-data-order comparison at an 8M-parameter proxy scale,
   with the learning rate swept on **both** arms until each optimum was interior
   (6 AdamW arms, 3 Muon arms). Every Muon arm beat every AdamW arm; Muon's loss
   was flat across a 10x LR range, which also removes the per-shape LR pilot this
   project currently pays for. Proxy scale — a direction and a sign, not a
   transferable multiplier. **Do not multiply this with result 1**: both act on
   tokens-to-target and may share a mechanism. F2 is worth 2.2x–3.7x; one
   matched-token pilot varying both together resolves where.

## Closed by proof, not measurement

**Gauge-Quotient Muon (X15).** Proposed: Muon's spectral flattening may promote
numerical noise in functionally dead parameter directions into full-sized
updates, so project the gauge out first. Measured two exact continuous
symmetries (SwiGLU up/down scaling; per-head V/O rotation) on the real
checkpoint:

| | gradient | quantisation error | null |
|---|---:|---:|---:|
| FFN scaling gauge | **0.000%** | 0.097% | 0.098% |
| V/O rotation gauge | **0.00x fair share** | 1.00x fair share | — |

The gradient figure is **exactly zero, and forced**: if the loss is invariant
along a symmetry orbit, the directional derivative along any tangent of that
orbit is identically zero, so the gradient is always exactly orthogonal to the
gauge tangent space. The proposed mechanism therefore cannot operate through the
gradient; the only channel is numerical error, which measures as perfectly
isotropic (1.00x fair share) in a subspace occupying 0.098% (FFN) and 3.08%
(V/O) of the parameters.

This closure is stronger than the empirical ones — it will not become true at
another scale or checkpoint. It also self-validates: a mis-derived symmetry
would have produced a visibly nonzero gradient projection rather than exactly
zero, and a unit test confirmed a pure gauge direction projects to 1.000000.

Note the corrected threshold: "<1% of energy" was the wrong null. A gauge
subspace of dimension d inside ambient D captures d/D of any isotropic vector's
energy for free; the meaningful quantity is enrichment above that.

## Directions closed by measurement — do not re-propose

Each was tested against the real trained checkpoint on held-out text.

| Direction | Why it is closed |
|---|---|
| Linear / oscillator / state-space attention as a **retrofit** | Attention is high-rank (684/1024 at 1% error) and its within-row logit spread is 10.0 against the Alman–Song threshold of 2.63 — **3.8x into the provably hard regime**. |
| FMM / multipole / sparse attention at S=1024 | Far-field blocks need rank 40/128; 37% of mass is in the far field. Measured 1.53x on attention, **1.04x end to end**. |
| Any attention replacement at S=1024 | Attention is **10.8% of arithmetic**; the ceiling is 1.12x. Becomes worthwhile only at S >= 16k. |
| Low-rank / structured weight matrices | Trained matrices are at 92.8% of full rank; factorisation **costs 1.34x more** than dense. Training filled the budget it was given. |

The same measurement that closes low-rank weights **opens** mixture-of-experts:
a capacity-saturated model wants more parameters at constant FLOPs per token.

## Published

The open mathematical problem space — 54 precisely posed questions across ten families, each anchored
to a measured constant, with the four closed directions stated as boundary conditions carrying their
crossover conditions — is public at:

**https://alpha.donto.org/research/alpha-open-problems-2026-08-03.html**

An external critique from the Harmonic GPT research program prompted the retraction above,
three new measurements, and accepted reformulations of Q10, Q20, Q21, Q23, Q31, Q36-37 and
Q44. Reply published at:

**https://alpha.donto.org/research/alpha-response-to-harmonic-2026-08-03.html**

A complete program briefing — everything we are doing, written for an agent who has never seen
it, with no questions in it — is at:

**https://alpha.donto.org/research/alpha-program-briefing-2026-08-03.html**

Source markdown is linked from the page and mirrored in the research tree. Served from
`/srv/alpha-research` via a `handle_path /research*` block in the `alpha.donto.org` vhost, ahead of the
`:3104` app proxy so reports stay up independently of the workbench.

## Working rules

- Save all research under the mounted research tree; never only in `/tmp` or scrollback.
- Device-independent claims get measured on CPU here, for free, before any pod is rented.
- Report genuinely interesting or strategy-changing findings to Discord; skip routine noise.
- A rejected idea stays rejected unless its **mechanism** changes.
- No promotion from theory, from llvmpipe correctness, or from one favourable sample.
