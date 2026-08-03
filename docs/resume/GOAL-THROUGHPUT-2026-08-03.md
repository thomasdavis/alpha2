# Standing goal — maximise training throughput on the foundation shape

**Set:** 2026-08-03 · **Status:** ACTIVE
**Parent goal:** [`GOAL-EXTREME-PERFORMANCE-2026-08-03.md`](GOAL-EXTREME-PERFORMANCE-2026-08-03.md) (10× cost reduction)
**Derivation:** `donto-resources/research/alpha-helios-reimagined/experiments/x16_throughput_ladder.py`

---

## The goal

> Raise sustained training throughput on the **exact frozen foundation shape** from
> **7,253.8 tokens/s** to **30,000 tokens/s committed**, with a **45,000 tokens/s
> stretch** — every rung gated on numerical parity, and no rung earned by changing
> what is computed unless that change has already been shown loss-neutral.

| Target | tokens/s | vs today | GPU-hours for the 1.942B-token run | Cost at $0.69/h |
|---|---:|---:|---:|---:|
| Today | 7,254 | 1.0× | 74.4 | $51.31 |
| **Committed** | **30,000** | **4.1×** | **18.0** | **$12.41** |
| **Stretch** | **45,000** | **6.2×** | 12.0 | $8.27 |
| Hard BF16 roofline | 252,909 | 34.9× | 2.1 | $1.47 |

Shape is fixed throughout: 18 layers, d=640, 10 heads × 64, FFN 1,728, vocab 12,288, S=1,024, batch 24, 24,576 tokens/step.

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

**The committed target is reached at R4, in pure FP32, with no tensor cores and no change to the arithmetic.** That is deliberate: it does not depend on the one fact nobody has measured.

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
| 1 | Print `host_build_ms` beside `dispatch_gpu_us` in the trainer | **instrumentation complete 2026-08-03; physical value open** | Confirms or refutes the host-bound model that the whole ladder rests on |
| 2 | Microbenchmark the FP32-accumulate cooperative-matrix path | **complete 2026-08-03 on RTX 4090** | Shape-dependent 0.61–0.90x of F16 accumulation; 4.99–5.81x selected FP32, so mixed precision stays open |
| 3 | R1 — static-graph record and replay | days | Largest single rung (1.84×), and it makes every later kernel gain actually visible |
| 4 | R2 → R3 → R4 | — | Reprofile after each; the ordering may change once R1 lands |
| 5 | R5 / R6 | — | Only if step 2 says yes |

Nothing after step 1 should start before step 1 finishes. The ladder's arithmetic is only as good as the host/GPU split it assumes.

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

The host split is still open. Five exact-shape attempts on one RTX 4090 host
failed before step one because the driver's usable device budget is below the
full FP32 graph peak. Those failures diagnosed and removed a 53 MiB smoke-test
leak, exposed 704 MiB of temp-slab tail waste, and added exact buffer classes;
none changed the shape or arithmetic. The next probe moves the unchanged graph
to a higher-memory one-GPU device rather than weakening the measurement.

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

Sustained warm median ≥ 30,000 tokens/s on the exact foundation shape, with full parity evidence preserved and checksum-verified on the mounted drive, reproduced from the selected source commit, and the pod removed or actively running the accepted job.
