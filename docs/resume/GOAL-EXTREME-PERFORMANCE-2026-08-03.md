# Standing goal — extreme performance by rethinking the problem

**Set:** 2026-08-03 · **Status:** ACTIVE · **Owner:** ajax + agent
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

## Gates

| Gate | Definition | State |
|---|---|---|
| **G0** | Physical baseline: exact FLOP accounting, roofline position, four-factor decomposition, reproducible from scripts. | **met 2026-08-03** |
| **G1a** | `host_build_ms` printed beside `dispatch_gpu_us` in the trainer, confirming or refuting the host-bound model of X8. ~1 hour, local, no GPU. **Run this first** — it changes how every candidate is scored. | open |
| **G1** | Reference-stack control run of the exact foundation shape on one 4090 (~$0.70, <=1 h), tokens/s and MFU recorded. Decides whether Helios is the production engine or the research engine. | open |
| **G2** | The 51% unattributed step interval explained and either eliminated or accounted for in a corrected ledger. **Diagnosed 2026-08-03 (X8): host-bound, unoverlapped, static graph rebuilt every step.** | diagnosed, unfixed |
| **G3** | >= 3x end-to-end over 7,253.8 tokens/s on the exact foundation shape, with exact-loss and gradient parity under existing promotion rules. | open |
| **G4** | >= 10x reduction in GPU-dollars to a fixed held-out loss, demonstrated on a bounded pilot with matched tokens, not extrapolated. | open |
| **G5** | Foundation run executed under the improved recipe with complete mounted evidence. | open |

**Order.** G1a first (free, local, ~1 h), then G1 (~$0.70) — the two cheapest
experiments in the program, and together they price every Tier-1 item. Kernel work
resumes only after both, and then against a known target instead of against the
previous kernel.

**Third result already banked** — from preserved logs, no new runs
(`X8-THE-MISSING-HALF-OF-THE-STEP.md`):

3. **Helios appears host-bound and unoverlapped.** Only 5–7 command submissions
   and 1–3 waits per step, so the missing half is *not* submission overhead. The
   step's op graph is **static** (1,444 identical ops on 17 of 20 steps) yet
   rebuilt in TypeScript every iteration, with 687 allocator slab fallbacks per
   step. Step time looks like `host_build + gpu_execute`, not `max(...)`.
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
