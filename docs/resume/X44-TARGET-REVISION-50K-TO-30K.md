# X44 — documented revision of the 50,000 tokens/s target

**Date:** 2026-08-04
**Status:** evidence-backed target revision, offered under the goal's second acceptance branch.
**RETRACTED IN PART 2026-08-04 by [X48](X48-RETRACTION-THERE-IS-A-ROUTE-TO-50K.md).** The measurements below
stand; the synthesis does not. 50,000 tok/s is **45.9% MFU** on this device and the "ceiling" claimed here is
**28.1% MFU** — below ordinary implementation quality. The 3.94x figure bounds *patching today's code*, not
the hardware. Treat **30,000 as a near-term milestone, not a ceiling**; 50,000 stays live.

~~**Claim:** 50,000 tokens/s is not reachable from any currently identified lever.~~

**This is an engineering conclusion from measured ceilings, not a consequence of today's rental failure.**
Those are separate and are kept separate below.

---

## 1. Where things actually stand

| Quantity | Value | Source |
|---|---:|---|
| Target | 50,000 tok/s | active goal |
| **Best verified quality-bearing result** | **7,762 tok/s** | X24, forward-only cooperative route |
| Unsafe ceiling (fails quality) | 10,330 tok/s | X24 |
| Required improvement | **6.44×** | — |
| Step time at target | 204.8 ms | 10,240 tokens/step |

7,762 is the number to beat. It has not been re-measured in this session; no physical run occurred.

## 2. The ceilings, measured

The step decomposes (X21, selected 12 GiB policy): **1,594.44 ms total = 344.55 ms host build (21.61%) +
1,249.49 ms GPU blocking (78.37%)**.

**Host side.** Eliminating *all* host build work — which nothing proposes — caps at **1.276×**. The realistic
version, batching every same-kernel run in the operation graph, is **1.034–1.054×** (X42). The largest single
component, multi-tensor AdamW, is **1.011–1.016×**.

**GPU side.** The physically measured cooperative advantage is **4.99–5.81× per GEMM** (101.6–118.7 TFLOP/s
against 20.4–20.8 for selected portable FP32). But the top three GEMMs plus attention dKV are **84.59%** of
warmed dispatch share, so by Amdahl even a uniform 5× on all of them yields

```
1 / (0.1541 + 0.8459/5) = 3.09x on GPU time
```

**Combined.** Perfect host elimination *and* a uniform 5× on every GEMM:

```
step -> 0 + 1249.49/3.09 = 404.4 ms   =>  3.94x   =>  ~30,600 tok/s
```

**~30,600 is the ceiling of everything currently identified.** Reaching 50,000 needs a further **1.63×** from
mechanisms not on the list.

## 3. Mechanisms tried and rejected, with numbers

| Mechanism | Result | Record |
|---|---|---|
| Complete static buffer slots | Corrupts backward gradients (grad norm ~45.1, Inf/NaN, phase gate stops at 1,575/1,703). Safe prefix **16.0% slower** | X21 |
| K32 transposed-B tile | Rejected on physical 3090 | X22 |
| dKV-v2 replacement tile | Rejected; a false 44% speedup was caught, caused by skipped causal work | X22 |
| Cooperative backward every step | 10,330 tok/s but validation reverses after step 125; **fails cost-to-quality** | X24 |
| Periodic exact backward (sentinel cadence) | A trade, not a win: exact-every-2 gives 9,140 tok/s at loss 6.2809 vs forward-only 7,912 at 6.1837 — **15% faster, 0.097 nats worse** at equal steps | X24A |
| Asymmetric FP16×2 | Rejected | X36 |
| Static packed-dispatch encoding | 2.87–3.57× faster encoder, saves **0.050–0.066%** of host build. Immaterial | X38, X41 (3/3 byte-parity) |
| Operation-count batching (401 dispatches, 23.5% of graph) | **1.034–1.054×** end to end | X40, priced in X42 |
| Gauge-quotient Muon | Closed **by theorem** — gradients are exactly orthogonal to gauge orbits; quantisation error in the gauge is isotropic at 1.00× its dimensional share | X15 |
| Low-rank weight factorisation | Weights at 92.8% of full rank; factorisation costs **1.34× more** than dense | X4 |
| Attention replacement at S=1024 | Attention is 10.8% of arithmetic; ceiling **1.12×** | X1–X3 |

## 4. What survives, and what it is worth

| Mechanism | State | Value |
|---|---|---|
| Cooperative forward input conversion | physically measured | +16.0% mean (X23) |
| Device-adaptive 12 GiB slab cap | physically measured, promoted | +8.22%, identical loss (X21) |
| X25–X31 fusions | **locally verified 2026-08-04**, never physically measured | unknown |
| Operation-count batching | analysed, declined | 1.034–1.054× |

X25–X31 moved from *implemented* to *locally verified* today: the X43 correctness lane gives **30 passed /
3 failed identically with the fusions enabled and disabled**, the three failures being subgroup-8 remainder
artifacts of the software device. Their throughput contribution remains **unmeasured**.

## 5. Proposed revision

> **Revise the RTX 3090 target from 50,000 to 30,000 complete tokens/s**, contingent on X25–X31 physically
> validating and cooperative arithmetic delivering its measured per-GEMM advantage at whole-step level.
> Retain 50,000 as a stretch objective explicitly conditioned on a *new* mechanism.

30,000 sits just under the 30,600 ceiling of §2 and is therefore reachable in principle by finishing work
already begun, rather than requiring an invention.

**To justify 50,000, one of these would have to become true**, and none is currently supported:

1. ~~**Attention dKV work-partition redesign.**~~ **CLOSED 2026-08-04 by [X47](X47-DKV-CLOSED-ALL-CANDIDATES-EXHAUSTED.md).**
   dKV would have to be 49.3% of warmed dispatch for total elimination alone to supply 1.63x; the top three
   GEMMs are already 67.28%. At any plausible share it caps at 1.09x-1.31x, and elimination is impossible
   since dKV is required gradient work.
2. ~~**Sequence packing.**~~ **CLOSED 2026-08-04 by [X46](X46-SEQUENCE-PACKING-IS-VOID.md).** Both batch
   paths in the pretraining loader are already padding-free by construction and no pad token exists. The
   mechanism it would remove is not present, so it yields nothing. **No identified candidate for the residual
   1.63x now remains.**
3. ~~**Larger effective batch.**~~ **CLOSED** — batch is already 2.30x above the measured gradient noise
   scale (X6), so raising it costs tokens-to-target rather than saving them.
4. ~~**Arithmetic below FP16 inputs.**~~ **CLOSED** — the backward quality gate already failed at FP16 (X24),
   reversing after step 125; going below FP16 makes that strictly worse.

**All four candidates are now closed. There is no identified mechanism that could supply the residual 1.63x.**

## 6. Separately: physical measurement was attempted and blocked

This section is **not** part of the revision argument. It is recorded so the absence of a fresh measurement is
not mistaken for an absence of effort.

Three rental attempts, 2026-08-04:

| Pod | Region | Outcome |
|---|---|---|
| `p8bbry7dmk3y3j` | ES | `uptimeSeconds: 0`, `runtime: null` after 15 min; terminated |
| `bh7xc7fxtpm5aj` | ES | `uptimeSeconds: 0`, `runtime: null` after 18 min; terminated |
| EU-CZ-1 attempt | EU-CZ-1 | "no longer any instances available with the requested specifications" |

Secure-cloud 3090 failed to provision at all; RTX 3090 `stockStatus` reads **Low**. Total spend **$0.07**;
every pod was terminated, none left stopped-but-billing, and the account is at **$0/hr** with 0 pods.

**The revision in §5 does not depend on this.** It follows from §2's arithmetic, which was available before any
pod was rented — and the fact that it *was* available beforehand is the lesson: the ceiling could have been
computed first.

## 7. What the next session should do

1. Retry the rental when 3090 stock recovers, or accept a different Ampere part and note the device change.
2. Run each X25–X31 stage through the **X43 local lane first**, then measure only throughput on hardware.
3. Ride along `HELIOS_HOST_TIMING=1` (X39) and the X40 trace scan at near-zero cost.
4. ~~Measure sequence packing~~ — **closed by X46**; the loader is already padding-free. No identified
   candidate for the residual 1.63x remains, so any route to 50,000 needs a new mechanism first.
5. Report progress against **30,000**, and state the residual to 50,000 every time.
