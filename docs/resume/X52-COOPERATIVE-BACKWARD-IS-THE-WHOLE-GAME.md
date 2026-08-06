# X52 — cooperative *backward* is the whole game; everything else is second-order

**Date:** 2026-08-04 · **Evidence:** E1 arithmetic over X1 accounting + trace audit. Free.
**Sharpens:** X48's budget, which assumed 5x on all GEMM without checking what cooperative can reach.

## The audit

Cooperative dispatch is gated by `coopShapeIsEnabled` (permissive by default —
`COOP_SHAPE_ALLOW.size === 0` allows everything), tile alignment, and
`canUseCoopMatmulDtypes`.

**Alignment is not the blocker.** Of 291 matmul operations in the preserved RTX 3090
trace, **273 (94%) have all logical dimensions divisible by 16**. The 18 that do not are
flash attention `[100, 1024, 64]`, where 100 = 10 batch x 10 heads.

So the real gate is dtype — and cooperative is **forward-only**, because X24 measured the
backward path failing cost-to-quality (10,330 tok/s but validation reverses after step 125).

## The consequence, and it is large

X1's accounting: linear forward is **29.7%** of step FLOPs, linear backward **59.4%** —
backward is *twice* the arithmetic of forward.

| Cooperative coverage | at 5x per-GEMM | at 7x per-GEMM |
|---|---:|---:|
| **Forward linear only** (today) | **1.31x** | **1.34x** |
| Forward + backward linear | 3.48x | 4.23x |

**Forward-only caps arithmetic gain at ~1.31x even with an infinitely good kernel**,
because Amdahl is dominated by the 59.4% of FLOPs that cannot use the path.

X48's budget assumed 5x across all GEMM and derived 3.09x on GPU time. That figure
**requires backward**. Without it the budget does not reach 30,000, let alone 50,000.

## What this reorders

The program's top priority is now unambiguous and quantified:

> **Making cooperative backward quality-safe is worth 1.31x -> 4.23x on arithmetic.
> Nothing else on the list is within an order of magnitude of that.**

For comparison, everything else measured this session:

| Lever | Worth |
|---|---:|
| **Cooperative backward unlocked** | **~3.2x** |
| dKV redesign (X47) | ≤1.31x, and impossible to fully realise |
| Operation-count batching (X42) | 1.03–1.05x |
| DGC run-splitting (X49) | host-side only, unmeasured |
| Sequence packing (X46) | 1.00x — void |

## The specific open mechanism

The handoff names "sentinel-corrected cooperative backward" as next high-upside. Note
carefully what has and has not been tried:

- **Tried and rejected (X24A):** periodic *exact* backward at cadence 2/4/8. Exact-every-2
  gives 9,140 tok/s at loss 6.2809 against forward-only's 7,912 at 6.1837 — 15% faster,
  0.097 nats worse at equal steps. A trade, not a win.
- **Untested:** **low-rank exact residual correction** — computing the backward GEMM
  cooperatively and correcting only the components that matter, rather than periodically
  recomputing the whole thing exactly.

The distinction matters: cadence-based correction pays full exact cost on some steps.
Residual correction pays a small cost on every step. They are different mechanisms with
different cost curves, and only the first has been falsified.

## Why the quality failure is plausible and where to look

X24's backward reversal after step 125 is consistent with FP16 range/precision loss in
gradient accumulation rather than in the product itself. Two things worth measuring
before designing a correction, both free on a real checkpoint:

1. the dynamic range of backward GEMM inputs versus forward — if backward operands span
   a wider exponent range, FP16 inputs lose more;
2. whether the divergence concentrates in specific layers or parameter families, which
   would make a *selective* rather than uniform correction sufficient.

Both are E2 measurements on stored gradients and need no hardware.
