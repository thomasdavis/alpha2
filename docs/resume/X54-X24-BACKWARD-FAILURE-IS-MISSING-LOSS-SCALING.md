# X54 — X24's cooperative-backward quality failure is missing loss scaling

**Date:** 2026-08-04 · **Evidence:** E2 measurement on the real checkpoint + E0 source audit. Free.
**Significance:** X52 established that unlocking cooperative backward is worth **1.31x -> 4.23x** on
arithmetic — the single largest lever in the program. This identifies why it fails, and the fix already
exists in the codebase.

## The measurement (X53)

Dynamic range of the operands each GEMM family consumes, on the real checkpoint over held-out text:

| Operand family | median \|x\| | p1 \|x\| | % subnormal in FP16 | % flush-to-zero |
|---|---:|---:|---:|---:|
| forward activations | 2.131e-01 | 3.569e-03 | 0.874% | 0.0020% |
| forward weights | 2.818e-02 | 5.202e-04 | 0.148% | 0.0001% |
| **backward grad-outputs** | **4.725e-07** | **3.363e-09** | **99.518%** | **15.147%** |

FP16's smallest normal is 6.104e-5. Backward grad-outputs sit at **4.7e-7 — two orders of magnitude
below it**, and are **4.5 x 10^5 times smaller** than forward activations.

**99.5% of backward operands land in FP16 subnormal range, and 15.1% flush entirely to zero.**

That is not approximation error. It is destruction of roughly a seventh of the gradient signal, every
step, which is exactly the shape of X24's failure: a path that looks fine briefly and then reverses
after step 125 as the damage accumulates.

## The cause

`packages/train/src/trainer.ts:1573`

```ts
const useLossScaling = !!deps.mixedPrecision;
```

Loss scaling — the standard remedy, present and working in this codebase, auto-tuning from 128.0 with
reduction tracking — is gated on the **`mixedPrecision` flag**.

The cooperative backward path is enabled independently (`HELIOS_ENABLE_COOP_BACKWARD=1`) and performs
its FP16 conversion **tile-locally inside the shader** (X23: FP16 conversion with FP32 global storage).
So it introduces FP16 casts on a run where `mixedPrecision` is false, and therefore **runs with
`lossScale = 1.0`**.

> The cooperative path created a new FP16 cast site without connecting it to the machinery that exists
> to protect FP16 cast sites.

## The fix

Drive loss scaling from *whether anything casts to FP16*, not from the `mixedPrecision` flag alone —
i.e. `useLossScaling = mixedPrecision || coopBackwardEnabled`.

Sanity check on magnitude: at the existing initial scale of 128, grad-outputs move from 4.7e-7 to
6.0e-5 — just at the normal boundary. The auto-tuner would need to climb to roughly **2^16** to place
them near 3.1e-2, comfortably mid-range. The existing tuner increases until overflow is detected, so
this should happen automatically, but the **starting point and climb rate should be checked** rather
than assumed.

## Why this reorders the program again

X52 showed cooperative backward is worth ~3.2x on arithmetic and that nothing else measured is within
an order of magnitude. This turns that from an open research problem — where the handoff proposed
sentinel correction and low-rank residual correction — into **wiring up existing machinery**.

Both proposed corrections may now be unnecessary. They were designed to compensate for a *numerical
error in the product*; the measurement says the error is in the *input representation*, upstream of
the product, and has a standard cheap remedy.

## What must be verified before believing this

1. **Re-run X53 at an early checkpoint.** These figures come from a trained model. Early gradients are
   larger, and X24's reversal appeared after step 125 — the trend across training matters more than
   the absolute numbers.
2. **Confirm the cooperative backward path really runs with `lossScale = 1.0`** by instrumenting the
   value on a run with `HELIOS_ENABLE_COOP_BACKWARD=1` and `mixedPrecision` false.
3. **Then re-run X24's comparison with loss scaling active.** The prediction is falsifiable and sharp:
   the 10,330 tok/s backward path should stop reversing after step 125 and track the forward-only
   trajectory. If it still reverses, the diagnosis is wrong and the residual-correction work is back on.

None of steps 1–2 needs hardware. Step 3 does.
