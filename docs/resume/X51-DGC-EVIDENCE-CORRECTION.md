# X51 — correction to X49's evidence, and a measurement trap for whoever validates it

**Date:** 2026-08-04 · **Evidence:** E0 source + preserved logs. Free.

## Correction

X49 stated "every recorded profile shows dgc=0" as evidence the DGC path never fires.
That observation is true but **confounded**: `PROFILE_GPU_TIMESTAMPS` disables DGC at
backend.ts:1697 (`!PROFILE_GPU_TIMESTAMPS && this.dgcReady && ...`). Every preserved
`dgc=0` reading — 4 in the transposed-A benchmark set — comes from a **timestamped**
run, where DGC is off by construction. Non-timestamped sustained runs do not emit the
`[gpu_ops]` line at all, so they provide no reading either way.

**So the profiles are not evidence for X49's premise.**

## What the premise actually rests on, and it still holds

1. **Structural code argument.** The DGC fast path requires *every* op in a flush to be
   eligible. A real flush carries ~243 mixed ops (matmuls, reductions, elementwise), so
   `allEligible` cannot be true. This is independent of any profile.
2. **Direct measurement.** A non-timestamped probe of a 40-op eligible run followed by
   one non-eligible op: `dgcFlushes=0` without the split, `dgcFlushes=1` with it.

X49's change and its parity validation are unaffected. Only the evidence sentence was
overstated, and it is corrected here rather than quietly amended.

## The trap this creates

**A timestamped profile cannot validate X49.** Timestamping disables DGC, so a profiled
run will report `dgc=0` and show none of X49's effect — a false negative that looks like
"the change does nothing".

Validation on hardware must use a **non-timestamped sustained run** and read
`dgcFlushes` from `getGpuStepStats()`, comparing against
`HELIOS_DISABLE_DGC_SPLIT_RUNS=1` as the control.

Expected on the real graph (replayed from the preserved trace): **+2 flushes per step**
against a 5–7 baseline, routing **127 ops (7.5% of the graph)** through DGC.

## Generalisable

Instrumentation that changes the code path it measures will hide the thing being
measured. X8 flagged this for timestamps perturbing scheduling; this is the stronger
form — timestamps *disable a feature outright*. Before trusting any profile, check what
the profiler switches off.
