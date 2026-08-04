# X50 — the gradient norm is ~383 operations, and the 127-add run is its reduction tree

**Date:** 2026-08-04 · **Evidence:** E0/E1 source audit + preserved trace. Free.

## Finding

`HeliosBackend.totalSumOfSquares` (backend.ts:3050) computes the global gradient norm as:

1. `sumOfSquares` per parameter tensor — 128 calls, ~2 ops each for large tensors
   (`sum_sq_reduce_stride` + `sum_reduce`) → **~256 ops**
2. a **pairwise tree** over the 128 scalar partials: 64 + 32 + 16 + 8 + 4 + 2 + 1
   = **127 separate `add` dispatches**

Total ~383 operations, **22.5% of the 1,703-op graph**, to produce one scalar.

This identifies the 127-op `add` run X40 found at operation 1189: it is not gradient
accumulation, it is the gradient-norm reduction tree. X49 now routes that exact run
through DGC as a single submit.

## The remaining inefficiency

127 dispatches to sum 128 floats. Each partial is a separate 1-element buffer, so the
tree exists only because the scalars are not contiguous.

**Fix:** have `sumOfSquares` write into a slot of one shared partials buffer, then a
single `sum_reduce` over 128 contiguous elements. **127 dispatches → 1.**

Requires an output-offset parameter on the reduction path; `acquireOutputRegion`
currently allocates a fresh region per call.

## Status

Not implemented. X49 already removes most of the *host* cost of these 127 ops by
batching them into one DGC submit, so the marginal value of the contiguous-buffer fix
is now lower than the raw op count suggests — it should be priced against the
post-X49 profile, not the pre-X49 one. That is the X42 error and it must not repeat.

The ~256 per-tensor sum-of-squares ops are untouched by both and remain the larger half.
