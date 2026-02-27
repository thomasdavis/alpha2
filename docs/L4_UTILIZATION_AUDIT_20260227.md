# L4 Utilization Audit (Helios/Vulkan)

Date: 2026-02-27

## Verdict
Not fully utilizing L4 tensor-core/cooperative-matrix capability before this loop.

## What Was Leaving Performance On The Table
1. Cooperative matmul required exact divisibility (`M%coopM==0 && N%coopN==0 && K%coopK==0`) in TS dispatch.
2. Native cooperative-matrix property selection used first-match order from the driver, which is not guaranteed to pick the highest-throughput shape.

## Changes Made In This Loop
1. Added a generic padded-coop 2D matmul path in Helios (`packages/helios/src/backend.ts`):
   - For large 2D GEMMs, if dimensions are non-aligned, pad A/B to cooperative tile sizes, run coop matmul, then slice output back to `[M,N]`.
   - Guarded by a strict overhead cap (`COOP_PAD_MAX_OVERHEAD=20%`) and minimum FLOPs threshold (`COOP_PAD_MIN_FLOPS=2,000,000`) to avoid regressions on small/awkward shapes.
   - Applies to both regular matmul and matmul-with-B-transposed.
2. Improved cooperative tile selection in native Vulkan probe (`packages/helios/native/helios_vk.c`):
   - Scans all valid `f16 x f16 -> f32` subgroup cooperative properties.
   - Picks best candidate by highest `M*N*K` score (tie-break by larger `M*N`), instead of first-match.
3. Added batched padded-coop matmul support:
   - Extended GPU `slice`/`scatterSlice` with 3D kernels (`slice_3d`, `scatter_slice_3d`) for on-GPU batched pad/crop.
   - Added batched coop padding path in matmul dispatch (`padded_batched` accounting).
4. Added direct cooperative path for `matmulTransposedA`:
   - New coop kernel variants: `transposed_a` and `transposed_a_batched`.
   - Dispatch now uses direct coop for aligned dimensions and keeps rewrite path as fallback.
5. Added coop-hit telemetry:
   - `HeliosBackend.getMatmulCoopStats()` reports direct/padded/rewrite dispatch counts and hit rate.
   - CLI train prints `coop_matmul: ...` summary after training.

## Validation
- Build: `npm run build -w @alpha/helios`
- Benchmark: `scripts/run-compiled-benchmark.sh 100`
- Inference sanity: 3 compiled-binary prompts completed successfully.

Latest benchmark snapshots in this loop series:
- `+11.25%` (`2cfbe7d`) — padded coop coverage expansion.
- `+0.82%` (`88acdb5`) — transposed-A coop rewrite route.
- `+4.17%` (`6c518ea`) — rewrite gating by viable coop coverage.
- `+4.96%` (current working tree) — batched coop padding + direct transposed-A coop + telemetry.

## Remaining Gaps (Next Loops)
1. Benchmarks in this workspace are on Intel iGPU; L4-specific gains still need direct measurement on GCloud L4.
2. Tile-size policy for non-coop matmul is static (`16`/`32` threshold by `M*N`) and not yet shape+device auto-tuned.
3. No per-op trace-level attribution for coop hit/miss reasons yet (only aggregate counters).
