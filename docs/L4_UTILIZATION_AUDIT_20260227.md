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

## Validation
- Build: `npm run build -w @alpha/helios`
- Benchmark: `scripts/run-compiled-benchmark.sh 100`
- Inference sanity: 3 compiled-binary prompts completed successfully.

Latest benchmark snapshot in this loop:
- avg tok/s: `2457.343`
- speedup vs previous stable benchmark: `+11.25%`

## Remaining Gaps (Next Loops)
1. No padded coop path for batched matmul (currently 2D only).
2. No cooperative path yet for `matmulTransposedA` kernel family.
3. Tile-size policy for non-coop matmul is static (`16`/`32` threshold by `M*N`) and not yet shape+device auto-tuned.
4. No runtime telemetry for “coop hit rate” per step; adding this would quantify tensor-core usage directly.
