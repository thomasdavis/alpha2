# Flash Attention Forward Coop2 Final Checkpoint

Date: 2026-03-03

## Goal

Beat CUDA on the forward benchmark key:

- `flash_attn_fwd_b1_h16_t512_d64`

Reference target from CUDA runs on L4 is typically around `0.126-0.127 ms`.

## Where We Are Now

We are in the last microseconds.

- Best observed direct coop2 sample in stable settings: `0.127962 ms`
- Typical direct median in good repeat runs: around `0.1288-0.1292 ms`
- Typical default-path median remains higher due wrapper/integration overhead in many runs.

Net: direct coop2 can effectively tie CUDA on best samples, but we do not yet have a reliable median win over CUDA.

## Forward Path Work Completed

### Kernel architecture and feature usage

- Implemented and stabilized coop2 forward kernel in:
  - `packages/helios/src/kernels/attention-coop2.ts`
- Uses NV coop2 primitives:
  - `OpCooperativeMatrixReduceNV` for row reductions
  - `OpCooperativeMatrixPerElementOpNV` for scale/mask/exp callbacks
  - coop-matrix math for online softmax recurrence and PV accumulation
- Added softcap-aware variants and f16 input variants.
- Added `_nolse` variants for controlled endcap experiments.
- Added QT and LS variant support through naming/dispatch.
- Added scope variants (`wg` / `sg`) and fallback behavior.

### Backend and dispatch integration

- Forward routing prefers coop2 by default where eligible:
  - `packages/helios/src/backend.ts`
- Added telemetry hooks for flash dispatch path/kernel/pipeline behavior.
- Added and used probe entrypoints via backend:
  - `qk`, `qk_mask`, `qk_softmax`, `pv`
- Added env-driven tuning knobs and pipeline key specialization.
- Added per-backend caching of flash env knobs (reduces repeated parsing overhead in hot call paths).

### Benchmark tooling

- Added repeat microbench harness:
  - `scripts/bench-flash-repeat.ts`
- Supports alternating run order to reduce bias (`default-first` vs `direct-first`).
- Reports decision metrics:
  - median, p90, trimmed mean, best, worst
- Added optional kernel debug capture in repeat output.

## Regressions and Reverts (Important)

Multiple endcap micro-optimizations were tested and reverted because they regressed or were unstable:

- reciprocal-normalization substitution (`invL * O`)
- coop-space LSE composition single-store variant
- LSE log path substitution (`log2 * ln2`)
- several writer-lane/endcap reshuffles
- extra conditional fast-splits in LSE write path

Conclusion from these attempts: we are no longer getting reliable gains from local endcap instruction edits.

## Reliable Benchmark Protocol (Current)

This is the protocol we treat as decision-grade for this flash key.

### Environment

```bash
export HELIOS_FLASH_FWD_PREFER_COOP2=1
export HELIOS_FLASH_FWD_COOP2_STRICT=1
export HELIOS_FLASH_COOP2_F16_INPUT=1
export HELIOS_FLASH_COOP2_QT=2
export HELIOS_FLASH_COOP2_LS=128
export HELIOS_FLASH_COOP2_SCOPE=workgroup
export HELIOS_FLASH_COOP2_BC=16
export HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0
```

### Repeat microbench

```bash
npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6 --order=alternate
```

### One-shot sanity probe

```bash
npx tsx scripts/tmp_flashcmp.ts
```

### Acceptance style used

- Prefer median improvements, not best single sample.
- Require non-regressing p90 for accepted speed wins.
- Use alternating order to reduce call-order bias.

## Representative Results (Good Stable Run)

From repeat microbench with the config above:

- `flashAttention` median: `0.130660 ms`
- `flashAttentionCoop2` median: `0.129209 ms`
- `probe_qk` median: `0.103269 ms`
- `probe_pv` median: `0.075351 ms`
- wrapper delta median: `0.001584 ms`
- best direct sample in that run: `0.127962 ms`

## Key Interpretation

- Wrapper overhead is small but still present in many runs.
- Core kernel is very close to CUDA; the remaining gap is in the low microseconds and highly sensitive to runtime variance.
- We are at the point where measurement discipline and structural overlap/scheduling changes matter more than micro-op edits.

## Current Best Known Config for This Key

- coop2 path enabled and strict
- f16 input path enabled
- `QT=2`
- `LS=128`
- `scope=workgroup`
- `BC=16`
- `skipLseWrite=0` for correctness benchmarking

## Next Practical Direction

If we continue optimization, the next meaningful forward lever is structural overlap (loader/compute role split with double-buffered KV staging) with strict single-variable A/B validation. Micro endcap edits are no longer giving reliable wins.

