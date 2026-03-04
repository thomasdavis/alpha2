# Flash Attention Coop2 Checkpoint (L4, Driver 590)

Date: 2026-03-04

## Goal

Beat CUDA on:

- `flash_attn_fwd_b1_h16_t512_d64`

CUDA reference on this host class is typically `0.126-0.128 ms` for this key.

## What Changed In This Loop

Primary code changes:

- `packages/helios/src/kernels/attention-coop2.ts`
- `packages/helios/src/backend.ts`

### 1) Coop2 constant-softcap path: precompute scale*invSoftCap once

In `attention-coop2.ts`:

- Added one precomputed scalar in main:
  - `scaleInvSoftCap = scale * constInvSoftCap` (only when `useSoftCapConst`)
- Updated per-element callbacks (`scaleSoftCap`, `scaleMask`, `scaleMaskCausalNoOob`) to use that precomputed value in const-softcap variants.
- Rewired callsites to pass `scaleInvSoftCap` instead of `scale` for const-softcap branches.

Intent:

- remove one multiply from hot per-element softcap math in constant-softcap kernels (`_sc30`).

### 2) Default path wrapper tax reduction

In `backend.ts` (`flashAttention`):

- Added `_flashFwdCoop2Ready: boolean | null`.
- Replaced per-call `try/catch` with:
  - fast direct call when readiness is known good,
  - single probe attempt when unknown,
  - cached scalar fallback when known bad.

Intent:

- avoid paying JS exception-guard overhead on every timed forward call once coop2 has been proven healthy.

### 3) Documentation cleanup

Removed ad-hoc temporary docs and kept only this checkpoint as the single status artifact.

## Current Benchmark Protocol (Reliable)

### Primary decision-grade protocol

Use repeat harness:

```bash
HELIOS_WG_SIZE=128 \
HELIOS_FLASH_FWD_PREFER_COOP2=1 \
HELIOS_FLASH_COOP2_QT=2 \
HELIOS_FLASH_COOP2_SCOPE=workgroup \
HELIOS_FLASH_COOP2_LS=128 \
HELIOS_FLASH_COOP2_F16_INPUT=1 \
HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6 --order=alternate
```

Decision metric:

- median + p90 (not single best sample)

### Cross-stack comparison protocol

Use focused `bench-ops` with higher warmup:

```bash
HELIOS_WG_SIZE=128 \
HELIOS_FLASH_FWD_PREFER_COOP2=1 \
HELIOS_FLASH_COOP2_QT=2 \
HELIOS_FLASH_COOP2_SCOPE=workgroup \
HELIOS_FLASH_COOP2_LS=128 \
HELIOS_FLASH_COOP2_F16_INPUT=1 \
HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  npx tsx scripts/bench-ops.ts --iters=20 --warmup=40 \
    --only=flash_attn_fwd_b1_h16_t512_d64,flash_attn_coop2_fwd_sc_b1_h16_t512_d64,flash_attn_coop2_probe
```

## Results From This Loop

## A) Decision-grade repeat harness (latest)

Command:

```bash
npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6 --order=alternate
```

Latest medians:

- `flashAttention` (default path): `0.108859 ms`
- `flashAttentionCoop2` (direct): `0.108287 ms`
- `probe_qk`: `0.076977 ms`
- `probe_pv`: `0.072334 ms`
- wrapper delta (`default - direct`) median: `0.000731 ms`

Interpretation:

- default path and direct coop2 are now effectively at parity in steady repeat runs.
- this is below the CUDA `~0.127 ms` reference for this key.

## B) Focused cross-stack run (bench-ops, warmup=40)

Command:

```bash
npx tsx scripts/bench-ops.ts --iters=20 --warmup=40 --only=flash_attn_fwd_b1_h16_t512_d64,flash_attn_coop2_fwd_sc_b1_h16_t512_d64,flash_attn_coop2_probe
```

Observed:

- `flash_attn_fwd_b1_h16_t512_d64 = 0.120 ms`
- `CUDA = 0.128 ms`
- `flash_attn_coop2_fwd_sc_b1_h16_t512_d64 = 0.104 ms`

Result:

- Helios win (`1.07x`) on this run.

## C) Warmup-6 bench-ops still shows variance

With `--warmup=6`, focused `bench-ops` can still fluctuate and produce slower rows (for example around `0.148 ms` vs CUDA `0.127 ms`) despite the same kernel path.

Working conclusion:

- short warmup run quality is not decision-grade for this key.
- use repeat harness medians and/or `bench-ops --warmup=40` for publishable comparisons.

## Current Best Known Config (L4)

- `HELIOS_FLASH_FWD_PREFER_COOP2=1`
- `HELIOS_FLASH_COOP2_QT=2`
- `HELIOS_FLASH_COOP2_SCOPE=workgroup`
- `HELIOS_FLASH_COOP2_LS=128`
- `HELIOS_FLASH_COOP2_F16_INPUT=1`
- `HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0` (correctness path)
- `HELIOS_FLASH_COOP2_DOUBLE_BUF=0`

## Status

For `flash_attn_fwd_b1_h16_t512_d64` on L4/driver 590:

- Helios is now in the CUDA-beating regime under decision-grade repeat benchmarking and wins in focused `bench-ops` with adequate warmup.
- Remaining issue is warmup-6 variability in `bench-ops`, not core coop2 kernel throughput.

## Next Steps

1. Keep using `bench-flash-repeat` (7x40x6, alternate order) as the kernel regression gate.
2. Keep `bench-ops --warmup=40` as the cross-stack publish metric for this key.
3. If additional margin is needed: implement overlap (`_db` role-split loader/compute), but only after proving a net gain against this checkpoint baseline.
