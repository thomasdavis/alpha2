# Flash Attention Coop2 Checkpoint (L4, Driver 590)

Date: 2026-03-04

## Goal

Win the forward key on L4:

- `flash_attn_fwd_b1_h16_t512_d64`

CUDA reference on this host class is typically around `0.126-0.128 ms` for this key.

## What Changed In This Pass

Primary file edited:

- `packages/helios/src/kernels/attention-coop2.ts`

### 1) Cheaper O normalization

- Replaced per-PV-tile matrix division (`O / l`) with:
  - one cooperative-matrix reciprocal compute (`lInv = 1 / l`)
  - per-tile multiply (`O * lInv`)

This removes repeated matrix-division work in the output endcap.

### 2) LSE writer-lane gating

- Kept only one writer lane per row (`myColInRow == 0`) for LSE writes.
- Moved `m/l` scratch loads and `log` computation inside that writer-only branch.
- Non-writer lanes no longer perform wasted scratch loads / log math.

### 3) Endcap ordering

- Moved LSE scratch/materialization block after O normalize/store path so the `m/l` barrier is no longer in front of O writeback for the common in-bounds path.

### 4) LSE log path variant

- Switched writer-lane LSE log from:
  - `log(l)`
- to:
  - `log2(l) * ln(2)`

Kept because it did not regress in current host runs and aligns with existing exp2 usage style.

## Environment Used For Bench

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
HELIOS_FLASH_FWD_PREFER_COOP2=1
HELIOS_FLASH_COOP2_SCOPE=workgroup
HELIOS_FLASH_COOP2_LS=128
HELIOS_FLASH_COOP2_QT=2
HELIOS_FLASH_COOP2_F16_INPUT=1
HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0
HELIOS_FLASH_COOP2_DOUBLE_BUF=0
```

Host:

- `alpha-bench-l4-coopdbg-20260228084511`
- NVIDIA L4
- Driver `590.48.01`

## Key Results

### A) Focused CUDA compare (`bench-ops`) with larger warmup

Command:

```bash
npx tsx scripts/bench-ops.ts \
  --iters=20 --warmup=40 \
  --only=flash_attn_fwd_b1_h16_t512_d64,flash_attn_coop2_fwd_sc_b1_h16_t512_d64,flash_attn_coop2_probe
```

Representative runs in this pass:

1. `flash_attn_fwd_b1_h16_t512_d64 = 0.105 ms` vs CUDA `0.127 ms`  -> Helios win
2. `flash_attn_fwd_b1_h16_t512_d64 = 0.121 ms` vs CUDA `0.126 ms`  -> tie/near win
3. `flash_attn_fwd_b1_h16_t512_d64 = 0.111 ms` vs CUDA `0.128 ms`  -> Helios win
4. `flash_attn_fwd_b1_h16_t512_d64 = 0.133 ms` vs CUDA `0.127 ms`  -> tie/slight loss

Takeaway: with higher warmup, Helios is now in the tie/win band for this key on this host.

### B) Decision-grade repeat harness (`bench-flash-repeat`)

Command:

```bash
npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6 --order=alternate
```

Observed in this pass:

- `flashAttentionCoop2` median: `0.0984 ms`
- `flashAttention` median: `0.0984 ms`
- wrapper delta median: ~`0.0002 ms`

Note: this harness is excellent for intra-Helios A/B and regression detection; keep using it for kernel iteration.

## Reliability Notes

Two things were consistently true in this pass:

1. Warmup depth matters a lot for flash-vs-CUDA row stability.
2. Short warmup (`6`) produced frequent pessimistic Helios rows in `bench-ops`.

Recommended policy for publishable flash-vs-CUDA comparison:

- Use `bench-ops` with `--warmup=40` for this key.
- Run at least 3 independent repetitions and report median-of-medians.
- Keep `scope=workgroup`, `LS=128`, `QT=2`, `F16_INPUT=1` fixed while comparing.

## Current Best Known Flash Config (This Host)

- coop2 preferred forward
- `scope=workgroup`
- `LS=128`
- `QT=2`
- `BC=16`
- f16 input conversion enabled
- LSE writes enabled (correctness path)
- double-buffering disabled for this key (`_db` did not consistently improve default row)

## Remaining Work

Even with tie/win samples, run-to-run spread still exists.

Next structural item (if more headroom needed):

- true load/compute overlap for KV staging (subgroup role split) while preserving current winner config.

But for the current benchmark key, this pass moved Helios into practical parity and repeated wins under reliable warmup.
