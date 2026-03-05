# Super Chat 12k Run Analysis (2026-03-05)

## Run identity

- Run ID: `super_chat_20260305150148_iz8f`
- Host: L4 (`alpha-bench-l4-coopdbg-20260228084511`)
- Start checkpoint: `runs/super_chat_rca_resume5600_nopurge_20260305_144259/checkpoint-8400.json`
- Final step: `12000` (completed)
- Config style: conservative tail LR, grad-norm gating disabled, high live-alloc thresholds, no purge storms

## Completion and reliability

- Training reached step `12000` and emitted `training_complete`.
- Event counts:
  - `training_started=1`
  - `eval_summary=18`
  - `checkpoint_saved=18`
  - `samples_generated=18`
  - `train_status_warning=8`
  - `spike_skip=0`
  - `gpu_mem_purge=0`
  - `training_complete=1`
- Interpretation:
  - The stability path worked operationally: no spike-skip storms, no purge-trigger collapse, no allocator-cap crash.
  - Discord/reporting path stayed active (samples were generated each eval/sample interval and no remote errors were logged).

## Loss and validation behavior

- Metric range in this run:
  - Steps: `8401 -> 12000` (`n=3600` points)
  - `loss_first=4.751509`
  - `loss_last=4.756973`
  - `loss_min=4.636910`
  - `loss_max=5.022566`
- Smoothed trend:
  - Start avg (first 200): `4.718169`
  - End avg (last 200): `4.711426`
  - Delta: `-0.006743` (very small improvement)
  - Last-1000 linear slope: `-0.00000131` loss/step (near-flat)
- Validation:
  - `val_first=4.700218 @ step 8600`
  - `val_best=4.700218 @ step 8600`
  - `val_last=4.725533 @ step 12000`
  - Net: slight degradation from best as training continued.

## Throughput / runtime profile

- `ms_per_iter_median=82.956`
- `ms_per_iter_p90=115.103`
- `tokens_per_sec_median=49375.6`
- `tokens_per_sec_p10=35588.7`
- `tokens_per_sec_p90=53826.1`

This indicates acceptable throughput consistency for a long resumed tail run.

## Output quality signal (samples)

Latest samples are still mostly nonsensical/subword-noise (despite low-ish loss):

- Prompt: `"<|user|> Tell me about yourself. <|assistant|>"`
- Output excerpt: `"... fs toaere iinas ewt eefre, of aoin ..."`

Additional prompts also show degraded coherence and character artifacts.

## Root-cause conclusions (this run + preceding RCA chain)

1. Structural instability (crashes/stalls) was mitigated.
- Resume mismatch guard + allocator cap raise + no-purge spike path removed catastrophic failure modes.

2. Optimization objective is now the blocker, not runtime stability.
- The model converges to a plateau around `~4.70-4.75` and does not produce coherent chat outputs.

3. Current 1.33M architecture + current data/tokenizer/training objective is underfitting for quality.
- Low movement in smoothed loss, slight val regression, poor sample quality at 12k strongly indicates capacity/objective bottleneck.

## What this means for the 3.0-3.5 target

With current setup (4L/128D/4H, current corpus + objective), this run does **not** indicate a path to `3.0-3.5` by simply extending steps.

Most likely required to move further:

- Increase model capacity (at least 6L/192D class) while preserving the now-stable runtime controls.
- Improve tokenizer/data alignment for chat structure quality (special-token behavior and text normalization checks).
- Add quality-linked eval metrics (repetition/degeneracy and bucketed val loss by sample length) as hard stop criteria.

## Artifacts

- Long run log and RCA timeline:
  - `docs/super-chat-training-run-log.md`
- Trainer fixes:
  - `packages/train/src/trainer.ts`
- Native allocator cap fix:
  - `packages/helios/native/helios_vk.c`

