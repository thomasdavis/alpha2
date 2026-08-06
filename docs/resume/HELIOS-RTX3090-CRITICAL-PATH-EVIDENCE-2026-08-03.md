# Helios RTX 3090 critical-path and dKV tile evidence

**Date:** 2026-08-03  
**Hardware:** NVIDIA GeForce RTX 3090, 24,576 MiB, Vulkan driver 570.124.04  
**Base repository revision:** `cded24a0b526cb7ab31a1afc2ac55b7fb019a4f8`  
**Artifact root:** `/mnt/donto-data/donto-resources/benchmarks/alpha-helios-3090-kernel-profile-20260803-r7/`  
**Scope:** exact 97,098,880-parameter foundation training step at batch 10, context 1,024, scalar FP32  

## Outcome

The first exact RTX 3090 per-kernel profile moves the binding constraint back from host allocation to four device
kernels. On the second, warmed timestamped step, the three generic GEMM layouts plus attention dKV backward
account for 84.59% of measured dispatch time. None of the first two follow-up candidates is selected:

- transposed-B `R42CK32` is numerically close but slightly slower in-kernel and neutral end to end;
- the pre-existing four-query dKV-v2 kernel is numerically valid but 4.50x slower than selected dKV.

A dKV tile sweep then exposed a correctness bug in the experimental non-square configurations. The old kernel
used the key-tile ordinal as the query-tile ordinal and loaded only one query row per invocation. That identity is
valid only when `Br == Bc`. Correcting causal block indexing and staging all `Br` rows restores exact learning
trajectory behaviour, but removes the apparent speedup. The selected 32 x 32 dKV tile remains fastest.

This is a useful negative result and a correctness improvement, not the requested 50,000 tokens/s breakthrough.
The strongest verified 3090 window remains 6,551.38 tokens/s, leaving a 7.632x gap to 50,000 tokens/s.

## Warmed kernel breakdown

The selected K16 baseline's warmed step measured 1,233.4973 ms of dispatch time:

| Rank | Kernel | Calls | Time | Dispatch share |
|---:|---|---:|---:|---:|
| 1 | `matmul_transposed_R42C` | 91 | 312.7565 ms | 25.36% |
| 2 | `matmul_transposed_a_R42C` | 91 | 286.4044 ms | 23.22% |
| 3 | `matmul_R42` | 91 | 230.7820 ms | 18.71% |
| 4 | `flash_attn_bwd_dkv_32_32_64` | 18 | 213.4468 ms | 17.30% |
| 5 | `flash_attn_bwd_dq_32_16_64` | 18 | 40.0794 ms | 3.25% |

Measured operation families were 69.48% matrix multiplication and 24.54% attention backward. This ranking is
the current constraint ledger: further allocator work cannot by itself produce the next large multiplier.

## R42CK32: physically rejected

K32 doubles the coalesced transposed-B reduction tile, halves load/barrier rounds, and raises shared memory from
4 KiB to 8 KiB. It had already passed an awkward-shape local numerical smoke. The 3090 result is:

| Measurement | K16 control | K32 candidate | Decision |
|---|---:|---:|---|
| warmed target-kernel time, 91 calls | 312.7565 ms | 315.9571 ms | candidate 1.02% slower |
| warmed total dispatch | 1,233.4973 ms | 1,238.1222 ms | candidate 0.37% slower |
| sustained samples after warmup | 21 | 21 | matched |
| sustained mean | 5,279.91 tok/s | 5,306.18 tok/s | +0.4975%, noise-sized |
| sustained median | 5,337.75 tok/s | 5,347.04 tok/s | +0.1740%, noise-sized |
| maximum loss difference | - | `4.3392e-5` | not exact |
| maximum gradient-norm difference | - | `3.3260e-3` | not exact |

The uninstrumented result is too small and inconsistent with the direct kernel timer. It also changes reduction
order enough for trajectories to diverge after several updates. It remains opt-in and is not added to the
foundation launcher.

## dKV-v2: decisively rejected

The pre-existing dKV-v2 generator batches four query rows in one loop body to expose instruction-level
parallelism. A new opt-in selector made it physically testable without changing the default path.

| Measurement | selected dKV | dKV-v2 | Change |
|---|---:|---:|---:|
| warmed target-kernel time, 18 calls | 213.4468 ms | 959.8356 ms | +349.7% |
| warmed total dispatch | 1,233.4973 ms | 1,938.5758 ms | +57.16% |
| step-2 loss | 9.062634468078613 | 9.062634468078613 | exact |
| step-2 gradient norm | 11.065296... | 11.065296... | exact at log precision |

The extra register pressure and generated code dominate any loop-level ILP benefit on this device.

## Non-square dKV tiles: false speedup and corrected boundary

Before the correctness repair, 64 x 32 and 32 x 16 appeared dramatically faster, at 118.3317 and 147.7972 ms.
Both produced `NaN` gradient norms and step-2 loss 9.5455 instead of 9.0626. They were skipping required query
rows, not computing the same operation faster.

Two generator defects were corrected:

1. the first causal query block is `floor(kBlockOff / Br)`, not `kBlockIdx`, when tile widths differ;
2. a `Bc`-wide workgroup must cooperatively load `Br / Bc` query rows, not leave `Br - Bc` shared rows unwritten.

After repair:

| Tile `(Br,Bc)` | Warmed dKV | Loss / gradient | Relative to 32 x 32 |
|---|---:|---|---:|
| `(32,32)` | 213.4468 ms | finite, selected trajectory | baseline |
| `(64,32)` | 240.9761 ms | finite, selected trajectory | +12.90% |
| `(32,16)` | 305.4776 ms | finite, selected trajectory | +43.12% |

The generator now rejects configurations where `Br` is not an integer multiple of `Bc`; dKV-v2 remains on its
square-tile contract. The final default regression exited cleanly, reproduced loss 9.062634468078613 and gradient
norm 11.065, and measured selected dKV at 213.3248 ms.

## Experimental hygiene

- The pod ran from 21:37:19.664Z and was observed at 22:14:45Z: 2,245.336 measured seconds, or USD 0.137215 at
  USD 0.22/hour. Deletion then succeeded and `runpodctl pod list -o json` returned `[]`.
- The fitting inputs retained their frozen hashes; all runs use seed 42 and identical model/data/tokenizer flags.
- Full timestamp profiles, matched sustained metrics, incorrect configurations, repaired configurations, and the
  final default regression are all retained. Failed/false-positive branches were not deleted.
- The first two timestamp processes completed their metrics but returned 139 during native teardown on driver
  570.124.04. Later profiles and the final regression exited zero. The teardown fault is preserved rather than
  recast as a successful process exit.

## Next constraint

The top three GEMM layouts remain 67.28% of warmed dispatch time. Reaching 50,000 tokens/s cannot come from dKV
tile tuning or another allocator percentage point. The next faithful gates should address the arithmetic path:

1. measure BF16/FP16 tensor-core accumulation against the exact behavioural/numerical contract;
2. build a device-fingerprint autotuner for the three dominant GEMM layouts;
3. test fused GEMM epilogues only where the trace proves a removable global-memory round trip;
4. pursue persistent subgraphs only after measuring whether dispatch/control time remains material inside the
   post-allocation step;
5. keep algorithmic token reduction separate from raw training-token/s so the 50k claim remains honest.
