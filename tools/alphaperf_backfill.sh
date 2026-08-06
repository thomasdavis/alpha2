#!/usr/bin/env bash
# Backfill alphaperf.db with the measured history of the 30k program.
#
# Every row here was measured on the RunPod 3070 during the session that added
# it, at the commit named. Re-running this is idempotent for the commit_log,
# finding, gate, kernel, gemm and isa tables carry an append per run — so run it
# ONCE per fresh DB (after `alphaperf.py init`), not repeatedly.
#
# The baseline commit is aecaa40 (start of the 2026-08-06 session); the numbers
# below are what that session measured.
set -euo pipefail
cd "$(dirname "$0")/.."
A="python3 tools/alphaperf.py"

# ---- end-to-end gate, before and after the session --------------------------
$A gate aecaa40 native 19116 --batch 24 --median-ms 80.4 --gpu-ms 75.9 --loss 9.5818 --held-gb 4.22 --note "baseline, start of 2026-08-06 session"
$A gate aecaa40 vulkan 11084 --batch 6 --median-ms 34.6 --loss 9.5236 --note "baseline"
$A gate 50849af native 19453 --batch 24 --median-ms 79.0 --gpu-ms 74.7 --loss 9.5818 --note "SHFL warp reduction (first butterfly)"
$A gate bb11fe1 native 19988 --batch 24 --median-ms 76.8 --gpu-ms 72.2 --loss 9.5818 --held-gb 4.23 --note "second butterfly cross-warp"
$A gate 0d0a87d native 19729 --batch 24 --median-ms 77.9 --loss 9.5818 --note "session end (run-to-run spread ~1-2pct from 210MHz idle clock)"
$A gate 0d0a87d vulkan 11991 --batch 6 --median-ms 32.0 --loss 9.5236 --note "session end"

# ---- isolated kernel microbenchmarks, [1536,640] unless noted ---------------
# micro-norm-bandwidth.mjs, 345 GB/s elementwise control on the same card
$A kernel aecaa40 layerNorm 1536x640 65.1 120.9 --pct 35 --note "shared-mem tree, before SHFL"
$A kernel aecaa40 rmsNorm 1536x640 37.2 211.6 --pct 61 --note "before SHFL"
$A kernel aecaa40 layerNormBackward 1536x640 174.2 67.7 --pct 20 --note "before SHFL"
$A kernel aecaa40 softmax 15360x64 43.1 182.4 --pct 53 --note "before SHFL"
$A kernel 50849af layerNorm 1536x640 56.4 139.5 --pct 40 --note "first butterfly (warp only)"
$A kernel 50849af rmsNorm 1536x640 33.5 234.5 --pct 68 --note "first butterfly"
$A kernel 50849af layerNormBackward 1536x640 157.0 75.1 --pct 22 --note "first butterfly"
$A kernel 50849af softmax 15360x64 31.1 252.7 --pct 73 --note "first butterfly"
$A kernel bb11fe1 layerNorm 1536x640 34.4 228.9 --pct 66 --note "second butterfly — AT ROOFLINE (control 34.2us)"
$A kernel bb11fe1 rmsNorm 1536x640 26.7 294.2 --pct 85 --note "second butterfly"
$A kernel bb11fe1 layerNormBackward 1536x640 110.8 106.5 --pct 31 --note "second butterfly"
$A kernel bb11fe1 softmax 15360x64 32.6 241.4 --pct 70 --note "second butterfly"
# the non-GEMM half, confirmed at roofline (why fusion is the only remaining lever there)
$A kernel bb11fe1 gelu 1536x2560 85.0 370.0 --pct 100 --note "at roofline"
$A kernel bb11fe1 geluBackward 1536x2560 117.0 402.0 --pct 100 --note "at roofline"
$A kernel bb11fe1 transpose 24x64x10x64 28.0 281.0 --pct 81 --note "at roofline but PURE OVERHEAD — foldable into GEMM addressing"
$A kernel bb11fe1 addInplace 512x2560 46.0 342.0 --pct 99 --note "at roofline"

# ---- GEMM rate probes (probe-gemm-rate.mjs, M=1536, L2-evicted, pool-filled) -
$A gemm bb11fe1 "qkv B^T"    1536 1920  640 nt 1 20.91 180 --note "projection, L2-resident"
$A gemm bb11fe1 "qkv fwd"    1536 1920  640 nn 1 18.88 200
$A gemm bb11fe1 "mlp fc B^T" 1536 2560  640 nt 1 20.48 246
$A gemm bb11fe1 "mlp fc fwd" 1536 2560  640 nn 1 19.22 262
$A gemm bb11fe1 "lm head B^T" 1536 12288 640 nt 1 21.60 1118
$A gemm bb11fe1 "attn proj fwd" 1536 640 640 nn 1 15.47 81
$A gemm bb11fe1 "mlp fc dW"  640 2560 1536 ta 1 17.88 282 --note "weight gradient"
$A gemm bb11fe1 "qkv dW"     640 1920 1536 ta 1 17.63 214
$A gemm bb11fe1 "attn proj dW" 640 640 1536 ta 1 12.63 100
# the batched attention gap — nt is 2.2x slow at identical shapes
$A gemm bb11fe1 "attn qk nn" 64 64 64 nn 240 3.38 37 --blocks 240
$A gemm bb11fe1 "attn qk nt" 64 64 64 nt 240 1.48 82 --blocks 240 --note "2.2x SLOW — staging request count, fix=stage 32k"
$A gemm bb11fe1 "attn qk ta" 64 64 64 ta 240 3.45 36 --blocks 240

# ---- commits, with what each moved -----------------------------------------
$A commit 50849af "capture SHFL, reduction stops using barriers for its warp" --before 19116 --after 19453
$A commit 0f52874 "probe: batched transposed-B is 2-3x slow, the staging request count" --note "diagnosis only, no code change to kernel"
$A commit 8f1a6d1 "ISA coverage register — a missing encoder is an absent thought" --note "user's explicit stubs ask"
$A commit 80282d0 "capture cp.async — LDGSTS, LDGDEPBAR, DEPBAR" --note "gate for f16-in-memory GEMM"
$A commit bb11fe1 "reduction cross-warp step is a second butterfly, not a walk" --before 19453 --after 19988
$A commit 0d0a87d "record cp.async hardware finding — bit-correct is not wired"

# ---- findings and REFUTATIONS (a dead lever must stay dead) -----------------
$A finding reduction "SHFL was missing — every reduction ran a shared-mem tree with a block barrier per step" --value "layerNorm 65->34us" --status confirmed
$A finding reduction "non-GEMM half is AT ROOFLINE (gelu 370, geluBwd 402, addInplace 342 GB/s vs 345 control)" --value "fusion is the only remaining non-GEMM lever" --status confirmed
$A finding gemm "step is now cleanly GEMM-bound" --value "69% of GPU" --status confirmed
$A finding attention "batched transposed-B (Q@K^T) is 2-3x slow: 4 staging requests/warp vs 1" --value "~2.1ms, fix=stage 32k tile" --status todo
$A finding cpasync "cp.async ENCODED bit-for-bit but hardware probe read ZEROES — async scoreboard not wired by encoding alone" --value "decode ptxas pipeline control fields first" --status todo
$A finding gemm "the 30k path: cp.async+f16-in-memory cut non-tensor staging instrs -> tensor-bound -> f16-accumulate finally pays (was 4pct because not tensor-bound)" --status inprogress
$A finding gemm "f16 accumulate alone worth ~4pct not a factor — kernel is not tensor-pipe-bound (issue-bound on staging)" --value "4pct" --status refuted --note "refuted as a standalone lever"
$A finding gemm "bigger batch LOWERS GEMM rate (operands leave cache): m1536 19.2, m6144 17.1, m12288 10.9 TFLOP/s" --status refuted --note "no shape at 100M sustains more than m1536"
$A finding gemm "split-K net negative — emulated at equal arithmetic, +8pct time" --status refuted
$A finding gemm "flash attention worth ~3pct not 12 — score matrix is 16KB/head at seq64" --value "3pct" --status refuted

# ---- ISA coverage snapshot -------------------------------------------------
$A isa HMMA encoded "tensor cores, 45.5 TFLOP/s from registers"
$A isa LDSM encoded "ldmatrix, whole fragment per shared load"
$A isa SHFL encoded "warp reductions without a barrier (closed 2026-08-06)"
$A isa RED encoded "RED.E.ADD.F32 scatter, embedding gradient"
$A isa LDGDEPBAR encoded "cp.async commit_group"
$A isa DEPBAR encoded "cp.async wait_group N"
$A isa "LDG.E.128" captured "encoded, no caller — 128-byte staging request"
$A isa LDGSTS captured "cp.async — encoded, hardware scoreboard not yet wired"
$A isa HFMA2 missing "packed f16 FMA — precondition for f16-in-memory"
$A isa LDGSTS_pipeline missing "the cp.async multi-stage control-field discipline"
$A isa FSETP missing "float compare to predicate"
$A isa I2FP missing "int->float, so a count can scale inside a kernel"

echo
echo "backfill complete:"
$A latest

# ---- the operation universe (the registry now lives IN this DB) -------------
$A op-import gpu-op-universe/catalog/operation-registry.json

# the primitives + kernels built this session, along the implementation ladder
$A op hephaestus.warp.shfl optimized --ref hephaestus/sm86_mem.c:hp_shfl --commit 50849af --note "warp shuffle; reduction rewritten on it"
$A op hephaestus.memory.ldgsts tested --ref hephaestus/sm86_mem.c:hp_ldgsts --commit 22c53e0 --note "cp.async; hardware-validated copy primitive"
$A op hephaestus.memory.ldgdepbar encoded --ref hephaestus/sm86_mem.c:hp_ldgdepbar --commit d23e5ce
$A op hephaestus.memory.depbar encoded --ref hephaestus/sm86_mem.c:hp_depbar --commit d23e5ce
$A op prometheus.reduction.warp-reduce optimized --ref prometheus/reduction.c:pr_emit_tree_warp_reg --commit bb11fe1 --note "two-butterfly warp reduction; layerNorm 65->34us at roofline"
$A op prometheus.gemm.cpasync-f16-nt tested --roofline 28 --ref prometheus/hmma.c:emit_hmma_cpasync_f16 --commit 1317004 --note "cp.async f16 GEMM NT single-buffered; correct 128x64x64; not yet measured"
$A op prometheus.gemm.hmma-staged measured --tflops 21 --roofline 28 --ref prometheus/hmma.c:emit_hmma_staged --commit aecaa40 --note "shipping staged GEMM; 15-21 TFLOP/s, issue-bound on staging"

# the autoresearch loop's experiment memory
$A experiment "Warp-shuffle reduction removes the block barriers that made layerNorm 5x off roofline" --op prometheus.reduction.warp-reduce --lever warp-shuffle --before 19116 --after 19988 --verdict confirmed --commit bb11fe1
$A experiment "f16-accumulate alone doubles the tensor ceiling so should be a factor" --lever f16-accumulate --verdict refuted --unit TFLOP/s --note "only 4pct because GEMM is issue-bound on STAGING not tensor-bound; pays only after cp.async"
$A experiment "Bigger batch amortises launches and fills the GEMM better" --lever batch-size --before 19 --after 17 --unit TFLOP/s --verdict refuted --note "rate FALLS with M as operands leave cache"
$A experiment "cp.async + f16 staging deletes 28 of 42 k-step instructions, doubling tensor fraction to 78pct" --op prometheus.gemm.cpasync-f16-nt --lever cp.async --verdict inprogress --commit 1317004 --note "emitter correct; perf pending double-buffer + measurement"
