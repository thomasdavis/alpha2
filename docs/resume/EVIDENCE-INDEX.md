# Canonical evidence index

This index answers “where is the proof?” without requiring a future session to search the whole data
disk.

The mandatory mounted-disk preservation policy and cross-experiment register are:

    /mnt/donto-data/donto-resources/research/alpha-helios/PRESERVATION-POLICY.md
    /mnt/donto-data/donto-resources/research/alpha-helios/EVIDENCE-REGISTER.md

Raw research, controls, failures, rejected candidates, machine metadata, and checksums must remain on the mounted
drive; repository prose is an index, not a substitute for those artifacts.

## 2026-08-03 Helios exact profiler and register-blocked GEMM

Authoritative evidence and rejected-measurement record:

    docs/resume/HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md

Program and AMD compatibility contract:

    docs/resume/HELIOS-OPTIMIZATION-AND-AMD-PROGRAM-2026-08-03.md

The exact RTX 4090 profile found generic FP32 GEMMs consumed about 80.2% of measured dispatch time. The new
portable 2 x 2 register-blocked kernel reduced exact one-step dispatch time from 3,216,809.1 to 2,030,423.3
microseconds (-36.9%), generic-matmul time from 2,541,982.1 to 1,361,243.1 microseconds (-46.5%), and raised
matched steady median throughput from the historical 3,579 to 4,513 tokens/s (+26.1%). The six-step printed
training and validation trajectory matched. Both the default and optimized implementations passed 105 real GPU
parity/gradient tests.

Implementation and executable checks:

- `scripts/smoke-helios-dispatch-profiler.mjs`
- `scripts/smoke-helios-matmul-autotune.mjs`
- `HELIOS_PROFILE_GPU_TIMESTAMPS=1` for exact non-replayed per-dispatch timing
- `HELIOS_MATMUL_REG2X2=1` for the measured portable candidate

The odd-dimension test covers all three GEMM layouts at `M=113`, `N=157`, `K=93`; it passes on RTX 4090 and
Mesa llvmpipe. Physical AMD validation remains open and is not implied by software Vulkan evidence.

### Gradient-buffer ownership follow-up

Canonical mounted evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-gradient-ownership-20260803/

Corrected profiler labels showed that 637 `scale_vec4x2` dispatches were autograd gradient clones. The selected
ownership-aware tape transfers the final consumer's buffer and clones only true aliases. In the exact graph it
removed 728 operations (2,431 to 1,703) and reduced dispatch time from 2,186,644.2 to 1,926,446.8 microseconds.
A current-source, trace-on control toggled by `ALPHA_DISABLE_GRADIENT_BUFFER_MOVE=1` measured 4,121.0 tokens/s;
the candidate measured 6,123.2 (+48.6%). The longer selected trace-off run measured 18 warm steps at
p10/median/p90 6,432.6 / 6,567.7 / 6,666.5 tokens/s, with a minimum of 6,411.7 and maximum of 6,677.8. Matched
losses and validation loss were exact and maximum gradient-norm difference was `6.913e-7`. A fixed-order bounded
embedding-gradient gather closed an intermittent one-ulp atomic-order replay mismatch without replacing the fast
production scatter; the case passed in 10 fresh GPU processes and the physical suite passed 29 files / 283 tests.

The same evidence record preserves the rejected dKV V2 experiment: +74.7% dKV GPU time, +15.6% full-graph
dispatch time, and -2.8% steady throughput. It is not selected.

### Layout-aware R4x2/R2 follow-up

Canonical mounted evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-r42-portfolio-20260803/

A portable 16 x 8-workgroup, 4 x 2-output kernel improved ordinary and transposed-A multiplication but regressed
transposed-B, so the selected portfolio uses R4x2 / R2 / R4x2 by layout. Across 18 warm production steps, median
throughput rose from 6,567.7 to 6,836.8 tokens/s (+4.10%), with p10/p90 6,638.4 / 6,970.6. Maximum loss and
gradient-norm differences were `9.537e-7` and `4.308e-8`; terminal held-out loss, learning rate, and clipping
coefficients matched. The full RTX 4090 suite passed 29 files / 283 tests. The artifact README and digest ledger
cover the all-R4x2 profile, selected hybrid profile, sustained run, physical/software smokes, and full test output.

The first portfolio was bound as kernel policy `layout-portfolio-r42-r2-v1` before the coalesced follow-up below
superseded it. At the measured median, the frozen token contract estimated to 78.9 device-hours or
USD 54.44 at USD 0.69/hour before run overhead. This is engine evidence only, not a new model version.

### Coalesced transposed-B follow-up

Canonical mounted evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-coalesced-20260803/

R42C remaps physical B loads so adjacent X invocations traverse contiguous K elements and transpose only into
shared memory. The paired R2C control was correct but neutral. R42C reduced transposed-B time from 570,078.2 to
467,672.1 us (-17.96%), exact full-graph dispatch time from 1,759,004.2 to 1,640,182.0 us (-6.75%), and raised
the 18-warm-step median from 6,836.8 to 7,048.9 tokens/s (+3.10%). Its p10/p90 was 6,844.8 / 7,200.8.

Maximum loss and gradient-norm differences were `9.537e-7` and `3.681e-8`; terminal held-out loss, learning rate,
and clipping coefficients matched. The full RTX 4090 suite passed 29 files / 283 tests. The foundation launcher
supersedes the prior policy with `layout-portfolio-r42c-r2-v2`, binding the coalesced flag and transposed-B R4x2
selection. At the measured median, the frozen contract estimates to 76.5 device-hours or USD 52.80 at USD
0.69/hour before overhead.

### Coalesced transposed-A follow-up

Canonical mounted evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-a-coalesced-20260803/

R42C-A remaps physical `[K,M]` A loads so adjacent X invocations traverse contiguous M values and transpose only
into the shared `[32,16]` tile. Two exact controls measured 336,395.8 and 338,954.0 us across 91 transposed-A
calls; three candidates measured 290,239.8-292,475.6 us. Candidate median is 13.61% below the control midpoint.

In a candidate-first 20-step production run followed by its control, warm median throughput rose from 7,085.0
to 7,253.8 tokens/s (+2.38%). Loss was exact across all steps, maximum gradient-norm difference was `2.154e-8`,
and terminal validation, learning rate, and clipping coefficients matched. The full RTX 4090 suite passed 29
files / 283 tests. The launcher now binds `layout-portfolio-r42c-r42ca-r2-v3` at selected source commit
`028e9b31524e6d89b2caee76dad2ae47b8896e03`. At the measured median, the frozen contract estimates to 74.37
device-hours or USD 51.31 at USD 0.69/hour before overhead.

### Experimental R42CK32 local preflight

Canonical mounted evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-r42ck32-local-preflight-20260803/

Source commit `2ca869249da901763b7f4a69db939226753b198f` adds an opt-in transposed-B R42C shader with a 32-wide K tile and
8 KiB total shared memory. The intended `matmul_transposed_R42CK32` dispatch passed the awkward
`M=113`, `N=157`, `K=93` Mesa smoke with maximum absolute error `3.338e-6`; the selected K16 R42C/R42C-A control
smokes and the complete local 233-pass/50-gated/0-fail suite also passed. This archive is a local correctness
preflight, not physical speed evidence. K32 remains absent from the selected launcher until a future physical
K16/K32 A/B establishes end-to-end value.

RunPod `wtupxv15debnvh` was deleted after the selected and experimental source were pushed. Its pre-deletion
dirty worktree, untracked scripts, all stashes, and small root artifacts were preserved with transfer hash parity
at:

    /mnt/donto-data/donto-resources/benchmarks/alpha-runpod-shutdown-wtupxv15debnvh-20260803/

## 2026-08-02 Helios chat-throughput sweep

Authoritative outcome:

    docs/resume/HELIOS-CHAT-THROUGHPUT-SWEEP-OUTCOME-2026-08-02.md

Canonical mounted-drive evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-chat-throughput-20260802/

The eight-row, identical-window RTX 4090 sweep found no safe full-context
improvement over the 5,333.6 tok/s FP32 reference. Both cooperative/mixed
precision paths failed correctness. The synchronized attribution row measured
502 ms forward versus 2,682.5 ms backward. All six successful compressed
checkpoints pass `zstd -t` and their per-row digest ledgers.

| Artifact | SHA-256 |
|---|---|
| `SUMMARY.md` | `84f52591b97e9542a5f9988517da6730d5cc766df1ef0b4a9990481be58e2663` |
| `summary.json` | `a28c9c53799d3054bd8947f6cdffa0b6f27c1f0fbffedb42bd6f8a8d1feb1c3f` |
| `ARTIFACTS.sha256` | `4ed1c05cee21aca81f9710b03c00477b0d6a11d8d60cc8475bedc15f90dcbcb5` |

## 2026-08-02 foundation-candidate feasibility and LR pilot

Decision and exact pilot contract:

    docs/resume/FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md

Canonical measured evidence:

    /mnt/donto-data/donto-resources/benchmarks/alpha-foundation-feasibility-20260802/

Frozen corpus controls:

| Artifact | SHA-256 |
| --- | --- |
| `foundation-2b-manifest.json` | `be6975e2ffe327beafdc35174321c79a778b3ac33e248eba28ab591081dcb2e0` |
| `foundation-val-005-64m.txt` | `17e30fa2e50e1a1f116cceed95381b76edd1be595d402c4dd053bd55a7eafd60` |
| `foundation-val-005-64m.manifest.json` | `f010da477d29189211d04ee05253906310658e0b61aac06069d48c84be24f384` |

The 136.9M row was rejected on measured cost. The 97,098,880-parameter row at
batch 24 is authorized only for the three 384-step LR arms. No foundation
checkpoint, HF/BLAH version, or Discord improvement exists yet.

## 2026-08-02 same-dataset recipe audit

Canonical audit:

    docs/resume/SAME-DATASET-RECIPE-AUDIT-2026-08-02.md

Mounted evidence:

    /mnt/donto-data/donto-resources/research/alpha-same-dataset-recipe-audit-20260802/

The audit found that the successful public same-Smoltalk SFT recipe used packed
full-sequence causal loss for two epochs, whereas Alpha's flagship used one
un-packed assistant-only pass. The proposed synthetic V12 generation is parked
until the existing-data recipe is tested from the clean base.

Both declared V12 learning-rate arms are now complete and rejected:

    docs/resume/CHAT-RECIPE-V12-LR3E4-OUTCOME.md
    docs/resume/CHAT-RECIPE-V12-LR1E3-OUTCOME.md

The `1e-3` arm completed 2,000/2,000 finite steps and all eight frozen
checkpoint evaluations. Its best regression window was step 1,000 at 34/69
structural passes with 36 loops, versus the public Alpha baseline at 55/69 with
24 loops. No checkpoint was selected or published. Native checkpoints and the
compact evaluation mirror are preserved under:

    /mnt/donto-data/alpha-runs/alpha-chat-recipe-v12-20260802/

Frozen V12 contract:

    docs/resume/CHAT-RECIPE-V12-CONTRACT.md

Validated corpus:

    /mnt/donto-data/alpha-corpora/chat-recipe-v12/

| Artifact | SHA-256 / result |
|---|---|
| train text, 450,402 rows | `e15e19f100040565faac1ed0381ed6e3db2a06c2b9a197b756fc0dd7c20b8f2a` |
| test text, 23,710 rows | `0b6e240d5ffcbb3a26d961bcd81f37787830ff9ebfe37d4e0faa528fcdcd701c` |
| corpus manifest | `68365ae0e2e6c4289a5ab1fd4458fd67b92085dd15475f4ccbe6723448046617` |
| structure | 474,112 / 474,112 clean |
| exact train/test overlap | zero after nine held-out exclusions |
| tokenizer parity | 4,096 / 4,096 sampled rows exact |

## 2026-08-02 chat foundations v11

Authoritative outcome:

    docs/resume/CHAT-FOUNDATIONS-V11-OUTCOME.md

Run root:

    /mnt/donto-data/alpha-runs/alpha-chat-foundations-v11-20260802/

Research and evaluation mirror:

    /mnt/donto-data/donto-resources/research/alpha-chat-foundations-v11-20260802/

The step-300 native checkpoint SHA-256 is
`6226c1443741058089f110b89dfa341e0325851098d3aaf049a501c1ca3393f9`. The reference-blinded GPT-5.5
review SHA-256 is `29355fb8a4e8093472b08f0bb4438964383749c00dd2be8faf625ea468a40a1a`; it ranked V8 first and selected
`NONE`. V11 is a rejected negative result.

The operator-requested experimental publication is Hugging Face revision
`29f0372fb94c1d249421daca50c3fbd263dc1309` and BLAH model `Mq5PrXS1MUk2yl0eSKUXwA`. BLAH run
`XEDqvFu4Adbj86rKEVUqEg` completed all 24 eval definitions with mean `0.3625`, below the earlier Alpha
run's `0.395833`. The exact model, run, results, logs, and eval definitions are preserved under
`blah-evaluation/` with hashes recorded in the outcome document.

## 2026-07-31 chat repair

Root:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/

Primary files:

| Evidence | Purpose |
|---|---|
| `full-end2/checkpoint-1200.json` | selected native corrective checkpoint |
| `full-end2/run/config.json` | full executed configuration |
| `full-end2/run/repair-contract.json` | exact source, input hashes, and intervention contract |
| `full-end2/run/metrics.jsonl` | complete 2,200-step trajectory |
| `full-end2/eval-step-*/` | corrected free-generation comparisons at steps 1,200–2,200 |
| `full-end2/nvidia-gate/` | exact 46/46 NVIDIA assertion evidence |
| `full-end2/hf-alpha-60m-chat-repair-1200/` | standard six-file portable export |
| `final-heldout/` | untouched 100-chat/200-QA result after verified copy |
| `final-heldout-pair-analysis.json` | recomputed frozen-input and machine-gate comparison |

Selected checkpoint SHA-256:

    399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec

Corrective corpus root:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/

The corpus manifest binds 33,113 train and 1,767 development conversations. No AlphaCorpus candidate was used.
See `CHAT-REPAIR-2026-07-31.md` for the prompt-boundary diagnosis and honest output examples.

Final selected frozen outputs:

| File | SHA-256 |
|---|---|
| `final-heldout/chat-results.jsonl` | `3f1a178299468be0549f32f7c871445de2113ed652bfd82c3068588445311570` |
| `final-heldout/qa-results.jsonl` | `137a3981401e0563dd1bdde2e2fc86aa04112363deb10a879d10b3fb495c9300` |
| `final-heldout/summary.json` | `997535ef15a9cd00a44c7c7d84474539688a317d98112d25695995061b9699af` |
| `final-heldout-pair-analysis.json` | `8e6b245c9932ca93887549a6e839ce61337eb52a7925a4d3bc9930a978b29763` |

Final machine result: 55/100 structural, 70/100 nonempty, 31 loops, and 0/200 closed-book QA exact. Gate FAIL.

## 2026-07-31 chat repair v2

Root:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/

Authoritative narrative:

    docs/resume/CHAT-REPAIR-V2-2026-07-31.md

Primary files:

| Evidence | Purpose |
|---|---|
| `pilot-a/evaluations/baseline-step1200-development/` | published baseline on the frozen v2 selector |
| `pilot-a/evaluations/pilot-a-eval-step-{200,400,600,800}/` | bounded continuation outputs and exact shared-ID comparisons |
| `pilot-b/run/config.json` | executed clean-base configuration |
| `pilot-b/run/repair-contract.json` | exact clean-base input and intervention contract |
| `pilot-b/run/metrics.jsonl` | complete 1,600-step finite trajectory |
| `pilot-b/run/MANIFEST.sha256` | all eight optimizer-bearing checkpoints plus run state |
| `pilot-b/evaluations/pilot-b-eval-step-{400,800,1200,1600}/` | raw outputs, machine summaries, exact comparisons, and fixed panels |
| `pilot-b/remote-logs/` | launcher and evaluator logs copied from the pod |
| `pilot-b/runpod-before-termination.txt` | exact Alpha and unrelated pod state before removal |
| `pilot-b/runpod-after-termination.txt` | proof the Alpha pod was removed and the unrelated pod preserved |
| `hf-recovery-archive/CHECKSUMS.sha256` | complete 53-file public recovery payload seal |

The exact comparable 69-prompt result rejected every v2 checkpoint. The published baseline had 24 loop flags;
Pilot A's best was 30, and the clean-base control's best was 29. All v2 checkpoints were nonempty on all 96
development prompts, demonstrating that response initiation and conversational competence are separate.

Clean-base checkpoint hashes:

| Step | SHA-256 |
|---:|---|
| 200 | `689c377420dc928b34200aef75fe06738160a6c8c45adafe8ea5fb9c2488bad4` |
| 400 | `276d8fe12f30ffa9acc80336712baa5aac4d459b89e585ad536352fe61574332` |
| 600 | `d0b440470c7863afa75e470fea19548f8bc9ddc15c951e380c2f54ee416151dd` |
| 800 | `fc83b3cd8493e1b554a436a61025a80a13359317e0ad0327ec0320ebafafa0b4` |
| 1,000 | `f50b47c61788a69305ad94ea6bd428762e3ce8c2fe8e75e3139d231fe62b8f5a` |
| 1,200 | `ffac13d2fde9de551224c9764e26a0f36acf2b5acc64a6b30fad0ef092afdce1` |
| 1,400 | `fd1968b554b0b460ae9e0d49fc8e1a1da0b701d0a027976ba0e5c826dd1ca930` |
| 1,600 | `1aa3e071d1999254903b95b1c46cd3ab8907f826ebf3cf3c2078c7c52c318be8` |

Public negative-result archive:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints/tree/c1117378c0bc8b81b408be09c000f80ea9f027d7/chat-repair-v2-20260731

Anonymous publication proof:

| Item | Verified value |
|---|---|
| Immutable revision | `c1117378c0bc8b81b408be09c000f80ea9f027d7` |
| Nested files | 53 |
| Checkpoint LFS metadata | step 800 and step 1,600 SHA/size both matched |
| README SHA-256 | `3235310d50eb4da238d8658106eb484abb3d2f96068f259730f6b7f6206ec953` |
| `CHECKSUMS.sha256` SHA-256 | `b733f5704e722faadd2e6e46cd9505be44e7952da75d3d001aa65ac92cc6cf5f` |

Post-rejection live browser proof:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/public-verification/space-live-selected-step1200-20260731.png
    SHA-256 54a56df6d34bcfac0e68727953feb4bc2846c77226730f876ae9a2f19d685d14

The real browser submitted `Hey, how is your day going?`, displayed `It's going well, thank you. How about you?`,
retained the visible quality-fail status, one main landmark, and no horizontal overflow. Health and evidence
endpoints bound the runtime to selected step 1,200 and its exact checkpoint SHA.

The sealed-final suite SHA-256 is
`8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d`; it was never executed or inspected.
No v2 model-improvement Discord announcement was made.

## 2026-08-01 chat repair v3 local preflight

Authoritative records:

    docs/resume/CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md
    docs/resume/CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md

Implementation commits:

    8341dd0  train: implement matched rollout unlikelihood preflight
    5753ca9  research: bind v3 to native context
    b367f6b  research: accelerate v3 rollouts with native parity
    957a02b  research: freeze v3 checkpoint evaluation
    db7daed  research: make v3 evaluation freeze reproducible

Canonical corrected freeze:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r3-20260801/

Deterministic replay:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r3-replay-20260801/

The canonical and replay builds contain identical bytes for all six selected/excluded artifacts. The freeze is
native to the selected checkpoint's 512-token context, caps prompts at 384 tokens, reserves 128 generated tokens,
and has zero over-limit rows.

| Artifact | SHA-256 |
|---|---|
| rollout candidates | `c8df6ccd79c4eb813d87c48eee9d2462837a944d24aeba1263c87515282e670a` |
| positive cohort | `3c9dcc8d44db15491dc94e0167e864da4fc436a49edbdbf9bac6b4b0652377da` |
| rollout exclusions | `bbea8330f6730eba9e60f578c125bddd092537f6b1e82d67d5afdece39551e2d` |
| fresh development selector | `0133dcda7d6ae3d5d7ed315e528e6cf566f332a355ed6189525f7a9f2b90c683` |
| qualitative panel | `c4c869f6c1dc30a9fa644d5e45782683f200db4f80bc9c54995abf0dd0983000` |
| development exclusions | `7e574f35703d80c1c0bca7a6599a079a29fd4729270854beff174e2d9e116557` |
| canonical freeze manifest | `976ef6b37949c729a2abad77f50f46c685dcb63269af1a1963dca58428e11231` |

Corrected native rollout smoke:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-smoke-r3-20260801/

The 24-row, six-per-source native smoke contains 946 generated decisions. Its first output terminated at learned
EOS but had 4-gram repeat rate `0.5102040816326531`, confirming a mechanically eligible failed trajectory. The
batched fp32 Transformers export reproduced every one of the 946 selected tokens, every runner-up token ID,
output text, and stop exactly. Maximum chosen-logit drift was `2.2649765014648438e-05`. Evidence:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-hf-cpu-smoke-r3-20260801/
    raw-rollouts.jsonl SHA-256 f60c80f972ca5449689f7c440e36ef1e73828e351face5c2122d411cc5bf7317
    native-parity-report.json SHA-256 04ecc1f3883e53a79cd3caa6b6cf3011a74b1492c3eddda19d796b587a3ce290

Both smokes are partial diagnostic evidence, not the complete rollout ledger. The compiler requires this PASS
report if the accelerated generator produces the complete ledger.

Canonical development-only evaluation freeze:

    /mnt/donto-data/donto-resources/research/
      alpha-chat-repair-v3-evaluation-freeze-r2-canonical-20260801/

Independent byte-identical replay:

    /mnt/donto-data/donto-resources/research/
      alpha-chat-repair-v3-evaluation-freeze-r2-replay-20260801/

| Artifact | SHA-256 |
|---|---|
| evaluation contract | `c0270b2fb544fec5e03addb168841c20183ab7b7522a0937e3e0647ae0b509ce` |
| exact eligible-69 prompts | `4ba67c07fea204bbc76d76fb2b9208519bdd0029aa48046bb8143b6bcdedb584` |

The first r1 attempt is retained as negative provenance evidence: its 69 prompt bytes replayed exactly, but its
contract hash included wall-clock/output-path variation. R2 removes those non-semantic inputs and `cmp` passes on
both files. The eligible-69 set is disjoint from fresh96 by normalized prompt, preserves original v2 order, and
ranges to 508 prompt tokens. Two partial reference-blinded CPU evaluator smokes are preserved at:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-eval-hf-cpu-smoke-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-eval-regression-hf-cpu-smoke-20260801/

Local result: 223 tests passed, 50 NVIDIA assertions skipped on non-NVIDIA llvmpipe, and 0 failed. The full
rollout ledger, compiled mask cohort, real 50/50 NVIDIA proof, and both training arms remain open. No improvement
or candidate exists yet. The following r2 directories are retained but superseded because their 1,024-token
freeze did not match the selected checkpoint's native context:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r2-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r2-replay-20260801/
    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-smoke-20260801/

## Terminal run

Root:

    /mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/

Primary files:

| Evidence | Purpose |
|---|---|
| checkpoint-30322.json | terminal native ALPH checkpoint despite historical extension |
| metrics.jsonl | complete 30,322-row SFT trajectory |
| sft-contract.json | immutable input, source, optimizer, and target-step contract |
| terminal-sft-verification.json | terminal parameter and input audit |
| flagship-sft-analysis.json | strict execution analyzer |
| terminal-finalizer-status.json | finalizer outcome and mirrored-artifact binding |
| frozen-eval-pair-analysis.json | base-versus-chat machine D3 adjudication |
| frozen-chat-semantic-review-report.json | blinded 100-case semantic failure |
| hf-export-parity.log | Alpha-versus-Transformers parity |
| terminal-manifest.sha256 | terminal remote artifact seal |

Terminal checkpoint SHA-256:

    6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8

## Frozen evaluation

Root:

    /mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/frozen-eval-chat/

Important files and hashes:

| File | SHA-256 |
|---|---|
| chat-results.jsonl | bc369665e98ec49ae141e271508fa289d6fcbc7acc14fe8632360ba1f64fe161 |
| qa-results.jsonl | 82d3254f02f7c900e395ae82387256097a9926c4e651544215a993af5a5d0cd7 |
| summary.json | c4751b33d19f09fbb84f223397af63897975980dfcf52172e9e18905ae955930 |

Frozen manifest SHA-256:

    bf6e6ea4e7fb9ccffd2bab6283de42fe33e681679883da06d691f06cb867ac68

Machine pair report SHA-256:

    92da0b3bf5bd984c579ded700c1b2f9bfe928fe010a5352f65d1a15aea3d48c6

Semantic report SHA-256:

    35cc1a87fad2c4f258cfdbd5859d0a0106c0f2c1e8bdd0d6e5ada303a0ffc1e9

## Checkpoint sample series

The identical eight-prompt, non-frozen greedy comparisons live under:

    ad-hoc-discord-checkpoint-15000/
    ad-hoc-quality-checkpoint-17000/
    ad-hoc-quality-checkpoint-18000/
    ...
    ad-hoc-quality-checkpoint-30000/

Each directory contains results/chat-results.jsonl and results/summary.json. These are diagnostic
samples, not frozen-eval substitutes.

The two Discord-approved qualitative comparisons are:

    discord-progress/quality-improvement-15000-to-17000-casual-chat.txt
    discord-progress/quality-improvement-20000-to-21000-casual-chat.txt

They include the same input, before and after output, and an honest aggregate boundary.

## SFT corpus and masking

Root:

    /mnt/donto-data/alpha-corpora/sft-text-v2/

Files:

- sft-v2.txt — 511,428 rendered conversations.
- sft-v2.txt.manifest.json — sources, counts, ordering, hashes, and trimming.
- length-audit.json — tokenizer-bounded length distribution.
- mask-audit.json — independent assistant-only mask verification.

Corpus SHA-256:

    ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb

The mask audit sampled 1,032 rows and passed atomic role markers, assistant-only state transitions,
supervised final EOS, and zero over-bound rows. This rules out padding loss but not the unshuffled
source-order problem.

## Publication reports

Corrective release:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/hf-space-published-v2.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/hf-cold-load/report.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/backend-health-public.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/backend-evidence-public.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/chat-response.json
    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/public/browser/

Corrective public identities:

| Artifact | Revision / SHA-256 |
|---|---|
| Standard model revision | `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0` |
| Native archive revision | `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60` |
| Static Space revision | `d87e0950baf0a16ccd2859c2cee6314602ba2881` |
| Installed backend bundle | `c2bd8a24387584cf0eae11082adef235e62a7d12b901c749e5ddd23b18b642f4` |

Historical terminal publication:

    /mnt/donto-data/alpha-runs/hf-chat-publication-experimental-published-20260730.json
    /mnt/donto-data/alpha-runs/hf-checkpoint-publication-published-20260730.json
    /mnt/donto-data/alpha-runs/hf-static-space-publication-published-v2-20260730.json
    /mnt/donto-data/alpha-runs/hf-chat-cold-load-b481f469-20260730/report.json

All report PASS refers to packaging, upload, identity, or cold-load verification. It does not supersede
the failed D3 quality reports.

## Space and backend proof

Root:

    /mnt/donto-data/alpha-runs/alpha-60m-space-runtime-5bd723d-20260730/

Evidence includes:

- Caddy configuration before and staged/current copies;
- desktop and 390-pixel mobile screenshots;
- an after-empty-EOS screenshot;
- public API and service proof;
- compiled backend provenance.

## Recovery bundle

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

Read RESUME.md and verify MANIFEST.sha256. The mirrored hf-archive subdirectory is the exact upload
payload for the public training-checkpoint repository.

## Repository record

- Terminal archive tag: alpha-60m-archive-20260730
- Terminal closeout commit: f5162239ae330e98880f89bf950dc69a9125a38e
- Space runtime source: 5bd723db49b15df1b80a279a016c68727270bacc
- Certified SFT training source: c333bf247fbe87b85d01f3d34789b46615dd1034

Do not use an old HANDOFF live endpoint or pod ID as current truth. The historical section is retained
only to reconstruct the paid trajectory.
