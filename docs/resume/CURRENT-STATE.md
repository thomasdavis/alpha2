# Current state

## 2026-08-03 superseding update

The three-arm foundation learning-rate pilot is complete and selected peak learning rate `0.002`. The packed
train cache contains 2,058,181,632 verified tokens and the planned foundation contract consumes 1,941,995,520
tokens over 79,020 steps. No full foundation run has begun and no new model checkpoint has been published.

Helios now has exact per-dispatch Vulkan timestamp attribution and five selected engine improvements. A portable
16 x 16-workgroup, 2 x 2-output register-blocked GEMM first reduced exact one-step dispatch time by 36.9%,
generic-matmul time by 46.5%, and raised matched steady median throughput from the historical 3,579 to 4,513
tokens/s. Corrected physical-kernel labels then exposed 637 `scale_vec4x2` calls that were actually autograd
gradient clones. Ownership-aware buffer forwarding clones only genuine aliases and moves the final consumer's
buffer. On a same-source trace-on ablation it removed 728 operations and raised median throughput from 4,121.0
to 6,123.2 tokens/s (+48.6%). A longer trace-off production run measured 18 warm steps at p10/median/p90
6,432.6 / 6,567.7 / 6,666.5 tokens/s. Its median is +45.5% over the selected register-blocked baseline and
+83.5% over the historical recipe.

Matched six-step losses and held-out validation loss are exact; the maximum gradient-norm difference was
`6.913e-7`. An intermittent one-ulp replay difference from repeated-token embedding-gradient atomic order was
fixed with a bounded fixed-order gather; the production model retains the efficient scatter. The replay case
passed in 10 fresh GPU processes and the default-on ownership path passed the complete physical RTX 4090 suite:
29 files and 283 tests, including operation/model gradients and 20/100-step training trajectories. A proposed four-query dKV unroll was
also measured and rejected: it made dKV 74.7% slower and the full dispatch graph 15.6% slower. This is an engine
improvement only; there is no new Discord sample, Hugging Face model, runtime, or BLAH version.

The third result is a layout-aware portable GEMM portfolio. A new 16 x 8-workgroup, 4 x 2-output kernel wins for
ordinary and transposed-A multiplication, while R2 remains faster for transposed-B. The selected R4x2/R2 path
measured 18 warm production steps at p10/median/p90 6,638.4 / 6,836.8 / 6,970.6 tokens/s, a 4.10% median gain
over the prior selected engine and about 91% over the historical 3,579-token/s path. Maximum loss and
gradient-norm differences were `9.537e-7` and `4.308e-8`; the terminal held-out loss, learning rate, and clipping
coefficients matched. The full RTX 4090 suite again passed 29 files / 283 tests.

The fourth result makes the transposed-B R4x2 global load contiguous in physical K and transposes only into
shared memory. A paired R2C control was correct but neutral. R42C reduced transposed-B GPU time from 570,078.2 to
467,672.1 us (-17.96%) and exact full-graph dispatch time from 1,759,004.2 to 1,640,182.0 us (-6.75%). Across 18
warm steps it measured p10/median/p90 6,844.8 / 7,048.9 / 7,200.8 tokens/s, another 3.10% median gain. Maximum
loss and gradient-norm differences were `9.537e-7` and `3.681e-8`; terminal validation, learning rate, and
clipping coefficients matched. The complete physical suite passed 29 files / 283 tests.

The fifth result applies the same physical-layout reasoning to transposed-A. R42C-A assigns adjacent X
invocations to adjacent M values in physical `[K,M]` A, then transposes into the unchanged shared tile. Across
matched timestamp profiles, candidate transposed-A median was 291,701.2 us versus a 337,674.9 us control
midpoint (-13.61%). In candidate-first and control-second 20-step production runs, warm median improved from
7,085.0 to 7,253.8 tokens/s (+2.38%), with candidate p10/p90 7,052.7 / 7,360.2. Loss was exact across all 20
steps, maximum gradient-norm difference was `2.154e-8`, and terminal validation, learning rate, and clipping
matched. The physical suite again passed 29 files / 283 tests.

One additional candidate is implemented but not selected. R42CK32 doubles the coalesced transposed-B reduction
tile from 16 to 32, increasing shared memory from 4 KiB to 8 KiB and halving load/barrier rounds. At exact source
commit `2ca869249da901763b7f4a69db939226753b198f`, Mesa llvmpipe dispatched the intended shader on the awkward
`113 x 157 x 93` edge shape and matched the CPU reference to `3.338e-6` maximum absolute error. The complete
local suite passed 233 tests with 50 physical-GPU gates skipped and 0 failures. This does not establish speed;
the K32 flag is absent from the selected launcher. The required RTX 3090 comparison is now complete and rejects
K32: warmed target-kernel time was 315.9571 ms versus 312.7565 ms for K16, while 21 steady samples per arm showed
only +0.4975% mean and +0.1740% median throughput. Maximum loss and gradient-norm differences were `4.3392e-5`
and `3.3260e-3`, so the candidate also failed exact trajectory parity.

The first RTX 3090 critical-path profile attributes 84.69% of warmed dispatch time to the three selected GEMM
layouts plus attention dKV backward. A wired four-query dKV-v2 candidate was 4.50x slower and was rejected.
Non-square dKV tile experiments initially appeared much faster but produced `NaN` gradients; the kernel assumed
equal query/key tile ordinals and loaded too few query rows. Correct causal indexing and cooperative staging
restore the exact trajectory, after which `(64,32)` and `(32,16)` are 12.90% and 43.12% slower than selected
`(32,32)`. The correctness repair remains; the square 32 x 32 path remains selected. See
`HELIOS-RTX3090-CRITICAL-PATH-EVIDENCE-2026-08-03.md`.

The optimized portfolio remains selected through `HELIOS_MATMUL_REG4X2=1`,
`HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1`, `HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1`,
`HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1`, and
`HELIOS_MATMUL_REG2X2=1` pending broader hardware evidence. Its scalar Vulkan contract is vendor-neutral and passes an awkward-shape compiler/numerical smoke through Mesa
llvmpipe, but physical AMD validation is still open. The current RunPod catalog offers no AMD device to this
account, so Radeon Vulkan and Instinct ROCm/HIP validation require another provider or machine. The dedicated
Alpha pod `wtupxv15debnvh` was deleted after complete recovery on 2026-08-03. `runpodctl pod list` returned empty,
so no Alpha RunPod is billing. The dirty-worktree and stash recovery archive is
`/mnt/donto-data/donto-resources/benchmarks/alpha-runpod-shutdown-wtupxv15debnvh-20260803/`.

At the sustained median 7,253.8 tokens/s the current 1,941,995,520-token contract is about 74.37 hours before
evaluation/checkpoint overhead, or approximately USD 51.31 at the observed USD 0.69/hour price. Optimization
continues against exact GPU time before the multi-day run. With naive dKV unrolling and clone-scale materialization
now closed, the next measured targets are a CODA-controlled GEMM-epilogue slice, correct operation-specific
low precision, real attention-backward redesign, column-sum/reductions, transposes, and deeper operation-graph
quotienting. See
`HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md` and
`HELIOS-OPTIMIZATION-AND-AMD-PROGRAM-2026-08-03.md`.

## 2026-08-02 superseding update

The V12 public-recipe control is now closed. Both packed full-sequence
Smol-SmolTalk pilots (`3e-4` and `1e-3`) completed 2,000 finite steps and all
declared frozen checkpoint evaluations, but neither approached the current
public model. The `1e-3` arm's best regression window was 34/69 structural with
36 loops; the public baseline remains 55/69 with 24 loops. No V12 checkpoint
was selected, published, or posted to Discord. The result redirects the active
model program toward a better-trained small foundation and teacher
distillation. The correctness-gated Helios throughput/phase sweep is now also
complete. It found no safe 3x flag-level improvement: the full-context FP32
reference averaged 5,333.6 tok/s, workgroups 128/256 were slower, pool growth
was neutral, and cooperative/mixed-precision paths failed correctness. A
synchronized row attributed roughly 84% of forward-plus-backward time to
backward propagation. The full-context feasibility probe then rejected the
136.9M configuration on measured cost and selected a 97,098,880-parameter
candidate at batch 24 for a bounded three-way LR pilot. It sustained 3,563.7
tok/s; batch 32 failed before step one. No long pretraining run has begun. See
`CHAT-RECIPE-V12-LR1E3-OUTCOME.md` and
`HELIOS-CHAT-THROUGHPUT-SWEEP-OUTCOME-2026-08-02.md` and
`FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md`.

V11 Phase M completed 300 finite all-token bridge steps from V8 step 200 over the unchanged 10,862 reviewed
synthetic conversations. It improved response initiation to 615/615 nonempty, EOS-terminated development
replies, but increased loops from V8's 5 to 12 and did not beat V8 in the reference-blinded 100-case review.
GPT-5.5 ranked the hidden V8 reference first and selected `NONE`, so Phase S was not run.

At the operator's request, the rejected step-300 checkpoint was preserved as a separate, versioned negative
result rather than replacing the earlier Alpha:

| Item | Current exact state |
|---|---|
| V11 decision | rejected as an improvement; quality `FAIL` |
| Native checkpoint | step 300, SHA `6226c1443741058089f110b89dfa341e0325851098d3aaf049a501c1ca3393f9` |
| HF repository | `ajaxdavis/alpha-chat-v11-m300-experimental` |
| HF revision | `29f0372fb94c1d249421daca50c3fbd263dc1309` |
| Runtime | `https://donto.org/alpha-v11-m300` |
| BLAH model | `Mq5PrXS1MUk2yl0eSKUXwA`, alias `alpha-v11-m300` |
| BLAH run | `XEDqvFu4Adbj86rKEVUqEg`, completed 24/24 with one errored eval |
| BLAH comparison | 0.3625 versus earlier Alpha 0.395833; 4 wins / 12 ties / 8 losses |
| Next training intervention | V12 clean-base packed full-sequence Smol-SmolTalk recipe replication; synthetic contrast-family generation is parked |

Every future publication must increment the public version and create a new BLAH model record. Existing model
records remain immutable historical comparisons. Full evidence and output examples are in
`CHAT-FOUNDATIONS-V11-OUTCOME.md`.

The same-dataset audit and frozen V12 contract now supersede the earlier proposal
for new synthetic generation. The V12 corpus build is complete at
`/mnt/donto-data/alpha-corpora/chat-recipe-v12/`: 450,402 train rows and 23,710
test rows, both 100% structurally clean; nine exact test/train overlaps were
excluded and none remain; a systematic 4,096-row sample has exact native/HF
token-ID parity. Train SHA-256 is `e15e19f1...`, test SHA-256 is `0b6e240d...`,
and manifest SHA-256 is `68365ae0...`. The dedicated Alpha RTX 4090 pod
`wtupxv15debnvh` was verified live and idle before launch and was deleted after the later 2026-08-03 engine
gates and recovery. No V12 training claim
exists until the declared pilot produces finite checkpoints and free-generation
evidence.

## Active result

The operator reopened Alpha training on 2026-07-31 to recover the original chatty-model goal. The first
corrective run selected checkpoint 1,200 and published it with an honest quality `FAIL`. Repair v2 then tested
both bounded continuation from that checkpoint and a clean-base control on a corrected 1,024-token corpus.
Every v2 checkpoint answered every development prompt, but all regressed on repetition, stopping, and
qualitative contingency. No v2 checkpoint was selected, the sealed-final suite was never opened, the public
chat model remained unchanged, and the paid Alpha pod was removed after verified recovery.

A prompt-level mechanism analysis completed locally on 2026-08-01. On the exact 69 generation-eligible prompts,
every v2 checkpoint created more new loops than it fixed, with median onset near generated token 18–24. Only
3/68,964 supervised targets crossed the same repetition threshold, and a calibrated BGE diagnostic found no
reliable semantic-contingency gain. The next intervention is a matched control of train-only,
rollout-conditioned repetition unlikelihood. Its local implementation and corrected 512-token freeze are complete,
and its development-only evaluation contract now replays byte-for-byte. It remains unexecuted on NVIDIA and does
not authorize a pod.

| Item | Current state |
|---|---|
| Active source | training `57c065e35c7564688726dafb404efaff952d860b`; prompt fix `cf4ad61` |
| Initialization | clean pre-SFT base SHA `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049` |
| Corrective data | 33,113 train / 1,767 development conversations |
| Full run | 2,200 steps, complete and finite |
| Selected checkpoint | step 1,200, SHA `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec` |
| Repair development result | 48/48 nonempty, 48/48 EOS, 0 role leaks, 5 loops |
| Final frozen result | 55/100 structural, 70/100 nonempty, 31 loops, QA 0/200; gate FAIL |
| Paid pod | `ksotbczj60mntk` removed; `runpodctl pod list` empty after removal |
| AlphaCorpus | paused side project; no candidate entered this run |
| Standard model | revision `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0` |
| Native archive | revision `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60` |
| Public Space | revision `d87e0950baf0a16ccd2859c2cee6314602ba2881` |
| Live backend | step 1,200, quality FAIL, source `e55cb23` |
| Repair v2 | Pilot A rejected; clean-base control rejected; no selection |
| V2 exact comparison | baseline 24 loops; best v2 still 29 loops on the same 69 prompts |
| V2 sealed final | never executed or inspected |
| V2 recovery archive | revision `c1117378c0bc8b81b408be09c000f80ea9f027d7`, 53 files |
| V2 Alpha pod | `omn3hktwqs7r5l` removed; unrelated pod left untouched; final live list empty |
| V2 mechanism analysis | complete; clean-target / self-amplifying-prefix hypothesis best supported |
| Repair v3 | local code/freeze PASS; 24-row native/export parity 946/946 exact; full rollouts/GPU/training still open |
| V3 evaluation freeze | fresh96 + panel24 + exact v2 eligible69 bound; contract SHA `c0270b2f`; canonical/replay exact |
| V3 local tests | 223 passed / 50 NVIDIA-gated / 0 failed; TypeScript and Python syntax clean |

The selected model is materially more conversational than the archived terminal checkpoint, but its untouched
result is not structurally reliable and its semantic behavior is weak. Repair v2 did not improve it. Alpha is
not yet a dependable chatbot, and no further run is authorized without a genuinely new finite intervention.

Canonical new evidence:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/

Full account:

    docs/resume/CHAT-REPAIR-2026-07-31.md
    docs/resume/CHAT-REPAIR-V2-2026-07-31.md
    docs/resume/CHAT-REPAIR-V2-MECHANISM-ANALYSIS-2026-08-01.md
    docs/resume/CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md
    docs/resume/CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md

## Archived terminal baseline

## Program status

The first Alpha program closed on 2026-07-30. Its engineering execution succeeded; its creative objective
did not. Its evidence remains immutable even though the later corrective run supersedes it as the model candidate.

| Item | Frozen state |
|---|---|
| Repository archive tag | alpha-60m-archive-20260730 |
| Archived source commit | f5162239ae330e98880f89bf950dc69a9125a38e |
| Training source commit | c333bf247fbe87b85d01f3d34789b46615dd1034 |
| Base-pretrain source | e561f66 |
| Architecture | Llama-form 16 layers, 512 width, 8 heads, 57,688,576 parameters |
| Tokenizer | Alpha byte BPE, vocabulary 12,288, block size 1,024 |
| Pretraining | 61,036 steps, 1,000,013,824 tokens |
| SFT | 30,322 steps, one epoch, 496,795,648 padded positions |
| Terminal SFT checkpoint | SHA-256 6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8 |
| Terminal execution gate | PASS |
| Terminal chat-quality gate | FAIL |
| Alpha RunPod | removed |
| Further training authorization | superseded by the bounded 2026-07-31 corrective run |

## Terminal quality truth

- Chat structural pass: 2/100.
- Empty response: 92/100.
- EOS termination: 94/100.
- Degenerate loop: 6/100.
- Blinded semantic review: 0 PASS, 0 BORDERLINE, 100 FAIL.
- Closed-book QA: 0/200 exact and 0 contained.

The two structurally passing outputs were not useful answers. This remains the truth about that checkpoint and
must not be rewritten as though the later repair had been present.

## Durable storage

Local continuation bundle:

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

Historical native archive revision before corrective evidence:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
    revision 7198d1a1f094ffe88d06399ea99fecbd78fa8b66

Historical standard model revision before corrective publication:

    https://huggingface.co/ajaxdavis/alpha-60m-chat
    revision b481f46924b7a4777a029de1ffb44c06cc925d4c
    safetensors SHA-256 6bb349085512c45fe5cf732209a82a5c5196d2d7a12f0aea16bdb042546dca92

Current selected continuation source:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints/blob/ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60/checkpoints/chat-repair-selected-step-1200.alph
    SHA-256 399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec

Rejected v2 recovery branch:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints/tree/c1117378c0bc8b81b408be09c000f80ea9f027d7/chat-repair-v2-20260731
    step 800 SHA-256 fc83b3cd8493e1b554a436a61025a80a13359317e0ad0327ec0320ebafafa0b4
    step 1,600 SHA-256 1aa3e071d1999254903b95b1c46cd3ab8907f826ebf3cf3c2078c7c52c318be8

Those v2 files are optimizer-bearing recovery states, not selected public models.

Current standard model:

    https://huggingface.co/ajaxdavis/alpha-60m-chat
    revision ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0
    safetensors SHA-256 a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688

The native archive, not model.safetensors, is the continuation source of truth.

## Serving state

The free static Hugging Face Space is revision
`d87e0950baf0a16ccd2859c2cee6314602ba2881`. It calls the exact Alpha CPU backend at
https://donto.org/alpha-60m. The backend is installed as alpha2-hf-backend.service, loopback-only behind
Caddy, nice 19, CPU-capped, and memory-capped at 3 GB.

The backend has no alternate model and reports quality_gate=FAIL. Empty EOS is returned as a successful
model response with an explicit alpha.empty_eos marker.

## Current authority boundary

Allowed without renewed training authorization:

- inspect or improve documentation;
- verify hashes and public metadata;
- maintain the existing CPU serving path;
- add tests and repair code without running training;
- prepare a proposed experiment contract.

Requires explicit renewed authorization after this closeout:

- create or repurpose a RunPod for Alpha;
- execute any training or continuation step;
- run another frozen evaluation intended to tune against the frozen set;
- change public quality claims without binding them to the selected checkpoint's exact evaluation;
- delete native checkpoints, failed outputs, or canonical evidence.
