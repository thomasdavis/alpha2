# Current state

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
`wtupxv15debnvh` was verified live and idle before launch. No V12 training claim
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
