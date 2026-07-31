# Current state

## Active result

The operator reopened Alpha training on 2026-07-31 to recover the original chatty-model goal. The bounded
corrective run is complete and checkpoint 1,200 is selected. Final held-out evaluation is complete and the paid
pod is removed; publication and live serving closeout are complete. Do not start another training run.

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

The selected model is materially more conversational than the archived terminal checkpoint, but its untouched
result is not structurally reliable and its semantic behavior is weak. It is not yet a dependable chatbot.

Canonical new evidence:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/

Full account:

    docs/resume/CHAT-REPAIR-2026-07-31.md

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
