# Frozen current state

## Program status

The Alpha 60M program closed on 2026-07-30. The engineering execution succeeded; the creative objective
did not. No new run is authorized.

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
| Further training authorization | none |

## Terminal quality truth

- Chat structural pass: 2/100.
- Empty response: 92/100.
- EOS termination: 94/100.
- Degenerate loop: 6/100.
- Blinded semantic review: 0 PASS, 0 BORDERLINE, 100 FAIL.
- Closed-book QA: 0/200 exact and 0 contained.

The two structurally passing outputs were not useful answers. The program must always be described as
a mechanically successful failed-quality research artifact, never as a chat model release.

## Durable storage

Local continuation bundle:

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

Public native archive:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
    revision 7198d1a1f094ffe88d06399ea99fecbd78fa8b66

Public standard model:

    https://huggingface.co/ajaxdavis/alpha-60m-chat
    revision b481f46924b7a4777a029de1ffb44c06cc925d4c
    safetensors SHA-256 6bb349085512c45fe5cf732209a82a5c5196d2d7a12f0aea16bdb042546dca92

The native archive, not model.safetensors, is the continuation source of truth.

## Serving state

The free static Hugging Face Space is revision
be0bd0428631d1585b13ddf9e93a8ed2d9254606. It calls the exact Alpha CPU backend at
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

Requires explicit renewed authorization:

- create or repurpose a RunPod for Alpha;
- execute any training or continuation step;
- run another frozen evaluation intended to tune against the frozen set;
- change public quality claims;
- delete native checkpoints, failed outputs, or canonical evidence.
