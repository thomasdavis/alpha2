# Alpha chat recipe V12 `1e-3` pilot outcome

Date: 2026-08-02

**Decision:** rejected. No checkpoint was selected, published to Hugging Face,
registered as a new BLAH model, or sent to Discord. The result also closes the
V12 public-recipe experiment: neither declared learning-rate arm justified a
full two-pass Smol-SmolTalk continuation.

This arm was the closest controlled Alpha replication of the published
SmolLM2 Smol-SmolTalk SFT stage. It started from the clean pre-SFT checkpoint,
used packed 1,024-token windows, and applied ordinary next-token loss to the
complete rendered conversation. It used peak learning rate `1e-3`, minimum
learning rate `1e-4`, 200 warmup steps, batch size 16, and 2,000 optimizer
steps. Symbiogenesis, assistant-only masking, RCR-UL, and bespoke synthetic
curricula were disabled. The only intended difference from the rejected
`3e-4` arm was learning rate.

## Immutable identity

- training source commit:
  `7754238337524c8caf5d16d3ac24a55f874b5b9c`;
- evaluation and profiler source commit:
  `28affb6`;
- clean parent SHA-256:
  `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049`;
- native tokenizer SHA-256:
  `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`;
- rendered training corpus SHA-256:
  `e15e19f100040565faac1ed0381ed6e3db2a06c2b9a197b756fc0dd7c20b8f2a`;
- rendered validation corpus SHA-256:
  `0b6e240d5ffcbb3a26d961bcd81f37787830ff9ebfe37d4e0faa528fcdcd701c`;
- corpus manifest SHA-256:
  `68365ae0e2e6c4289a5ab1fd4458fd67b92085dd15475f4ccbe6723448046617`;
- evaluation-freeze SHA-256:
  `3e5a35d01644961bf464c627b527cf99290b1ed6f56467ebaccfbe86a4c66908`.

The native corpus contains 277,479,707 training tokens. This pilot exposed
32,768,000 packed tokens, about 11.8% of one corpus pass. Its intentionally
short horizon tested whether the public recipe produced a viable
free-generation direction before spending the full two-pass budget.

## Execution and numerical result

The RTX 4090 run completed all 2,000 declared steps without a non-finite loss
or gradient. Across 1,893 ordinary non-checkpoint steps from step 100 onward,
mean throughput was 5,363.3 tokens/second and median throughput was 5,389.7
tokens/second. End-to-end training took 6,154.3 seconds.

Validation loss generally improved from 2.3501 at step 250 to its minimum
1.8569 at step 1,750, then ended at 1.8740. This apparently healthy
teacher-forced trajectory did not correspond to healthy conversation.

Every selected checkpoint was copied from the pod, compared against a remote
SHA-256, compressed losslessly, tested with `zstd -t`, and decompressed back to
the declared raw SHA-256. The final step-2,000 checkpoint is:

```text
raw SHA-256: 28fb5578daa89aceb497f1f80560c12a7b6727487039212d2da232307f72df86
zstd SHA-256: 8394cba548db5ce25d877c1240802af2cdc2026a034cc613f40023753e0800e7
```

The first `scp` copy of the final checkpoint was only 637,655,040 of
692,528,817 bytes and failed the remote/local hash comparison. It was never
accepted. `rsync --append-verify` completed the file, after which the raw and
decompressed hashes matched. This preservation failure is recorded because a
successful transport command is not evidence of complete checkpoint bytes.

Before evaluation, the NVIDIA suite passed 50/50 tests, including 100-step
CPU/GPU trajectory parity and mixed-precision finiteness. Every checkpoint's
portable Hugging Face export also achieved 100% token top-1 agreement with the
native model and stayed below the declared `1e-3` logit-error tolerance.

## Frozen free-generation trajectory

All outputs were generated greedily for at most 128 new tokens. `Structural`
means nonempty, stopped without a role leak or repetition failure, and is only
a mechanics gate; it is not a semantic-quality score.

| Step | Selector structural / 96 | Selector loops | Selector role leaks | Regression structural / 69 | Regression loops | Regression role leaks | Release structural / 6 | Release loops |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 23 | 82 | 3 | 23 | 49 | 5 | 0 | 6 |
| 500 | 19 | 79 | 7 | 22 | 52 | 6 | 1 | 6 |
| 750 | 43 | 73 | 7 | 32 | 48 | 5 | 2 | 4 |
| 1,000 | 40 | 69 | 10 | 34 | 36 | 2 | 0 | 5 |
| 1,250 | 31 | 69 | 24 | 24 | 46 | 11 | 0 | 5 |
| 1,500 | 30 | 76 | 14 | 24 | 39 | 12 | 0 | 4 |
| 1,750 | 25 | 78 | 22 | 25 | 46 | 12 | 0 | 4 |
| 2,000 | 30 | 73 | 21 | 21 | 46 | 11 | 1 | 4 |

The selected public Alpha baseline remains far better mechanically:

| Model | Selector structural | Selector loops | Selector leaks | Regression structural | Regression loops | Regression leaks | Release structural |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Public Alpha baseline | 83/96 | 35 | 0 | 55/69 | 24 | 0 | 6/6 |
| V12 `1e-3` best regression window, step 1,000 | 40/96 | 69 | 10 | 34/69 | 36 | 2 | 0/6 |

Step 1,000 had the arm's best regression structural count. Relative to the
public model it lost 43 selector passes, added 34 selector loops, lost 21
regression passes, and added 12 regression loops. Later checkpoints did not
recover; role leakage became much worse after step 1,000.

## What the model actually did

The model learned common answer openings and occasionally produced a short,
clean sentence, but it did not acquire dependable semantic contingency. Its
failure modes were concrete:

- `What is DNA?` at step 1,000 began, "DNA is a fundamental concept in
  mathematics," then repeatedly called it a number-theory concept;
- a promise-versus-prediction question was answered as though both were
  mathematical uncertainty concepts, followed by a circular definition;
- an institutional-identity question repeatedly alternated between "a
  separate entity" and "not necessarily a separate entity";
- ordinary answers reused generic tutoring, support, or planning templates
  unrelated to the prompt;
- many generations repeated one clause until the 128-token cap or emitted a
  new user-role marker.

The short clean outputs do not offset the distribution-level result. No sample
was posted to the improvement-only Discord channel because this arm was a
regression.

## Evaluation quota incident

The first evaluation location was the pod's network `/workspace`. Four
checkpoints completed, then checkpoint 1,250 export stopped with Linux error
122 (`EDQUOT`) while closing a file. The filesystem reported ample global free
space, but the volume's tenant quota was exhausted. Both partial attempts were
preserved. Four complete manifests were copied byte-for-byte to local pod
storage, and evaluation resumed there. The wrapper skipped the four finished
manifests and completed steps 1,250 through 2,000. This was an artifact-storage
failure, not a model or numerical failure.

## Scientific interpretation

The result rejects the narrow hypothesis that Alpha's main problem was merely
using assistant-only, unpacked SFT instead of the public packed full-sequence
recipe. Both declared learning rates removed immediate silence but remained
much worse than the public Alpha checkpoint on stopping, loops, role
boundaries, and semantics.

It does **not** establish that Smol-SmolTalk is bad or that full-sequence SFT is
generally ineffective. The public SmolLM2 checkpoint applies this stage to a
vastly more pretrained base. Alpha's base saw about one billion tokens, and
the V12 arms exposed less than one eighth of one SFT pass. The finite pilot does
establish that no evidence justifies spending two full passes on this weak
trajectory.

The next intervention is therefore not another SFT mixture on the same parent.
The active order is:

1. complete the same-workload Helios throughput and phase-profile sweep;
2. select only a numerically correct speed configuration;
3. train a stronger small foundation on the already sealed three-billion-token
   corpus;
4. apply sequence-level teacher distillation and conversational SFT;
5. select exclusively by frozen generated conversation;
6. publish a new Hugging Face/BLAH version only for an honest winner.

## Preserved evidence

Native checkpoints and run evidence:

```text
/mnt/donto-data/alpha-runs/alpha-chat-recipe-v12-20260802/lr1e3/
```

Compact evaluation evidence, including every generated output, suite summary,
audit, parity report, and manifest:

```text
/mnt/donto-data/alpha-runs/alpha-chat-recipe-v12-20260802/evaluations/lr1e3/
```

The large, reproducible `model.safetensors` copies were intentionally omitted
from the local evaluation mirror because the native checkpoints are already
preserved. Their content hashes remain in every evaluation manifest.

Top-level ledger hashes:

```text
evaluation artifact ledger: 4e02efa2ca825e1641d559292f9be74830ec7f45574d0c283cd573a84fb2b72c
run artifact ledger:        b8bfef0e36413e46de2a49fc87a306974bfd4513ddb344ea33b93e87223562ad
raw checkpoint ledger:      deb1fc65818102b38ff12ce671ebb62eefd8b307ea07302886c106bff55951f5
compressed checkpoint ledger: 5b8097d65f61863c332e9254ce1ff4d49ec913a14bc7c69dad4d77f2044dbc23
```

The retained V12 `1e-3` run is about 1.6 GiB, and the compact evaluation
mirror is about 12 MiB. Both remain well below the 15 GiB artifact review
threshold.
