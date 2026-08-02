# Alpha chat foundations v10 — independent-wave contract

**Status:** generation and review in progress; no checkpoint selected or published  
**Product:** a natural conversational model, not a benchmark-answer artifact  
**Compute boundary:** one dedicated Alpha GPU; no unrelated pod may be touched

## Why this intervention exists

Alpha's public checkpoint answers readily but is not reliably conversational. Fresh BLAH logs show direct
semantic failures, instruction failures, circular paraphrases, fabricated context, and repeated continuations.
These are not mainly refusals or empty-output failures. The prompt reaching the server is rendered with the
same atomic user/assistant markers used in training, so the broad failure cannot be attributed to a mismatched
chat template.

The first reviewed foundations intervention, v8, established a useful direction without reaching release
quality. In a reference-blinded review of 100 unchanged development conversations, its step-200 checkpoint
earned 12 PASS, 21 BORDERLINE, and 67 FAIL, versus 3 PASS, 9 BORDERLINE, and 88 FAIL for the public checkpoint.
That is a material relative gain and an unacceptable absolute result. It was not published.

V10 tests one narrow follow-up hypothesis:

> A second independently generated and reviewed realization of the same balanced conversational curriculum
> will improve unseen semantic contingency by increasing unique situations, while the stable initialization,
> assistant-only objective, low learning rate, and retained repetition-unlikelihood branch preserve stopping.

This is not permission to repeat training indefinitely. Every checkpoint remains rejectable.

## Frozen evidence

The following are selection evidence and remain excluded from training by exact normalized user-turn checks:

- the frozen 24-item BLAH baseline;
- every visible v4 development prompt;
- the original v8 development population;
- the earlier reviewed semantic-development conversations;
- the inherited sealed final, which remains unopened.

The exact BLAH prompt texts and judge answers are not converted into training examples. A generated candidate
that collides exactly with a frozen prompt is rejected; the compiler does not use a semantic keyword blacklist.

## Sampled-decoding baseline

Greedy generation is necessary but insufficient. BLAH chat, health checks, and registered-default calls sample
at temperature 0.7; a checkpoint can therefore look acceptable under one continuation while assigning too much
probability to loops or irrelevant text.

`scripts/evaluate_chat_sampling_robustness.py` runs the unchanged 615-prompt v8 development population three
times with deterministic seeds at temperature 0.7, top-k 40, and a 128-token allowance. The complete 5,535
trajectories for the three controls are archived at:

```text
/mnt/donto-data/donto-resources/benchmarks/alpha-evals-20260802/sampling-baselines-r3/
```

| Checkpoint  |  Runs | Degenerate loops | Did not stop by allowance | Empty | Structural pass |
| ----------- | ----: | ---------------: | ------------------------: | ----: | --------------: |
| Public      | 1,845 |              150 |                        60 |     1 |           1,784 |
| Stable U1   | 1,845 |               11 |                        10 |     2 |           1,833 |
| V8 step 200 | 1,845 |                3 |                         0 |     9 |           1,836 |

These are structural measurements, not semantic scores. They show that v8 made the sampled distribution much
safer, while the prior blinded review shows that semantic usefulness remained poor. V10 must satisfy both sides.

## BLAH request-metadata correction

BLAH's eval execution already supplies deterministic parameters to the model, but the stored
`raw_data.request` object was populated from the model's registered defaults. This made the evidence record say
temperature 0.7 even when execution used temperature 0. The diagnostic bug was fixed in BLAH commit `5807651`;
future eval records will report the parameters actually sent. Historical outputs remain valid, but their stored
request-parameter summary must not be treated as execution truth.

## Synthetic-data intervention

V10 does not add a generic web corpus, replay public outputs, or increase the weight of the old rows. It creates
one new independent wave:

1. The existing GPT-5.5 blueprint allocates 100 batches across ten foundations.
2. GPT-5.4 generates 64 conversations per batch with new situations and wording.
3. GPT-5.5 independently reviews every candidate for correctness, contingency, naturalness, and compactness.
4. The existing v8 compiler catalogs every accepted and rejected candidate and excludes all exact holdout,
   conversation, and normalized-user-turn collisions.
5. The v10 merger gives the original v8 development set permanent ownership of development; all surviving new
   wave rows are training candidates.
6. Cross-wave exact conversation and normalized-user-turn deduplication is repeated at merge time.

The ten focus families are:

- foundational answers;
- quantitative reasoning;
- instruction control;
- context grounding;
- negation and contrast;
- uncertainty honesty;
- language and pragmatics;
- multi-turn update;
- premise resistance;
- ordinary conversation.

The source objects contain natural-language turns. The compiler injects Alpha's exact chat delimiters only when
materializing training bytes. Rejected candidates, rejected model attempts, review concerns, source prompts,
event logs, hashes, and split decisions remain preserved on the mounted research drive.

## Generation and review identities

| Role      | Model   | Purpose                                                       |
| --------- | ------- | ------------------------------------------------------------- |
| Planner   | GPT-5.5 | Existing frozen semantic allocation and variation constraints |
| Generator | GPT-5.4 | Candidate natural-language conversations                      |
| Reviewer  | GPT-5.5 | Independent structured adjudication                           |

The generator and reviewer run through the Codex subscription lane. No per-token API key is embedded in the
project. Structured output is validated against the committed JSON schemas. Invalid outputs are preserved and
retried; they cannot silently enter the corpus.

## Merge invariants

`scripts/merge_chat_foundations_v10_corpus.ts` must prove:

- every source manifest and materialized output still matches its recorded SHA-256;
- every accepted training row passed independent review;
- the original v8 development file is copied byte-for-byte;
- no new-wave row enters development;
- exact conversation hashes are unique across waves;
- normalized user turns are unique across waves;
- inherited visible-prompt exclusions passed in every component corpus;
- all source candidates, including rejections, appear in the merged catalog;
- the sealed final remains uninspected.

The expected merged training population is at least 9,000 conversations. This lower bound is a construction
sanity check, not an argument that row count produces intelligence.

## Mask and training contract

Before GPU training, every train and development row must pass an exhaustive assistant-only state-machine mask
audit at block size 512. The audit must show:

- user and structural targets are masked;
- assistant content targets are supervised;
- the terminal end-of-text target is supervised;
- role markers remain atomic;
- no row exceeds the block;
- every row, not a sample, was inspected.

Training starts from the exact stable U1 parameter checkpoint:

```text
0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb
```

The optimizer is reset. The U1-derived RCR-UL negative trajectories are hash-preservingly rebound to the new
positive rows and remain active at weight 0.5.

The initial run will use:

| Setting                      | Value                                     |
| ---------------------------- | ----------------------------------------- |
| Context                      | 512 tokens                                |
| Batch                        | 16 conversations                          |
| Objective                    | assistant-only SFT plus RCR-UL            |
| Learning rate                | 5e-6 to 5e-7 cosine                       |
| Warmup                       | 25 steps                                  |
| Candidate checkpoints        | every 50 steps                            |
| First four assistant targets | 4x weight                                 |
| Terminal EOS                 | 2x weight                                 |
| Planned ceiling              | 400 steps; earlier checkpoints may select |

The 400-step ceiling exposes about 6,400 conversation rows before deterministic reshuffling, rather than
cycling over the small corpus for many epochs. Falling validation loss cannot authorize continuation.

## Selection and publication

Every checkpoint is first evaluated with the unchanged greedy selector, regression population, release probes,
and export parity. Only structurally eligible candidates enter semantic comparison. Finalists are then evaluated
with repeated temperature-0.7/top-40 sampling against the same controls.

A candidate must:

- materially improve blinded semantic PASS and BORDERLINE counts over v8 step 200;
- preserve direct response contingency across every focus family;
- introduce no role leaks;
- preserve or improve v8's sampled loop and stopping behavior;
- show the direction at adjacent checkpoints rather than one lucky point;
- remain better when candidate identities are hidden from the reviewer.

Only a locally selected winner may be uploaded to Hugging Face, deployed to the public inference server,
registered on BLAH, and evaluated in a fresh BLAH run. Discord receives no generation, loss, or routine progress
message. It receives a comparison only after a genuine output-level improvement, with the same input, before and
after outputs, and the aggregate boundary.

If no checkpoint clears the local gate, nothing public changes. The rejected run and all evidence remain
restartable, and the next synthetic wave must be based on its actual failure clusters rather than blind scale.
