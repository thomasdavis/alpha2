# Alpha chat repair — 2026-07-31

## Executive result

Alpha has recovered the basic mechanics of conversation. The selected corrective checkpoint starts a response
on every case in the 48-prompt development suite, stops on EOS on every case, leaks no role markers, and sounds
recognizably conversational on many ordinary prompts. The archived terminal model had produced only eight
nonempty answers in 100 prompts and 92 empty strings.

This is not a claim that Alpha is a strong reasoner. The selected checkpoint still produces shallow answers,
misreads idioms and conceptual questions, and enters a repetition loop in five of 48 development cases. The
result is best described as **structurally chatty, semantically immature**.

No further training run is currently justified. The final untouched evaluation failed the predeclared quality
gate, and the portable export, public publication, and restart archive preserve that result before any later
experiment spends more GPU.

## Product goal and scope

The active goal is the original Alpha product:

> Build a small model that is pleasant and effective in ordinary conversation.

Its first job is to answer, stay responsive to the user’s latest move, contribute an appropriate next thought,
and stop cleanly. Linguistics, ontology, philosophy, evidence use, and synthetic conceptual curricula remain
future specializations. They cannot compensate for an inert or mechanically awkward base interlocutor.

AlphaCorpus is therefore paused as a side project. Its SQLite ledger, public explorer, PRDs, and unreviewed
synthetic candidates are preserved, but none of those rows entered this corrective training run.

Model scale is not a benchmark or identity claim. The current architecture is simply the existing artifact that
could be repaired quickly on one rented GPU. Future scale is governed by one-GPU training and serving feasibility
and by measured conversational gain per dollar.

## What failed in the archived model

The terminal SFT checkpoint’s sealed 100-chat evaluation measured:

| Metric | Terminal checkpoint |
|---|---:|
| Structural pass | 2 / 100 |
| Nonempty | 8 / 100 |
| Immediate or eventual EOS | 94 / 100 |
| Empty | 92 / 100 |
| Degenerate loops | 6 |
| Human semantic result | 0 PASS / 100 FAIL |

That failure had **two independent mechanisms**.

### 1. Real answer-initiation undertraining

The old SFT loader walked a source-grouped corpus without epoch shuffling. Token-averaged loss let long answer
interiors dominate short answers and response starts. The model could score well under teacher forcing because
the gold answer prefix was supplied at every position, while still choosing EOS as the first generated token.

The failure was worse on longer prompts: all terminal nonempty outputs came from prompts of 84 tokens or less,
and no prompt over 300 tokens received a response. More of the same ordered epoch was not a reliable remedy;
checkpoint samples oscillated while held-out loss remained plausible.

### 2. A generation-only tokenizer-boundary defect

Every production generation path rendered the final boundary as:

    <|assistant|> 

The SFT corpus rendered known assistant content as:

    <|assistant|> Hello

These strings are not equivalent under Alpha’s byte-level BPE. The content’s leading space is normally absorbed
into the first content token. The generation-only trailing space instead became a standalone token after the
assistant marker, creating a boundary the model never saw in training.

The exact diagnostic was:

| Text | Token IDs | Interpretation |
|---|---|---|
| `<|assistant|>` | `[257]` | correct generation boundary |
| `<|assistant|> ` | `[257, 32]` | erroneous standalone space token |
| `<|assistant|> Hello` | `[257, 400, 11713]` | content begins with a space-owning token |

On checkpoint 1,200, the old boundary entered a code-fence loop under both the fast inference path and the slow
autograd reference path. Removing only the terminal space produced an ordinary response and EOS. This ruled out
the fast inference engine as the cause of that behavior.

Commit `cf4ad61` removes the generation-only space from:

- frozen chat and QA evaluation;
- the native Hugging Face-compatible API;
- the exported Hugging Face Jinja chat template;
- frozen-suite construction and verification.

Historical assistant turns keep the space before their known content. Regression tests distinguish these two
cases. The fix passed package builds, repository typechecking, 19 focused tests, 211 central tests with 46
NVIDIA-only tests gated, and the single timed-out analyzer passed when rerun alone in 4.81 seconds.

Because the terminal 100-case evaluation used the defective boundary, its recorded outputs remain valid evidence
of what the published server produced at that revision, but they are not a clean estimate of the checkpoint under
the corrected protocol.

## Corrective corpus

The repair corpus was intentionally compact and conversation-first. It was constructed from already staged,
licensed sources and did not use AlphaCorpus synthetic candidates.

| Source | Selected conversations |
|---|---:|
| SODA | 30,000 |
| SmolTalk2 everyday conversation | 2,260 |
| Smol concise subset | 1,436 |
| OASST2 | 1,184 |
| **Total** | **34,880** |

The deterministic split contains 33,113 training and 1,767 development conversations. Exact duplicate rendered
conversations were removed. Every row fits the 512-token corrective context without cutting an assistant answer.
The train and development mask audits passed.

Important limitation: SODA contributes about 86% of the selected rows. It supplies useful short turn-taking and
roleplay but also encourages invented personal identities, scene continuation, and vague human-to-human dialogue.
This is visible in the model’s tendency to answer as a fictional participant instead of a precise assistant. A
future corpus revision should increase direct assistant answers and reduce roleplay dominance, but that is not a
reason to launch another run during this closeout.

Canonical inputs:

| Artifact | SHA-256 |
|---|---|
| `train.txt` | `298ca5332fc23e19bdd5f8cdb2a1a94e54dadfb48579ab6a58097f20135e1744` |
| `dev.txt` | `412d9dcc5bd36af38140f288fce9a94b9de9287fd0bdc5e9bc10101c1c2f4645` |
| `manifest.json` | `792256f22e43777fbc188141b11358d29c25773bea7f50583fd0ba213081bfa6` |
| tokenizer | `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24` |
| clean pre-SFT base | `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049` |

Corpus root:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/

## Training intervention

The run changed the learning problem rather than merely extending the failed epoch:

1. Initialize from the clean pre-SFT base, not the failed terminal SFT checkpoint.
2. Deterministically shuffle conversations every epoch.
3. Normalize each conversation’s supervised loss so long answers do not dominate short ones.
4. Weight the first four assistant content tokens by 8x.
5. Weight terminal EOS by 2x, independently of answer-start weighting.
6. Exclude EOS from the answer-start category.
7. Select checkpoints by free generation before considering validation loss.

Executed full-run contract:

| Setting | Value |
|---|---:|
| Steps | 2,200 |
| Batch | 32 conversations |
| Context | 512 tokens |
| Learning rate | `5e-5` cosine to `5e-6` |
| Warmup | 100 steps |
| Start weighting | first 4 content tokens at 8x |
| EOS weighting | 2x |
| Epoch shuffle | deterministic |
| Conversation weighting | equal |

The run completed in 4,421.4 seconds on a rented RTX 4090 at approximately 8.5–8.9K tokens/second. A larger GPU
was not required for this workload: the 4090 made the complete controlled run and checkpoint sweep comfortably
fit the comparison window. Bigger hardware should be rented only when a larger batch, model, or experiment is
known to use it rather than as a status symbol.

The fail-closed NVIDIA suite passed 46/46 on the exact training source before the full run.

## Checkpoint selection

All results below use deterministic greedy decoding and the corrected terminal assistant boundary.

| Step | Nonempty | Structural / EOS | Role leaks | Loops | Mean / max 4-gram repetition |
|---:|---:|---:|---:|---:|---:|
| 1,200 | 48/48 | 48/48 | 0 | 5 | 0.04034 / 0.400 |
| 1,600 | 48/48 | 46/48 | 0 | 6 | 0.07994 / 0.928 |
| 1,800 | 48/48 | 46/48 | 0 | 5 | 0.05887 / 0.928 |
| 2,000 | 48/48 | 46/48 | 0 | 4 | 0.05652 / 0.928 |
| 2,200 | 48/48 | 48/48 | 0 | 7 | 0.06499 / 0.542 |

Checkpoint 1,200 is selected because it is the cleanest overall start/stop/repetition trade-off and its ordinary
conversation samples are stronger. Step 2,000 reached a lower validation loss and one fewer detected loop, but
two responses failed to stop and maximum repetition was substantially worse. This is direct evidence that
teacher-forced validation loss is not an adequate chat-model selector.

Selected native checkpoint:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/full-end2/checkpoint-1200.json

SHA-256:

    399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec

## Representative outputs

These are exact deterministic outputs from the selected checkpoint under the corrected prompt. They are not
teacher-written exemplars.

| Input | Output | Assessment |
|---|---|---|
| “Hey, how’s your day going?” | “It’s going well, thanks. How about you?” | Natural, direct turn-taking |
| “You won’t believe what happened at lunch.” | “What happened?” | Concise and appropriately responsive |
| “Continue this in two sentences: The elevator opened onto a beach.” | “The elevator opened onto a beach.” | Nonempty and fluent, but merely repeats the seed and ignores the two-sentence instruction |
| resolved financial-bank follow-up | “Well, it’s possible that the bank was crowded. But it’s not a riverbank.” | Retains the resolved sense, but does not answer why Mondays might be worse |
| “The plan is held together with string. What does that imply?” | “It means that the plan is held together with string. It’s a string that is used to hold the plan together with string.” | Literal, repetitive, misses the idiom |
| conceptual fairness prompt | “People really mean when they call a decision fair. They’re not really sure what they’re talking about.” | Conversational surface, weak analysis |
| cat-or-dog opinion prompt | “A cat or a dog? A cat or a dog? A dog? A dog?” | Repetition failure |

The improvement Discord post showed the before/after contrast and explicitly reported the remaining weaknesses.
No later checkpoint earned another post because none improved the qualitative trade-off.

## What this result proves—and does not prove

It proves that:

- the old empty-response behavior was repairable without resuming the failed SFT epoch;
- answer-start and EOS supervision must be controlled separately;
- deterministic shuffling and equal conversation weighting are practical in Alpha’s trainer;
- free generation can select a materially better checkpoint than held-out loss;
- the generation-only assistant-space bug affected native evaluation and portable serving;
- the selected model can participate in short ordinary exchanges instead of immediately terminating.

It does not prove that:

- Alpha understands philosophy, ontology, pragmatics, or linguistics deeply;
- every nonempty answer is relevant or coherent;
- the current SODA-heavy mixture is the right long-term data distribution;
- longer multi-turn conversation is stable;
- factual QA is strong;
- repetition is solved;
- the selected checkpoint passes the old predeclared zero-loop D3 gate.

## Final untouched evaluation

The 100-chat/200-QA frozen suite was held until checkpoint selection. Its input hashes are:

| Input | SHA-256 |
|---|---|
| 100 chat prompts | `6c463debaaf4f59452bc5e88ce85ca81f64c8a9e91974822609b5ac0883f7121` |
| 200 closed-book QA items | `bbbeec574f5dd25a07ec1ff1c5a184d1709397345c394aa793f6a6d5d3c30a62` |

| Metric | Selected step 1,200 |
|---|---:|
| Structural pass | 55 / 100 |
| Nonempty | 70 / 100 |
| EOS terminated | 56 / 100 |
| Empty | 30 / 100 |
| Role leaks | 0 / 100 |
| Degenerate loops | 31 / 100 |
| Mean / maximum 4-gram repetition | 0.141388 / 0.840 |
| Closed-book QA exact / contained | 0 / 200 / 0 / 200 |
| Closed-book QA mean token F1 | 0.015292 |
| Predeclared structural/repetition gate | **FAIL** |

The untouched suite exposes a large development-to-evaluation gap. The repair improved the archived server's
2/100 structural pass and 92/100 empty profile, but it did not produce a dependable chatbot: 30 frozen prompts
were still empty, 44 did not terminate on EOS, and 31 crossed the loop threshold. The full outputs are preserved
without model-judge relabeling.

| Output | SHA-256 |
|---|---|
| `chat-results.jsonl` | `3f1a178299468be0549f32f7c871445de2113ed652bfd82c3068588445311570` |
| `qa-results.jsonl` | `137a3981401e0563dd1bdde2e2fc86aa04112363deb10a879d10b3fb495c9300` |
| `summary.json` | `997535ef15a9cd00a44c7c7d84474539688a317d98112d25695995061b9699af` |
| pair analysis | `8e6b245c9932ca93887549a6e839ce61337eb52a7925a4d3bc9930a978b29763` |

The pair analyzer verified the same frozen inputs and matching model shape except for the predeclared context
change from 1,024 to 512 tokens. Its result is FAIL. Relative to the clean base it gained 55 structural passes
and reduced detected loops by 68, while remaining far short of the release gate.

## Portable export

The selected checkpoint has been exported as a standard zero-custom-code `LlamaForCausalLM` directory:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/full-end2/hf-alpha-60m-chat-repair-1200/

| File | SHA-256 |
|---|---|
| `model.safetensors` | `a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688` |
| `config.json` | `d9080edae93005a8738604d86bfafa93429fa839ee2ca4c0e1f8901074c8616a` |
| `generation_config.json` | `097ffca0d0dec269ef98214f0399f091452f88b57cc6334208a9904f68e59fd6` |
| `tokenizer.json` | `37372c9b1bdbf7d9655444e90247bef957018d0d7ff0b668d1330e28d97c44cf` |
| `tokenizer_config.json` | `75b8671684f5643da2c5bf397b5a4f021c12842fb355c6431d409f4e91c60245` |
| `chat_template.jinja` | `8c08c7e2eaec9375356477033e1532b814fb75aff8dcf1cc1bf2b04fd93a00a7` |

The exported chat template uses `<|assistant|>` with no trailing generation-only space. Stock Transformers loaded
the export with zero custom code. Across five prompts covering 87 token positions—including single-turn and
multi-turn chat boundaries—tokenizer IDs matched 5/5, top-1 logits matched 87/87, and maximum absolute logit
difference was `5.531e-05` against the native Alpha reference (`1e-3` threshold). A stock message-list generation
smoke test returned `" It's going well. I'm doing alright. "` rather than empty EOS.

## Artifact map

| Evidence | Location |
|---|---|
| Full native run contract and metrics | `alpha-chat-repair-20260731/full-end2/run/` |
| Selected native checkpoint | `alpha-chat-repair-20260731/full-end2/checkpoint-1200.json` |
| Corrected development evaluations | `alpha-chat-repair-20260731/full-end2/eval-step-{1200,1600,1800,2000,2200}/` |
| NVIDIA gate evidence | `alpha-chat-repair-20260731/full-end2/nvidia-gate/` |
| Hugging Face export | `alpha-chat-repair-20260731/full-end2/hf-alpha-60m-chat-repair-1200/` |
| Final frozen evaluation | `alpha-chat-repair-20260731/final-heldout/` after copy and verification |
| Corrective corpus | `/mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/` |

All run paths above are beneath `/mnt/donto-data/alpha-runs/` unless absolute. The project-owned corrective archive
remains well below the operator’s 15 GiB pause threshold.

## Closeout checklist

- [x] Build compact deterministic corrective corpus.
- [x] Add epoch shuffle, equal conversation weighting, and independent start/EOS controls.
- [x] Prove the NVIDIA path on the exact source.
- [x] Complete the full corrective run.
- [x] Evaluate multiple checkpoints with free generation.
- [x] Select checkpoint 1,200.
- [x] Diagnose and fix the generation-boundary defect in all paths.
- [x] Export a standard Hugging Face payload.
- [x] Finish, copy, hash, and independently recompute the untouched 100-chat/200-QA evaluation.
- [x] Prove native/Transformers logit parity on the selected export.
- [ ] Update the public model card and upload the selected model.
- [ ] Add the native checkpoint and complete run evidence to the recovery archive.
- [ ] Update the public Space/backend to the selected hash and verify real browser output.
- [x] Terminate pod `ksotbczj60mntk` and verify no paid Alpha pod remains.
- [ ] Commit and push the closeout documentation and exact public revisions.

## If training is resumed later

Do not simply continue from step 2,200. Begin from the selected checkpoint or clean base under a new written
contract and isolate one intervention at a time. The highest-value next questions are:

1. Can a more assistant-like, less roleplay-heavy corpus reduce shallow persona continuation without reviving EOS?
2. Can repetition be reduced through data and selection without sacrificing short natural replies?
3. How long can the model preserve conversational state across turns under its real context window?
4. Does a larger one-GPU-feasible model learn the same curriculum more semantically, or does data quality remain
   the limiting factor?

Keep the final frozen suite sealed from training decisions. Add new development prompts for any new intervention
and treat the existing 100-chat outputs as evaluation evidence, not tuning data.
