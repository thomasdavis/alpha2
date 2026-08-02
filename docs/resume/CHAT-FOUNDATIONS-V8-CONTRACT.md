# Alpha chat foundations v8 — finite experiment contract

## Product question

Can a compact, independently reviewed synthetic curriculum make Alpha answer the user's actual question, follow small instructions, use supplied context, handle ordinary corrections, and participate naturally in conversation without losing the anti-loop stability already present in U1?

Alpha is still a chat model. This intervention is not a benchmark-answer patch, a factual memorization run, a programming curriculum, or an ontology product. It targets the missing foundations that prevent Alpha's existing language ability from becoming reliable conversation.

## Frozen evidence behind the intervention

The public BLAH baseline contains 24 eligible results. Alpha earned full credit on 6 and a mean score of 0.37708333333333327. Raw outputs show semantic imitation, wrong answers, circular responses, and repetition. They do not show refusal or token-limit truncation as the main failure mode. The live prompt format matches Alpha's trained chat template.

The exact 24 BLAH prompts, all visible v4 development suites, and the prior reviewed semantic-development conversations are exact normalized holdouts. They may be used to reject a colliding synthetic candidate, but their target answers do not enter training. The inherited sealed final remains unopened.

## Synthetic-data construction

The intervention deliberately uses synthetic data as the training substance, not merely as an evaluation accessory:

1. GPT-5.5 planned 100 bounded batches across ten conversational foundations.
2. GPT-5.4 generated 6,400 candidate conversations: 42 single-exchange and 22 two-exchange conversations in every batch.
3. GPT-5.5 independently reviewed every candidate exactly once for semantic correctness, response contingency, naturalness, and compactness.
4. The compiler admits only candidates with an `accept` decision and scores of at least 4/4/4/3 on those dimensions.
5. Every generated candidate remains in `catalog.jsonl`, including rejected candidates, scores, concerns, raw turns, and final admission status.

The focuses are:

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

The reviewer accepted 5,776/6,400 candidates under its own decision. Four accepted decisions fell below the stricter compiler thresholds. The compiler then excludes exact normalized holdout collisions and repeated normalized user turns. No semantic keyword filter, answer blacklist, replay corpus, or public-model output is used.

## Development split

The statistical and leakage unit is a whole generation batch, not an individual row. For every focus, exactly one of its ten batches is selected by the lowest `sha256(seed, dev-batch, batch_id)`. All surviving candidates from that batch go to development; none go to training.

The deterministic pre-commit smoke build produced:

| Population              |  Rows |
| ----------------------- | ----: |
| Generated and cataloged | 6,400 |
| Training                | 5,141 |
| Development             |   615 |
| Accepted overall        | 5,756 |
| Rejected overall        |   644 |

Rejection causes can overlap. They were 628 independent-review/threshold failures, 14 repeated normalized user-turn collisions, and 2 exact normalized holdout collisions.

## Rendering and loss contract

The source database objects contain natural-language turns without chat delimiters. The compiler injects Alpha's exact atomic markers only when materializing the training text:

```text
<|user|> ... <|assistant|> ... <|end_of_text|>
```

Every materialized row must:

- begin with a user turn;
- alternate user and assistant turns;
- end after an assistant response with one EOS;
- tokenize to at most 512 tokens;
- mask all user and structural targets;
- supervise all assistant content targets;
- supervise the final EOS;
- preserve the three chat markers as atomic tokenizer items.

Before training, `verify_sft_masks.ts --every 1 --block 512` must pass on every training and development row. A sampled mask audit is not sufficient.

## Initialization and anti-loop branch

Training starts from the exact mechanically stable U1 checkpoint:

```text
0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb
```

The optimizer and schedule start fresh. The prior immutable U1-derived RCR-UL rollout trajectories are deterministically rebound to the new positive rows without changing any negative token IDs or penalty positions. This is a pairing operation, not new generation. The unlikelihood branch remains active at weight 0.5 so the semantic repair does not silently discard the only intervention that previously improved loop mechanics.

## Finite GPU schedule

The declared initial schedule is:

| Setting                 |               Value |
| ----------------------- | ------------------: |
| Steps                   |               1,200 |
| Batch size              |    16 conversations |
| Context block           |          512 tokens |
| Optimizer               |         fresh AdamW |
| Learning rate           | 1e-5 to 1e-6 cosine |
| Warmup                  |           100 steps |
| Checkpoints             |     every 200 steps |
| First assistant targets |       first 4 at 4x |
| Final EOS               |                  2x |
| RCR-UL                  |   0.5, epsilon 1e-6 |

The run is finite. Additional epochs are not automatically authorized by a falling loss.

## Selection

Every declared checkpoint at steps 200, 400, 600, 800, 1,000, and 1,200 is evaluated only after training finishes. Evaluation uses the unchanged visible selector, regression population, release probes, and generation-parity checks.

Selection is based on actual outputs:

- semantic correctness;
- response contingency;
- instruction satisfaction;
- context use;
- natural stopping;
- nonempty initiation;
- loop and role-leak behavior;
- ordinary conversational quality.

Validation loss cannot select. A merely nonempty response cannot select. The BLAH suite is rerun only after one local checkpoint wins under the frozen rule. Discord receives an update only if the model genuinely improves, accompanied by model outputs and the reason the change matters.

## Storage rule for this run

Historical local repair artifacts already occupy roughly 14 GiB. The v8 corpus and audit artifacts are small, but v8 checkpoint families will remain on the dedicated GPU pod during selection. Only the selected checkpoint and its necessary recovery evidence are mirrored locally, preventing this campaign from crossing the user's 15 GiB soft pause threshold through redundant checkpoints.

## Failure interpretation

- If correct relational data does not improve unseen conversational behavior, reject the curriculum hypothesis rather than adding blind epochs.
- If semantic behavior improves but loops regress, adjust the positive/RCR balance in a new declared intervention.
- If an earlier checkpoint is best, later loss improvement does not overrule it.
- If no checkpoint beats the current public model, publish nothing and preserve the null result.
- If a local winner fails the frozen BLAH rerun, diagnose the raw BLAH logs before constructing the next non-leaking intervention.
