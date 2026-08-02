# Alpha chat foundations v11 — all-token bridge contract

Date: 2026-08-02

## Product boundary

Alpha remains a small, one-GPU conversational model. The product target is a
model that answers the latest turn naturally and usefully. It is not a
benchmark-answer store, a fact memorizer, a programming model, or an ontology
serializer.

## Evidence behind this intervention

V8 remains the best local semantic checkpoint, but it still fails most of a
reference-blinded 100-case conversational panel. V10 doubled the number of
independently generated and reviewed situations while keeping initialization,
assistant-only SFT, and the anti-loop branch stable. V10 was worse than V8:
`7 PASS / 18 BORDERLINE / 75 FAIL` versus `13 / 27 / 60` in a direct blinded
comparison.

The dominant failures are not empty output or refusal. They are:

- repeating a noun or number from the prompt instead of applying an operation;
- responding to an earlier assistant turn rather than the latest user move;
- giving the right initial polarity and then contradicting it;
- restating a request instead of performing it;
- failing to distinguish unknown personal facts from supplied context;
- losing negation, exception, and update structure;
- producing fluent tautologies instead of ordinary useful text.

Another independent SFT wave is therefore not justified.

## Hypothesis

Assistant-only supervision may be too sparse for the current foundation. It
trains answer tokens and does backpropagate through the prompt, but it never
directly trains the model to predict and organize the user-language transitions
that define these synthetic conversations.

V11 tests whether an all-token causal-language-modeling bridge over the exact
reviewed V10 bytes strengthens the language and state-transition substrate. The
bridge is followed only after evaluation by a short assistant-only recovery
stage that restores explicit response policy and the proven anti-loop branch.

This is a causal objective test, not permission for blind extra epochs.

## Fixed data

Phase M uses the existing V10 train and development files byte-for-byte:

- 10,862 training conversations;
- 615 whole-batch-held-out development conversations;
- GPT-5.4 generation with GPT-5.5 independent review;
- exact V8 development ownership;
- exact normalized exclusion of visible development and BLAH prompts;
- no public-output replay and no sealed-final access.

The chat markers remain in the text because they are part of the exact model
format. Unlike SFT, every next-token target is supervised during Phase M,
including user language and structural transitions. No generated or reviewed
row is changed, added, or removed, so the intervention cannot be credited to a
new corpus.

## Initialization

Phase M starts from V8 step 200, the best local semantic checkpoint under direct
reference-blinded comparison:

```text
acae25cf38ab0ac7fbc621fad0d817c187514d27c792d5586ac722e54cb8254a
```

The optimizer and learning-rate schedule start fresh. The model architecture,
tokenizer, chat template, inference renderer, and decoding settings do not
change. Symbiogenesis remains off because this experiment changes the learning
objective, not the architecture or automatic configuration policy.

## Phase M: all-token bridge

| Setting | Value |
| --- | ---: |
| Objective | ordinary next-token loss over every token |
| Context | 512 tokens |
| Batch | 16 packed sequences |
| Steps | 300 |
| Learning rate | 1e-5 to 1e-6 cosine |
| Warmup | 25 steps |
| Checkpoints | 75, 150, 225, 300 |
| RCR-UL | off during the bridge |
| Assistant-only masking | off during the bridge |

The planned exposure is about 2.46 million token positions. This deliberately
revisits the compact corpus several times because the scientific variable is
whether direct modeling of the entire dialogue trajectory changes held-out
behavior. It is not evidence that unlimited repetition is safe.

Phase M checkpoints are not publishable chat models. They are candidates for
the recovery stage only.

## Mid-bridge evaluation

Every Phase M checkpoint receives:

- native-to-HF parity;
- the unchanged greedy selector, regression population, and release probes;
- the complete 615-prompt development diagnostic;
- a reference-blinded semantic comparison against V8 step 200;
- inspection for role continuation, user-turn generation, loops, contradiction,
  and failure to stop.

Validation loss cannot select. A checkpoint must show real semantic movement
without catastrophic conversational mechanics to become the Phase S parent.
If no checkpoint improves on V8, Phase S is not run and the objective hypothesis
is rejected.

## Phase S: response-policy recovery

Only a selected Phase M checkpoint may enter Phase S. Phase S reuses the same
reviewed V10 corpus with:

- assistant-only SFT;
- first-four assistant targets weighted 4x;
- terminal EOS weighted 2x;
- the exact row-matched U1 RCR-UL branch at weight 0.5;
- a fresh optimizer;
- at most 200 steps with 50-step checkpoints;
- learning rate no higher than 5e-6.

Phase S exists to recover “answer as the assistant and stop” behavior after the
all-token bridge. It is not allowed to rescue a Phase M checkpoint that showed
no semantic gain.

## Selection and public evaluation

A V11 checkpoint must beat V8 on reference-blinded semantic behavior and retain
sampled stopping and loop resistance. It must improve actual latest-turn
answers across multiple families, not merely produce longer text or a lower
loss.

Only a local winner may be uploaded to Hugging Face, deployed to the public
server, registered on BLAH, and entered into a fresh BLAH run. The exact BLAH
prompts and judge answers remain evaluation-only. Discord receives an update
only after a genuine same-input output improvement.

If V11 fails, the next intervention is a new synthetic corpus organized as
linked contrast families—base cases, paraphrases, minimal changes, updates,
hard negatives, and required invariants—rather than another pool of independent
polished answers.

## Storage and recovery

All research metadata, outputs, and reviews live on the mounted research drive.
Unselected weights remain on the dedicated GPU pod. Only a selected native
checkpoint is mirrored locally. Retained local project artifacts are checked
against the operator's 15 GiB pause threshold before any weight copy.

