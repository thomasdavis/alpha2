# Alpha chat foundation v6

Date: 2026-08-02

## Why this intervention exists

V4 and v5 both reject the hypothesis that a few thousand excellent semantic
conversations can install broad conversational competence by themselves. V4
continued the public roleplay-heavy chat checkpoint; v5 repeated the same
curriculum from the clean pretrained parent. Both produced fluent-looking but
semantically circular answers. Changing initialization altered repetition and
stopping, but did not teach transfer to ordinary unseen questions.

V6 supplies the missing foundation: broad direct instruction, explanations,
ordinary chat, correction, constraint following, and task diversity from the
already staged and audited canonical SFT corpus. Narrow semantic refinement is
deferred until a checkpoint can first answer ordinary questions coherently.

## Data boundary

The source is the immutable 511,428-conversation SFT v2 corpus. A streaming
builder removes an entire conversation when any user turn exactly matches a
normalized prompt from either:

- the visible development suites bound by the semantic-repair freeze; or
- the frozen 24-result BLAH baseline.

This is contamination control, not topic filtering. All other source bytes are
preserved verbatim. The builder does not inspect the sealed final suite, apply a
manual keyword taxonomy, or train on BLAH judge reasoning. Common public prompts
such as greetings and elementary facts are dynamically detected and excluded
when they match exactly.

## Training boundary

- initialization: clean pretrained parent;
- maximum steps: 8,000;
- checkpoints: every 1,000 steps;
- batch: 16;
- context: 512;
- optimizer: reset AdamW;
- learning rate: `2e-5` to `2e-6`, 500-step warmup;
- assistant-start weighting: 8x over four tokens;
- terminal-token weighting: 2x;
- no SODA-dominated repair replay;
- no v4 semantic micro-corpus repetition in this stage;
- no validation-loss selection.

The finite run samples only part of the large corpus. That is intentional: the
checkpoint trajectory tests whether diversity and direct instruction repair the
basic capability before committing to a full pass.

## Promotion gate

A candidate must improve actual answers, not merely output length or loop
counts. Required evidence includes:

- direct response to ordinary instructions;
- correct short explanations and distinctions on unseen prompts;
- natural conversational contingency;
- calibrated uncertainty rather than empty scripts;
- clean EOS behavior and no role leakage;
- matched regression and qualitative-panel inspection;
- exact HF/native parity.

BLAH is rerun only after local held-out selection. A BLAH score does not override
a semantic failure, and exact-overlap exclusions are recorded so the public run
cannot be presented as hidden generalization.

The sealed final remains untouched until one checkpoint clears all visible
selection gates.
