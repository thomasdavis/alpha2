# Alpha public BLAH baseline audit

Date: 2026-08-02

**Decision:** the currently published `Alpha 60M Chat` is a useful mechanics
baseline, not an acceptable conversational model. It must not be treated as a
winner merely because it returns nonempty text or occasionally produces a good
greeting. The latest complete 24-evaluation run scored `0.3229166667`, and its
raw calls show failures in elementary instruction following, semantic
completion, factual association, negation, arithmetic, uncertainty, and
repetition control.

No new BLAH model was registered from this audit. Publication remains gated on
a locally frozen candidate that beats this baseline on ordinary dialogue and
semantic behavior, not only held-out loss or response initiation.

## Snapshot and scope

The read-only BLAH snapshot used here is preserved at:

```text
/mnt/donto-data/donto-resources/benchmarks/alpha-blah-evals-20260802/
```

It contains the model record, 500 recent raw call logs, all 307 result records
then visible, 24 evaluation definitions, and 22 run records. `SHA256SUMS`
identifies the exact bytes. The API-wide log statistics at capture time were:

- 962 calls;
- 26 recorded call errors;
- 10,897 ms mean latency;
- 371 evaluation, 320 health-check, 135 chatroom, 109 chat, 21 game, and 6
  direct-API calls.

The 500-call page itself contained 225 health checks, 182 evaluation calls, 86
chatroom calls, and 7 chat calls. None of those 500 rows had a transport error
or empty response. This distinction matters: the deployed model now reliably
produces *something*, but the content is usually not dependable.

## Latest complete run

Run `VzqH9rosl0_DEwkVa1PBNA` completed all 24 available evaluations:

| Measurement | Result |
| --- | ---: |
| Mean score | 0.3229166667 |
| Score 1.0 | 5 / 24 |
| Score 0.5 | 1 / 24 |
| Scores 0.1-0.4 | 11 / 24 |
| Score 0.0 | 7 / 24 |
| API `finish_reason=stop` | 24 / 24 |
| Judge rationales mentioning repetition or looping | 7 / 24 |
| Median output length | 17 tokens |
| Mean output length | 26.58 tokens |
| Minimum / maximum output length | 4 / 256 tokens |
| Responses at the runtime ceiling | 1 / 24 |

The score is not explained by widespread request failures or truncation. Only
one response reached the 256-token ceiling. Most bad answers ended normally and
briefly. The dominant problem is therefore what the model chooses to say, not
the platform cutting otherwise-correct answers off.

The five full-credit cases show the narrow competence that does exist: simple
greeting behavior, literal copying, and a few memorized or high-frequency
associations. They do not compensate for failures on equally short prompts.

## Failure families with raw evidence

### 1. It recognizes a task frame but cannot supply the missing semantic item

The most characteristic failure is a fluent shell around an absent answer:

> Prompt: `What is 1 + 1?`
>
> Alpha: `1 + 1 is a simple equation that can be used to solve problems. It's a simple equation that can be used to solve problems.`

> Prompt: `What is the opposite of hot?`
>
> Alpha: `The opposite of hot is the opposite of hot. The opposite of hot is the opposite of hot.`

> Prompt: `Complete this saying: The early bird catches the`
>
> Alpha: `The early bird catches the`

> Prompt: `Name one animal.`
>
> Alpha: `What animal is it?`

These are not refusals. Alpha understands that the prompt concerns an equation,
an opposite, a completion, or an animal, but fails to cross from the task's
language into the requested answer. This is consistent with a base that learned
surface continuations but received too little broad semantic exposure.

### 2. It fails exact output constraints

> Prompt: `Answer in exactly three words: what colour is the sky on a clear day?`
>
> Alpha: `The sky on a clear day is a clear day. It is a clear day.`

> Prompt: `List exactly four colours, one per line, nothing else.`
>
> Alpha: `The first one is the blue, the second is the blue, and the third is the blue.`

> Prompt: `Reply with only this JSON object and nothing else: {"colour": "the colour of grass"}`
>
> Alpha: `I'm sorry, I didn't mean to make you feel like this. I just wanted to make sure that you were able to see it.`

The JSON case is not merely malformed JSON; it ignores the requested operation
entirely and falls into an unrelated apology template. Training data should
therefore include matched natural-language constraints, exact copying, short
closed-form answers, and hard negative continuations—not only more long helpful
answers.

### 3. It makes elementary semantic and factual substitutions

> Prompt: `What is the capital of France?`
>
> Alpha: `The capital of France is the United States of America.`

> Prompt: `What is 7 plus 6?`
>
> Alpha: `7 plus 6 is a prime number. It's a prime number.`

These failures are relevant even though Alpha is not intended to memorize the
world. A conversational model still needs basic language grounding and common
relations: a capital should be a city, a sum should be a number, and an animal
request should elicit an animal. "Fact-light" cannot mean relation-light.

### 4. It does not reliably express epistemic limits

> Prompt: `What did I eat for breakfast this morning?`
>
> Alpha: `I was just trying to get some breakfast. I was just trying to get some breakfast.`

Instead of saying that it cannot know the user's private past, Alpha invents a
first-person frame and repeats it. This is simultaneously an uncertainty,
perspective, and conversational-role failure.

### 5. It degenerates after finding a locally correct phrase

In fictional-context evaluations Alpha can copy the supplied answer and then
repeat a templated clause until the ceiling. A representative response begins:

> `The capital of Zephyria is Marrowind. Marrowind is a city in the city of Marrowind. Marrowind is a city in the city of Marrowind...`

The answer token is present, so isolated exact-match scoring would overstate the
capability. The raw output correctly reveals degeneration. This is why Alpha's
selection gate counts loops and inspects full continuations.

### 6. Live stochastic greetings are unstable

The health check asked the identical prompt `Say hello in one sentence.` 225
times in the captured 500-call page. Recent replies included:

- `Say hello in another sentence.`
- `What's a sentence in another sentence?`
- `I don't know what to say.`
- `I can't do that.`
- `Okay, I'll try to say hello.`
- `Hello! How can I help you today?`

`Say hello in another sentence.` alone occurred 47 times across the 500 recent
responses. The model can generate a good greeting, but it does not robustly
associate the instruction with greeting behavior at BLAH's temperature `0.7`.
Repeated sampled evaluation is therefore mandatory for any replacement; one
greedy golden sample is insufficient.

## Prompt-template and termination diagnosis

The platform sends an OpenAI-style user message to Alpha's inference endpoint.
The runtime renders it as:

```text
<|user|> PROMPT <|assistant|>
```

This is the same role-marker convention used by the training corpus, including
system-message folding. The failures above are not explained by an obvious
BLAH/chat-template mismatch.

Inspection did reveal a separate serving defect: serialized multi-turn
training dialogues place the next `<|user|>` directly after the assistant's
answer, while the old live runtime stopped only on `<|end_of_text|>`. It could
therefore stream a model-generated user turn. Commits `4f704c7` and `bdd927c`
make the runtime stop at any atomic user, assistant, or whole-dialogue boundary
without weakening the frozen strict-EOS model gate. This prevents role text
from leaking to clients, but it cannot turn a circular or false answer into a
correct one.

## What training data the evidence asks for

The observed deficits motivate concrete, family-linked additions rather than a
generic request for more chats:

1. **Answer-bearing microdialogues:** arithmetic, opposites, category members,
   sentence completion, and common relation ranges, with short direct targets.
2. **Constraint transformations:** the same underlying answer rendered as one
   word, exactly three words, a numbered list, one item per line, and literal
   copying, paired with polished wrong-format negatives.
3. **Epistemic perspective:** private facts, missing evidence, supplied
   evidence, correction, and first/second/third-person attribution.
4. **Stop-and-yield examples:** many correct one-clause answers followed by the
   next user turn in the serialized stream, so the role boundary is frequent
   and unambiguous.
5. **Anti-loop contrast pairs:** a correct concise continuation versus the same
   continuation followed by duplication, circular definition, or topic drift.
6. **Sampled robustness cases:** multiple valid surface realizations for the
   same prompt so the model is not dependent on one brittle memorized opening.

These additions can improve post-training, but the public-comparator audit also
shows that data format is not the whole gap. The closest coherent public
SmolTalk SFT model sits on a base exposed to orders of magnitude more
pretraining tokens than Alpha's clean base. The ongoing V12 experiment tests
the public full-sequence packed recipe without pretending that it recreates
that foundation.

## Replacement gate

A new BLAH model/version may be registered only after a checkpoint:

- beats the current frozen baseline on full-response selector and regression
  suites;
- improves actual semantic panels, not only structural counts;
- remains stable under repeated temperature-0.7 sampling;
- answers ordinary short prompts directly;
- avoids role leakage through the corrected runtime;
- is exported with native/Hugging-Face logit parity;
- and is published under a new immutable model/checkpoint identity.

Until then, the current public model remains the honest baseline and retains
its `quality_gate=FAIL` label.
