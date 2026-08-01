# Alpha semantic-chat repair v4 — local preflight

**Status:** data hypothesis and generation smoke only; no GPU run launched

**Product objective:** improve actual conversational meaning while preserving the selected model's reliable
response initiation and natural stopping.

## Why v3 is not the next release

Repair v3 supplied causal evidence that rollout-conditioned repetition unlikelihood works mechanically. At U1
step 400 it reduced fresh-development loop flags from 35 to 6 and introduced no new loop on the paired baseline
population. A matched live probe through `evals.blah.dev`, however, showed no semantic gain. It remained unable
to explain a simple biological concept, distinguish a promise from a prediction, honor an explicit request for
empathy without advice, explain a familiar ambiguity, or sustain a basic identity discussion. U1 step 400 is
therefore rejected for promotion. More anti-loop pressure would optimize the symptom rather than the product.

## Newly measured data mismatch

The selected step-1,200 model was trained from the clean pretraining checkpoint on a 34,880-conversation repair
corpus intended to emphasize short natural dialogue. Its actual composition is:

| Source                           |   Rows | Share |
| -------------------------------- | -----: | ----: |
| SODA                             | 30,000 | 86.0% |
| SmolTalk2 everyday conversations |  2,260 |  6.5% |
| SmolTalk concise subset          |  1,436 |  4.1% |
| OASST2                           |  1,184 |  3.4% |

Only 519 rows contain one user turn and one assistant turn. The builder required four or more turns for the
SmolTalk source, excluding the direct instruction/answer form by construction. A read-only census of the
SmolTalk portion of the canonical SFT corpus found 186,043 one-exchange rows, including 58,457 whose user and
assistant turns both fit a compact 96-word envelope. Many are code or arbitrary formatting tasks and cannot be
accepted wholesale, but the measurement proves that the repair selected conversational surface form at the cost
of direct semantic instruction.

SODA is a legitimate synthetic social-dialogue resource, but Alpha's sampled rows often cast the assistant as a
friend, relative, teacher, or fictional participant. They teach conversational rhythm, not a consistently
helpful interlocutor. Several live failures resemble that policy: “What's wrong with you?” in response to a user
requesting understanding, or a generic question where a distinction was requested.

## Research support

- TinyStories demonstrates that restricted-vocabulary, high-quality synthetic data can teach coherent generation
  to very small models: <https://arxiv.org/abs/2305.07759>.
- Baby Llama shows that distillation can materially improve a compact student trained under a small-data regime:
  <https://arxiv.org/abs/2308.02019>.
- Dialogue Distillation combines augmented dialogue with ranking and teacher distillation specifically to prevent
  noisy pairs from degrading a dialogue model: <https://aclanthology.org/2020.emnlp-main.277/>.
- CLASS-IT reports that sequential conversational and lecture-aligned curricula can outperform a merged small-model
  curriculum, while also warning that interaction tuning does not automatically create broad transfer:
  <https://arxiv.org/abs/2510.25364>.
- SmolLM3's released recipe treats everyday/multi-turn weakness as a targeted data problem and generates missing
  domains with a stronger teacher rather than assuming aggregate SFT volume is enough:
  <https://huggingface.co/blog/smollm3>.

These works motivate high-quality synthetic distillation and staged curricula. They do not establish that the
planned intervention will work for Alpha; the pilot must decide that empirically.

## V4 hypothesis

Starting from the selected public checkpoint, a compact curriculum dominated by concise, semantically checked
teacher conversations will improve response contingency and conceptual correctness more than another pass over
the role-play-heavy repair corpus. A limited replay of natural multi-turn dialogue should preserve conversational
rhythm. Repetition treatment is deferred unless the new positive curriculum actually produces a loop regression.

The first intervention changes the positive data distribution. It does not change architecture, decoding,
tokenizer, chat template, loss implementation, or add v3 unlikelihood.

## Candidate data design

The pilot corpus will combine:

1. newly generated, short synthetic conversations from `gpt-5.4`, using a structured-output schema;
2. the complete 2,260-row SmolTalk2 everyday split, subject to the existing structural/token audits;
3. a small, provenance-preserving sample of OASST2 or another natural dialogue source only after quality review;
4. no bulk SODA replay in the first pilot.

Synthetic examples focus on ordinary explanation, conceptual distinctions, pragmatic intent, empathy without
forced advice, language and ambiguity, evidence-sensitive reasoning, correction, disagreement, and delayed
multi-turn use. Categories allocate generation; they are not model-visible labels and are not runtime rules.

The generator returns structured candidate objects using
`schemas/chat-semantic-repair-candidates.schema.json`. Raw candidates, rejected candidates, generation logs,
teacher identity, prompt revision, hashes, review decisions, and rendered exposure remain preserved. No JSON is
scraped from free text.

## Contamination control

The generator does not receive development answers or sealed-final material. Before release construction,
candidate user turns must be checked against every development prompt by normalized exact hash and semantic
similarity. Entire candidate families are rejected when they collide. The six live Blah probe concepts used to
make the v3 release decision are also reserved from training so they remain useful regression evidence.

## Generation economy

`gpt-5.4` is the initial teacher because the operator selected it for bulk synthetic generation. Generation is
batched and resumable. The orchestration layer requests complete structured batches, validates exact identifiers
and turn alternation, and retains raw failures rather than repeatedly spending on rows that already passed.
Reviewer calls operate over batches, with escalation only for ambiguity or disagreement; generation count is not
treated as accepted count.

The initial target is a few thousand accepted conversations, not a giant corpus. Scaling is conditional on a
measured free-generation gain. This keeps Codex subscription use and GPU spend proportional to evidence.

## Evaluation and stop decision

The candidate is compared with the public step-1,200 checkpoint at identical decoding settings on:

- the frozen v3 fresh-development and regression populations;
- the preserved blinded conversational panel;
- a held-out semantic-contingency suite with whole-family isolation;
- matched calls through Blah on ordinary explanation, empathy, ambiguity, and conceptual discussion;
- loop, role-leak, nonempty, EOS, repetition, and answer-length diagnostics.

Success requires better meaning, not merely fewer loops or more words. A candidate is rejected if it improves
direct answers by becoming lecture-like, ignores user intent, introduces repetition, or loses ordinary social
conversation. Hugging Face and the public Blah registration remain unchanged until an honestly superior
checkpoint passes the release gate.
