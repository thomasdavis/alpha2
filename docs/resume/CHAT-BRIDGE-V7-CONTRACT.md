# Alpha direct-semantic bridge v7

**Status:** local implementation; corpus and GPU run are not yet evidence

## Decision

V6 is rejected. Starting from the clean pretrained checkpoint and exposing it to 8,000 steps of the broad
canonical SFT distribution did not recover the public model's mechanics or install reliable meaning. Across all
eight declared checkpoints, fresh-selector structural passes stayed between 58/96 and 68/96 with 47–57 loops;
the selected public baseline remains 83/96 with 35 loops. Step 8,000 also answered scurvy with niacin, invented a
Mexican “Parmesan cuisine,” fabricated a Tokyo venue, and failed a basic arithmetic update. Loss did not predict
useful free generation.

The next intervention preserves the one component that has worked causally: v3 U1 step 400 reduced fresh loops
to six while keeping response initiation and stopping strong. U1 did not improve meaning, so v7 adds positive
semantic supervision without changing its architecture or adding more repetition pressure.

## Corpus contract

V7 combines:

- all 5,104 reviewed v4 semantic-chat training conversations;
- 40,000 compact single-user/single-assistant conversations selected from the canonical SmolTalk source span;
- a development set consisting of the 265 reviewed semantic rows plus 2,000 disjoint direct rows.

Direct rows are eligible by provenance, two-turn conversational shape, and exact Alpha-tokenizer length at or
below 384 tokens. Selection and interleaving use SHA-256 order. There is no topic-name, answer-key, programming,
or hand-written semantic lookup. This is intentionally a broad direct-answer bridge; its residual domain mix is
measured rather than silently described as philosophy-only data.

Every visible v4 evaluation user turn, every frozen 24-item BLAH prompt, and every semantic-development user turn
is excluded by normalized exact match. The inherited sealed final is not opened. Train and development rows are
disjoint by full conversation hash, and every selected row retains origin, source line where available, exact
token count, split, and deterministic order in the catalog.

## Training contract

- initialization: v3 U1 step 400, SHA-256
  `0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb`;
- fresh AdamW optimizer; no optimizer or RNG continuation;
- 2,800 steps, batch 16, block 512: approximately one pass over 45,104 train conversations;
- learning rate `1e-5` to `1e-6`, 200 warmup steps;
- checkpoints every 400 steps;
- deterministic conversation shuffle, equal conversation weighting, first four assistant tokens at 8x, EOS at
  2x, and no unlikelihood term;
- one GPU path at a time, deterministic evaluation, no selection by loss.

## Release decision

Every checkpoint must pass exact Alpha/Hugging Face export parity and the unchanged visible free-generation
suites. A candidate must materially improve response meaning and contingency while staying near U1's loop and
termination behavior. Concrete panel answers—not aggregate loss—decide whether any checkpoint is eligible.

If v7 improves meaning but reintroduces repetition, the next bounded operation may be a short U1-style
repetition cleanup initialized from the selected v7 checkpoint. If v7 merely produces fluent wrong answers,
generic lectures, or superficial verbosity, it is rejected and no cleanup is justified.

The sealed final, Hugging Face model, BLAH registration, and Discord remain untouched until a candidate clears
the visible gate.
