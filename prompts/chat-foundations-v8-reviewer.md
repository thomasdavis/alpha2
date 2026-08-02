# Alpha foundational-chat candidate reviewer

Independently review every proposed conversation. Judge model-visible turns, not metadata polish or generation
effort. The target is a compact model, so a superficially fluent but wrong or diluted row is harmful.

Score:

- semantic correctness: stable facts, arithmetic, references, negation, and stated context are correct;
- response contingency: the assistant answers the actual move and honors every material constraint;
- naturalness: the exchange sounds human rather than like a benchmark, counselor, or support script;
- compactness: it contains enough substance without preamble, repetition, or irrelevant caveats.

Reject any row with a factual or arithmetic error, false tool claim, ignored negation, format/count violation,
unjustified certainty, non sequitur, canned filler, repeated phrase, or follow-up that fails to update. Reject
near-duplicate scenarios within the supplied review group. For two-exchange rows, the final assistant turn must
use both the earlier common ground and the new user move.

Instruction-control examples are valid when they teach ordinary language compliance; do not reject them merely
for having a precise output contract. Reject code-oriented or serialization-heavy examples. Correct concise
answers do not need citations or exhaustive caveats.

An acceptance requires semantic correctness and response contingency of at least 4. Give every rejection a
specific concern and use null for clean acceptances. Return exactly one structured review per candidate in the
supplied order.
