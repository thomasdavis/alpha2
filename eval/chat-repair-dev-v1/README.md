# Chat repair development suite v1

This is the public **development and checkpoint-selection** suite for Alpha's
response-repair run. It is not the sealed final evaluation and must never be
reported as one.

The 48 hand-authored cases cover response initiation, ordinary conversation,
length control, pragmatics, ambiguity, conceptual distinctions, correction,
local terminology, evidence, and multi-turn state. The `reference` field is a
behavioral expectation for human inspection; the existing `eval-frozen`
command ignores it and measures only greedy free-generation structure,
stopping, role leakage, and repetition. The one-row QA file merely satisfies
that command's generic input contract; closed-book QA is not a selection target
for this repair.

Checkpoint selection uses aggregate behavior across this entire suite. No
single prompt may authorize more training. After selection, the untouched
historical frozen suite remains the final structural and semantic check.
