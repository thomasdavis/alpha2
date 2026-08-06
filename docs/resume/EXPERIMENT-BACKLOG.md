# Ordered repair and experiment backlog

No item in this document authorizes a run. It defines the work that should be proposed if the user
later reopens Alpha.

## Scientific objective

Produce a model that starts a relevant answer reliably, remains coherent under its own generated
history, terminates cleanly, and passes the unchanged D3 gate. Lower teacher-forced loss is supporting
evidence, not the objective.

## 2026-07-31 repair-v2 finding

The previous backlog's initiation and data-order work was implemented and tested. Repair v2 answered all 96
development prompts at every selector checkpoint, from both the selected checkpoint and the clean base. That
success did not yield effective conversation: exact-comparison loops increased, EOS control weakened, and fixed
qualitative panels showed semantic nonanswers and repeated-phrase attractors.

The next problem is therefore not “make the model emit any token.” It is:

- make the response contingent on the user's actual move;
- maintain semantic direction under the model's own generated history;
- terminate after completing the answer rather than after a template fragment or repetition attractor;
- preserve ordinary short social dialogue while improving instruction and conceptual continuation;
- demonstrate improvement on new families rather than repeatedly fitting the visible v2 outputs.

No item below authorizes GPU spend. The next proposal must introduce a genuinely new, single declared
intervention and a fresh development selector.

## Historical P0 — implemented before repair v2

### 1. Deterministic SFT shuffling

Replace the monotonic source-ordered walk with a deterministic epoch permutation bound to:

- corpus SHA-256;
- seed;
- epoch number;
- train/validation assignment;
- launcher contract.

Acceptance:

- every training example appears exactly once per epoch;
- no duplicate or missing indices;
- restart at any batch reproduces the same future order;
- train and validation identities remain disjoint;
- a unit test proves identical permutation across processes and a changed permutation across epochs.

Do not use a hand-maintained source-name switch. The mechanism must operate from dynamic manifest
metadata and stable example identities.

### 2. Source-balanced sampling

The old corpus was 88 percent SmolTalk by rows and consumed in source blocks. Add a manifest-driven
sampler that can cap per-source dominance while preserving exact provenance.

Candidate policy to test locally:

- select source weights from counts and measured quality, not a hardcoded string table;
- interleave sources within batches;
- report effective row and supervised-token shares;
- preserve an unweighted control path.

The sampler contract must include the selected weight vector and manifest hash.

### 3. Measure and protect answer initiation

Add an evaluation that records, for each prompt:

- EOS rank, logit, probability, and margin over the best content token at the first assistant position;
- first 1, 4, 8, and 16 generated tokens;
- prompt token length and source;
- whether generation ended immediately, looped, leaked roles, or reached useful content.

Possible training changes, tested one at a time:

- bounded extra weight on the first few assistant targets;
- conversation-normalized loss so long answers do not dominate;
- a minimum-answer-length auxiliary objective derived from the data;
- short-dialogue curriculum before the broad mixture.

Do not globally ban EOS or suppress it in the server. That would hide the learned failure and create
non-terminating loops.

### 4. Make generation quality a checkpoint selector input

The future selector must rank checkpoints using a sealed non-frozen development set and preserve:

- structural response rate;
- immediate-EOS rate;
- semantic verdicts;
- repetition and role leakage;
- teacher-forced loss;
- exact prompt and output hashes.

The frozen 100-chat/200-QA set remains final admission only and must not be tuned against.

### 5. Add loader and selection regression tests

Required tests:

- shuffle completeness, determinism, resume continuity, and epoch change;
- source mixing across consecutive batches;
- assistant masks unchanged by shuffling;
- padding remains weight zero;
- answer-start metric recomputation from raw logits;
- selector rejects an attractive single sample when aggregate behavior regresses;
- selector rejects immediate-EOS improvement achieved by non-terminating loops.

## Historical P1 — completed and rejected by repair v2

The clean-base and continuation controls described below have now been executed in the bounded v2 form. Neither
beat the selected baseline. Do not rerun this matrix unchanged.

Preferred starting point: the clean base-pretrain step 61,036 checkpoint, not terminal SFT.

Run a small, predeclared matrix in sequence, never concurrently:

1. Control: old loss recipe with deterministic shuffling only.
2. Source balance: same as control plus manifest-driven source interleaving.
3. Start weighting: winning data recipe plus one bounded answer-start weighting choice.

Each candidate should receive exactly the same number of supervised assistant tokens and the same
development prompts. Stop early if immediate EOS or loops cross the rejection thresholds in
ACCEPTANCE-GATES.md.

The first paid pilot exists to identify a direction, not to finish a model. No flagship continuation
may start until one candidate beats the archived baseline on aggregate generation and preserves
mechanical correctness.

## Next candidate interventions to research locally

These are hypotheses to study and pre-register, not a menu to sweep:

1. **Hard-negative continuation learning:** from the same conversation prefix, distinguish a directly contingent
   continuation from a polished but irrelevant, parroting, or repetition-prone continuation. The negative must be
   natural enough that superficial formatting cannot identify it.
2. **Short-horizon sequence-stability objective:** directly measure whether the model's own first generated tokens
   increase the probability of a coherent completion rather than a repeated phrase. Any auxiliary loss must be
   validated against real free generation and must not globally ban repeats.
3. **Response-policy balance:** measure direct answer, clarification, acknowledgement, continuation, and closure
   behavior dynamically from the data. Avoid a hardcoded string policy table or a universal follow-up-question
   signature.
4. **Semantic-contingency curriculum:** concentrate on paired user moves that differ minimally but demand different
   replies, with whole-family holdouts and same-words/different-intent controls.
5. **Foundation adequacy control:** determine whether the base has enough general language structure to learn the
   intervention under a one-GPU budget. Treat this as a controlled comparison, not a size benchmark.

Before choosing one, perform no-training analyses over v2 hidden states/logits and compare loop-onset versus
stable responses. The analysis must identify a testable mechanism, not merely a new corpus preference.

## Later architecture and capacity decision

If a properly controlled semantic-contingency intervention remains unlearnable, decide explicitly between:

- a smaller, higher-quality and shorter curriculum matched to the current foundation; or
- a different one-GPU-feasible architecture or foundation with a separately budgeted pretrain.

Do not silently expand model scale or corpus cost inside an SFT repair. That is a new program.

## Experiments not worth running first

- another unmodified epoch from step 30,322;
- changing temperature or top-k and calling that training improvement;
- globally blocking EOS for a minimum number of tokens;
- selecting checkpoint 28,500, which does not exist;
- tuning directly against the frozen final set;
- judging from one greeting;
- changing multiple data, objective, optimizer, and architecture variables in one run;
- using a framework other than Alpha for training.

## Expected first implementation files

- packages/train/src/data.ts — deterministic permutation and resumable cursor.
- packages/train/src/trainer.ts — contract wiring and source/answer-start metrics.
- packages/model/src/gpt.ts — only if a bounded weighting scheme is admitted.
- packages/tests/src/sft-masking.test.ts — preserve mask invariants.
- new focused tests for shuffle/resume and generation selection.
- a new launcher; never mutate the old one-epoch contract in place.

## Documentation required for every candidate

Each candidate directory must contain:

- hypothesis and single changed variable;
- source commit and clean/dirty state;
- input and checkpoint hashes;
- exact token/step budget;
- guard and termination policy;
- raw metrics and generation outputs;
- machine verdict and human rationale;
- spend;
- explicit decision: reject, retain for comparison, or admit to the next gate.
