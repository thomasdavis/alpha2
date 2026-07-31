# PRD-14 — Conversational response-policy control plane

**Status:** planning contract derived from D5 evidence; no migration or backfill authorized before D5 closeout

**Applies to:** family blueprints, generation allocation, natural-language prompt compilation, human review,
release balancing, and ordinary-chat/AlphaPact evaluation

**Authority:** PRD-00, PRD-01, PRD-02, PRD-03, PRD-04, PRD-05, PRD-06, PRD-09, and future D5 adjudication

**Current aggregate evidence:** 48 current D5 candidates, 48 distinct free-form `intendedResponsePolicy`
strings, and zero normalized `response_policy_target` rows

**Scientific authority created:** none; aggregate structural evidence is not a judgment about any candidate

## 1. Purpose

Alpha is meant to be chatty in the sense of responsive, momentum-preserving, adaptive, and present—not in the
sense of always long, warm, inquisitive, or opinionated. A synthetic-data factory cannot reliably teach that
behavior if every item carries an unconstrained prose instruction that cannot be counted, compared, composed,
or audited.

This PRD defines a versioned and extensible response-policy control plane. It lets the orchestrator express
what conversational move is appropriate, why it is appropriate, which competing moves are prohibited, and how
the policy should change with context. It also separates:

- what the blueprint intended;
- what natural-language instruction the worker received;
- what response the worker actually produced;
- what a deterministic instrument observed;
- what a human reviewer judged; and
- how a release balanced the resulting evidence.

The control plane is researcher-side structure. Alpha still sees and produces ordinary natural language.

## 2. D5 finding and interpretation boundary

The canonical D5 ledger currently contains:

```text
current candidate versions:             48
distinct intendedResponsePolicy strings: 48
response_policy_target rows:              0
```

This proves that the current policy field is effectively one-off prose and that the normalized target table is
not yet used. It does not prove that the instructions are poor, that the responses violate them, or that any
candidate should be accepted or rejected.

The original strings remain immutable candidate evidence. A future normalized mapping supplements them; it
never rewrites or replaces them.

## 3. Design principles

### 3.1 Policy is purpose-relative

The right move depends on the user's apparent goal, the current Question Under Discussion, shared ground,
missing evidence, conversational stage, and social stakes. “Ask a question” or “answer directly” is not good in
isolation.

### 3.2 Policies compose

A response may need to answer directly, preserve one uncertainty, challenge a hidden premise, give one
counterexample, and then stop. The data model represents compatible moves and their ordering rather than
forcing one label per response.

### 3.3 Policies have negative constraints

The target includes what must not happen: do not ask ritualistically, do not universalize a local term, do not
lecture, do not invent ambiguity, do not bury a direct answer, and do not claim evidence that is absent.

### 3.4 Design and observation are different evidence

An intended `challenge` does not make the output a challenge. A human may observe hedging, accommodation, or a
lecture. Both facts remain in separate rows.

### 3.5 Taxonomy is data, not a code enum

Policy concepts, dimensions, relations, examples, and proposals are versioned SQLite records. Code validates
schema and referential integrity, but does not contain a permanent switch statement over policy names. New
recurring moves can be proposed and reviewed without a schema migration.

### 3.6 Natural language stays natural

The worker sees a compiled plain-language instruction appropriate to the task. It does not see internal policy
IDs, JSON, database columns, or a checklist to parrot. The compiler varies wording while preserving the bound
policy meaning.

### 3.7 Distribution matters

Even individually reasonable responses can form a bad curriculum if too many ask questions, over-explain,
agree, hedge, or use the same rhythm. Policy allocation and observed behavior are audited at family, batch,
cohort, and release levels.

## 4. What a response policy is

A response policy is a purpose-bounded contract describing the conversational action Alpha should take next.
A policy instance conceptually contains:

```text
<trigger, obligation, optional moves, prohibited moves, ordering, depth, initiative,
 question need, closure, epistemic stance, interpersonal stance, evidence behavior,
 common-ground effect, rationale, scope>
```

It is not:

- a canned sentence template;
- a personality trait;
- a topic label;
- a philosophical doctrine;
- a universal safety rule;
- a required final question;
- a fixed word count; or
- a substitute for the hidden semantic contract.

## 5. Policy dimensions

The dimensions below seed the ontology. Their values remain versioned data and may expand through reviewed
proposals.

### 5.1 Primary conversational obligation

What useful move must the response accomplish?

- orient to the interaction;
- answer the explicit question;
- acknowledge and ground;
- distinguish senses or cases;
- explain or exemplify;
- test an interpretation;
- clarify a decisive uncertainty;
- challenge a premise;
- construct or test a counterexample;
- repair or retract locally;
- compare frameworks;
- attribute a claim or perspective;
- identify missing evidence;
- formulate a retrieval question;
- summarize current common ground;
- recover a prior thread;
- propose or negotiate terminology;
- express a provisional judgment;
- invite a useful continuation; or
- close naturally.

These are initial concepts, not a closed code list. A new move must state its boundary and nearest existing
policies.

### 5.2 Answer timing

- answer before qualification;
- give a minimal conditional answer;
- answer after one necessary clarification;
- explain why the question cannot yet be answered;
- defer pending evidence;
- refuse the premise while answering the underlying need; or
- continue an already established inquiry without restating the question.

This dimension prevents “it depends” or a question from becoming a universal opening.

### 5.3 Initiative

- reactive only;
- add one useful next distinction;
- offer an optional example;
- surface one high-value fork;
- propose a reframing;
- retrieve or request evidence; or
- deliberately avoid expanding the scope.

Initiative is not verbosity. One sentence can contribute a new foothold.

### 5.4 Question necessity

Question behavior must be selected for a reason:

- prohibited because the answer is already complete;
- unnecessary and should be omitted;
- optional after a substantive partial answer;
- useful for a real fork;
- necessary before a responsible answer;
- necessary for missing evidence; or
- rhetorical/question-form language that is not an information request.

Observed questions are separately classified using the blinded D5 review categories. A policy target cannot
declare its own question successful.

### 5.5 Closure

- answer and stop;
- concise natural closure;
- leave an optional next foothold without a question;
- ask one justified question;
- summarize an unresolved set;
- explicitly park the issue;
- hand off to retrieval/evidence gathering; or
- continue because the interaction is mid-trajectory.

### 5.6 Depth and compression

Depth is relative to the user's need and the conceptual work, not a global word range:

- fragment/backchannel;
- one direct sentence;
- short answer plus reason;
- compact worked example;
- medium exploration;
- deep analysis because requested or necessary; or
- progressive disclosure with a concise first layer.

The exact rendered length remains an observed value. A policy expresses functional depth.

### 5.7 Epistemic stance

- confident within supplied evidence;
- calibrated uncertainty;
- preserve finite alternatives;
- distinguish testimony from endorsement;
- separate observation from interpretation;
- identify missing evidence;
- challenge an unsupported premise;
- treat a local stipulation as local; or
- suspend judgment.

Hedging is not automatically calibrated. Review checks whether uncertainty tracks an actual dependency.

### 5.8 Disagreement and correction

- accommodate a harmless local convention;
- acknowledge then qualify;
- disagree directly with reasons;
- steelman before critique when it materially helps;
- correct a factual or conceptual premise locally;
- preserve a theory-relative alternative;
- reject a false equivalence;
- revise only dependent commitments; or
- admit Alpha's own mistake.

### 5.9 Common-ground effect

- establish a local term;
- reuse an established term efficiently;
- test whether a pact was accepted;
- notice drift;
- recover an earlier distinction;
- revise the pact;
- mark scope change;
- keep an unresolved alternative live; or
- avoid re-explaining already shared material.

### 5.10 Evidence and retrieval behavior

- reason only from supplied evidence;
- attribute each claim;
- compare conflicting sources;
- distinguish valid time and record time;
- request the smallest missing evidence;
- formulate a search query;
- state what retrieval could decide;
- abstain from entity-specific invention; or
- answer conceptually while separating the missing factual premise.

### 5.11 Interpersonal stance

- neutral direct;
- warm but not performative;
- skeptical;
- playful;
- reflective;
- tentative;
- emotionally attentive;
- concise professional; or
- technically precise when the user requests it.

Stance never overrides the substantive move. Style-scrubbed review tests whether value remains.

### 5.12 Adaptation target

- match established vocabulary;
- simplify without condescension;
- increase technical precision;
- shorten after common ground accumulates;
- slow down after confusion;
- respect the user's framing while marking its scope;
- switch example domain; or
- preserve the user's preferred local term.

## 6. Policy relations and composition

Response policies form a graph, not a flat label set.

Required relation types include:

- `broader_than` / `narrower_than`;
- `compatible_with`;
- `usually_precedes` / `usually_follows`;
- `in_tension_with`;
- `prohibits`;
- `repairs`;
- `observational_confusion_with`;
- `requires_context`;
- `specializes_for_evidence`;
- `specializes_for_pact_stage`; and
- `candidate_successor` for reviewed taxonomy evolution.

A policy composition records:

- required components;
- optional components;
- prohibited components;
- order constraints;
- condition under which an optional component activates;
- maximum conversational initiative;
- closure rule; and
- rationale tied to the family/scene purpose.

Composition is validated structurally and later reviewed semantically. The system must not assume every graph
combination is coherent.

## 7. Logical schema

### 7.1 Policy ontology

| Table | Purpose |
|---|---|
| `response_policy` | Stable policy concept |
| `response_policy_version` | Definition, boundary, positive/negative examples, authority, digest |
| `response_policy_dimension` | Extensible dimension definition |
| `response_policy_dimension_value` | Versioned value within a dimension |
| `response_policy_relation` | Typed graph relation between policies or values |
| `response_policy_proposal` | Candidate new policy from repeated uncaptured behavior |
| `response_policy_review` | Human/expert decision and rationale |
| `response_policy_alias` | Search/alignment phrase, not destructive normalization |

Aliases help retrieval but do not collapse semantically different policies merely because their wording is
similar.

### 7.2 Targets and compiled instructions

| Table | Purpose |
|---|---|
| `response_policy_target` | Target for a scene/message/candidate with necessity and rationale |
| `response_policy_target_component` | Required/optional/prohibited policy component and priority |
| `response_policy_target_condition` | Contextual activation condition tied to semantic state or evidence |
| `response_policy_order_constraint` | Required ordering among components |
| `response_policy_compilation` | Exact natural-language instruction compiled for a model call |
| `response_policy_compiler_version` | Versioned compiler/prompt method and digest |
| `response_policy_compilation_basis` | Exact target components used by the compilation |

The existing `response_policy_target` table is a preliminary shape. A future migration may preserve and extend
it, but must not rewrite historical rows or silently reinterpret its `policy_slug` field.

### 7.3 Observed behavior

| Table | Purpose |
|---|---|
| `response_move_observation` | Observed move in a generated or evaluated response |
| `response_move_span` | Exact message span supporting the observation |
| `response_policy_fit` | Target-to-observation match, partial fit, conflict, or not-applicable |
| `question_policy_observation` | Necessity and effect of an actual question |
| `closure_observation` | Answer-and-stop, natural close, canned invitation, truncation, or continuation |
| `initiative_observation` | Useful initiative, scope drift, or inertness |
| `adaptation_observation` | Evidence of register/depth/common-ground adaptation |
| `policy_violation` | Specific required move missed or prohibited move produced |

Each observation records authority: deterministic, model-proposed, human, expert, or executable. Model-
proposed observations are not promoted to human fact.

### 7.4 Distribution and allocation

| Table | Purpose |
|---|---|
| `response_policy_allocation` | Predeclared target distribution by campaign/cohort/release scope |
| `response_policy_allocation_cell` | Policy/dimension target, tolerance, rationale, and weighting |
| `response_policy_distribution_run` | Exact population, method, revision, and timestamp |
| `response_policy_distribution_observation` | Intended or observed counts/rates |
| `response_policy_distribution_finding` | Concentration, absence, mismatch, or uncertain measurement |
| `response_policy_balance_adjudication` | Human decision about whether the distribution is acceptable |

Target distributions need not be uniform. They reflect product purpose: ordinary chat may need abundant direct
answers and answer-and-stop, while an ambiguity cohort legitimately contains more clarifications.

## 8. Evidence layers

For every candidate, the ledger should be able to reconstruct:

1. **Blueprint target:** the structured policy contract chosen before generation.
2. **Compiled instruction:** the exact natural-language policy instruction shown to the worker.
3. **Raw legacy field:** the original `intendedResponsePolicy` string, if any.
4. **Generated behavior:** exact assistant message bytes.
5. **Deterministic observations:** length, question ending, role sequence, and other non-semantic signals.
6. **Critic proposals:** dimension-specific findings with model/prompt provenance.
7. **Blind human observation:** naturalness and question behavior without seeing the target.
8. **Contract-aware human fit:** whether the behavior satisfied the intended policy.
9. **Family distribution diagnosis:** whether siblings cover distinct policies.
10. **Release distribution:** whether selected training material creates an undesirable policy prior.

No layer overwrites another. A mismatch is useful training-factory evidence.

## 9. Generation integration

### 9.1 Allocation

Before surface generation, the allocator chooses policy compositions based on:

- family purpose and semantic pressure;
- scene stage and user need;
- current batch/release coverage;
- missing ordinary-chat behavior;
- risk of question, lecture, or agreement concentration;
- difficulty and model capability; and
- negative-example allocation.

Allocation is stored before the worker call so it cannot be retrofitted to describe whatever the model wrote.

### 9.2 Natural-language compilation

The compiler translates the structured target into concise task-specific prose. It should:

- lead with the substantive obligation;
- include only constraints relevant to this scene;
- state question/closure requirements explicitly when important;
- avoid internal labels;
- vary surface wording under a versioned paraphrase method;
- preserve meaning across variants; and
- record every compiled byte in the model-call ledger.

The compiler is not a giant template map. It uses a versioned model or compositional renderer whose exact
input/output remains reviewable. If a model compiler is used, structured output and independent validation are
required.

### 9.3 Worker output

The worker must answer the conversation, not explain the policy. Outputs that say “I will answer directly,”
name the target, or reveal rubric language receive a leakage finding.

### 9.4 Negative construction

Hard negatives should differ in policy fit while remaining plausible and fluent:

- ritual question versus answer-and-stop;
- warm agreement versus warranted challenge;
- long lecture versus compact conceptual move;
- generic hedging versus finite legitimate alternatives;
- direct but insensitive reply versus appropriate interpersonal adaptation;
- overconfident fact invention versus evidence-aware conceptual answer;
- correct local repair versus collateral conceptual churn; and
- useful initiative versus topic hijacking.

Negatives remain labeled by the specific violation. “Bad response” is insufficient provenance.

## 10. Review integration

### 10.1 Pass A

Blind review continues to judge what the response actually does:

- direct responsiveness;
- conversational naturalness;
- appropriate depth/length;
- question necessity;
- desire to continue;
- substantive value after style removal; and
- observed findings with exact evidence.

No structured target is shown.

### 10.2 Pass B

After campaign-wide blindness gates pass, the target composition and compiled instruction are revealed. The
reviewer judges:

- whether the target was appropriate for the scene;
- whether the compiled prose represented the target accurately;
- whether the response satisfied required components;
- whether it produced prohibited components;
- whether ordering and closure were appropriate;
- whether the policy was overprescriptive; and
- whether a taxonomy gap caused the mismatch.

### 10.3 Pass C and closeout

Family synthesis checks policy diversity and recurring failure. Campaign closeout distinguishes:

- bad policy allocation;
- bad policy definition;
- bad natural-language compilation;
- worker noncompliance;
- reviewer uncertainty;
- missing taxonomy; and
- acceptable context-sensitive variation.

These failure loci imply different repairs and must not be collapsed into “model quality.”

## 11. Release construction

Every candidate selected for a release retains target and observed policy evidence. A release manifest reports:

- target policy/dimension coverage;
- human-observed move coverage;
- target/observation mismatch rates;
- question necessity distribution;
- closure distribution;
- depth and length distribution;
- initiative and adaptation distribution;
- family/provider/prompt concentration;
- negative-policy examples and their allowed use; and
- unresolved policy judgments.

Selection should avoid teaching accidental global rules. In particular:

- not every conceptual answer asks a follow-up;
- not every disagreement starts with praise;
- not every ambiguity triggers clarification;
- not every philosophical issue receives a long essay;
- not every warm response mirrors emotion;
- not every answer supplies a technical term; and
- not every conversation must remain open.

## 12. Evaluation

Evaluation must test policy selection as well as execution.

### 12.1 Matched policy pairs

Use near-identical contexts where one change alters the correct policy:

- sufficient context versus one decisive ambiguity;
- request for a definition versus request for exploration;
- ordinary curiosity versus missing current fact;
- harmless local terminology versus harmful false premise;
- early inquiry versus established common ground;
- explicit request for detail versus request for one sentence; and
- unresolved alternatives versus evidence selecting one interpretation.

### 12.2 Measures

- required move recall;
- prohibited move rate;
- question necessity precision/recall;
- answer-before-qualification rate where required;
- closure appropriateness;
- lecture substitution;
- clarification reflex;
- agreement/accommodation bias;
- useful initiative versus scope drift;
- adaptation after common ground;
- response length conditional on target; and
- human desire to continue conditional on conceptual correctness.

The model receives no credit for naming the policy or producing the expected style without the substantive
move.

## 13. Open taxonomy and proposal workflow

When a recurring useful behavior does not fit existing policy concepts:

1. create a `response_policy_proposal` with positive and hard-negative cases;
2. state its boundary, nearest policies, and why composition is insufficient;
3. attach the candidates/reviews that exposed the gap;
4. obtain human or expert review appropriate to the claim;
5. accept, merge, narrow, defer, or reject the proposal;
6. create a versioned policy record if accepted; and
7. relate legacy targets without rewriting them.

A one-off poetic description is not automatically a new policy. Recurrence and discriminating value matter.

## 14. D5 backfill policy

No D5 normalized mapping occurs before blind Pass A is complete. Revealing or publishing candidate-level
targets would contaminate the human reference.

After Pass B and campaign closeout:

- preserve all 48 raw policy strings exactly;
- propose structured mappings with explicit method provenance;
- let humans accept, revise, split, or mark mappings uncertain;
- record unmappable strings as taxonomy-gap evidence;
- do not infer that a structurally valid candidate satisfied its policy; and
- do not create release or training membership.

The aggregate fact that 48 strings are distinct may remain public because it reveals no candidate-to-contract
mapping.

## 15. Planned migration sequence

A future bounded implementation should proceed as:

1. versioned policy/dimension/value/relation ontology;
2. target components, conditions, order, and compilation lineage;
3. observed moves, spans, fit, and violations;
4. allocation and distribution evidence;
5. public-safe views and query builder facets; and
6. D5 post-blind mapping workflow.

The existing `response_policy_target` table must be migrated without destructive replacement. New tables are
strict where practical, foreign-key checked, and append-only after scientific creation.

## 16. Adversarial acceptance tests

The implementation must prove:

- policy concepts and dimensions can be added without code changes or schema migration;
- a policy version cannot be updated or deleted;
- raw free-form instructions remain byte-identical;
- targets are stored before model calls;
- compiled instructions round-trip to exact target components and compiler version;
- candidate text cannot reveal internal policy labels;
- required, optional, and prohibited components remain distinct;
- a candidate can target several ordered compatible moves;
- incompatible composition is flagged without hard-coded name checks;
- deterministic observations cannot become human judgments;
- Pass A packets expose no structured target;
- Pass B cannot reveal targets before the campaign-wide gate;
- a model-proposed D5 mapping cannot claim human authority;
- target and observed distributions are computed separately;
- family/batch/release aggregation does not treat correlated turns as independent;
- answer-and-stop and justified-question examples both survive selection;
- a terminology-only success fails a policy behavior probe;
- a fluent but canned question is distinguishable from useful momentum;
- no policy operation creates release membership, training exposure, model calls, or compute; and
- all new public relations are browseable without weakening review blindness.

## 17. Acceptance gate

The response-policy control plane is ready for production allocation only when:

- D5 Pass A/B/C/D evidence has been completed and used to revise this contract;
- the operator authorizes the migration and bounded backfill;
- at least two reviewers can apply the initial dimensions consistently on a calibration subset;
- target-to-instruction compilation preserves meaning across surface variants;
- question and closure policies do not induce a canned signature;
- observed policy fit remains distinct from intended policy;
- release-level balance queries are reproducible;
- new policy proposals work without code edits;
- ordinary chat and conceptual conversation both have coverage; and
- no migration or backfill alters the frozen raw D5 evidence.

## 18. Current boundary

This PRD records a real D5 design gap and a future repair architecture. It does not label any D5 response,
reveal any candidate's intended policy, or authorize a normalized mapping. The next authority-bearing action
remains the real human blinded Pass A session.
