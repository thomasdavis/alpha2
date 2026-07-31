# PRD-04 — Quality, review, and adjudication

## 1. Purpose

Synthetic data becomes useful training evidence only after its claims, conceptual boundaries, dialogue behavior,
and provenance survive review. This PRD defines a layered quality system that preserves disagreement and avoids
letting fluent teacher models validate their own philosophical style.

Quality is not one number. A candidate can be source-faithful but conversationally wooden, natural but
conceptually empty, philosophically interesting but inappropriate for the intended training cohort, or valuable
as a hard negative while unacceptable as a positive target.

## 2. Quality dimensions

### 2.1 Structural validity

- schema-valid output;
- roles and order correct;
- no hidden annotations or delimiters leaked into model-visible text;
- required fields present;
- complete source/span references;
- allowed length and language;
- deterministic renderer compatibility.

Structural validity is necessary and not evidence of semantic quality.

### 2.2 Source fidelity

- every source-dependent assertion is stated, reasonably inferred, or explicitly marked speculative;
- quotation and paraphrase remain distinguishable;
- source speaker, compiler, and later analyst are not conflated;
- chronology and modality are preserved;
- contradiction is not silently resolved;
- absence in a source is not represented as explicit negation;
- no invented cause, intent, identity, or provenance.

### 2.3 Conceptual validity

- target distinction has a defensible boundary;
- positives, negatives, and borderline cases match that boundary;
- required inferences follow;
- prohibited inferences do not;
- a counterexample actually engages the claim;
- repair addresses the counterexample without unrelated qualification;
- alternative analyses are neither collapsed nor multiplied gratuitously;
- analogy preserves the intended relation and names its limits;
- purpose-sensitive representation advice answers the declared competency question.

### 2.4 Linguistic validity

- natural grammar and reference;
- phenomenon genuinely instantiated rather than named in commentary;
- no unnatural construction used solely to make an annotation obvious;
- ambiguity type correctly identified;
- implicature, presupposition, and speech act are not confused;
- language variety claims match their authority status;
- paraphrase preserves the intended content at the relevant granularity.

### 2.5 Conversational quality

- direct engagement with the preceding move;
- appropriate length and depth;
- useful momentum;
- adaptation to established terms and user state;
- no automatic restatement;
- no automatic follow-up question;
- clarification only when it changes the response materially;
- no generic essay or counselor cadence;
- natural repair, disagreement, and closure;
- response feels authored for this conversation.

### 2.6 Pedagogical value

- evidence is dense enough to teach the distinction;
- minimal contrasts isolate the relevant difference;
- examples are neither trivial nor overloaded;
- hard negatives are plausible;
- surface variation does not obscure the contract;
- the response models a behavior Alpha should actually perform;
- the unit adds information beyond semantic near-neighbors.

### 2.7 Plurality and calibration

- uncertainty matches evidence;
- genuine ambiguity retains important analyses;
- unsupported possibilities are excluded;
- clarification is requested only if it could resolve the relevant uncertainty;
- theory-relative claims name or imply their scope;
- public commitment is not mistaken for private belief;
- provisional working assumptions are not universalized.

### 2.8 Safety, rights, and authority

- license and allowed use established;
- no private or identifying content without authorization;
- community-specific language not falsely represented as attested;
- sensitive categories carry appropriate review and release restrictions;
- the assistant does not reinforce harmful or manipulative behavior merely to sound agreeable;
- the unit is suitable for the intended release and experiment.

### 2.9 Novelty and distributional contribution

- not a semantic duplicate unless intentionally linked;
- contributes a new family, projection, transformation, linguistic realization, interaction style, or hard
  boundary;
- does not reinforce an already dominant teacher/template signature;
- has a declared role in a release allocation.

## 3. Hard gates versus scored dimensions

Hard failures include:

- fabricated source content;
- hidden-evaluation answer leakage;
- invalid or self-contradictory semantic contract;
- culturally unauthorized authenticity claim;
- private data violation;
- train/private-eval family leakage;
- missing generation lineage;
- structured-output parse invented from free text;
- renderer/source mismatch;
- use of a rejected target as a positive without explicit negative labeling.

Scored dimensions remain separate. A release policy may require minima, but a weighted average cannot let
beautiful prose compensate for a false claim.

## 4. Review layers

### R0 — Deterministic validation

Schemas, constraints, spans, hashes, role order, source presence, length, held-out words, delimiter leakage,
and exact duplicates.

### R1 — Automated diagnostic analysis

Similarity clustering, style signatures, lexical leakage, question rate, response-length outliers, source
entailment candidates, category coverage, and consistency between hidden state and expected deltas.

The first D5 R1 materialization is recorded in Execution 05. It stores current-version candidate-level word
and character n-gram similarities, campaign/family distributions, and dynamically discovered word n-gram
signatures as `surface_distribution_only`. These are review nominations, not semantic clusters, quality
scores, critic decisions, or admissibility judgments.

### R2 — Model critics

Independent dimension-specific critics produce findings with evidence. They may recommend accept, repair,
reject, restrict, or human adjudication. Their recommendations are not final.

### R3 — Human stratified audit

Humans review a predeclared sample from every batch, oversampling high-risk categories, disagreements, model-
accepted edge cases, and model-rejected candidates that might expose critic bias.

### R4 — Expert/human full review

Required for frozen evaluation anchors, contested philosophy, complex counterexamples, cultural authority,
rights ambiguity, and public claims of validity.

### R5 — Release-level audit

Reviews the distribution, leakage, provenance, coverage, style, rights, and reconstructability of a cohort,
not just individual units.

For conversational action, [PRD-14](PRD-14-RESPONSE-POLICY-CONTROL-PLANE.md) requires reviewers to keep the
blueprint policy, compiled worker instruction, observed response move, question/closure behavior, and release
distribution in separate evidence layers. A declared target never proves that a generated response achieved it.

### D5 candidate passes and family synthesis

For a bounded family calibration, candidate review proceeds through distinct evidence layers:

- **Pass A** judges model-visible conversation without family, lineage, contract, validator, or model cues;
- **Pass B** reveals the family and hidden contract only after Pass A is sealed and judges contract fit;
- **Pass C** compares every sibling only after all current candidates have sealed A and B evidence;
- **Pass D** is later adjudication and campaign synthesis, not an automatic consequence of Pass C.

Pass C records coverage pressure, strongest and weakest units, semantic duplicate groups, shared errors and
style signatures, response-policy imbalance, metadata mismatch, missing negatives, blueprint repair,
uncertainty, and family disposition. Every structurally rejected sibling also receives a separate disposition
of content utility, validator correctness, semantic type, remedy, and automatic-accept/reject hazards.

The workflow must fail closed before its prerequisites, bind every response to an exact evidence snapshot,
retain the raw submission, and leave candidate, release, and training state unchanged. The executable D5
implementation is recorded in [Execution 06](EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md).

Pass A repeat presentations are presentation-level reliability evidence, not additional candidate reviews.
Each repeat uses a fresh opaque ID, hides its source review in the model-visible packet, and stores its response
separately. Stability measures outcome, policy, confidence, and dimension-score agreement while preserving the
possibility that a stable judgment is wrong or that a changed judgment is justified. The executable contract
is [Execution 07](EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md).

Pass D now has an executable non-binding campaign-closeout path. It requires the same human adjudicator's
complete A/B population, hidden-repeat stability, all family syntheses, every required structural disposition,
and the current authoritative analysis run. It records candidate adjudications and their exact bases, failure
clusters, conversational-distribution assessments, uncertainty, and recommended D5 states. The schema forces
zero execution authority and the transaction creates no lifecycle transition, release member, or training
exposure. See [Execution 08](EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md). The live relations remain empty
until a real human completes the preceding passes.

## 5. Reviewer calibration

Before reviewing production data, every model and human reviewer sees a calibration set containing:

- clear accepts;
- clear rejects;
- plausible but invalid counterexamples;
- overqualified repairs;
- genuine plural-analysis cases;
- false ambiguity;
- source-attribution traps;
- natural answer-and-stop cases;
- canned follow-up questions;
- culturally restricted cases;
- examples where experts legitimately disagree.

Track dimension-level agreement, bias, severity, false acceptance, false rejection, and uncertainty. A reviewer
may be qualified for conversational naturalness and unqualified for mereology. Capability profiles are scoped.

Every sealed human A/B session also records the reviewer's declared competence scope, start/end, interruption,
fatigue, and material conditions. These declarations are provenance rather than automatic authority scores.
They are append-only, bound to exact packet/submission hashes, and must validate before any review evidence is
written. The first executable implementation is [Execution 14](EXECUTION-14-D5-REVIEW-SESSION-PROVENANCE.md).

Calibration expires after material rubric, model, or prompt changes.

## 6. Counterexample protocol

The Counterexample Game found that a language-model judge accepted roughly twice as many candidate
counterexamples as human philosophers, while repeated repair greatly lengthened definitions without improving
accuracy ([Drucker and Mahowald](https://arxiv.org/abs/2605.03936)). Alpha therefore treats counterexamples as
high-risk research objects.

A valid counterexample must:

1. instantiate the antecedent conditions of the target claim or definition;
2. fail the predicted classification or consequence;
3. avoid changing an unrelated background assumption;
4. remain plausible or explicitly fictional under the family's authority type;
5. state why it engages the rule;
6. survive at least one adversarial attempt to explain it away;
7. support a local repair or a justified rejection of the original distinction.

Repairs are scored for:

- locality;
- explanatory compression;
- retained coverage;
- new false positives/negatives;
- number of added clauses;
- conceptual stability across later cases.

Longer is not better. A repair that becomes an exception list is rejected even if it fits the seen cases.

## 7. Admissible-set review

For ambiguity or pluralism, reviewers assign:

- required analyses;
- optional but defensible analyses;
- excluded analyses;
- evidence that would discriminate;
- whether clarification can resolve the set;
- whether plurality is linguistic, evidential, theoretical, perspectival, temporal, granular, or cultural.

Quality requires both precision and recall. Listing every imaginable reading is overcoverage, not nuance.

## 8. Conversational-quality review

Reviewers answer behaviorally specific questions:

- Did the first sentence address the user's actual move?
- Did the response contribute something beyond paraphrase?
- Was a distinction relevant to the user's purpose?
- Did the answer choose an appropriate depth?
- Was a question necessary, useful, optional, or formulaic?
- Did the response preserve established local terminology?
- Did it make an unwarranted claim about the user's mental state?
- Could friendly framing be removed while leaving a substantive contribution?
- Would the reviewer willingly continue?
- Did the response stop when complete?

The `desire to continue` judgment never substitutes for conceptual validity. Warm shallowness and cold rigor are
separate failure modes.

## 9. Synthetic-user audit

Synthetic-user batches are compared on:

- politeness and question-rate concentration;
- unnaturally complete task descriptions;
- convenient disclosure of hidden goals;
- turn length and grammatical uniformity;
- lack of repair, hesitation, or reference;
- demographic or dialect stereotyping;
- difficulty distribution;
- success inflation relative to human interaction.

At least some final evaluation uses real consented human–Alpha interaction. Synthetic users may support
training and regression; they cannot alone establish product success.

## 10. Style-signature controls

Detect and limit:

- repeated opener/closer n-grams;
- “The key distinction is...” overuse;
- fixed two-sided enumeration;
- excessive em dashes, headings, and bullet lists;
- reflexive “it depends”;
- automatic “Which sense do you mean?”;
- essay-style thesis/conclusion symmetry;
- therapy-like mirroring;
- unearned praise;
- technical labels immediately followed by textbook definitions;
- teacher-specific punctuation and cadence.

Style-scrubbed review removes politeness and framing, then judges the underlying intellectual move.

## 11. Adjudication outcomes

Allowed outcomes:

- `accept_as_positive`;
- `accept_as_negative`;
- `accept_as_ambiguous_set`;
- `accept_with_scope_restriction`;
- `repair_local`;
- `regenerate_from_blueprint`;
- `revise_blueprint`;
- `split_family`;
- `merge_as_projection`;
- `restrict_requires_authority`;
- `defer_theory_disagreement`;
- `reject_invalid`;
- `reject_duplicate`;
- `reject_style`;
- `reject_source_fidelity`;
- `reject_policy`.

Adjudication selects lifecycle state and use; it does not erase reviews or assert global philosophical truth.

## 12. Batch acceptance

A batch must satisfy predeclared thresholds for:

- deterministic validity;
- source fidelity;
- human-audited conceptual precision;
- critic false-accept rate;
- duplicate and template concentration;
- required style/length/question distributions;
- category and transformation coverage;
- high-risk review completeness;
- rights and authority status;
- cost per accepted family structure;
- absence of evaluation leakage.

A high individual accept rate may be suspicious if it reflects an easy repetitive prompt. Yield is interpreted
with novelty and coverage.

## 13. Quality states and release tiers

- **Raw:** every attempt, no quality claim.
- **Structurally valid:** machine contract passed.
- **Bronze:** automated critics passed; not human-calibrated enough for strong claims.
- **Silver:** batch human audit and release policy passed.
- **Gold:** unit or family received required expert/human review and may support primary training/evaluation.
- **Contested:** valuable preserved disagreement; release only with its analysis set and scope.
- **Restricted:** retained but excluded from general distribution.
- **Red/negative:** deliberately retained failure or hard negative with verified label.

These are not a single quality ladder. A `Red` hard negative may be more scientifically valuable than a Bronze
positive.

## 14. Human review efficiency

Use models to organize, not replace, scarce human judgment:

- cluster candidates so one reviewer can compare siblings;
- highlight precise disputed spans and contract deltas;
- present source and expected state beside the candidate;
- blind teacher/model identity where practical;
- interleave accepted and rejected items;
- track reviewer fatigue and order;
- collect short rationales and uncertainty;
- prioritize cases where a decision changes many descendants;
- reuse adjudicated anchors for later calibration.

Do not ask humans to review raw millions of rows. Review decisions should occur at the highest leverage level:
blueprint, family, batch distribution, and strategically sampled realizations.

## 15. Audit sampling

Every production batch receives:

- uniform random sample;
- category-stratified sample;
- high critic-disagreement sample;
- high-confidence model-accept sample;
- high-confidence model-reject sample;
- novelty/outlier sample;
- nearest-evaluation-neighbor sample;
- source/cultural risk sample;
- repaired-lineage sample.

Sample selections and inclusion probabilities are recorded so estimates are interpretable.

## 16. Stop rules

Quality production stops if:

- calibrated human false-accept rate exceeds threshold;
- a teacher or critic revision changes behavior materially;
- repetitive style dominates;
- conceptual errors cluster around a blueprint;
- evaluation leakage is detected;
- source/license status becomes uncertain;
- a cultural authority objection arises;
- review backlog exceeds the approved unreviewed inventory;
- batch cost rises without marginal family or coverage gain.

## 17. Acceptance criteria

The review system is ready only when:

- each quality construct has anchored examples;
- judges are calibrated by dimension and cannot self-authorize;
- counterexample validity receives explicit checks;
- accepted and rejected samples receive blind human audit;
- review disagreement remains visible;
- batch-level style and duplication audits work;
- plurality is scored for undercoverage and overcoverage;
- naturalness and conceptual contribution remain separate;
- every release decision is traceable to reviews, policies, and evidence;
- reviewers can reverse a decision by supersession without deleting its history.
