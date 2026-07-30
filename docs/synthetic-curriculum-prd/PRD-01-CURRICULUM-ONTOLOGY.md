# PRD-01 — Curriculum ontology and categorical coverage

## 1. Purpose

This PRD defines how Alpha Corpus describes what a synthetic unit is *about*, what intellectual operation it
teaches, what conversational role it plays, and what should change across related cases. It begins with the
analytical lenses in Donto's canonical mega-extraction prompt and expands them for dialogue, linguistics,
philosophy, and social interaction.

The taxonomy is intentionally comprehensive but never closed. It is a coverage instrument, experimental
language, and query interface—not a decree that all meaning fits a fixed ontology. A generator may mint a new
lens or distinction when existing categories lose something important. Such proposals enter the same review,
definition, provenance, and versioning process as any other research object.

## 2. Requirements

The ontology SHALL:

1. distinguish planned variables from post hoc observations and computed measurements;
2. allow multiple categories per family, scene, turn, claim, or transformation;
3. represent uncertain, contested, culturally governed, and theory-relative annotations;
4. preserve rejected category proposals and superseded definitions;
5. support hierarchical, overlapping, and typed cross-category relations;
6. support both sparse annotation and deliberately exhaustive annotation;
7. treat “unknown,” “not applicable,” “not reviewed,” and “reviewers disagree” as different states;
8. keep model-visible language independent of internal category identifiers;
9. make every category definition and revision sourceable;
10. allow queries at any granularity without forcing every unit to populate every dimension.

## 3. Four kinds of fields

Every field in the ledger belongs to one of four meta-classes.

### 3.1 Design variables

Chosen before generation. Examples: target lens, family, intended intervention, dialogue stage, source
condition, desired length, user stance, and held-out projection. These are causal inputs, not claims that the
result actually realized them.

### 3.2 Semantic ground truth

Specified by a family designer or adjudicator. Examples: active local definition, required and prohibited
inferences, admissible interpretations, dependencies, valid-time change, and what should remain invariant.
Ground truth may be executable, expert-adjudicated, set-valued, theory-relative, or intentionally unresolved.
Its authority kind must be recorded.

### 3.3 Observed annotations

Judgments about an actual candidate or model response. Examples: naturalness, unnecessary question, hidden
implication, invalid counterexample, drift, style signature, or whether the response actually performed the
planned dialogue act. Multiple judgments can coexist.

### 3.4 Derived measurements

Computed from stored artifacts. Examples: token count, lexical overlap, embedding-neighbor density,
teacher concentration, family coverage, question rate, response-length distribution, revision locality,
duplicate-cluster size, and exposure frequency. Derived values identify their method and version.

Mixing these classes is prohibited. In particular, an intended `counterexample` is not automatically an
observed valid counterexample.

## 4. Category record

Each category has:

- a stable opaque identifier;
- preferred name and aliases;
- concise and extended definitions;
- meta-class eligibility;
- broader, narrower, related, incompatible, overlap, projection-of, and historical-successor relations;
- positive, negative, borderline, and culturally restricted examples;
- applicability and exclusion notes;
- provenance and responsible author;
- authority level and review state;
- version and valid interval;
- retirement or supersession link;
- recommended generation uses;
- known shortcuts and annotation hazards.

Category names are not training labels unless an experiment explicitly renders them. Renaming a category does
not rewrite old records.

## 5. Foundational Donto-derived analytical lenses

### 5.1 Taxonomy, type theory, and categorization

What kinds exist? What is an instance, type, token, subclass, phase, role, prototype, cluster, or family-
resemblance category? When are boundaries sharp, graded, purpose-relative, or disputed? Curriculum families
must include tempting category errors, cross-classification, type/token confusion, and cases where several
taxonomies serve different questions.

### 5.2 Mereology

Parts, wholes, members, portions, components, ingredients, layers, regions, fragments, and collections. Cover
transitivity and its failures, functional versus material parts, essential versus replaceable parts, shared
parts, scattered objects, boundaries, holes, fusion, overlap, constitution, and group membership. Do not
collapse `part of`, `member of`, `contained in`, `made of`, and `located in`.

### 5.3 Identity and persistence

When does something remain the same through change? Cover continuants and processes, role change, renaming,
replacement, copying, branching, merging, repair, institutional succession, versioned documents, fictional
identity, bodily continuity, memory criteria, social recognition, and record linkage. Preserve the difference
between entity identity and claim equivalence.

### 5.4 Topology and spatial relation

Containment, contact, connection, adjacency, boundary, interior, exterior, crossing, enclosure, separation,
orientation, path, region, scale, and viewpoint-dependent spatial descriptions. Include linguistic frames of
reference and cases where everyday topology differs from mathematical topology.

### 5.5 Chronology and temporal structure

Before, after, during, overlap, recurrence, duration, interval, instant, deadline, age, order, valid time,
record time, remembered time, and counterfactual time. Cover tense, aspect, temporal anaphora, historical truth,
late reports, backdating, future commitments, and events with uncertain or disputed boundaries.

### 5.6 Causation and etiology

Cause, enable, prevent, trigger, maintain, mediate, constitute, correlate, explain, motivate, and merely precede.
Include causal chains, overdetermination, omission, intervention, feedback, probabilistic causation, mechanisms,
and narrative temptation. The negative curriculum must strongly punish invented causality.

### 5.7 Teleology and function

Purpose, use, design function, biological function, selected effect, social function, assigned purpose, user
goal, malfunction, repurposing, side effect, and apparent purpose. Distinguish what something is for, why an
agent uses it, what it tends to do, and what consequence sustains a practice.

### 5.8 Agency, thematic roles, and case grammar

Agent, patient, experiencer, instrument, beneficiary, recipient, source, goal, stimulus, causer, bearer, and
affected party. Cover collective agency, distributed responsibility, coercion, automation, institutional
action, passive constructions, omitted agents, and ambiguous role assignment.

### 5.9 Epistemology

Knowledge, belief, justification, evidence, testimony, observation, inference, confidence, doubt, ignorance,
defeaters, reliability, expertise, authority, hearsay, memory, disagreement, and epistemic injustice. Separate
source reports from endorsement and private belief from public commitment.

### 5.10 Deontology, norms, and law

Obligation, permission, prohibition, entitlement, duty, rule, exception, jurisdiction, convention, enforcement,
violation, excuse, and responsibility. Include conflicts of norms, defeasible rules, constitutive versus
regulative rules, legal versus moral classification, and the difference between describing and endorsing a
norm.

### 5.11 Axiology and evaluation

Good, bad, better, worse, admirable, harmful, fair, beautiful, useful, meaningful, and worthy relative to
criteria, agents, cultures, and purposes. Cover thick concepts, value pluralism, incommensurability, tradeoffs,
and the distinction between evaluation and prediction.

### 5.12 Modality

Possibility, necessity, actuality, capability, permission, likelihood, counterfactuality, disposition, and
normative modality. Include scope ambiguity, de re/de dicto contrasts, modal subordination, possible-world
reasoning, and the difference between cannot, must not, and did not.

### 5.13 Qualia structure

Formal role (what kind of thing), constitutive role (what it consists of), telic role (what it is for), and
agentive role (how it came about), plus coercion between these readings. Use ordinary noun phrases and
nominalizations, not only technical vocabulary.

### 5.14 Lexical semantics and linguistics

Polysemy, homonymy, synonymy, antonymy, entailment, presupposition, selectional preference, coercion,
metaphor, metonymy, idiom, lexicalization, derivation, compounding, semantic roles, ambiguity, vagueness,
prototype structure, and cross-linguistic mismatch. Include morphology, syntax, phonology, information
structure, discourse markers, and language variation where relevant.

### 5.15 Social ontology

Institutions, offices, roles, status, membership, collective acceptance, power, authority, money, law,
marriage, corporations, borders, records, reputation, and socially constructed categories. Cover contestation,
recognition, imposed classification, intersectionality, and the difference between a social fact and universal
natural fact.

### 5.16 Process and event structure

States, activities, accomplishments, achievements, transitions, phases, subevents, participants, culmination,
interruption, repetition, habituality, and event identity. Connect aspectual language to records that may split
or merge events differently.

### 5.17 Constitution and material

Made of, constituted by, realized in, embodied by, composed from, and supported by. Include statue/clay,
document/content, institution/people, software/execution, artifact/material, and whether constitution implies
identity.

### 5.18 Dependence and grounding

Existential, modal, explanatory, causal, conceptual, institutional, and evidential dependence. Cover
grounding direction, circularity, levels, supervenience, realization, and cases where “depends on” is too
underspecified to license an inference.

### 5.19 Genetic, provenance, and origin relations

Created by, derived from, copied from, inherited from, translated from, extracted from, cited by, transmitted
through, and descended from. Include mixed origin, uncertain lineage, common sources, independent witnesses,
and transformations that preserve or alter content.

### 5.20 Comparison and similarity

Exact, similar, analogous, broader, narrower, inverse, overlapping, decomposed, incompatible, and merely
co-occurring. Always ask “similar in what respect?” Include false bridges, surface similarity without
structural identity, and structural similarity without shared vocabulary.

### 5.21 Quantity and measurement

Count, amount, rate, proportion, range, unit, scale, precision, uncertainty, detection limit, ordinal rank,
normalization, and aggregation. Cover measurement error, operationalization, incomparable scales, base-rate
neglect, and the ontology of the measured property versus the measuring procedure.

### 5.22 Disposition, power, and capacity

Fragility, solubility, skill, authority, opportunity, tendency, vulnerability, and causal power. Distinguish a
capacity from its exercise, a disposition from observed frequency, and absence of manifestation from absence
of the disposition.

### 5.23 Speech acts and communication

Assertion, question, request, command, promise, warning, invitation, concession, denial, correction, apology,
threat, joke, quotation, report, and metalinguistic proposal. Cover indirect acts, uptake, felicity conditions,
quoted versus endorsed content, and asymmetric authority.

### 5.24 Phenomenology and experience

Perception, sensation, emotion, mood, attention, embodiment, first-person authority, seeming, memory, and the
difference between experience and external cause. Do not treat reported inner states as independently verified
facts.

### 5.25 Open lens

The generator may propose a lens not represented above. It must explain the recurring distinction, show
positive and negative cases, identify what questions it improves, and compare it with nearest existing lenses.
Novel-sounding names without useful boundaries are rejected.

## 6. Dialogue-native and research-expanded lenses

### 6.1 Pragmatics and implicature

Literal content, implicature, explicature, presupposition, deixis, reference, accommodation, common knowledge,
relevance, politeness, irony, understatement, and conversational repair. Teach when a missing premise should be
accommodated, challenged, or clarified.

### 6.2 Discourse and information structure

Topic, focus, contrast, givenness, anaphora, coherence relations, discourse referents, rhetorical relations,
ellipsis, turn connection, and topic shift. Include what a pronoun or “that” points to after several turns.

### 6.3 Common ground and public commitments

User commitments, Alpha commitments, shared commitments, attributed commitments, denials, tentative
acceptance, live alternatives, Questions Under Discussion, and repair state. A local working assumption is not
necessarily a private belief or universal endorsement.

### 6.4 Inferential conceptual pacts

Purpose-bounded local meanings or representational choices with licensed consequences, prohibited
consequences, dependencies, scope, and revision conditions. PRD-05 gives the formal benchmark.

### 6.5 Metalinguistic negotiation

Cases in which speakers negotiate how a term should be used while appearing to dispute the world. Cover
descriptive, normative, political, practical, and ameliorative reasons for choosing a meaning, plus cases where
participants should preserve different usages rather than force agreement.

### 6.6 Intent and plan interpretation

Infer likely communicative goals from wording, context, actions, constraints, corrections, and downstream
effects. Store multiple hypotheses and discriminating questions. Avoid mind-reading claims or equating intent
with outcome.

### 6.7 Argumentation and dialectic

Premise, conclusion, support, rebuttal, undercutting, qualification, burden, counterexample, analogy,
dilemma, reductio, steelmanning, and explanatory comparison. Distinguish a surprising case from a valid
counterexample and a longer definition from an improved one.

### 6.8 Hermeneutics and interpretation

Speaker meaning, text meaning, historical context, genre, audience, authorial intention, reception, charitable
interpretation, symptomatic reading, and interpretive pluralism. Teach what evidence could decide among
readings and what may remain underdetermined.

### 6.9 Rhetoric and framing

Framing, emphasis, euphemism, dysphemism, metaphor, analogy, omission, emotional appeal, stance, register,
credibility, and persuasive design. Analysis must not automatically imply manipulation or bad faith.

### 6.10 Semiotics and representation

Sign, symbol, icon, index, token, inscription, representation, reference, denotation, connotation, code, and
interpretant. Include diagrams, labels, maps, records, names, and how representations can be useful without
being identical to what they represent.

### 6.11 Narrative and explanation

Plot, sequence, narrator, point of view, character, motive, causal story, retrospective coherence, genre, and
counter-narrative. Teach that narrative order and explanatory satisfaction are not causal proof.

### 6.12 Translation and cross-linguistic conceptualization

Lexical gaps, partial equivalence, grammaticalization, cultural salience, classifier systems, evidentiality,
honorifics, kinship terms, and incompatible segmentation. Avoid declaring one language's ontology canonical.

### 6.13 Culture, standpoint, and authority

Insider/outsider categories, community ownership, situated knowledge, historical power, contested naming,
and who has authority to validate a usage. Synthetic generation may explore generic structures but cannot
manufacture community attestation.

### 6.14 Power, institutions, and ideology

Who may name, classify, authorize, exclude, record, enforce, or erase? Cover bureaucratic categories,
legibility, institutional incentives, resistance, path dependence, and competing descriptive versus critical
accounts.

### 6.15 Emotion, affect, and interpersonal stance

Emotion concepts, appraisal, valence, arousal, expression, empathy, face, vulnerability, reassurance,
frustration, and relational tone. Teach recognition without canned therapy language or unearned certainty about
another person's internal state.

### 6.16 Attention, salience, and relevance

What matters for the current question, what can be safely omitted, what distracts, and how salience differs
from importance or frequency. This lens is central to resisting verbose enumeration.

### 6.17 Absence, negation, and unknown

Explicit negative, missing record, unobserved case, unknown value, not applicable, prohibited value,
counterfactual absence, silence, and evidence of absence. These distinctions are essential to evidence-first
behavior.

### 6.18 Granularity and scale

Fine versus coarse descriptions, zoom levels, temporal aggregation, spatial resolution, taxonomic depth,
institutional versus individual analysis, and when loss from collapsing levels is acceptable for a question.

### 6.19 Analogy and structural mapping

Map roles and relations across domains, identify what is preserved, and state where the analogy breaks.
Include matched false analogies and cases with different surface vocabulary but the same relational structure.

### 6.20 Counterfactuals and intervention

What would change under a controlled alteration, what remains fixed, and which dependencies transmit change?
Cover nearest-world assumptions, impossible antecedents, causal versus evidential counterfactuals, and branch
comparison.

### 6.21 Conceptual change and history

Semantic shift, theory change, category revision, scientific revolution, institutional redefinition, reclaimed
terms, and historical concepts that should not be projected unchanged into the present.

### 6.22 Formal, mathematical, and logical structure

Set, relation, function, order, equivalence, graph, symmetry, quantifier, implication, contradiction,
paraconsistency, and proof. Formal language is used when clarifying; it does not dominate model-visible
conversation or assume contested phenomena have one formalization.

### 6.23 Learning, pedagogy, and explanation

Example selection, scaffolding, misconception diagnosis, analogy choice, minimal contrast, retrieval practice,
teach-back, and adapting an explanation to the learner. “Explain simply” and “remove the important
qualification” are not synonyms.

### 6.24 Conversational ethics and epistemic conduct

Honesty, calibrated confidence, respectful disagreement, refusal to fabricate, correction, attribution,
privacy, manipulation, dependency, and awareness of asymmetric expertise. This is not generic safety prose;
it is behavior embedded in ordinary conversations.

## 7. Linguistic coverage system

Each release should report coverage across, at minimum:

- sentence purpose: declarative, interrogative, imperative, exclamative, fragment, backchannel;
- clause structure: simple, coordination, subordination, relative, complement, adjunct, parenthetical;
- argument structure and alternations;
- tense, aspect, mood, evidentiality, modality, negation;
- reference: pronouns, definites, demonstratives, names, descriptions, bridging, deixis;
- ambiguity source: lexical, syntactic, scope, reference, pragmatic, discourse, social;
- information structure: topic, focus, contrast, correction, afterthought;
- register: casual, reflective, technical, playful, skeptical, tentative, emotionally charged;
- speech act and indirectness;
- discourse relation: cause, contrast, concession, elaboration, correction, explanation, sequence;
- lexical frequency and construction rarity;
- response length and rhythm;
- dialect/variety status: generic generated variation, licensed attestation, community-reviewed, or prohibited;
- language and translation provenance.

The system must not equate “complexity” with long sentences. Short utterances can involve difficult ellipsis,
implicature, reference, repair, or stance.

## 8. Conversational-function taxonomy

Candidate turns can target:

- greet and orient;
- answer directly;
- acknowledge without parroting;
- clarify minimally;
- infer and test intent;
- distinguish senses;
- give an example or counterexample;
- propose terminology;
- accept a local stipulation;
- refuse a misleading stipulation;
- track a live alternative;
- challenge a premise;
- steelman;
- repair a claim;
- retract locally;
- attribute a view;
- compare frameworks;
- state missing evidence;
- ask for or use a source;
- retrieve-ready query formulation;
- summarize current common ground;
- recover after digression;
- change depth or register;
- express provisional judgment;
- close naturally;
- answer and stop.

The `answer and stop` category receives deliberate representation so Alpha does not learn to append a question
to every response.

## 9. Transformation taxonomy

Linked families use typed interventions:

- paraphrase;
- irrelevant-detail addition;
- relevant-detail addition;
- minimal meaning-changing edit;
- lexical substitution;
- syntactic alternation;
- register shift;
- speaker or perspective shift;
- purpose/QUD shift;
- evidence addition, withdrawal, contradiction, or source change;
- valid-time or record-time shift;
- granularity shift;
- category-boundary case;
- positive-to-negative or known-to-unknown shift;
- analogy projection;
- false-analogy projection;
- counterexample;
- definition repair;
- scope restriction or expansion;
- local-term adoption;
- local-term drift;
- explicit renegotiation;
- delayed reuse after distractors;
- interruption and recovery;
- composed transformations;
- ordering-sensitive composition;
- ordering-insensitive composition.

Each edge declares expected commitments to preserve, add, retract, leave plural, reattribute, temporalize, or
mark unsupported.

## 10. Source and evidence conditions

Families may be:

- source-free fictional micro-world;
- grounded in a synthetic passage;
- grounded in a licensed real passage;
- multi-source corroborating;
- multi-source conflicting;
- source with unreliable narrator;
- source with missing context;
- later source revising earlier source;
- primary/secondary/tertiary source comparison;
- record-time/valid-time conflict;
- quoted claim versus analyst interpretation;
- retrieval-required because evidence is absent.

Every grounded unit identifies whether the expected response should accept, qualify, challenge, abstain, or ask
for another source.

## 11. Style and interaction diversity

Coverage must include users who are:

- concise, expansive, uncertain, confident, impatient, playful, formal, colloquial, skeptical, associative,
  emotionally invested, mistaken, partially right, or explicitly exploring;
- asking one question, thinking aloud, correcting Alpha, offering a theory, changing their mind, or requesting
  a particular depth;
- cooperative, distracted, defensive, or productively adversarial without becoming abusive caricatures.

Synthetic diversity cannot be established by attaching demographic labels to generic prose. Human ecological
validation is needed before claims about real social varieties.

Assistant variation must cover:

- short direct answer;
- answer plus one example;
- concise distinction;
- exploratory response;
- explicit disagreement;
- uncertainty with discriminating evidence;
- natural humor or metaphor where appropriate;
- structured explanation only when structure helps;
- no-question closure;
- necessary clarification;
- tentative hypothesis;
- source-conditioned analysis.

Style distributions are measured so a dominant teacher signature can be detected rather than assumed away.

## 12. Difficulty model

Difficulty is multidimensional:

- number of active distinctions;
- dependency depth;
- context distance;
- number of live alternatives;
- lexical rarity;
- syntactic complexity;
- degree of surface/structural mismatch;
- number and composition of interventions;
- evidence conflict;
- required revision locality;
- social or cultural sensitivity;
- degree of theory dependence;
- response-policy ambiguity;
- required brevity.

A scalar difficulty band may be derived for sampling, but the component vector remains stored.

## 13. Family design minimum

A mature concept family should contain, where applicable:

- a statement of purpose and competency questions;
- a latent distinction or operation;
- at least two positive cases;
- at least two hard negatives;
- one borderline or ambiguous case;
- one paraphrase-invariance case;
- one minimal change that should alter the answer;
- one false bridge;
- one alternate projection;
- one ordinary conversational use;
- one challenge and local repair;
- one `answer and stop` realization;
- prohibited shortcuts and terminology leakage;
- explicit dependencies and invariants;
- a theory/culture/authority classification;
- evaluation probes.

Not every family needs all elements. Omissions must be deliberate and recorded.

## 14. Open-lens governance

Any generator, reviewer, researcher, or later model may propose a new category. A proposal includes:

1. recurring phenomenon;
2. why existing categories are insufficient;
3. positive and negative examples;
4. nearest existing categories and typed relations;
5. useful competency questions;
6. risk of superficial or culturally unauthorized labeling;
7. proposed generation and evaluation use;
8. source or argument provenance.

Possible outcomes are `accepted`, `accepted_as_alias`, `accepted_as_narrower`, `accepted_as_crosscutting`,
`deferred`, `contested`, `rejected_no_boundary`, `rejected_duplicate`, and `restricted_requires_authority`.
Rejected proposals remain queryable.

## 15. Coverage accounting

No release is described solely by row count. Its report includes:

- independent family count;
- projections per family;
- transformation-edge coverage;
- dialogue-function coverage;
- lens coverage and intersections;
- hard-negative and false-bridge density;
- ambiguity kind and admissible-set distribution;
- source-condition distribution;
- response-length and question-rate distribution;
- teacher, prompt, and reviewer concentration;
- semantic duplicate clusters;
- accepted, rejected, disputed, and unreleased counts;
- human-calibrated fraction;
- train/development/private-evaluation separation at family and generator-template levels.

Coverage dashboards may reveal gaps; they must not induce generators to pad shallow examples merely to fill a
cell.

## 16. Acceptance criteria

PRD-01 is implemented only when:

- all categories are versioned records rather than hardcoded string enums;
- the Donto prompt crosswalk in Appendix A is lossless at the level of named lenses and extraction principles;
- categories can overlap and receive typed relations;
- unknown, not-applicable, unreviewed, disputed, and prohibited are distinct;
- open-lens proposals round-trip with full provenance;
- planned, observed, ground-truth, and derived fields cannot be conflated by the schema;
- coverage can be reported at family level and release level;
- a family can mint a novel distinction without changing model-visible training syntax;
- culturally restricted categories can be excluded from releases by policy rather than deletion.
