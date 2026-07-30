# Appendix A — Donto mega-prompt to Alpha curriculum crosswalk

## 1. Source and purpose

This appendix records how the canonical Donto extraction method informs Alpha Corpus without turning Alpha
into an extraction engine.

Source inspected:

`/mnt/donto-data/workspace/donto/apps/donto-agent/prompts/extract_broad.txt`

Related sources:

- `/mnt/donto-data/donto-resources/vision/DONTO-CANON.md`;
- `/mnt/donto-data/donto-resources/vision/DONTO-ABUNDANCE.md`;
- `/mnt/donto-data/workspace/donto/docs/DONTO-PRD.md`;
- `/mnt/donto-data/workspace/donto/docs/DONTO-CALCULUS.md`.

The prompt's genius is not simply its long list of topics. It combines exhaustive multi-lens inspection with
free predicate minting, evidence fidelity, contradiction preservation, inverse relations, provenance, and an
open category for distinctions the list did not anticipate. Alpha Corpus adopts that epistemic posture.

## 2. Lens crosswalk

| Donto lens | Alpha curriculum target | Conversational realization | Required boundary pressure |
|---|---|---|---|
| Taxonomy/type theory | kinds, instances, roles, phases, prototypes | “Is a student a kind of person or a role?” | graded versus sharp categories; type/token errors |
| Mereology | part/member/component/portion/ingredient | “Is a committee member part of the committee in the same sense as the chair is part of the furniture?” | non-transitivity, overlap, functional/material parts |
| Identity/persistence | sameness through change | renamed institutions, repaired objects, former roles | copying, branching, replacement, social recognition |
| Topology/spatial | boundary, enclosure, contact, path | holes, rooms, maps, borders | linguistic frame versus physical relation |
| Chronology/time | intervals, order, valid and record time | historical role versus present record | late reports, uncertain boundaries, aspect |
| Causation/etiology | cause, enable, prevent, explain | sequence versus cause in a story | omission, correlation, mechanism, overdetermination |
| Teleology/function | purpose, design, use, selected/social function | “What is this rule for?” | use versus function; malfunction; repurposing |
| Agency/thematic roles | agent, patient, experiencer, instrument | active/passive reports, collective decisions | omitted agents, coercion, institutional action |
| Epistemology | evidence, testimony, belief, knowledge | conflicting sources and confidence | report versus endorsement; defeaters; ignorance |
| Deontology/norms/law | permission, obligation, prohibition | rule conflicts and exceptions | legal versus moral; constitutive versus regulative |
| Axiology/value | criteria and plural values | “Better for whom and for what?” | fact/evaluation, incommensurability, thick concepts |
| Modality | possible, necessary, capable, permitted | “Could,” “must,” and “must not” | scope, de re/de dicto, actual/nonactual |
| Qualia structure | formal/constitutive/telic/agentive roles | noun interpretation and coercion | one lexical item selecting different roles |
| Lexical semantics/linguistics | polysemy, entailment, coercion, morphology | natural minimal contrasts and repair | same word/different sense; cross-language mismatch |
| Social ontology | roles, offices, institutions, collective facts | status, membership, recognition | power, imposed categories, contestation |
| Process/event structure | states, activities, culmination, subevents | “Was building” versus “built” | split/merge events; interruptions; habituals |
| Constitution/material | made of, constituted by, realized in | statue/clay, document/content | constitution does not automatically equal identity |
| Dependence/grounding | causal, conceptual, institutional dependence | “In what sense does it depend on that?” | direction, type, circularity, underspecification |
| Genetic/provenance/origin | copied, derived, inherited, translated | source lineages and versions | common source versus independent corroboration |
| Comparison/similarity | exact, close, broad, narrow, inverse | non-equivalence dialogue | similarity respect; false bridges |
| Quantity/measurement | count, rate, unit, precision, error | operational definitions | property versus measuring procedure; aggregation |
| Disposition/capacity | tendency, skill, power, vulnerability | “It never broke; is it fragile?” | manifestation versus possession |
| Speech acts/communication | assertion, request, promise, report | indirect acts and uptake | quoted versus endorsed; felicity and authority |
| Phenomenology/experience | perception, feeling, seeming | first-person reports | experience versus external cause; attribution |
| Open lens | newly discovered distinction | model/researcher proposes useful category | novelty must show boundary, recurrence, and query use |

## 3. Source-sweep crosswalk

The Donto prompt also requires exhaustive passes over source shape. Alpha converts them as follows:

| Donto source sweep | Alpha data object |
|---|---|
| Every named/implied entity | scene participants, discourse referents, latent roles |
| Every attribute | commitments, states, qualities, measurements, descriptions |
| Every relationship and inverse | typed family relations and bidirectional conversational tests |
| Events and subevents | trajectory transitions, event structure, before/after states |
| Quantities/qualifiers/disputes | measurement units, modality, confidence, disagreement |
| Containment/organization hierarchy | mereological and institutional projections |
| Provenance/epistemics | source-conditioned dialogue and evidence anchors |
| Contradiction/corroboration | plural claim states, rebuttal/support, source independence |
| Structured cells/tables | interpretation of schema, missingness, rows, aggregation |
| Figurative/euphemistic framing | rhetoric, metaphor, stance, social meaning |
| Notable quotes | quotation, attribution, speech acts, authorial voice |

## 4. Donto operational principles and Alpha equivalents

### 4.1 Emit free now; align later

**Donto:** freely mint predicates, then align them dynamically and at query time.
**Alpha Corpus:** let family designers propose distinctions and lenses; record nearest categories and typed
relations; do not force every insight into a closed enum. Release queries may later fold or select them.

### 4.2 Evidence or honest hypothesis

**Donto:** a claim is anchored or explicitly interpretive/hypothetical.
**Alpha Corpus:** source-conditioned content separates source claim, generator inference, family ground truth,
and model-visible response. Unsupported content cannot be smuggled in as a fact.

### 4.3 Contradiction is legal state

**Donto:** incompatible claims coexist with provenance.
**Alpha Corpus:** rival analyses, reviewer disagreement, user/Alpha commitments, and conflicting sources coexist.
The system can select a scoped response without deleting alternatives.

### 4.4 No destructive overwrite

**Donto:** retract or supersede.
**Alpha Corpus:** revisions, rejection, retirement, redaction, and adjudication create append-only successors.

### 4.5 Confidence is not maturity

**Donto:** model confidence does not equal evidence maturity.
**Alpha Corpus:** teacher confidence, critic score, human review, executable oracle, and cultural authority remain
separate dimensions.

### 4.6 Identity is a hypothesis

**Donto:** entity resolution is scoped and revisable.
**Alpha Corpus:** duplicate candidates, same concept, same family, and cross-domain realization are proposed
relations with provenance, not destructive merges.

### 4.7 Schema alignment is typed and scoped

**Donto:** exact, close, broader, narrower, inverse, decomposition, value mapping, incompatible, local
specialization, and not-equivalent relations.
**Alpha Corpus:** these become explicit non-equivalence curricula, projection maps, false bridges, and release
selection relations.

### 4.8 Loss is reported

**Donto:** representation/export operations report what they omit or collapse.
**Alpha Corpus:** renderers, cohort queries, granularity changes, and alignment choices declare lost messages,
annotations, alternatives, or dependencies.

### 4.9 Reproducible releases

**Donto:** query results and evidence state must be reproducible.
**Alpha Corpus:** sealed SQLite snapshots, cohort manifests, renderers, tokenizers, and exposures reconstruct
every training artifact.

## 5. Donto typed-edge crosswalk

### Predicate/alignment edges

- exact → paraphrase/equivalent representation;
- close → similar but unsafe universal substitution;
- broader/narrower → taxonomy and granularity transformations;
- inverse → relation-direction tests;
- decomposition → frame or multi-relation analysis;
- value mapping → translation/unit/category alignment;
- incompatible → false bridge or conflicting framework;
- derived → explicit inference dependency;
- local specialization → inferential conceptual pact;
- not equivalent → Non-Equivalence Judge families.

### Argument edges

- supports → evidence addition;
- rebuts → contradictory claim;
- undercuts → source/reliability defeater;
- qualifies → scope, modality, or time restriction;
- explains → explanatory relation without automatic cause;
- alternative analysis → admissible set;
- same evidence/different analysis → perspective/theory plurality;
- same claim/different schema → cross-projection mapping;
- supersedes → local revision with historical preservation.

## 6. What Alpha must not inherit from extraction

The Donto mega-prompt is optimized for exhaustive source deconstruction. Alpha conversation requires different
surface behavior. Do not train Alpha to:

- enumerate every lens in every answer;
- emit large JSON structures by default;
- name a predicate for every observation;
- turn casual conversation into source extraction;
- give inverse relations when the user did not need them;
- dump provenance machinery into a simple answer;
- sound like a knowledge-engineering report;
- ask every possible ontological question;
- mistake exhaustiveness for relevance.

The dataset generator uses Donto's depth; the assistant uses conversational salience.

## 7. Additional Alpha-only lenses

Donto's prompt is expanded with dialogue-native categories:

- common ground and Questions Under Discussion;
- inferential conceptual pacts;
- metalinguistic negotiation;
- intent and plan interpretation;
- pragmatics and implicature;
- discourse/information structure;
- argumentation and dialectic;
- rhetoric and framing;
- hermeneutics;
- semiotics;
- narrative;
- analogy and false bridges;
- conceptual change;
- attention/salience;
- emotion and interpersonal stance;
- pedagogy;
- conversational ethics;
- human–model coordination;
- answer-and-stop behavior.

## 8. Acceptance test for the crosswalk

The crosswalk is faithful only if a researcher can take any named Donto lens or core invariant, find its Alpha
representation, see how it becomes natural conversation, and identify at least one boundary case that prevents
mere keyword training. New Donto lenses discovered later enter through the open-lens process rather than
requiring a rewrite of the curriculum system.
