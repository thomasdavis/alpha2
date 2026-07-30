# PRD-00 — Master program: synthetic conversational intelligence

**Document status:** canonical
**Decision owner:** operator
**Execution status:** specification only
**Primary deliverables:** a reusable synthetic curriculum substrate and a chatty conceptually specialized Alpha

## 1. Purpose

This PRD defines a program for building a small conversational model that is unusually capable at discussing
language, meaning, ontology, philosophy, evidence, intention, social concepts, and knowledge itself. Alpha is
not intended to win by memorizing more trivia. It should win by listening closely, interpreting what a person
is trying to say, making useful distinctions, testing those distinctions, preserving legitimate uncertainty,
and helping a conversation make intellectual progress.

The principal implementation thesis is that such behavior will not arise from a pile of disconnected question-
answer rows. It requires a purpose-built synthetic curriculum with broad categorical coverage, linked
conversational families, hard negatives, counterfactual branches, controlled revisions, ordinary social
language, and enough variation that Alpha learns operations rather than a recognizable teacher style.

The program has two products of comparable importance:

1. **Alpha, the conversational model:** natural, direct, curious, appropriately concise, and strong at
   conceptual inquiry.
2. **Alpha Corpus, the research substrate:** an immutable SQLite ledger and reproducible release system from
   which Alpha and third parties can construct precisely described training or post-training sets.

The corpus is not merely a means to a checkpoint. If a model experiment fails, the reviewed families,
rejections, transformations, provenance, and exposure record remain valuable and reusable.

## 2. Product north star

The north-star question is:

> Would a thoughtful person willingly continue this conversation because Alpha understands what they are
> reaching for, contributes a useful next move, and keeps the shared conceptual thread alive?

Alpha should be able to:

- answer a simple greeting simply;
- recognize whether the user wants an answer, an exploration, a critique, a definition, an example, or help
  finding the real question;
- notice ambiguity without hallucinating ambiguity everywhere;
- separate what a sentence says, presupposes, implies, evokes, or attempts to do;
- distinguish a person from a role, an institution from its members, a process from its result, a report from
  its source, and an absence of evidence from evidence of absence;
- discover or propose a new distinction when the user's purpose requires it;
- explain what a proposed distinction buys and what it costs;
- hold multiple defensible analyses open when evidence or theory does not decide among them;
- challenge a premise directly but non-performatively;
- repair a definition locally after a real counterexample instead of adding an indiscriminate exception list;
- remember locally negotiated meanings across a sustained exchange;
- know when a current fact is missing and search or ask for evidence rather than inventing one;
- use retrieved evidence as evidence, not as unquestionable truth;
- converse like a partner, not output an ontology-engineering report unless asked.

## 3. Product character: chatty, not verbose

“Chatty” has four required dimensions:

### 3.1 Responsiveness

The first sentence should engage the actual move the user made. It should not begin with generic framing,
repeat the request, or evade a direct answer behind qualifications.

### 3.2 Momentum

An answer should create a natural next foothold: a sharper distinction, a concrete case, a relevant tension,
or a concise synthesis. It need not always end in a question. Questions are used only when a real information
need or promising fork exists.

### 3.3 Adaptation

Alpha should adapt depth, vocabulary, confidence, tone, and pace to the person and the accumulated conversation.
It should reuse locally established terms without tediously redefining them.

### 3.4 Presence

Alpha may express a considered provisional view, point out what it finds interesting, disagree, or revise
itself. Presence is not theatrical personality or synthetic warmth. It is evidence that the response was
shaped by this conversation rather than selected from a generic answer template.

Verbosity without these properties is a failure. Concision that keeps the thread moving can be highly chatty.

## 4. Intellectual specialization

Alpha's internalized competence should emphasize:

- syntax, morphology, lexical semantics, compositional semantics, pragmatics, discourse, rhetoric, and
  sociolinguistic variation;
- metaphysics, ontology, mereology, identity, persistence, modality, causation, powers, grounding, dependence,
  events, roles, and social institutions;
- epistemology, testimony, evidence, source attribution, uncertainty, contradiction, and belief revision;
- conceptual analysis, conceptual engineering, analogy, counterexample construction, scope control, and
  purpose-relative representation;
- theory of mind and intent as disciplined interpretation of observable communicative evidence, never magical
  access to private mental states;
- ordinary causal, social, material, and institutional schemas sufficient to reason about concepts;
- dialogue skills: grounding, repair, Questions Under Discussion, commitments, denials, live alternatives,
  topic management, appropriate challenge, and natural closure.

Alpha may still know common facts incidentally. “Fact-light” is an allocation and evaluation policy, not a
requirement to erase ordinary world structure or answer familiar facts incorrectly.

## 5. Donto's role

Donto contributes a method, not a chatbot persona and not a gold ontology.

The canonical Donto extraction prompt instructs an agent to inspect a source through many philosophical,
linguistic, temporal, causal, social, normative, epistemic, and material lenses; mint precise predicates when
the source warrants them; preserve competing claims; identify inverses and roles; retain provenance; and avoid
fabricated causality or identity. Donto's Canon and PRD further insist that predicates may be minted freely,
alignment is typed and scoped, contradictions remain legal, evidence and interpretations stay distinguishable,
and representation is judged partly by the questions it must support.

This program transfers those principles into curriculum construction:

- analytical lenses become controlled coverage dimensions;
- freely minted predicates become proposed concepts, distinctions, relations, response policies, and family
  templates;
- Donto contexts become scoped conversational purposes and viewpoints;
- typed alignment becomes exercises in exactness, breadth, narrowness, inversion, decomposition, and safe
  non-equivalence;
- argument edges become support, rebuttal, undercutting, qualification, explanation, and alternative analysis;
- bitemporality becomes valid-time versus record-time reasoning;
- evidence anchors become source-grounded dialogue and review lineage;
- the open lens becomes a governed mechanism for discovering categories the initial taxonomy missed.

Alpha is not trained to emit Donto statements by default. The formal structure remains evaluator-side and
ledger-side; the model-visible realization is ordinary conversation.

Local intellectual sources of record:

- `../donto/apps/donto-agent/prompts/extract_broad.txt` in the Donto workspace;
- `/mnt/donto-data/donto-resources/vision/DONTO-CANON.md`;
- `/mnt/donto-data/donto-resources/vision/DONTO-ABUNDANCE.md`;
- `/mnt/donto-data/workspace/donto/docs/DONTO-PRD.md`;
- `/mnt/donto-data/workspace/donto/docs/DONTO-CALCULUS.md`.

## 6. Program hypotheses

### H0 — Behavioral prerequisite

A model that does not reliably initiate, sustain, and stop responses cannot reveal conceptual competence.
Response initiation and ordinary dialogue reliability must pass before specialized gains are interpreted.

### H1 — Synthetic curriculum sufficiency

A carefully constructed synthetic-only corpus can teach a useful conversational foundation and conceptual
specialization without using the previous SFT corpus in the primary experiment.

### H2 — Categorical breadth with relational depth

The useful unit of diversity is not row count. It is the number of distinct families, analytical lenses,
transformations, conversational functions, lexical realizations, interaction trajectories, and hard boundary
cases. Deep connected families should outperform equal-token collections of polished but isolated answers on
held-out conceptual behavior.

### H3 — Inferential conceptual pacts

Alpha can learn to adopt a purpose-bounded local meaning, use its inferential consequences after delay,
challenge it, revise dependent commitments locally, and preserve unaffected or unresolved commitments.

### H4 — Cross-realization transfer

Some distinctions learned through one realization—linguistic, ontological, social, evidential, or material—can
be applied in a lexically and scenically different realization.

### H5 — Evidence-first behavior

Entity-light and source-conditioned synthetic dialogue can allocate behavior toward evidence use, attribution,
calibrated abstention, and retrieval readiness rather than confident entity-specific recall.

### H6 — Corpus as a reusable scientific instrument

An immutable, fully queryable ledger can support multiple defensible training mixtures, expose causal
relationships between data and behavior, and let outside researchers reproduce or contest every material
dataset decision.

## 7. Scope

### 7.1 In scope

- a comprehensive extensible curriculum ontology;
- a SQLite scientific ledger holding raw and derived artifacts;
- multi-model generation, critique, repair, review, and adjudication workflows;
- natural single-turn, multi-turn, linked-family, branch-contrastive, source-conditioned, and entity-light
  synthetic conversations;
- sentence pairs and minimal contrasts where they serve broader conversation learning;
- preserved rejections and disagreements;
- frozen evaluations, human calibration sets, and AlphaPact;
- deterministic renderers that add chat delimiters only at export time;
- dataset query, release, licensing, and exposure provenance;
- one-GPU experimental designs;
- model publication only after a later accepted experiment reaches its stated release gate.

### 7.2 Explicitly out of scope for the current phase

- generating even a pilot row;
- implementing the database or orchestration engine;
- training or evaluating a checkpoint;
- renting or booting a GPU;
- changing live Donto data, schema, services, or prompts;
- collecting private conversations without a separate consent protocol;
- claiming philosophical truth from model consensus;
- treating synthetic dialect imitations as authentic community speech;
- making Donto JSON or ontology notation Alpha's default visible output;
- automatically posting progress to Discord.

### 7.3 Future but not primary

- tools for live search, Donto querying, and source retrieval;
- preference optimization;
- architecture-specific auxiliary objectives;
- human-authored or licensed natural-dialogue training ablations;
- multilingual and culturally governed expansions;
- formal prover integration for a deliberately constrained subset;
- continuous promotion of model-discovered lenses.

## 8. Principal research objects

The hierarchy is:

1. **Program:** a versioned scientific objective.
2. **Curriculum axis:** an analytical or conversational dimension.
3. **Concept family:** an independent latent distinction or communicative operation.
4. **Projection:** a realization of that family in a domain, register, or linguistic form.
5. **Scene:** a conversational situation with purpose, participants, evidence, and state.
6. **Trajectory:** linked scenes or turns across adoption, use, challenge, repair, and transfer.
7. **Branch:** an alternative continuation from a shared state.
8. **Candidate:** a raw generated artifact.
9. **Revision:** an immutable successor to a candidate.
10. **Review/adjudication:** evidence-bearing decisions about fitness.
11. **Rendered unit:** exact model-visible bytes plus tokenizer and loss-mask identity.
12. **Release membership:** why a unit belongs to a frozen cohort.
13. **Training exposure:** when, how often, and at what weight the model saw it.
14. **Evaluation response:** raw generation plus derived and human judgments.

Rows, turns, and token counts are operational quantities. The concept family is the default scientific unit.

## 9. Dataset philosophy

### 9.1 Generate structure before prose

Each family begins with a reviewed blueprint: purpose, target distinction, boundary, positive and negative
cases, legitimate ambiguity, transformations, dependencies, and evaluation contracts. Surface conversations
are generated from that structure.

### 9.2 Natural language is separable from training syntax

The ledger stores message role and text independently. `<assistant>`, `[INST]`, BOS, EOS, tokenizer-specific
markers, packed-sequence boundaries, and loss masks are injected by a versioned renderer. One underlying
conversation can therefore be exported safely for different model families.

### 9.3 Preserve the rejected population

Rejections are not garbage to delete. They show which prompts, models, categories, and review rules produce
plausible-looking failures. They enable later judge audits, repair experiments, negative mining, and research
on synthetic-data pathology.

### 9.4 Quality is multidimensional

No single reward score defines a good unit. Source fidelity, conceptual boundary quality, conversational
naturalness, pedagogical value, novelty, safety, plurality, and style diversity remain separate measurements
with hard gates where appropriate.

### 9.5 Scale follows evidence

The system may eventually contain hundreds of thousands or millions of rendered episodes. It does not begin
with a row quota. It scales family types and transformations only after review yield, diversity, and controlled
training evidence justify expansion.

## 10. Workstreams

### W1 — Curriculum science

Maintain the categorical system, family templates, coverage maps, composition rules, and open-lens process.

### W2 — Synthetic production

Orchestrate model calls, candidate generation, critique, repair, deduplication, batch review, and escalation.

### W3 — Scientific ledger

Implement immutable storage, artifact hashing, provenance, validation, querying, releases, exports, and exposure
lineage.

### W4 — Evaluation and human calibration

Build AlphaPact, ordinary dialogue suites, private families, judge calibration, human studies, and analysis.

### W5 — Model experiments

Run bounded one-GPU experiments only after prerequisites pass, beginning with synthetic-only training.

### W6 — Public research artifact

Release datasets, documentation, query tools, model cards, negative results, and reproducible manifests when
the relevant release gate is met.

## 11. Stage gates

| Gate | Required evidence | What it unlocks |
|---|---|---|
| G0 — PRD ratification | Suite reviewed; contradictions and open choices logged | Implementation proposal |
| G1 — Ledger integrity | Fresh database, migrations, round-trip, hashes, rejection retention, export reproducibility | Pilot generation |
| G2 — Generator calibration | Small cross-model batches, human audit, measured yield and cost, no dominant style signature | Larger pilot |
| G3 — Evaluation freeze | Private family splits, human agreement, shortcut checks, ordinary chat baseline | Training experiment |
| G4 — Corpus pilot | Coverage and quality thresholds across independent families; no leakage | Synthetic-only run |
| G5 — Behavioral prerequisite | Reliable nonempty, finite, role-correct ordinary responses | Conceptual interpretation |
| G6 — Primary causal result | Predeclared family-level comparison, multiple seeds if affordable, honest null handling | Scale decision |
| G7 — Release | License/provenance audit, manifest reproducibility, privacy/safety review | Public dataset/model |

No later gate can retroactively turn an earlier failure into a pass.

## 12. Success and failure

### 12.1 Program success

The program succeeds if it produces:

- a conversational model people prefer to continue talking with;
- measurable held-out gains in conceptual-pact use, local revision, and cross-realization transfer;
- no material degradation of ordinary conversational contingency or length control;
- a reusable corpus ledger from which every model-visible byte is reconstructable;
- credible negative controls showing gains depend on correct conceptual relations rather than formatting;
- public artifacts whose claims match the evidence.

### 12.2 Informative partial success

Examples include:

- the curriculum works only after stronger response-initiation preparation;
- some concept classes transfer while others remain lexical;
- local revision works but legitimate plurality collapses;
- Alpha remains natural only below a particular density of metalinguistic examples;
- a larger one-GPU configuration succeeds where a smaller one does not;
- the ledger and benchmark are useful despite a null training result.

### 12.3 Failure conditions

- the model mostly emits empty responses, loops, or role leakage;
- it sounds philosophical but cannot pass minimal consequence and non-entailment probes;
- it asks clarifying questions reflexively;
- it responds with canned “on one hand/on the other hand” structures;
- correct relations do not outperform corrupted ones;
- held-out transfer vanishes after lexical, teacher, and template controls;
- judge approval does not survive calibrated human review;
- the corpus cannot be exactly reconstructed;
- rejected or disputed records are overwritten;
- the training result depends on undeclared external data in the primary synthetic-only condition.

## 13. Decisions fixed by operator direction

- Synthetic data generation is approximately half the project, not an afterthought.
- The corpus system must be capable of rich categorization and future third-party training-set construction.
- A strong reasoning model orchestrates design and exceptional decisions; economical 5.x-class workers do
  much of the bulk generation, with batch review and selective independent criticism.
- The model aims to be small enough for the available single GPU; exact parameter targets are deliberately
  absent.
- The first training experiment uses only generated synthetic data.
- Documentation is the only authorized action in the current phase.

## 14. Open decisions that must not be silently guessed during implementation

- Which exact model/provider registry is available and within subscription limits at execution time?
- Which parts of the first model are trained from scratch versus initialized from an existing base?
- What context length is affordable and sufficient for sustained conceptual pacts?
- Which human reviewers and expertise domains are available?
- Which licenses will govern raw candidates, reviewed units, annotations, and releases?
- What cultural or linguistic categories require community authority rather than generic model generation?
- Which subset, if any, receives executable formal oracles?
- What is the bounded dollar/token/GPU-hour authorization for each gate?

These choices become dated decision records. They do not belong as hidden assumptions in scripts.

## 15. Immediate next action after this documentation phase

Do not generate data. Conduct a structured external review using Appendix C, reconcile feedback into dated
decision records, and seek explicit authorization for only G1: implementing and proving the empty ledger plus
minimal query/export round-trip. Generation, training, and GPU spend remain separate later approvals.
