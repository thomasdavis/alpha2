# PRD-10 — Research claims, prior art, and publication strategy

## 1. Purpose

This PRD prevents Alpha's product vision, engineering rigor, and scientific novelty from being conflated. The
combination is distinctive, but nearly every broad ingredient has precedent. Claims must therefore name the
specific learning intervention, held-out behavior, and negative controls.

This is a bounded audit, not proof that no adjacent work exists. It must be refreshed before submission.

## 2. Product claim versus scientific claim

### Product claim

Alpha aims to be a small, natural, intellectually alive conversational partner specialized in language,
ontology, philosophy, evidence, and conceptual inquiry while relying on external retrieval for long-tail facts.

That is a coherent product identity. It is not by itself a novel scientific result.

### Primary scientific claim

> At equal synthetic-data and compute budgets, coherent multi-turn conceptual trajectories teach a small
> conversational model to establish purpose-bounded local meanings, apply their inferential consequences,
> revise dependent commitments locally, preserve legitimate alternatives, and transfer the underlying
> distinction to an unseen realization better than isolated or conceptually corrupted synthetic examples,
> without degrading ordinary conversation.

### Data/infrastructure contribution

> Alpha Corpus is an immutable scientific ledger for synthetic conversational curricula in which concept
> families, transformations, admissible analyses, rejections, reviews, rendered bytes, and model exposures are
> reconstructable and independently selectable for later training or post-training.

This is primarily a rigor and reuse claim until empirical work shows that the ledger enables otherwise
difficult science.

## 3. Closest research

| Area | Primary work | Collision with Alpha | Surviving opening |
|---|---|---|---|
| Conceptual pacts | [Kennington 2024](https://aclanthology.org/2024.lrec-main.327/) | Temporary partner-specific conceptualization and small dynamic language models already exist | Inferential consequences, revision dependencies, purpose and cross-domain transfer |
| Convention post-training | [Hua, Wang, and Artzi 2025/2026](https://arxiv.org/abs/2508.06482) | Targeted post-training for efficient convention formation, including document-grounded text | Conceptual boundary/revision rather than naming or compression; synthetic-only foundation |
| Human–AI convention mismatch | [Jones et al. 2026](https://arxiv.org/abs/2602.08208) | Humans and models form conventions but heterogeneous pairs lag | Explicitly train and test human–Alpha negotiation, not Alpha–Alpha success |
| Common ground | [Anikina et al. 2025 survey](https://aclanthology.org/2025.luhme-1.2/) | Common ground is a broad mature field | Particular synthetic intervention and inferential-pact benchmark |
| Public commitments | [Maudet et al. 2015](https://aclanthology.org/W15-0131/) | Dialogue commitments, denial, correction and ambiguity have formal precedent | Training natural conversational behavior with dependent local revision |
| QUD/common-ground tracking | [Khebour et al. 2024](https://aclanthology.org/2024.lrec-main.318/) | Shared beliefs, Questions Under Discussion and update operations already modeled | Open-ended conceptual pact trajectories and small-model curriculum effects |
| Metalinguistic negotiation | [Plunkett and Sundell 2023](https://doi.org/10.1007/s11245-023-09941-2) | Negotiating how a term should be used is established philosophy of language | Operational training/evaluation over natural multi-turn model behavior |
| Conceptual engineering with LLMs | [Allen 2024](https://arxiv.org/abs/2312.03749) | LLM-supported stipulative classification and rationale generation exist | Learning local update behavior rather than applying a supplied definition |
| Entity-light learning | [Knowledgeless Language Models](https://arxiv.org/abs/2607.12831) | Entity anonymization and context reliance are directly occupied | Entity-light multi-turn conceptual dialogue and synthetic-only controlled data |
| Synthetic multi-agent review | [Review-Instruct](https://aclanthology.org/2025.findings-acl.851/) | Candidate/reviewers/chairman generation is established | Family contracts, preserved rejection/disagreement and causal data use |
| Counterexample repair | [The Counterexample Game](https://arxiv.org/abs/2605.03936) | Iterative definition/counterexample repair directly overlaps | Human-calibrated local revision, compression and conversation rather than repeated definition growth |
| Synthetic users | [Lost in Simulation](https://arxiv.org/abs/2601.17087) | Simulated users can miscalibrate behavior and success | Treat simulations as candidates and include real human product validation |
| Dialogue-only training | [Dialogue Is Not Enough](https://arxiv.org/abs/2510.20358) | More dialogue alone does not guarantee broad competence | Controlled semantic structure inside dialogue plus ordinary language substrate |
| Structured language-learning tasks | [L2T](https://aclanthology.org/2026.acl-short.27/) | Direct linguistic learning tasks during pretraining are occupied | Natural collaborative conversation and language–ontology transfer |
| Behavioral transformations | [CheckList](https://aclanthology.org/2020.acl-main.442/), [Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/) | Invariance, directional expectation and minimal perturbation are established | Set-valued commitment deltas and training intervention, not evaluation alone |
| Metamorphic families | [NormWorlds-CF](https://arxiv.org/abs/2607.03957) | Solver-backed root families, transformations and change records are close | Open conversational semantics, purpose, attribution, plurality, and cross-realization transport |
| Ambiguity preservation | [AmbiEnt](https://arxiv.org/abs/2304.14399) | Recognition and retention of multiple meanings are established targets | Typed lifecycle, revision locality and inference under finite admissible sets |
| Belief revision | [Belief-R](https://aclanthology.org/2024.emnlp-main.586/) | Revising conclusions while retaining unaffected beliefs is active work | Beyond logical premises into time, perspective, local terms, ontology and ordinary dialogue |

## 4. Claims Alpha must not make

Alpha must not claim to have invented:

- conceptual pacts;
- convention formation;
- common-ground or commitment tracking;
- counterfactual/minimal-pair augmentation;
- metamorphic relations;
- synthetic multi-agent generation;
- model-judge review;
- conceptual engineering;
- ambiguity-aware evaluation;
- belief-revision locality;
- entity anonymization or knowledgeless models;
- dialogue-specialized small models;
- a provenance database merely by using SQLite;
- predicate minting as an isolated idea;
- philosophical fine-tuning;
- synthetic data at scale.

## 5. Strongest differentiators

### 5.1 Inferential, not merely referential, pacts

The pact changes what follows, what does not follow, what remains unresolved, and which later conclusions
depend on it.

### 5.2 Natural conversation as the behavioral surface

Formal states and category labels govern generation and scoring but do not become Alpha's default voice.

### 5.3 Synthetic-only controlled foundation

The primary run deliberately tests whether the generated curriculum can supply both language substrate and
specialization, rather than applying a thin intervention to an opaque pretrained base.

### 5.4 Donto-derived open categorical abundance

The curriculum uses a broad lens system and allows new distinctions to be minted with provenance, examples,
boundaries, and later query-time alignment. This is more than selecting topics from a fixed ontology.

### 5.5 Relation visibility as a causal variable

The same underlying semantic content is compared when isolated, placed in coherent trajectories, or
corrupted. SQLite relationship metadata alone is explicitly not treated as learning signal.

### 5.6 Local revision plus plurality

The system jointly scores what must change, what must not change, and what may remain a finite set of analyses.

### 5.7 Cross-realization transport with false bridges

The test requires a relation learned in one realization to produce the corresponding delta in another while
rejecting surface-similar non-isomorphisms.

### 5.8 Complete data-to-behavior lineage

Every accepted and rejected candidate, rendered byte, exposure, and output can be related to a family and
review decision. The claim is usefulness for causal audit, not novelty from normalization alone.

## 6. Falsifiable hypotheses

### H1 — Synthetic conversational viability

The synthetic-only release can produce reliable ordinary free conversation. Failure is measured by nonresponse,
loops, role leakage, stopping, contingency, and human judgments—not hidden by loss.

### H2 — Relational trajectory effect

Correct coherent trajectories outperform isolated and corrupted conditions on whole-family revision locality.

### H3 — Cross-realization transport

Training on one projection improves held-out application in another after lexical, template, and scenario
controls.

### H4 — Legitimate plurality

Linked plural-analysis training improves admissible-set precision and recall without raising false-ambiguity
or overhedging rates.

### H5 — Conversational non-degradation

Conceptual specialization does not reduce directness, naturalness, answer-and-stop behavior, or desire to
continue.

### H6 — Open-lens utility

Some reviewed newly minted lens/family types produce held-out query or behavior lift beyond nearest existing
categories. A beautiful name without lift or stable boundary fails.

### H7 — Evidence-first allocation

Entity-light source-conditioned synthetic training improves attribution, context use, abstention, and conflict
handling without destroying ordinary world-schema reasoning.

## 7. Reviewer-resistant controls

- equal targeted tokens and response starts;
- correct versus corrupted relation;
- attention-visible versus metadata-only linkage;
- whole-family and cross-projection holdout;
- lexical terminology removal;
- same-word/different-structure control;
- different-word/same-structure control;
- teacher/provider/template/source grouping;
- human-calibrated counterexamples;
- style-scrubbed conceptual review;
- ordinary-chat non-degradation;
- family-level statistics;
- multiple checkpoints and seeds within budget;
- private evaluation frozen before generation release;
- synthetic users separated from human product evidence;
- no private-test selection.

## 8. Null-result value

Publishable or useful negative results may include:

- synthetic-only language fails despite mechanically sound training;
- relational packaging does not outperform isolated examples;
- correct and corrupted relationships perform alike;
- local rules are learned lexically but do not transfer;
- plurality training causes overhedging;
- conceptual curriculum damages natural conversation;
- response initiation remains a distinct failure mode;
- particular Donto lenses are too underdetermined for reliable synthetic ground truth;
- model critics systematically approve invalid counterexamples;
- the one-GPU model configuration cannot integrate the target despite a stronger feasible control.

Negative results must retain exact corpus and exposure evidence.

## 9. Publication decomposition

### Paper 1 — AlphaPact

**Working title:** *Beyond Naming: Inferential Conceptual Pacts in Dialogue*

Contributions:

- behavioral construct;
- benchmark and human adjudication;
- adoption, drift, revision locality, scope, plurality and transfer;
- existing-model analysis.

### Paper 2 — Alpha Curriculum

**Working title:** *Linked Conceptual Trajectories Teach Local Revision in Synthetic-Only Conversational Models*

Contributions:

- equal-budget isolated/linked/corrupted study;
- family-held-out primary result;
- ordinary-chat non-degradation;
- one-GPU reproducibility.

### Paper 3 — Alpha Evidence

**Working title:** *Entity-Light Synthetic Dialogue for Evidence-First Conversation*

Contributions:

- fictional/anonymized/source-conditioned conditions;
- context-memory conflict;
- attribution, abstention, and retrieval readiness.

### Research artifact — Alpha Corpus Ledger

Contributions:

- reusable normalized corpus and rejection population;
- queryable category/family topology;
- deterministic exports;
- exact generation-to-exposure lineage.

The ledger paper follows demonstrated use; infrastructure scope does not delay the first causal test.

## 10. Claim register

Every public claim receives:

- precise statement;
- primary/secondary/exploratory status;
- preregistered endpoint;
- population and unit of analysis;
- comparison/control;
- supporting artifacts;
- counterevidence;
- nearest prior art;
- surviving narrower wording;
- limitations;
- responsible author and adjudication date.

“The model understands ontology” is never an acceptable standalone claim. Prefer “the model preserved person
identity while retracting a time-qualified role on held-out families under the frozen protocol.”

## 11. Novelty statement

The defensible novelty statement is:

> Alpha investigates whether a one-GPU conversational model trained only on a deliberately constructed
> synthetic corpus can learn inferential conceptual pacts: purpose-bounded local meanings whose consequences
> remain usable after delay, whose dependent commitments can be revised without collateral churn, whose
> legitimate alternatives can remain unresolved, and whose underlying distinctions can transfer across
> lexically isolated linguistic and ontological realizations. The decisive intervention holds synthetic family
> content and budget as constant as practical while varying whether its conceptual trajectory is isolated,
> coherently model-visible, or corrupted, and requires ordinary conversational behavior not to degrade.

## 12. Prior-art refresh protocol

Before any paper or public novelty claim:

1. search current arXiv, ACL Anthology, major dialogue/ML venues, PhilPapers/discipline sources, and relevant
   dataset/system repositories;
2. read primary papers rather than search snippets;
3. record date and query scope;
4. add collisions and counterclaims;
5. revise claim wording before running expensive confirmatory studies where possible;
6. label preprints and publication status accurately;
7. distinguish search finding from proof of absence.

## 13. Acceptance criteria

Research claims are ready only when:

- all broad novelty collisions are disclosed;
- the primary endpoint and controls are frozen;
- product value is separated from novelty;
- the relational signal is model-visible;
- synthetic-only and pretrained-post-training questions are not conflated;
- human product evaluation is not replaced by synthetic users;
- model judges are calibrated and subordinate on contested cases;
- null outcomes have predeclared interpretations;
- public wording is supported by exact artifacts and family-level analysis.
