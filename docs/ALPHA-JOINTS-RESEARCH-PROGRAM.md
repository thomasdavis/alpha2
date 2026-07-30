# Alpha Joints: typed semantic equivariances across linguistic and ontological projections

**Status:** proposed successor research program; no generation or training authorized  
**Date:** 2026-07-30  
**Repository:** `alpha2`  
**Predecessor brief:** [`RESEARCH-MODEL-DATASET-BRIEF.md`](RESEARCH-MODEL-DATASET-BRIEF.md)  
**Target student:** Alpha, 57,688,576 parameters, 1,024-token context  
**Model-visible medium:** natural-language conversation only  
**Canonical pilot data substrate:** delimiter-independent, versioned SQLite  

## 1. Executive decision

The project will no longer organize itself around “generate 200,000 good conversations.” That remains a
possible later production ceiling, conditional on evidence. The scientific object is now an
**Executable Conceptual Neighborhood**: a linked family of natural-language episodes whose controlled
differences specify what a concept requires the model to preserve, add, retract, qualify, attribute, or
leave legitimately plural.

The central question is:

> Can a sub-100M conversational language model learn typed semantic transformations in one domain and
> transport the corresponding commitment changes into a lexically different, previously unseen
> domain—while preserving legitimate ambiguity rather than forcing a single answer?

The decisive evidence is **cross-domain abstraction transfer**. A model receives no credit merely for
using the correct technical term. It must apply a distinction learned through one realization to an
unseen realization in another domain: for example, from temporary institutional roles to the semantics
of “former student,” or from evidential language to provenance-preserving database advice.

The first pilot is **24–60 deeply specified concept families**, with approximately 30 as the decisive
design point, not 300 neighborhoods or 200,000 rows. A 300-neighborhood study is now a later expansion
gate. Scaling is forbidden until correct relation visibility beats independent targeted examples and
corrupted-relation controls on lexically isolated, whole-family cross-projection tests.

## 2. Authority boundary

This document authorizes research and documentation only. It does **not** authorize:

- generating pilot or production data;
- changing Alpha's model, trainer, loader, objective, tokenizer, serving code, or public model;
- provisioning a GPU, RunPod, or other paid compute;
- running initiation repair, midtraining, SFT, auxiliary-head, interpretability, or positive-control
  experiments;
- building the SQLite schema or importing records;
- changing or tuning against the archived frozen evaluation;
- deleting, replacing, or relabeling the failed Alpha run or its evidence.

Each later stage requires a separate contract with a cost ceiling, exact artifacts, stop conditions, and
acceptance gates.

## 3. The three programs must remain separated

The predecessor brief allowed three valuable projects to compete for the same result. Alpha Joints
orders them and prevents causal confusion.

| Program | Scientific role | Primary question | May claim conceptual novelty? |
|---|---|---|---|
| **P0 — Alpha Initiation** | Prerequisite engineering experiment | Can the student reliably begin and sustain ordinary replies? | No; it removes a measurement confound |
| **P1 — Alpha Joints** | Core learning-science experiment | Does visible, correct relational structure produce cross-projection commitment-delta transport? | Yes, if controls pass |
| **P2 — Alpha Ledger** | Comprehensive scientific data substrate, implemented in phases | Can every dataset object, decision, exposure, and result be reconstructed, queried, and audited? | Separate systems contribution only after independent evaluation |

P0 must pass before P1 training. P2 is comprehensive by design, but P1 must not wait for every expensive
derived table to be materialized before the decisive scientific objects are frozen. A systems paper and
a learning paper may eventually emerge; their claims, ablations, and acceptance criteria remain
separate. Phased implementation is not permission to discard data or narrow the eventual ledger.

## 4. Replacement novelty statement

> **Alpha Joints investigates whether a sub-100M conversational language model can learn typed semantic
> equivariances from relational families of natural-language interactions and transport those
> equivariances across lexically isolated linguistic and ontological projections. Each family specifies
> commitments that must remain invariant, commitments that must change under temporal, evidential,
> perspectival, or granularity interventions, and finite sets of interpretations that may legitimately
> remain unresolved. The decisive comparison holds the underlying semantic episodes and budget constant
> where possible, accounts separately for added comparison or supervision tokens, and varies whether
> transformation relations are invisible, attention-visible, or explicitly supervised. Success
> requires localized commitment revision, calibrated preservation of plurality, whole-family and
> composed-transformation generalization, and cross-projection transfer that survives terminology,
> template, teacher, and false-analogy controls.**

Unlike solver-defined metamorphic rule worlds, the target relations concern open-ended conversational
semantics, attributed viewpoints, purpose-sensitive representation choices, and set-valued admissible
analyses. Unlike conventional contrast sets and counterfactual augmentation, the primary endpoint is
transport of a structured output delta into an unseen domain rather than label consistency within one
task.

This is a proposed claim, not a finding. It survives only if correct relation visibility adds value beyond:

- the response-initiation repair;
- an equal-token generic conversation control;
- the same targeted episodes treated as independent rows;
- technical-vocabulary memorization;
- a formatting, context-length, or gradient-co-location effect;
- randomly paired neighborhoods or correctly formatted but permuted delta labels;
- a larger student's capacity advantage;
- evaluator or teacher-family bias.

The first learning paper must not lead with SQLite, synthetic generation, parameter-count novelty, or a
claim that Alpha “understands concepts.” The database is reproducibility infrastructure. Synthetic
teachers are candidate constructors. The behavioral claim is narrower: the model preserves and changes
specified commitments under controlled transformations and transports those transformation laws into
held-out projections. This publication positioning does not reduce the Alpha Ledger’s scope.

## 5. Frozen Alpha baseline

### 5.1 Verified facts

| Item | Frozen result |
|---|---:|
| Parameters | 57,688,576 |
| Architecture | Llama-form, 16 layers, width 512, 8 attention heads |
| Tokenizer | Alpha byte BPE, vocabulary 12,288 |
| Context | 1,024 tokens |
| Base pretraining | 1,000,013,824 tokens |
| SFT conversations | 511,428 |
| SFT epochs | 1 |
| SFT padded positions | 496,795,648 |
| Final train / held-out loss | 1.7579851 / 1.6439665 |
| Terminal structural pass | 2 / 100 |
| Terminal empty responses | 92 / 100 |
| Terminal loops | 6 / 100 |
| Blinded semantic review | 0 PASS / 100 FAIL |
| Closed-book QA | 0 / 200 exact; 0 contained |

The existing SFT corpus is already overwhelmingly synthetic. SmolTalk accounts for 450,402 of 511,428
rows, or 88.0675%, and [SmolTalk describes itself as a synthetic SFT dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
Synthetic generation alone is therefore not the proposed intervention.

### 5.2 Evidence locations

| Evidence | Location |
|---|---|
| Program closeout | `HANDOFF.md` |
| Current-state summary | `docs/resume/CURRENT-STATE.md` |
| Failure analysis | `docs/resume/FAILURE-ANALYSIS.md` |
| Acceptance gates | `docs/resume/ACCEPTANCE-GATES.md` |
| Evidence index | `docs/resume/EVIDENCE-INDEX.md` |
| Original corpus contract | `docs/SFT_CORPUS.md` |
| Frozen evaluation | `docs/FROZEN_EVAL.md` |
| SFT manifest | `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt.manifest.json` |
| Terminal run | `/mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/` |
| Native checkpoints | `ajaxdavis/alpha-60m-training-checkpoints`, revision `7198d1a1f094ffe88d06399ea99fecbd78fa8b66` |

SFT corpus SHA-256:
`ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb`.

Terminal native checkpoint SHA-256:
`6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8`.

### 5.3 Evidence status

- **Fact:** the terminal model usually selected EOS immediately, and all nonempty terminal responses were
  still unusable.
- **Strong diagnosis:** token-averaged teacher forcing did not protect answer initiation; long ordered
  source blocks plausibly contributed recency bias and forgetting.
- **Unproven causal claim:** shuffling or answer-start weighting alone will repair the behavior.
- **Research hypothesis:** once initiation is independently repaired, relational conceptual
  neighborhoods can teach transfer at this scale.

No later document may silently upgrade the diagnosis or hypothesis into a verified cause.

## 6. Prior-art adjudication

The novelty is not the parameter count, synthetic teachers, controlled variation, linked transformation
families, preserve/change records, family-level holdouts, relation-aware rewards, linguistic tasks,
ontology competency questions, ambiguity preservation, belief revision, plural perspectives, or granular
provenance individually. Each has close prior work.

| Work | Verified overlap | Gap relevant to Alpha Joints |
|---|---|---|
| [NormWorlds-CF](https://arxiv.org/abs/2607.03957) | July 2026 preprint with 270 solver-verified root families, 1,080 canonical-to-variant pairs, compact change records, family splits, and a metamorphic-relation reward on 1.7B/4B models | The closest collision; its closed rule worlds do not test open-ended conversational semantics, linguistic–ontology transport, or finite admissible interpretations in a sub-100M student |
| [Variation Theory counterfactual augmentation](https://aclanthology.org/2025.findings-acl.50/) | Generates controlled variations intended to expose what varies and what remains invariant | Classification/active learning rather than structured conversational commitment deltas |
| [CheckList](https://aclanthology.org/2020.acl-main.442/) | Establishes minimum-functionality, invariance, and directional-expectation behavioral tests | Evaluation methodology, not cross-projection training and transport |
| [Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/) | Uses small, meaningful perturbations to probe local decision boundaries | Mostly task-local label changes rather than set-valued semantic transport |
| [Metamorphic testing survey](https://arxiv.org/abs/2605.13898) | Places invariance and predictable relations among transformed executions in a broad established testing lineage | Alpha must specify a narrower open-ended semantic relation, not claim metamorphic testing itself |
| [Counterfactually augmented data](https://arxiv.org/abs/1909.12434) | Uses minimal edits to change a target while retaining irrelevant content | Does not by itself establish cross-domain abstraction; a controlled SNLI study found no general transfer advantage and sometimes reduced robustness ([Huang et al.](https://arxiv.org/abs/2010.04762)) |
| [AmbiEnt](https://arxiv.org/abs/2304.14399) | Contains 1,645 linguist-annotated ambiguous examples and evaluates preservation of multiple meanings | Ambiguity is not novel; the remaining opening is set-valued equivariance under typed interventions and transport |
| [Baby Llama](https://arxiv.org/abs/2308.02019) | Distilled an ensemble into a 58M-parameter LLaMA trained from a 10M-word BabyLM corpus | Not open-ended conceptual dialogue or cross-domain linguistic–ontology transfer |
| [LLM-designed study plans](https://aclanthology.org/2025.babylm-main.33/) | Teacher automatically designed 56 tasks and generated a multitask pretraining corpus competitive with same-size human text | Independent tasks, not commitment-delta neighborhoods or plural analyses |
| [L2T](https://aclanthology.org/2026.acl-short.27.pdf) | Mixed 14 Language Learning Tasks with ordinary pretraining; improved and accelerated linguistic competence in 500M/1B models | Not tiny conversational students, ontology, or cross-projection transfer |
| [ContingentChat](https://aclanthology.org/2025.babylm-main.25/) | Targeted teacher–student post-training improved grammaticality and cohesion in a 100M-word BabyLM | Dialogue contingency, not cross-projection semantic-delta transport |
| [Llamalogue](https://aclanthology.org/2025.babylm-main.29/) | Dialogue-only pretraining improved dialogue continuation while underperforming on most standard BabyLM benchmarks | Warns that specialization can interfere; no language–ontology bridge |
| [Speaking of Language](https://aclanthology.org/2026.bigpicture-main.9/) | Identifies natural and symbolic metalanguage as an understudied NLP research area | Research agenda, not the proposed tiny-student relational curriculum |
| [LingGym](https://aclanthology.org/2025.emnlp-main.69/) | Tests metalinguistic inference from IGT and grammatical descriptions in 18 typologically diverse grammars | Evaluation rather than a conversational training object or ontology transfer |
| [WALS metalinguistic evaluation](https://arxiv.org/html/2602.02182v2) | Converts WALS features across 2,660 languages into explicit questions and finds fragmented, resource-linked performance | Factual feature QA, not transformations among commitments |
| [Ontology Generation using LLMs](https://arxiv.org/html/2503.05388v1) | Generates OWL drafts from user stories and competency questions; evaluates across ten ontologies | Ontology authoring, not natural-language conceptual behavior in a tiny student |
| [Standpoint EL](https://arxiv.org/abs/2302.13187) | Formally represents diverse, possibly conflicting standpoints while retaining tractable reasoning | External logical representation rather than learned conversational behavior |
| [Polyvocal ontology proposal](https://aclanthology.org/2026.eacl-srw.46/) | Proposes perspective-aware extraction into epistemically separate, provenance-bearing knowledge graphs | A research proposal for extraction/KGs, not student internalization |
| [The Counterexample Game](https://arxiv.org/abs/2605.03936) | Iterates definition, counterexample, and repair; finds diminishing returns and a permissive LM judge | One repair chain rather than typed neighborhoods and cross-domain transfer |
| [Belief-R](https://aclanthology.org/2024.emnlp-main.586/) | Tests belief revision after new evidence and exposes a trade-off between updating and preserving unaffected beliefs | Not linguistic ambiguity, attribution, temporality, and ontology jointly |
| [DeltaLogic](https://arxiv.org/abs/2604.02733) | Uses minimal premise edits to expose belief-revision failures in logical reasoning models | Closed logical cases rather than open conversational projection transport |
| [RippleEdits](https://arxiv.org/abs/2307.12976) | Tests whether knowledge edits propagate to related facts without disturbing unrelated ones | Model editing rather than learning semantic transformation families |
| [Relation learning and cross-domain generalization](https://arxiv.org/abs/1910.05065) | Treats structured relational representation as a basis for transfer across domains | Makes cross-domain transfer an established goal; Alpha must define the transported delta and leakage controls precisely |
| [OriginBlame](https://arxiv.org/html/2607.13037v1) | Propagates record/token-level provenance and resolves exact forget sets | Provenance engineering without rich conceptual semantics |
| [Generated-corpus redundancy study](https://arxiv.org/abs/2606.29605) | Shows raw synthetic token volume can greatly overstate unique information in a clinical corpus | Different domain, but directly warns against row/token count as the research variable |

NormWorlds-CF invalidates any claim that linked executable families, explicit change records,
family-level splits, or metamorphic-relation supervision are new. CheckList, contrast sets, Variation
Theory, counterfactual augmentation, and metamorphic testing likewise establish the broader methodological
lineage. AmbiEnt establishes ambiguity preservation as an existing target; Belief-R, DeltaLogic, and
RippleEdits establish revision locality as an existing target; Standpoint EL and polyvocal ontology work
establish perspective-preserving representation as an existing target.

The narrower surviving opening found in this bounded search is their conjunction as one causal question:

1. a sub-100M conversational student;
2. open-ended, natural-language commitment states rather than solver-defined rule-world labels;
3. finite required/permitted/forbidden commitments and admissible analyses;
4. transport of a typed delta between lexically and scenically isolated linguistic and ontological
   projections;
5. identical-content treatments in which relation visibility increases from absent, to optimizer-only,
   to attention-visible, to explicitly supervised;
6. false-bridge and corrupted-relation controls;
7. behavioral transfer paired with predeclared cross-projection probing and causal intervention.

This is a search finding, not proof of absence. It must be narrowed again if closer work appears.

## 7. The canonical scientific object

Alpha Joints does not currently define a mathematical topology. “Topology of concepts” is therefore
retired as a formal claim. The precise object is a **typed semantic transition system** whose edges
specify invariance, equivariance, or set-valued change over behavioral commitment states. “Conceptual
neighborhood” remains the intuitive dataset term; “executable semantic transformation family” and
“typed semantic equivariance” are the methodological terms.

### 7.1 Executable Conceptual Neighborhood

A neighborhood is represented conceptually as:

```text
N = (V, E, Gamma, P)
```

where:

- `V` is a set of model-visible natural-language conversational episodes;
- `E` is a set of typed, directed transformations between episodes;
- `Gamma` is the hidden set of commitment and analysis constraints associated with nodes and edges;
- `P` is the set of domain projections through which the latent distinction is realized.

“Executable” means the structure can produce controlled comparisons and score relational behavior. It
does not mean the model sees JSON, a graph language, logical notation, or database IDs.

### 7.2 Episode

An episode is one complete learning interaction:

- a short user/assistant exchange;
- a two-to-six-turn dialogue;
- a minimal contrast with explanation;
- a counterexample and repair;
- an evidence update;
- a competency-question discussion;
- a teach-back or transfer check.

Episodes remain serialization units. Neighborhoods are the allocation, splitting, and scientific units.
Counting episodes without counting distinct neighborhoods and transformation coverage is insufficient.

### 7.3 Commitment

A commitment is a proposition, distinction, attribution, or qualified stance that a defensible answer
should express or preserve. The term is behavioral: it does not assume that the model literally stores
propositional beliefs.

A commitment record includes:

- stable identifier and natural-language gloss;
- scope and presuppositions;
- polarity or status;
- source/speaker/theory attribution where applicable;
- valid-time and record-time qualification where applicable;
- granularity or perspective;
- confidence and whether the commitment is required, permitted, or forbidden;
- relations to other commitments.

### 7.4 Expected commitment delta

For an edge from episode `i` to episode `j`, the hidden contract is:

```text
Delta(i -> j) = {
  preserve,
  add,
  retract,
  pluralize,
  attribute,
  temporalize,
  change_granularity,
  request_clarification,
  remain_unresolved
}
```

Each set contains commitment identifiers and conditions. Categories are not mutually exclusive across
the whole edge; a temporal update may preserve entity identity, retract a current role, add a historical
role, and retain an attributed colloquial description.

### 7.5 Admissible analysis set

Some episodes have one expected analysis. Others have a finite set of defensible readings or modeling
choices. The contract records:

- admissible analyses;
- excluded analyses and why they fail;
- evidence that would select among admissible analyses;
- whether clarification is possible and the smallest useful question;
- whether plurality is permanent, theory-relative, perspective-relative, or merely unresolved.

The goal is not maximal hedging. Proposing ten unsupported readings where two are defensible is a failure.

Every scorable state must distinguish:

- **required commitments:** every defensible answer must retain them;
- **permitted commitments:** an answer may include them under an admissible analysis;
- **forbidden commitments:** the available evidence or contract excludes them;
- **admissible analyses:** finite coherent groupings of commitments;
- **excluded analyses:** plausible-looking but unsupported or contradicted groupings;
- **resolvability:** whether further evidence or a minimal clarification can reduce the admissible set.

### 7.6 Projection

A projection is one realization of a latent distinction. Required projection families are:

- linguistic form and interpretation;
- ontology or knowledge representation;
- ordinary social/institutional reasoning;
- evidence, records, and provenance;
- temporal change;
- cross-linguistic or translation realization where authority and sources permit.

No neighborhood needs every projection. Every high-priority latent distinction needs at least two, and
the bridge subset needs linguistic plus ontological projections by definition.

### 7.7 Cross-projection delta transport

Let `s` denote a hidden behavioral commitment state, and let transformation `e` induce a licensed change
`Delta_e(s)`. Let `phi_(p->q)` map a latent distinction from source projection `p` into target projection
`q`. The primary hypothesis is that the learned transformation approximately commutes with projection:

```text
phi_(p->q)(Delta_e(s)) ~= Delta_e'(phi_(p->q)(s))
```

In plain language: after learning what a transformation changes in projection `p`, does the model make
the corresponding change in an unseen projection `q`? The two sides need not share words, scenarios, or
surface answer form. They must share a reviewed semantic dependency structure and expected commitment
delta.

For example, training may show through tenant and officeholder cases that ending a role retracts current
role membership while preserving bearer identity and historical attribution. A held-out linguistic test
then asks about “former student” without exposing *student*, *former*, *graduation*, *enrolment*, or close
scenario analogues during training. An essential-category negative control must reject mechanical
over-transfer of the same delta.

### 7.8 Relation classes

The primary relations are:

- **invariance:** paraphrase or irrelevant detail must preserve the commitment state;
- **equivariance:** a time, evidence, perspective, or granularity change must induce a corresponding,
  typed output change;
- **set-valued equivariance:** an intervention changes the finite admissible-analysis set without
  necessarily selecting one answer;
- **composition:** multiple interventions must compose when their deltas are compatible and expose
  order sensitivity when they are not.

## 8. Transformation taxonomy

| Transformation | Controlled change | Required behavior |
|---|---|---|
| Paraphrase | Surface form only | Preserve substantive commitments |
| Register/style shift | Audience or formality | Preserve content while adapting expression |
| Irrelevant detail addition | Add non-diagnostic information | Ignore distraction and preserve answer |
| Minimal meaning change | One semantically decisive feature | Change exactly affected commitments |
| Lexical substitution | Replace a term, sometimes across senses | Preserve or change according to sense, not string similarity |
| Evidence addition | Add supporting or defeating evidence | Revise only dependent conclusions |
| Evidence retraction | Withdraw one source/observation | Retract its support without deleting unrelated claims |
| Temporal shift | Move event or query time | Preserve historical truth and update current status |
| Perspective shift | Change speaker, institution, or theory | Preserve attribution; do not flatten perspectives |
| Granularity shift | Move between coarse and fine descriptions | Map levels without declaring them identical |
| Competency-question shift | Change what the user needs to ask | Change modeling recommendation when requirements change |
| Counterexample | Add a case challenging a definition | Repair locally or accept plural structure |
| Clarification response | Supply information requested by prior turn | Resolve only the ambiguity addressed |
| Genuine ambiguity | Withhold disambiguating evidence | Return the defensible analysis set |
| Cross-domain projection | Re-realize the latent distinction | Transfer structure without relying on memorized vocabulary |
| Composition | Apply two or more transformations | Produce the composition of their legitimate deltas |

Every edge must state what changed and what was held constant. A supposed minimal pair with multiple
uncontrolled differences cannot enter the pilot.

### 8.1 Formally checked core, not universal formalization

The micro-pilot should predeclare a constrained semantic calculus for the subset with sufficiently clear
contracts:

- identity preservation;
- role acquisition and termination;
- source attribution;
- valid time versus record time;
- evidence support and withdrawal;
- collective versus member;
- coarse versus fine granularity;
- explicit ambiguity branching.

A later implementation may use Lean, Datalog, answer-set programming, or a small transition calculus to
check whether deltas are internally consistent, whether compatible compositions commute, when order
should matter, whether an invariant is contradicted, and whether an admissible-analysis set is empty or
overgenerated. This document authorizes only the specification of that oracle, not its implementation.

Solver use would not itself be novel after NormWorlds-CF. The proposed contribution is connecting a
checked core to open-ended conversational commitment states, cross-domain projection transport, and
finite set-valued interpretations without pretending that every linguistic or ontological dispute can
be reduced to a proof obligation.

## 9. Core cross-domain distinctions

| Latent distinction | Linguistic projection | Ontological/evidence projection | Candidate held-out transfer |
|---|---|---|---|
| Role versus bearer | “student,” “former student,” temporary predicates | Tenant, officeholder, patient, institutional role | Train ontology; test lexical/modifier behavior |
| Collective versus members | Agreement with “committee,” distributive/collective readings | Group identity, membership, component parthood | Train group ontology; test agreement/reference |
| Event versus object | Nominalization, eventive/stative readings | Reification and event identity criteria | Train ontology; test nominalization |
| Individuation | Mass/count alternation and coercion | Counting, entity resolution, instance boundaries | Train linguistic contrasts; test database choices |
| Evidence versus assertion | Evidentials, reported speech, attribution | Claim, source, testimony, confidence | Train language; test conflicting catalogue records |
| Polysemy versus identity | Related lexical readings | One entity, several entities, or erroneous merge | Train lexical cases; test record resolution |
| Aspect and event boundaries | Progressive, perfective, telicity | Process, culmination, completion, event identity | Train aspect; test event reconciliation |
| Translation mismatch | Different lexical partitions | Alignment without false equivalence | Train ontology alignment; test translation advice |
| Identity through change | Names, definites, “same,” “former” | Versioned artifacts, institutions, renamed entities | Train thought experiments; test documents/institutions |
| Granularity | Hypernymy, lexical specificity, discourse focus | Partonomies and level-dependent categories | Train biological/organizational cases; test translation |
| Modality and commitment | May, must, counterfactual, conditional | Possibility, necessity, policy, disposition | Train linguistic modality; test ontology advice |
| Time of truth versus report | Tense, temporal adverbs, reported speech | Valid time versus record time | Train record cases; test narrative interpretation |

The strict scoring rule is:

> Correct terminology without preservation of the relevant distinction receives no conceptual-transfer
> credit.

## 10. Worked neighborhood: student

### 10.1 Base episode

**User:** Is a student a kind of person, or a role a person has?  
**Expected behavior:** distinguish a persistent person from a temporally and often institutionally
dependent role; acknowledge that ordinary class language can still be useful.

Core commitments:

- a person can bear the student role;
- the bearer and role are not identical;
- the role can begin or end while the person's identity persists;
- an ontology choice depends partly on required queries.

### 10.2 Paraphrase edge

**User:** Is being a student what somebody is, or something they are for a while?  
**Delta:** preserve all four core commitments despite changed wording and lower technical register.

### 10.3 Temporal intervention

**User:** She graduated yesterday. Is she still the same person? Is she still a student?  
**Delta:**

- preserve person identity;
- retract current institutional student-role status under the stated assumptions;
- add historical role attribution;
- preserve the truth of an earlier time-qualified “she is a student” claim.

### 10.4 Perspective intervention

**User:** Her mother still calls her “my student daughter,” but the registrar says she is no longer
enrolled.  
**Delta:**

- keep both usages attributed;
- distinguish colloquial identity-description from institutional status;
- avoid forcing the registrar's criteria onto the mother's speech or vice versa;
- explain which classification controls which practical query.

### 10.5 Competency-question intervention

**User:** I only need to print a current class list. Do I really need to model “student” as a role?  
**Delta:**

- recommend the simplest representation adequate for the narrow requirement;
- preserve the conceptual distinction;
- state what historical or multi-role questions the simplification cannot answer.

### 10.6 Linguistic projection

**User:** Why does “former student” make sense, while “former person” sounds very different?  
**Expected transfer:** use the role/bearer distinction to explain modifier interpretation without merely
repeating “student is a role.” Discuss coercion or unusual contexts rather than declaring “former person”
logically impossible.

### 10.7 Counterexample

**User:** What about someone studying independently without any institution?  
**Delta:** distinguish institutional and activity-based readings; refine rather than discard the earlier
analysis; do not manufacture unlimited ambiguity.

### 10.8 Negative controls

- Changing the person's name must not change the conceptual answer.
- Adding an irrelevant favorite color must not alter role status.
- Replacing “graduated” with “enrolled yesterday” must reverse the relevant temporal delta.
- Replacing “student” with a clearly essential category should not mechanically reproduce the role
  analysis.

This is one neighborhood. Seven polished independent answers would not encode or test these relations.

## 11. P0 — isolate response initiation first

### 11.1 Why P0 is separate

A model that emits EOS cannot expose its conceptual competence. If the conceptual dataset also repairs
response initiation, any later gain is causally ambiguous. P0 is therefore a prerequisite calibration
experiment using ordinary conversational material with no ontology or advanced metalinguistic content.

The research claim for P0 is narrow:

> In this small assistant model, free-generation failure at the first assistant position can be repaired
> and measured independently of conceptual curriculum content.

### 11.2 Required controlled factors

All P0 arms begin from the same archived base checkpoint and use the same underlying ordinary
conversation pool and supervised-token budget. At minimum isolate:

- monotonic versus deterministic shuffled/interleaved order;
- ordinary token averaging versus an explicitly predeclared answer-start or episode-normalized
  intervention;
- original length distribution versus substantial short, complete replies;
- checkpoint selection by held-out loss versus checkpoint selection by predeclared free generation.

A compact factorial is preferable to changing all four at once. The design must contain at least one
arm where only order changes and one where only answer-start treatment changes.

### 11.3 P0 frozen evaluation

Construct a new initiation-only suite before P0 training. It must not reuse the terminal frozen suite as
tuning data. It should cover:

- greetings and ordinary questions;
- direct factual questions answerable from the prompt;
- requests for short explanation;
- lightweight disagreement and correction;
- ambiguous prompts requiring a single clarification;
- prompts from 1–64, 65–160, 161–300, and 301–700 tokens;
- one-turn and multi-turn context;
- exact repeated prompts for determinism checks.

Measure first-token EOS rank/margin, nonempty rate, structural completion, relevance, answer length,
looping, and variation by prompt-length band.

### 11.4 P0 admission gate

P1 training remains closed until a predeclared P0 arm achieves, on a private final suite:

- at least 99% nonempty responses overall and within every prompt-length band;
- no systematic first-token EOS preference;
- no degenerate loops;
- at least 98% structurally complete replies;
- stable results across more than one checkpoint window and seed;
- no material regression in basic relevance or language quality;
- a complete causal report separating order, sampling/loss, and length effects.

These thresholds may be revised before the suite is frozen, never after results are seen.

## 12. P1 pilot corpus

### 12.1 Size and composition

The first scientific pilot contains **24–60 deeply formalized concept families**, with approximately 30
as the decisive design point. It deliberately sacrifices breadth for trustworthy transformations,
negative controls, and projection isolation.

Candidate starting families are:

| Concept family | Candidate projections |
|---|---|
| Role versus bearer | Student, tenant, officeholder, patient, software permission |
| Claim versus source | Reported speech, catalogue records, database provenance |
| Valid time versus record time | Narrative tense, employment records, versioned facts |
| Collective versus members | Linguistic agreement, committees, organizations |
| Event versus object | Nominalization, event records, reification |
| Type versus token | Words, documents, products, schema instances |
| Evidence versus assertion | Evidentials, testimony, confidence-bearing claims |
| Identity through change | Former roles, renamed institutions, document versions |
| Part versus member | Components, teams, body parts, organizational structure |
| Granularity | Lexical specificity, taxonomies, organization levels, translation mismatch |

The list is a candidate set, not a hand-maintained semantic ontology. Admission depends on whether human
reviewers can certify a shared relation, isolatable projections, and meaningful false-bridge controls.
The statistical unit is the concept family, not its episode count.

The scientific counts reported for the pilot are:

- unique neighborhoods;
- unique latent distinctions;
- transformation edges by type;
- commitments and delta operations;
- admissible-analysis sets;
- projection families;
- sourced phenomena;
- accepted episodes and model-visible tokens;
- unique-information and duplicate-cluster estimates.

Episode count is last, not first.

### 12.2 Required neighborhood roles

Every accepted concept family should include:

1. at least three genuinely different projections;
2. four to six primitive transformations;
3. two composed transformations;
4. one false-analogy projection;
5. one same-word/different-relation control;
6. one different-words/same-relation control;
7. one ambiguity or clarification case where scientifically justified;
8. explicit required, permitted, and forbidden commitments;
9. a surface-preserving paraphrase and irrelevant-detail control;
10. at least one minimal meaning-changing intervention.

The aggregate pilot must cover every transformation in section 8 at a predeclared minimum frequency.

### 12.3 Domain content

High-priority linguistic topics:

- reference, predication, identity, deixis, and anaphora;
- lexical sense, polysemy, coercion, and ambiguity;
- mass/count, number, collectives, and individuation;
- tense, aspect, mood, modality, evidentiality, and event structure;
- presupposition, entailment, implicature, attribution, and common ground;
- argument structure, roles, valency, voice, and alternations;
- category diagnostics and gradient/theory-relative grammatical judgments;
- information structure, discourse coherence, repair, and perspective;
- cross-linguistic category mismatch with attested, authorized evidence.

High-priority ontological topics:

- type/token, class/instance, role/bearer, quality, disposition, and function;
- identity, individuation, persistence, constitution, and dependence;
- parthood, membership, collectives, boundaries, and granularity;
- events, processes, states, participants, causal relations, and temporal qualification;
- claims, evidence, sources, confidence, record time, and valid time;
- open-world incompleteness, contradiction, correction, and supersession;
- perspective, standpoint, social/institutional categories, and authority;
- alignment, near-equivalence, translation mismatch, and erroneous merger;
- competency questions and purpose-sensitive modeling advice.

The pilot should prefer distinctions that can be controlled and projected. It should not attempt
encyclopedic coverage of linguistics or metaphysics.

## 13. Splits designed around transfer

Random episode-level train/test splitting is prohibited.

### 13.1 Whole-neighborhood holdout

At least 20% of neighborhoods are held out in full. The student sees neither their episodes nor their
surface variants, sources, generation templates, or close semantic duplicates. This measures transfer to
new conceptual material.

### 13.2 Cross-projection holdout

For a separate set of training neighborhoods, expose one or more projections and withhold another.
Examples:

- train roles through tenant, patient, and officeholder ontology; test “student” and “former student”;
- train evidential/reporting distinctions; test conflicting catalogue provenance;
- train mass/count individuation; test entity-resolution decisions;
- train aspectual boundaries; test whether records describe one event or two.

The held-out projection must use different surface vocabulary and scenarios. A technical term appearing
on both sides invalidates a strong transfer claim unless a separate no-term analysis is reported.

### 13.3 Composed-transformation holdout

Train individual transformations and test selected compositions, such as:

- temporal shift plus perspective shift;
- evidence retraction plus granularity change;
- paraphrase plus minimal meaning edit;
- competency-question change plus ambiguous source claims.

The expected composed delta is frozen before generation of model responses.

### 13.4 Template, teacher, and source isolation

Splits also group by:

- seed conceptual family;
- source work and source fragment;
- generation prompt/template family;
- teacher campaign;
- lexical and scenario cluster;
- contrast/counterexample family;
- human author or reviewer where author style could leak.

All group assignments are materialized and hashed in SQLite before any model training.

### 13.5 False bridges and jargon ablation

Every major family needs controls designed to separate abstraction from association:

- a true bridge with different words and scenarios but the same reviewed dependency structure;
- a false bridge with similar vocabulary or topic but a different dependency structure;
- a same-word/different-relation case;
- a different-words/same-relation case;
- a technical-jargon version and a version with terms such as *role*, *provenance*, *valid time*,
  *granularity*, and *ambiguity* removed or replaced.

Transfer credit is awarded for preserving the dependency and commitment delta, never for producing the
right technical noun. Performance that does not distinguish true from false bridges is topical
association, not abstraction.

### 13.6 Development and final evaluation

- **Training:** model-visible episodes only.
- **Development:** public structure and metrics, distinct neighborhoods; usable for design selection.
- **Private final:** sealed neighborhoods, interventions, and answer contracts inaccessible to generation,
  training, and checkpoint-tuning agents.
- **Human audit:** blinded samples from all splits, with the private final adjudicated only after model
  and checkpoint choices are locked.

## 14. Decisive controlled experimental arms

All primary arms use the same base checkpoint, tokenizer, architecture, P0 initiation method, optimizer
budget, target supervised-token budget, prompt-length envelope, checkpoint cadence, and frozen
evaluations. Any tokens or objective terms introduced to expose a relation are itemized rather than
hidden inside an “equal-token” label.

| Arm | Data/training treatment | Question isolated |
|---|---|---|
| **A — generic** | Equal-token ordinary dialogue | How much comes from P0 repair and token budget? |
| **B — independent** | Targeted episodes randomized and attention-separated | Is targeted content alone sufficient? |
| **C — co-batched** | Correct same-family episodes in one minibatch but separate attention contexts | Does gradient co-location/rehearsal change learning? |
| **D — packed neighborhood** | The same content tokens placed in one jointly attention-visible sequence, with no edge labels | Does direct contextual comparison matter? |
| **E — explicit relation** | Natural-language comparison and/or explicit delta supervision, reported as distinct subarms | Does explicit relational information add value beyond visibility? |

### 14.1 Why relation visibility is the causal variable

A standard causal language model with state reset between examples cannot observe an edge merely because
two examples are consecutive or share a minibatch. Arm C changes gradient covariance, optimizer
trajectory, local rehearsal, and interference; it does not make the semantic relation attention-visible.
Calling C “relational learning” without that qualification would be a category error.

B, C, and D therefore preserve the targeted content as closely as possible while progressively changing
relation visibility: absent, optimizer-only, and attention-visible. E changes the information or
objective and must be interpreted separately. Token accounting must report semantic episode tokens,
comparison/rendering tokens, supervised tokens, and context utilization rather than asserting equality
where packing necessarily changes model-visible context.

### 14.2 Relation-corruption controls

Every relation-visible treatment requires matched negative controls:

- **random-edge control:** pair unrelated families while preserving formatting, length, schedule, and
  packing;
- **permuted-delta control:** preserve label frequency but scramble which transformation receives which
  delta;
- **corrupted-invariant control:** preserve surface similarity while marking an actually affected
  commitment as held constant, or vice versa;
- **false-bridge control:** pair topically or lexically plausible projections that do not instantiate the
  same dependency structure.

Correct relations must beat equally regular but incorrect relations. Otherwise any gain is attributable
to formatting, longer context, rehearsal, or generic contrast rather than semantic relational structure.

### 14.3 E is not allowed to rescue D invisibly

E changes the model-visible information or objective. If only E works, the supported claim is that
explicit comparison or delta supervision helps. It does not show that co-batching, packing alone, or the
database representation caused transfer. Natural-language comparison and an auxiliary loss must be
separate E subarms if both are tested.

### 14.4 Specialist factorial

Using the strongest primary treatment, compare equal-token:

- linguistic specialist;
- ontology specialist;
- integrated linguistic–ontology model.

Interpretation:

- specialists succeed, integrated fails: capacity interference or mixture design problem;
- integrated matches specialists but does not transfer: coexistence without shared abstraction;
- integrated uniquely transfers across projections: evidence for the central thesis;
- all fail while positive control succeeds: 57.7M capacity or foundation limit;
- all models fail: data/evaluation/training design likely inadequate.

### 14.5 Seeds and selection

- Prefer three independent seeds for decisive comparisons.
- Declare which comparisons can begin with successive halving.
- Select checkpoints by a predeclared composite of P0 stability, ordinary conversation, and conceptual
  development metrics.
- Never select on the private final suite.
- Report every seed, not only the best run.

## 15. Continued pretraining is a first-class contender

The project must not assume assistant-only SFT is the correct mechanism. [L2T](https://aclanthology.org/2026.acl-short.27.pdf)
shows that structured learning tasks mixed with ordinary next-token pretraining can accelerate linguistic
competence, though at much larger 500M/1B scales and token budgets.

After the initial data-organization comparison, test equal-token strategies:

1. **Targeted SFT only:** all conceptual material rendered as conversation.
2. **Conceptual midtraining plus small SFT:** descriptions, contrasts, paraphrases, examples,
   counterexamples, and transformation sequences in a continued-pretraining phase, followed by a smaller
   conversational alignment phase.
3. **Optional later preference stage:** penalize flattening, fabricated certainty, overhedging, and generic
   essay behavior only after supervised effects are understood.

Midtraining and SFT token budgets, source material, and exposure counts must be reconciled. A gain from
more total tokens is not a gain from phase structure.

## 16. Larger positive control

Run the strongest pilot comparison on one 150M–300M student using the same data, tokenizer policy where
possible, objectives, and evaluations.

Purpose:

- distinguish curriculum failure from 57.7M capacity failure;
- detect integrated-capability interference;
- establish that the evaluation is learnable;
- compare whether transfer emerges at a larger scale before concluding the relational object is wrong.

The larger model is a diagnostic control, not a replacement for Alpha and not a product launch. It
requires explicit cost authorization. If architectures differ, parameter scale and architecture effects
must not be conflated.

## 17. Candidate generation and review

### 17.1 Generate neighborhoods, not isolated answers

A generation task receives:

- one latent distinction and boundary conditions;
- source-grounded phenomena where factual claims are involved;
- required projection families;
- requested transformation edges;
- commitments and candidate delta contract;
- negative controls and likely confounds;
- style/length constraints;
- prohibited leakage vocabulary for held-out projections.

The authoring teacher proposes the neighborhood. An independent reviewer challenges:

- whether the latent distinction is coherent;
- whether each intervention changes only what it claims;
- whether commitments and deltas are complete;
- whether proposed plurality is genuine;
- whether the projection actually shares structure rather than a loose analogy;
- whether a source supports every attested language claim;
- whether a simpler confound explains the expected answer.

### 17.2 Teacher choice

Use a blinded bake-off over frontier teachers available at generation time. Evaluate the whole
neighborhood, not one attractive episode. The author, semantic challenger, naturalness judge, and final
reviser should not all be the same model family.

Record exact provider, model/version, prompt revision, parameters, date, raw output, transformation,
review, and terms. “Generated by a SOTA model” is not provenance.

### 17.3 Human authority

Humans dominate adjudication for:

- whether a counterexample genuinely defeats a definition;
- whether two analyses are admissible;
- whether a cross-domain projection preserves the same abstraction;
- theory-sensitive linguistic analyses;
- cultural, Indigenous, signed, low-resource, or community-governed language material;
- final private evaluation contracts.

[The Counterexample Game](https://arxiv.org/abs/2605.03936) found its LM judge accepted roughly twice as
many counterexamples as humans and that longer repair chains increased verbosity without improving
accuracy. Model votes cannot replace conceptual adjudication.

### 17.4 Candidate volume

Generate only enough candidates to fill the approved pilot neighborhoods under measured rejection and
revision rates. Preserve failed and rejected attempts. Report:

- candidate-to-accepted rate by edge and projection;
- invalid-minimal-intervention rate;
- duplicate/near-duplicate rate;
- teacher/judge disagreement;
- human correction rate;
- source-verification failure;
- unique-information estimates.

Do not infer scientific value from generated-token count. A 2026 production-scale study found severe
redundancy in one LLM-generated clinical corpus, illustrating why volume is not unique information;
its domain-specific numbers are not assumed to transfer to Alpha.

## 18. Evaluation as conceptual conservation laws

The primary endpoint is:

> **Cross-projection commitment-delta accuracy on lexically isolated, whole-family holdouts.**

This endpoint is evaluated against independent targeted training and matched corrupted-relation
controls. No single aggregate score may replace the component failures below.

### 18.1 Relational metrics

| Metric | Definition | Failure exposed |
|---|---|---|
| **Invariance accuracy** | Fraction of required preserved commitments retained across an edge | Distractibility or surface memorization |
| **Delta precision** | Fraction of changed commitments that should have changed | Gratuitous revision |
| **Delta recall** | Fraction of required additions/retractions/qualifications expressed | Failure to respond to decisive evidence |
| **Revision locality** | Unaffected commitments retained after an intervention | Global belief churn |
| **Analysis-set precision** | Proposed readings that are actually admissible | Indiscriminate hedging |
| **Analysis-set recall** | Required admissible readings preserved | Premature flattening |
| **Overhedging rate** | Unwarranted plurality or clarification | Avoidance disguised as nuance |
| **Attribution retention** | Claims remain attached to source/speaker/theory | Perspective collapse |
| **Temporal integrity** | Current and historical status are distinguished | Time-insensitive contradiction |
| **Granularity integrity** | Levels are mapped without false identity | Category collapse |
| **Competency sensitivity** | Advice changes with the user's required questions | Memorized ontology dogma |
| **Cross-projection transfer** | Latent distinction applied in an unseen domain realization | Jargon/topic memorization |
| **Compositional delta accuracy** | Combined interventions yield the correct combined changes | Edge memorization without composition |
| **False-bridge rejection** | Plausible but structurally invalid correspondences are rejected | Analogical over-transfer |
| **Jargon-ablation retention** | Transfer survives removal/replacement of technical terms | Terminology memorization |

### 18.2 Two evaluation channels

Use both, and never merge their scores:

1. **Diagnostic constrained probes:** controlled choices, set selection, or short propositions designed to
   reveal exact commitment state with low parsing ambiguity.
2. **Free conversational generation:** natural replies judged for the same commitments plus relevance,
   coherence, appropriate depth, and conversational quality.

The first locates representation or decision failures. The second is the actual product behavior. A
model that passes forced-choice probes but cannot express the distinction conversationally has not met
the goal.

### 18.3 Scoring free text

Do not use brittle exact-string rules for open-ended answers. The evaluation contract should combine:

- deterministic checks only where the output contract is truly deterministic;
- controlled diagnostic questions with executable keys;
- semantic commitment extraction using a versioned, schema-validated evaluator;
- human adjudication of stratified and high-risk cases;
- independent judge calibration against those human decisions;
- preserved raw outputs and per-commitment decisions.

Judge accuracy, false-plurality rate, and disagreement must be reported. No aggregate judge score may
hide which commitment was missed or invented.

### 18.4 Ordinary conversation remains a gate

Every conceptual checkpoint is also tested for:

- nonempty response and immediate EOS;
- relevance and directness;
- looping/repetition;
- response length appropriateness;
- repair and follow-up;
- prompt-length robustness;
- regression to generic textbook voice;
- regression against the P0 and base language suites.

Conceptual vocabulary cannot compensate for losing ordinary conversation.

### 18.5 Statistical unit and analysis

The inferential unit is the concept family or neighborhood, never the episode. Episodes and surface
realizations within a family are dependent observations. The analysis plan should use paired family-level
comparisons or a hierarchical model with predeclared effects for:

- concept family;
- transformation type;
- projection pair;
- teacher/template or constructor;
- model seed.

Report uncertainty across families and seeds, not confidence intervals that treat thousands of related
episodes as independent. Multiple-comparison handling and exclusions must be frozen before the private
final is opened.

### 18.6 Conceptual sample efficiency

In addition to accuracy, report:

- cross-projection gain per unique concept family;
- gain per transformation edge;
- gain per human-review hour;
- gain per model-visible and supervised token;
- marginal value of the second, third, and later projection;
- marginal value of composition examples;
- unique-information estimates after semantic duplicate clustering.

The strongest result would be that a small number of deeply structured families produce more held-out
transport than many independent targeted answers under equal token and review budgets.

## 19. P2 — comprehensive Alpha Ledger in SQLite

### 19.1 Completeness rule and implementation phases

Alpha Ledger is intended to represent the complete evolving dataset: accepted content, candidates,
rejections, revisions, competing analyses, linguistic and ontological description, transformations,
sources, authority, reviews, renderings, tokenizations, model exposures, runs, and results. “Pilot
schema” means the first implemented slice of that complete design. It does not mean that only successful
examples or paper-critical fields are retained.

The governing distinction is:

- **tracking completeness:** no scientifically relevant input, output, decision, failure, or lineage is
  discarded;
- **materialization phase:** expensive derived structures may begin as immutable content-addressed
  artifacts referenced by SQLite and later become normalized query tables;
- **paper scope:** the first learning paper need not claim the ledger as its novelty even though the
  ledger records the entire experiment.

Implement a table only if it supports at least one named requirement:

- reconstruct model-visible text;
- represent a neighborhood, transformation, commitment, analysis, or projection;
- enforce a split/leakage rule;
- trace source/license/generation/review lineage;
- reproduce a release/export/run;
- calculate a predeclared metric;
- honor a restriction or withdrawal.

Everything else still requires a preservation path. Large raw generations, tokenizer arrays, parser
outputs, embeddings, and proof artifacts may live as immutable content-addressed files or blobs with
hash, media type, byte length, producer, schema/version, and lifecycle recorded in SQLite. They may not
live only in an ephemeral work directory or be deleted because their normalized table is deferred.

### 19.2 Delimiter independence remains mandatory

Canonical utterance text contains no model-specific role marker, BOS, EOS, or chat-template wrapper.
Role and boundary semantics live in relational records. A versioned export renderer injects
`<|user|>`, `<|assistant|>`, `<|end_of_text|>`, or a future model's delimiters after cohort selection.

The database must distinguish:

- what a participant said;
- the participant's semantic/discourse role;
- delimiters injected by a renderer;
- tokenizer-specific token IDs;
- which rendered token positions receive supervision.

No import path may recover roles by parsing delimiter-like text out of canonical utterances.

### 19.3 First-class experimental tables

These objects may not be hidden as generic annotations because they define the paper's intervention.

| Table | Core fields | Scientific purpose |
|---|---|---|
| `conceptual_neighborhood` | stable ID, revision, latent distinction, scope, status, source/review state | Unit of allocation, split, and inference |
| `episode` | neighborhood, stable episode ID, revision, episode role, status | Model-visible learning interaction |
| `message` | episode revision, ordinal, semantic role, plain utterance, language/variety, hash | Delimiter-free conversation |
| `text_span` | message revision, character/byte boundaries, normalization map, span kind | Stable anchor for linguistic, semantic, source, and review claims |
| `phenomenon` | revisioned concept/phenomenon ID, label, definition, framework, parent/relations, provenance | Extensible vocabulary without baking one theory into columns |
| `linguistic_analysis` | message/span, analyst or model, framework/version, status, confidence, source | Preserve competing morphological, syntactic, semantic, pragmatic, discourse, and typological analyses |
| `linguistic_feature_assertion` | analysis, span, phenomenon, value/object, qualification, evidence | Query sentence types, constructions, meanings, functions, and uncertainty |
| `ontological_analysis` | episode/message/span, analyst or model, framework/version, competency question, status | Preserve competing identity, class, role, part, event, time, evidence, and granularity analyses |
| `ontological_assertion` | analysis, subject, relation, object/value, scope, time, perspective, evidence | Query ontological commitments without forcing one global ontology |
| `pedagogical_target` | family/episode/edge, capability, difficulty, prerequisite, intended contrast | Track what a unit is designed to teach or test |
| `transformation_type` | stable type, definition, required controls, version | Shared intervention vocabulary |
| `transformation_edge` | source episode, target episode, type, changed factor, held-constant contract | Executable relation among cases |
| `commitment` | neighborhood, proposition gloss, scope, status, source, time/perspective/granularity | Expected conceptual content |
| `episode_commitment` | episode, commitment, required/permitted/forbidden, qualification | Node-level answer contract |
| `expected_commitment_delta` | edge, commitment, preserve/add/retract/pluralize/attribute/etc., condition | Edge-level conservation/change contract |
| `admissible_analysis_set` | episode/edge, set identity, plurality kind, resolution conditions | Legitimate ambiguity structure |
| `admissible_analysis` | set, analysis, required/permitted, evidence, exclusion boundary | Individual defensible reading |
| `projection` | neighborhood, domain, phenomenon, vocabulary exclusion, source | One realization of the latent distinction |
| `cross_domain_pair` | source projection, held-out projection, expected shared commitments | Transfer contract |
| `invariance_constraint` | edge or pair, commitment, severity, rationale | Explicit unaffected content |
| `evaluation_intervention` | base episode, transformation sequence, split, sealed expected delta | Reproducible relational evaluation case |
| `relation_visibility_treatment` | arm, context boundary, packing, comparison text, objective signal | Distinguish invisible, optimizer-only, attention-visible, and explicit relations |
| `bridge_control` | source/target projections, true/false status, lexical similarity, reviewed dependency | True bridge, false bridge, and jargon-ablation control |
| `relation_corruption` | source edge, corruption type, permutation seed, resulting contract | Matched random-edge, permuted-delta, and invariant-corruption controls |
| `formal_oracle_case` | transformation sequence, assumptions, expected state/set, proof/check status | Checked-core oracle without forcing every case into formal logic |

### 19.4 Cross-cutting ledger tables

| Family | Tables | Purpose |
|---|---|---|
| Revisions | `unit_revision`, `revision_parent`, `content_hash` | Never overwrite a scientific object |
| Sources | `source_work`, `source_file`, `source_fragment`, `source_claim`, `license`, `authority_record`, `restriction` | Grounding, permission, and cultural authority |
| Generation | `model_version`, `prompt_template`, `generation_task`, `generation_attempt`, `candidate_origin` | Exact teacher lineage, including failures |
| Review | `reviewer`, `rubric`, `judgment`, `judgment_score`, `disagreement`, `adjudication` | Human/model decisions and conflict |
| Similarity | `signature`, `similarity_edge`, `duplicate_cluster`, `cluster_member` | Duplicate control and leakage |
| Splits/releases | `split`, `split_assignment`, `release`, `release_member`, `cohort`, `cohort_member` | Materialized scientific membership |
| Rendering | `renderer`, `rendering_profile`, `delimiter_definition`, `render_event`, `render_segment` | Reproducible model input without contaminating text |
| Tokenization summary | `tokenizer`, `tokenization`, `token_sequence_blob`, `tokenization_summary`, `supervision_span` | Exact token reconstruction and high-value audits |
| Derived language structures | `sentence`, `wordform`, `morpheme`, `constituent`, `dependency_arc`, `coreference_chain`, `discourse_relation`, `semantic_role` | Fully queryable linguistic descriptions with analysis/version provenance |
| Derived ontology structures | `entity_mention`, `entity_hypothesis`, `identity_relation`, `class_hypothesis`, `part_relation`, `event_hypothesis`, `temporal_relation`, `evidence_link` | Fully queryable, contradiction-preserving ontological descriptions |
| Exposure | `training_example`, `training_example_member`, `exposure_manifest`, `exposure_summary`, `checkpoint_exposure` | Know what every run and checkpoint could have learned from |
| Runs | `training_run`, `run_data_binding`, `checkpoint`, `evaluation_run`, `finding` | Bind results to code/data/evaluation hashes |
| Audit | `schema_version`, `decision_record`, `audit_event`, `release_artifact` | Evolution and integrity |

### 19.5 Deferred materialization, never deferred preservation

The following high-volume normalized rows may be materialized after the core ledger if their scale would
delay the pilot; their table contracts remain part of the complete design:

- one row for every token occurrence;
- every candidate parse node and arc from every parser/version;
- every derived ontological assertion from every analyzer/version;
- per-step unit/token exposure if the trainer can instead emit a compact verified exposure manifest;
- full-text publication mirrors and public review applications;
- generalized provenance projections whose immediate query contract is not yet stable.

For every deferred table family, the ledger still records the raw or compact source artifact, content
hash, schema/version, producer, relevant unit IDs, and deterministic derivation recipe. Deferral is not
rejection, omission, or permission to throw data away. The predecessor schema remains the design
reservoir, and later model runs may materialize different linguistic, ontological, sentence-type, or
token-level views from the same preserved evidence.

### 19.6 Pilot database invariants

1. Canonical utterances are immutable and delimiter-free.
2. Revisions never destroy prior text, commitments, analyses, judgments, or decisions.
3. Every accepted episode belongs to exactly one revisioned neighborhood.
4. Every transformation edge names changed and held-constant factors.
5. Every evaluation edge has an expected delta frozen before model output.
6. Every required commitment is traceable to source, construction, or explicit hypothetical status.
7. Competing analyses coexist; adjudication relates rather than deletes them.
8. Splits are assigned at neighborhood/source/template/semantic-family level and materialized.
9. A deterministic renderer plus tokenizer reconstructs exact model bytes/tokens from a sealed release.
10. Model-visible exports contain no audit metadata unless explicitly designed as natural-language
    content.
11. Unknown license or cultural authority fails closed.
12. Release, export, run binding, and evaluation counts/hashes reconcile.
13. A relation-visible case has a matched visibility/corruption control or a documented exclusion.
14. Required, permitted, and forbidden commitments remain distinguishable in storage and scoring.
15. Every deferred derived object has a hash-addressed preserved source and a reproducible derivation
    record in SQLite.
16. Rejected, failed, conflicting, and superseded objects remain queryable rather than being filtered out
    of the ledger.

### 19.7 Required database queries

The pilot schema fails review if it cannot answer, reproducibly:

- list every neighborhood with a temporal edge but no attribution or historical-truth constraint;
- find cross-domain pairs whose training and held-out projections share suspicious vocabulary;
- show every commitment expected to retract while an unrelated commitment must remain invariant;
- enumerate admissible-analysis sets lacking an excluded-analysis boundary;
- select all bridge neighborhoods held out in full and prove none of their source/template clusters are
  in training;
- trace one output score back to checkpoint, rendered episode bytes, edge contract, source, teacher, and
  human adjudication;
- render the same episodes for two chat templates without modifying messages;
- compare delimiter and supervised-content token overhead across renderers;
- list rejected counterexamples and the human/model disagreement that rejected them;
- materialize the semantic episode multiset for arms B–E and reconcile all added packing/comparison and
  supervision tokens rather than claiming false byte equality;
- list every true bridge with its false-bridge, same-word/different-relation, and jargon-ablation control;
- reconstruct each corrupted-relation treatment from its source edge and permutation seed;
- identify formal-oracle cases whose composed delta is inconsistent, order-sensitive, empty, or
  overgenerated.

## 20. Later relation-aware training intervention

Arm E introduces explicit relation information only after B–D have isolated targeted content,
optimizer-level co-location, and attention-visible juxtaposition. Natural-language comparison and an
auxiliary training signal are separate subarms because the former changes input content while the latter
changes the objective.

Candidate temporary prediction targets include:

- preserve;
- add;
- retract;
- pluralize/branch;
- attribute;
- temporalize;
- change granularity;
- request clarification.

Possible mechanisms:

1. a temporary relation-prediction head removed at inference;
2. a contrastive objective bringing paraphrases and cross-domain realizations of one distinction closer
   while separating minimally different conclusions;
3. preference pairs favoring local, attributed revision over global churn, flattening, or overhedging;
4. natural-language comparison episodes, reported separately because they change model-visible content.

The auxiliary intervention must not be introduced into every primary arm. If it alone succeeds, the
claim is limited to explicit delta supervision. It cannot be attributed to storage structure,
co-batching, packing, or relational existence in the research ledger.

## 21. Mechanistic opportunity at 57.7M

The small model is an experimental advantage if internal analyses are predeclared rather than used to
decorate a behavioral story after the fact.

Candidate question:

> Do the same internal directions, features, heads, or compact subspaces support a latent distinction
> across its linguistic and ontological projections?

Examples:

- “former student” and temporally dependent institutional roles;
- collective agreement and group/member ontology;
- aspectual boundaries and event identity;
- evidential language and source attribution;
- mass/count coercion and individuation in entity resolution.

Possible analyses after a behavioral effect exists:

- layerwise linear probes trained on one projection and tested on lexically isolated projections;
- representational similarity across paired projections with lexical controls;
- causal activation patching or directional intervention on a predeclared subset;
- comparison of specialists and integrated models;
- testing whether an intervention changes both projection behaviors in the predicted direction while
  leaving unrelated commitments stable.

A shared probe direction is not by itself a learned concept. The evidence ladder requires: (1)
behavioral transfer, (2) lexical, scenario, and false-bridge controls, (3) cross-projection probe
generalization, and (4) a causal intervention with predicted effects and unrelated-commitment stability.
The result must replicate across multiple families and seeds. A valid negative finding is that the model
learns isolated vocabulary-specific tasks without a shared abstraction.

## 22. Cultural and linguistic authority

Relational structure does not relax source ethics.

- Never fabricate examples attributed to a real language, variety, speaker, community, or theory.
- Distinguish public access, legal license, scholarly attribution, and community authority.
- Use attested examples or explicitly constructed examples reviewed by appropriate expertise.
- Keep language-specific evidence separate from a teacher's generalization.
- Do not treat English categories as the latent concept and other languages as exotic projections.
- Do not turn contested analyses into one canonical answer for scoring convenience.
- Do not infer that a typological database licenses unrestricted derivative generation from every cited
  grammar.
- Restricted, sacred, personally sensitive, or community-governed material is excluded without explicit
  authority.
- Signed-language phenomena cannot be reduced to written glosses without stating the modality loss.
- A cross-domain analogy that encodes stereotypes about social roles, dialects, occupations, disability,
  gender, race, or community identity must be rejected or reviewed by appropriate humans.

## 23. Scaling law for this program

The following is a decision ladder, not a promise to run.

| Stage | Scientific object | Admission evidence for next stage |
|---|---|---|
| Design | 24–60 hand-audited concept-family specifications | Reviewers agree projections, edges, deltas, and false bridges are coherent |
| Candidate micro-study | Approximately 30 complete families | Measured construction/revision cost and human agreement |
| P0 | Initiation-only controlled runs | Section 11 gate passes |
| Decisive micro-pilot | Approximately 30 deep families | B–E and corruption controls, three seeds where required, private final |
| Breadth expansion | Up to 300 neighborhoods | Correct relations beat independent and corrupted controls on transport/plurality |
| 20K episode scale | More neighborhoods, not paraphrase inflation | Transfer and revision-locality gains persist |
| 50K episode scale | Underrepresented transformations/projections | Scaling curve remains positive without conversation regression |
| 200K maximum | Only if unique neighborhoods and edge coverage justify it | Predeclared information/coverage need, not pipeline throughput |

Stop scaling when:

- additional episodes mostly duplicate existing commitment transitions;
- whole-neighborhood or cross-projection transfer plateaus;
- integrated-model interference worsens;
- teacher/human rejection rises materially;
- ordinary conversation regresses;
- the larger control learns but Alpha does not and a capacity conclusion is sufficiently supported;
- cost exceeds the approved ceiling.

## 24. Interpretation matrix

| Result | Supported conclusion | Unsupported conclusion |
|---|---|---|
| P0 passes; all P1 arms fail | Initiation confound removed; pilot method/evaluation/capacity remains problematic | “Conceptual neighborhoods cannot work” |
| A ≈ B ≈ C ≈ D ≈ E | No tested content or relation-visibility treatment helps at this scale | “Synthetic data never helps” |
| B > A; C/D ≈ B | Targeted content helps; co-batching and attention visibility add no measured benefit | “The model learned a relation” |
| C > B; D ≈ C | Gradient co-location or rehearsal changes learning, but attention visibility adds no gain | “The semantic edge was observed” |
| D > C and correct packing > random/false packing | Attention-visible correct comparison adds value | “Explicit supervision is unnecessary at all scales” |
| E > D | Explicit comparison or delta supervision adds value beyond packing | “The database structure caused transfer” |
| Correct relation ≈ corrupted relation | Gains arise from formatting, context, rehearsal, or generic contrast | “Relational semantics helped” |
| Transfer works; plurality collapses | A single-state transformation was learned; set-valued behavior was not | “Ambiguity preservation succeeded” |
| Specialists > integrated | Joint capacity or mixture interference | “Linguistics and ontology are unrelated” |
| Integrated > specialists on transfer | Joint training supports cross-domain abstraction | “The same internal feature caused both” |
| 150M–300M succeeds; 57.7M fails | Curriculum is learnable; Alpha scale/foundation likely limits joint target | “Parameter count alone is causal” |
| Diagnostic probes pass; free chat fails | Knowledge may be accessible under constraints but not conversationally expressed | “The product goal is met” |
| Free chat sounds good; deltas fail | Eloquence without conceptual conservation | “The model understands ontology” |

## 25. Research-agent work program

Each track returns a dated, source-linked memo. Agents may disagree. No track may start a run or edit
code under this document.

### J0 — Prior art and novelty

Search for relational curricula, graph-structured training examples, counterfactual/minimal-intervention
learning, metamorphic training/testing, solver-verified transformation families, ambiguity-aware
alignment, belief revision, cross-domain transfer, ontology verbalization, metalinguistic training,
small-model distillation, and provenance-bearing datasets. Maintain a NormWorlds-CF comparison chart and
narrow the novelty whenever closer work is found.

### J1 — P0 initiation design

Design the smallest controlled experiment separating data order, episode normalization, answer-start
treatment, length distribution, and checkpoint selection. Specify exact new frozen prompts and causal
interpretation.

### J2 — Neighborhood formal semantics

Formalize nodes, edges, commitment states, delta composition, contradictions, valid-time/record-time,
perspectives, and admissible-analysis sets. Specify the constrained formal-oracle subset, distinguish
invariance/equivariance/set-valued relations, and identify impossible or underspecified compositions.

### J3 — Linguistic neighborhoods

Propose sourced, cross-linguistically responsible neighborhoods with controlled transformations. Audit
theory assumptions and distinguish text-representable competence from phonetic/signed modalities.

### J4 — Ontology neighborhoods

Develop framework-neutral identity, role, part, event, time, evidence, granularity, and competency-
question cases. Avoid treating one upper ontology as truth.

### J5 — Cross-projection transfer

Design projection pairs with genuine shared structure, lexical-confound controls, held-out vocabulary,
false bridges, jargon ablations, held-out scenarios, and falsification criteria. Identify analogies too
weak to support a transfer claim.

### J6 — Generation and human review

Design teacher bake-off, prompts, counterexample challenge, review rubrics, human expertise allocation,
rejection ledger, and unique-information audit.

### J7 — Relational evaluation

Define executable commitment scoring, diagnostic versus free-generation channels, uncertainty, human
sample sizes, judge calibration, whole-neighborhood holdout, composition tests, and final gates.

### J8 — Training design and controls

Specify arms A–E, random-edge/permuted-delta/corrupted-invariant controls, specialist factorial,
midtraining/SFT comparison, seed policy, positive-control model, honest token accounting, checkpoint
selection, and compute-efficient successive halving.

### J9 — Alpha Ledger SQLite

Review the comprehensive schema, phased materialization plan, delimiter-independent rendering,
immutability, query set, scale, migrations, and hash-addressed preservation contract. Produce a data
dictionary and invariant test plan, not implementation.

### J10 — Mechanistic analysis

Predeclare probes and interventions capable of distinguishing shared abstraction from vocabulary-specific
features. Specify multiple-comparison and researcher-degree-of-freedom controls.

### J11 — Authority, licensing, and harm

Define inclusion, exclusion, community authority, source licensing, privacy, restricted material,
stereotype audit, and withdrawal rules for every projection family.

## 26. Research-return template

Every return includes:

1. track and date;
2. bottom-line recommendation;
3. claims with primary sources;
4. closest counterevidence;
5. concrete design or amendment;
6. causal variable isolated;
7. falsification criterion;
8. unresolved questions;
9. exact requested text changes;
10. search/reproducibility record.

Do not use paper titles or abstracts as proof of an experimental detail that requires reading the method
or results. Label preprints, workshop papers, thesis proposals, benchmarks, systems, and replicated
findings distinctly.

## 27. External-assessment adjudication

Two external assessments were supplied by the operator on 2026-07-30. Their citations were checked
against primary arXiv or ACL Anthology pages where accessible. Recommendations are preserved here as
decisions rather than silently absorbed.

| Recommendation | Disposition | Effect on program |
|---|---|---|
| Replace combination novelty with linked conceptual transformations | **Accepted in first draft; superseded** | NormWorlds-CF required the narrower transport claim |
| Make 200K a conditional ceiling | **Accepted** | Sections 1, 12, 23 |
| Repair initiation first and separately | **Accepted** | P0 in section 11 |
| Treat midtraining seriously | **Accepted** | Section 15 |
| Separate database and learning-paper claims | **Accepted** | P0/P1/P2 split; ledger remains comprehensive |
| Reduce dependence on model judges | **Accepted** | Human-dominant conceptual adjudication |
| Use 150M–300M positive control | **Accepted conditionally** | Required design; execution needs cost authority |
| Make neighborhood first-class in SQLite | **Accepted** | Section 19.3 |
| Add conservation/delta metrics | **Accepted** | Section 18 |
| Compare independent versus linked identical content | **Accepted and refined** | Arms B–E vary actual relation visibility |
| Add relation prediction head | **Deferred as explicit ablation** | Arm E / section 20, after B–D |
| Claim the combination has never been done | **Rejected** | Only bounded prior-art finding retained |
| Treat every recommendation as established truth | **Rejected** | Each remains falsifiable and source-scoped |
| Treat linked executable families and preserve/change records as novel | **Rejected after NormWorlds-CF collision** | Section 6 narrows the claim |
| Replace “concept topology” with typed semantic equivariance | **Accepted** | Sections 1, 4, 7 |
| Make cross-projection commitment-delta transport primary | **Accepted** | Sections 4, 7.7, 18 |
| Shrink the first pilot from 300 to 24–60 deep families | **Accepted** | Sections 1, 12, 23 |
| Treat co-batching as model-visible relation exposure | **Rejected** | Arm C isolates optimizer-level effects only |
| Add packed, explicit, and corrupted-relation treatments | **Accepted** | Arms D/E and section 14.2 |
| Add false bridges and jargon ablation | **Accepted** | Sections 13.5 and 18 |
| Add a limited formal semantic calculus | **Accepted as a design track, not code authority** | Section 8.1 |
| Require behavioral plus causal internal transfer | **Accepted conditionally after behavior exists** | Section 21 |
| Lead the first paper with SQLite or synthetic generation | **Rejected** | P2 remains supporting infrastructure |

## 28. Open decisions before any generation

1. What formal semantics best handles delta composition without pretending natural-language commitments
   are fully logical propositions?
2. How many transformations per neighborhood are needed to infer a boundary rather than memorize a
   chain?
3. Can attention-visible packing expose the relation without explicit comparison language, and how can
   its context/format effects be matched?
4. Which concepts support genuine linguistic–ontology projections, and which are only analogies?
5. What lexical and scenario controls are sufficient for a cross-projection claim?
6. How should set-valued answers be scored when humans disagree on admissibility?
7. What P0 intervention is minimally sufficient and least entangled with P1?
8. Is the one-billion-token base foundation adequate for any P1 arm?
9. What architecture/tokenizer should the 150M–300M control use to remain interpretable?
10. How should midtraining and SFT receive equal-token comparison when their supervision masks differ?
11. Which SQLite fields are indispensable for the pilot and which are infrastructure enthusiasm?
12. What human expertise and community authority are actually available?
13. What is the cost ceiling for three seeds, five arms plus corruptions, specialists, phase comparison,
    and positive control—and which comparisons use successive halving?
14. What negative result is sufficient to close Alpha Joints rather than scale it?

## 29. Definition of research readiness

No candidate generation begins until the repository has:

- reconciled J0–J11 returns with preserved disagreement;
- a prior-art claim chart;
- a frozen neighborhood/delta formalism;
- 30 hand-written, independently audited neighborhood specifications;
- a source/license/authority plan;
- a human-review staffing and sample plan;
- a frozen split and relational-evaluation protocol;
- a P0 experiment contract and newly frozen initiation suite;
- an arms A–E design with explicit context visibility, honest token accounting, and matched corruption
  controls;
- specialist and positive-control interpretation rules;
- a reviewed comprehensive SQLite data dictionary, phased-materialization plan, and invariant/query test
  plan;
- a generation micro-study contract with budget and stop conditions;
- explicit user authorization for that next stage.

## 30. Definition of pilot success

The pilot succeeds scientifically only if:

1. P0 passes independently.
2. Data, code, runs, and evaluation contracts reconcile by hash.
3. Ordinary conversation does not materially regress.
4. At least one relation-visible treatment beats independent targeted units across seeds on the primary
   cross-projection commitment-delta endpoint.
5. Correct relations beat random-edge, permuted-delta, corrupted-invariant, and false-bridge controls.
6. The gain holds on lexically isolated whole-family and composed-transformation holdouts.
7. The integrated model transfers beyond technical-word matching and survives jargon ablation.
8. Revision locality improves without increasing overhedging or collapsing legitimate analysis sets.
9. Human adjudication confirms the automated direction.
10. The result distinguishes content, optimizer co-location, attention visibility, explicit supervision,
    corruption, and capacity explanations.
11. Any mechanistic claim includes behavioral transfer, cross-projection probing, causal intervention,
    unrelated-commitment stability, and replication.
12. Failures and null results remain public in the evidence bundle.

A model that merely becomes chattier has passed P0, not Alpha Joints. A model that says “role,”
“provenance,” or “ambiguity” more often has not demonstrated the core claim.

## 31. Initial primary-source bibliography

- Zhang, [NormWorlds-CF: Solver-Verified Counterfactual Normative Reasoning with Metamorphic-Relation GRPO](https://arxiv.org/abs/2607.03957).
- Gebreegziabher et al., [Leveraging Variation Theory in Counterfactual Data Augmentation](https://aclanthology.org/2025.findings-acl.50/).
- Ribeiro et al., [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/).
- Gardner et al., [Evaluating Models' Local Decision Boundaries via Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/).
- Zheng et al., [Bidirectional Empowerment of Metamorphic Testing and Large Language Models](https://arxiv.org/abs/2605.13898).
- Kaushik et al., [Learning the Difference that Makes a Difference with Counterfactually-Augmented Data](https://arxiv.org/abs/1909.12434).
- Huang, Liu, and Bowman, [Counterfactually-Augmented SNLI Training Data Does Not Yield Better Generalization Than Unaugmented Data](https://arxiv.org/abs/2010.04762).
- Liu et al., [We're Afraid Language Models Aren't Modeling Ambiguity](https://arxiv.org/abs/2304.14399).
- Timiryasov and Tastet, [Baby Llama](https://arxiv.org/abs/2308.02019).
- Kamzela, Lango, and Dušek, [Multi-task pretraining with LLM-designed study plans](https://aclanthology.org/2025.babylm-main.33/).
- Yamaguchi, Mi, and Aletras, [Enhancing Linguistic Competence through Language Learning Tasks](https://aclanthology.org/2026.acl-short.27.pdf).
- Salhan et al., [ContingentChat](https://aclanthology.org/2025.babylm-main.25/).
- Padovani et al., [Dialogue Is Not Enough to Make a Communicative BabyLM](https://aclanthology.org/2025.babylm-main.29/).
- Schneider and Anastasopoulos, [Speaking of Language](https://aclanthology.org/2026.bigpicture-main.9/).
- Yang et al., [LingGym](https://aclanthology.org/2025.emnlp-main.69/).
- Arčon et al., [Evaluating Metalinguistic Knowledge across the World's Languages](https://arxiv.org/html/2602.02182v2).
- Lippolis et al., [Ontology Generation using Large Language Models](https://arxiv.org/html/2503.05388v1).
- Gómez Álvarez, Rudolph, and Strass, [Standpoint EL](https://arxiv.org/abs/2302.13187).
- Miranda and Nalepa, [Polyvocal ontology-based perspective-aware extraction proposal](https://aclanthology.org/2026.eacl-srw.46/).
- Drucker and Mahowald, [The Counterexample Game](https://arxiv.org/abs/2605.03936).
- Yang et al., [Belief-R](https://aclanthology.org/2024.emnlp-main.586/).
- Dhanda, [DeltaLogic: Minimal Premise Edits Reveal Belief-Revision Failures](https://arxiv.org/abs/2604.02733).
- Cohen et al., [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://arxiv.org/abs/2307.12976).
- Webb et al., [A Theory of Relation Learning and Cross-domain Generalization](https://arxiv.org/abs/1910.05065).
- Xue, [OriginBlame](https://arxiv.org/html/2607.13037v1).
- Lazem and Teahan, [Generated clinical-corpus redundancy measurement](https://arxiv.org/abs/2606.29605).

## 32. Working conclusion

Alpha Joints does not seek a database full of beautiful answers. It asks whether a small conversational
model can transport a typed semantic change across domains while preserving the commitments that should
remain fixed and the interpretations that should legitimately remain plural.

The key experimental contrast is:

> identical targeted content with relations invisible, optimizer-co-located, attention-visible, or
> explicitly supervised—plus equally regular but semantically corrupted controls.

The key transfer test is:

> learn a boundary in one projection and preserve it in another without relying on the same vocabulary.

The key revision test is:

> change exactly what new evidence, time, perspective, or competency requirements license—and preserve
> everything else.

A roughly 30-family pilot that demonstrates those behaviors would be more consequential than 200,000
audited conversations that only make Alpha sound informed. If the pilot fails cleanly, the result still
locates whether the limit is initiation, targeted content, gradient co-location, attention visibility,
explicit supervision, foundation, capacity, or the underlying transport hypothesis.
