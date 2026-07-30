# Alpha Joints: executable conceptual neighborhoods for a small conversational model

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

> Can a 57.7M-parameter conversational language model learn the topology of conceptual commitments—what
> remains invariant, what changes under a controlled intervention, and what may legitimately remain
> plural—when examples are organized as linked natural-language transformations rather than independent
> instruction rows?

The decisive evidence is **cross-domain abstraction transfer**. A model receives no credit merely for
using the correct technical term. It must apply a distinction learned through one realization to an
unseen realization in another domain: for example, from temporary institutional roles to the semantics
of “former student,” or from evidential language to provenance-preserving database advice.

The first pilot is approximately **300 conceptual neighborhoods and 4,800–7,200 accepted episodes**, not
200,000 rows. Scaling to 20K, 50K, or 200K episodes is forbidden until linked neighborhoods beat
equal-token independent targeted examples on unseen-neighborhood and cross-projection tests.

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
| **P1 — Alpha Joints** | Core learning-science experiment | Does linked relational training produce transferable commitment structure? | Yes, if controls pass |
| **P2 — Alpha Ledger** | Minimal scientific data substrate | Can every decisive object and result be reconstructed and audited? | Separate systems contribution only after independent evaluation |

P0 must pass before P1 training. P2 must support P1, but P1 must not wait for every ambitious provenance
feature imagined in the predecessor brief. A systems paper and a learning paper may eventually emerge;
their claims, ablations, and acceptance criteria remain separate.

## 4. Replacement novelty statement

> **Alpha Joints investigates whether a 57.7M-parameter conversational language model can acquire
> reusable conceptual boundaries from an executable curriculum of linked natural-language
> transformations. Each conceptual neighborhood specifies which commitments should remain invariant
> under paraphrase, which should change under a minimal contextual or evidential intervention, and which
> may legitimately remain plural across interpretations, perspectives, granularities, and times. We
> test whether these distinctions transfer between paired linguistic and ontological realizations,
> whether belief revision remains localized to evidence-dependent commitments, and whether disagreement
> can be preserved without collapsing into either contradiction failure or indiscriminate hedging. The
> contribution is not synthetic-data scale, but a relational data representation, training
> intervention, and behavioral evaluation for learning the topology of concepts.**

This is a proposed claim, not a finding. It survives only if linked organization adds value beyond:

- the response-initiation repair;
- an equal-token generic conversation control;
- the same targeted episodes treated as independent rows;
- technical-vocabulary memorization;
- a larger student's capacity advantage;
- evaluator or teacher-family bias.

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

The novelty is not the parameter count, synthetic teachers, linguistic tasks, ontology competency
questions, dialogue alignment, counterexample repair, plural perspectives, belief revision, or granular
provenance individually. Each has close prior work.

| Work | Verified overlap | Gap relevant to Alpha Joints |
|---|---|---|
| [Baby Llama](https://arxiv.org/abs/2308.02019) | Distilled an ensemble into a 58M-parameter LLaMA trained from a 10M-word BabyLM corpus | Not open-ended conceptual dialogue or cross-domain linguistic–ontology transfer |
| [LLM-designed study plans](https://aclanthology.org/2025.babylm-main.33/) | Teacher automatically designed 56 tasks and generated a multitask pretraining corpus competitive with same-size human text | Independent tasks, not commitment-delta neighborhoods or plural analyses |
| [L2T](https://aclanthology.org/2026.acl-short.27.pdf) | Mixed 14 Language Learning Tasks with ordinary pretraining; improved and accelerated linguistic competence in 500M/1B models | Not tiny conversational students, ontology, or cross-projection transfer |
| [ContingentChat](https://aclanthology.org/2025.babylm-main.25/) | Targeted teacher–student post-training improved grammaticality and cohesion in a 100M-word BabyLM | Dialogue contingency, not reusable conceptual topology |
| [Llamalogue](https://aclanthology.org/2025.babylm-main.29/) | Dialogue-only pretraining improved dialogue continuation while underperforming on most standard BabyLM benchmarks | Warns that specialization can interfere; no language–ontology bridge |
| [Speaking of Language](https://aclanthology.org/2026.bigpicture-main.9/) | Identifies natural and symbolic metalanguage as an understudied NLP research area | Research agenda, not the proposed tiny-student relational curriculum |
| [LingGym](https://aclanthology.org/2025.emnlp-main.69/) | Tests metalinguistic inference from IGT and grammatical descriptions in 18 typologically diverse grammars | Evaluation rather than a conversational training object or ontology transfer |
| [WALS metalinguistic evaluation](https://arxiv.org/html/2602.02182v2) | Converts WALS features across 2,660 languages into explicit questions and finds fragmented, resource-linked performance | Factual feature QA, not transformations among commitments |
| [Ontology Generation using LLMs](https://arxiv.org/html/2503.05388v1) | Generates OWL drafts from user stories and competency questions; evaluates across ten ontologies | Ontology authoring, not natural-language conceptual behavior in a tiny student |
| [Standpoint EL](https://arxiv.org/abs/2302.13187) | Formally represents diverse, possibly conflicting standpoints while retaining tractable reasoning | External logical representation rather than learned conversational behavior |
| [Polyvocal ontology proposal](https://aclanthology.org/2026.eacl-srw.46/) | Proposes perspective-aware extraction into epistemically separate, provenance-bearing knowledge graphs | A research proposal for extraction/KGs, not student internalization |
| [The Counterexample Game](https://arxiv.org/abs/2605.03936) | Iterates definition, counterexample, and repair; finds diminishing returns and a permissive LM judge | One repair chain rather than typed neighborhoods and cross-domain transfer |
| [Belief-R](https://aclanthology.org/2024.emnlp-main.586/) | Tests belief revision after new evidence and exposes a trade-off between updating and preserving unaffected beliefs | Not linguistic ambiguity, attribution, temporality, and ontology jointly |
| [OriginBlame](https://arxiv.org/html/2607.13037v1) | Propagates record/token-level provenance and resolves exact forget sets | Provenance engineering without rich conceptual semantics |
| [Generated-corpus redundancy study](https://arxiv.org/abs/2606.29605) | Shows raw synthetic token volume can greatly overstate unique information in a clinical corpus | Different domain, but directly warns against row/token count as the research variable |

The current search did **not** find a direct precedent combining all of these:

1. a roughly 58M conversational student;
2. linked natural-language interventions around one conceptual boundary;
3. explicit preserve/add/retract/plural/attribute/temporalize contracts;
4. paired linguistic and ontological projections;
5. whole-neighborhood, composed-transformation, and cross-projection holdouts;
6. evaluation of revision locality and legitimate plurality;
7. delimiter-independent, provenance-bearing data representation.

This is a bounded search finding, not proof of absence. The prior-art track remains open through paper
submission.

## 7. The canonical scientific object

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

The first scientific pilot contains approximately 300 accepted conceptual neighborhoods:

| Family | Neighborhoods | Purpose |
|---|---:|---|
| Primarily linguistic | 100 | Meaning/form distinctions with ontology/evidence projections where appropriate |
| Primarily ontological | 100 | Identity, roles, parts, events, time, evidence, perspective, competency questions |
| Explicit language–ontology bridge | 100 | Cross-domain realizations designed for projection holdout |
| **Total** | **300** | |

Each neighborhood should contain approximately eight structural episode roles and two or three surface
realizations where diversity adds information. Expected accepted volume is 4,800–7,200 episodes.

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

Not every neighborhood needs every transformation, but every accepted neighborhood must include:

1. a base episode;
2. a surface-preserving paraphrase;
3. a minimal meaning-changing intervention;
4. an irrelevant-detail negative control;
5. either an evidence, temporal, perspective, or competency-question intervention;
6. a counterexample or boundary case;
7. an admissible-analysis or clarification case where the concept supports one;
8. a projection into another domain or a documented reason why transfer is not meaningful.

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

### 13.5 Development and final evaluation

- **Training:** model-visible episodes only.
- **Development:** public structure and metrics, distinct neighborhoods; usable for design selection.
- **Private final:** sealed neighborhoods, interventions, and answer contracts inaccessible to generation,
  training, and checkpoint-tuning agents.
- **Human audit:** blinded samples from all splits, with the private final adjudicated only after model
  and checkpoint choices are locked.

## 14. Decisive equal-token experimental arms

All primary arms use the same base checkpoint, tokenizer, architecture, P0 initiation method, optimizer
budget, supervised-token budget, prompt-length envelope, checkpoint cadence, and frozen evaluations.

| Arm | Data/training treatment | Question isolated |
|---|---|---|
| **A — generic control** | Shuffled ordinary conversations, matched tokens | How much comes from P0 repair and token budget? |
| **B — independent targeted units** | Same targeted episodes, fully shuffled as independent rows | Is topical curation alone sufficient? |
| **C — linked schedule** | Same episode bytes and tokens, but neighborhood/edge-aware batching and rehearsal; standard objective | Does relational organization/scheduling help without a new loss? |
| **D — linked plus delta signal** | Arm C plus a relation-aware auxiliary or preference signal | Does explicitly learning transformations add value? |

### 14.1 Why B versus C matters

B and C contain exactly the same model-visible episodes and token multiset. Their difference is whether
training preserves neighborhood relations in batching, ordering, contrast exposure, and rehearsal. This
is the cleanest test of relational organization without changing content or architecture.

The study must acknowledge a real possibility: with a standard causal objective and no state across
examples, schedule alone may be too weak to expose an edge. A null B-versus-C result would motivate D;
it would not prove the neighborhood representation scientifically useless.

### 14.2 D is not allowed to rescue a failed C invisibly

D changes the objective. It must be reported as a distinct intervention. If only D works, the supported
claim is that relational supervision helps—not that relational data organization alone helps.

### 14.3 Specialist factorial

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

### 14.4 Seeds and selection

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

## 19. P2 — minimum viable SQLite substrate

### 19.1 Scope rule

The pilot database exists to execute and audit the Alpha Joints experiment. It is not authorized to
become a general scientific-data platform before P1 has evidence.

Implement a table only if it supports at least one named requirement:

- reconstruct model-visible text;
- represent a neighborhood, transformation, commitment, analysis, or projection;
- enforce a split/leakage rule;
- trace source/license/generation/review lineage;
- reproduce a release/export/run;
- calculate a predeclared metric;
- honor a restriction or withdrawal.

Full token-occurrence lineage, extensive model-run exposure warehouses, generalized annotation graphs,
and publication infrastructure remain designed in the predecessor brief but deferred until a named
scientific query requires them.

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

### 19.4 Minimal support tables

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
| Runs | `training_run`, `run_data_binding`, `checkpoint`, `evaluation_run`, `finding` | Bind results to code/data/evaluation hashes |
| Audit | `schema_version`, `decision_record`, `audit_event`, `release_artifact` | Evolution and integrity |

### 19.5 Deferred tables

Defer unless an approved query justifies the cost:

- one row for every token occurrence;
- generalized concept/ontology stores beyond pilot requirements;
- exhaustive syntactic parses for episodes whose objective does not depend on a parse;
- per-step unit/token exposure if the trainer can instead emit a compact verified exposure manifest;
- full-text publication mirrors and public review applications;
- provenance features whose sole justification is hypothetical future reuse.

Deferral is not rejection. The predecessor schema remains the design reservoir.

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
- materialize the exact episode multiset for arms B and C and prove the content/token multiset matches.

## 20. Later relation-aware training intervention

Only after arms B and C establish the data/schedule effect should the project introduce an auxiliary
relation signal.

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

The auxiliary intervention must not be introduced into every primary arm. Otherwise a success cannot be
attributed to the relational dataset rather than the objective.

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

- layerwise linear probes trained on training projections and tested on held-out projections;
- representational similarity across paired projections with lexical controls;
- causal activation patching or directional intervention on a predeclared subset;
- comparison of specialists and integrated models;
- testing whether an intervention changes both projection behaviors or only one vocabulary family.

A shared probe direction is not by itself a learned concept. Strong evidence requires transfer, causal
intervention, lexical/scenario controls, and replication across neighborhoods. A valid negative finding
is that the model learns isolated vocabulary-specific tasks without a shared abstraction.

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
| Design | 30 hand-audited neighborhood specifications | Reviewers agree edges and deltas are coherent |
| Candidate micro-study | 30 complete neighborhoods | Measured generation/revision cost and human agreement |
| P0 | Initiation-only controlled runs | Section 11 gate passes |
| Pilot | 300 neighborhoods / 4.8K–7.2K episodes | B/C/D comparison, three seeds where required, private final |
| 20K episode scale | More neighborhoods, not paraphrase inflation | Transfer and revision-locality gains persist |
| 50K episode scale | Underrepresented transformations/projections | Scaling curve remains positive without conversation regression |
| 200K maximum | Only if unique neighborhoods and edge coverage justify it | Predeclared information/coverage need, not pipeline throughput |

Stop scaling when:

- additional episodes mostly duplicate existing commitment topology;
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
| A ≈ B ≈ C | Targeted content and organization show no benefit at tested scale | “Synthetic data never helps” |
| B > A; C ≈ B | Targeted content helps; linked scheduling adds no measured benefit | “Model learned graph topology” |
| C > B | Relational scheduling/organization helps with identical content | “Auxiliary relation objectives are unnecessary at all scales” |
| D > C ≈ B | Explicit delta supervision helps; organization alone did not | “The data representation alone caused transfer” |
| Specialists > integrated | Joint capacity or mixture interference | “Linguistics and ontology are unrelated” |
| Integrated > specialists on transfer | Joint training supports cross-domain abstraction | “The same internal feature caused both” |
| 200M succeeds; 58M fails | Curriculum is learnable; Alpha scale/foundation likely limits joint target | “Parameter count alone is causal” |
| Diagnostic probes pass; free chat fails | Knowledge may be accessible under constraints but not conversationally expressed | “The product goal is met” |
| Free chat sounds good; deltas fail | Eloquence without conceptual conservation | “The model understands ontology” |

## 25. Research-agent work program

Each track returns a dated, source-linked memo. Agents may disagree. No track may start a run or edit
code under this document.

### J0 — Prior art and novelty

Search for relational curricula, graph-structured training examples, counterfactual/minimal-intervention
learning, belief revision, conceptual spaces, cross-domain transfer, ontology verbalization, metalinguistic
training, small-model distillation, and provenance-bearing datasets. Maintain a claim chart and narrow
the novelty whenever closer work is found.

### J1 — P0 initiation design

Design the smallest controlled experiment separating data order, episode normalization, answer-start
treatment, length distribution, and checkpoint selection. Specify exact new frozen prompts and causal
interpretation.

### J2 — Neighborhood formal semantics

Formalize nodes, edges, commitment states, delta composition, contradictions, valid-time/record-time,
perspectives, and admissible-analysis sets. Identify impossible or underspecified compositions.

### J3 — Linguistic neighborhoods

Propose sourced, cross-linguistically responsible neighborhoods with controlled transformations. Audit
theory assumptions and distinguish text-representable competence from phonetic/signed modalities.

### J4 — Ontology neighborhoods

Develop framework-neutral identity, role, part, event, time, evidence, granularity, and competency-
question cases. Avoid treating one upper ontology as truth.

### J5 — Cross-projection transfer

Design projection pairs with genuine shared structure, lexical-confound controls, held-out vocabulary,
and falsification criteria. Identify analogies too weak to support a transfer claim.

### J6 — Generation and human review

Design teacher bake-off, prompts, counterexample challenge, review rubrics, human expertise allocation,
rejection ledger, and unique-information audit.

### J7 — Relational evaluation

Define executable commitment scoring, diagnostic versus free-generation channels, uncertainty, human
sample sizes, judge calibration, whole-neighborhood holdout, composition tests, and final gates.

### J8 — Training design and controls

Specify arms A–D, specialist factorial, midtraining/SFT comparison, seed policy, positive-control model,
token equality, checkpoint selection, and compute-efficient successive halving.

### J9 — Alpha Ledger SQLite

Review the minimal schema, delimiter-independent rendering, immutability, query set, scale, migrations,
and what must remain deferred. Produce a data dictionary and invariant test plan, not implementation.

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

The assessment that triggered this successor brief was supplied by the operator on 2026-07-30. Its
citations were checked against primary arXiv or ACL Anthology pages where accessible. It is preserved
here as decisions rather than silently absorbed.

| Recommendation | Disposition | Effect on program |
|---|---|---|
| Replace combination novelty with linked conceptual transformations | **Accepted** | Sections 4, 7–10 |
| Make 200K a conditional ceiling | **Accepted** | Sections 1, 12, 23 |
| Repair initiation first and separately | **Accepted** | P0 in section 11 |
| Treat midtraining seriously | **Accepted** | Section 15 |
| Separate database and learning papers | **Accepted** | P0/P1/P2 split and scoped SQLite |
| Reduce dependence on model judges | **Accepted** | Human-dominant conceptual adjudication |
| Use 150M–300M positive control | **Accepted conditionally** | Required design; execution needs cost authority |
| Make neighborhood first-class in SQLite | **Accepted** | Section 19.3 |
| Add conservation/delta metrics | **Accepted** | Section 18 |
| Compare independent versus linked identical content | **Accepted** | Arms B/C |
| Add relation prediction head | **Deferred as explicit ablation** | Arm D / section 20, after data-only comparison |
| Claim the combination has never been done | **Rejected** | Only bounded prior-art finding retained |
| Treat every recommendation as established truth | **Rejected** | Each remains falsifiable and source-scoped |

## 28. Open decisions before any generation

1. What formal semantics best handles delta composition without pretending natural-language commitments
   are fully logical propositions?
2. How many transformations per neighborhood are needed to infer a boundary rather than memorize a
   chain?
3. Can arm C expose relational structure through schedule alone, or does it require model-visible pairing?
4. Which concepts support genuine linguistic–ontology projections, and which are only analogies?
5. What lexical and scenario controls are sufficient for a cross-projection claim?
6. How should set-valued answers be scored when humans disagree on admissibility?
7. What P0 intervention is minimally sufficient and least entangled with P1?
8. Is the one-billion-token base foundation adequate for any P1 arm?
9. What architecture/tokenizer should the 150M–300M control use to remain interpretable?
10. How should midtraining and SFT receive equal-token comparison when their supervision masks differ?
11. Which SQLite fields are indispensable for the pilot and which are infrastructure enthusiasm?
12. What human expertise and community authority are actually available?
13. What is the cost ceiling for three seeds, four arms, specialists, phase comparison, and positive
    control—and which comparisons use successive halving?
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
- an arms A–D design with equal-token accounting;
- specialist and positive-control interpretation rules;
- a reviewed minimal SQLite data dictionary and invariant/query test plan;
- a generation micro-study contract with budget and stop conditions;
- explicit user authorization for that next stage.

## 30. Definition of pilot success

The pilot succeeds scientifically only if:

1. P0 passes independently.
2. Data, code, runs, and evaluation contracts reconcile by hash.
3. Ordinary conversation does not materially regress.
4. At least one linked treatment beats independent targeted units on a predeclared relational metric
   across seeds.
5. The gain holds on whole-neighborhood or composed-transformation holdout, not only variants of seen
   cases.
6. The integrated model shows cross-projection transfer beyond technical-word matching.
7. Revision locality improves without increasing overhedging.
8. Human adjudication confirms the automated direction.
9. The result can distinguish content, schedule, objective, and capacity explanations.
10. Failures and null results remain public in the evidence bundle.

A model that merely becomes chattier has passed P0, not Alpha Joints. A model that says “role,”
“provenance,” or “ambiguity” more often has not demonstrated the core claim.

## 31. Initial primary-source bibliography

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
- Xue, [OriginBlame](https://arxiv.org/html/2607.13037v1).
- Lazem and Teahan, [Generated clinical-corpus redundancy measurement](https://arxiv.org/abs/2606.29605).

## 32. Working conclusion

Alpha Joints does not seek a database full of beautiful answers. It seeks a controlled world of
neighboring cases whose differences specify the shape of a concept.

The key experimental contrast is:

> identical targeted content as independent episodes versus the same content organized through explicit
> conceptual neighborhoods, controlled transformations, and commitment deltas.

The key transfer test is:

> learn a boundary in one projection and preserve it in another without relying on the same vocabulary.

The key revision test is:

> change exactly what new evidence, time, perspective, or competency requirements license—and preserve
> everything else.

A 5,000–10,000-episode pilot that demonstrates those behaviors would be more consequential than 200,000
audited conversations that only make Alpha sound informed. If the pilot fails cleanly, the result still
locates whether the limit is initiation, data organization, objective, foundation, capacity, or the
underlying hypothesis.

