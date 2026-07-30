# Donto × Alpha: predicate birth, semantic deconstruction, and small models that know how knowledge works

> **Product-scope correction (2026-07-30):** Predicate Birth and Survival is a possible Donto
> application, not Alpha's primary identity. Alpha's governing goal is a small, natural, chatty
> interlocutor with deep practical understanding of language, ontology, and philosophy. See
> [Alpha's chatty research-model north star](ALPHA-CHATTY-RESEARCH-MODEL-NORTH-STAR.md). Donto integration
> and research novelty remain subordinate to that conversational goal.

**Status:** research agenda for criticism and revision; not an implementation plan

**Date:** 2026-07-30

**Authoring posture:** deliberately broad in possibilities, narrow in claims

**Current authorization boundary:** documentation and research only; no data generation, training,
RunPod provisioning, or live Donto writes are authorized by this document

**Related experimental brief:** [Alpha Joints research program](ALPHA-JOINTS-RESEARCH-PROGRAM.md)

---

## 1. Executive judgment

The user's instinct is coherent. Donto and Alpha are not unrelated projects that happen to mention
language. They meet at one unusually interesting question:

> **How much intelligence about the structure, limits, and transformation of knowledge can a very small
> model learn when factual memory is deliberately made secondary?**

Donto's premise is that generating candidate claims and relations is now cheap, while holding them
without premature collapse, anchoring them to evidence, distinguishing incompatible analyses, and
deciding what to ask next are the scarce operations. Alpha's small size makes it a possible scientific
instrument for asking which of those operations can be learned independently of encyclopedic recall.

The present Alpha Joints direction is valuable, but I would change its role. **Typed semantic
transformations should be the experimental grammar, not necessarily the headline capability.** The most
Donto-native candidate capability is:

> **Given a source passage and carefully chosen neighboring cases, can a sub-100M model discover a
> reusable relation worth naming, explain what that relation commits us to and what it does not, reject
> tempting false equivalences, and transport the distinction into a lexically different domain?**

I call this proposed program **Predicate Birth and Survival**. “Birth” is the free minting of a relation;
“survival” is whether the relation remains faithful, discriminating, reusable, alignable, and useful when
the substrate later asks real questions. This is not the same as producing ever more predicate strings.

The strongest first scientific object would be a **Predicate Birth Neighborhood**: a small family of
source-grounded examples, counterexamples, paraphrases, perspective changes, temporal changes, false
analogies, and cross-domain projections that collectively specify when a proposed predicate applies.
Alpha Joints already supplies much of the control structure needed to build and evaluate these
neighborhoods.

This recommendation is provisional. A bounded primary-source search found close and sometimes direct
prior art in Open Information Extraction, open-world relation discovery, dynamic schema induction,
ontology learning, question-driven schema discovery, and LLM-assisted predicate invention. In
particular, **ADVENT** now uses LLMs to invent named auxiliary predicates and Prolog to verify them, and
**AutoSchemaKG** induces schemas and relations from tens of millions of documents. Predicate invention,
dynamic schema induction, and “schema-free KG construction” therefore cannot be Alpha's broad novelty
claims. The surviving opening is the combination of:

- a sub-100M conversational student;
- source-bound, open-ended predicate birth from natural language rather than structured ILP examples;
- explicit positive, negative, ambiguous, perspectival, temporal, and non-equivalent cases;
- transport of the same semantic distinction across lexically isolated domains;
- evaluation by downstream query utility and safe non-collapse, not label match alone;
- a contradiction-preserving substrate in which rejected, competing, and superseded predicates remain
  inspectable;
- factual retrieval kept external so the learned object is an epistemic operation rather than a store of
  world trivia.

That exact conjunction appears promising, but it is a **research hypothesis, not a certified novelty
claim**.

---

## 2. Why Donto makes this question natural

### 2.1 Donto is not trying to finish an ontology

The canonical Donto program is an evidence-first operating system for contested reality. Its durable
object is not a single clean graph but a trail showing:

- who or what asserted a claim;
- what source fragment supports it;
- when it was asserted and when it was supposed to be true;
- which incompatible claims remain open;
- under which identity and alignment hypotheses a query was answered;
- how the claim's standing changes as evidence arrives;
- what observation or search would best reduce uncertainty next.

Its core loops are **HOLD**, **JUDGE**, and **STEER**:

1. **HOLD:** emit, anchor, and preserve claims—including incompatible ones.
2. **JUDGE:** align predicates, test identities, register arguments, review, and re-rank.
3. **STEER:** decide what evidence or action would be most discriminating next.

This differs from a conventional ontology project. Donto does not want a model to force every source
into one canonical vocabulary at ingestion time. It wants the source's distinctions preserved first and
their relationships adjudicated later, under a query-specific lens.

### 2.2 Free predicates are a feature, but unexamined predicates are not yet knowledge

The abundance canon says: emit freely now; defer typing, alignment, joining, and identity resolution to
query time. That is the correct governing direction. It prevents a fixed schema from deleting distinctions
before anyone knows which distinctions will matter.

But free minting creates a second problem. A minted label by itself does not tell us:

- whether the relation is actually supported by the source;
- whether its arguments are in the right direction;
- whether it is a relation, a role, an event, a property, or a disguised sentence;
- which nearby predicates are exact, broader, narrower, inverse, decomposed, merely close, or
  incompatible;
- whether the relation survives paraphrase;
- whether it incorrectly merges part, member, material, portion, and constitution;
- whether it is specific enough to be useful but general enough to recur;
- which questions it helps answer;
- which evidence would falsify or refine it.

The future research target should therefore not be “mint more.” It should be **mint with an inspectable
birth record, then let relations survive or fail under use**.

### 2.3 The live substrate exposes the actual bottleneck

Read-only inspection on 2026-07-30 found the following current shape. The largest row counts below are
PostgreSQL planner estimates and should be treated as approximate; the bounded semantic tables were
counted exactly.

| Live object | Current scale | Research implication |
|---|---:|---|
| Statements | ~43.21 million | Donto already has claim abundance |
| Evidence links | ~4.44 million | Anchoring exists at meaningful scale but remains far from universal |
| Predicate symbols | 1,157,848 | Relation naming is already radically open |
| Implicit predicates | 1,156,079 | Most minted symbols have not been promoted into a curated vocabulary |
| Active predicates | 1,768 | Promotion is extremely sparse relative to birth |
| Predicate descriptors | 95 | Definitions, signatures, and examples are the conspicuous missing layer |
| Accepted and candidate close matches | 96,353 | Alignment exists, but most volume is `close_match` rather than strong equivalence |
| Accepted exact equivalences | 2,064 | Safe collapse is rare—which is healthy, but needs better judgment |
| Accepted non-equivalences | 12 | Explicitly recording tempting but false merges is almost untouched |
| Arguments | 3,085 | Reasons for and against claims are tiny relative to the claim store |
| Identity edges / hypotheses | 169 / 51 | Identity-as-hypothesis is implemented but sparse |
| Claim frames / frame roles | 31 / 68 | Rich n-ary event and measurement structure is nearly empty |

This suggests a large opportunity between raw extraction and mature epistemic state. Donto is already
good at keeping symbols. It needs much more help describing what those symbols mean, differentiating
them from near neighbors, finding their argument roles, identifying when a binary triple should become a
frame, and choosing which uncertainties deserve the next unit of attention.

### 2.4 Donto's formal gaps are also possible learning targets

The current Donto Calculus explicitly lists several unbuilt or incomplete areas:

- acceptability semantics over argument graphs;
- a production inference calculus;
- a unified standing vector;
- persistent, shareable lens objects;
- a unified result envelope including losses;
- formal semantics for n-ary frames;
- quantity, unit, and uncertainty reasoning;
- confidence composition;
- policy algebra;
- correct negative and absent polarity under predicate alignment.

Alpha should not be used to paper over those formal gaps with plausible prose. Some are database or
calculus engineering problems. But they define excellent behavioral tasks: can a model recognize when a
claim is undercut rather than rebutted, when absence is not negation, when a time change preserves
historical truth, when an alignment is safe for retrieval but unsafe for inference, or when a binary
relation destroys an essential participant role?

---

## 3. What “intelligent about knowledge itself” can mean operationally

This phrase should be made testable. It need not mean consciousness, philosophical wisdom, or a complete
theory of knowledge. For this program it can mean competence at transformations over epistemic objects.

A model is more intelligent about knowledge when it can reliably do things such as:

1. **Distinguish assertion from implication.** Say what a passage states, presupposes, suggests, merely
   permits, and leaves unresolved.
2. **Separate claim from source.** Preserve who said what without treating reported content as endorsed
   fact.
3. **Separate entity from role.** Preserve a bearer while roles begin, end, or conflict.
4. **Separate event from its linguistic packaging.** Recognize a process, culmination, result, or
   nominalized event across different sentences.
5. **Respect time.** Distinguish valid time, record time, current status, former status, and later
   correction.
6. **Respect granularity.** Know when a coarse representation is adequate for one question and destructive
   for another.
7. **Preserve legitimate plurality.** Retain several admissible analyses without inventing unlimited
   possibilities.
8. **Localize revision.** Change only commitments that depend on retracted or superseded evidence.
9. **Invent a useful distinction.** Name a relation that explains a recurring contrast not already covered
   by the available vocabulary.
10. **Reject a false analogy.** Notice that similar words or scenarios do not instantiate the same
    relation.
11. **Compose a query lens.** State which identity, alignment, time, source, polarity, and maturity choices
    are appropriate for a user's question.
12. **Ask the next discriminating question.** Seek evidence that would split live hypotheses, not merely
    collect more of the same.
13. **Know when not to collapse.** Treat “close enough for retrieval” as different from “identical for
    logical inference.”
14. **Know when to stop decomposing.** Estimate whether another analytical aperture will add supported,
    non-redundant structure or only fluent noise.

These are all measurable without requiring the model to memorize the capitals, presidents, diseases,
APIs, or programming languages of the world.

---

## 4. The architectural thesis: facts outside, epistemic operations inside

The proposed system has three different memories and should not confuse them.

| Layer | What it stores | Candidate implementation |
|---|---|---|
| **Parametric competence** | linguistic distinctions, semantic operations, revision behavior, search planning, predicate contracts | Alpha or a later small model |
| **World memory** | source passages, claims, evidence, contradiction, time, identity and predicate hypotheses | Donto plus external search/retrieval |
| **Scientific memory** | every generated candidate, review, split, prompt, exposure, run, metric, and artifact | comprehensive Alpha SQLite ledger |

The model should not need to know a historical fact before it can reason about whether two sources
disagree, whether one source merely reports another, or what question would resolve an identity conflict.
At inference time it can retrieve the relevant source material. This is adjacent to work that explicitly
decouples knowledge storage from reasoning optimization, such as
[RARE](https://arxiv.org/abs/2503.23513), but Alpha's target would be epistemic representation and
predicate behavior rather than benchmark question answering.

This separation also makes the small size scientifically meaningful. If a 58M model succeeds on a
source-contained task using invented names and synthetic worlds, the result cannot easily be dismissed as
factual memorization. If it fails while a 300M control succeeds, that gives a useful capacity threshold.

---

## 5. Recommended flagship question: Predicate Birth and Survival

### 5.1 Primary research question

> **Can a sub-100M conversational model learn to invent source-faithful, reusable natural-language
> predicates from contrastive examples, state their applicability and non-applicability conditions, and
> transport those predicates across lexically isolated domains—without treating a fixed ontology as the
> gold answer?**

### 5.2 Secondary questions

1. Does attention-visible comparison of neighboring cases outperform the same token multiset presented
   independently?
2. Can the model distinguish genuine predicate birth from paraphrasing an existing relation?
3. Can it prefer a broad-but-safe relation over a brittle sentence-specific label without collapsing real
   distinctions?
4. Can it discover that one natural-language expression requires several predicates or an n-ary frame?
5. Can it state non-entailments and counterexamples as reliably as positive examples?
6. Can a predicate learned through a linguistic projection be reused in a mereological or database
   projection, and vice versa?
7. Does an invented predicate improve a downstream Donto query, evidence retrieval, contradiction
   diagnosis, or next-question choice?
8. Can the model preserve a finite set of admissible predicates when ontology choice is genuinely
   purpose- or theory-relative?
9. Can the same small model remain conversational and explain the distinction in ordinary language?

### 5.3 Why “survival” matters more than “invention”

An LLM can always generate a new phrase. That is not a research result. A predicate survives only if it
passes several independent pressures:

- **source pressure:** its instances are genuinely supported by the passage;
- **contrast pressure:** it separates intended positives from hard negatives;
- **paraphrase pressure:** its application does not depend on one wording;
- **counterexample pressure:** the definition can be refined without becoming vacuous;
- **transfer pressure:** it applies in an unseen projection of the same distinction;
- **false-bridge pressure:** it does not transfer where only superficial analogy exists;
- **alignment pressure:** its relation to existing predicates is typed rather than flattened;
- **query pressure:** it helps answer a named competency question;
- **evidence pressure:** its standing can change as source support changes;
- **human pressure:** reviewers can understand and contest the proposed relation.

Donto can preserve failed births. Failure does not require deletion; it requires a recorded reason and low
standing.

---

## 6. The Predicate Birth Neighborhood

### 6.1 Scientific object

A Predicate Birth Neighborhood, abbreviated **PBN**, is a linked family:

```text
PBN = <sources, episodes, transformations, commitments,
       candidate predicates, competency questions, adjudications>
```

where:

- **sources** are the passages or synthetic micro-world statements available to the model;
- **episodes** are ordinary natural-language interactions about those sources;
- **transformations** connect controlled variants such as paraphrases, time shifts, speaker changes,
  counterexamples, and false analogies;
- **commitments** specify what each episode requires, permits, forbids, attributes, or leaves plural;
- **candidate predicates** are possible relations with contracts, not one mandatory label;
- **competency questions** say what a representation must support;
- **adjudications** preserve reviews, disagreements, revisions, and evidence.

The model-visible text remains natural language. IDs, chat delimiters, IRIs, JSON, and training wrappers
are injected by deterministic renderers after selection from SQLite. This preserves the existing Alpha
decision that training content should not be polluted by serialization syntax unless syntax learning is a
separate experiment.

### 6.2 Predicate contract

A useful predicate candidate needs more than a name. Its hidden scientific contract should be capable of
recording:

- provisional label and alternative lexicalizations;
- natural-language gloss and definition;
- arity;
- argument roles and direction;
- expected subject and object kinds, if defensible;
- temporal dependence;
- intensional versus extensional behavior where relevant;
- positive instances;
- near-miss and hard-negative instances;
- necessary conditions;
- sufficient conditions, if any;
- known non-entailments;
- exceptions and boundary cases;
- whether several analyses remain admissible;
- relationship to existing predicates: exact, close, broader, narrower, inverse, decomposition,
  incompatible, or no known mapping;
- source anchors and hypothesis status;
- competency questions the predicate supports;
- known questions for which the predicate is too coarse;
- review, revision, and survival history.

The student does not have to emit this as a rigid record. It can answer in ordinary language; a later
renderer or evaluator maps the answer to the contract. A structured auxiliary head can be studied only
after the natural-language capability is established.

### 6.3 Set-valued answers

Predicate labels are not unique gold strings. “holds a temporary institutional role” and “bears a
time-qualified institutional status” may be equally defensible. A neighborhood should therefore record:

- required semantic commitments;
- permitted semantic commitments;
- forbidden semantic commitments;
- admissible predicate families;
- excluded predicate families;
- distinctions that additional evidence could resolve;
- distinctions that remain theory- or purpose-relative.

Evaluation should compare behavior, applicability, and consequences—not exact wording.

---

## 7. Worked example: discovering a role predicate

### 7.1 Source-contained cases

The model sees several ordinary cases:

> Lina enrolled at the college in 2022. She graduated yesterday. The registrar still retains her student
> record, but she is no longer enrolled.

> Emre became treasurer in March. His term ended in June. The association still retains his financial
> reports.

> Jo rented the apartment for six months. The lease ended, but Jo is the same person and the apartment is
> the same apartment.

The intended recurring distinction is not the words *student*, *treasurer*, or *tenant*. It is that a
persistent bearer can enter and leave a temporally and institutionally dependent role.

### 7.2 Predicate birth prompt

> What relationship do these cases share that would be useful to name? Explain when it applies, when it
> stops applying, and what remains the same.

An admissible answer might propose a relation such as **bears a time-qualified role**, explain that the
person persists while the role begins and ends, and preserve historical role attribution.

### 7.3 Counterexample

> A statue was melted down and recast into a bell. Is this just another case of a bearer losing a role?

The model should reject the transfer. This case concerns material constitution and artifact identity, not
merely role termination.

### 7.4 Cross-projection test

All training examples containing *student*, *former*, *graduation*, and *enrolment* are excluded. The
model is then asked:

> Why does “former student” make sense, while “former person” ordinarily sounds very different?

Credit requires preserving the bearer/role distinction without merely using the technical word *role*.

### 7.5 Competency-sensitive representation

> I only need to print today's class list. Do I need all that temporal structure?

The correct answer may recommend a simpler current-membership relation for that narrow purpose while
stating which later questions—when someone enrolled, whether they graduated, or which roles overlapped—
the simplification cannot answer.

This is Donto-native because no representation is declared globally correct. Adequacy is judged against
a question, and richer distinctions remain available rather than being destroyed.

---

## 8. Worked example: intent without mind reading

### 8.1 Source dialogue

> A: Can you join the meeting on Thursday?
>
> B: I can probably make the afternoon work.

A shallow intent label might be `accept`. A richer deconstruction distinguishes:

- an answer about availability;
- a weak positive commitment;
- uncertainty carried by *probably*;
- restriction to the afternoon;
- cooperative engagement with the request;
- no commitment to the morning;
- no evidence about why the speaker is uncertain.

Possible predicates include **signals conditional availability for**, **tentatively commits to**, and
**narrows acceptable time to**. These are not necessarily synonyms; they expose different useful
questions.

### 8.2 Controlled transformations

Compare:

- “Yes, I will be there Thursday afternoon.”
- “I might be able to make the afternoon work.”
- “Thursday afternoon is open on my calendar.”
- “I wish I could, but Thursday is impossible.”
- “My assistant says Thursday is open.”
- “I can probably make the afternoon work, but do not schedule around me yet.”

The model must track commitment strength, source, condition, and cancellation. It must not infer a hidden
psychological cause or convert politeness conventions into certain facts.

### 8.3 Why intent is scientifically attractive and dangerous

Intent deconstruction is close to the user's interest because it tests semantics, pragmatics, attribution,
uncertainty, and conversational repair without requiring encyclopedic knowledge. But it is culturally and
contextually loaded. The [PUB benchmark](https://aclanthology.org/2024.findings-acl.719/) already covers
implicature, presupposition, reference, and deixis, and reports wide variation plus a human–model gap.
Alpha's novelty cannot be “we evaluate pragmatics.” It would need to test open predicate birth,
commitment locality, and culturally qualified competing analyses rather than another multiple-choice
pragmatics suite.

---

## 9. Worked example: mereology as a high-value testbed

Part–whole language is an excellent laboratory because ordinary language collapses relations that
formal knowledge systems must distinguish:

- component–object;
- member–collection;
- portion–mass;
- material–object;
- place–area;
- phase–process;
- feature–activity;
- constitution without identity;
- dependent boundary;
- functional part versus arbitrary fragment.

Consider:

> The committee lost three members but continued to exist.

> The engine lost a piston and stopped functioning.

> The loaf lost a slice but remained a loaf.

> The statue is bronze, but the bronze is not the statue under every identity criterion.

> The shoreline moved even though no detachable object called “the shoreline” moved.

A model that assigns all of these `partOf` has learned almost nothing. A useful model should discover
which contrasts matter for persistence, counting, transitivity, replacement, function, and query
behavior. Existing research already shows that language models can possess fragments of everyday
part-structure knowledge while violating coherence constraints at high rates
([Gu, Dalvi Mishra, and Clark 2023](https://aclanthology.org/2023.acl-long.106/)). There is also direct
work on meronymic ontology extraction. The opportunity is not mereology itself, but **using mereology to
test whether a tiny model can invent and transport distinctions while avoiding false transitivity and
query-destructive collapse**.

---

## 10. How this changes Alpha Joints

Alpha Joints remains useful. Its linked transformations, whole-neighborhood splits, false bridges,
set-valued analyses, relation-visibility arms, and cross-projection tests are exactly the experimental
controls this new program needs.

The change is the predicted object:

| Current Alpha Joints emphasis | Predicate Birth and Survival emphasis |
|---|---|
| Produce the correct answer after a semantic transformation | Discover or refine the relation that explains the transformation |
| Track commitment deltas | Use deltas to define a predicate's applicability contract |
| Test cross-projection reasoning | Test cross-projection predicate reuse and false-bridge rejection |
| Treat a neighborhood as linked episodes | Treat it as the evidence and counterevidence surrounding a predicate birth |
| Evaluate invariance and revision | Also evaluate relation usefulness, fragmentation, over-collapse, and survival |

This is an extension, not a rejection. The earlier document should remain intact as the controlled
learning-science design until third-party review decides whether the new headline is stronger.

---

## 11. Closest prior art and the claim it removes

The following table is deliberately adversarial. Every row removes an easy novelty claim.

| Work | What already exists | What Alpha therefore cannot claim | Possible remaining distinction |
|---|---|---|---|
| [Open Information Extraction](https://aclanthology.org/2020.emnlp-main.690/) | Schema-free subject–predicate–object extraction from text | “We extract open predicates rather than use a fixed schema” | Predicate contracts, contradiction, plurality, and cross-domain survival |
| [Abstractive OpenIE](https://aclanthology.org/2023.emnlp-main.376/) | Predicates may contain words absent from the source and may express inferred relations | “Our predicates are abstractive rather than copied spans” | Explicit epistemic status, counterexamples, and query-tested survival |
| [Open Relation Extraction and Grounding](https://aclanthology.org/I17-1086/) | Relation-type naming and grounding to KB schemas | “We name open relations and compare them to existing predicates” | Preserve no-match and typed non-equivalence rather than forcing grounding |
| [Linking Surface Facts to Large-Scale KGs](https://aclanthology.org/2023.emnlp-main.445/) | Benchmarks recognize out-of-KG predicates and show that detecting them is hard | “We distinguish new predicates from existing ones” | Birth contracts plus downstream survival in an open epistemic substrate |
| [KNoRD](https://arxiv.org/abs/2305.13533) | Known and novel relation discovery with hard negatives and long-tail classes | “We discover novel relation classes in open-world data” | Truly open natural-language contracts, set-valued analyses, and transfer beyond dataset labels |
| [CEO](https://arxiv.org/abs/2305.13521) | Open-domain event ontology induction with meaningful names | “We induce a new event schema from corpora” | Small-model semantic transport and evidence-preserving predicate ecology |
| [LLMs4OL](https://arxiv.org/abs/2307.16648) | LLM term typing, taxonomy discovery, and non-taxonomic relation extraction | “We use LLMs for ontology learning” | Open birth rather than recognition over a predefined relation set |
| [OntoLearner](https://arxiv.org/abs/2607.01977) | 180 ontologies, 22 domains, standardized ontology-learning benchmarks | “We provide the first cross-domain LLM ontology-learning substrate” | Donto-native open relations, ambiguity, query use, and sub-100M transport |
| [Generative Ontology Induction](https://arxiv.org/abs/2607.16201) | Domain-agnostic class-level schema discovery from document corpora | “We generate a domain ontology from examples” | Predicate-level epistemic contracts and survival rather than static blueprint coverage |
| [ScheMatiQ](https://aclanthology.org/2026.acl-demo.22/) | A research question plus corpus is turned into a steerable schema and grounded database | “We are the first to induce purpose-sensitive schemas from questions” | Query-relative non-collapse and learned predicate transformations in a tiny model |
| [AutoSchemaKG](https://aclanthology.org/2026.acl-long.942/) | Dynamic schema induction and KG construction over 50M+ documents, entities and events | “Schema-free autonomous KG construction is new” | Small-model epistemic competence, contradictions, typed mappings, and survival rather than scale |
| [ADVENT](https://arxiv.org/abs/2607.01585) | LLM-generated named auxiliary predicates, formal Prolog verification, and cross-task reuse | “LLM-driven predicate invention with verification is new” | Natural-language source grounding, open epistemic contracts, ambiguity, and cross-domain semantic transport |
| [Competency-question generation](https://aclanthology.org/2025.ldk-1.15/) | LLM-assisted generation and filtering of ontology competency questions | “Using competency questions to steer ontology design is new” | Use them as interventions and survival tests rather than generated documentation |
| [RARE](https://arxiv.org/abs/2503.23513) and [KARD](https://arxiv.org/abs/2305.18395) | Externalize knowledge and train smaller models on reasoning with retrieved evidence | “A small model can reason while facts live outside its parameters” | Train epistemic decomposition and predicate behavior rather than answer rationales |

Two local findings narrow the claim further:

1. `semholo` already conducted an extensive prior-art audit and found that broad novelty claims around
   semantic transport, sheaf obstruction, geometry, holonomy, query determinacy, and concept invention
   collide with substantial literature. “Semantic holonomy” should not be revived as a headline.
2. Alpha Joints already documents direct precedents for metamorphic transformation families, contrast
   sets, belief revision, ambiguity-aware alignment, and relational curricula. Linked examples are an
   experimental method, not by themselves the contribution.

### 11.1 The direct ADVENT collision

ADVENT is especially important because it appeared in July 2026 and uses the exact phrase *predicate
invention*. It asks an LLM to identify implicit patterns in structured relational examples, propose named
auxiliary predicates and logic definitions, and refine them through Prolog execution. On transformed
poker-hand tasks it reports that formal verification improves success and that accumulated inventions aid
later tasks.

Alpha must differ in substance, not branding:

- ADVENT's input is structured ILP background knowledge and positive/negative examples; Alpha's would be
  natural-language sources and conversations.
- ADVENT seeks a rule that solves a target classification; Alpha would preserve multiple admissible
  representations tied to different competency questions.
- ADVENT's verification is deductive execution; Alpha would combine executable cores with evidence,
  temporal, perspectival, pragmatic, and human judgments that cannot all be reduced to Prolog.
- ADVENT uses frontier LLMs as inventors; Alpha asks what a sub-100M student can internalize and transfer.
- ADVENT reuses invented helper predicates across structured tasks; Alpha's strongest test is transport of
  a semantic distinction into a lexically isolated domain plus downstream Donto utility.

### 11.2 The direct AutoSchemaKG collision

AutoSchemaKG dynamically extracts entities, events, relations, and concepts from web-scale corpora. It
reports 900M+ nodes, 5.9B edges, and downstream multi-hop QA gains. This is a formidable collision with
any claim framed as “LLMs can build schema-free knowledge graphs.”

Donto's difference is not that it is bigger or more automatic. It is that it treats contradiction,
evidence, bitemporality, identity, and mappings as first-class hypotheses and refuses irreversible schema
collapse. Alpha's research must measure those differences directly. If the only result is more QA recall,
AutoSchemaKG and many GraphRAG systems already occupy the space.

### 11.3 The OntoLearner warning

OntoLearner reports that failure varies strongly with ontological structure, relation type, and domain;
larger models do not reliably solve taxonomy discovery, and output discipline can matter more than
additional “thinking.” That is a warning for Alpha. A 58M model may learn surface relation phrases yet
fail on hierarchy, closure, or global consistency. The pilot therefore needs a larger positive control and
must not infer deep ontology competence from fluent definitions.

---

## 12. Defensible novelty statement, if the decisive experiment works

> **Predicate Birth and Survival investigates whether a sub-100M conversational language model can
> induce reusable relation contracts from source-grounded natural-language contrast families and
> transport those contracts across lexically isolated linguistic, mereological, pragmatic, and
> ontological projections. Each relation contract specifies supported instances, hard negatives,
> non-entailments, admissible competing analyses, temporal and perspectival qualifications, typed links
> to nearby predicates, and competency questions it enables. The decisive comparison holds source and
> episode tokens constant while varying whether correct relations among cases are invisible,
> attention-visible, or explicitly supervised. Success requires downstream query utility, localized
> revision, false-bridge rejection, and preservation of non-equivalence in a contradiction-preserving
> substrate—not merely generation of a plausible predicate name.**

This statement is not ready for publication until a broader systematic review confirms the exact gap.

---

## 13. Candidate program portfolio

The user asked for many possible fascinating directions. They should be preserved as a portfolio rather
than quietly collapsed into the recommended one.

### 13.1 Alpha Predicate Foundry / Predicate Birth and Survival

**Question:** Can a tiny model discover relations worth naming and make their boundaries inspectable?

**Donto value:** directly enriches predicate descriptors, alignments, non-equivalences, query expansion,
and future inference safety.

**Best feature:** clean bridge between Alpha Joints and the live million-predicate ecology.

**Main risk:** “predicate invention” and dynamic schema induction have direct prior art; the experiment
must center survival, transfer, contradiction, and query utility.

### 13.2 Alpha Epistemic Compiler

**Question:** Can a small model compile a paragraph into claims, sources, attestations, valid-time
qualifiers, modalities, competing readings, argument relations, and n-ary frames while keeping its output
conversational and source-faithful?

**Donto value:** highest immediate utility. It could become a learned front end to Donto's HOLD loop.

**Main risk:** resembles rich information extraction, semantic parsing, AMR, OpenIE, proposition
decomposition, and event extraction. Novelty would depend on contradiction, provenance, and open
predicate birth—not extraction alone.

### 13.3 Alpha Mereology Laboratory

**Question:** Can a small model learn and transport distinctions among component, member, portion,
material, phase, boundary, constitution, and identity?

**Donto value:** prevents one of the most damaging classes of predicate collapse and supports better
frames and identity hypotheses.

**Main risk:** narrower application and substantial formal/multilingual annotation expertise.

**Recommended role:** first deep domain within Predicate Birth and Survival, not necessarily a separate
flagship.

### 13.4 Alpha Intent Microscope

**Question:** Can a small model deconstruct requests, refusals, commitments, presuppositions,
implicatures, audience design, repair conditions, and uncertainty without pretending to read minds?

**Donto value:** turns conversations and agent actions into richer claim/commitment trails.

**Main risk:** cultural authority, annotator disagreement, and large existing pragmatics literature.

**Recommended role:** one carefully governed projection, with multilingual and community review.

### 13.5 Alpha Lens Composer

**Question:** Given a competency question, can the model propose a reversible Donto lens—scope, time,
identity hypothesis, predicate closure, maturity, polarity, contradiction posture, source preference, and
export safety—and explain its losses?

**Donto value:** directly activates a major unbuilt object in the Lens Spec.

**Main risk:** evaluation depends on a lens registry and unified query envelope that do not yet exist.

**Recommended role:** later joint model/substrate paper after lens infrastructure is real.

### 13.6 Alpha Non-Equivalence Judge

**Question:** Can a small model tell exact equivalence from close match, broader/narrower, inverse,
decomposition, incompatible, and no-supported-mapping under a stated query scope?

**Donto value:** enormous. Accepted exact mappings are rare, and explicit accepted non-equivalence is
nearly absent.

**Main risk:** ontology alignment is mature prior art, and static pair classification could become dull.

**Recommended role:** a survival pressure and high-value practical benchmark within the flagship.

### 13.7 Alpha Frame Composer

**Question:** Can the model recognize when binary triples lose essential participant structure and propose
an event, measurement, experiment, speech-act, or decision frame with typed roles?

**Donto value:** the live frame layer is nearly empty.

**Main risk:** substantial overlap with semantic role labeling, FrameNet, AMR, UCCA, and event schema
induction.

**Recommended role:** one output mode of the Epistemic Compiler.

### 13.8 Alpha Contradiction Cartographer

**Question:** Can the model distinguish direct contradiction, incompatible classification, source
disagreement, time mismatch, scope mismatch, granularity mismatch, undercutting, and merely different
vocabulary?

**Donto value:** helps turn preserved contradictions into useful argument graphs and proof obligations.

**Main risk:** conflict taxonomies and NLI are crowded; the novelty would be localized revision and
evidence-seeking.

### 13.9 Alpha Negative-Space Reader

**Question:** Can the model distinguish what a passage denies, presupposes, implicates, fails to assert,
leaves unknown, and makes impossible under a local micro-world?

**Donto value:** directly addresses negative and absent polarity semantics.

**Main risk:** absence is extremely easy to hallucinate. Evaluation must be tightly source-contained.

### 13.10 Alpha STEER

**Question:** Given live hypotheses and evidence, can the model propose the next search, observation,
record request, or question with highest expected discriminatory value?

**Donto value:** closest to Donto's north-star STEER loop.

**Main risk:** information-gain evaluation is hard without environments or future outcomes.

**Recommended role:** second major paper after representation quality is proven.

### 13.11 Alpha Search-Native Scholar

**Question:** Can a fact-light model retrieve passages, deconstruct them, mint provisional predicates,
anchor its claims, and explain which uncertainty remains?

**Donto value:** realizes the desired division between external facts and internal epistemic competence.

**Main risk:** retrieval quality, reasoning, citation, and predicate learning become confounded.

**Recommended role:** integration stage, not the first causal experiment.

### 13.12 Alpha Decomposition Controller

**Question:** Which next analytical aperture—syntax, event structure, intent, mereology, causation,
provenance, counterfactual, or another model-invented axis—will produce the most supported,
non-redundant value per unit of compute?

**Donto value:** directly attacks the economics of maximal extraction without reducing depth.

**Main risk:** needs a trustworthy value function; otherwise it rewards more verbose output.

### 13.13 Alpha Claim Ontogeny

**Question:** Can the model understand the difference between a newly observed assertion, an anchored
claim, a corroborated claim, a contested claim, a promoted claim, and a superseded claim—and identify
what evidence licenses each transition?

**Donto value:** teaches maturity and standing rather than undifferentiated confidence.

**Main risk:** much of this behavior may be implemented more reliably as deterministic governance.

### 13.14 Alpha Query-Relative Collapse License

**Question:** Can the model decide when two predicates or entities may be treated as equivalent for one
question while preserving their distinction for another, and report the information loss?

**Donto value:** perhaps the purest expression of query-time joining.

**Main risk:** the local Semantic Holonomy audit found close prior art around query determinacy,
abstraction refinement, and reversible collapse. It is also difficult to evaluate before Donto lenses and
loss reports exist.

### 13.15 Alpha Concept Boundary Learner

**Question:** Given positive, negative, and ambiguous neighboring cases, can the model state the latent
distinction without being given its name?

**Donto value:** this is the simplest stepping stone toward predicate birth.

**Main risk:** can degrade into ordinary concept learning unless cross-domain transfer and open naming
are required.

### 13.16 Alpha Epistemic Compression

**Question:** What is the smallest set of anchored, qualified commitments from a passage that preserves
answers to a declared family of questions?

**Donto value:** balances generative abundance against query utility without deleting the full source.

**Main risk:** “minimal” is always relative to the query set; a fixed compression target would contradict
Donto's open future-question posture.

### 13.17 Alpha Predicate Ecology

**Question:** Across time, which freely minted predicates recur, split, merge only under a lens, acquire
counterexamples, or become useful for new questions?

**Donto value:** turns the existing million-predicate population into a longitudinal scientific object.

**Main risk:** observational analysis of the current graph could be dominated by extractor artifacts and
historical prompt styles.

**Recommended role:** later empirical paper after predicate birth contracts exist.

---

## 14. Comparative decision matrix

Scores are provisional, on a five-point scale. Higher is better except **risk**, where five means high
risk. They are decision aids, not quantitative findings.

| Program | Novelty opening | Direct Donto value | 58M feasibility | Evaluation clarity | First-pilot cost | Risk | Recommended position |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Predicate Birth and Survival | 4 | 5 | 3 | 4 | 3 | 4 | **Primary candidate** |
| Epistemic Compiler | 3 | 5 | 3 | 3 | 3 | 4 | Practical companion |
| Mereology Laboratory | 3 | 4 | 4 | 4 | 2 | 2 | First deep projection |
| Intent Microscope | 3 | 4 | 3 | 2 | 3 | 5 | Governed projection only |
| Lens Composer | 4 | 5 | 3 | 3 | 4 | 4 | Later, after lens runtime |
| Non-Equivalence Judge | 2 | 5 | 4 | 5 | 2 | 2 | Core evaluation/task |
| Frame Composer | 2 | 5 | 3 | 4 | 3 | 3 | Compiler subtask |
| Contradiction Cartographer | 3 | 5 | 3 | 4 | 3 | 3 | Second-wave task |
| Negative-Space Reader | 3 | 4 | 2 | 3 | 3 | 4 | Diagnostic task |
| STEER | 4 | 5 | 2 | 2 | 5 | 5 | Long-term north star |
| Search-Native Scholar | 2 | 5 | 3 | 2 | 4 | 4 | Integration demonstration |
| Decomposition Controller | 4 | 5 | 2 | 2 | 4 | 5 | Later research |
| Claim Ontogeny | 3 | 4 | 4 | 4 | 2 | 2 | Curriculum component |
| Query-Relative Collapse | 4 | 5 | 2 | 2 | 5 | 5 | High-upside later paper |
| Concept Boundary Learner | 2 | 4 | 5 | 5 | 1 | 1 | First stepping stone |
| Epistemic Compression | 4 | 4 | 2 | 2 | 4 | 5 | Theoretical side program |
| Predicate Ecology | 4 | 5 | 3 | 3 | 4 | 4 | Longitudinal Donto study |

My current ranking is:

1. **Predicate Birth and Survival** as the scientific flagship.
2. **Mereology Laboratory** and **Non-Equivalence Judge** as the cleanest initial domains and pressures.
3. **Epistemic Compiler** as the practical Donto pathway.
4. **STEER** and **Query-Relative Collapse** as the most intellectually ambitious later targets.

---

## 15. Non-negotiable prerequisite: do not confound knowledge behavior with response initiation

The archived Alpha checkpoint is not currently a conversational research instrument. Its sealed terminal
evaluation produced:

- 92/100 empty responses caused by immediate EOS;
- 94/100 responses terminated with EOS;
- only 8/100 nonempty responses;
- six degenerate loops;
- 2/100 structural passes;
- 0/100 semantic passes under blinded inspection;
- 0/200 closed-book QA exact or contained answers.

The model may contain useful token-level regularities, but free generation cannot reveal them reliably.
Any future conceptual experiment must therefore keep **P0 response initiation** separate from
**P1 predicate learning**.

Before the model can be used for this research, a future authorized run must meet a frozen gate such as:

- at least 99% nonempty responses on a balanced initiation suite;
- no systematic first-token EOS preference;
- no degenerate loops;
- stable initiation across short, medium, and long prompts;
- ordinary short-answer, explanation, clarification, disagreement, and multi-turn behavior;
- checkpoint selection by free generation, not teacher-forced loss alone.

P0 should contain no advanced ontology content. Otherwise an apparent predicate gain could merely be the
first dataset that taught the model to start talking.

This document does **not** authorize such a run. It specifies the gate for a later decision.

---

## 16. A decisive micro-pilot

### 16.1 Start with depth, not 200,000 rows

The first experiment should use **24–60 deeply controlled concept families**, not 200,000 independent
chats. The unit of science is a semantic distinction with several projections and adversarial neighbors,
not an episode count.

A 24-family core could include:

| Family | Linguistic projection | Ontological / Donto projection | Key false bridge |
|---|---|---|---|
| Role versus bearer | *student*, *former student* | time-qualified institutional role | material transformation |
| Type versus token | a word versus this occurrence | work, edition, copy, record | member of a class |
| Member versus component | *committee member* | collection membership | engine component |
| Component versus material | *wooden handle* | functional part / constituent material | arbitrary portion |
| Portion versus countable part | *some water*, *a slice* | mass portions and individuation | detachable component |
| Constitution versus identity | statue and bronze | coincident entities | ordinary role change |
| Collective versus members | singular/plural agreement | group persistence | set extensionality assumed globally |
| Event versus object | nominalization | event reification and participants | result object |
| Process versus culmination | progressive/perfective | event boundary and completion | mere temporal duration |
| Cause versus correlation | causal language | argument/evidence relation | temporal succession |
| Necessary versus sufficient | conditionals | constraint and implication direction | frequent association |
| Disposition versus manifestation | *fragile* versus *broke* | capacity and realization | observed event alone |
| Claim versus source | reported speech | attestation and provenance | endorsement by recorder |
| Evidence versus assertion | evidentials | evidence links and claim standing | confidence score |
| Valid time versus record time | tense and temporal adverbs | bitemporality | publication date assumed as event date |
| Current versus historical truth | *is* / *was* / *former* | open and closed validity | correction treated as erasure |
| Speaker versus content | quotation and attribution | claim node versus agent | same sentence, new speaker |
| Assertion versus presupposition | factive and change-of-state triggers | required versus background commitment | mere implication |
| Entailment versus implicature | scalar and conversational inference | asserted versus defeasible claim | lexical synonymy |
| Deontic versus descriptive | *must*, *may*, *is* | obligation versus observed behavior | prediction |
| Polysemy versus coreference | related word senses | predicate split versus entity merge | homonymy assumed |
| Absence versus negation | missing mention | open-world unknown versus explicit negative | database null as false |
| Granularity shift | general/specific wording | coarse/fine schema and loss | exact equivalence |
| Identity through change | rename, repair, replacement | identity hypothesis | loose similarity |

Each family should contain:

- three or more genuinely different projections;
- four to six primitive transformations;
- two composed transformations;
- at least one false analogy;
- at least one same-words/different-relation control;
- at least one different-words/same-relation control;
- one hard negative;
- one admissible-plurality case where appropriate;
- several competency questions;
- an explicit predicate contract or finite admissible set;
- at least two independently reviewed surface realizations.

### 16.2 Source conditions

Use three evidence regimes:

1. **Synthetic micro-worlds:** invented names, entities, institutions, and rules; facts are fully supplied in
   the prompt.
2. **Ordinary-language vignettes:** familiar situations whose answer depends on the text, not obscure
   background facts.
3. **Retrieved passages:** external documents with frozen source fragments and anchors, used only after
   the closed-source result is understood.

The first causal claim should be made on regimes 1 and 2. Regime 3 demonstrates utility but introduces
retrieval and source-selection confounds.

### 16.3 Whole-family and lexical isolation

Splits occur at family and projection level. A test item must not share:

- the latent family when measuring unseen-family generalization;
- the held-out projection when measuring cross-projection transfer;
- distinctive jargon;
- scenario templates;
- teacher phrasing;
- proper names;
- source documents;
- construction programs;
- reviewer-specific paraphrase style.

For a role/bearer test on *former student*, training should exclude not only that phrase but closely
related enrolment, graduation, and school scenarios. Success should depend on transporting the
distinction, not recalling a lexical pattern.

---

## 17. Experimental arms: relation visibility is the causal variable

A standard causal LM does not know that two training examples are linked merely because they come from
the same database neighborhood. The comparison must vary what relational structure the model can
actually observe.

| Arm | Model exposure | What it isolates |
|---|---|---|
| **A — Generic dialogue** | Equal-token ordinary conversations | P0 and token-budget effects |
| **B — Independent targeted** | Predicate-neighborhood episodes shuffled and attention-separated | Targeted content effect |
| **C — Co-batched** | Correctly related episodes in one optimizer batch but separate attention contexts | Gradient co-location and rehearsal |
| **D — Packed comparison** | The same episode text in one attention-visible sequence, without edge labels | Direct comparison effect |
| **E — Explicit relation** | Natural-language comparison tasks or an auxiliary predicate/delta objective | Explicit relational supervision |
| **F — Corrupted relation** | Same format and schedule as D/E, but wrong pairings or contracts | Formatting and regularization control |

All feasible arms should hold constant:

- base checkpoint;
- model architecture;
- accepted episode text;
- model-visible token count;
- optimizer-step budget;
- answer-start intervention;
- evaluation and checkpoint-selection policy;
- random seeds;
- prompt renderer except where relation visibility necessarily changes it.

If E alone works, the claim is not that neighborhood organization is sufficient; it is that explicit
relation supervision is required. If D does not beat B, relational proximity may not teach anything. If F
matches D or E, gains likely come from formatting or rehearsal rather than correct semantics.

### 17.1 Positive capacity control

Repeat the decisive comparison on one 150M–300M model. The 58M target remains scientifically central,
but a larger control separates two null results:

- the curriculum does not express the intended capability;
- the curriculum is learnable, but 58M lacks capacity or foundation.

### 17.2 Continued pretraining versus conversational post-training

Run, at equal targeted-token budgets, at least:

- targeted SFT only;
- targeted continued pretraining followed by small conversational SFT;
- generic continued pretraining plus targeted SFT.

Broad semantic distinctions may need to be installed before instruction tuning. The previous failed run
does not establish that SFT is the right learning stage.

---

## 18. Training tasks inside one Predicate Birth Neighborhood

The same semantic family can produce several task types. They should be separately tagged so later runs
can include or exclude them.

### 18.1 Observation

Describe only what the passage licenses. Distinguish explicit content, defeasible inference, and unknown.

### 18.2 Contrast discovery

Given two or more cases, state the difference that changes the answer.

### 18.3 Predicate birth

Propose one or more useful names for the recurring relation and explain the argument roles.

### 18.4 Contract induction

State positive conditions, exclusions, non-entailments, temporal or perspectival qualifications, and
boundary cases.

### 18.5 Application

Decide whether the predicate applies to a new case and justify the decision using the contract.

### 18.6 Counterexample repair

Revise a predicate definition after a genuine counterexample without making it vacuous or merely adding
the example as an exception.

### 18.7 False-bridge rejection

Explain why a superficially similar case instantiates a different relation.

### 18.8 Cross-projection transport

Apply a learned relation in a domain with different vocabulary and entities.

### 18.9 Alignment

Relate a candidate predicate to nearby existing ones using typed mappings and a stated scope.

### 18.10 Competency test

Choose which predicate or representation is adequate for a given question and state the lost questions.

### 18.11 Decomposition

Recognize when one surface relation must split into several predicates or an n-ary frame.

### 18.12 Search planning

State what source or observation would distinguish the live candidate contracts.

### 18.13 Teach-back conversation

Explain the distinction at novice, intermediate, and expert depth; answer follow-ups; correct a user's
misunderstanding; and remain natural rather than turning every response into an ontology lecture.

---

## 19. Evaluation: a predicate earns survival

No single scalar captures success. Report a vector and preserve the components.

### 19.1 Source fidelity

- **anchor support:** proportion of required assertions supported by the supplied source span;
- **unsupported commitment rate:** commitments introduced without evidence;
- **attribution accuracy:** claims remain attached to the correct speaker or source;
- **epistemic-status accuracy:** asserted, inferred, hypothetical, denied, and unknown are distinguished.

### 19.2 Boundary quality

- **positive coverage:** intended positive cases accepted;
- **hard-negative rejection:** near misses rejected;
- **false-bridge rejection:** superficial analogies rejected;
- **non-entailment accuracy:** forbidden consequences are not claimed;
- **overbreadth:** unrelated cases incorrectly captured;
- **overspecificity:** valid paraphrases or new instances rejected.

### 19.3 Reuse and transport

- **within-projection reuse:** new lexical realizations in the same domain;
- **cross-projection transfer:** new domain and vocabulary;
- **jargon-free transfer:** technical labels removed or replaced;
- **same-word/different-relation accuracy:** lexical shortcuts resisted;
- **different-word/same-relation accuracy:** surface differences ignored when appropriate.

### 19.4 Predicate ecology

- **fragmentation rate:** many needless predicates for one reusable distinction;
- **over-collapse rate:** real distinctions merged;
- **typed-alignment accuracy:** exact, close, broader, narrower, inverse, decomposition, incompatible, or
  no-match;
- **alignment safety calibration:** retrieval-safe is not automatically export- or inference-safe;
- **descriptor completeness:** definition, roles, examples, counterexamples, and scope are recoverable.

### 19.5 Query utility

- **competency-question coverage:** which declared questions become answerable;
- **answer lift:** improvement over passage-only or raw-triple baselines;
- **retrieval lift:** useful evidence found because of the predicate;
- **loss awareness:** questions made unanswerable by a coarse representation are identified;
- **revision locality:** new evidence changes only dependent answers.

### 19.6 Plurality

- **admissible-set precision:** proposed analyses are defensible;
- **admissible-set recall:** important alternatives retained;
- **overhedging rate:** unsupported alternatives invented;
- **clarification efficiency:** asks the smallest question that can actually reduce the set;
- **permanent-plurality recognition:** does not promise that more evidence will settle a theory-relative
  choice.

### 19.7 Conversation

- nonempty-response rate;
- first-token EOS rate;
- loop/repetition rate;
- directness;
- contextual depth control;
- follow-up coherence;
- correction and repair;
- ordinary language without forced jargon;
- refusal to manufacture ontology detail when the user asked a simple question.

### 19.8 Human and executable oracles

Prefer, in order:

1. executable micro-world constraints;
2. source-verifiable anchors;
3. controlled transformations with hand-authored deltas;
4. multiple human judgments at family level;
5. model judges for softer conversational properties only.

Do not allow the same teacher family to generate, adjudicate, and score the decisive test. Existing work
on iterative counterexample generation has shown that model judges can accept invalid counterexamples;
fluent adjudication is not ground truth.

### 19.9 Statistical unit

The unit of inference is the concept family or neighborhood, not the episode. Use paired or hierarchical
analysis with effects for:

- concept family;
- transformation type;
- projection pair;
- source regime;
- teacher or author;
- rendering template;
- reviewer;
- model seed.

Thousands of variants from thirty families do not constitute thousands of independent observations.

---

## 20. Synthetic data: useful, but the object is not 200,000 chats

Using a state-of-the-art teacher is sensible. It should generate **candidate semantic neighborhoods**, not
an undifferentiated mountain of polished replies.

### 20.1 Teacher roles

Separate roles where possible:

- **constructor:** proposes source cases and latent distinctions;
- **surface realizer:** writes diverse natural-language variants;
- **adversary:** generates hard negatives and false bridges;
- **contract critic:** identifies missing conditions and overbreadth;
- **source verifier:** checks that claims are licensed by the passage;
- **formal checker:** executes the tractable subset;
- **human reviewer:** accepts, rejects, or records plural analyses.

Model identity, version, prompt, sampling settings, timestamp, inputs, outputs, and cost/usage metadata must
be preserved for every generation.

### 20.2 Candidate multiplication

For each accepted episode, retain all candidates and rejection reasons. It may be reasonable to generate
five to twenty candidates per desired accepted item. The rejected population is scientifically valuable:
it reveals teacher shortcuts, ambiguous specifications, recurring false bridges, and annotation
difficulty.

### 20.3 Diversity is not word substitution

Require diversity along independent axes:

- lexical realization;
- syntax;
- discourse position;
- speaker relationship;
- language or dialect where authority exists;
- institution and social setting;
- temporal structure;
- evidence structure;
- granularity;
- projection domain;
- ambiguity type;
- counterexample type.

Paraphrasing one template 200,000 times is not a large conceptual dataset.

### 20.4 Freeze evaluation before generation

The final evaluation families, construction recipes, jargon exclusions, and leakage rules must be frozen
and hashed before large-scale teacher generation. Otherwise the teacher pipeline will gradually tune
itself to the test.

### 20.5 Cultural and linguistic authority

Pragmatic force, politeness, kinship, personhood, classification, evidentiality, and ontology are not
culturally neutral. The database must distinguish:

- source-attested analysis;
- teacher-proposed analysis;
- linguist-reviewed analysis;
- community- or speaker-reviewed analysis;
- disputed analysis;
- rejected analysis;
- pending review.

No synthetic teacher is authorized to become the final authority for a living language or community's
conceptual categories.

---

## 21. Why comprehensive SQLite tracking is correct

There is no good scientific reason to discard generated candidates, rejected analyses, tokenizations,
reviews, or lineage. The earlier warning against letting the database become the first paper's headline
should not be misunderstood as a recommendation to track less.

The correct principle is:

> **Track everything; materialize and normalize it in phases; do not let schema construction delay the
> first causal test.**

“Tracked” does not require every large byte sequence to be stored inline in one SQLite page. Large model
files, raw media, and checkpoint tensors may live as content-addressed artifacts, but SQLite must contain
their hash, size, media type, producer, location, lifecycle state, and relationship to every use. If an
artifact is not reconstructible and linked from the ledger, it is not tracked.

### 21.1 Why SQLite is a good pilot substrate

- portable as one primary file plus hash-addressed artifacts;
- transactional and inspectable;
- supports foreign keys, recursive CTEs, FTS, JSON columns, and rigorous migrations;
- easy to snapshot, hash, publish, and query from multiple languages;
- separates the scientific dataset from chat delimiters and training serialization;
- makes every future sampling decision reproducible;
- can preserve conflicting analyses without choosing one canonical row;
- can later export candidate knowledge into Donto without making Donto the training database.

### 21.2 SQLite and Donto have different jobs

| Alpha SQLite | Donto Postgres |
|---|---|
| Experimental ledger and dataset source of truth | Live contradiction-preserving knowledge substrate |
| Stores candidate/rejected/accepted training objects and exposures | Stores claims, evidence, alignment, identity, arguments, and query state |
| Reconstructs exactly what every model saw | Answers real substrate queries under lenses |
| Optimized for portable releases and experimental lineage | Optimized for concurrent services and tens of millions of statements |
| May contain synthetic micro-worlds and hidden evaluation contracts | Must not receive hidden gold answers or test leakage |

The first integration should export only to a sealed Donto candidate context or scratch instance. No
model-generated predicate should be silently promoted into the canonical live graph.

### 21.3 Comprehensive logical table families

The ledger should ultimately represent at least the following.

#### Sources and provenance

- source;
- source version;
- document;
- fragment;
- evidence anchor;
- license and policy;
- acquisition event;
- content hash;
- author or community authority;
- valid and transaction times.

#### Conceptual structure

- concept family;
- projection;
- neighborhood;
- episode;
- message;
- transformation;
- transformation edge;
- commitment;
- expected commitment delta;
- admissible analysis set;
- false bridge;
- competency question;
- invariance or equivariance constraint.

#### Predicate birth

- predicate candidate;
- lexicalization;
- definition revision;
- argument role;
- type/signature hypothesis;
- positive instance;
- hard negative;
- non-entailment;
- exception;
- ambiguity membership;
- nearest existing predicate;
- typed alignment proposal;
- alignment scope and safety tier;
- survival event;
- retirement, split, scoped collapse, or supersession event.

#### Claims and frames

- claim;
- claim kind;
- polarity;
- modality;
- source attribution;
- temporal qualifier;
- frame;
- frame role;
- argument edge;
- evidence and counterevidence;
- hypothesis status;
- standing observations.

#### Generation and review

- provider;
- model and immutable revision;
- prompt template and revision;
- generation request;
- generation candidate;
- token usage;
- sampling settings;
- automated check;
- human review;
- adjudication;
- rejection reason;
- revision chain;
- reviewer authority and conflict-of-interest metadata.

#### Rendering and exposure

- rendering profile;
- delimiter/chat template version;
- rendered artifact;
- tokenizer and revision;
- token sequence hash;
- training example;
- sampling cohort;
- split assignment;
- curriculum position;
- model exposure;
- batch/packing membership;
- contamination or overlap finding.

#### Runs and artifacts

- experiment;
- arm;
- seed;
- run;
- checkpoint;
- optimizer and scheduler state;
- environment and hardware;
- code revision;
- metric definition and revision;
- metric observation;
- generation sample;
- failure event;
- external artifact;
- release and manifest.

### 21.4 Track token occurrences too—but at the right phase

Token occurrence lineage can eventually answer valuable questions:

- Which concept families contributed a token or span?
- How often did the model see a technical word before a jargon-free test?
- Did a supposedly held-out lexical form leak through another renderer?
- Which checkpoints were exposed to which delimiter sequence?
- How much supervision did each semantic distinction receive?

It is therefore worth supporting. But tens or hundreds of millions of token-occurrence rows need not block
the first 24-family micro-pilot. Preserve every rendered sequence and tokenizer hash immediately; then
materialize occurrence rows when an analysis requires them. That is phased computation, not phased
provenance.

### 21.5 Required reconstructibility invariants

The ledger is not comprehensive unless it can answer:

1. What exact natural-language messages did checkpoint X see?
2. Which delimiter and tokenizer turned them into which token sequence?
3. Which source fragments and analyses produced each message?
4. Which candidates were rejected, by whom, and why?
5. Which concept families and projections were wholly absent from training?
6. Which teacher, prompt, and model version produced each candidate?
7. Which competency questions motivated a predicate birth?
8. Which later queries made that predicate useful or harmful?
9. Which mappings were judged retrieval-safe but inference-unsafe?
10. Can a released dataset and every evaluation split be rebuilt byte-for-byte?

### 21.6 Current storage constraint

As of 2026-07-30, `/mnt/donto-data` is 97% full with about 32 GB free, and `/` is 93% full with about
15 GB free. This is an operational stop sign for large new corpora or checkpoints, not an argument for
discarding provenance. Before any generation or run, capacity must be reclaimed or durable artifact
storage expanded. The ledger itself will be small relative to checkpoints and token corpora, but its
artifact policy must assume storage is scarce.

---

## 22. Donto integration without contaminating the experiment

### 22.1 Shadow first

Future integration should proceed through a sealed shadow context:

1. model receives a frozen passage and permitted retrieved context;
2. model produces natural-language analysis;
3. deterministic or separately evaluated renderer creates candidate predicates and claims;
4. citer attaches honest evidence or marks the candidate unanchorable;
5. candidate output enters a scratch Donto context at hypothesis-only maturity;
6. downstream queries and reviewers test utility;
7. nothing is promoted automatically.

### 22.2 Donto as evaluator, not answer key

The existing graph contains valuable examples and failure modes, but it must not be treated as complete
ground truth. Its million predicates include historical extractor artifacts, redundant labels, and
under-described relations. Use Donto to test:

- whether a new predicate recurs across independent sources;
- whether it retrieves useful evidence;
- whether it creates or resolves an alignment ambiguity;
- whether a query needs it;
- whether it causes unsafe over-expansion;
- whether it improves contradiction diagnosis;
- whether it supports a useful next evidence question.

Do not reward a model merely for matching the current graph's most frequent label.

### 22.3 Natural language first, renderer second

Donto's current extraction interface uses compact structured facts. Alpha's research target should remain
natural-language competence first. This avoids spending scarce 58M capacity on JSON punctuation and IRI
formatting and preserves ordinary conversational evaluation. A renderer can later map:

```text
“The registrar attributes a 2022 enrolment to Lina, and the passage says that status ended yesterday.”
```

into candidate claims, time bounds, attestation edges, and predicate descriptors. Renderer fidelity is a
separate measured component.

### 22.4 Birth is not promotion

The Donto lifecycle should distinguish:

- a predicate string observed in one extraction;
- a described predicate candidate;
- a predicate with recurring supported instances;
- a predicate with reviewed scope and negative cases;
- a predicate aligned for query expansion;
- a predicate safe for export;
- a predicate safe for logical inference.

The model may help at each stage, but it cannot collapse them into one confidence score.

---

## 23. Assessment of Alpha's existing training data for this goal

### 23.1 Pretraining corpus

The archived 58M base saw one billion tokens from a globally shuffled educational/web mixture:

- 50% FinePDFs-Edu;
- 30% DCLM;
- 20% FineWeb-Edu.

This was a sensible, licensed, broad-text choice for proving the from-scratch stack. It is not evidence
that the model was pretrained deeply enough to support the new research. One billion tokens is only about
17 tokens per parameter, and the frozen base model produced severe repetition and essentially no useful
free generation. Broad educational prose also optimizes ordinary next-token prediction; it does not
explicitly teach source attribution, counterexample boundaries, mereological distinctions, predicate
alignment, or query-relative ontology choice.

The existing pretraining data is therefore **not bad**, but it is **insufficient and not targeted to this
capability**. It should remain a baseline. A future experiment should compare continued pretraining on
conceptual neighborhoods against an equal-token sample from the same broad mixture.

### 23.2 SFT corpus

The final SFT corpus contained 511,428 structurally validated English conversations from Smol-SmolTalk,
SmolTalk2, OASST2, and a small SODA component. It was hash-pinned, length-audited, assistant-mask-audited,
and decontaminated against frozen evaluation. Those are genuine strengths.

But it was not a good causal intervention for chat initiation or the proposed knowledge behavior:

- it was consumed in long source-grouped blocks without shuffling;
- long assistant answers dominated token-averaged loss;
- the first answer token received negligible weight relative to continuation tokens;
- 203,074 conversations needed prefix trimming to fit 1,024 tokens;
- the mixture was broad and heterogeneous rather than organized around a small number of semantic
  distinctions;
- no relational neighborhood structure was visible to the model;
- free generation oscillated independently of held-out teacher-forced loss.

The corpus may contain many good individual conversations. The failed result does not prove those rows
are linguistically poor. It proves that **corpus quality, ordering, weighting, model capacity, training
stage, and evaluation cannot be collapsed into one word: “good.”**

### 23.3 What should be retained

Retain from the existing program:

- the byte-level tokenizer and standard Llama export;
- the full hash and exposure provenance;
- assistant-only masking implementation;
- frozen free-generation gates;
- native and Transformers parity;
- checkpoint archives;
- the broad pretraining mixture as a control;
- the current model as a negative baseline.

Do not assume the terminal SFT checkpoint is the best initialization for conceptual research. Compare at
least the base checkpoint, any repaired chat checkpoint, and a larger positive control under a frozen
protocol.

---

## 24. How to spend future RunPod compute scientifically

No GPU should be provisioned merely because this document exists. Once the research design, evaluation,
storage, and authorization gates are satisfied, compute should be released in escalating stages.

### 24.1 Stage R0 — zero GPU

- complete the prior-art review;
- secure the storage margin;
- define 24 concept families;
- formalize the tractable micro-world subset;
- build and review the frozen evaluation first;
- specify all arms and stop rules;
- create the comprehensive SQLite data dictionary;
- test leakage and reconstruction on tiny fixtures;
- obtain linguistic or community review where needed.

### 24.2 Stage R1 — answer-initiation repair

Use the smallest run that can distinguish SFT recipes. Compare shuffling, source balancing, episode
normalization, short-answer coverage, and explicit first-token weighting. Stop unless the nonempty and
loop gates pass.

### 24.3 Stage R2 — concept-boundary smoke test

Use perhaps 8–12 families to determine whether the model can learn positive/hard-negative boundaries at
all. This is not a novelty experiment; it prevents an expensive null result caused by basic incapacity.

### 24.4 Stage R3 — decisive 24–60-family relational-visibility experiment

Run the equal-token arms B, D, E, and corrupted F first. Add A and C if budget permits or if they answer a
specific causal question. Use multiple seeds and a 150M–300M positive control for the strongest contrast.

### 24.5 Stage R4 — scale to 300 families only on transfer evidence

Scale when the correct-relation condition beats independent and corrupted controls on:

- whole-family transfer;
- false-bridge rejection;
- non-equivalence;
- query utility;
- ordinary conversational behavior.

Do not scale because the generator can produce rows quickly.

### 24.6 Stage R5 — Donto shadow integration

Run on frozen, policy-eligible Donto passages in a scratch context. Measure survival against real query
behavior, not just curated tests.

### 24.7 Stage R6 — search-native and STEER experiments

Only after source-contained deconstruction works should the model control retrieval, propose evidence
requests, or decide which aperture to run next.

### 24.8 Compute accounting

The unit economics should report:

- GPU-hours per accepted family;
- training tokens per latent distinction;
- performance gain per unique family;
- performance gain per human-review hour;
- marginal value of each additional projection;
- marginal value of hard negatives and composed interventions;
- cost per predicate that survives a downstream query test;
- cost per promoted claim or safe alignment—not raw generated row count.

This is a better use of a 3090 than another undifferentiated half-billion-token SFT epoch.

---

## 25. Formal and executable subset

Not every philosophical or linguistic interpretation can be solver-verified. A selected core can.

### 25.1 Candidate primitives

- entity persistence;
- role acquisition and termination;
- membership change;
- component replacement;
- source attribution;
- valid and record time;
- explicit positive, negative, and unknown polarity;
- evidence support and withdrawal;
- necessary and sufficient conditions;
- part/member/portion distinctions;
- finite ambiguity branches;
- coarse/fine query loss.

### 25.2 What the checker should verify

- a proposed delta does not alter declared invariants;
- positive examples satisfy the contract;
- hard negatives fail for the intended reason;
- composition order commutes when specified;
- non-commuting transformations remain distinguishable;
- an admissible set is nonempty;
- a predicate definition does not entail a forbidden consequence;
- a query can or cannot be answered from the representation as declared.

### 25.3 What remains human-adjudicated

- literary interpretation;
- culturally dependent pragmatic force;
- theory-relative ontology design;
- social-category boundaries;
- contested identity criteria;
- whether a relation is illuminating rather than merely technically consistent;
- natural conversational quality.

Using a solver is not itself novel—ADVENT and NormWorlds-CF already make that clear. Its role is to
provide hard oracles for the subset where formal claims are possible.

---

## 26. Mechanistic opportunity at 58M

The small model is valuable because its internal representations are tractable enough to study.

After a behavioral result, ask whether the same internal feature supports:

- temporary roles in institutional records and *former student*;
- member/collection distinctions and collective agreement;
- event boundaries in aspect and event reconciliation;
- source attribution in reported speech and catalogue provenance;
- absence/negation in language and open-world database queries.

A serious mechanistic claim requires more than a linear probe:

1. train a probe on one projection;
2. test it on a lexically isolated projection;
3. identify a stable direction or subspace across seeds;
4. intervene on that representation during inference;
5. observe the predicted behavior change in both projections;
6. verify that unrelated commitments remain stable;
7. compare against random directions and false-bridge families.

Possible outcomes are all informative:

- shared causal feature across projections;
- probe transfer without causal effect;
- vocabulary-specific features only;
- behavioral transfer only in the larger control;
- no transfer at either scale.

Do not claim “the model has an ontology neuron” from a visually attractive activation plot.

---

## 27. Failure modes and what each would mean

| Failure | Plausible interpretation | Required response |
|---|---|---|
| Immediate EOS returns | P0 still unsolved | Stop conceptual interpretation; repair initiation |
| Independent and correct-linked arms tie | Relational structure did not help at this scale/objective | Do not scale neighborhoods on faith |
| Correct and corrupted links tie | Formatting, packing, or rehearsal explains gains | Reject the semantic-relation claim |
| Technical-jargon tests pass; jargon-free tests fail | Terminology memorization | Strengthen lexical isolation; no abstraction claim |
| True and false bridges transfer equally | Topical analogy rather than relation transport | Redesign family and negative controls |
| Positive cases pass; hard negatives fail | Overbroad predicate | Add counterexample pressure; report poor boundary quality |
| Many one-off labels appear | Predicate fragmentation | Increase reuse pressure without forcing collapse |
| Everything maps to a few generic predicates | Over-collapse | Reward distinction coverage and no-match decisions |
| Definitions sound good but queries do not improve | Eloquence without utility | Fail survival; keep candidate unpromoted |
| Query lift improves but evidence fidelity drops | Retrieval shortcut or hallucinated relation | Require source and anchor gate |
| Ambiguity recall is high but precision low | Indiscriminate hedging | Penalize unsupported analyses |
| Larger control succeeds; 58M fails | Capacity or foundation threshold | Valuable capacity result; reconsider model scale |
| Both scales fail | Object, data, objective, or hypothesis may be wrong | Audit specification before more compute |
| Model writes valid structure but cannot converse | Semantic parser, not chat model | Report separately; do not claim conversational competence |
| Conversation improves without transfer | P0/SFT success only | Do not claim predicate learning |
| Donto utility improves only on seen queries | Query-set overfitting | Expand held-out competency questions |
| Human reviewers disagree systematically | Genuine plurality or bad specification | Preserve disagreement; do not average it away |

Negative results should remain publishable. A clean finding that 58M learns local relation labels but no
cross-domain abstraction is more valuable than a vague claim after scaling until one metric moves.

---

## 28. What not to do

- Do not train another full epoch of the same ordered SFT mixture.
- Do not use teacher-forced loss as the primary checkpoint selector.
- Do not call a plausible relation name predicate understanding.
- Do not use the current Donto predicate inventory as an unquestioned gold ontology.
- Do not force every candidate into an existing relation.
- Do not equate close match with exact equivalence.
- Do not allow query-safe alignment to imply inference safety.
- Do not make all model-visible output JSON or ontology syntax in the first experiment.
- Do not evaluate open predicates by exact label string.
- Do not randomly split episodes from the same concept family across train and test.
- Do not let one teacher family generate, verify, and judge the decisive result.
- Do not use model judges as final authority on philosophical counterexamples.
- Do not call synthetic row volume conceptual diversity.
- Do not make the SQLite implementation the headline learning contribution.
- Do not discard raw candidates, failed analyses, or negative runs.
- Do not write experimental output directly into canonical Donto contexts.
- Do not rebrand the program as semantic holonomy without resolving the local prior-art audit.
- Do not provision RunPod before storage, eval, and authorization gates pass.
- Do not claim “world first” from the present bounded search.

---

## 29. Possible paper ladder

This program can produce stepping stones rather than one all-or-nothing flagship.

### Paper 0 — The response-initiation phase transition

**Claim:** token-averaged teacher-forced SFT can hide first-assistant-token failure in very small models.

**Evidence needed:** controlled weighting, answer-length, shuffling, and free-generation ablations across
seeds. The current failed run is motivating evidence, not sufficient causal proof.

### Paper 1 — When is a relation worth naming?

**Claim:** predicate birth contracts learned from contrast families outperform independent targeted
examples on hard negatives and cross-projection transfer.

**Evidence needed:** correct versus corrupted relational visibility, whole-family holds, false bridges,
larger control.

### Paper 2 — Predicate survival in an open epistemic substrate

**Claim:** query utility and typed non-collapse can select useful new predicates without imposing one
write-time ontology.

**Evidence needed:** Donto shadow contexts, held-out competency questions, retrieval and loss reports.

### Paper 3 — A small epistemic compiler

**Claim:** a fact-light model can convert retrieved passages into source-attributed, temporally qualified,
ambiguity-preserving claim structures.

**Evidence needed:** source fidelity, frame quality, renderer fidelity, Donto query utility.

### Paper 4 — Learning what evidence to seek

**Claim:** the model chooses searches or observations that reduce live epistemic uncertainty better than
generic retrieval.

**Evidence needed:** executable environments, information-gain outcomes, cost-aware baselines.

### Paper 5 — Mechanisms of cross-domain semantic transport

**Claim:** shared causal internal features support the same distinction in linguistic and ontological
projections.

**Evidence needed:** behavioral transfer, cross-projection probes, causal interventions, seed replication.

---

## 30. Research-agent work packages

Third-party agents should attack the proposal rather than merely decorate it.

### A0 — Novelty and closest-work search

- Search beyond the sources in this brief.
- Find direct precedents for source-grounded predicate birth, relation-contract induction, query-tested
  predicate survival, and sub-100M cross-domain transfer.
- State which claim each precedent removes.
- Separate peer-reviewed work from unreviewed preprints.
- Report search terms, databases, dates, and negative search findings.

### A1 — Formal object

- Critique the Predicate Birth Neighborhood.
- Decide whether it is best modeled as a transition system, algebra, program synthesis task, concept
  lattice, or another object.
- Specify the smallest executable core.
- Identify any misuse of “predicate,” “relation,” “concept,” “property,” “role,” or “ontology.”

### A2 — Mereology

- Design families covering component, member, portion, material, place, phase, boundary, function, and
  constitution.
- Identify false transitivity traps.
- Propose cross-linguistic cases and authority requirements.
- Find existing datasets and licenses.

### A3 — Semantics and pragmatics

- Design source-contained tests for assertion, presupposition, implicature, deixis, reference,
  illocution, commitment, and repair.
- Separate linguistically defensible plurality from annotator uncertainty.
- Prevent mind-reading and English-centric assumptions.

### A4 — Ontology and knowledge representation

- Critique the predicate contract and alignment types.
- Specify competency-question evaluation.
- Identify where n-ary frames are mandatory.
- Design query-relative adequacy and information-loss tests.

### A5 — Small-model learning

- Estimate which tasks are plausible at 58M, 150M, and 300M.
- Recommend continued-pretraining/SFT objectives.
- Design answer-initiation controls.
- Identify token budgets and capacity confounds.

### A6 — Synthetic curriculum

- Design constructor, adversary, critic, and reviewer prompts.
- Propose teacher-isolation and template-isolation splits.
- Estimate candidate-to-accepted ratios.
- Define semantic duplicate detection without brittle string lists.

### A7 — Evaluation and statistics

- Formalize survival metrics.
- Define executable and human oracles.
- Specify hierarchical analysis and power at 24–60 families.
- Design corrupted-relation, false-bridge, and jargon controls.

### A8 — SQLite scientific ledger

- Produce a comprehensive logical data model.
- Keep delimiters and renderers separate from semantic content.
- Specify immutable IDs, revision chains, hashes, and reconstructibility queries.
- Design phased materialization without losing any lineage.
- Estimate storage for candidates, renders, token occurrences, and checkpoints.

### A9 — Donto integration

- Map predicate contracts to existing Donto tables and identify genuine schema gaps.
- Design shadow-context evaluation.
- Define query-utility experiments and promotion boundaries.
- Ensure policies and evidence anchoring propagate.

### A10 — Mechanistic analysis

- Predeclare probe and intervention methods.
- Identify appropriate random and false-bridge controls.
- Avoid post-hoc neuron storytelling.

### A11 — Philosophy and research ethics

- Critique the program's epistemological assumptions.
- Identify where formal verification is inappropriate.
- Address cultural authority, plural ontologies, and social-category harms.
- Define what “intelligence about knowledge” does and does not mean.

---

## 31. Required return format for research agents

Each external assessment should include:

1. **Bottom line:** proceed, revise, or abandon—and why.
2. **Closest prior work:** direct links and exact collision.
3. **Strongest surviving claim:** one falsifiable sentence.
4. **Fatal confounds:** what would make a positive result uninterpretable.
5. **Decisive experiment:** smallest design that resolves the key uncertainty.
6. **Negative result:** what outcome should stop scaling.
7. **Data implications:** what must be represented in SQLite.
8. **Donto implications:** which live substrate operation benefits.
9. **Capacity implications:** what result at 58M versus 300M would mean.
10. **Corrections to this brief:** cite section numbers and replacement text where possible.

Agents should not assume the recommended direction is correct. A well-supported argument for the
Epistemic Compiler, STEER, mereology, or another option is welcome.

---

## 32. Why this program is fascinating even if Alpha fails

The underlying fascination is not “can a tiny model answer trivia?” It is that language continually
creates candidate ontologies before anyone writes a schema:

- a speaker turns an activity into an object through nominalization;
- *former* reveals which categories behave like roles;
- a plural noun can denote members, a collective, or an institution;
- a report can carry a claim without endorsing it;
- a tense shift changes current truth while preserving historical truth;
- a metaphor proposes a new structural resemblance;
- a correction withdraws one commitment without deleting a person or event;
- a question reveals which distinctions a representation must preserve;
- disagreement can be about facts, vocabulary, theory, scale, evidence, or purpose.

Donto is built to retain this expanding possibility space. Alpha can test whether the operations that
create and discipline that space are learnable at small scale. A model that knows few facts but can
reliably say:

> “These sources do not yet disagree; they use different time windows,”

or:

> “This relation is useful for retrieval, but treating it as exact would make a false inference,”

or:

> “Your paragraph supports three attributed commitments, one presupposition, and two admissible readings;
> the following search would distinguish them,”

could be scientifically and practically valuable even if it cannot name a president or write code. It
would use search and Donto for changing world facts while devoting its limited parameters to the grammar
of evidence, distinction, disagreement, and inquiry.

That division of labor is not guaranteed to work. Finding the smallest scale at which it begins—or
showing that these operations do not separate cleanly from factual language modeling—is itself a serious
result.

---

## 33. Open decisions before any data generation

1. Is Predicate Birth and Survival a stronger headline than Alpha Joints, or should it be a task inside
   Alpha Joints?
2. Which exact difference from ADVENT, AutoSchemaKG, ScheMatiQ, and OntoLearner is both novel and
   measurable?
3. Is *predicate* the right unit, or should the program center relations, frames, commitment operators,
   or question-conditioned representation fragments?
4. What formal language captures the executable subset without pretending to formalize all
   interpretation?
5. Which 24 concept families have the clearest contracts and strongest Donto relevance?
6. Which three projections per family are genuinely structural rather than author-imposed analogies?
7. What is the minimum false-bridge set capable of detecting topical transfer?
8. How will admissible predicate sets be adjudicated when reviewers disagree?
9. What counts as predicate reuse rather than paraphrase?
10. What downstream Donto query establishes usefulness without circularly rewarding the intended
    annotation?
11. How should fragmentation and over-collapse be traded without creating one arbitrary scalar?
12. Which existing Donto predicates are safe and policy-eligible as observational test material?
13. Which Donto data are extractor artifacts that should never become supervision?
14. Should the first student start from the base checkpoint, a newly repaired chat checkpoint, or a new
    small pretrained model?
15. Is 58M sufficient for the basic boundary task? What evidence would justify 150M or 300M as the main
    student rather than only a control?
16. What P0 initiation intervention is least entangled with conceptual content?
17. Should predicate birth be learned through continued pretraining, SFT, pairwise preference, an
    auxiliary head, or a staged combination?
18. How many seeds and families provide meaningful uncertainty estimates within the compute budget?
19. Which teacher models can be used within budget and licensing constraints?
20. How will teacher, template, source, and reviewer isolation be enforced?
21. Which languages and communities can be included responsibly in the first pilot?
22. What model-visible content remains natural language, and what is hidden evaluator metadata?
23. Which SQLite tables must be materialized for the micro-pilot, and which can initially reference
    immutable artifacts?
24. Where will checkpoint and corpus artifacts live given current disk pressure?
25. What negative result closes the direction rather than merely prompting a larger run?

---

## 34. Definition of research readiness

No paid model run is research-ready until all of the following exist:

- a frozen one-sentence primary claim;
- an explicit distinction from the closest prior work;
- a dated, primary-source bibliography and search log;
- 24 or more reviewed concept-family specifications;
- accepted definitions of predicate birth, reuse, survival, fragmentation, and over-collapse;
- a frozen final evaluation built before training data expansion;
- whole-family, cross-projection, lexical, teacher, template, and source splits;
- hard negatives, false bridges, and corrupted-relation controls;
- a separate P0 response-initiation protocol;
- equal-token experimental arms;
- a larger positive-control plan;
- predeclared checkpoint selection and stopping rules;
- executable or human adjudication rules for every primary metric;
- a statistical analysis plan at family level;
- a comprehensive SQLite data dictionary and reconstructibility tests;
- storage capacity for all artifacts plus safe working margin;
- a shadow-only Donto integration contract;
- explicit authorization for the exact RunPod spend.

---

## 35. Definition of pilot success

The strongest arm must, across predeclared seeds:

1. pass the independent response-initiation gate;
2. beat independent targeted episodes at equal tokens;
3. beat the corrupted-relation control;
4. improve on whole-family and cross-projection holds;
5. reject false bridges rather than merely repeat a semantic theme;
6. retain hard-negative precision;
7. preserve admissible plurality without overhedging;
8. type predicate relationships more accurately than a similarity baseline;
9. improve at least one held-out competency-question or retrieval task;
10. retain ordinary conversational behavior;
11. survive jargon ablation;
12. preserve every run, output, failure, and review in the SQLite ledger.

A model that merely emits more relations has not succeeded. A model that becomes chatty has passed P0.
A model that produces elegant definitions but no transfer has learned local rhetoric. A model that
improves retrieval by flattening distinctions has violated the Donto thesis.

---

## 36. Local evidence map

These files were used to understand the project. Future researchers should read them before treating
this agenda as canonical.

| Local file | Why it matters |
|---|---|
| `/mnt/donto-data/donto-resources/vision/DONTO-CANON.md` | Canonical HOLD/JUDGE/STEER vision and trail-centered north star |
| `/mnt/donto-data/donto-resources/vision/DONTO-ABUNDANCE.md` | Emit-free, defer-joining thesis and relationship-generating apertures |
| `/mnt/donto-data/workspace/donto/docs/PLANS.md` | Canonicity and current planning index |
| `/mnt/donto-data/workspace/donto/docs/DONTO-PRD.md` | Ten invariants, fourteen object families, language pilot, predicate and alignment contract |
| `/mnt/donto-data/workspace/donto/docs/DONTO-CALCULUS.md` | Formal statement/evidence/time/alignment machinery and explicit gaps |
| `/mnt/donto-data/workspace/donto/docs/DONTO-LENS-SPEC.md` | Query-relative identity, predicate folding, time, contradiction, source, and export stance |
| `/mnt/donto-data/workspace/donto/docs/EXTRACTION-MAXIMALISM.md` | Linguistic, pragmatic, counterfactual, and interpretive deconstruction agenda |
| `/mnt/donto-data/workspace/donto/apps/donto-agent/prompts/extract_broad.txt` | Current free-predicate extraction apertures and source-faithfulness posture |
| `/mnt/donto-data/workspace/donto/docs/sheaf-prd/PRD-00-master.md` | Local-to-global consistency and obstruction program |
| `/mnt/donto-data/workspace/semholo/research/RELATED-WORK.md` | Prior-art warning against broad semantic-holonomy claims |
| `/mnt/donto-data/workspace/semholo/docs/CLAIM-REGISTER.md` | Narrowed local novelty adjudication |
| `/mnt/donto-data/workspace/semholo/docs/PILOT-FINDINGS.md` | Evidence that shared response schemas do not automatically create shared semantic coordinates |
| `/mnt/donto-data/workspace/donto-web/apps/home/content/reports/donto-cameron-winter-deconstruction-2026-06-04.md` | Concrete example of plural, hypothesis-only interpretive predicate minting |
| `/mnt/donto-data/workspace/alpha2/GOAL.md` | Complete archived model program and execution evidence |
| `/mnt/donto-data/workspace/alpha2/docs/resume/FAILURE-ANALYSIS.md` | Evidence-backed initiation/EOS diagnosis |
| `/mnt/donto-data/workspace/alpha2/docs/ALPHA-JOINTS-RESEARCH-PROGRAM.md` | Existing controlled transformation and SQLite research design |

The live row counts in Section 2.3 are a 2026-07-30 read-only snapshot and will drift.

---

## 37. Initial primary-source bibliography

### Open extraction, relation discovery, and predicate invention

- Hohenecker et al. [Systematic Comparison of Neural Architectures and Training Approaches for Open
  Information Extraction](https://aclanthology.org/2020.emnlp-main.690/), EMNLP 2020.
- Pei, Jindal, and Chang. [Abstractive Open Information Extraction](https://aclanthology.org/2023.emnlp-main.376/),
  EMNLP 2023.
- Yu, Huang, and Ji. [Open Relation Extraction and Grounding](https://aclanthology.org/I17-1086/),
  IJCNLP 2017.
- Radevski et al. [Linking Surface Facts to Large-Scale Knowledge Graphs](https://aclanthology.org/2023.emnlp-main.445/),
  EMNLP 2023.
- Hogan, Li, and Shang. [Open-world Semi-supervised Generalized Relation Discovery Aligned in a
  Real-world Setting](https://arxiv.org/abs/2305.13533), 2023.
- Yu et al. [ADVENT: LLM-Driven Automatic Predicate Invention for ILP](https://arxiv.org/abs/2607.01585),
  arXiv 2026.

### Ontology and schema induction

- Giglou, D'Souza, and Auer. [LLMs4OL: Large Language Models for Ontology Learning](https://arxiv.org/abs/2307.16648),
  2023.
- Giglou et al. [OntoLearner: A Modular Python Library for Ontology Learning with Large Language
  Models](https://arxiv.org/abs/2607.01977), arXiv 2026.
- Xu, Zhang, and Chen. [CEO: Corpus-based Open-Domain Event Ontology Induction](https://arxiv.org/abs/2305.13521),
  2023.
- Sergienko. [Generative Ontology Induction](https://arxiv.org/abs/2607.16201), arXiv 2026.
- Bai et al. [AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction
  from Web-Scale Corpora](https://aclanthology.org/2026.acl-long.942/), ACL 2026.
- Levy et al. [ScheMatiQ: From Research Question to Structured Data through Interactive Schema
  Discovery](https://aclanthology.org/2026.acl-demo.22/), ACL 2026.
- Mahlaza et al. [On the Feasibility of LLM-based Automated Generation and Filtering of Competency
  Questions for Ontologies](https://aclanthology.org/2025.ldk-1.15/), LDK 2025.

### Linguistic and epistemic diagnostics

- Sravanthi et al. [PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics
  Capabilities](https://aclanthology.org/2024.findings-acl.719/), ACL Findings 2024.
- Gu, Dalvi Mishra, and Clark. [Do language models have coherent mental models of everyday
  things?](https://aclanthology.org/2023.acl-long.106/), ACL 2023.
- Roy Dipta and Ferraro. [If We May De-Presuppose: Robustly Verifying Claims through
  Presupposition-Free Question Decomposition](https://aclanthology.org/2025.starsem-1.20/), *SEM 2025.
- Wilie et al. [Belief Revision: The Adaptability of Large Language Models
  Reasoning](https://aclanthology.org/2024.emnlp-main.586/), EMNLP 2024.

### Externalized knowledge and small-model reasoning

- Kang et al. [Knowledge-Augmented Reasoning Distillation for Small Language Models in
  Knowledge-Intensive Tasks](https://arxiv.org/abs/2305.18395), 2023.
- Wang et al. [RARE: Retrieval-Augmented Reasoning Modeling](https://arxiv.org/abs/2503.23513), 2025.

This is a targeted opening review, not a systematic review. It should be expanded before any public
novelty claim.

---

## 38. Working conclusion

The current direction is **near the right intellectual territory but not yet the sharpest experiment**.
Alpha Joints correctly identifies controlled transformations, invariants, localized revision, legitimate
plurality, and cross-domain transfer as the mechanisms that distinguish conceptual learning from a pile of
good answers. Donto reveals what those mechanisms should be used for: not simply answering ontology
questions, but creating and disciplining a living vocabulary of evidence-bound relations.

My strongest present recommendation is:

> **Use Alpha Joints to build Predicate Birth Neighborhoods, and test whether a fact-light small model can
> invent relations whose boundaries survive counterexamples, false analogies, new domains, and real
> Donto questions.**

Begin with mereology, roles and identity, evidence and attribution, time, and absence versus negation.
They are philosophically rich, linguistically visible, operationally important to Donto, and more
tractable than unrestricted intent or literary interpretation. Add intent as a governed projection, not as
the sole program. Keep world facts in retrieved sources. Keep every candidate and failure in SQLite.
Keep model-visible content natural. Treat Donto's million predicates as an ecology to study, not a schema
to memorize.

The most exciting eventual system is not a tiny oracle. It is a small **epistemic specialist** that can
read retrieved evidence, notice distinctions, mint provisional relations, preserve disagreement, explain
losses, and ask the next useful question—while Donto remembers the world and everything the model has
ever claimed about it.

That is worth investigating. It is also narrow enough to fail honestly.
