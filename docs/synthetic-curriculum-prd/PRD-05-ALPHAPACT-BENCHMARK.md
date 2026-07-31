# PRD-05 — AlphaPact: inferential conceptual pacts in conversation

## 1. Purpose

AlphaPact is the central behavioral construct and frozen evaluation program for Alpha's specialized
conversational intelligence. It tests whether a model can co-construct a temporary, purpose-sensitive
conceptual system with a person and then use, challenge, revise, suspend, or transfer it appropriately.

The benchmark is not a vocabulary quiz and not merely a reference game. A model receives credit only when a
local meaning or representational choice changes its downstream inferences in the right way.

## 2. Research lineage and surviving contribution

Psycholinguistic “conceptual pacts” conventionally describe temporary partner-specific agreements about how to
conceptualize and refer to an object. Small dynamically constructed language models have already been used to
model such reference-resolution pacts ([LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.327/)).
Post-training has also improved convention formation and communication efficiency
([Hua, Wang, and Artzi](https://arxiv.org/abs/2508.06482)), while recent human–AI work shows that humans and
models can each form conventions within same-type pairs yet fail to coordinate as well with each other
([Jones et al.](https://arxiv.org/abs/2602.08208)).

Common-ground tracking, public commitments, and Questions Under Discussion are established research objects
([common-ground survey](https://aclanthology.org/2025.luhme-1.2/),
[public commitments](https://aclanthology.org/W15-0131/),
[common-ground tracking](https://aclanthology.org/2024.lrec-main.318/)).

Alpha therefore does not claim to invent pacts, common ground, or conventions. Its narrower target is an
**inferential conceptual pact**: a locally negotiated interpretation whose licensed and prohibited
consequences, scope, dependencies, live alternatives, and revision behavior can be tested in natural
conversation and transported into a different realization.

## 3. Definition

An inferential conceptual pact (ICP) is a public, purpose-bounded conversational arrangement concerning how a
term, distinction, analogy, category, evidential standard, or representation will function in the current
inquiry.

An evaluator-side pact state at turn `t` is conceptually:

`P_t = <term, scope, purpose, licensed inferences, prohibited inferences, admissible analyses, unresolved alternatives, dependencies>`

It sits beside separate public commitment stores:

- user commitments;
- Alpha commitments;
- shared commitments;
- attributed third-party/source commitments;
- denials;
- Questions Under Discussion.

The formal representation is hidden from Alpha. It is not JSON the model must emit.

## 4. Behavioral criterion

Alpha has formed the pact only if it can:

1. acknowledge or negotiate the proposed local rule naturally;
2. apply it to a new positive case;
3. reject an attractive inference that the pact prohibits;
4. preserve it after intervening turns;
5. distinguish an actual challenge from an unrelated question;
6. revise commitments that depend on a changed rule;
7. preserve independently supported commitments;
8. notice purpose or scope change;
9. keep unresolved alternatives live when no agreement was reached;
10. recover the pact following interruption;
11. use the pact more efficiently as common ground accumulates;
12. transfer the underlying distinction into an unfamiliar domain;
13. reject a superficially similar but structurally different case.

Repeating the agreed definition, using the agreed nickname, or stating the technical category does not by
itself pass.

## 5. Pact classes

### 5.1 Local lexical definition

“By *witness* here, let us mean an independently originating source, not every document that repeats the
claim.”

### 5.2 Category-boundary agreement

“For this inventory, count detachable power supplies as components but not packaging.”

### 5.3 Evidential standard

“Use *evidence* for observations attributable to a source; call later conclusions *interpretations*.”

### 5.4 Ontological modeling choice

“For this registry, model student as a time-qualified role, not an essential kind of person.”

### 5.5 Analogy mapping

“Treat document versions like a branching lineage for this discussion, but do not infer biological inheritance.”

### 5.6 Normative/evaluative criterion

“Call a process fair here only if affected parties had a meaningful opportunity to contest it.”

### 5.7 Temporal convention

“Present status means valid on the event date, not what the database currently records.”

### 5.8 Granularity convention

“At this stage treat each committee as one decision-making unit, unless internal dissent is relevant.”

### 5.9 Source and attribution convention

“A claim belongs to the diarist unless the diarist explicitly quotes another speaker.”

### 5.10 Metalinguistic proposal

“Let's reserve *intent* for a plan supported by behavior and use *effect* for what actually happened.”

## 6. Pact lifecycle

Each family may realize some or all of:

1. **Need:** a confusion or purpose creates pressure for a distinction.
2. **Proposal:** user or Alpha proposes a local interpretation.
3. **Negotiation:** the other participant accepts, narrows, rejects, or offers an alternative.
4. **Adoption:** a working pact enters shared ground.
5. **Immediate application:** simple consequence.
6. **Delayed application:** consequence after distractors.
7. **Hard negative:** tempting prohibited consequence.
8. **Challenge:** case pressures the boundary.
9. **Repair or refusal:** pact changes locally or survives.
10. **Scope shift:** purpose/time/speaker changes.
11. **Recovery:** pact is retrieved after interruption.
12. **Transfer:** structurally analogous new domain.
13. **False bridge:** similar surface, different structure.
14. **Closure:** summarize, leave alternatives open, or answer and stop.

## 7. Example family: evidence versus interpretation

### Base

User: “For this investigation, can we use *evidence* only for observations tied to a source and call the later
conclusions *interpretations*?”

Expected Alpha behavior: accept the operational distinction, note its purpose, and avoid declaring it the only
valid use of *evidence*.

### Application

The diary says, “I saw Thomas slam the door.” A historian says Thomas was furious.

Required consequences:

- the diary is evidence that the diarist reported seeing the action;
- anger is an interpretation unless additional observation supports it;
- a diary report is not automatically proof the event occurred exactly as described.

### Prohibited consequence

Do not say the historian's claim is direct observation merely because it appears in a scholarly source.

### Revision

The participants refine the pact into `direct record`, `reported observation`, and `later interpretation`.
Only earlier claims affected by the finer distinction should be relabeled.

### Invariant

An independently established publication date should not change.

### Transfer

Apply the structure to a news article quoting a witness and an editor's headline.

### False bridge

A thermometer reading is also mediated by an instrument, but the relevant source/dependence structure differs
from copied testimony. Alpha must not force the same categories without examining the purpose.

## 8. Family construction

Each AlphaPact family defines:

- purpose/QUD;
- participants and asymmetries;
- initial commitments and denials;
- local term or representation;
- scope and validity interval;
- licensed and prohibited inferences;
- dependencies;
- admissible alternatives;
- pact lifecycle transitions;
- must-change probes;
- must-not-change probes;
- still-unresolved probes;
- out-of-scope probes;
- requires-evidence probes;
- false-bridge probes;
- terminology and scenario holdouts;
- human-adjudication needs.

The hidden contract may be executable for fictional micro-worlds and set-valued/expert-adjudicated for open
philosophical cases.

## 9. Benchmark composition

The initial instrument should contain enough independent families to estimate family-level performance rather
than inflate certainty with turns. A planning target is 120–200 families across at least eight pact classes
and several domain projections. The precise number remains gated by human agreement and review capacity, not a
production quota.

The benchmark contains:

- ordinary easy cases to establish face validity;
- difficult but adjudicable cases;
- theory-relative sets;
- source-conditioned cases;
- natural multi-turn trajectories;
- counterfactual branches from shared prefixes;
- terminology-scrubbed variants;
- cross-domain transfer;
- false bridges;
- real-human interaction subset when authorized.

## 10. Splitting and leakage control

Split at the highest dependency level:

- latent contract family;
- projection family;
- scenario constructor/template;
- teacher model/prompt family;
- source family;
- lexical cluster;
- counterexample method;
- human author where relevant.

Private evaluation receives entire families and their semantic relatives. A paraphrase of a private item is a
leak even if its row ID is new.

## 11. Evaluation modes

### 11.1 Single-turn probes

Fast diagnosis of application, non-entailment, ambiguity, and response policy.

### 11.2 Stateful multi-turn conversation

Measures adoption, delay, drift, repair, scope, recovery, and natural interaction.

### 11.3 Branch comparison

From the same prefix, compare correct and conceptually corrupted continuations.

### 11.4 Cross-projection transfer

Learn or establish a distinction in one domain, then test it in a lexically isolated domain.

### 11.5 Human–Alpha interaction

Participants negotiate a concept and pursue a real question. Required for product validity; synthetic users
alone are inadequate.

### 11.6 Long-context survival

Measure when the pact is lost, repeated, or distorted over sustained conversation. If native context is too
short, a separately evaluated conversational state/memory mechanism may be introduced; results must identify
which system maintained the pact.

## 12. Primary measurements

### 12.1 Pact adoption

Does the next relevant answer follow the accepted local interpretation rather than a default meaning?

### 12.2 Inferential consequence accuracy

Required commitments derived and prohibited commitments avoided.

### 12.3 Revision locality

`1 - (unaffected commitments incorrectly changed / unaffected commitments tested)`

Also report missed dependent revisions. A model that changes nothing is stable but unresponsive.

### 12.4 Pact drift

Rate at which Alpha silently reverts, broadens, narrows, or substitutes the pact without a legitimate update.

### 12.5 Scope sensitivity

Correctly distinguishes local task use, speaker use, historical use, and universal claims.

### 12.6 Alternative-set precision and recall

Penalizes both collapse and indiscriminate possibility listing.

### 12.7 Recovery

Correct reuse after distractors or interruption without excessive restatement.

### 12.8 Cross-projection transport

Applies the relevant inferential delta in a new domain without relying on held-out terminology.

### 12.9 False-bridge rejection

Refuses superficially tempting transfer when the dependency structure differs.

### 12.10 Conversational efficiency

Conditional on correctness, does Alpha become appropriately more economical as common ground accumulates?

### 12.11 Interaction quality

Directness, contingency, momentum, adaptation, presence, length control, question necessity, and desire to
continue.

## 13. Aggregate score policy

There is no single “AlphaPact score” for scientific adjudication. Report a profile with family-clustered
uncertainty. Primary outcomes are predeclared for each study, with revision locality recommended for the first
relational-curriculum test.

No model receives conceptual credit for:

- repeating technical vocabulary;
- restating the pact;
- merely being concise;
- producing a longer nuanced answer;
- asking for clarification when the intended reading is clear;
- listing every possible interpretation;
- satisfying a model judge without human-calibrated evidence.

## 14. Baselines

Before curriculum training, evaluate:

- humans on an adjudicated subset;
- the archived Alpha baseline as a failure reference where technically possible;
- one already conversational small model as an instrument-validity baseline;
- strong current models as approximate ceilings and failure sources;
- lexical/nearest-neighbor heuristics on controlled probes;
- a symbolic oracle on executable fictional families.

The purpose is to validate the construct and reveal shortcuts before spending GPU time.

## 15. Human study protocol outline

Subject to a later consent/ethics decision:

- participants receive a discussion goal, not hidden rubric language;
- they may propose or negotiate terminology naturally;
- conversations are long enough for adoption and reuse;
- blind pairwise comparisons control order and response length;
- post-conversation questions test shared understanding and willingness to continue;
- personal/sensitive content is excluded or separately governed;
- raw human dialogue is not automatically placed in training;
- human turns used for evaluation remain private if promised;
- synthetic and human results are clearly distinguished.

## 16. Failure signatures

- **Definition echo:** repeats rule but cannot apply it.
- **Default reversion:** returns to public meaning after delay.
- **Pact absolutism:** treats local rule as universally correct.
- **Conceptual churn:** changes unrelated commitments during repair.
- **Exception accretion:** adds clauses rather than improving the boundary.
- **Plurality collapse:** forces one answer.
- **Plurality explosion:** lists unsupported alternatives.
- **Agreement bias:** accepts a harmful or incoherent stipulation without challenge.
- **Clarification reflex:** asks a question where context is sufficient.
- **Technical-label shortcut:** succeeds only when category names are present.
- **False transfer:** applies an analogy based on vocabulary.
- **Thread loss:** forgets QUD or local term.
- **Lecture substitution:** explains the topic instead of participating in the exchange.

## 17. Acceptance criteria

AlphaPact is ready to freeze only when:

- independent human reviewers agree sufficiently on executable/adjudicable subsets;
- legitimate disagreement is represented as answer sets, not majority-vote fiction;
- tests distinguish pact use from definition repetition;
- family, lexical, teacher, template, and source leakage checks pass;
- false bridges are matched and credible;
- ordinary conversational quality is rated separately;
- primary metrics aggregate by family;
- private evaluation artifacts are sealed before generation of the training release;
- benchmark model failures demonstrate a useful difficulty range;
- the exact evaluation harness and decoder are versioned;
- no benchmark content enters the primary training corpus.

The physical freeze, private-payload, contamination-closure, and evaluation-run requirements that enforce
these criteria are specified in [PRD-13](PRD-13-EVALUATION-FIREWALL-AND-FREEZE.md). This benchmark definition
does not itself authorize D6 implementation or item authoring before D5 closeout.
