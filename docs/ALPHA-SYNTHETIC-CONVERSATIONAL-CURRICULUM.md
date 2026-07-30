# Alpha's synthetic conversational curriculum

## Building an entity-light, evidence-first linguist-philosopher through deeply structured synthetic data

**Status:** proposed data-and-model research program; documentation only

**Date:** 2026-07-30

**Authorization boundary:** this document does not authorize data generation, database construction,
training, GPU provisioning, Donto writes, or publication claims

**Governing product definition:**
[Alpha's chatty research-model north star](ALPHA-CHATTY-RESEARCH-MODEL-NORTH-STAR.md)

**Supporting methods:** [Alpha Joints](ALPHA-JOINTS-RESEARCH-PROGRAM.md) and the
[Donto research agenda](DONTO-ALPHA-RESEARCH-AGENDA.md)

---

## 1. Executive decision

Alpha remains a small, natural, intellectually alive conversational model specialized in language,
pragmatics, ontology, and philosophy. The crucial addition is that **building the synthetic curriculum
is half of the research program**.

The project has two equal scientific artifacts:

1. **The Alpha Corpus:** a large, deeply categorized, provenance-complete synthetic curriculum of
   ordinary conversations, linguistic contrasts, conceptual inquiries, evidence-conditioned dialogues,
   counterexamples, repairs, conceptual pacts, and cross-domain transformations.
2. **The Alpha Model:** a one-GPU conversational model trained and evaluated to discover how much of
   that curriculum becomes usable, natural conversational intelligence.

The data is not preparatory plumbing. Its taxonomy, generation process, adversarial construction,
review history, rejected population, relational structure, and exact model exposure are research
outputs in their own right.

The corrected hypothesis is:

> **A large synthetic curriculum of richly categorized, linked, multi-turn conceptual conversations can
> teach a one-GPU language model to establish shared meanings, notice relevant ambiguity, challenge
> claims with valid cases, revise ideas locally, and carry distinctions across domains—without making
> ordinary conversation stiff, verbose, or lecture-like.**

The word *large* matters. A few exquisite concept families can validate the pipeline, but they cannot
supply the lexical, interactional, stylistic, and conceptual diversity needed to shape a whole
conversational model. The production ambition remains on the order of **hundreds of thousands of
accepted learning units**, generated from a much larger candidate population. That is a scaling horizon,
not permission to lower quality or bypass pilot gates.

The word *synthetic* also matters. The corpus will be deliberately authored with several Codex 5.x
teacher variants and selective Claude-based criticism or alternative generation. It will not be a dump
of one teacher answering random prompts. Generation will be a versioned scientific workflow whose every
input, output, critique, revision, rejection, and release decision is stored in SQLite.

---

## 2. What this program is trying to build

Alpha should be good to talk with about questions such as:

- What did somebody imply rather than assert?
- Is this disagreement about the world, the evidence, or the words?
- Is a student a kind of person or a role a person can bear?
- What remains the same when an institution changes its members and buildings?
- Is a hole a part, a boundary, an absence, or a dependent feature?
- Does this counterexample defeat the definition or merely expose another reading?
- Why does one phrasing sound accusatory while another sounds neutral?
- Which interpretation fits the context, and which evidence would decide?
- What representation is adequate for the questions we actually want to ask?
- What should change when a source retracts a claim, and what should remain stable?

It should also handle ordinary conversation naturally. It must greet, answer, react, clarify, disagree,
repair, joke, follow a thread, and stop without turning every exchange into a seminar.

The intended division of labor is:

| Alpha learns internally | External retrieval or Donto supplies |
|---|---|
| Language structure and conversational judgment | Exact quotations and source passages |
| Ordinary causal and social schemas | Names, dates, measurements, and current facts |
| Categories, relations, roles, events, parts, and time | Rare domain records and conflicting archives |
| Pragmatic and interpretive reasoning | Provenance and larger evidence collections |
| Conceptual comparison and revision | Search over the world beyond the prompt |
| Knowing when evidence is missing | The missing evidence itself |

This is better described as **entity-light and evidence-first** than literally knowledgeless. Alpha still
needs ordinary world structure. It cannot understand promises, institutions, intentions, bodies,
events, tools, evidence, or parts while knowing nothing about how such things generally behave. What it
does not need is a large inventory of entity-specific trivia.

---

## 3. The scientific center: conversational concept formation

The strongest scientific object from the external feedback is the **conceptual pact**: a temporary,
purpose-sensitive understanding that interlocutors establish about how a term, category, analogy, or
distinction is being used in the present conversation.

Examples include:

- “By *person* here, I mean the legal entity rather than the human being.”
- “Let's call a source that copies another source a dependent witness.”
- “For this discussion, count a hole as a structural feature, not a material component.”
- “When I say the university survived, I mean institutional continuity, not unchanged membership.”
- “Use *evidence* only for observations; call the rest interpretation.”

A capable Alpha should be able to:

1. notice that a local meaning has been proposed;
2. distinguish it from the default public meaning;
3. adopt it without mechanically restating it every turn;
4. use it efficiently in later reasoning;
5. recognize a later conflict with it;
6. distinguish intentional revision from accidental drift;
7. explain what the choice permits and excludes;
8. preserve unresolved alternatives when no single pact has been accepted;
9. abandon or revise the pact when the user changes the purpose;
10. remember that a different conversation may use the term differently.

Recent evidence from referential communication is unusually relevant. Human pairs became more accurate
and efficient as they established compact referring expressions, while frontier model pairs remained
verbose and did not show comparable common-ground formation. This does not prove the broader Alpha
hypothesis, because the study involved a visual reference task. It does identify a concrete failure that
the synthetic curriculum can target: **conversations should become semantically richer and
linguistically more economical as shared understanding accumulates**.

The model sees only natural conversation. Researchers may represent the hidden state with commitments,
denials, local terms, live alternatives, evidence dependencies, purposes, and revision edges. That
hidden representation is a measurement device, not the model's conversational voice.

---

## 4. Why synthetic data is half the project

No existing corpus is likely to contain the required combination at sufficient density:

- ordinary, varied, responsive dialogue;
- explicit but natural discussion of language and meaning;
- philosophically serious examples and counterexamples;
- ontology without machine-facing formalism;
- controlled ambiguity and perspective;
- conceptual pacts that evolve over several turns;
- source-conditioned disagreement and revision;
- the same distinction realized across unrelated domains;
- hard negative cases that look similar but require different conclusions;
- calibrated moments of asking, challenging, conceding, retrieving, and stopping.

Human conversations contain these phenomena, but they are rare, difficult to license, poorly annotated,
and rarely constructed with the contrasts needed for controlled learning. Textbooks explain concepts but
usually do not model responsive inquiry. Generic chat corpora provide interaction but spread their token
budget across facts, coding, task completion, and assistant conventions irrelevant to Alpha.

Synthetic generation allows the project to control:

- which conceptual distinction is active;
- what the user knows and intends;
- what the assistant should and should not infer;
- when a clarification is necessary;
- what evidence is available;
- which commitment changes after a new turn;
- which tempting analogy must be rejected;
- how the same idea appears in another domain;
- whether the answer should stop, explain, challenge, or ask;
- how conversational style and depth vary.

But synthetic generation also creates predictable hazards:

- one homogeneous teacher voice;
- unnaturally cooperative users;
- excessive politeness and affirmation;
- every answer ending with a question;
- overlong explanations;
- concept labels leaking into every example;
- invented facts and false quotations;
- superficial counterexamples;
- the generator and judge sharing the same blind spots;
- thousands of variants that are not genuinely independent;
- clean conversations with none of the interruptions, mistakes, or partial understanding found in real
  dialogue.

The data program exists to exploit the controllability while measuring and resisting these hazards.

---

## 5. Scale: candidates, accepted units, and independent structure

The original ambition of a large synthetic corpus should not be discarded. It should be made more
precise.

### 5.1 Production horizon

A reasonable production horizon, conditional on smaller validation stages, is:

- thousands of independent semantic and conversational families;
- tens of thousands of dialogue blueprints and trajectories;
- hundreds of thousands of accepted learning units;
- many more raw candidates, branches, critiques, and rejected outputs;
- enough ordinary conversation to prevent the specialist material from becoming Alpha's only voice.

The exact total should be governed by coverage and marginal value, not a round-number quota. Reaching
200,000 accepted units would be meaningful only if those units occupy a deliberately designed
capability space. Two hundred thousand paraphrases of a few templates would be a large file and a small
curriculum.

### 5.2 Unit hierarchy

The ledger must distinguish:

| Unit | Meaning |
|---|---|
| **Domain** | A broad area such as pragmatics, mereology, or evidence |
| **Concept family** | One transferable distinction, operation, or conversation problem |
| **Scene family** | A family situated in one scenario with common hidden state |
| **Blueprint** | The planned dialogue trajectory and intervention structure |
| **Dialogue** | One complete model-visible conversation |
| **Turn** | One speaker contribution within a dialogue |
| **Utterance or span** | A smaller linguistic object when annotation requires it |
| **Contrast pair** | Two scenes or utterances whose controlled difference matters |
| **Branch** | A counterfactual continuation from a shared dialogue prefix |
| **Learning unit** | The serialized object selected for a particular training objective |
| **Exposure** | One actual presentation of a unit during a model run |

Counts at these levels are not interchangeable. Ten surface dialogues from one blueprint do not become
ten independent conceptual observations. A training release may contain many useful dependent variants,
while evaluation and statistical claims remain grouped by family.

### 5.3 Multi-resolution data

The corpus should include several resolutions:

- sentence and phrase contrasts for dense linguistic structure;
- single-turn replies for response initiation, directness, and length control;
- short dialogues for interpretation and repair;
- medium dialogues for conceptual pacts and counterexamples;
- sustained dialogues for cumulative reasoning and drift;
- source-conditioned dialogues for evidence and attribution;
- linked cross-domain families for transfer.

The model should not learn that intellectual content always arrives in long exchanges. A one-sentence
reply can display excellent pragmatic and ontological judgment.

---

## 6. The master categorization system

Every accepted unit should be richly categorized. Categories are not ornamental tags; they determine
coverage, sampling, splits, evaluation, and later analysis.

No single label is sufficient. Each scene occupies a multidimensional design space.

### 6.1 Conversational function

- greet or acknowledge;
- answer directly;
- elaborate;
- exemplify;
- contrast;
- clarify;
- ask for evidence;
- ask a necessary disambiguating question;
- decline an unnecessary question;
- challenge a premise;
- concede;
- repair a misunderstanding;
- revise an earlier position;
- summarize shared ground;
- keep alternatives open;
- suggest a factual lookup;
- change depth or register;
- continue a line of inquiry;
- close naturally.

### 6.2 Linguistic level

- phonetics and phonology where text can support the phenomenon;
- morphology and word formation;
- syntax and constituent structure;
- lexical semantics;
- compositional semantics;
- reference and anaphora;
- quantification and scope;
- tense and aspect;
- modality;
- information structure;
- discourse coherence;
- pragmatics;
- sociolinguistic variation;
- historical change;
- typology;
- translation and lexical partitioning;
- language acquisition and metalinguistic explanation.

### 6.3 Pragmatic phenomenon

- literal force versus conversational act;
- implicature;
- presupposition;
- deixis;
- indirect request;
- politeness and face;
- irony and sarcasm;
- euphemism;
- under-specification;
- strategic silence;
- reported speech;
- quotation versus endorsement;
- topic and focus;
- accommodation;
- common-ground mismatch;
- clarification and grounding;
- conversational repair;
- speaker commitment;
- audience design;
- register shift.

### 6.4 Ontological domain

- type and token;
- class and instance;
- role and bearer;
- object, event, state, and process;
- identity through change;
- part, member, component, portion, and feature;
- collection and collective;
- boundary, hole, absence, and negative entity;
- constitution and dependence;
- function, purpose, and realization;
- causation and enabling condition;
- agency and responsibility;
- social and institutional entities;
- time-indexed existence and status;
- granularity;
- modality and counterfactual possibility;
- evidence, claim, source, and testimony;
- ambiguity and theory-relative analysis.

### 6.5 Philosophical operation

- identify the proposition under dispute;
- distinguish verbal, factual, normative, and framework disagreement;
- expose a hidden premise;
- strengthen an argument before criticizing it;
- construct a valid counterexample;
- explain why the counterexample engages the claim;
- refine a concept without adding arbitrary exceptions;
- test necessary and sufficient conditions;
- identify a false dichotomy;
- compare explanatory costs;
- preserve a genuine unresolved alternative;
- distinguish purpose-relative usefulness from universal truth;
- identify what evidence would discriminate between accounts;
- revise only the affected commitment;
- recognize when a question is malformed or conflates levels.

### 6.6 Epistemic condition

- directly stated;
- strongly implied;
- weakly suggested;
- presupposed;
- inferred from background structure;
- reported by a source;
- disputed by another source;
- retracted;
- temporally superseded;
- unknown;
- explicitly false;
- absent from the evidence;
- ambiguous;
- theory-relative;
- dependent on a competency question;
- requiring retrieval.

### 6.7 Conceptual-pact stage

- proposal;
- negotiation;
- acceptance;
- rejection;
- implicit adoption;
- efficient reuse;
- compression;
- challenge;
- repair;
- deliberate revision;
- accidental drift;
- scope change;
- suspension;
- abandonment;
- transfer to another case;
- recovery after interruption.

### 6.8 Interactional shape

- cooperative;
- skeptical;
- confused;
- adversarial but good-faith;
- emotionally invested;
- tentative;
- playful;
- impatient;
- novice-to-expert;
- peer-to-peer;
- user teaching the model;
- model correcting itself;
- asymmetric evidence;
- interrupted inquiry;
- topic return after digression.

### 6.9 Response policy

- answer and stop;
- answer and illustrate;
- answer and qualify;
- answer and gently challenge;
- ask one necessary question;
- offer alternatives without a question;
- say what must be retrieved;
- defer judgment;
- acknowledge uncertainty precisely;
- repair before continuing;
- summarize and close.

Question necessity must be explicitly annotated. The corpus must not teach the signature “make two
distinctions, then always ask which one the user means.”

### 6.10 Style and surface form

- concise, medium, and sustained;
- informal, neutral, and scholarly;
- warm, direct, dry, playful, and reflective;
- simple and technically precise vocabulary;
- sentence fragments and complete prose;
- varied rhythm and paragraph structure;
- several user expertise levels;
- regional and social varieties where responsibly sourced;
- explicit terminology versus terminology-free explanation.

### 6.11 Difficulty and transfer

- direct recognition;
- near-neighbor discrimination;
- hard negative;
- misleading surface cue;
- false analogy;
- incomplete context;
- competing evidence;
- multiple admissible analyses;
- composed intervention;
- lexical isolation;
- scenario isolation;
- cross-domain projection;
- long-horizon retention.

### 6.12 Factual load

- no entity-specific facts;
- fictional entities with ordinary world structure;
- anonymized real entities;
- supplied self-contained passage;
- sourced historical material;
- conflicting supplied sources;
- current facts requiring retrieval;
- quotation requiring verification.

These categories should be represented relationally in SQLite. They should not be flattened into one
comma-separated tag field.

---

## 7. Curriculum families

The corpus needs breadth of interaction and depth of structure. Its major families should include the
following.

### 7.1 Ordinary conversational nucleus

Natural greetings, reactions, requests, stories, opinions, follow-ups, misunderstandings, humor, mild
conflict, and topic changes. These scenes protect Alpha from becoming an ontology lecturer.

They should still contain careful interactional design:

- what move the user is making;
- whether a follow-up is useful;
- appropriate response length;
- what local context must be retained;
- what would sound mechanical or over-eager.

### 7.2 Language and meaning conversations

Users ask why a phrase sounds odd, what a sentence can mean, how two expressions differ, why translation
is difficult, or how context changes interpretation. The assistant responds naturally at varying depths.

These should include both technical and nontechnical renderings of the same insight.

### 7.3 Pragmatic interpretation conversations

Dialogs about implied criticism, indirect requests, humor, politeness, presuppositions, social risk,
strategic ambiguity, and speaker commitment. The assistant must avoid pretending to know private mental
states.

### 7.4 Ontological inquiry

Ordinary questions about what things are, what persists, what counts as part of what, whether roles are
entities, when an event ends, whether an absence can cause something, or how institutions survive.

The aim is flexible conceptual reasoning, not memorization of one ontology.

### 7.5 Philosophical co-investigation

Multi-turn exploration of definitions, premises, examples, counterexamples, analogies, and consequences.
The user may revise the claim, resist the model's distinction, or expose that the model's counterexample
misses the point.

### 7.6 Conceptual-pact conversations

The interlocutors establish a local term or distinction and later reuse, challenge, refine, or abandon
it. The hidden state records what changed after each turn.

### 7.7 Evidence-conditioned conversations

The user supplies one or more passages. Alpha distinguishes statement, implication, attribution,
conflict, time, missing evidence, and interpretation. These examples begin early in the curriculum even
though live search integration comes later.

### 7.8 Cross-domain families

One distinction appears in several unrelated settings:

- role and bearer in students, officeholders, permissions, and theatrical characters;
- source and claim in reported speech, archives, scientific testimony, and rumors;
- valid time and record time in narrative, employment, databases, and legal status;
- collective and members in grammar, committees, ecosystems, and teams;
- event boundaries in aspect, historical records, workflows, and rituals.

The model should learn the distinction without requiring the same vocabulary.

### 7.9 Hard negatives and false bridges

Pairs that share words, tone, or topic but do not instantiate the same structure. These prevent Alpha
from treating every apparent analogy as insight.

### 7.10 Repair and failure conversations

The assistant misunderstands, overstates, asks an unnecessary question, proposes an invalid
counterexample, or loses a local definition. A later turn diagnoses and repairs the failure. Some scenes
should show the user making the mistake; others should show the assistant making it.

### 7.11 Retrieval-aware conversations

Alpha recognizes that a quotation, attribution, date, or current fact requires checking. It can still
analyze the conceptual issue while marking the factual dependency.

### 7.12 Meta-conversational scenes

The user asks Alpha why it interpreted a sentence in a certain way, whether a clarification was really
necessary, or how their shared terminology changed. These make conversational reasoning explicit
without forcing formal output.

---

## 8. Worked synthetic family: a conceptual pact about evidence

This example shows how one seed becomes a family rather than one polished answer.

### 8.1 Hidden family objective

Teach the distinction between:

- an observation;
- a report of somebody else's claim;
- an interpretation;
- a conclusion licensed by several pieces of evidence.

The objective is not to memorize those four labels. It is to negotiate and use the distinction in
conversation.

### 8.2 Base pact

**User:** “For this discussion, can we reserve *evidence* for things somebody actually observed and
call the rest interpretation?”

**Desired assistant move:** recognize the proposed local convention, note one consequence, and adopt it
without insisting that it is the only legitimate public definition.

### 8.3 Efficient reuse

Several turns later:

**User:** “The diary says Thomas was angry. Is that evidence?”

The correct answer depends on who wrote the diary and what the wording reports. Alpha should use the
local pact without restating the entire agreement.

### 8.4 Challenge

**User:** “But facial expression is already an interpretation. Maybe our definition is too strict.”

Alpha should recognize a challenge to the pact rather than treating it as a new unrelated question.

### 8.5 Local revision

The pair may revise the working vocabulary:

> direct record, reported observation, and later interpretation

The prior turns do not become meaningless. The revision should preserve which earlier conclusions still
hold and identify which labels changed.

### 8.6 False drift

A later user says:

> “So the historian's conclusion is direct evidence.”

Alpha should notice that this conflicts with the accepted distinction unless the user is deliberately
proposing another revision.

### 8.7 Purpose shift

The user then explains that they only need a broad reading list, not a proof standard. Alpha may agree
that the finer distinction can be collapsed for that limited purpose without claiming the categories are
universally identical.

### 8.8 Cross-domain projection

The same family can project into:

- scientific observation versus theoretical interpretation;
- eyewitness testimony versus hearsay;
- a database record versus an inference derived from several records;
- a sentence's literal content versus a pragmatic inference.

### 8.9 Surface realizations

The generation system should create variants where:

- the assistant answers and stops;
- the assistant offers an example;
- the user is skeptical;
- the assistant initially misunderstands and repairs;
- no technical vocabulary appears;
- the conversation is informal;
- the same hidden structure occurs with entirely different nouns and verbs.

All variants share a family identity. They are not treated as independent evidence during evaluation.

---

## 9. The synthetic generation team

The corpus should be generated by a **division of epistemic labor** across multiple model calls, model
variants, prompts, and roles.

Several Codex 5.x models available at generation time can serve as the main teacher fleet. Claude Code
may be used selectively as an independent critic, alternative author, or adjudication input. Exact model
identifiers, versions, reasoning settings, prompts, and dates must be stored for every call because
product names and behavior change.

No teacher is treated as a source of truth merely because it is strong.

### 9.1 Generation roles

| Role | Responsibility |
|---|---|
| **Curriculum architect** | Selects the target cell in the coverage design |
| **Phenomenon specialist** | Specifies the linguistic, pragmatic, ontological, or philosophical structure |
| **Dialogue planner** | Creates the multi-turn trajectory and branch points |
| **User simulator** | Writes a plausible user with partial knowledge, goals, and changing reactions |
| **Assistant author** | Produces natural candidate responses consistent with the hidden plan |
| **Counterexample attacker** | Attempts to break a definition or analogy |
| **False-bridge constructor** | Builds a superficially similar case that should not transfer |
| **Pact-state analyst** | Records local terms, commitments, alternatives, and revisions |
| **Pragmatics critic** | Checks speech acts, implication, politeness, and intent calibration |
| **Linguistics critic** | Checks the described language phenomenon and invented examples |
| **Ontology critic** | Checks category, identity, time, part-whole, and dependency reasoning |
| **Evidence critic** | Checks attribution, source support, non-entailment, and uncertainty |
| **Conversation critic** | Checks directness, momentum, adaptation, presence, and length |
| **Style diversifier** | Produces genuinely different interactional realizations |
| **Adversarial reviewer** | Searches for hidden assumptions, leakage, and canned patterns |
| **Review chair** | Reconciles critiques without erasing disagreement |
| **Human adjudicator** | Decides high-value or contested conceptual cases |

One model call may fill more than one role during early calibration, but the database must record that
role overlap. Production-quality units should not be generated and finally approved by the same model,
prompt family, and reasoning path.

### 9.2 Teacher diversity

Diversity should come from:

- different Codex 5.x teacher variants;
- selective Claude generation or criticism;
- independently designed prompt families;
- different role assignments;
- varied hidden user goals and expertise;
- varied conversational trajectories;
- source-grounded and source-free cases;
- adversarial construction, not only paraphrase.

Teacher diversity is not automatically intellectual diversity. Models may share training data and
stylistic priors. Cross-model agreement is evidence to inspect, not proof.

### 9.3 Structured outputs and natural model-visible text

Generation agents should submit metadata through schema-validated structured-output or tool-call
interfaces. Do not scrape JSON from free prose. Validation failures remain recorded attempts.

The actual conversation remains separate natural-language content. It should not contain internal IDs,
tags, pact states, JSON, scoring rubrics, or assistant delimiters. Those are injected or linked later.

---

## 10. The generation workflow

The production process should be a resumable graph, not one prompt that asks a teacher to “write a good
conversation.”

### Step 1 — Select a coverage cell

Choose a deliberate combination of:

- domain;
- phenomenon;
- conversational function;
- pact stage;
- evidence condition;
- difficulty;
- style;
- response policy;
- target length;
- transfer relation.

Selection should prioritize coverage gaps and research questions, not whichever prompts happen to be
easy to generate.

### Step 2 — Create or select a concept seed

A seed may come from:

- a linguistic phenomenon;
- an ontological distinction;
- a philosophical problem;
- a real source passage;
- a recurring Donto alignment problem;
- an observed conversational failure;
- a teacher-proposed concept subjected to review.

The seed record describes why the family matters and what would constitute a shallow imitation.

### Step 3 — Specify the hidden state

Record:

- initial commitments;
- user goal;
- possible interpretations;
- unavailable information;
- source dependencies;
- intended conversational move;
- what should remain invariant;
- what a later intervention should change;
- forbidden implications;
- whether a question is necessary.

### Step 4 — Design a trajectory

The dialogue planner creates a sequence such as:

1. ordinary opening;
2. emergence of a distinction;
3. local agreement;
4. efficient reuse;
5. challenge or counterexample;
6. revision;
7. new application;
8. natural close.

Not every conversation uses this sequence. Reusing one arc would create another canned signature.

### Step 5 — Generate the user side independently

The user simulator should not merely ask questions optimized for the assistant's prewritten answer. It
should have:

- a partial understanding;
- a conversational purpose;
- preferences about depth;
- possible misconceptions;
- realistic reactions;
- permission to reject the assistant's framing;
- occasional digressions or shorthand.

### Step 6 — Generate candidate assistant continuations

Generate several materially different candidates:

- direct and concise;
- example-first;
- distinction-first;
- gently challenging;
- clarification-seeking where warranted;
- answer-and-stop.

The system should not assume one universally ideal response. Some scenes may retain several acceptable
continuations.

### Step 7 — Branch the conversation

Create controlled branches from the same prefix:

- the user accepts the distinction;
- the user rejects it;
- new evidence arrives;
- the user changes the purpose;
- the user uses the local term incorrectly;
- an ambiguity becomes resolved;
- a second ambiguity remains;
- the assistant's earlier response is challenged.

Branches make local revision trainable and measurable.

### Step 8 — Construct hard negatives

Create:

- an invalid counterexample;
- a false analogy;
- the same vocabulary with a different structure;
- different vocabulary with the same structure;
- a tempting but unsupported factual claim;
- an unnecessary clarification question;
- a polished but conceptually empty response.

### Step 9 — Run specialist critiques

Each critic returns granular judgments tied to exact turns and hidden expectations. Critiques are stored
even when later rejected.

### Step 10 — Revise without laundering history

Revisions create new immutable versions linked to their parents. The rejected original remains
available. A revised dialogue is not allowed to erase evidence of the teacher's initial failure.

### Step 11 — Diversity and contamination audit

Check lexical, semantic, structural, stylistic, source, prompt-template, teacher, and family overlap.

### Step 12 — Human adjudication where required

Humans review disputed counterexamples, contested interpretations, cultural claims, linguistic examples,
and high-value evaluation families.

### Step 13 — Release selection

Select accepted units into explicit cohorts. Do not duplicate rows to express sampling weight. A unit is
stored once and exposure policies refer to it.

---

## 11. Avoiding the synthetic assistant voice

Synthetic data can make Alpha sound like the average of its teachers: polished, agreeable, heavily
qualified, and predictably structured. The corpus must deliberately resist that convergence.

### 11.1 Follow-up question control

Every candidate should be labeled for whether a follow-up question is:

- necessary;
- useful but optional;
- unnecessary;
- evasive;
- already answered by context.

Matched variants should include:

- answer and stop;
- answer and add one example;
- answer and continue the thought;
- challenge without asking;
- ask one discriminating question;
- explicitly state that no clarification is needed.

### 11.2 Length control

For the same intellectual move, generate short, medium, and sustained realizations. Reviewers should
penalize answers whose added length does not add conceptual or conversational value.

### 11.3 User realism

Users should:

- use fragments and informal language;
- misunderstand in plausible ways;
- change their mind;
- resist the assistant;
- provide incomplete context;
- refer back with pronouns and shorthand;
- sometimes want an answer rather than a seminar;
- occasionally contribute the better distinction.

They should not all sound like benchmark writers.

### 11.4 Intellectual style variation

Some assistant responses should lead with an intuition, others with an example, a counterexample, a
distinction, a tentative objection, or a concise answer. Technical language should appear only when it
helps the particular interlocutor.

### 11.5 Style-scrubbed review

Reviewers should sometimes evaluate a response after removing greetings, affirmation, hedging, and
friendly framing. If no useful intellectual move remains, the response is shallow regardless of how
pleasant it sounds.

### 11.6 Template detection

Measure repeated:

- openings;
- transition phrases;
- answer structures;
- follow-up forms;
- qualification patterns;
- paragraph counts;
- rhetorical questions;
- examples and analogy shapes.

Semantic and syntactic clustering should supplement lexical duplicate checks. The solution is not a
hand-maintained blacklist of phrases.

---

## 12. Entity-light and evidence-first data

Knowledgeless Language Models provide direct evidence that anonymizing named entities during pretraining
can suppress entity-linked recall and improve several context-grounded behaviors. Alpha should learn
from that result without copying it uncritically or claiming the broad idea as novel.

### 12.1 Data conditions to compare

For selected families, create parallel conditions:

- ordinary named entities;
- fictional but naturalistic entities;
- anonymized entities with type-preserving descriptions;
- supplied evidence passages;
- evidence passages that conflict with familiar associations.

Anonymization must preserve the ordinary relational structure needed for reasoning. Replacing every
person, institution, location, and object with opaque symbols may teach placeholder manipulation rather
than conversation.

### 12.2 Evidence diversity

Evidence-conditioned scenes should include:

- one clear source;
- two corroborating sources;
- directly conflicting sources;
- sources from different times;
- one source copying another;
- observation versus later interpretation;
- incomplete evidence;
- irrelevant evidence;
- plausible but misleading evidence;
- a source with uncertain authority;
- a passage that supports only part of the user's claim;
- a case where no conclusion is licensed.

### 12.3 Context-memory conflict

Some examples should deliberately contradict familiar associations. Alpha must learn to distinguish:

- “the supplied passage stipulates X”;
- “my background expectation would normally be Y”;
- “the evidence is too weak to choose”;
- “the source itself should be challenged.”

This is more demanding than clean passage question answering.

### 12.4 Facts remain usable content

Historical, scientific, social, and ordinary factual passages can provide rich conceptual material. They
should be selected because they exercise interpretation, evidence, time, categorization, or language—not
because Alpha must memorize their entities.

---

## 13. Quality gates

Every candidate moves through explicit gates. Passing one gate cannot conceal failure at another.

### Gate Q0 — Structural integrity

- valid dialogue and turn structure;
- required metadata present;
- exact raw generation preserved;
- no delimiter contamination in canonical content;
- no truncated turns;
- provenance complete.

### Gate Q1 — Conversational quality

- direct response to the actual move;
- natural turn-taking;
- appropriate length;
- no canned follow-up;
- no lecture-mode drift;
- plausible user behavior;
- useful momentum or natural closure.

### Gate Q2 — Linguistic and conceptual validity

- phenomenon correctly instantiated;
- distinction actually matters;
- example and counterexample are valid;
- no forbidden entailment;
- terminology used correctly where present;
- no fake depth through abstraction.

### Gate Q3 — Pact and dialogue-state integrity

- local meanings tracked;
- accepted and rejected commitments distinguished;
- deliberate revisions recognized;
- accidental drift penalized;
- unresolved alternatives retained;
- later turns depend coherently on earlier ones.

### Gate Q4 — Evidence integrity

- claims attached to the right source;
- quotation and paraphrase distinguished;
- observation and interpretation separated;
- unsupported factual additions rejected;
- time and perspective preserved;
- retrieval need identified honestly.

### Gate Q5 — Diversity

- not a near-duplicate;
- not a template paraphrase;
- adds a real interactional, conceptual, lexical, or structural variant;
- does not worsen teacher-style concentration;
- contributes to an underrepresented coverage cell.

### Gate Q6 — Split and leakage safety

- family assignment frozen;
- no train-test conceptual sibling leakage;
- source, template, teacher, lexical, and scenario overlap checked;
- private evaluation never used as a generation prompt.

### Gate Q7 — Human authority

Human approval is required for categories where model consensus is not reliable, including:

- disputed counterexample validity;
- genuine versus invented ambiguity;
- subtle pragmatic intent;
- theory-relative philosophy;
- linguistic claims about languages the reviewer does not authoritatively know;
- culturally sensitive categorization;
- final private evaluation admission.

The Counterexample Game found that a model judge accepted approximately twice as many counterexamples
as human reviewers and that longer repair chains became more verbose without improving accuracy. Alpha
must not use model-only review to manufacture philosophical quality.

---

## 14. Review and adjudication

### 14.1 Preserve plural judgments

Review records should distinguish:

- accepted;
- accepted with revision;
- rejected;
- contested;
- theory-relative;
- requires specialist review;
- structurally valid but conversationally poor;
- conversationally strong but conceptually shallow.

Do not force all disagreements into one winner.

### 14.2 Independent review

Where practical:

- the author should not be the sole reviewer;
- one Codex variant should critique another;
- Claude may provide an independent family of objections;
- humans adjudicate the highest-impact disagreements;
- the review chair sees the critiques but not an instruction to preserve the original.

### 14.3 Calibration sets

Maintain human-adjudicated calibration sets for:

- counterexample validity;
- ambiguity legitimacy;
- question necessity;
- conceptual contribution;
- pact preservation;
- evidence support;
- conversational naturalness.

Model judges must report performance against these sets. A judge is not trusted globally because it
performed well on one category.

### 14.4 Rejection is data

Rejected candidates can later support:

- preference pairs;
- failure classifiers;
- prompt improvement;
- teacher comparison;
- research on philosophical-sounding shallowness;
- analysis of synthetic style collapse;
- adversarial evaluation.

Never delete or overwrite them.

---

## 15. The Alpha SQLite ledger

The complete corpus should live as a **versioned scientific object in SQLite**. SQLite is not merely the
final export format. It is the project memory for how every piece of data came to exist and how it was
used.

Large immutable payloads may be stored in content-addressed files when that is operationally sensible,
but SQLite must retain their hashes, sizes, media types, locations, ownership, license state, and every
relationship required for reconstruction.

### 15.1 Core principles

1. Raw generations are immutable.
2. Revisions create new records linked to parents.
3. Rejections are preserved.
4. Canonical messages contain no training delimiters.
5. Every model-visible byte is reconstructible.
6. Every training exposure is attributable to a release and rendering profile.
7. Splits occur at the family level before surface multiplication.
8. Competing analyses can coexist.
9. Sources and evidence spans remain first-class.
10. No aggregate score replaces per-item judgments.
11. Every teacher and reviewer action is versioned.
12. Data-selection weights do not duplicate canonical rows.

### 15.2 Project and policy tables

Provisional logical tables:

- `project`;
- `research_question`;
- `policy`;
- `policy_revision`;
- `authorization_event`;
- `taxonomy_release`;
- `quality_gate_definition`;
- `split_policy`;
- `render_policy`;
- `review_policy`;
- `license_policy`.

These establish which rules governed a generation or release. A later policy never silently rewrites an
earlier decision.

### 15.3 Teacher and prompt tables

- `provider`;
- `teacher_model`;
- `teacher_model_version`;
- `teacher_capability_profile`;
- `teacher_role`;
- `prompt_family`;
- `prompt_template`;
- `prompt_template_revision`;
- `prompt_instance`;
- `reasoning_configuration`;
- `generation_job`;
- `generation_attempt`;
- `generation_output`;
- `generation_failure`;
- `usage_record`;
- `cost_record`.

Record the exact Codex or Claude model identifier and configuration seen by the generation system. A
label such as “Codex 5.x” is a planning category, not sufficient provenance.

### 15.4 Source and evidence tables

- `source`;
- `source_version`;
- `source_fragment`;
- `source_license`;
- `source_authority_note`;
- `evidence_span`;
- `quotation`;
- `paraphrase_link`;
- `evidence_support_edge`;
- `evidence_conflict_edge`;
- `source_dependency`;
- `retrieval_requirement`.

Every sourced factual or linguistic claim should be traceable to the fragment used during generation or
review.

### 15.5 Taxonomy tables

- `domain`;
- `phenomenon`;
- `phenomenon_relation`;
- `conversational_function`;
- `philosophical_operation`;
- `ontological_operation`;
- `pragmatic_operation`;
- `epistemic_condition`;
- `interactional_shape`;
- `response_policy`;
- `style_dimension`;
- `difficulty_dimension`;
- `factual_load_class`;
- `failure_mode`;
- `coverage_cell`.

Many-to-many join tables should preserve which analyst assigned each category, with confidence,
rationale, and review status.

### 15.6 Family and scene tables

- `concept_family`;
- `family_version`;
- `family_objective`;
- `family_risk`;
- `scene_family`;
- `scene_blueprint`;
- `trajectory`;
- `trajectory_step`;
- `branch_point`;
- `branch`;
- `contrast_pair`;
- `transformation`;
- `cross_domain_projection`;
- `false_bridge`;
- `hard_negative`;
- `competency_question`.

These tables separate independent conceptual structure from surface realizations.

### 15.7 Dialogue tables

- `dialogue`;
- `dialogue_version`;
- `participant`;
- `participant_profile`;
- `turn`;
- `message_content`;
- `utterance_span`;
- `dialogue_act`;
- `turn_dependency`;
- `repair_link`;
- `question_necessity`;
- `closure_type`;
- `conversation_quality_annotation`.

The natural-language message is stored independently of speaker delimiters and rendering syntax.

### 15.8 Conceptual-pact and commitment tables

- `conceptual_pact`;
- `pact_term`;
- `pact_scope`;
- `pact_purpose`;
- `pact_state`;
- `pact_state_transition`;
- `commitment`;
- `commitment_status`;
- `commitment_dependency`;
- `denial`;
- `live_alternative`;
- `admissible_analysis`;
- `forbidden_entailment`;
- `revision_event`;
- `invariance_requirement`;
- `drift_event`;
- `recovery_event`.

The state after each relevant turn should be reconstructible without forcing one philosophically final
analysis.

### 15.9 Candidate and revision tables

- `candidate`;
- `candidate_parent`;
- `candidate_role`;
- `candidate_variant`;
- `revision`;
- `revision_reason`;
- `rejection`;
- `rejection_reason`;
- `accepted_use`;
- `supersession`;
- `duplicate_cluster`;
- `contamination_flag`.

The system must distinguish a candidate dialogue from a candidate turn, interpretation, counterexample,
or annotation.

### 15.10 Review tables

- `review_assignment`;
- `reviewer`;
- `reviewer_type`;
- `review_judgment`;
- `review_dimension_score`;
- `review_rationale`;
- `review_disagreement`;
- `adjudication`;
- `adjudication_member`;
- `human_authority_note`;
- `judge_calibration_result`;
- `quality_gate_result`.

Raw reviewer output and normalized judgments should both be preserved.

### 15.11 Release and split tables

- `dataset_release`;
- `release_parent`;
- `release_manifest`;
- `release_member`;
- `cohort`;
- `cohort_member`;
- `split`;
- `split_assignment`;
- `holdout_reason`;
- `sampling_policy`;
- `sampling_weight`;
- `exclusion`.

A release is an immutable manifest, not a mutable query result.

### 15.12 Rendering and token exposure tables

- `render_profile`;
- `delimiter_profile`;
- `rendered_unit`;
- `rendered_message`;
- `tokenizer_version`;
- `token_sequence`;
- `token_occurrence`;
- `loss_mask`;
- `training_example`;
- `training_exposure`;
- `exposure_order`;
- `exposure_weight`.

Token-occurrence materialization may be phased for storage reasons, but exact rendered text, tokenizer,
mask, order, and hashes must be preserved from the first training release.

### 15.13 Model-run and evaluation tables

- `model_configuration`;
- `training_run`;
- `training_stage`;
- `checkpoint`;
- `checkpoint_artifact`;
- `generation_configuration`;
- `model_output`;
- `evaluation_suite`;
- `evaluation_item`;
- `evaluation_run`;
- `evaluation_output`;
- `evaluation_judgment`;
- `human_dialogue_session`;
- `pairwise_preference`;
- `failure_observation`;
- `null_result`.

The ledger connects model behavior back to the exact synthetic units and exposures that may have shaped
it.

### 15.14 Required reconstructibility queries

The database must answer:

- Which teacher, prompt revision, source fragments, and critiques produced this turn?
- Which rejected candidates share its parent blueprint?
- Which conceptual pact was active at each turn?
- Which commitments were meant to change after the counterexample?
- Was the final follow-up question necessary?
- Which independent family owns this surface realization?
- Which release and training run exposed the model to it?
- How many times and in what order was it seen?
- Which evaluation families are lexically or structurally related?
- Which teacher styles dominate the current release?
- Which taxonomy cells are underfilled?
- Which human disagreements remain unresolved?
- Can the exact model-visible byte and token sequence be regenerated?
- Did a conceptually improved unit reduce conversational naturalness?
- Which data cohorts correlate with response initiation, hedging, or lecture mode?

---

## 16. Coverage steering

Generation should be driven by a live coverage model derived from SQLite.

### 16.1 Coverage is multidimensional

Do not say “we have 10,000 pragmatics examples” and assume coverage. Ask how many involve:

- indirect requests;
- a skeptical user;
- no clarification question;
- an evidence passage;
- medium-length dialogue;
- a deliberate pact revision;
- a false-bridge negative;
- informal language;
- terminology-free transfer.

### 16.2 Quotas are allocation hypotheses

Set provisional target distributions over:

- ordinary versus specialist conversation;
- linguistic, pragmatic, ontological, philosophical, and epistemic content;
- dialogue length;
- response policy;
- user stance;
- teacher model and prompt family;
- source-grounded versus constructed scenes;
- positive, negative, ambiguous, and revision cases;
- concept families and projections.

Rebalance from measured gaps and training evidence. Do not duplicate stored units to meet a quota;
sampling policies govern exposure.

### 16.3 Marginal information

A candidate should earn admission by adding at least one meaningful form of novelty:

- new concept family;
- new projection;
- harder boundary case;
- new conversational move;
- new style or user profile;
- new evidence condition;
- new pact transition;
- new response length;
- valid disagreement;
- useful negative.

Lexical novelty alone is insufficient.

---

## 17. Split and leakage policy

### 17.1 Split before multiplication

Concept families, sources, templates, and intended projections should be assigned to development or
private evaluation before large-scale surface generation. Otherwise later variants will leak the hidden
structure across splits.

### 17.2 Leakage dimensions

Audit:

- exact and near text overlap;
- semantic duplicate clusters;
- shared scenario skeletons;
- shared latent contracts;
- teacher and prompt-template fingerprints;
- named entity overlap;
- source overlap;
- explanation order;
- counterexample structure;
- local-term reuse;
- dialogue trajectory overlap.

### 17.3 Evaluation privacy

Private evaluation prompts, sources, hidden pact states, and admissible judgments must never be supplied
to generation agents. Error analysis can motivate a new training family, but the final evaluation family
must remain untouched.

### 17.4 Statistical unit

Research claims should treat the independent concept or scene family as the main unit. Turns, branches,
and paraphrases are dependent observations even when they provide useful training signal.

---

## 18. Dataset release ladder

Use immutable release layers:

### Raw

Every generation attempt, including malformed, empty, duplicated, and rejected outputs.

### Bronze

Structurally valid candidates with complete provenance.

### Silver

Candidates that pass automated and model-based review but may still await specialist or human
adjudication.

### Gold

Accepted training units satisfying the release's declared gates.

### Red

Preserved failures and hard negatives suitable for preference data, diagnosis, or evaluation but not
ordinary positive training.

### Frozen evaluation

Human-adjudicated, isolated families never exposed during generation or training.

Each release receives a manifest, database snapshot or reproducible query contract, content hashes,
taxonomy version, prompt versions, source-license summary, and coverage report.

---

## 19. Training representations derived from the ledger

The same canonical data can support several experimental exports without being duplicated or rewritten.

Possible representations include:

- ordinary next-token conversational text;
- assistant-only conversational supervision;
- paired contrast scenes;
- branch continuations from a common prefix;
- preference pairs from accepted and rejected candidates;
- source passage followed by natural dialogue;
- pact establishment followed by delayed reuse;
- corrupted-neighborhood controls;
- short-answer initiation cohorts;
- continued language-modeling material built from natural explanations and dialogues.

All exports should inject user and assistant delimiters only during rendering. Metadata and hidden state
must not leak into model-visible text unless an experiment explicitly tests natural-language
metalinguistic explanation.

The completed Alpha run showed why exposure accounting matters: long answer interiors can dominate loss,
ordered source blocks can create recency effects, and good teacher-forced continuation does not prove
free response initiation. The SQLite ledger must therefore preserve episode weighting, first-response
positions, sequence order, masks, truncation, and length distributions.

---

## 20. The data experiments

The synthetic corpus should support controlled tests of the data itself.

### 20.1 Generic versus targeted

Compare ordinary high-quality chat with the same token budget of language-, pragmatics-, ontology-, and
philosophy-centered conversation.

### 20.2 Independent versus linked

Present the same targeted content as independent scenes or linked multi-turn families. Test whether
linkage improves pact formation, local revision, and cross-domain reuse.

### 20.3 Correct versus corrupted relations

Pair scenes correctly or pair them with structurally wrong siblings. If both help equally, the gain is
not evidence of relational learning.

### 20.4 Single-teacher versus teacher mixture

Hold content targets constant while varying generation and review diversity. Measure conceptual quality,
voice concentration, and template dependence.

### 20.5 Entity-normal versus entity-light

Compare ordinary, fictionalized, anonymized, and evidence-supplied versions. Measure evidence reliance,
naturalness, and conceptual transfer without claiming that one intervention is universally best.

### 20.6 Review depth

Compare minimal screening, multi-model review, and human-adjudicated subsets. Quantify the marginal value
of each review layer.

### 20.7 Question policy

Compare data in which teachers naturally overproduce follow-up questions with data explicitly balanced
for answer-and-stop, optional continuation, and necessary clarification.

### 20.8 Scale curves

Measure performance as accepted coverage grows. Count independent families, projections, interaction
types, and accepted tokens—not just rows.

### 20.9 Evidence timing

Compare introducing evidence-conditioned dialogue early throughout the curriculum versus attaching it
only after conceptual conversation has been learned.

### 20.10 Atomic contrasts versus full dialogues

Test whether sentence pairs teach dense linguistic distinctions efficiently and whether full dialogues
are required to make those distinctions operational in conversation.

---

## 21. Evaluation aligned with the product

### 21.1 Hard product gate

Alpha must reliably answer, remain relevant, avoid loops, control length, track the conversation, and
sound natural. No conceptual metric compensates for failure here.

### 21.2 Interaction and conceptual contribution stay separate

Score at least two channels:

1. **Interaction quality:** responsiveness, momentum, adaptation, presence, repair, and length control.
2. **Conceptual contribution:** valid distinction, relevant example, real counterexample, premise
   tracking, local revision, evidence discrimination, and transfer.

Warmth cannot hide shallowness. Technical accuracy cannot hide conversational failure.

### 21.3 Conceptual-pact evaluation

Test whether Alpha:

- recognizes a local stipulation;
- reuses it efficiently;
- avoids reintroducing rejected terminology;
- distinguishes pact from public convention;
- notices drift;
- handles deliberate revision;
- preserves unresolved alternatives;
- transfers the local distinction to a new case;
- becomes more concise as common ground grows.

### 21.4 Pragmatic policy

Evaluate when Alpha should:

- answer;
- ask;
- challenge;
- accommodate;
- retrieve;
- leave alternatives open;
- stop.

Correctly identifying an ambiguity but choosing an irritating conversational action is not success.

### 21.5 Evidence-conditioned behavior

Test:

- source attribution;
- contradiction;
- time and granularity;
- incomplete and misleading evidence;
- context that conflicts with familiar associations;
- unsupported conclusions;
- recognition that retrieval is necessary.

### 21.6 Sustained conversation

The eventual experience may be a long discussion, but initial claims must fit the system's demonstrated
context and memory. Evaluate bounded multi-turn sessions first. Any rolling conversation-state ledger or
memory compression must be an explicit component with separate tests for omission, distortion, stale
commitments, and privacy.

### 21.7 Human desire to continue

In blinded pairwise sessions, ask whether reviewers would voluntarily continue the conversation. Control
for answer length and presentation so verbosity does not win by default.

---

## 22. A practical data-building roadmap

No stage below is authorized by this document. Each requires an explicit generation contract, storage
budget, provider policy, and stopping condition.

### D0 — Freeze taxonomies and private evaluation

- finalize the first taxonomy release;
- define family and dialogue units;
- freeze private conversational and conceptual-pact evaluations;
- define human review protocols;
- define source and license rules;
- approve the initial SQLite logical schema.

### D1 — Generator calibration

- generate a small but deliberately diverse candidate batch;
- measure schema failures, teacher signatures, repetition, invalid counterexamples, and review
  disagreement;
- revise prompts and taxonomies;
- do not train on this batch merely because it exists.

### D2 — Curated pilot corpus

- construct several thousand accepted units across all major data resolutions;
- include ordinary conversation, conceptual dialogue, evidence, pacts, hard negatives, and repair;
- conduct human calibration;
- prove exact reconstruction and split isolation.

### D3 — First data ablations

- compare independent and linked scenes;
- compare teacher mixtures;
- compare entity conditions;
- measure whether synthetic style is becoming dominant;
- use only bounded one-GPU experiments under separate authorization.

### D4 — Medium-scale expansion

- expand successful families and undercovered cells;
- add new independent concepts rather than multiplying only successful templates;
- introduce more sourced passages and cross-domain projections;
- publish a full data card and failure report.

### D5 — Production-scale synthetic curriculum

- grow toward hundreds of thousands of accepted units only when earlier releases show marginal value;
- retain a much larger raw and rejected candidate population;
- build several training cohorts rather than one monolithic “final dataset”;
- use SQLite to generate reproducible releases for different hypotheses.

### D6 — Ongoing ecology

- add newly discovered conversation failures;
- preserve dataset lineage;
- compare teacher generations over time;
- retire harmful units without deleting history;
- continue building the corpus as a reusable research asset independent of any one checkpoint.

The labor, review attention, and intellectual ownership of D0–D6 should be treated as roughly half the
program. Training is not the only “real work.”

---

## 23. One-GPU design constraint

The model should remain as small and efficient as possible while achieving the target, because the
project has one GPU for training and serving experiments.

That constraint affects the data program:

- every training token must justify its role;
- redundant surface variants should not crowd out independent concepts;
- short, information-dense scenes matter;
- ordinary chat must remain sufficiently represented;
- length buckets and exposure weights must be deliberate;
- cohorts should permit bounded experiments rather than requiring full-corpus runs;
- exact checkpoints and data releases must support interruption and resumption;
- evaluation should select behavior, not merely loss.

The document deliberately does not prescribe a size ladder. The engineering target is simply a model
that fits the available hardware and meets the conversational gates.

---

## 24. Donto's role

Donto remains important in four ways:

1. **Inspiration:** its contradiction-preserving, evidence-first worldview identifies valuable
   conversational distinctions.
2. **Source of difficult cases:** real predicate variation, alignment uncertainty, evidence conflict,
   time, and granularity can seed later families.
3. **Future retrieval substrate:** it can supply claims and sources without requiring Alpha to memorize
   them.
4. **Ecological evaluation:** Alpha can eventually propose interpretations or distinctions in a shadow
   context and be evaluated for usefulness and safe non-collapse.

Donto is not the training database. The Alpha SQLite ledger records scientific construction and model
exposure. Donto records claims about sources and the world. Their responsibilities may connect later but
must not collapse.

No synthetic candidate should become a live Donto claim merely because a teacher generated it.

---

## 25. Failure modes

### Data is large but shallow

Symptoms: high row count, low independent-family count, repeated rhetorical structure, weak transfer.

Response: stop scale-up and add new concept families, user goals, interaction patterns, and hard
negatives.

### The model becomes a lecturer

Symptoms: long definitions, headings, constant qualifications, weak engagement with the user's move.

Response: increase ordinary dialogue, short responses, answer-and-stop variants, user resistance, and
length-controlled preference data.

### The model always asks a question

Symptoms: every answer ends with “which sense do you mean?” or “would you like an example?”

Response: enforce question-necessity balance and evaluate unnecessary-question rate.

### Synthetic users are too cooperative

Symptoms: users accept every distinction and conveniently supply the exact missing fact.

Response: independent user simulation, disagreement trajectories, interruptions, partial understanding,
and human-authored user turns.

### Counterexamples sound clever but miss

Symptoms: revisions grow longer while the original boundary remains untested.

Response: granular validity criteria, adversarial critics, and human adjudication.

### Entity-light becomes worldless

Symptoms: Alpha manipulates placeholders but lacks ordinary causal, social, and material understanding.

Response: preserve typed world structure, fictional naturalistic situations, and grounded supplied
passages.

### The model judges share the same blind spot

Symptoms: unanimous automated acceptance contradicted by humans.

Response: cross-family teachers, calibrated judges, source checks, and preserved disagreement.

### Metadata leaks into voice

Symptoms: Alpha talks about “commitment deltas,” tags, or hidden rubrics when ordinary language would do.

Response: strict separation of canonical messages from researcher-side structure and targeted leakage
tests.

### The database delays all learning evidence

Symptoms: schema work expands indefinitely while no bounded corpus has passed human review.

Response: implement complete logical preservation in phases; materialize expensive derived tables when
needed, while never discarding raw provenance.

### Training consumes the whole project

Symptoms: generation is rushed so another run can start.

Response: treat corpus releases, reviews, and coverage results as first-class milestones. No GPU run
creates data quality after the fact.

---

## 26. What not to do

- Do not ask one model to generate 200,000 chats in one repeated template.
- Do not count turns, rows, variants, and independent families as the same thing.
- Do not accept data because two model judges agree.
- Do not make every conversation philosophical.
- Do not teach JSON, code, or metadata in the core model-visible corpus.
- Do not hide source-free factual invention inside fluent dialogue.
- Do not treat all ambiguity as a reason to ask the user.
- Do not reward verbosity as depth.
- Do not use a fixed list of phrases as the main diversity detector.
- Do not duplicate rows to express sampling weights.
- Do not leak related families across evaluation splits.
- Do not overwrite rejected or superseded candidates.
- Do not make Donto's current predicates the answer key.
- Do not claim entity-light training, conceptual pacts, synthetic multi-agent review, or dialogue
  specialization as individually novel.
- Do not let a new training run begin merely because a large candidate count has been reached.

---

## 27. Prior-art implications

This document accepts the following constraints on novelty:

- [Knowledgeless Language Models](https://arxiv.org/abs/2607.12831) already studies named-entity
  anonymization as a way to suppress parametric recall and improve evidence-conditioned behavior.
  Alpha's contribution cannot be “facts outside the model” in general.
- [Dialogue Is Not Enough to Make a Communicative BabyLM](https://arxiv.org/abs/2510.20358) shows that
  dialogue-centric data can improve a narrow dialogue-continuation test while failing to deliver broad
  communicative competence. Dialogue volume alone is not the intervention.
- [LVLMs and Humans Ground Differently in Referential Communication](https://aclanthology.org/2026.acl-long.410/)
  reports that frontier model pairs remained verbose and failed to show the increasing efficiency of
  human conceptual pacts in its task. This supplies a timely target, not proof that Alpha's broader
  version is unique.
- [Conceptual Pacts for Reference Resolution Using Small, Dynamically Constructed Language Models](https://aclanthology.org/2024.lrec-main.327/)
  already models temporary referential pacts. Alpha must go beyond object naming into negotiated
  conceptual distinctions, revision, and cross-domain conversation.
- [Review-Instruct](https://aclanthology.org/2025.findings-acl.851/) already uses candidate, reviewer,
  and chair roles to generate multi-turn instruction data. Multi-agent generation itself is not novel.
- [Data Selection for Multi-turn Dialogue Instruction Tuning](https://aclanthology.org/2026.findings-acl.130/)
  treats whole-dialogue trajectory, topic grounding, information progress, and answer-form consistency
  as selection signals. Alpha should build on rather than rediscover these concerns.
- [The Counterexample Game](https://arxiv.org/abs/2605.03936) shows both the promise and the limits of
  iterative conceptual repair, including over-permissive model judging and verbosity without accuracy
  gains.
- [Dialogue is the Plan](https://aclanthology.org/2026.acl-short.63/) argues for shared commitments,
  grounding, and repair as part of the dialogue process rather than an output wrapper. Alpha applies a
  related principle to conceptual conversation rather than agentic task planning.
- [Understanding Common Ground Misalignment in Goal-Oriented Dialog](https://aclanthology.org/2025.acl-long.161/)
  finds that subtle, context-dependent common-ground failures remain difficult. Common-ground detection
  itself is not an untouched problem.
- [Real or Robotic?](https://aclanthology.org/2026.findings-acl.2060/) reports that LLM simulations cover
  a narrow range of human dialogue style and dynamics. Synthetic user simulation requires explicit
  validation rather than an assumption of human realism.

The strongest surviving contribution is the combination of:

- a product-first conversational goal;
- a large, deeply categorized synthetic curriculum;
- conceptual-pact formation and revision beyond reference naming;
- linguistic, pragmatic, ontological, philosophical, and epistemic cross-domain transfer;
- equal-token linked-versus-independent and corrupted-relation controls;
- entity-light, evidence-conditioned dialogue;
- human-dominant adjudication of conceptual validity;
- complete SQLite lineage from generation through exposure and behavior;
- preservation of ordinary conversational naturalness as a hard endpoint.

That conjunction is a hypothesis to test, not a novelty claim to announce before the evidence exists.

---

## 28. Work packages for research agents

### DATA-0 — Taxonomy audit

Test whether the categorization system is complete, orthogonal enough to allocate data, and usable by
reviewers. Identify conflated axes and missing conversational phenomena.

### DATA-1 — Synthetic-generation architecture

Design the Codex 5.x and Claude role graph, schema-validated outputs, retry policy, failure preservation,
and reproducibility contract.

### DATA-2 — Natural dialogue

Research how to generate varied, non-canned users and responses. Define metrics for teacher-style
concentration, unnecessary questions, and lecture mode.

### DATA-3 — Linguistic curriculum

Define high-value phenomena, minimal contrasts, ordinary explanations, cross-linguistic authority, and
tests that do not reward terminology alone.

### DATA-4 — Pragmatics and conceptual pacts

Define pact lifecycles, common-ground state, drift, repair, challenge policy, and efficient reuse across
turns.

### DATA-5 — Ontology and philosophy

Define operations, valid counterexamples, purpose-relative modeling, competing analyses, and human
adjudication protocols.

### DATA-6 — Evidence-first curriculum

Design entity-light, fictionalized, anonymized, sourced, conflicting, incomplete, and misleading
evidence conditions.

### DATA-7 — SQLite ledger

Turn the logical table families and invariants into a staged schema proposal. Prove exact reconstruction
and immutable history before any bulk generation.

### DATA-8 — Quality and adjudication

Create calibration sets, review rubrics, judge validation, rejection taxonomies, and human escalation
rules.

### DATA-9 — Leakage and diversity

Design family-aware splitting, semantic duplication detection, prompt and teacher fingerprinting, and
coverage steering.

### DATA-10 — Data ablations

Specify equal-token comparisons for independent versus linked scenes, correct versus corrupted
relations, teacher mixtures, entity conditions, and review depth.

### DATA-11 — One-GPU training interface

Specify how cohorts, lengths, masks, exposure weights, order, checkpoints, and free-generation gates
connect the ledger to bounded training without predetermining architecture size.

Each agent must distinguish verified prior work, proposed design, unresolved question, and personal
recommendation.

---

## 29. Readiness gates before generation

Bulk synthetic generation should not begin until:

- the product north star is frozen;
- the first private evaluation families are isolated;
- the taxonomy has a versioned release;
- the canonical unit hierarchy is defined;
- the SQLite logical schema has been independently reviewed;
- prompt and model provenance can be stored exactly;
- natural-language content is separated from rendering delimiters;
- rejected outputs can be preserved cheaply;
- source and licensing policies are explicit;
- human-review responsibilities are known;
- teacher and judge calibration sets exist;
- candidate and accepted counts are reported separately;
- storage admission has been checked;
- generation authorization and provider-use limits are explicit.

Training should not begin merely because generation has begun. A training release must additionally
have:

- passed coverage and diversity audit;
- passed split leakage audit;
- passed human review at the declared level;
- a frozen manifest and hashes;
- exact rendering and exposure rules;
- a bounded experiment that identifies what the release is supposed to test.

---

## 30. Definition of success for the data half

The Alpha Corpus succeeds when it is:

- large enough to shape a conversational model rather than demonstrate a few ideas;
- organized around independent conceptual and interactional structure;
- richly categorized without exposing metadata in dialogue;
- diverse across teachers, prompts, users, trajectories, styles, lengths, and domains;
- strong in ordinary conversation as well as specialist inquiry;
- full of real hard negatives, repairs, and competing analyses;
- evidence-conditioned from early stages;
- able to support conceptual-pact and cross-domain experiments;
- human-calibrated where models are unreliable;
- immutable, reconstructible, and queryable in SQLite;
- inclusive of failures and rejected candidates;
- capable of producing multiple controlled training cohorts;
- demonstrably useful under one-GPU experimental constraints.

The data program fails if it creates a large text file whose provenance, independent structure,
coverage, and causal value cannot be recovered.

---

## 31. Final program definition

Alpha is a small, one-GPU conversational linguist-philosopher whose factual knowledge is deliberately
secondary to language, interpretation, ontology, inquiry, and evidence use.

Half of the work is to build Alpha's synthetic world of conversations:

- ordinary and strange;
- concise and sustained;
- cooperative and resistant;
- clear and ambiguous;
- sourced and hypothetical;
- linguistically dense and socially natural;
- full of local terms, changing commitments, counterexamples, repairs, false analogies, and genuine
  unresolved alternatives.

Codex 5.x teachers and selective Claude critics can generate that world at scale. SQLite will remember
how every scene was conceived, generated, reviewed, revised, rejected, selected, rendered, and shown to
the model. Human reviewers will remain the final authority where conceptual validity cannot be reduced
to a mechanical check.

The other half is to discover what a one-GPU model can learn from it without losing the thing the user
actually wants: a natural, present, insightful conversational partner.

The research question is not whether synthetic data can make Alpha sound philosophical. It is whether a
scientifically constructed synthetic curriculum can teach Alpha to **participate in the formation of
meaning with another person**.
