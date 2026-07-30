# Alpha research-model dataset program

**Status:** research brief; proposed design, not an authorized generation or training run  
**Date:** 2026-07-30  
**Scope:** data, curriculum, provenance, and evaluation for a future Alpha continuation  
**Canonical repository:** `alpha2`  
**Current model:** 57,688,576 parameters, 12,288-token byte-BPE vocabulary, 1,024-token context  

## 1. The objective

Build the data program for a small, genuinely conversational research model whose distinctive strengths
are:

- natural, eager, coherent conversation;
- a strong practical grasp of how language works;
- the ability to notice and explain linguistic structure, ambiguity, implication, and variation;
- careful reasoning about categories, identity, relations, parts, events, time, evidence, and
  contradiction;
- intellectual curiosity, calibrated uncertainty, and the ability to reason with a person rather than
  merely emit a reference answer.

Broad encyclopedic recall is not the priority. Programming, source code, APIs, JSON production, and
tool-use formats are explicitly out of scope for this curriculum. The desired model is closer to a
small conversational linguist and ontologist than a miniature general-purpose coding assistant.

The working hypothesis is that a deliberately constructed corpus of approximately **200,000 accepted
training units** could move Alpha much closer to that target. A unit may be a short dialogue, a
sentence contrast, a guided analysis, or an ontology problem expressed as natural conversation. The
number is a research hypothesis, not a quota that overrides quality.

This document is designed to be handed to independent research agents. Their first job is to test and
improve the hypothesis, not to endorse it.

### 1.1 Research posture and possible novelty

This is an attempt at a novel research program, not a routine dataset assembly exercise. The candidate
novelty is the **combination** of:

- a very small language model specialized for open-ended conversation rather than broad assistant
  coverage;
- linguistic metareasoning as a central capability rather than an evaluation afterthought;
- ontology, contested categorization, evidence, and temporal qualification taught in ordinary language;
- contrastive and multi-analysis examples that preserve ambiguity instead of forcing one gold answer;
- a large but tightly targeted synthetic curriculum whose accepted units retain source and generation
  provenance;
- behavioral admission based on free conversation, response initiation, and explanation—not only
  teacher-forced loss or multiple-choice accuracy.

It may be new as a combined program. That is **not yet a verified novelty claim**. Small-model synthetic
curricula, language-model distillation, linguistics benchmarks, ontology verbalization, natural-language
inference, philosophical dialogue corpora, and domain specialization all have substantial prior art.
Research track R0 must look for the closest precedents and report both overlap and the narrower claim
that survives. “Nobody has done this” is prohibited until that review is complete.

Novelty is not itself a success criterion. A known method that works is preferable to a novel mixture
that cannot be distinguished from teacher imitation. The research contribution should ultimately be a
falsifiable result: which data forms and curricula do or do not give a model at this scale robust
conversational, linguistic, and ontological competence?

## 2. Authority boundary

This brief does **not** authorize:

- generating the corpus;
- starting or resuming training;
- provisioning a GPU or RunPod;
- modifying model, trainer, loader, inference, service, or publication code;
- changing the frozen evaluation or the public claims about Alpha;
- deleting or replacing the failed run, its samples, or its checkpoints.

The current task is research and documentation only. Any future data generation must have an explicit
generation contract. Any future training must have a separate, explicit training authorization.

## 3. Non-negotiable product intent

The future model should:

1. **Answer rather than disappear.** It must reliably begin a response across short, medium, and long
   prompts.
2. **Sound like an interlocutor.** It should respond to the particular person and conversational move,
   not paste a generic textbook block.
3. **Handle language as structure and action.** It should reason about form, meaning, use, context,
   variation, change, and acquisition.
4. **Handle ontological disagreement without flattening it.** It should distinguish alternative
   categorizations, evidence, time, perspective, and levels of granularity.
5. **Explain at the right depth.** It should be able to give a one-sentence answer, a worked example,
   or a sustained research dialogue without assuming that longer is always better.
6. **Admit uncertainty and multiple analyses.** It should not invent a fact, language example, source,
   consensus, or single correct analysis merely to sound confident.
7. **Stay in natural language.** Internal corpus metadata may be structured, but model-visible training
   content in this program must not teach JSON, code, markup-heavy protocols, or tool-call habits.

The model does not need to know every fact in the world. It does need enough grounded examples to learn
the difference between understanding a question, manipulating words, and merely sounding plausible.

## 4. Frozen baseline: what actually happened

Alpha's completed SFT run was mechanically successful and qualitatively unsuccessful:

| Item | Frozen result |
|---|---:|
| Parameters | 57,688,576 |
| Base pretraining | 1,000,013,824 tokens |
| SFT conversations | 511,428 |
| SFT padded positions | 496,795,648 |
| SFT epochs | 1 |
| Final train / held-out loss | 1.7579851 / 1.6439665 |
| Terminal structural chat pass | 2 / 100 |
| Terminal empty responses | 92 / 100 |
| Terminal degenerate loops | 6 / 100 |
| Blinded semantic assessment | 0 PASS / 100 FAIL |
| Closed-book QA | 0 / 200 exact; 0 contained |

The current source mix is:

| Source | Rows | Share |
|---|---:|---:|
| SmolTalk | 450,402 | 88.0675% |
| SmolTalk2 everyday conversations | 121 | 0.0237% |
| SmolTalk2 system chats | 32,776 | 6.4087% |
| OASST2 | 3,439 | 0.6724% |
| SODA | 24,690 | 4.8277% |
| **Total** | **511,428** | **100%** |

Two facts change the direction of this project:

1. The corpus is already overwhelmingly synthetic. SmolTalk describes itself as a synthetic SFT
   dataset. Synthetic data by itself is therefore not the missing ingredient.
2. The corpus was traversed monotonically in source order, without shuffle, while token-averaged
   teacher-forced loss gave very little relative influence to the first token of each response. That
   recipe learned some answer continuation but not a stable policy for starting an answer.

The existing corpus is structurally valid. It is not demonstrated to be a balanced curriculum for the
specific capabilities now desired. More generic synthetic instruction data would repeat the same
mistake at a different scale.

### 4.1 Evidence packet for external researchers

External researchers should use the following frozen evidence rather than reconstructing the story from
informal summaries:

| Evidence | Location or identifier | What it establishes |
|---|---|---|
| Program closeout | `HANDOFF.md` | Frozen outcome, publication state, archives, and no-run authority boundary |
| Current-state summary | `docs/resume/CURRENT-STATE.md` | Architecture, exact terminal metrics, and durable checkpoints |
| Failure analysis | `docs/resume/FAILURE-ANALYSIS.md` | Evidence for first-token EOS, prompt-length effect, source ordering, and ruled-out causes |
| Experiment backlog | `docs/resume/EXPERIMENT-BACKLOG.md` | Proposed future repairs and evidence gates |
| Acceptance gates | `docs/resume/ACCEPTANCE-GATES.md` | Existing execution and quality contracts |
| Evidence index | `docs/resume/EVIDENCE-INDEX.md` | Canonical reports, hashes, and run artifacts |
| SFT corpus contract | `docs/SFT_CORPUS.md` | Rendering, validation, exact corpus state, and completed outcome |
| Frozen evaluation | `docs/FROZEN_EVAL.md` | Prompt construction, metrics, and terminal evaluation contract |
| SFT manifest | `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt.manifest.json` | Source hashes, counts, output hash, and construction record |
| Terminal evidence | `/mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/` | Exact metrics, checkpoint, samples, and machine reports |
| Native continuation archive | `ajaxdavis/alpha-60m-training-checkpoints`, revision `7198d1a1f094ffe88d06399ea99fecbd78fa8b66` | Base, best surviving SFT, terminal SFT, optimizer/RNG, tokenizer, and audits |
| Public standard model | `ajaxdavis/alpha-60m-chat`, revision `b481f46924b7a4777a029de1ffb44c06cc925d4c` | Reproducible Transformers export of the failed terminal model |

The final SFT corpus SHA-256 is
`ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb`. The terminal native checkpoint
SHA-256 is `6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8`.

### 4.2 Base-data and capacity context

The base model is Llama-form with 16 layers, width 512, eight attention heads, a 12,288-token byte-BPE
vocabulary, and a 1,024-token context. It saw approximately one billion training tokens—not the entire
staged corpus.

The staged pretraining text was built from selected retained shards of
`HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled`: 1,857,705 accepted documents,
11,700,002,843 characters, and roughly three billion estimated tokens across six output shards. The
flagship run consumed exactly 1,000,013,824 tokens from its selected first three shards. This is broad
educational web/document data, not a controlled foundation in linguistics, dialogue, or ontology.

Researchers must therefore consider three distinct limits:

1. **Foundation limit:** one billion general pretraining tokens may not have established enough robust
   language competence for specialized SFT to reorganize.
2. **Capacity limit:** 57.7M parameters may not simultaneously retain general fluency and represent the
   proposed conceptual breadth.
3. **Post-training limit:** the data order, mixture, token weighting, and teacher-forced objective may
   have hidden competence that free generation could not access.

The current evidence does not causally separate these limits.

### 4.3 Evidence status: fact, diagnosis, and hypothesis

Third-party reviews must preserve these epistemic boundaries:

- **Verified facts:** exact run configuration, hashes, finite tensors, source ordering, 92 empty terminal
  outputs, prompt-length breakdown, and native/Transformers reproduction.
- **Strong diagnosis:** unstable first-token response initiation reconciles low teacher-forced validation
  loss with empty greedy generation; monotonic source blocks and token weighting are plausible major
  contributors.
- **Unproven causal claim:** that shuffling, answer-start weighting, or a new corpus would by itself fix
  the model. No controlled post-closeout ablation was run.
- **Research hypothesis:** that a targeted 200K accepted-unit curriculum can create the desired niche at
  this model scale.

Agents should challenge diagnoses with evidence, but must not rewrite hypotheses as established causes
or use the failed outcome to claim that the proposed method cannot work.

## 5. The core research question

> What compact mixture of natural conversations, linguistic contrasts, explanations, counterexamples,
> and ontology problems gives a roughly 58M-parameter model the best attainable conversational and
> conceptual competence without spending its limited capacity on broad factual recall, code, or
> machine-facing formats?

This question has several possible answers. Research agents must consider at least:

- targeted continued pretraining followed by a smaller SFT set;
- one integrated conversational curriculum;
- staged curricula from short response initiation to deeper dialogue;
- contrastive examples and explanations versus dialogue-only examples;
- whether 200,000 accepted units is too few, sufficient, or excessive for this model and tokenizer;
- whether 57.7M parameters are themselves the binding constraint.

No agent should assume that a large teacher can simply pour its knowledge into a small student. The
corpus must be designed for what this student can represent and reliably generate.

### 5.1 Competing hypotheses to keep alive

The research program should explicitly compare, rather than prematurely collapse, these explanations:

| ID | Hypothesis | Prediction | Evidence that would weaken it |
|---|---|---|---|
| H1 | Generic SFT content was misaligned with the intended niche | Targeted data improves niche evals at equal token budget | No advantage over a size-matched shuffled generic subset |
| H2 | Monotonic source order caused recency bias and forgetting | Shuffled/interleaved training is more stable across checkpoints | The same oscillation and endpoint occur after controlled interleaving |
| H3 | Token-averaged loss undertrained response starts | Episode or answer-start emphasis sharply reduces immediate EOS | EOS collapse persists despite verified start-token learning |
| H4 | The base model is under-pretrained | More or better domain foundation data helps more than SFT redesign | Strong gains from SFT redesign at the unchanged base checkpoint |
| H5 | The model is too small for the joint target | Individual niches work but the combined curriculum interferes | A controlled combined model matches all specialist subsets |
| H6 | Synthetic teacher style, not knowledge, dominated learning | Multi-teacher and style-controlled data improve robustness | Style diversity changes outputs but not capability scores |
| H7 | The tokenizer/context make analysis inefficient | Fragmentation and truncation predict errors by language/task | Errors remain after controlling length and tokenization burden |
| H8 | Greedy decoding magnified a narrow calibration defect | Alternative calibrated decoding recovers useful latent behavior | Non-greedy output remains empty, looping, or semantically poor |
| H9 | SFT is the wrong place to teach deep concepts | Targeted continued pretraining plus small SFT beats SFT-only | SFT-only matches it at equal data/compute |
| H10 | Evaluation design mistakes fluency for understanding | Contrastive and transfer tests reverse rankings from style judges | Rankings remain stable under blinded human and adversarial tests |

This table should evolve as agents find better explanations. Removing a hypothesis requires an explicit
adjudication entry and evidence, not consensus by repetition.

### 5.2 What would count as an interesting research result?

Positive and negative results are both valuable if the comparison is clean. Examples include:

- a targeted 20K subset outperforming 200K generic conversations on held-out compositional tasks;
- contrast sets improving minimal-pair sensitivity but hurting ordinary conversational naturalness;
- ontology dialogue transferring to unseen categorization problems rather than only learned terms;
- multi-teacher data reducing stylistic collapse without improving correctness;
- answer-start intervention eliminating empty responses while revealing a separate capacity ceiling;
- a finding that 57.7M parameters cannot sustain the joint target even though narrower specialists can;
- evidence that continued pretraining is necessary before conversational instruction can teach the
  desired conceptual behavior.

The program should publish failures and null results with the same hashes and provenance as successes.

## 6. What “200,000 examples” means

The target is **200,000 accepted, auditable training units after filtering**, not 200,000 unreviewed
teacher completions.

A training unit is one coherent learning episode. It may contain:

- a user question and an assistant answer;
- a two-to-six-turn conversation;
- two minimally different sentences followed by a natural-language comparison;
- an incorrect analysis followed by diagnosis and repair;
- an ambiguous utterance followed by two or more defensible readings;
- an ontology scenario followed by questions, distinctions, and counterexamples;
- a teach-back exchange in which the user tests or refines an explanation.

The generation pool will probably need to be several times larger than the accepted corpus. Candidate
volume must be set only after a small audited generation study measures rejection, duplication, and
revision rates. “We generated 200K” is not a completion criterion.

### 6.1 Strawman capability allocation

The following allocation is deliberately concrete so researchers have something falsifiable to
criticize. It is **not yet frozen**.

| Capability family | Accepted units | Primary purpose |
|---|---:|---|
| Conversational foundations and repair | 40,000 | Response initiation, relevance, turn-taking, clarification, disagreement, tone, follow-up |
| Morphology and syntax | 35,000 | Form, composition, grammaticality, dependencies, alternations, argument structure |
| Semantics and pragmatics | 30,000 | Meaning, ambiguity, inference, presupposition, implicature, reference, modality |
| Discourse, sociolinguistics, change, and acquisition | 20,000 | Coherence, information structure, register, variation, change, learning, interaction |
| Typology and cross-linguistic comparison | 25,000 | Language diversity without treating English categories as universal |
| Ontology and metaphysical analysis | 30,000 | Identity, category, relation, part, event, time, dependence, process, role, evidence |
| Ambiguity, contradiction, evidence, and provenance | 10,000 | Multiple analyses, conflicting claims, temporal qualification, epistemic restraint |
| Integrated research dialogues | 10,000 | Sustained synthesis across linguistic and ontological questions |
| **Total** | **200,000** | |

These are primary labels, not isolated silos. A dialogue about whether “the committee” denotes one
entity or many may primarily count as ontology while also exercising semantics, agreement, discourse,
and social categories. Every accepted unit should carry secondary coverage labels in its audit metadata.

### 6.2 Orthogonal format allocation

The same 200,000 units should also be audited by interaction form. A possible starting distribution is:

| Interaction form | Accepted units | Notes |
|---|---:|---|
| Natural two-to-six-turn conversations | 70,000 | Includes casual, exploratory, explanatory, and disagreement dialogues |
| Sentence contrasts and minimal pairs | 35,000 | Form or context changes one interpretation or acceptability judgment |
| Guided analysis and Socratic dialogue | 30,000 | Assistant helps the user discover a distinction rather than lecturing immediately |
| Definition, example, and counterexample sets | 20,000 | Operational understanding, not memorized dictionary prose |
| Ontology scenarios and competency questions | 20,000 | Natural-language modeling choices and their consequences |
| Error diagnosis, repair, and revision | 15,000 | User or assistant starts with a plausible mistake |
| Multiple-valid-analysis cases | 10,000 | Explicitly preserves ambiguity, theory-dependence, or incomplete evidence |
| **Total** | **200,000** | |

Capability and format are two independent views of the same units. They must not be added together.

### 6.3 Illustrative unit sketches

These sketches are **not approved training rows**. They clarify the intended intellectual and
conversational shape for researchers designing or criticizing the corpus.

#### A. Ordinary conversational repair

**User:** I keep using “ontology” when I think I mean “taxonomy.” Am I mixing them up?  
**Assistant behavior:** Answer directly, preserve the user's underlying idea, distinguish a hierarchy of
labels from an account of entities and relations, offer one ordinary example, and invite the user to test
the distinction on their project. Do not begin with a history of philosophy.

Why it belongs: conversational tact, terminology repair, concise ontology, and a useful follow-up.

#### B. Minimal syntactic contrast

**User:** Why does “Which book did Mira say that Lee bought?” sound ordinary, but “Which book did Mira
meet the person who bought?” often sounds worse?  
**Assistant behavior:** Notice the extraction contrast, explain the relevant structural constraint in
plain language, acknowledge dialect/context and theory differences, and draw a simple bracketing or
substitution test without pretending one label is the explanation.

Why it belongs: structure-sensitive reasoning, gradient judgment, and explanation by evidence.

#### C. Morphology without English as the template

**User:** If one word can express what English needs a whole sentence for, is it still really one word?  
**Assistant behavior:** Separate phonological, morphological, and orthographic notions of “word”; explain
that diagnostics can disagree; use sourced or transparently schematic examples; avoid presenting
polysynthesis as exotic compression.

Why it belongs: cross-linguistic category caution and multiple diagnostics.

#### D. Lexical ambiguity versus referential uncertainty

**User:** In “I went to the bank,” is the sentence ambiguous or do I just not know which bank you mean?  
**Assistant behavior:** Compare lexical readings with uncertainty among referents under one reading,
show how context can resolve each, and admit that the distinction can blur in real lexical analysis.

Why it belongs: a language–ontology bridge that generalizes beyond a memorized definition.

#### E. Presupposition under disagreement

**User:** If I say “Nila stopped singing,” what have I actually claimed?  
**Assistant behavior:** Distinguish the asserted change from the usual background assumption that Nila
had been singing, test both under negation or questioning, and note that context can challenge or
accommodate the background.

Why it belongs: inference types, conversational context, and compact diagnostic reasoning.

#### F. Type, instance, and role

**User:** Is “student” a kind of person or just something a person is doing for a while?  
**Assistant behavior:** Explain why a modeling system might treat student as a role borne by a person,
contrast that with ordinary class language, explore temporal change and institutional dependence, and
ask what questions the user's ontology must support.

Why it belongs: ontology as a modeling choice connected to time and use, not label memorization.

#### G. Identity through change

**User:** If every plank in a boat is replaced, when did it stop being the same boat?  
**Assistant behavior:** Map several criteria of identity, explain why the facts may underdetermine one
answer, connect each criterion to practical consequences, and avoid treating the familiar thought
experiment as proof of a preferred metaphysics.

Why it belongs: sustained but chatty reasoning with genuinely alternative analyses.

#### H. Part, member, and material

**User:** Is a violinist part of an orchestra in the same way a wheel is part of a bicycle?  
**Assistant behavior:** Distinguish membership from component parthood, show why both use “part” in
ordinary speech, examine persistence when the member leaves, and produce a counterexample to an overly
simple transitivity rule.

Why it belongs: relation properties, polysemy, and counterexample construction.

#### I. Contradictory sourced claims

**User:** One catalogue says the photograph was taken in 1912 and another says 1914. Which date should
the database store?  
**Assistant behavior:** Preserve both sourced claims, distinguish event time from record time, ask what
evidence supports each, explain confidence and possible reconciliation, and resist silently choosing a
winner.

Why it belongs: evidence-first ontology and useful real-world conversation.

#### J. Translation and category mismatch

**User:** Why can't a dictionary just give one exact word in the other language?  
**Assistant behavior:** Explain differences in sense boundaries, grammar, register, cultural practice,
and context; use careful examples; distinguish “no one-word equivalent” from “cannot be translated.”

Why it belongs: lexical ontology, pragmatics, typology, and conversational myth correction.

#### K. User challenges the assistant

**User:** You called that ungrammatical, but everyone in my family says it.  
**Assistant behavior:** Reconsider the claim, distinguish variety-specific grammar from a prestige norm,
ask for context, and correct itself without defensiveness or empty praise.

Why it belongs: repair, sociolinguistic humility, and responsiveness to counterevidence.

#### L. Teach-back and productive transfer

**Assistant:** A role depends on a context in a way the bearer need not. Can you think of another example
besides “student”?  
**User:** Maybe “tenant,” because the person stays the same after moving out?  
**Assistant behavior:** Validate the structural insight, refine the relevant institutional relation, and
offer a near-counterexample that tests whether the user now understands the distinction.

Why it belongs: multi-turn learning and transfer, not one-way exposition.

### 6.4 Anti-examples

Reject or heavily revise units with these patterns:

- a user question followed by a polished encyclopedia article that never acknowledges the question's
  wording or purpose;
- ten superficial paraphrases labeled as ten different examples;
- confident statements about an actual language with no attested source;
- a minimal pair whose two sentences differ in several uncontrolled ways;
- an ontology answer that declares a category “correct” without a competency question or assumptions;
- an ambiguity example whose alleged readings are not independently plausible;
- a dialogue in which every assistant turn begins with the same praise or summary phrase;
- content made longer to satisfy a token target;
- an answer that teaches terminology but cannot apply it to a fresh case;
- an apparently balanced debate in which one side is a fabricated or harmful straw person;
- source text lightly paraphrased beyond the license or presented without provenance;
- a judge-approved example retained despite knowledgeable human disagreement that was never recorded.

## 7. Model-visible curriculum design

### 7.1 Conversation should be learned as behavior

The dataset must show the model how good conversation unfolds, not simply attach a user marker to an
essay. It should contain:

- direct answers before elaboration;
- context-sensitive acknowledgements without repetitive filler;
- appropriate follow-up questions when a real ambiguity blocks an answer;
- willing, specific disagreement;
- collaborative reformulation when the user is searching for a term;
- corrections that preserve the useful part of the user's thought;
- brief answers when the question is brief;
- deeper answers when the user invites analysis;
- references to earlier turns without copying them;
- graceful recovery after interruption, misunderstanding, or topic shift;
- uncertainty stated precisely rather than as ritual hedging;
- examples invented transparently and facts distinguished from illustrations.

Avoid a synthetic house style. In particular, the accepted set should not be dominated by headings,
three-item lists, “Certainly!”, “Great question”, “As an AI”, repeated conclusions, or identical answer
openings. These regularities are easy for a small model to memorize and can become generation attractors.

### 7.2 Protect answer initiation

Because terminal Alpha usually chose EOS at the first assistant position, the curriculum must make
response starts a first-class measured object.

Required properties include:

- every accepted conversation has a substantive, nonempty assistant start;
- opening tokens and opening constructions are diverse;
- short, complete answers are common enough not to be drowned out by long answers;
- prompts span conversational moves, styles, and lengths;
- repeated assistant prefixes are measured and capped;
- answer-start accuracy and EOS margin are evaluated separately from average token loss;
- examples are weighted or sampled as episodes, not allowed to matter only in proportion to answer
  length.

The final implementation may use curriculum scheduling, loss weighting, sampling, or another mechanism.
This brief defines the required behavior, not the code solution.

### 7.3 Proposed length bands

The exact bands require tokenizer measurement, but the accepted corpus should deliberately cover:

| User-side prompt length | Proposed share |
|---|---:|
| 1–64 tokens | 45% |
| 65–160 tokens | 35% |
| 161–300 tokens | 15% |
| 301–700 tokens | 5% |

| Assistant answer length | Proposed share |
|---|---:|
| 8–40 tokens | 30% |
| 41–140 tokens | 45% |
| 141–320 tokens | 20% |
| 321–650 tokens | 5% |

These are not invitations to pad answers. They prevent the corpus from equating quality with length and
ensure that short conversational turns have material training weight. Complete units should fit the
1,024-token context without dropping earlier turns. Any future context-bound trimming must be audited;
silent prefix trimming is unacceptable for this curriculum.

## 8. Linguistic coverage specification

“All the complexities of linguistics” cannot literally be exhausted in 200,000 examples. The achievable
requirement is systematic coverage of major levels of analysis, their interactions, and the places
where languages differ. Research track R2 must turn this section into a defensible coverage ontology.

### 8.1 Sound, writing, and form

- articulatory and acoustic phonetic distinctions expressed accurately in text;
- phonemes, allophones, contrast, neutralization, and distribution;
- syllable structure, stress, tone, intonation, rhythm, and prosody;
- phonological processes and rule/constraint alternatives;
- sound-symbol and writing-system distinctions;
- orthography versus pronunciation;
- grapheme, character, segment, and transliteration distinctions;
- morphemes, roots, stems, affixes, clitics, reduplication, and suppletion;
- inflection versus derivation and the cases that challenge that distinction;
- isolating, agglutinative, fusional, polysynthetic, and non-concatenative patterns without treating
  typological labels as rigid boxes.

Text-only representation cannot teach acoustic perception or production. Examples must state that
modality limit rather than pretending written descriptions are speech data.

### 8.2 Morphosyntax and syntax

- word classes and category diagnostics;
- constituency and dependency perspectives;
- agreement, concord, case, alignment, and indexation;
- argument structure, valency, transitivity, voice, and applicatives;
- word order and information-structure interactions;
- negation, interrogation, relativization, coordination, and subordination;
- control, raising, ellipsis, displacement, locality, and binding;
- tense, aspect, mood, modality, and evidentiality;
- definiteness, specificity, number, gender, classifiers, and noun classes;
- grammaticalization and constructional alternatives;
- gradient acceptability, dialect variation, context dependence, and theory-dependent analyses.

The corpus should favor diagnosis over label recitation: what evidence would distinguish two parses,
what changes under a minimal contrast, and what remains unresolved?

### 8.3 Semantics and pragmatics

- lexical sense, polysemy, homonymy, synonymy, antonymy, and lexical fields;
- compositionality and non-compositional expressions;
- reference, denotation, predication, and deixis;
- scope, quantification, negation, modality, and intensional contexts;
- events, states, participants, thematic roles, aspect, and temporal interpretation;
- entailment, contradiction, presupposition, conventional implicature, and conversational implicature;
- anaphora, coreference, bridging, and discourse reference;
- ambiguity and underspecification at lexical, syntactic, semantic, and pragmatic levels;
- metaphor, metonymy, irony, indirect speech acts, politeness, and common ground;
- context change, accommodation, repair, and meaning negotiation.

Examples must distinguish “one likely interpretation” from “the sentence can only mean this.” Multiple
valid readings are a feature, not noise to be removed.

### 8.4 Discourse, society, history, and learning

- coherence, cohesion, topic, focus, givenness, contrast, and rhetorical relations;
- turn-taking, adjacency pairs, repair, grounding, and conversational inference;
- register, style, genre, audience design, and stance;
- dialect, sociolect, ethnolect, idiolect, code-switching, and translanguaging;
- descriptivism versus prescriptivism and the social power behind “correctness” claims;
- language contact, borrowing, convergence, divergence, and language change;
- comparative reconstruction and the limits of historical inference;
- first- and additional-language acquisition;
- processing difficulty versus grammaticality;
- signed and spoken languages without treating sign as encoded speech;
- language documentation, vitality, reclamation, authority, and consent.

### 8.5 Typological safeguards

English should not silently supply the universal template. The curriculum should cover genuinely
different systems of alignment, constituent order, morphology, reference, information structure,
evidentiality, classification, and modality.

However, typological breadth must not be manufactured by inventing unattested sentences in languages
the teacher does not reliably know. For low-resource, Indigenous, signed, and minoritized languages:

- use attested examples with clear source and license, or explicitly labeled constructed examples
  reviewed by someone with appropriate authority;
- preserve the source's analysis and uncertainty;
- do not present a community, language, or variety as homogeneous;
- distinguish public availability from ethical permission to use;
- never turn sacred, restricted, personally sensitive, or community-governed material into synthetic
  training data without authorization.

## 9. Ontology coverage specification

Ontology here means both careful reasoning about what kinds of things there are and practical knowledge
organization. It does not mean teaching the model to emit OWL, RDF, JSON-LD, SQL, or another formal
syntax. Formal ideas should appear through natural-language cases, questions, explanations, and
counterexamples.

### 9.1 Core distinctions

- type and token; class and instance;
- universal and particular;
- identity, similarity, equivalence, and representation;
- individuation and criteria of identity;
- category boundaries, prototypes, family resemblance, and graded membership;
- essential versus accidental properties;
- intrinsic versus relational properties;
- concrete, abstract, fictional, hypothetical, and information entities;
- continuants and occurrents; objects, events, processes, and states;
- roles, functions, dispositions, capacities, qualities, and realizations;
- dependence, constitution, participation, inherence, and realization.

### 9.2 Structure, relation, and change

- parthood, proper parthood, overlap, boundaries, collectives, and sums;
- spatial, temporal, causal, comparative, and social relations;
- relation arity, direction, symmetry, asymmetry, transitivity, and inverses;
- time, change, persistence, stages, histories, and temporal parts;
- event identity, causal chains, enabling conditions, and prevention;
- modality, possibility, necessity, counterfactuals, and dispositions;
- granularity and levels of description;
- the difference between a thing, a record of it, a name for it, and a claim about it.

### 9.3 Knowledge and contested reality

- observation, testimony, inference, hypothesis, and definition;
- claim, evidence, source, provenance, confidence, and authority;
- truth at a time versus when a claim was recorded;
- incomplete information and the open-world assumption;
- contradiction without immediate collapse or winner-take-all deletion;
- retraction, supersession, correction, and reinterpretation;
- disagreement caused by evidence versus vocabulary, granularity, time, or perspective;
- ontology alignment and the difference between synonymy, near-equivalence, and relatedness;
- polysemy versus mistaken entity merger;
- competency questions: what a categorization must let a user ask and distinguish.

The curriculum should not pretend that one upper ontology settles metaphysics. It should teach how a
framework's commitments help with some questions and distort others.

### 9.4 Language–ontology bridge

High-value integrated topics include:

- whether a noun phrase implies an entity;
- mass/count alternations and individuation;
- plural reference, groups, and collectives;
- event nominalization and reification;
- lexical polysemy versus multiple entities;
- names, descriptions, reference, and identity over time;
- tense/aspect choices and event boundaries;
- social categories, roles, institutions, and changing criteria;
- translation mismatches caused by incompatible lexical or conceptual partitions;
- when two datasets disagree because of ontology rather than factual error;
- how evidential and modal marking changes what a speaker commits to.

This bridge is likely the most distinctive part of the project and should receive explicit evaluation,
not remain an incidental overlap.

## 10. Grounding and source policy

Synthetic generation should combine the flexibility of a frontier teacher with the discipline of
attested references. Teacher memory alone is not a source.

### 10.1 Source classes

Each candidate unit should be associated with one of these non-model-visible provenance classes:

1. **Attested:** grounded in a cited, licensed primary example or dataset record.
2. **Constructed from attested pattern:** new surface wording generated from a sourced phenomenon;
   construction and transformation are recorded.
3. **Illustrative hypothetical:** an invented case used to explain a general distinction, clearly not
   represented as an attested fact about a language or community.
4. **Analytical synthesis:** compares multiple cited analyses or frameworks.
5. **Conversational behavior:** a non-factual interaction designed to teach turn-taking, repair, tone,
   or explanation style.

These statuses must never be silently collapsed.

### 10.2 Candidate reference families

The following are starting points for source research, not blanket approval to copy:

- [Universal Dependencies guidelines](https://universaldependencies.org/guidelines.html) for a
  cross-linguistic morphological and syntactic inventory;
- [World Atlas of Language Structures](https://wals.info/feature) for typological feature coverage;
- [OntoLex-Lemon](https://www.w3.org/2016/04/ontolex/) for the language–ontology interface;
- [Basic Formal Ontology](https://bfo-ontology.github.io/bfo-2020.html) as one explicit upper-ontology
  tradition to analyze and contrast;
- [OWL 2 Direct Semantics](https://www.w3.org/TR/owl2-direct-semantics/) for formal-semantic concepts
  that can be translated into natural-language competency questions;
- [OBO Foundry principles](https://obofoundry.org/principles/fp-000-summary.html) for ontology
  governance and interoperability questions.

Every dataset, treebank, article, grammar, ontology, and example collection requires its own license and
use audit. A project's website license does not automatically cover every contributed item. Citation is
required but does not replace permission.

### 10.3 Provenance record

For every candidate and accepted unit, preserve outside the model-visible conversation:

- stable unit identifier and version;
- primary and secondary capability labels;
- interaction form and difficulty;
- language or variety, where applicable;
- attested, constructed, hypothetical, synthesis, or behavior status;
- source URL, bibliographic identifier, license, and exact source location where applicable;
- teacher provider, exact model/version, generation date, prompt-template version, and seed;
- judge and reviser identities/versions;
- transformations from source to candidate;
- automated filter results;
- reviewer decision, rationale, and revision history;
- train, development, or frozen-evaluation assignment;
- content hash and near-duplicate cluster identifier.

The physical encoding within and around the canonical SQLite artifact remains an implementation
decision. Provenance must be append-only or versioned so a revised unit does not erase its history.

### 10.4 Dataset-substrate decisions

**Decision D1 — SQLite is canonical.** The final novel dataset will be represented in a versioned SQLite
database. Flat chat text, model token arrays, Hugging Face exports, sample packs, and manifests will be
deterministic, content-hashed **derivatives** of a sealed database release. They will not become
independent sources of truth.

**Decision D2 — content is delimiter-independent.** Canonical message text will never contain a training
delimiter merely because the current model expects one. User, assistant, system, quoted-speaker, and
other discourse roles live in relational fields. A versioned export renderer injects delimiters such as
`<|user|>`, `<|assistant|>`, and `<|end_of_text|>` only after a release/cohort is selected for a specific
model and tokenizer. A future model can render the same semantic conversation with a different chat
template without editing, parsing, or cleaning the source utterances.

**Decision D3 — rich description is part of acceptance.** Every accepted training unit must be described
structurally, linguistically, ontologically, pedagogically, and in provenance/review terms. Description
may include competing annotations rather than one forced answer. Annotation completeness will be a
release gate, not optional documentation added after training.

These decisions do not authorize creating the database now. R5, R7, R8, R9, and R11 must review and
amend the logical design before implementation.

The database exists to answer questions such as:

- Which accepted units teach implicature through disagreement rather than definition?
- Which sentences contain a quantified noun phrase, an anaphor, and two defensible scope readings?
- Which ontology examples involve roles that change over time?
- Which exact source span supports a claim about a real language?
- Which units were authored by one teacher and approved only by a correlated judge?
- Which answer openings, sentence structures, lexical items, delimiters, or content-token sequences
  dominate a rendered export?
- Which examples were rejected, revised, or disputed, and why?
- Which train release and exact unit revisions did a model checkpoint actually see?
- How often did a run expose the model to each capability, language, interaction form, sentence type,
  token type, and difficulty band?
- Can a future run select only short conversational repairs involving ambiguity, or exclude every unit
  with uncertain licensing, without rebuilding history by guesswork?

The database should represent disagreement. A sentence can receive competing analyses from different
annotators or theories. A unit can be useful for several phenomena. An annotation can be uncertain,
superseded, or rejected without deleting the earlier judgment.

### 10.5 Logical object flow

```text
source document -> source span -> generation task -> candidate unit -> unit revision
                                                              |
                                                              +-> messages -> sentences -> tokenizations
                                                              |
                                                              +-> annotations -> reviews -> decisions
                                                              |
                                                              +-> duplicate clusters / provenance edges

accepted unit revisions -> dataset release -> curriculum cohort -> rendered export -> model run
```

The arrows mean traceable relationships, not destructive conversion. Source material, raw candidates,
revisions, rejected candidates, and release membership remain independently inspectable.

### 10.6 Proposed table families

This is a logical schema inventory, not SQL DDL. Names and boundaries are open to specialist review, but
an implementation should not collapse semantically different records into one generic metadata blob.

#### A. Database governance and releases

| Table | Core records | Purpose |
|---|---|---|
| `schema_version` | migration/version, applied time, implementation commit, migration hash | Reconstruct the exact database interpretation |
| `dataset_project` | stable project ID, title, intent, authority policy | Separate this curriculum from future dataset programs |
| `dataset_release` | release ID, semantic version, status, parent release, seal time, content hash | Define immutable candidate, review, pilot, and training releases |
| `release_member` | release, unit revision, inclusion role, order key, sampling weight | Materialize exact membership without relying on a live query |
| `release_artifact` | release, artifact type, path/URI, byte count, hash, builder version | Link flat exports, manifests, reports, and token packs to their source release |
| `decision_record` | decision ID, scope, status, rationale, proposer, approver, time | Preserve why a design or inclusion decision changed |
| `audit_event` | actor, action, object type/ID, previous/new version, time, reason | Append-only history of material database changes |

Release states should distinguish at least working, candidate, reviewed, sealed, superseded, and
withdrawn. “Superseded” must not delete the old release. “Withdrawn” needs a recorded legal, ethical,
privacy, or quality reason and a tombstone that prevents accidental future export.

#### B. Units, conversations, and text revisions

| Table | Core records | Purpose |
|---|---|---|
| `unit` | stable unit ID, project, creation origin, current state | Identity of one learning episode across revisions |
| `unit_revision` | revision ID, unit, parent revision, content hash, created time, revision reason | Immutable version of a candidate or accepted unit |
| `conversation` | revision, conversation kind, turn count, declared setting | Conversation-level structure without flattening messages |
| `message` | revision, ordinal, semantic role, raw utterance text, text hash, language assertion | Preserve each turn without model-specific delimiters |
| `sentence` | message, ordinal, exact text, character/byte offsets, segmentation method/version | Query and annotate sentences while preserving original text |
| `text_span` | message or sentence, start/end offsets, span text hash | Anchor linguistic, ontology, source, and review annotations precisely |
| `unit_relation` | subject unit, relation concept, object unit, provenance | Link contrast partners, paraphrases, repairs, prerequisites, and adversarial variants |

Stable unit identity must be separate from a revision. If a reviewer fixes one word, the old text and all
annotations tied to it remain valid historical records; the new revision receives a new content hash and
its own review state.

Canonical message text stores only the utterance. For example, an assistant message stores `A role can
change while its bearer remains the same.`—not `<|assistant|>A role can change...<|end_of_text|>`.
System messages should be rare and explicitly justified. The model-visible “no JSON/code/tool format”
boundary applies to utterance text, not to internal relational metadata.

#### C. Renderers, delimiters, tokenizers, and supervision

| Table | Core records | Purpose |
|---|---|---|
| `renderer` | renderer ID/version, implementation/hash, supported conversation contract | Version the transformation from semantic messages to model input |
| `rendering_profile` | renderer, target model family, role-order policy, BOS/EOS policy | Declare one model-specific serialization without changing source text |
| `delimiter_definition` | profile, semantic role/boundary, emitted bytes, tokenizer token IDs | Track exactly what is injected at each boundary |
| `render_event` | unit revision, profile, output hash, byte count, status | Bind one unit to exact rendered bytes |
| `render_segment` | event, ordinal, segment kind, source message/span or delimiter, byte offsets | Distinguish injected scaffolding from content in the rendered stream |
| `tokenizer` | stable tokenizer ID, name, vocabulary size, model/config hashes, special-token policy | Version the exact tokenizer rather than naming it informally |
| `token_type` | tokenizer, token ID, exact byte sequence/display form, special-token class | Define each vocabulary item once per tokenizer |
| `tokenization` | render event or raw message, tokenizer, token count, sequence hash | Identify one reproducible tokenization event |
| `token_occurrence` | tokenization, ordinal, token type, byte/character span, segment kind, role, supervised flag | Permit exact token-level audits and future selection |
| `token_sequence_blob` | tokenization, encoding, compressed IDs/weights, hash | Efficient exact reconstruction alongside queryable occurrences |
| `tokenization_summary` | tokenization, delimiter/content counts, fragmentation metrics, error flags | Fast corpus analysis without scanning every occurrence |
| `supervision_span` | render event, start/end token ordinal, target role, weight policy, rationale | Record what a future training export supervises |
| `token_statistic` | release/cohort/export, tokenizer, token type or family, count, document frequency | Audit domination, rarity, delimiters, and tokenizer inequality |

This separation lets the database answer both “what did the person/assistant say?” and “what exact byte
and token sequence did run X consume?” Delimiters are injected data with their own provenance, not a
linguistic property of the utterance.

For 200,000 units, full token occurrences may mean tens of millions of rows. That is feasible in SQLite
but must be benchmarked. A compact sequence blob alone is not sufficient because it cannot answer the
desired token-level questions; an occurrence table alone may be wasteful. The proposed dual
representation gives exact compact reconstruction plus relational auditability. Both forms must hash to
the same token sequence, and redundant secondary indexes should be chosen by measured queries rather
than indexed indiscriminately.

Token records are tokenizer-relative. The original byte text remains canonical; re-tokenizing with a
future tokenizer creates new `tokenization` rows and never overwrites the old sequence.

#### D. Flexible linguistic, ontological, and pedagogical description

| Table | Core records | Purpose |
|---|---|---|
| `concept_scheme` | scheme ID, name, version, scope, maintainer, source/license | Hold Alpha, UD, typological, ontology, rubric, and imported vocabularies separately |
| `concept` | concept ID, scheme, preferred label, definition, status, provenance | Represent a phenomenon, capability, sentence type, speech act, ontology notion, or risk |
| `concept_label` | concept, language, label, label type, source | Support synonyms and multilingual labels without string-case logic |
| `concept_relation` | subject concept, relation concept, object concept, status, provenance | Build broader/narrower, prerequisite, contrast, overlap, and alignment graphs |
| `annotation` | target object/span, concept, assertion type, confidence, annotator, method, time | Attach concepts to units, messages, sentences, spans, or tokenizations |
| `annotation_evidence` | annotation, source span/review/judgment, support type, note | Explain why an annotation was asserted |
| `annotation_relation` | subject annotation, relation, object annotation | Express agreement, contradiction, refinement, or supersession among analyses |
| `competency_question` | concept(s), natural-language question, expected distinction, provenance | Connect categories to behavior the model should demonstrate |
| `learning_objective` | objective ID, behavior, difficulty, prerequisite concepts, success evidence | State why a unit belongs in training |
| `unit_objective` | unit revision, objective, primary/secondary role, strength | Permit several explicit pedagogical uses per unit |

Some analyses have internal structure that should not be flattened into tags:

| Table | Core records | Purpose |
|---|---|---|
| `analysis` | target, analysis kind, theory/scheme, annotator, method/version, confidence, status | Identity for one parse, reading, discourse analysis, or ontology interpretation |
| `analysis_node` | analysis, node ordinal, anchored span or abstract node, concept | Represent constituents, tokens, implicit arguments, entities, events, or discourse units |
| `analysis_edge` | analysis, source node, relation concept, target node, confidence | Represent dependencies, constituency links, semantic roles, discourse relations, or ontology relations |
| `interpretation` | target utterance/sentence, paraphrase, conditions, likelihood/status, analysis | Preserve distinct readings without forcing one gold meaning |
| `interpretation_relation` | two interpretations, compatible/contrast/refinement relation, rationale | Express ambiguity structure and theory disagreement |
| `mention` | analysis, span, mention type, form/head features | Anchor references and predications in text |
| `reference_chain` | analysis, chain type, discourse referent hypothesis | Group coreferential, bridging, split, or disputed mentions |
| `claim` | unit/message/span, proposition text, epistemic status, source/analysis | Separate what a dialogue claims from its surface sentence |
| `claim_relation` | subject claim, support/attack/entail/contradict/refine relation, object claim | Model reasoning and contradiction inside an example |
| `temporal_qualification` | target claim/entity/relation, valid interval, record interval, uncertainty | Describe time-sensitive ontology and provenance cases |
| `ontology_commitment` | analysis/unit, commitment concept, scope, explicit/implicit status, rationale | Record what an example assumes exists or distinguishes |

An `analysis` is always theory- and annotator-relative. For example, dependency and constituency parses
can coexist; two coreference analyses can disagree; an ontology reading can reify an event while another
treats the phrasing as a convenient description. The database must not manufacture reconciliation by
putting one analysis into privileged columns and the others into notes.

This concept graph prevents a brittle fixed column or enum for every conceivable sentence type. New
research can add a concept scheme or align two schemes without migrating the whole unit table. The
database must still validate scheme versions and relationship types; “flexible” must not mean an
uninspectable tag soup.

Concept schemes should cover at least:

- interaction forms, discourse functions, and speech acts;
- linguistic levels, phenomena, constructions, and diagnostics;
- morphosyntactic features and sentence/clause types;
- semantic, pragmatic, rhetorical, and coreference relations;
- repair, stance, register, variation, and social context;
- language, variety, modality, and attestation descriptors;
- ontology distinctions, entity/relation types, and competency questions;
- reasoning operations such as compare, infer, disambiguate, counterexample, revise, and teach back;
- difficulty, prerequisite, transfer, and compositionality dimensions;
- provenance status, source risk, cultural authority, and restriction;
- quality rubrics and known failure modes.

An annotation must identify its agent: human reviewer, named model/version, deterministic analyzer, or
imported source. Conflicting annotations can coexist until an adjudication explicitly relates them.

#### E. Sources, licenses, attestations, and provenance

| Table | Core records | Purpose |
|---|---|---|
| `source_work` | title, authors/maintainers, edition/version, identifiers, canonical URL | Bibliographic identity of a work or dataset |
| `source_file` | work, file/version, acquisition time, byte count, content hash, storage location | Pin the exact acquired object |
| `source_fragment` | file, locator, offsets/page/row, exact fragment hash, access class | Anchor an attested example or claim |
| `license` | identifier, version, terms URL/text hash, redistribution/derivative flags, reviewer | Record the reviewed legal instrument |
| `source_license` | work/file/fragment, license, jurisdiction/scope, effective time, confidence | Avoid assuming one license covers every contributed item |
| `authority_record` | community/custodian, material scope, permission status, conditions, evidence | Track ethical or community authority separately from copyright |
| `provenance_assertion` | subject object, predicate concept, object/source, actor, method, time | Express traceable derivation and support relationships |
| `source_claim` | fragment, normalized claim text, status, reviewer | Separate what a source says from what a teacher inferred |
| `restriction` | target, restriction type, export/training limits, effective time, rationale | Fail closed on sensitive, withdrawn, or unapproved material |

Raw copyrighted or restricted source text need not be embedded when the permission contract allows only
a locator and hash. The database should store the minimum material needed for verification under the
applicable authority and access rules.

#### F. Generation, prompts, teachers, and candidates

| Table | Core records | Purpose |
|---|---|---|
| `model_provider` | provider identity, terms version, access class | Separate provider from model version |
| `model_version` | provider, exact model/version, release date, capability notes, terms/output policy | Pin authors, judges, and revisers |
| `prompt_template` | stable template, immutable revision, text hash, intent, variables contract | Version every generation and judging instruction |
| `generation_campaign` | design, target coverage, budget ceiling, stop rule, authorization | Define a bounded candidate study or corpus campaign |
| `generation_task` | campaign, seed/source inputs, requested concepts, template revision, random seed | Preserve exactly what was asked |
| `generation_attempt` | task, model, parameters, start/end, raw response hash, status, cost/usage | Retain success, empty response, refusal, and failure |
| `candidate_origin` | unit revision, generation attempt, extraction method | Link a parsed candidate to its raw authoring event |
| `revision_attempt` | input revision, reviser, critique, proposed output, accepted result | Preserve adversarial and editorial transformations |

Teacher outputs must be stored before normalization so parsing or rendering changes can be audited. A
model that generated an example must never be recorded merely as “SOTA teacher”; exact provider,
version, prompt, parameters, time, and applicable terms are required.

#### G. Review, judging, disagreement, and decisions

| Table | Core records | Purpose |
|---|---|---|
| `reviewer` | human/model/tool identity, qualifications/version, independence group | Make correlated review visible |
| `rubric` | immutable rubric revision, scope, creator, hash | Version evaluation criteria |
| `rubric_dimension` | rubric, dimension concept, scale, anchors, hard-fail policy | Define what scores mean |
| `review_assignment` | target revision, reviewer, rubric, blind group, assignment time | Record who was asked to judge what |
| `judgment` | assignment, overall disposition, confidence, rationale, completion time | Preserve accept/revise/reject/abstain decisions |
| `judgment_score` | judgment, rubric dimension, value, evidence span/note | Query correctness, naturalness, grounding, and other dimensions separately |
| `review_disagreement` | related judgments, disagreement type, severity, status | Prevent averages from hiding substantive conflict |
| `adjudication` | target, inputs considered, decision, actor, rationale, supersedes | Resolve when necessary without deleting earlier judgments |
| `acceptance_decision` | unit revision, release scope, disposition, gate set, adjudication | Keep general quality separate from release-specific inclusion |

Human reviewers must be able to abstain or mark insufficient authority. A majority model vote is not an
adjudication when all models inherit the same unsupported claim.

#### H. Deduplication, leakage, and similarity

| Table | Core records | Purpose |
|---|---|---|
| `content_signature` | target, algorithm/version, signature/hash | Exact, n-gram, MinHash, and embedding signatures |
| `similarity_edge` | two targets, method/version, score, comparison scope | Preserve semantic and structural near-duplicate evidence |
| `duplicate_cluster` | cluster ID, method/version, representative, status | Group candidate families for selection caps |
| `cluster_member` | cluster, target, membership score, decision | Inspect retained and rejected paraphrases |
| `leakage_rule` | rule revision, protected split/source/template relation, threshold | Version contamination policy |
| `leakage_finding` | target pair/cluster, rule, score, disposition, reviewer | Record why an item moved or was excluded |

Similarity is method-relative, not truth. Retain the model/embedding version and threshold; do not replace
humanly meaningful relations such as “minimal contrast partner” with an opaque duplicate label.

#### I. Splits, cohorts, curricula, and deterministic exports

| Table | Core records | Purpose |
|---|---|---|
| `split_definition` | split ID/version, purpose, freeze state, isolation policy | Define train, development, audit, and other partitions |
| `split_assignment` | unit revision, split, assignment reason, group key | Freeze membership at scenario/source/template-family level |
| `cohort` | release, cohort name/version, purpose, selection specification hash | Name a reusable subset for an ablation or curriculum |
| `cohort_member` | cohort, unit revision, sampling weight, order key, inclusion reason | Materialize exact selection instead of rerunning a mutable query |
| `curriculum_stage` | curriculum version, ordinal, cohort, target share, rehearsal policy | Represent stages and interleaving declaratively |
| `export_recipe` | renderer/profile/tokenizer/supervision versions, ordering, limits, hash | Pin how relational records become model input |
| `export_run` | recipe, release/cohort, artifact, counts, content hash, verification status | Produce reproducible flat text or token tensors |
| `export_member` | export, unit revision, render event/hash, ordinal, tokenization | Trace every training row back into SQLite |

Saved selection text alone is not enough because the database can evolve. Every cohort and release must
materialize member revision IDs and hash the ordered membership.

#### J. Future training-run lineage and exposure

| Table | Core records | Purpose |
|---|---|---|
| `training_run` | run ID, source commit, architecture/config hashes, base checkpoint, authorization | Bind model work to exact code and authority |
| `run_data_binding` | run, release/export/curriculum, manifest hash, tokenizer | State exactly what data contract the run used |
| `run_checkpoint` | run, step/tokens, artifact hash, metrics link, selection status | Connect data exposure and observed behavior |
| `unit_exposure` | run/checkpoint window, unit revision, presentations, supervised tokens, weight | Measure what the model actually saw |
| `concept_exposure` | run/checkpoint window, concept, units, presentations, supervised tokens | Aggregate exposure by capability, sentence, or ontology type |
| `token_exposure` | run/checkpoint window, tokenizer/token type, count, supervised count | Audit actual token distribution rather than planned distribution |
| `evaluation_run` | checkpoint, frozen suite revision, decoder contract, judge set, report hash | Make behavioral selection reproducible |
| `data_effect_finding` | run comparison, hypothesis, estimate, uncertainty, evidence artifact | Preserve conclusions and null results from ablations |

The corpus database should record run lineage, but private frozen evaluation **content** should live in a
separately access-controlled database or sealed artifact. The training database may store its identifier,
version, and evaluation result hashes without exposing prompts to generation or curriculum agents.

### 10.7 Annotation-completeness contract

“Fully described” cannot responsibly mean that every sentence has one unquestionably correct parse or
ontology. It means that every accepted unit passes a declared description profile, and every omission or
uncertainty is visible.

The minimum proposed profile is:

| Level | Required description for every accepted unit |
|---|---|
| Structural | Unit kind, message roles/order, sentence boundaries, languages/varieties asserted, exact byte offsets, content hashes |
| Conversational | Speech act or discourse function of each turn, interaction form, repair/follow-up behavior, register/style, expected response behavior |
| Linguistic | Primary and secondary phenomena, level(s) of analysis, constructions/contrasts, ambiguity or acceptability status, reasoning operation, difficulty |
| Ontological | Entity/relation/category distinctions exercised, identity/time/granularity assumptions, competency question or “not applicable” rationale |
| Pedagogical | Learning objective, prerequisite, intended transfer, counterexample/contrast relation, why the unit belongs |
| Epistemic | Attested/constructed/hypothetical/synthesis/behavior status, claims, evidence, uncertainty, competing analyses |
| Provenance | Source or generation chain, teacher/template/version, transformations, content and source hashes, license/authority status |
| Quality | Automatic gates, judges, human review where required, disagreements, revisions, final acceptance rationale |
| Data use | Split/group, release membership, cohort eligibility, restrictions, renderer/tokenizer compatibility |

Sentence- and span-level annotation should be required wherever the learning objective depends on a
specific form, reading, relation, or evidence phrase. Token-level annotation should cover tokenization,
supervision, delimiters, special tokens, fragmentation, and any phenomenon whose evidence is genuinely
token-local. It should not invent token-level labels for a discourse property that only exists at the
conversation level.

Completeness reports must show, by concept and annotation level:

- required, present, not applicable, unresolved, disputed, and missing counts;
- automated versus human/model-supplied annotation;
- confidence and agreement distributions;
- source and theory coverage;
- units blocked from release by incomplete description.

### 10.8 Required database invariants

1. **Original utterance text is immutable.** Corrections create revisions.
2. **Semantic roles and delimiters are separate.** No canonical message text begins with a model-specific
   role marker or ends with model-specific EOS.
3. **Every rendered byte is traceable.** An export resolves to unit revisions, messages, injected
   delimiters, renderer, tokenizer, supervision, acceptance, and release.
4. **Rejections are evidence.** Rejected and failed candidates remain queryable with reasons unless a
   legal/ethical deletion obligation requires restricted tombstoning.
5. **Annotations are assertions, not magic columns.** They identify author, method, version, time,
   confidence, and evidence.
6. **Disagreement is legal state.** Competing parses, theory labels, judgments, and source claims can
   coexist.
7. **Splits are group-aware and frozen.** Paraphrases, source fragments, template families, and adversarial
   variants cannot leak across partitions through row-level randomization.
8. **Releases are immutable.** A change creates a child release and a new hash.
9. **Exports are deterministic.** The same sealed database, release, recipe, rendering profile, and
   tokenizer produce the same ordered bytes and token sequence.
10. **Model-visible and audit data are separate.** Metadata is never accidentally serialized into a
    training conversation.
11. **Access restrictions fail closed.** Unknown license or authority state excludes a unit from a public
    or training release until adjudicated.
12. **No unversioned automated analysis.** Segmentation, embeddings, judges, tokenizers, and classifiers
    carry exact method versions.
13. **Counts reconcile.** Release members, export rows, render hashes, tokenizations, and model-run
    exposure must agree or the release fails validation.

### 10.9 SQLite engineering requirements for later design

Before implementation, research agents should specify and benchmark:

- foreign-key enforcement and integrity checks;
- strict typing where SQLite supports it;
- transaction boundaries for generation and review imports;
- write-ahead logging during active construction and a safe sealing/export procedure;
- content-addressed external storage for large raw artifacts when embedding them would bloat backups;
- full-text search over messages, sentences, source claims, and reviewer rationales;
- indexes based on named audit queries, including concept, language, split, release, source, review state,
  teacher, hash, delimiter, and token-type access;
- the storage cost of tens of millions of `token_occurrence` rows;
- deterministic database snapshots, checksums, backup verification, and corruption tests;
- schema migration tests against every sealed release;
- a read-only publication copy with sensitive/restricted records excluded by an audited export;
- reproducible analytical views for accepted current units, unresolved reviews, coverage gaps, source
  risk, duplicate clusters, delimiter overhead, and run exposure;
- concurrency limits: SQLite is suitable as a corpus artifact and controlled review store, but concurrent
  generation workers may require a single-writer import queue or per-worker staging databases;
- portability: no critical meaning should live only in a host-specific path, undocumented extension, or
  transient virtual table.

The final `.sqlite3` file should ship with a schema/data dictionary, migration history, integrity report,
release manifest, and example read-only queries. Implementation details remain outside the present
Markdown-only task.

### 10.10 Database review questions

Research returns should answer:

1. Is SQLite still appropriate when full token occurrences and raw generation attempts are counted?
2. Which records belong inside the database versus content-addressed external artifacts?
3. Can the schema represent alternative linguistic theories without either hardcoding one or becoming
   unqueryable?
4. Which annotations can be automated reliably, and which require expert judgment?
5. How should confidence and disagreement be calibrated across humans and models?
6. What exact group keys prevent source, prompt-template, semantic, and adversarial leakage?
7. How can a future run compose a cohort by sentence/token/phenomenon type while preserving a frozen
   reproducible member list?
8. What tables or fields are required to honor withdrawal, restricted-use, or community authority?
9. How should private evaluation metadata be linked without leaking evaluation content?
10. What is the smallest public database that remains scientifically useful without redistributing
    restricted source material?
11. Should full token occurrences live in the canonical database, an attached immutable SQLite shard per
    tokenizer, or both—and what scientific queries would be lost by each choice?
12. What automated invariant proves that a delimiter cannot leak into canonical utterance text or that
    metadata cannot leak into a model-visible export?

## 11. Synthetic generation method

### 11.1 Do not freeze a teacher name prematurely

“Use a SOTA model” should mean a measured teacher-selection process, not loyalty to a model name that
will be stale by generation time. Research track R4 should conduct a small blinded bake-off over the
best available teachers using the same seed tasks. Compare:

- linguistic correctness;
- ontological precision;
- naturalness and conversational responsiveness;
- diversity without incoherence;
- calibration and willingness to preserve multiple analyses;
- source faithfulness;
- cost, rate limits, reproducibility, and terms governing generated outputs.

The strongest teacher for generation may not be the strongest judge. If feasible, use different model
families for authoring and adjudication so one model's stylistic and conceptual biases do not validate
themselves.

### 11.2 Proposed candidate pipeline

1. **Freeze the evaluation design first.** Define capabilities and reserve sources, scenario families,
   lexical items, and transformations that generation cannot see.
2. **Build a sourced seed bank.** Gather phenomena, distinctions, attested examples, ontology cases,
   misconceptions, and competency questions with license records.
3. **Generate multiple candidates.** Ask the authoring teacher for genuinely different interactions,
   not paraphrases of one template.
4. **Run deterministic validation.** Reject missing roles, empty answers, broken turns, context overflow,
   prohibited machine-facing formats, and exact duplicates.
5. **Run semantic and lexical deduplication.** Cluster prompts, answers, reasoning patterns, and openings;
   cap template families rather than retaining every paraphrase.
6. **Grounding check.** Verify attested claims against their source and check that constructed or
   hypothetical material is labeled honestly in metadata.
7. **Independent adjudication.** A different judge evaluates correctness, relevance, naturalness,
   pedagogical value, ambiguity handling, and cultural/source risk.
8. **Adversarial revision.** A reviser challenges the analysis, offers counterexamples, and repairs a
   candidate only when the repair is traceable.
9. **Human audit.** Specialists inspect stratified samples and all high-risk categories; rejection and
   disagreement rates remain visible.
10. **Select to the coverage matrix.** Accept by demonstrated value and underrepresented cells, not
    simply by aggregate judge score.
11. **Seal manifests and splits.** Hash content, provenance, filters, and split assignments before any
    training.

The final corpus must preserve rejected examples and reasons in a research ledger. Otherwise later
agents cannot tell whether an apparent gap was missed, attempted and rejected, or intentionally excluded.

### 11.3 Candidate-volume hypothesis

A reasonable research starting point is to generate **600,000–1,000,000 candidates** and select
200,000, but this range is not authorized and should not be budgeted until a 2,000–5,000-candidate study
measures the actual acceptance curve. The study itself must not become training data unless it passes the
same final pipeline.

Synthetic-data precedents support the pattern of controlled generation plus selection, not blind
volume. [TinyStories](https://arxiv.org/abs/2305.07759) demonstrated that carefully constrained
synthetic material can teach coherent language behavior to very small models.
[Cosmopedia](https://huggingface.co/blog/cosmopedia) emphasized diverse seed sources, prompt curation,
and deduplication. [Magpie](https://arxiv.org/abs/2406.08464) generated millions of candidates and
selected a much smaller high-quality set. None of these results proves that this exact 200K curriculum
will work for Alpha; they justify testing the method.

## 12. Quality gates for individual units

An accepted unit must be:

- structurally complete and nonempty;
- natural as a conversation or explicitly framed exercise;
- correct under its stated assumptions;
- responsive to the exact prompt;
- useful for a defined capability rather than merely on-topic;
- concise enough for its purpose and free of synthetic padding;
- clear about attested fact, construction, hypothesis, and disagreement;
- free of unsupported citations or invented language/community claims;
- free of hidden code/JSON/tool-use instruction unless the curriculum is later expanded by decision;
- within the context limit without semantic truncation;
- licensed and traceable where it depends on a source;
- sufficiently distinct from retained examples in wording, reasoning path, and interaction structure;
- assigned to no split whose isolation it would violate.

### 12.1 Automatic corpus-level audits

At minimum, report:

- exact and semantic duplicate rates;
- n-gram and embedding clusters by source, teacher, template, opening, and answer shape;
- distribution across every primary and secondary capability;
- prompt and answer token-length distributions;
- conversation-turn distributions;
- source, license, language, provenance-status, teacher, judge, and reviewer distributions;
- repeated answer openings and discourse templates;
- empty, EOS-only, truncated, malformed, and machine-format-bearing counts;
- lexical diversity without using it as a proxy for quality;
- disagreement rates between judges and humans;
- rejection and revision rates by category;
- train/evaluation similarity and source-family leakage.

All aggregate reports must link back to the exact units that produced the count.

## 13. Evaluation must precede generation

The terminal run showed that held-out teacher-forced loss can look good while free generation is
unusable. The future program must select data and checkpoints by behavior as well as loss.

### 13.1 Frozen evaluation families

Research track R8 should design a minimum 2,000-prompt suite spanning:

| Evaluation family | Strawman count | What it tests |
|---|---:|---|
| Response initiation and ordinary conversation | 300 | Nonempty, relevant, appropriately sized replies |
| Conversation repair and multi-turn reference | 250 | Clarification, correction, memory within context |
| Morphology and syntax | 300 | Analysis, contrasts, diagnostics, counterexamples |
| Semantics and pragmatics | 300 | Ambiguity, inference, context, reference |
| Typology and cross-linguistic reasoning | 250 | Non-Anglocentric distinctions and calibrated limits |
| Ontology | 300 | Identity, categories, relations, time, parts, events, evidence |
| Integrated language–ontology dialogue | 150 | Cross-domain synthesis |
| Adversarial uncertainty and contradiction | 150 | Multiple analyses, provenance, refusal to fabricate |
| **Total** | **2,000** | |

The final suite should use held-out scenario generators, source families, lexical realizations, and
phenomena combinations—not row-level random samples from generation templates. A private final subset
should remain unseen by corpus authors and tuning agents.

### 13.2 Required measurements

Measure at least:

- nonempty-response rate;
- immediate-EOS rate and first-token EOS margin;
- structural completion rate;
- relevance and instruction response;
- semantic correctness;
- conversational naturalness;
- answer-length appropriateness;
- repetition and loop rates;
- consistency across paraphrases and prompt-length bands;
- minimal-pair sensitivity;
- ambiguity recognition without indiscriminate hedging;
- counterexample quality;
- provenance and uncertainty calibration;
- cross-linguistic hallucination rate;
- ontology distinction and competency-question accuracy;
- multi-turn coherence;
- regression against base language competence.

Use deterministic metrics where they measure the real property, multiple independent model judges where
rubrics require judgment, and blinded human assessment on stratified samples. Judge agreement and known
judge failures must be reported.

### 13.3 Preliminary admission gates

These thresholds are proposals for R8 to critique:

- at least 99% nonempty responses across the complete suite and every prompt-length band;
- no systematic first-token EOS collapse;
- at least 98% structurally complete responses;
- no degenerate loop in the frozen suite;
- no category with less than 80% judged useful, and a materially higher overall useful rate;
- no significant loss of ordinary conversational ability in exchange for technical vocabulary;
- human review confirms the same direction as automated judges;
- a claimed improvement must hold across multiple prompt families, not one attractive sample.

No Discord or public progress claim should be made for a lower loss alone. Share model outputs only when
an aggregate frozen behavior metric improves, and show both the example change and why it is
representative of the measured improvement.

## 14. Curriculum, mixing, and training implications

This is a data brief, but the corpus cannot be designed independently of the failure mechanism.
Research track R9 must propose a training-facing curriculum that includes:

- deterministic shuffle with a recorded seed;
- balanced interleaving of source and capability families;
- caps preventing one teacher, template, source, or answer shape from dominating a window;
- episode-aware sampling so one long answer does not outweigh many successful response starts;
- deliberate short-answer and first-turn coverage throughout training, not only at the beginning;
- multi-turn examples introduced without erasing short conversational competence;
- development evaluations at fixed token intervals;
- checkpoint selection by generation behavior plus loss;
- explicit tests for catastrophic forgetting and source-recency effects;
- ablations that isolate data effects from loss, sampling, or architecture changes.

One attractive ordering hypothesis is:

1. response initiation, ordinary dialogue, short explanations, and repair;
2. sentence contrasts and core morphology/syntax;
3. semantics, pragmatics, and multiple interpretations;
4. ontology cases and knowledge distinctions;
5. typology and cross-linguistic comparison;
6. integrated multi-turn research dialogue;
7. balanced rehearsal of all earlier capabilities throughout later stages.

This order is not a recommendation to train in sealed blocks. The previous run demonstrated the danger
of long homogeneous blocks. Each stage should increase a capability's share while retaining a shuffled
rehearsal mixture.

### 14.1 Required future ablation ladder

No flagship should be used to discover whether the basic idea works. R1, R8, and R9 should design a
compute-bounded comparison in which each rung changes as few variables as possible. The exact sizes are
open, but the comparison should include:

1. the archived base checkpoint evaluated without new training;
2. a size-matched, deterministically shuffled subset of the current generic SFT data;
3. conversational-foundation data only;
4. targeted linguistics data only;
5. targeted ontology data only;
6. the proposed integrated mixture;
7. the integrated mixture without contrast/multiple-analysis units;
8. the integrated mixture with one teacher versus multiple teachers;
9. SFT-only versus targeted continued pretraining plus SFT;
10. ordinary token-averaged loss versus the selected answer-initiation/episode-aware intervention.

Not every comparison must become a separate full model. Fractional factorial or successive-halving
designs may be more efficient. What matters is preserving at least one control that distinguishes “new
data is better” from “the trainer changed” and one that distinguishes “more targeted” from merely “more
tokens.” Seeds and checkpoint-selection rules must be declared before results are read.

### 14.2 Scaling stages, each requiring new authorization

The expected future sequence is:

| Stage | Purpose | Output | Training allowed under this brief? |
|---|---|---|---|
| Prior-art and design review | Test novelty and assumptions | R0–R11 returns and revised brief | No |
| Candidate micro-study | Measure teacher diversity, rejection, revision, and cost | 2K–5K audited candidates; not automatically trainable | No |
| Frozen evaluation construction | Establish uncontaminated behavioral gates | Versioned private/public evaluation manifests | No |
| Small data ablation | Compare mechanisms cheaply | Predeclared run matrix and reports | Only with new authorization |
| 20K accepted-unit pilot | Test the end-to-end corpus method | Audited pilot corpus and behavior curve | Only with new authorization |
| 50K–100K scale check | Test whether gains persist and interference appears | Scaling and transfer evidence | Only with new authorization |
| 200K accepted corpus | Complete the approved coverage matrix | Sealed corpus, provenance, and audit | Generation contract required |
| Flagship continuation | Test the final research claim | Frozen behavioral and efficiency results | Explicit training contract required |

Advancing a stage requires its predecessor's artifacts and gates. A promising cherry-picked sample is not
authority to skip the ablation or evaluation stages.

### 14.3 Known program-level risks

- **Capacity illusion:** a frontier judge may reward fluent imitation even when the student has learned
  no transferable abstraction.
- **Coverage illusion:** a taxonomy with every cell populated may still contain shallow paraphrases.
- **Synthetic monoculture:** one teacher's favorite analyses, discourse markers, and cultural defaults
  may dominate thousands of apparently diverse units.
- **Evaluation circularity:** teacher-generated train and teacher-generated eval can reward the same
  latent template.
- **Concept interference:** linguistic and ontological terminology may displace ordinary conversational
  fluency in a small model.
- **Foundation mismatch:** SFT may be asked to teach abstractions that should have been learned during
  pretraining.
- **Tokenizer inequality:** morphologically rich or non-Latin data may consume far more tokens and
  receive less effective context.
- **False multilingualism:** scattered examples across many languages may teach exotic-looking tokens,
  not cross-linguistic understanding.
- **Jargon capture:** the model may learn labels such as “presupposition” or “continuant” without being
  able to use the distinction in an unfamiliar case.
- **Catastrophic forgetting:** specialization may erase the modest general language ability already in
  the base checkpoint.
- **Teacher hallucination laundering:** independent synthetic prose can turn a model's false belief into
  an apparently sourced lesson.
- **License and authority mismatch:** legally downloadable data may still be unsuitable for derivative
  generation, redistribution, or community use.
- **Benchmark overfitting:** repeated checkpoint inspection can turn a frozen evaluation into training
  feedback even without placing its rows in the corpus.
- **Selection bias:** preserving only beautiful examples can hide the actual rejection rate and make the
  method impossible to reproduce.
- **Compute confounding:** conclusions drawn from one seed, one checkpoint, or unequal token budgets may
  be noise.

Every research return should identify which of these risks it addresses and add any missing risk.

## 15. Safety, cultural authority, and epistemic conduct

The project must distinguish content safety from research honesty.

- Do not fabricate examples attributed to real languages, speakers, communities, researchers, or
  traditions.
- Do not infer community consent from technical access or an open webpage.
- Do not use restricted cultural or personal material.
- Do not teach the model that prestige varieties are inherently more logical, grammatical, or precise.
- Do not convert a contested analysis into an unqualified fact merely to simplify a lesson.
- Include respectful correction of linguistic prejudice and category essentialism without turning every
  dialogue into a lecture.
- Preserve disagreements between theories where the evidence does not settle them.
- Teach the difference between a harmful category, a category used descriptively, and a report that
  somebody else used that category.
- Audit teacher outputs for stereotypes encoded in names, occupations, dialects, grammaticality
  judgments, social roles, and ontology examples.
- Require domain or community review for high-risk language material.

Research track R10 must define what requires specialist review, community authority, exclusion, or an
explicitly limited claim.

## 16. Research-agent work program

Agents should work independently enough to expose disagreement. They must not edit code or start data
generation/training under this brief. Each agent should return a dated memo in section 18 or submit a
patch that adds only its labeled return and clearly proposed amendments.

### R0 — Prior art and novelty adjudication

Search broadly and systematically for the closest prior work: tiny/small language models trained on
synthetic curricula; BabyLM-style data-efficiency work; domain-specialized conversational models;
linguistically informed pretraining or instruction tuning; natural-language inference and minimal-pair
curricula; ontology verbalization and ontology QA; philosophical dialogue models; Socratic/tutoring
data; multi-teacher distillation; contradiction-preserving or uncertainty-aware dialogue; and
provenance-bearing synthetic datasets. Produce a claim chart distinguishing known components, close
combinations, and any narrower novelty that survives. A null novelty finding does not cancel the project;
it improves its experimental grounding.

### R1 — Capability and scale fit

Determine what conversational, linguistic, and ontological capabilities are realistically learnable by
a 57.7M-parameter, 1B-pretrained model with a 1,024-token context. Evaluate whether the binding problem
is data design, optimization, tokenizer/context, capacity, or some interaction. Recommend an empirical
way to distinguish them before a flagship run.

### R2 — Linguistic coverage ontology

Turn section 8 into a principled, cross-linguistic hierarchy of capabilities, phenomena, interactions,
difficulty levels, and failure modes. Use primary linguistic resources. Identify English-centric
assumptions, missing subfields, text-only modality limits, and areas where examples require expert or
community validation.

### R3 — Formal and applied ontology coverage

Turn section 9 into a framework-neutral coverage plan. Compare relevant upper-ontology and knowledge-
representation traditions without selecting one as metaphysical truth. Define natural-language tasks
that test understanding without teaching serialization syntax.

### R4 — Synthetic generation methodology and teacher selection

Review current primary research on synthetic corpora, instruction generation, knowledge distillation,
teacher bias, self-instruct methods, filtering, revision, and small-model curricula. Propose the teacher
bake-off, candidate multiplier, generation prompts, independence requirements, and stopping rule.

### R5 — Sources, licenses, and provenance

Build a source inventory for linguistic phenomena, attested examples, typology, lexical semantics,
conversation, and ontology. Record item-level licenses and restrictions. Specify a provenance model that
can survive publication, re-generation, correction, and withdrawal.

### R6 — Chatty behavior and conversation design

Operationalize “chatty” without equating it with verbosity, flattery, or filler. Define speech acts,
turn structures, repair behaviors, follow-up behavior, registers, persona boundaries, and naturalness
rubrics. Propose how to prevent one teacher's voice from becoming the model's only voice.

### R7 — Filtering, judging, and deduplication

Design deterministic, model-based, and human quality stages. Address correlated author/judge errors,
semantic duplicates, template leakage, superficial diversity, factual/source verification, adversarial
revision, and retention of rejected evidence.

### R8 — Evaluation and acceptance gates

Design the frozen evaluation before data generation. Define held-out construction, contamination
controls, metrics, rubrics, human sample sizes, judge calibration, uncertainty bounds, baseline
comparisons, and exact admission criteria for any future training scale-up.

### R9 — Curriculum, mixing, and data budget

Propose the 200K allocation, token budget, difficulty schedule, interleaving, repetition/rehearsal,
sampling unit, and ablations. Explicitly address the prior monotonic loader and answer-start failure.
Compare 200K high-quality units with smaller and larger alternatives.

### R10 — Cultural, language-authority, and harm review

Define rules for Indigenous, low-resource, signed, minoritized, historical, stigmatized, and community-
governed language data. Identify where public data remains ethically unsuitable, where synthetic
examples are unacceptable, and who can authorize or review inclusion.

### R11 — SQLite dataset substrate and annotation architecture

Critique sections 10.4–10.10 as a scientific data model, not merely a software schema. Determine whether
one canonical SQLite database can preserve raw candidates, revisions, message/sentence/span structure,
competing linguistic and ontological annotations, provenance, reviews, token occurrences, releases, and
model-run exposure at the proposed scale. Specify the delimiter-independent rendering contract,
annotation-completeness profiles, table normalization, indexes, storage estimates, integrity invariants,
migration/publication strategy, and any justified external or attached artifacts. Include representative
research queries that the design must answer and failure cases that would make SQLite the wrong choice.

## 17. Required research-return format

Every agent return must include:

1. **Track ID and date.**
2. **Bottom-line recommendation.** One paragraph, including whether it changes this brief.
3. **Claims and evidence.** Each important claim linked to a primary source.
4. **Candidate sources or datasets.** Include exact version, maintainer, scope, license, access method,
   and known limitations.
5. **Proposed design.** Concrete counts, categories, rubrics, or experiments where appropriate.
6. **Risks and counterarguments.** Include evidence against the preferred proposal.
7. **Falsification criteria.** What result would make the agent abandon or revise its recommendation?
8. **Unresolved questions.** State what cannot yet be answered.
9. **Changes requested to this brief.** Quote the affected heading and provide replacement wording.
10. **Reproducibility record.** Search date, query strategy, sources inspected, and any inaccessible
    material whose absence could change the conclusion.

Use exact excerpts sparingly and within copyright limits. Prefer primary papers, official specifications,
dataset cards, licenses, and maintained repositories over summaries. Do not cite a search-results page.

## 18. Agent research returns

This section is append-only. A later synthesis may supersede a recommendation, but it must not erase the
original return or disagreement.

| Track | Status | Latest return | Decision impact |
|---|---|---|---|
| R0 prior art and novelty | open | — | — |
| R1 capability and scale | open | — | — |
| R2 linguistic coverage | open | — | — |
| R3 ontology coverage | open | — | — |
| R4 generation method | open | — | — |
| R5 sources and licenses | open | — | — |
| R6 conversation design | open | — | — |
| R7 filtering and judging | open | — | — |
| R8 evaluation | open | — | — |
| R9 curriculum and mixing | open | — | — |
| R10 authority and harm | open | — | — |
| R11 SQLite substrate | open | — | — |

### R0 return

Pending.

### R1 return

Pending.

### R2 return

Pending.

### R3 return

Pending.

### R4 return

Pending.

### R5 return

Pending.

### R6 return

Pending.

### R7 return

Pending.

### R8 return

Pending.

### R9 return

Pending.

### R10 return

Pending.

### R11 return

Pending.

## 19. Decisions still required

The research phase must resolve these before generation begins:

1. Is 57.7M parameters a viable target for the desired behavior, or should Alpha remain an engine
   research artifact while a larger student is used for the research-model goal?
2. Should some of the 200K units support continued pretraining rather than assistant-only SFT?
3. What is the smallest experiment that distinguishes answer-start weighting, data quality, data order,
   and model capacity?
4. What average token length makes the corpus learnable without returning to long-answer domination?
5. Which teacher and independent judge families win a blinded domain bake-off?
6. What proportion of examples must be source-grounded versus transparently hypothetical?
7. Which languages and linguistic phenomena can be included with adequate accuracy, licensing, and
   authority?
8. What ontology traditions should be contrasted, and which distinctions are genuinely useful to an
   ordinary conversational user?
9. Which corpus artifacts can be public, and which require restricted storage or exclusion?
10. What frozen gates justify spending money on another training run?

## 20. Definition of research completion

Research is complete only when the repository contains an evidence-backed amendment or successor to
this brief that includes:

- reconciled returns from R0–R11, including preserved disagreements;
- a prior-art claim chart and defensible statement of contribution or replication;
- a reviewed SQLite logical schema, delimiter-independent rendering contract, annotation-completeness
  profile, scale estimate, and integrity/query acceptance tests;
- a frozen capability and coverage matrix;
- a source and license inventory;
- a teacher/judge/reviser selection method;
- an approved provenance and rejection ledger design;
- an exact candidate study with cost ceiling and stop conditions;
- a frozen evaluation construction plan and acceptance gates;
- a proposed final allocation and token budget;
- a curriculum and mixing plan that addresses the prior failure;
- an explicit cultural/language authority policy;
- a decision on 57.7M model viability;
- separate proposed contracts for generation and training.

Having prompts, a teacher subscription, or a target row count does not satisfy this definition.

## 21. Initial evidence base

These sources motivate the current brief and should be supplemented, challenged, and version-pinned by
the research tracks:

- Hugging Face, [SmolTalk dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk): the
  dominant existing source is already a broad synthetic SFT mixture, including capabilities outside
  this project's new focus.
- Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759): controlled synthetic language data can
  produce coherent behavior in unusually small models, while task simplicity matters.
- Ben Allal et al., [Cosmopedia](https://huggingface.co/blog/cosmopedia): large synthetic corpus design
  depends on seed diversity, prompt curation, and deduplication.
- Xu et al., [Magpie](https://arxiv.org/abs/2406.08464): large candidate generation followed by much
  smaller high-quality selection is a useful precedent.
- [Universal Dependencies introduction](https://universaldependencies.org/introduction.html) and
  [guidelines](https://universaldependencies.org/guidelines.html): cross-linguistic grammatical
  annotation with shared and language-specific distinctions.
- [WALS feature inventory](https://wals.info/feature): a starting map of documented structural diversity,
  not a complete linguistic theory or automatic license for every derived example.
- W3C, [OntoLex-Lemon](https://www.w3.org/2016/04/ontolex/): an explicit bridge between lexical material
  and ontological representation.
- [Basic Formal Ontology 2020](https://bfo-ontology.github.io/bfo-2020.html): one maintained upper-
  ontology framework whose commitments can seed contrasts and competency questions.
- W3C, [OWL 2 Direct Semantics](https://www.w3.org/TR/owl2-direct-semantics/): a formal reference for
  class, property, individual, and inference distinctions that should be rendered as natural-language
  reasoning rather than syntax-generation training.
- [OBO Foundry principles](https://obofoundry.org/principles/fp-000-summary.html): governance,
  interoperability, definition, documentation, and collaboration principles for applied ontology.

## 22. Working conclusion

A 200K synthetic curriculum is plausible, but **synthetic scale is not the thesis**. The thesis is that a
small model may learn a coherent conversational and conceptual niche when every accepted example is
chosen for that niche, grounded where it makes factual claims, balanced across real capability gaps,
and evaluated through free generation.

The most important shift from the failed run is:

> Do not ask whether the model predicts the teacher's next tokens cheaply. Ask whether it begins a useful
> response, understands the distinction being tested, stays in conversation, and knows what its evidence
> does and does not support.
