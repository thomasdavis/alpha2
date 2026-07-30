# PRD-07 — Synthetic-only experimental program

## 1. Purpose

This PRD defines the first model experiments after the corpus, ledger, and evaluation gates pass. The primary
experiment is intentionally radical and clean:

> Initialize Alpha without importing a pretrained language model and train it only on data generated and
> released by Alpha Corpus.

No previous Alpha pretraining corpus, old SFT conversations, web crawl, public instruction dataset, code
corpus, encyclopedic QA set, or hidden human dialogue enters the primary training mixture. Human-authored
material may calibrate reviewers and populate private evaluation, but not train the primary condition.

Later experiments may compare human, retrieved, public, or pretrained additions. Those are separate arms after
the synthetic-only result is understood.

## 2. Constraint

The model, context length, batch plan, optimizer state, and checkpointing strategy must fit and run safely on
the one GPU available to the operator. No fixed parameter count is a product requirement. Scale is an
engineering variable chosen from measured memory, throughput, stability, and conversational gain.

The binding Alpha rule remains: training FLOPs use Alpha's own training stack unless the operator explicitly
supersedes that decision in `docs/resume/DECISIONS.md`.

## 3. Why synthetic-only first

The first study asks whether the program can deliberately build the linguistic and conversational environment
it wants rather than inherit an opaque mixture and apply a thin specialist SFT. A clean result can show:

- which competencies the generated corpus can install;
- whether ordinary chat and conceptual specialization can coexist;
- whether Donto-derived categorical breadth matters;
- whether linked trajectories add value over independent synthetic scenes;
- whether entity-light construction produces evidence-ready behavior;
- where the corpus lacks basic language coverage;
- what the single-GPU capacity threshold appears to be.

Synthetic-only does not imply all facts are fictional nonsense. It means every training passage and dialogue
is generated under a known recipe and recorded in the ledger. Some future synthetic units may be grounded in
licensed source facts, but the primary release should prefer fictional or generic entities to keep factual
memorization secondary and provenance clean.

## 4. Experimental sequence

### E0 — Evaluation and harness validity

Before training:

- freeze ordinary-chat and AlphaPact suites;
- establish human and strong-model behavior on adjudicated subsets;
- verify exact free generation, decoding, role formatting, EOS, and metric aggregation;
- prove that train/eval family contamination detection works;
- create a tiny synthetic canary corpus that exercises the full export/training/evaluation lineage without
  making a capability claim.

E0 uses no paid GPU beyond a separately authorized bounded smoke test.

### E1 — Synthetic language substrate

Construct a broad synthetic foundation capable of teaching ordinary English rather than only meta-discussion.
It should include:

- short and medium natural dialogue;
- narrative, description, explanation, comparison, and reflection;
- diverse sentence structures and discourse relations;
- ordinary physical, social, causal, temporal, and institutional situations;
- language phenomena realized naturally;
- corrections, repair, reference, ellipsis, and implicature;
- many answer-and-stop examples;
- source-conditioned passages and discussion;
- entity-light fictional names, places, institutions, and events;
- no programming or JSON curriculum in the primary mixture.

This layer is synthetic but not conceptually shallow. It supplies the language substrate on which specialized
behavior depends.

### E2 — Response-initiation and ordinary-chat pilot

Train a bounded pilot selected to answer only:

- Does Alpha reliably begin a response?
- Can it produce one-sentence and medium answers?
- Does it stop appropriately?
- Can it sustain several turns without role leakage or loops?
- Does it respond to the actual preceding move?

Do not interpret ontology or philosophy scores if this gate fails.

### E3 — Synthetic conversational foundation run

Train on the synthetic language substrate plus ordinary conversational curriculum. Select checkpoints by free
generation, not teacher-forced loss alone. Measure response-start token behavior separately from answer
interiors.

Required frozen behaviors include greeting, direct question, disagreement, clarification, answer-and-stop,
short explanation, multi-turn reference, and recovery after a correction.

### E4 — Concept curriculum intervention

From the same valid starting checkpoint and equal targeted-token budgets, compare:

1. **Independent:** synthetic conceptual scenes shuffled and attention-separated.
2. **Trajectory-linked:** the same underlying distinctions realized as coherent stateful multi-turn dialogue.
3. **Relation-corrupted:** surface-matched trajectories containing wrong pact applications or revisions.
4. **Branch-contrastive:** shared prefixes with correct and polished hard-negative continuations, if the
   objective is implemented and separately declared.

The principal comparison is trajectory-linked versus independent and relation-corrupted on whole-family held-
out revision locality. The relation must be visible in model context or objective; a relationship stored only
in SQLite is not training signal.

### E5 — Cross-projection and plurality study

Test whether learned distinctions transfer between linguistic, ontological, social, material, and evidential
realizations, while preserving finite admissible alternatives. Include lexical isolation and false bridges.

### E6 — Entity-light evidence study

Only after E4 produces interpretable behavior, compare synthetic conditions such as:

- ordinary fictional names;
- type-preserving placeholders;
- variable entity specificity;
- visible supporting passage;
- incomplete passage;
- conflicting sources;
- familiar-looking claim contradicted by supplied evidence;
- retrieval-required prompt with missing evidence.

This studies evidence use without contaminating the first relational-curriculum result.

### E7 — Later external-data ablations

Possible later studies—each separately authorized—may add:

- an existing pretrained base;
- licensed human conversations;
- public pretraining text;
- real retrieval;
- Donto claims and source passages;
- preference optimization;
- multilingual or community-reviewed material.

These are not part of the primary synthetic-only claim.

## 5. Synthetic training mixture architecture

The primary release should be assembled from explicit components rather than one undifferentiated pool:

| Component | Purpose | Key risks |
|---|---|---|
| Ordinary conversational foundation | Initiation, contingency, turn-taking, stopping | generic assistant voice |
| Linguistic breadth | grammar, reference, discourse, pragmatics | textbook metalanguage |
| World-schema prose | ordinary entities, events, roles, causes, materials | incoherent synthetic worlds |
| Donto-lens conceptual scenes | ontology/philosophy specialization | lecture density |
| Inferential pact trajectories | local meanings and revision | canned negotiation pattern |
| Source-conditioned dialogue | evidence and attribution | clean-passage QA shortcut |
| Short-form cohort | brevity and answer-and-stop | shallow answers |
| Hard negatives/contrasts | boundary learning | obvious bad style |
| Narrative and playful material | language vitality and presence | losing conceptual allocation |

Mixture weights are experimental parameters recorded in the release and exposure ledger. They are not tuned on
private evaluation.

## 6. Token and episode accounting

Report:

- unique natural-language tokens before rendering;
- model-visible tokens after rendering;
- supervised assistant tokens;
- response-start positions;
- EOS positions;
- short/medium/long response distribution;
- independent family count;
- trajectories, branches, and transformation edges;
- repetition/exposure by unit and family;
- semantic duplicate-adjusted content estimate;
- teacher and prompt concentration.

Episode count alone is insufficient. A thousand paraphrases from one family are not a thousand independent
concepts.

## 7. Tokenizer and language considerations

The tokenizer is part of the synthetic-only system and is versioned. Before a primary run, verify:

- coverage of ordinary English morphology and punctuation;
- conversational contractions and fragments;
- philosophical/linguistic vocabulary without excessive fragmentation;
- names and fictional entities;
- Unicode and quoted material;
- special-token separation from natural text;
- stable chat rendering;
- no accidental evaluation vocabulary leakage from tokenizer training artifacts beyond ordinary lexical
  availability.

Tokenizer training data must follow the declared synthetic-only boundary if tokenizer exposure is included in
the scientific claim.

## 8. P0 response-initiation gate

The archived Alpha result demonstrated that healthy-looking teacher-forced loss can coexist with near-total
free-generation failure. The new program therefore tracks:

- first assistant-token loss and rank;
- first content-token loss and rank;
- immediate-EOS probability;
- nonempty response rate;
- completion length distribution;
- EOS termination rate;
- repetition/loop rate;
- role-token leakage;
- prompt-length sensitivity;
- multi-turn degradation.

Before specialized interpretation, the frozen ordinary suite should reach a predeclared high reliability bar
approximately equivalent to: nearly all prompts receive a nonempty role-correct response, immediate EOS and
loops are rare, short answers work, and outputs stop. The exact numeric gate is frozen before the run and not
adjusted around a checkpoint.

## 9. Primary causal study

### 9.1 Hypothesis

At equal synthetic targeted-token budgets, coherent stateful conceptual trajectories improve held-out local
revision and pact use relative to independent scenes and conceptually corrupted trajectories, without damaging
ordinary conversation.

### 9.2 Experimental controls

Match as closely as possible:

- base checkpoint;
- optimizer and steps;
- assistant target tokens;
- response-start count;
- sequence/context-length distribution;
- response-length distribution;
- family difficulty;
- teacher and prompt distribution;
- source condition;
- positive/negative balance;
- total compute.

### 9.3 Primary endpoint

Family-level revision locality on whole-family held-out AlphaPact cases.

### 9.4 Secondary endpoints

- pact adoption and drift;
- required/prohibited inference accuracy;
- admissible-set precision/recall;
- cross-projection transfer;
- false-bridge rejection;
- response initiation;
- ordinary conversational contingency;
- answer length and question necessity;
- human desire to continue.

### 9.5 Negative-control interpretation

If correct linkage does not beat relation corruption, any gain is likely due to longer context, formatting,
repetition, or regularization rather than conceptual structure.

## 10. Checkpoint selection

Predeclare checkpoint cadence and selection rule. Selection may combine:

- free-generation structural gate;
- ordinary-chat development score;
- conceptual development score;
- non-degradation constraints;
- training stability.

Never select on private evaluation, a favorite qualitative sample, or loss alone. Preserve all evaluated
checkpoints and outputs.

## 11. One-GPU execution policy

- choose a model/configuration from measured memory headroom and throughput;
- run arms sequentially with identical environment and code revision;
- checkpoint optimizer and RNG so interruption is recoverable;
- mirror checkpoints and evidence off the pod continuously;
- fail closed on unintended CPU fallback;
- verify real GPU execution and progress, not process existence;
- terminate paid infrastructure after the bounded run;
- do not run more arms than the authorized experiment requires;
- use early futility rules predeclared from development metrics, never private test.

## 12. Statistical analysis

- concept family is the default independent unit;
- report per-family paired differences where arms share families;
- cluster or model random effects for family, transformation, projection, teacher/template, and seed;
- distinguish confirmatory primary metrics from exploratory analyses;
- report nulls and adverse ordinary-chat changes;
- treat multiple branches from one family as dependent;
- include effect intervals, not just point estimates;
- avoid claims about “understanding” beyond the measured behavior.

## 13. Decision outcomes

### Continue and scale

Only if synthetic training produces reliable conversation and linked trajectories show a credible held-out gain
over both independent and corrupted controls.

### Repair corpus

If ordinary language or chat fails while the training stack is stable, analyze coverage, response-start
distribution, style concentration, and tokenizer/rendering before generating indiscriminately more rows.

### Capacity/configuration threshold

If the curriculum is behaviorally coherent in a stronger feasible configuration but not the smallest tested
configuration, report a one-GPU capacity result rather than call the curriculum invalid.

### Reject linkage hypothesis

If correct linkage does not beat controls across sound runs, keep the corpus and ledger but stop claiming
relational organization teaches the target.

### Reject synthetic-only sufficiency

If broad synthetic-only training remains linguistically inert or pathological despite sound data and training,
test a later external-base or licensed-human ablation. Do not rewrite the primary failure.

## 14. Release gate for a trained Alpha

A model release must disclose:

- exact synthetic dataset release and rendering;
- training stack/code/checkpoint lineage;
- one-GPU compute and cost;
- free-generation gate results;
- AlphaPact and ordinary-chat results;
- human-evaluation method;
- entity/factual limitations;
- known style and conceptual failures;
- whether any external pretrained weights or data entered the run;
- whether the model is a research artifact or recommended conversational model.

## 15. Acceptance criteria

The experimental program is ready to execute only when:

- the synthetic-only boundary is machine-auditable;
- frozen evaluation predates training release construction;
- the response-initiation gate and decoder are validated;
- all arms have model-visible differences precisely specified;
- relation corruption remains fluent and surface matched;
- family-level leakage checks pass;
- exact model exposure can be reconstructed;
- compute fits one GPU with verified headroom;
- stop/futility rules and budget are authorized;
- no prior Alpha run is resumed by accident;
- the operator separately approves the bounded experiment.
