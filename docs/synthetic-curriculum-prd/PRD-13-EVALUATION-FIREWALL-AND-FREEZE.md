# PRD-13 — Evaluation firewall, private vault, and freeze protocol

**Status:** implementation-ready planning contract; not authorized for execution before D5 closeout

**Applies to:** D6 AlphaPact, ordinary-chat, response-initiation, and conceptual evaluation construction

**Authority:** PRD-00, PRD-02, PRD-05, PRD-06, PRD-07, PRD-09, and a future explicit post-D5 operator decision

**Current physical state:** the canonical SQLite ledger has only a preliminary `evaluation_output` placeholder;
the complete evaluation catalog described here is not implemented or populated

**Public/private boundary:** every public ledger table remains browseable at `/corpus`; private evaluation
plaintext is never placed in the publicly served database or public blob tree before retirement

## 1. Purpose

Alpha's central empirical claim cannot be tested with prompts written after seeing the trained model, with
paraphrases of training families, or with a benchmark repeatedly inspected during curriculum construction.
This PRD defines the evidence firewall that must exist before an accepted synthetic training release can be
constructed.

The firewall must make five propositions machine-auditable:

1. evaluation families and their semantic relatives were fixed before training-release selection;
2. private prompts, expected states, and answer sets did not enter model-visible training bytes;
3. development results, private results, and qualitative debugging samples were used under different policies;
4. every generated output is tied to the exact checkpoint, decoder, prompt bytes, and environment that produced
   it; and
5. a later contamination discovery adds a visible correction or superseding suite instead of rewriting history.

This is not a benchmark-content authorization. It is the implementation contract that prevents future data
generation, release construction, and training from making the benchmark meaningless.

## 2. Product alignment

The evaluation program exists to answer whether Alpha is a natural conversational partner with unusually good
conceptual competence. It must not turn Alpha into a formal-state printer.

Model-visible evaluation input is ordinary natural language. Researcher-side contracts may describe:

- public commitments and denials;
- locally negotiated meanings;
- licensed and prohibited consequences;
- dependency-aware revisions;
- legitimate alternative analyses;
- evidence and attribution;
- scope, time, granularity, and purpose;
- required conversational policy; and
- expected invariants and changes.

Alpha receives credit through behavior in conversation, not by emitting these labels or using ontology jargon.
Ordinary conversational quality is measured separately from conceptual contribution so warmth, length, or a
philosophical voice cannot conceal a shallow move.

## 3. Gate and non-goals

### 3.1 Entry gate

D6 implementation and evaluation authoring begin only after:

- the complete D5 human census is sealed;
- hidden-repeat stability is recorded;
- all family syntheses and structural dispositions exist;
- the non-binding Pass D closeout exists;
- the operator adjudicates D5; and
- the operator authorizes a bounded D6 scope.

Infrastructure may be planned before that decision. No private prompt authoring, benchmark model call, human
study, or baseline run is implied by this document.

### 3.2 Non-goals

This PRD does not:

- authorize more synthetic training-data generation;
- authorize GPT-5.4, GPT-5.5, or another model call;
- authorize a GPU or Alpha training run;
- choose a fixed parameter count;
- declare the existing D5 candidates suitable for training;
- define one scalar score called intelligence;
- make a model judge the final authority on philosophical adequacy;
- publish private evaluation plaintext; or
- weaken the public all-table corpus browser.

## 4. Threat model

The system must defend against more than exact prompt duplication.

### 4.1 Direct leakage

- private prompt or answer text appears in a training candidate;
- a private payload is copied into a generation prompt, review packet, log, report, browser response, or public
  artifact;
- an evaluation export is accidentally selected by a training cohort;
- a tokenizer corpus or packing cache contains private plaintext.

### 4.2 Family leakage

- a private item is paraphrased with a new row ID;
- a sibling branch or repaired version enters training;
- a different projection of the same latent contract enters training;
- a source-derived evaluation item and its source-conditioned training analogue share the decisive structure;
- a private false bridge appears as a positive training example;
- a renamed family is assigned to another split.

### 4.3 Constructor leakage

- train and evaluation items share a prompt template, scenario constructor, teacher style, or counterexample
  recipe that makes the split predictable;
- the same author or model produces both sides with a recognizable signature;
- explanation order or rubric vocabulary becomes a shortcut.

### 4.4 Adaptive evaluation leakage

- private results influence checkpoint choice;
- a prompt is edited after inspecting a model failure;
- repeated private evaluation turns the suite into development data;
- qualitative examples are silently selected from private outputs;
- a judge prompt is tuned against the claimed final model.

### 4.5 Operational leakage

- private text enters shell history, process arguments, CI logs, stack traces, telemetry, Discord, browser
  storage, or a public error page;
- a backup or SQLite snapshot containing plaintext is placed under the public corpus root;
- a public all-table reader follows an external path to plaintext;
- a wrong decoder, tokenizer, or chat renderer changes what was actually evaluated.

## 5. Three-zone storage model

### 5.1 Zone P — public scientific ledger

The canonical public SQLite ledger remains the system of record for:

- suite and version identity;
- public metadata and construct definitions;
- public development items;
- family and dependency bindings;
- authority and review status;
- payload hashes, byte lengths, and encryption metadata;
- freeze manifests and signatures;
- metric and decoder definitions;
- contamination methods and results;
- model/checkpoint/run/output lineage;
- public aggregate observations; and
- corrections, retirements, and access-event summaries safe to disclose.

Every table in this zone remains visible through `/corpus`. A public row may prove that a private item exists
and was frozen without revealing its prompt, expected behavior, distractors, or answer set.

### 5.2 Zone V — sealed private evaluation vault

Private plaintext lives outside the publicly served corpus tree in a dedicated mounted-drive vault. The vault
contains content-addressed encrypted payloads and a private manifest. Encryption uses a mature external tool;
Alpha Corpus does not invent its own cryptography.

The decryption key or age identity:

- lives outside Git, SQLite, public blobs, logs, and command arguments;
- is referenced only by an opaque secret handle;
- is not available to the public web service;
- is not mounted into generation or training processes; and
- is required only by a local authorized evaluation runner.

Public SQLite stores the ciphertext digest, plaintext commitment digest, byte length, media type, vault object
identifier, encryption method/version, and verification state. It never stores the key or plaintext path.

### 5.3 Zone E — ephemeral evaluation workspace

An authorized run decrypts only the required suite version into a newly created, mode-restricted temporary
directory on the mounted drive. The runner:

1. verifies the frozen manifest before decryption;
2. verifies every plaintext digest after decryption;
3. passes prompt bytes through stdin or file descriptors rather than command arguments;
4. records output bytes immediately into the scientific artifact store;
5. removes the temporary plaintext on clean exit;
6. records an interrupted-cleanup event if deletion cannot be confirmed; and
7. never places the workspace beneath a public server root.

Removal of an ephemeral copy does not delete scientific evidence. The encrypted vault object and public
commitment remain immutable.

## 6. Logical schema

The table names below are first-class research objects, not generic annotations hidden in JSON.

### 6.1 Suite identity and freeze

| Table | Purpose |
|---|---|
| `evaluation_suite` | Stable benchmark construct and lifecycle |
| `evaluation_suite_version` | Frozen definition, policies, decoder set, primary outcomes, manifest digest |
| `evaluation_suite_parent` | Supersedes, repairs, derives-from, or retires relation |
| `evaluation_freeze` | Authority, timestamp, input population, manifest, and explicit freeze decision |
| `evaluation_freeze_basis` | Human reviews, adjudications, validator runs, and source decisions supporting freeze |
| `evaluation_policy` | Versioned development/private access and use policy |
| `evaluation_policy_binding` | Policy attached to suite version, item, run, or output |

An `evaluation_suite_version` is immutable. Correcting an item creates a new suite version and a typed parent
relation. A freeze cannot be edited into existence after a model run.

### 6.2 Families, items, trajectories, and probes

| Table | Purpose |
|---|---|
| `evaluation_family` | Independent statistical and semantic family root |
| `evaluation_family_version` | Versioned hidden family contract |
| `evaluation_family_relation` | Sibling, ancestor, projection, constructor, source, lexical, or semantic relation |
| `evaluation_item` | Stable item identity and public/private partition |
| `evaluation_item_version` | Versioned prompt trajectory metadata and payload commitment |
| `evaluation_turn` | Ordered natural-language turn commitment; plaintext may be vaulted |
| `evaluation_branch` | Shared-prefix counterfactual branch identity |
| `evaluation_probe` | Must-change, must-not-change, unresolved, scope, evidence, transfer, or false-bridge probe |
| `evaluation_expectation` | Required, permitted, forbidden, or set-valued outcome contract |
| `evaluation_dependency` | Which commitments or expected changes depend on which premise/pact/evidence |
| `evaluation_family_binding` | Binding to corpus family, source, constructor, template, author, teacher, or cluster |
| `evaluation_holdout` | Terminology, scenario, projection, source, and constructor holdout contract |

Private item rows expose safe metadata and commitments only. Model-visible turns and answer-bearing contracts
are ciphertext-backed until an authorized run.

### 6.3 Authority and adjudication

| Table | Purpose |
|---|---|
| `evaluation_review_assignment` | Blinded human/expert assignment |
| `evaluation_review` | Append-only review outcome and rationale |
| `evaluation_review_finding` | Specific ambiguity, invalidity, shortcut, cultural-authority, or scoring issue |
| `evaluation_adjudication` | Resolution or preserved set-valued disagreement |
| `evaluation_adjudication_basis` | Exact reviews/sources/executable checks supporting the decision |
| `evaluation_agreement_observation` | Family-clustered agreement statistic with method |
| `evaluation_disagreement_case` | Legitimate plurality or unresolved authority gap |

Model review can nominate problems but cannot masquerade as human authority. Human and expert roles remain
typed. Theory-relative cases may freeze with several admissible analyses rather than a manufactured consensus.

### 6.4 Decoder, metrics, and scoring

| Table | Purpose |
|---|---|
| `decoder_profile` | Exact temperature, sampling, stops, length, chat rendering, and seed policy |
| `metric_definition` | Versioned formula, unit, aggregation level, direction, and limitations |
| `metric_component` | Raw item/probe observation contributing to a metric |
| `metric_aggregation_plan` | Family-clustered aggregation and uncertainty method |
| `scoring_program` | Executable scorer artifact, environment, and digest |
| `scoring_program_test` | Positive, negative, boundary, and set-valued scorer fixtures |
| `judge_profile` | Optional model-judge contract and calibration limitations |

Primary outcomes, decoder profiles, stopping rules, and aggregation plans are frozen with the suite version.
Adding a post-hoc metric is legal only as a clearly labeled exploratory analysis.

### 6.5 Runs, outputs, and failures

| Table | Purpose |
|---|---|
| `evaluation_run` | Suite version, checkpoint, decoder, software, environment, purpose, and status |
| `evaluation_run_item` | Exact ordered item population and item-version digest |
| `evaluation_output` | Exact output blob, tokens/logprobs where available, timing, and status |
| `evaluation_failure` | Empty, immediate EOS, loop, timeout, OOM, invalid render, vault failure, or scorer failure |
| `behavior_annotation` | Human, executable, or calibrated-model annotation over an output |
| `pairwise_comparison` | Blinded order-controlled model-output preference |
| `metric_observation` | Raw or aggregated measurement tied to exact inputs |
| `statistical_analysis` | Predeclared or exploratory model specification and artifact |
| `run_adjudication` | Whether a run is valid for its declared comparison |

The preliminary existing `evaluation_output` table must be superseded or migrated into this relational
contract before real D6 data. A free-form `checkpoint_id` and `evaluation_item_id` without foreign keys is not
sufficient scientific lineage.

### 6.6 Human interaction

| Table | Purpose |
|---|---|
| `human_study` | Consent, population, compensation, protocol, and ethics status |
| `human_session` | Participant-pseudonym, conditions, ordering, timing, and consent version |
| `human_turn` | Separately governed human/model turns and vault/public policy |
| `human_session_event` | Pause, interruption, withdrawal, repair, and technical failure |
| `continue_preference` | Blinded desire-to-continue response and scale |

Human interaction data never becomes training data by default. Withdrawal and retention policy must be
specified before collection. Public aggregate results must not permit re-identification.

### 6.7 Contamination and split closure

| Table | Purpose |
|---|---|
| `split_dependency` | Declared dependency edge that forces objects into the same split closure |
| `split_closure_snapshot` | Frozen transitive closure for a suite or release decision |
| `contamination_probe` | Versioned exact, lexical, semantic, structural, source, and constructor method |
| `contamination_run` | Inputs, method versions, thresholds, software, and environment |
| `contamination_result` | Pair/group finding, score, severity, and disposition |
| `contamination_adjudication` | Human/expert decision on a proposed overlap |
| `lexical_holdout` | Held-out terminology or phrase family, versioned rather than hard-coded in software |
| `shortcut_probe` | Test for vocabulary, style, length, author, or template shortcuts |

Split closure is data-driven. It is not a hand-maintained map of predicate names or topic strings. Explicit
relations, source lineage, learned similarity, dynamically discovered signatures, and human adjudication all
contribute evidence.

### 6.8 Private payload commitments and access

| Table | Purpose |
|---|---|
| `private_payload` | Public metadata and commitments for an encrypted private object |
| `private_payload_location` | Opaque vault object reference and verification state |
| `private_payload_binding` | Item/version/turn/expectation relation to a payload |
| `private_payload_access_grant` | Purpose-bounded authorization, approver, expiry, and suite/run scope |
| `private_payload_access_event` | Append-only decrypt/read/use/cleanup result without plaintext |
| `private_payload_retirement` | Decision to publish, retain private, or destroy key material under policy |

The public all-table explorer may display every one of these tables. It must never resolve a vault reference,
serve a decryption key, or return plaintext.

## 7. Lifecycle and state machines

### 7.1 Suite lifecycle

```text
draft -> under_human_review -> calibrated -> leakage_audited -> frozen -> active -> retired
```

`frozen` requires a manifest and freeze decision. `active` requires a verified evaluation runner and a declared
access policy. Retirement does not delete historical runs.

### 7.2 Item lifecycle

```text
draft -> reviewed -> adjudicated | plural_admissible | rejected -> frozen
```

A rejected item remains in the ledger with its failure reason. A repair creates a new version or child item.

### 7.3 Run lifecycle

```text
planned -> materializing -> running -> completed | failed | interrupted -> adjudicated
```

A partially completed run remains visible. Empty responses and timeouts are outputs/failures, not silently
missing rows. A retry is another attempt tied to the same planned item.

### 7.4 Freeze immutability

No `UPDATE` or `DELETE` is allowed on versioned, frozen, output, annotation, contamination, or access-event
records. Mutable workflow heads may change state only through checked transitions. Scientific content changes
create new versions and typed supersession edges.

## 8. Evaluation composition

The first freeze must contain distinct suites or declared sub-suites for:

### 8.1 Response initiation

- nonempty response rate;
- immediate-EOS rate;
- first content token behavior;
- one-sentence response;
- medium response;
- answer-and-stop;
- role-token leakage;
- repetition loops; and
- appropriate stopping.

This gate is interpreted before conceptual scores.

### 8.2 Ordinary conversation

- contingency on the immediately preceding move;
- directness and appropriate length;
- multi-turn reference;
- correction and repair;
- disagreement without hostility or boilerplate;
- clarification only when necessary;
- adaptation to user register;
- conversational momentum;
- thread recovery; and
- human desire to continue.

### 8.3 AlphaPact

- pact adoption;
- delayed use;
- licensed and prohibited inference;
- revision locality;
- drift;
- scope shift;
- alternative preservation;
- cross-domain transport;
- false-bridge rejection; and
- efficiency conditional on correctness.

### 8.4 Language, pragmatics, and interpretation

- ambiguity type and resolvability;
- implicature and presupposition;
- reference and deixis;
- reported speech and evidential stance;
- intent/effect distinctions;
- discourse structure;
- metalinguistic negotiation;
- translation mismatch; and
- terminology-scrubbed structural transfer.

### 8.5 Ontology and philosophy in conversation

- role versus bearer;
- part versus member;
- group versus members;
- event versus object;
- identity through change;
- valid time versus record time;
- source versus claim;
- necessary versus sufficient conditions;
- granularity and purpose-relative representation;
- valid counterexamples and local conceptual repair; and
- legitimate theory-relative plurality.

These are conversational behaviors, not trivia questions or requirements to name a philosophical school.

## 9. Public development versus sealed private evaluation

### 9.1 Public development

Public items explain the construct, exercise the harness, and permit third-party reproduction. They may be used
for debugging and decoder validation. Results on them cannot establish final generalization.

### 9.2 Private evaluation

Private items are whole-family held out. Their semantic relatives, templates, projections, sources, and
constructor lineages are quarantined from training. Private results are not used to choose a checkpoint unless
the study explicitly declares a one-time terminal selection policy; the preferred policy is no private-based
selection at all.

### 9.3 Rotating and retirement

If a private suite becomes exposed, it is marked compromised and retired for future primary claims. Its
history remains public. A new suite version uses new independent families rather than cosmetic paraphrases.
Retired content may later be published for reproducibility after the replacement is sealed.

## 10. Leakage closure and release firewall

A training release cannot seal merely because no exact evaluation item ID is present. Its validation must
prove closure-level separation.

### 10.1 Required relation families

The closure includes:

- family ancestry and splits;
- projections and false bridges;
- branches, transformations, repairs, and paraphrases;
- source and source-fragment lineage;
- constructor and prompt-template lineage;
- author, teacher, critic, and reviewer generation roles where relevant;
- lexical clusters and held-out terminology;
- deterministic template signatures;
- learned semantic-similarity neighborhoods; and
- manually adjudicated relations.

### 10.2 Release-seal predicate

For every proposed training member, validation must show:

1. no direct binding to a private family or item;
2. no path through the frozen split-dependency closure;
3. no unresolved severe contamination result;
4. no private plaintext in rendered bytes, token IDs, caches, manifests, or tokenizer inputs;
5. no vault access by the generation/training principal; and
6. a contamination run whose exact methods and input snapshots are bound into the release manifest.

The result is a sealed validation artifact, not an ephemeral green log line.

### 10.3 Post-seal discovery

If contamination is discovered later:

- append the finding and evidence;
- mark the affected release with a warning state;
- identify every checkpoint with exposure to the release;
- produce a corrected release excluding the closure;
- keep the original release and result for scientific honesty; and
- do not retroactively describe the contaminated result as clean.

## 11. Evaluation-run protocol

An authorized runner performs the following fail-closed sequence:

1. resolve an immutable suite version and freeze;
2. verify suite, public ledger, private vault, checkpoint, tokenizer, renderer, decoder, and software digests;
3. verify that the purpose permits public or private access;
4. record a planned run and exact item order before inference;
5. materialize only required private payloads into the ephemeral workspace;
6. run free generation with the frozen decoder;
7. record every output, including empty output, EOS, loop, timeout, and crash;
8. run deterministic scorers before optional human/model annotation;
9. record raw item/probe observations before aggregates;
10. aggregate by independent family, not by correlated turn count;
11. clean and verify the ephemeral workspace;
12. adjudicate whether the run is valid; and
13. publish only policy-allowed metadata and aggregates.

A run never falls back silently to another model, decoder, prompt rendering, device, or checkpoint.

## 12. Metrics and interpretation

No single aggregate replaces the profile. Required families of measurements include:

- nonempty, immediate-EOS, loop, and stopping rates;
- ordinary conversational contingency and length control;
- pact adoption and drift;
- inferential consequence accuracy;
- revision locality plus missed dependent revision;
- alternative-set precision and recall;
- overhedging and false ambiguity;
- attribution and temporal integrity;
- cross-projection transport;
- false-bridge rejection;
- question necessity;
- interaction quality;
- desire to continue; and
- efficiency conditional on correctness.

Confidence intervals and tests use the concept/evaluation family as the primary statistical unit. Item- or
turn-level counts may be descriptive but cannot inflate precision.

Human and deterministic channels remain separate:

- executable fictional cases support exact state and delta scoring;
- human/expert-adjudicated cases support conceptual and conversational judgments;
- set-valued cases preserve legitimate alternatives;
- model judges support bounded naturalness triage only after calibration; and
- philosophical validity is never established solely by a model judge.

## 13. Public interface

The `/corpus` reader should eventually provide safe views such as:

- `v_evaluation_suite_public`;
- `v_evaluation_family_public`;
- `v_evaluation_freeze_manifest`;
- `v_evaluation_run_public`;
- `v_evaluation_family_result`;
- `v_contamination_summary`;
- `v_private_payload_commitment`; and
- `v_checkpoint_evaluation_exposure`.

The generic all-table browser remains available. Public rendering applies cell-level policy to any field that
could contain private or consent-governed text. A policy violation fails closed rather than returning a partly
redacted row whose shape leaks an answer.

Public API behavior remains read-only. No public route can create a suite, decrypt a payload, start a run,
submit an annotation, or change a freeze.

## 14. Planned operator interface

The eventual CLI should distinguish preparation from execution:

```text
alpha-corpus eval status
alpha-corpus eval suite-prepare
alpha-corpus eval import-public --file ...
alpha-corpus eval import-private --file ... --vault-secret-handle ...
alpha-corpus eval review-prepare --reviewer ...
alpha-corpus eval review-submit --file ...
alpha-corpus eval contamination-plan --suite ... --release ...
alpha-corpus eval contamination-run --execute ...
alpha-corpus eval freeze-plan --suite ...
alpha-corpus eval freeze --execute --decision ...
alpha-corpus eval run-plan --suite-version ... --checkpoint ...
alpha-corpus eval run --execute --authorization ...
alpha-corpus eval verify --run ...
```

Planning commands are read-only or create clearly typed draft workflow records. Model inference, private-vault
access, and freeze actions require explicit execution flags and authority records. No command interprets free
model prose as structured data.

## 15. Migration and validation plan

The future D6 implementation should be delivered in bounded migrations:

1. suite/family/item/version and public metadata;
2. payload commitments and vault access lineage;
3. review/adjudication and set-valued expectations;
4. decoder/metric/scorer contracts;
5. contamination and split closure;
6. runs, outputs, failures, annotations, and statistics;
7. human-study governance; and
8. public-safe views and release-firewall triggers/checks.

Before the first migration, create and verify a SQLite backup of the canonical ledger. Every migration is
hash-registered, additive, strict where practical, foreign-key checked, and covered by append-only triggers.
Fresh installation and upgrade of a copied canonical ledger must both pass.

## 16. Adversarial acceptance tests

The implementation is not accepted until tests prove at least:

- fresh migrations and idempotent reopen;
- immutable suite/item/freeze/output records reject update and delete;
- a changed private plaintext object fails its commitment hash;
- a changed ciphertext object fails vault verification;
- the wrong secret or suite version cannot decrypt or run;
- private sentinel phrases are absent from the public SQLite file, public blobs, exports, logs, and HTTP;
- the public all-table browser reveals metadata but no private plaintext;
- public-development items remain reproducible without vault access;
- a direct private item in a training cohort blocks release sealing;
- a renamed descendant or sibling blocks through split closure;
- a paraphrase or projection produces a contamination finding;
- a same-words/different-structure false positive can be adjudicated without deleting the finding;
- unresolved severe contamination blocks sealing;
- a decoder mismatch blocks an evaluation run;
- empty, EOS, loop, timeout, and crash results remain visible;
- retry outputs cannot overwrite prior outputs;
- private results cannot be silently used for checkpoint selection;
- family-level aggregation differs from naive correlated-item aggregation on a fixture;
- a set-valued expectation penalizes undercoverage and overcoverage;
- a compromised suite is superseded, not rewritten; and
- no evaluation operation creates training exposure or authorizes GPU work.

## 17. D6 acceptance gate

D6 is ready to unlock training-release construction only when:

- independent human reviewers support the adjudicable subset;
- disagreement is preserved where authority does not support one answer;
- response-initiation and ordinary-chat suites are operational;
- AlphaPact probes distinguish definition echo from actual pact use;
- false bridges, terminology scrubs, and constructor holdouts work;
- public and private suite versions are frozen with exact decoders and primary metrics;
- private plaintext is absent from public and training-visible artifacts;
- family/source/template/teacher/lexical/semantic closure audits pass;
- baseline failures show a useful difficulty range;
- the runner records every output and failure append-only;
- public all-table browsing remains safe and useful;
- the exact freeze predates training-release construction; and
- the operator explicitly accepts the D6 evidence and authorizes the next bounded stage.

## 18. Current gap and next action

The canonical D5 ledger already has content-addressed blobs, actors, provenance, analysis runs, family topology,
public browsing, append-only triggers, and a preliminary `evaluation_output` table. It does not yet have the
suite, item, expectation, metric, run, private-vault, human-study, contamination, or split-closure substrate
required by this PRD.

That gap is expected at the current gate. The next authority-bearing action remains real human D5 Pass A, not
D6 migration or benchmark authoring. After D5 closeout, this PRD gives the operator a bounded implementation
surface that can be accepted, amended, or rejected before any private evaluation content exists.
