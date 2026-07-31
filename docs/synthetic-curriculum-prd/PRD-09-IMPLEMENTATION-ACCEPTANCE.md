# PRD-09 — Implementation plan and acceptance gates

## 1. Purpose

This document turns the research vision into bounded implementation stages. It is not authority by itself;
direct operator decisions determine which bounded stages may execute.

Each stage ends in inspectable evidence. Later stages do not begin because code exists; they begin after the
preceding gate is explicitly adjudicated and the operator authorizes the next bounded scope.

## 2. Current state

The project is at **D5 — calibration generated, human conceptual adjudication pending**. The exact bounded
generation record is [Execution 01](EXECUTION-01-LEDGER-AND-CALIBRATION.md). The executable closeout protocol,
review form, and unratified next-step package are [PRD-12](PRD-12-D5-HUMAN-ADJUDICATION.md),
[Appendix D](APPENDIX-D-D5-REVIEW-INSTRUMENT.md), and
[Decision Packet 01](DECISION-PACKET-01-D5-NEXT-STEP.md). The first-class deterministic evidence checkpoint is
[Execution 05](EXECUTION-05-D5-SURFACE-EVIDENCE.md). The fail-closed family-synthesis and structural-
disposition implementation is [Execution 06](EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md).
The blinded repeat-presentation implementation is
[Execution 07](EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md).
The non-binding campaign-closeout implementation is
[Execution 08](EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md).
The deployed aggregate campaign-state panel is
[Execution 09](EXECUTION-09-D5-PIPELINE-VISIBILITY.md).
The campaign-wide executable Pass B blindness gate is
[Execution 10](EXECUTION-10-D5-PASS-B-BLINDNESS-GATE.md).
The exact exported-packet envelope gate for browser drafts and local human submissions is
[Execution 11](EXECUTION-11-D5-IMMUTABLE-REVIEW-ENVELOPE.md).
The generalization of that gate to every A/B/C/D human packet is
[Execution 12](EXECUTION-12-D5-ALL-PACKET-ENVELOPE-BINDING.md).
The implementation contract for the still-locked D6 evaluation substrate is
[PRD-13](PRD-13-EVALUATION-FIREWALL-AND-FREEZE.md).
The D5-derived response-policy normalization and distribution contract is
[PRD-14](PRD-14-RESPONSE-POLICY-CONTROL-PLANE.md); its migration and backfill remain locked until blind review
and closeout.
The deployed accessibility, fatigue-reduction, recovery, and responsive-navigation proof for the local-first
human instrument is
[Execution 13](EXECUTION-13-D5-REVIEW-WORKSPACE-HARDENING.md).
The first-class reviewer competence/session-condition record, legacy-packet compatibility proof, and current
public deployment are [Execution 14](EXECUTION-14-D5-REVIEW-SESSION-PROVENANCE.md).
The complete Pass A evidence contract, explicit non-numeric dimension states, rubric-v1-to-v2 supersession,
and current public deployment are
[Execution 15](EXECUTION-15-D5-REVIEW-EVIDENCE-COMPLETENESS.md).

Done:

- product north star fixed;
- synthetic-data work recognized as a principal half of the program;
- Donto prompt lenses cross-walked into an extensible curriculum ontology;
- comprehensive ledger, orchestration, quality, benchmark, release, experiment, and operations contracts
  written;
- first experimental data boundary fixed as synthetic-only;
- fixed parameter-count framing removed;
- one-GPU constraint recorded;
- D2/D5 ledger: ten hash-verified migrations, 135 tables, five current/public views, 198 append-only triggers,
  content-addressed blobs, and clean integrity/foreign-key validation;
- D3 canaries: 49 categories, 16 transformations, and six quarantined family blueprints;
- D4 orchestration: structured Codex calls, exact raw artifacts, bounded/idempotent tasks, validators, usage,
  and completed-response recovery;
- D5 generation portion: 12 GPT-5.4 calls, 48 candidates, 42 structurally valid and six retained rejections,
  with a full human-audit packet;
- D5 review substrate: executable blinded Pass A, contract-aware Pass B, family-level Pass C, separate
  structural-disposition evidence, and non-binding Pass D campaign closeout; versioned rubrics; append-only
  human submission; a deployed local-first browser workspace; a hidden-lineage-safe aggregate pipeline;
  mobile/desktop assignment navigation, incomplete-work recovery, packet-scoped position persistence, accessible
  score names, reviewer competence and session-condition declarations, and corrected contrast/landmarks;
  26/26 corpus tests; exact binding of every response to an exported immutable packet envelope; safe in-memory
  normalization of preserved v1 packets; required immediate-comprehension judgments, per-dimension evidence,
  explicit `not_applicable`/`uncertain` states, and complete finding repair contracts; 12 preserved superseded
  v1 assignments plus the first 12 of 48 Pass A assignments prepared under rubric v2 over the same
  candidate-content population with zero fabricated judgments. Pass C is proven to create zero assignments
  before all A/B prerequisites are sealed;
- D5 repeat substrate: future Pass A sessions can interleave six blinded consistency presentations without
  duplicating candidates or candidate reviews; responses, scores, findings, and derived stability remain
  append-only. The live repeat population is correctly zero before human Pass A. Pass B preparation now fails
  closed until all current candidates have sealed Pass A evidence, all required repeat-stability rows exist,
  and no first-class Pass A presentation session remains assigned;
- D5 closeout substrate: exact A/B/repeat/C/structural/analysis evidence is frozen into a resumable Pass D
  packet; candidate adjudication bases, failure clusters, distribution assessments, uncertainty, and
  recommended states are append-only; SQL forces zero execution authority; live preparation fails at missing
  Pass A and creates no assignment, lifecycle transition, release member, or training exposure;
- D5 deterministic evidence: one current-version snapshot, 236 scoped metrics, 2,256 pair/method surface
  edges, 488 dynamic signatures, and an append-only correction for one erroneous software-revision claim;
- GPT-5.5, training, GPU work, live Donto writes, and release remained unused during generation. A separately
  authorized factual Discord progress timer was installed after the calibration.

Open before D5 acceptance:

- implement Appendix D sections 3.1–3.4 as a pass-specific, contract-indexed Pass B worksheet without
  invalidating sealed Pass A rubric-v2 evidence;
- human blind and contract-aware review of all 48 calibration candidates;
- family synthesis for all six families and separate content/schema disposition for all six structural
  rejections;
- non-binding campaign closeout over the complete human and deterministic evidence;
- critic calibration, if a critic is later justified;
- human-grounded false-accept/false-reject measurement;
- operator decision on production generation.

No model critic ran in Execution 01. The current six rejections are deterministic
`unknown_secondary_lens` findings, so human disagreement with their disposition is a schema/taxonomy
diagnostic rather than a critic false-reject measurement. PRD-12 creates the human reference against which a
later, separately authorized critic can be measured.

## 3. Stage D1 — External review and decision reconciliation

### Work

- circulate the suite using Appendix C;
- obtain critiques from language, ontology/philosophy, dialogue, synthetic-data, data-systems, and small-model
  perspectives where possible;
- store reports in the repository or linked research archive;
- create a claim/counterclaim table;
- resolve contradictions into dated decisions or explicit open questions;
- identify minimal changes required before implementation.

### Acceptance

- every reviewer identifies expertise and source scope;
- feedback is traceable to exact PRD sections;
- conflicting advice remains visible;
- adopted changes cite rationale;
- no novelty statement depends on unverified reviewer assertions;
- operator ratifies or amends PRD-00.

### Unlocks

A bounded D2 implementation authorization.

## 4. Stage D2 — Empty ledger and migration substrate

### Work

- choose implementation language/runtime consistent with repository constraints;
- implement schema migrations and version registry;
- implement immutable IDs, versions, blobs, events, and digests;
- implement a small but representative vertical slice across program, taxonomy, family, dialogue, generation,
  review, release, rendering, and evaluation tables;
- implement public-safe views;
- document storage layout.

### Acceptance

- fresh database from zero succeeds;
- migrations match disk and are hash-verified;
- foreign keys and strict constraints enabled;
- append-only/sealed immutability tested;
- raw/canonical/rendered artifacts round-trip;
- rejection retention proven;
- database integrity survives forced interruption;
- all artifacts stay on approved mounted storage;
- no model calls have occurred.

### Unlocks

D3 authoring and query interface.

## 5. Stage D3 — Family authoring, taxonomy, and query interface

### Work

- load versioned taxonomy and Donto crosswalk;
- implement open-lens proposals and typed category relations;
- create family-blueprint authoring and validation workflow;
- implement semantic states, commitments, dependencies, deltas, projections, transformations, and branches;
- implement stable read views and example cohort queries;
- author a few hand-written synthetic canary families solely to validate representation, not train a model.

### Acceptance

- all PRD-01 meta-classes remain distinct;
- one family demonstrates plurality, one source evidence, one pact revision, and one false bridge;
- a new lens can be proposed/reviewed without schema migration;
- family ancestry/split propagation works;
- example queries in PRD-02 and PRD-06 succeed;
- canary artifacts are clearly tagged non-production;
- no external generation campaign has occurred.

### Unlocks

D4 orchestration calibration authorization.

## 6. Stage D4 — Generation orchestrator and call ledger

### Work

- implement model/provider aliases and exact revision registry;
- implement prompt/tool-schema versioning;
- implement schema-constrained calls;
- implement bounded campaigns, tasks, attempts, routing, budgets, raw artifacts, and idempotent resume;
- implement deterministic validators;
- implement orchestrator summaries and stop conditions;
- configure secrets outside Git/SQLite.

### Acceptance

- mocked/offline runs prove lifecycle and failure handling;
- a provider calibration probe occurs only under explicit call authorization;
- raw and structured outputs both retained;
- empty/schema-invalid/rate-limited responses remain failures;
- same task cannot double-commit;
- strongest-model and worker call counts visible;
- provider change invalidates calibration;
- no free-text JSON parsing;
- secrets absent from repository, database, and logs.

### Unlocks

D5 calibration generation.

## 7. Stage D5 — Small generation calibration

### Work

- select a small set of diverse family blueprints;
- compare worker models/prompt variants on surface tasks;
- compare critic profiles;
- perform human blind audit;
- estimate accepted family value, style concentration, error clusters, and cost;
- tune routing and stop rules.

### Acceptance

- every call is within a hard authorized budget;
- at least two materially different lens families tested;
- human audit covers high/low critic-confidence items;
- critic false-accept and false-reject behavior measured;
- question rate, length, style signature, and semantic duplicates reported;
- failed and rejected population retained;
- no claim that calibration rows form a training corpus;
- operator adjudicates whether production generation is justified.

### Unlocks

D6 evaluation construction and D7 pilot generation, separately.

## 8. Stage D6 — AlphaPact and ordinary-chat freeze

### Work

- author independent evaluation families;
- implement the public-ledger/private-vault firewall in PRD-13 before storing private item plaintext;
- establish authority type and adjudication;
- create public development and sealed private portions;
- benchmark humans, strong models, and available small baselines;
- test shortcuts and score reliability;
- freeze decoder and primary metrics;
- quarantine semantic relatives.

### Acceptance

- family-level independence and leakage audit pass;
- every public evaluation table remains browseable while private prompt/answer plaintext remains absent from
  the public SQLite file, blob tree, HTTP responses, logs, and training-visible artifacts;
- human agreement supports the adjudicable subset;
- set-valued cases preserve legitimate disagreement;
- terminology-scrub and false-bridge controls work;
- benchmark has nontrivial difficulty range;
- private content is sealed and excluded from public/training views;
- response-initiation suite is operational.

### Unlocks

Training-release construction, not training itself.

## 9. Stage D7 — Synthetic corpus pilot

### Work

- generate approved family allocation in resumable batches;
- review and adjudicate;
- construct ordinary-chat, linguistic, conceptual, pact, evidence, short-form, and negative cohorts;
- perform source/rights/style/duplication/leakage audits;
- seal pilot ledger and release candidate.

### Acceptance

- family-level coverage meets predeclared allocation;
- human-audited conceptual precision meets threshold;
- style/teacher concentration within limits;
- evaluation contamination absent;
- rendering is deterministic;
- release card and manifest complete;
- raw/rejected/contested inventory reconciles with all attempts;
- operator explicitly accepts a particular release for D8.

### Unlocks

Synthetic-only canary training.

## 10. Stage D8 — Training-stack canary and response pilot

### Work

- build/deploy pinned Alpha training stack;
- run fail-closed GPU and numerical tests;
- run a tiny end-to-end training canary;
- verify dataset/render/token/loss masks;
- run bounded response-initiation pilot;
- mirror and hash all artifacts;
- terminate paid infrastructure.

### Acceptance

- every intended NVIDIA test executed;
- no unintended CPU fallback;
- data bytes match sealed release;
- training steps and checkpoints advance;
- no nonfinite or silent kernel errors;
- free generation and metrics run exactly;
- pod termination verified;
- response pilot reaches the predeclared gate or is recorded as failure;
- no continuation occurs automatically.

### Unlocks

D9 primary synthetic-only run only after separate authorization.

## 11. Stage D9 — Synthetic-only conversational foundation

### Work

- randomly initialize selected one-GPU-feasible Alpha configuration;
- train only on the accepted synthetic release;
- monitor response-start, ordinary chat, and stability;
- select checkpoints under frozen rules;
- stop at authorized compute/budget or futility boundary.

### Acceptance

- exact synthetic-only exposure proven;
- response-initiation/ordinary-chat gate adjudicated;
- no private-eval tuning;
- artifacts mirrored and serving conversion verified if relevant;
- negative result retained honestly;
- operator decides whether conceptual intervention is interpretable.

### Unlocks

D10 causal curriculum study.

## 12. Stage D10 — Linked curriculum causal study

### Work

- create equal-budget independent, linked, and corrupted arms;
- verify relation visibility;
- run sequential one-GPU arms from the same checkpoint/configuration;
- evaluate frozen family-level outcomes;
- perform human review and statistical analysis.

### Acceptance

- arm differences and matching documented;
- exact exposure and seeds preserved;
- correct relation compared with corruption;
- primary endpoint reported regardless of sign;
- ordinary-conversation non-degradation reported;
- no scaling decision precedes adjudication.

### Unlocks

D11 scale, revise, or reject decision.

## 13. Stage D11 — Scale decision

Possible decisions:

- scale successful family types;
- revise weak taxonomy/generator areas;
- test capacity/configuration threshold within one GPU;
- separate plurality from pact revision;
- add entity-light evidence study;
- test a pretrained or human-data ablation;
- publish a null/negative result;
- stop the model program while releasing the ledger/benchmark.

The decision cites causal evidence, not generation-pipeline momentum.

## 14. Stage D12 — Public artifact

### Dataset release acceptance

- public-safe ledger reconstructed from policy views;
- manifests and hashes verified elsewhere;
- licenses, provenance, model use, human review, and limitations disclosed;
- private/restricted data absent;
- example exports reproduced;
- error-report process available.

### Model release acceptance

- ordinary conversation and declared specialized gates pass;
- checkpoint traceable to exact exposure;
- standard inference load verified;
- model card states synthetic-only status and factual limitations;
- no fallback/canned behavior disguises model failures;
- Space/demo, if built, serves the exact model;
- Discord post occurs only if a qualitative improvement case passes its separate contract.

## 15. Documentation deliverables by stage

Every stage adds:

- dated decision record;
- scope/authorization;
- execution plan;
- exact commands or interface versions once implementation exists;
- validation report;
- artifact manifest;
- failures and unresolved questions;
- next-stage recommendation;
- updated resume/handoff pointer.

## 16. Definition of “done” for this PRD suite

The current documentation task is done when:

- the complete suite exists under one indexed folder;
- GOAL.md identifies it as the current planning goal while preserving archive history;
- AGENTS.md and resume decisions make the authorization boundary unmistakable;
- internal links and source paths validate;
- no size-specific product identity remains in the suite;
- synthetic data is visibly a principal half of the project;
- the first experiment is unambiguously synthetic-only;
- all changes are committed, pushed, and verified on the remote branch.

The original documentation definition is complete. Execution 01 proves a bounded D2–D4 vertical slice and the
generation portion of D5. PRD-12 makes the remaining D5 human gate executable; it does not pass that gate or
authorize D6–D12.
