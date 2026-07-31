# PRD-02 — SQLite scientific ledger

## 1. Product statement

Alpha Corpus SHALL track everything material in SQLite.

That includes raw prompts and responses, incomplete and rejected candidates, every revision, competing
analyses, category definitions, sources, licenses, model identities, reviewer judgments, family topology,
rendered chat bytes, tokenization, loss masks, release membership, training order, checkpoint exposure, model
outputs, costs, failures, and later corrections.

The qualification is physical, not philosophical: the logical data model is comprehensive from the start,
while expensive materializations—such as one row per token occurrence—may be created lazily from immutable
source artifacts until a query or experiment needs them. Deferral may change *how* a fact is stored, never
whether it can be reconstructed.

## 2. Why SQLite

SQLite is appropriate for the canonical research artifact because it is:

- a single portable file that can accompany a release;
- transactional and durable;
- rich enough for normalized relational constraints, views, triggers, FTS, JSON metadata, and recursive
  queries;
- inspectable by researchers without operating a service;
- reproducible and hashable;
- suitable for local-first generation and review;
- easy to snapshot at release boundaries;
- compatible with later read replicas, Parquet exports, or a server index without surrendering canonicity.

SQLite is not the only runtime representation. Large blobs may be content-addressed files, token streams may
be exported to packed shards, and analytical mirrors may use DuckDB/Parquet. The SQLite ledger remains the
authority that says what each artifact is, where it came from, and how it was used.

## 3. Non-negotiable invariants

### L1 — No destructive overwrite

Scientific records are append-only. A correction creates a successor and a typed supersession relation. Rows
may be excluded from a release or cryptographically tombstoned under a future privacy policy; ordinary review
never deletes history.

### L2 — Content addressability

Every immutable text, binary artifact, rendered unit, prompt, response, manifest, and export has a digest over
canonical bytes. Duplicate bytes may share storage while retaining distinct provenance events.

### L3 — Exact generation reconstruction

A model call is reconstructable from provider, model ID, model revision if exposed, parameters, tool schema,
system/developer/user messages, attachments, prompt-template revision, seed when meaningful, request time,
response bytes, status, latency, and usage report.

### L4 — Exact model exposure reconstruction

For any training step or checkpoint, the system can determine the release, sampler, rendering profile,
tokenizer, packing, loss mask, order or RNG state, unit weights, and exact model-visible bytes.

### L5 — Delimiter independence

Canonical dialogue content stores roles and message text separately. Chat templates, `<assistant>`-style tags,
BOS/EOS, separators, padding, and packing sentinels exist only in versioned render artifacts.

### L6 — Rejections are first-class

Failed generations, malformed responses, duplicates, unsafe units, culturally unauthorized material,
invalid counterexamples, judge disagreements, and null outputs remain stored with reasons and lineage.

### L7 — Plurality is representable

The ledger can hold mutually incompatible analyses without selecting a global winner. Authority and scope are
explicit.

### L8 — Family-level split integrity

Train, development, public evaluation, and private evaluation membership is assigned at the declared
independence unit. Related branches, paraphrases, prompt templates, and lexical clusters cannot leak silently.

### L9 — Derived facts identify their derivation

Metrics, embeddings, classifications, and quality scores record algorithm/model, configuration, input digest,
and execution. They never masquerade as manually established ground truth.

### L10 — Database self-description

Schema version, migration history, invariants, validation reports, dataset card, licenses, and release
manifests travel with the database.

## 4. Identity strategy

All durable research objects use opaque stable identifiers, preferably UUIDv7 or equivalent time-sortable
random IDs. Human-readable slugs are mutable aliases. Natural-language labels, predicate names, and hashes are
not primary keys.

Each versioned object uses:

- stable logical object ID;
- immutable version ID;
- version number or parent relation;
- transaction time;
- optional valid interval;
- creator actor and process;
- status;
- content digest;
- reason for change.

## 5. Logical schema catalog

The catalog below is deliberately rich. Implementation may stage migrations, but any omitted table requires a
documented alternative that preserves the same reconstructability.

### 5.1 Program and governance

| Table | Purpose |
|---|---|
| `program` | Named research program, north star, authority, state |
| `program_version` | Immutable goal and protocol revisions |
| `decision` | Dated binding decisions and supersessions |
| `open_question` | Unresolved research or implementation decisions |
| `risk` | Risk, likelihood, impact, mitigation, owner, status |
| `stage_gate` | Gate definition and prerequisites |
| `gate_evidence` | Artifacts and adjudications offered for a gate |
| `gate_decision` | Pass/fail/defer plus authority and rationale |
| `actor` | Human, model, service, organization, or anonymous contributor |
| `actor_role` | Time-scoped roles and authorities |
| `policy` | Versioned privacy, cultural, release, safety, and quality policies |
| `policy_binding` | Which object/release/context a policy governs |

### 5.2 Taxonomy and ontology

| Table | Purpose |
|---|---|
| `category` | Stable category identity |
| `category_version` | Definition, examples, status, authority, hazards |
| `category_alias` | Names and language variants |
| `category_relation` | Broader/narrower/overlap/incompatible/projection/related/supersedes |
| `category_proposal` | Open-lens minting proposal |
| `category_proposal_review` | Reviews and outcomes of a proposal |
| `annotation_dimension` | Declared field/dimension and its meta-class |
| `annotation_value` | Versioned value record where values are not free text |
| `annotation_assignment` | Sparse scoped assertion that an object has a category/value |
| `annotation_disagreement` | Explicit conflict between assignments or reviewers |
| `coverage_requirement` | Desired coverage without conflating it with observed coverage |
| `coverage_observation` | Computed coverage for a cohort or release |

`annotation_assignment` identifies whether it is a design variable, semantic ground truth, observed judgment,
or derived result. It also records confidence, scope, source, and review state.

### 5.3 Concept families and semantic state

| Table | Purpose |
|---|---|
| `concept_family` | Stable scientific independence unit |
| `family_version` | Immutable family blueprint revision |
| `family_competency_question` | Questions the distinction should help answer |
| `family_projection` | Linguistic, ontological, social, material, evidential, etc. realization |
| `family_projection_relation` | True bridge, false bridge, partial mapping, withheld projection |
| `scene` | Situation, purpose, participants, context, target behavior |
| `scene_version` | Immutable scene content/contract revision |
| `trajectory` | Ordered stateful dialogue or case sequence |
| `trajectory_member` | Scene/turn order and branch identity |
| `branch_point` | Shared prefix and available continuations |
| `transformation` | Typed intervention definition |
| `transformation_edge` | Source state, target state, intervention and composition metadata |
| `semantic_state` | Hidden researcher-side commitment state at a point |
| `state_commitment` | Proposition, status, holder, scope, time, authority |
| `commitment_dependency` | Why one commitment depends on another, evidence, or pact |
| `expected_delta` | Preserve/add/retract/pluralize/attribute/temporalize/unsupported |
| `admissible_analysis_set` | Set-valued resolution object |
| `admissible_analysis_member` | Required/permitted/excluded reading and rationale |
| `discriminating_evidence` | Evidence or clarification that would reduce alternatives |
| `invariance_constraint` | Relation that must hold across outputs |
| `shortcut_hazard` | Vocabulary, template, scenario, or answer-form shortcut |

Commitments can be held by the user, Alpha, another quoted speaker, a source, or shared ground. Acceptance for
the purpose of a dialogue does not imply private belief.

### 5.4 Natural-language content

| Table | Purpose |
|---|---|
| `dialogue` | Logical conversation identity |
| `dialogue_version` | Immutable dialogue revision |
| `participant` | Participant identity and role within a dialogue |
| `message` | Stable message identity |
| `message_version` | Role, natural text, language, parent, transaction time |
| `message_span` | Addressable span for claims, errors, and annotations |
| `utterance_relation` | Reply, correction, paraphrase, elaboration, quotation, interruption |
| `local_term` | Conversation-scoped term or representational choice |
| `local_term_version` | Definition, scope, accepted status and revision |
| `dialogue_state_link` | Connects turns to semantic states before/after |
| `response_policy_target` | Answer/ask/challenge/stop/etc. design target |
| `source_attachment` | Source fragment visible to a message or scene |

Message content never includes a mandatory chat delimiter. A message may contain literal markup when it is
part of what a participant said; that is distinguished from rendering syntax.

The preliminary `response_policy_target` relation is expanded conceptually by
[PRD-14](PRD-14-RESPONSE-POLICY-CONTROL-PLANE.md). Raw free-form instructions remain immutable; future policy
definitions, target components, compiled instructions, observations, and distribution findings are separate
versioned evidence rather than a destructive string normalization.

### 5.5 Sources, evidence, and rights

| Table | Purpose |
|---|---|
| `source` | Work, document, dataset, observation, or synthetic micro-world |
| `source_version` | Immutable bibliographic and content state |
| `source_fragment` | Addressable quoted/paraphrased region |
| `evidence_anchor` | Link from a claim/annotation to a fragment and anchoring method |
| `source_relation` | Copies, cites, translates, summarizes, contradicts, derives from |
| `claim` | Source or analyst claim separate from the dialogue surface |
| `claim_version` | Polarity, modality, scope, time, interpretation level |
| `claim_argument_edge` | Supports, rebuts, undercuts, qualifies, explains, alternate analysis |
| `license` | Normalized rights instrument |
| `rights_assertion` | Actor's claim about allowed use and jurisdiction |
| `consent_record` | Consent scope for human-contributed material |
| `cultural_authority` | Required community/expert validation and restrictions |
| `privacy_classification` | Privacy status and review |
| `redaction` | Non-destructive redaction/tombstone record |

Synthetic sources remain explicitly synthetic. A teacher's invented passage cannot silently become an
attested real-world source.

### 5.6 Model, prompt, and tool registry

| Table | Purpose |
|---|---|
| `provider` | Provider/service identity and terms snapshot |
| `model` | Logical model family |
| `model_revision` | Exact model/version/API identity and capability profile |
| `model_role_profile` | Orchestrator, worker, critic, judge, repairer eligibility |
| `prompt_template` | Stable template identity |
| `prompt_template_version` | Exact messages/instructions and variables |
| `tool_schema` | Structured-output or tool-call schema version |
| `generation_recipe` | Model, prompts, parameters, role, retry and routing policy |
| `review_recipe` | Reviewer assignment and rubric configuration |
| `renderer` | Logical rendering system |
| `renderer_version` | Exact template and normalization rules |
| `tokenizer` | Logical tokenizer |
| `tokenizer_version` | Files/digests/configuration/special-token map |
| `software_component` | Repository, package, binary, or service identity |
| `software_revision` | Commit, build digest, environment and dependency lock |

Exact provider model IDs are runtime records rather than permanent prose assumptions. This lets the program
use later 5.x variants without rewriting old provenance.

### 5.7 Generation execution

| Table | Purpose |
|---|---|
| `generation_campaign` | Bounded objective, budget and cohort target |
| `generation_batch` | Atomic resumable batch |
| `generation_task` | One planned generation/review/repair action |
| `model_call` | Exact request/response lifecycle |
| `model_call_message` | Ordered input messages independent of prompt template |
| `model_call_tool` | Tool definitions and returned calls |
| `model_call_usage` | Provider usage, measured or estimated tokens and cost |
| `model_call_attempt` | Retries, transient failure, fallback routing |
| `raw_artifact` | Response body, log, attachment, or structured result digest |
| `candidate` | Candidate unit or blueprint produced by a call/human |
| `candidate_version` | Immutable content revision |
| `candidate_parent` | Generate, critique, repair, paraphrase, branch, merge lineage |
| `candidate_failure` | Null, truncation, schema, policy, quality, duplication, other failure |
| `generation_event` | Append-only state transition |
| `routing_decision` | Why orchestrator selected or escalated a model |
| `budget_event` | Reserved, charged, refunded, estimated, overrun, denied |

Raw free text is always retained even when a structured call also succeeds. Structured fields come from a
schema-valid tool/output path, never regex extraction from prose.

### 5.8 Review and adjudication

| Table | Purpose |
|---|---|
| `rubric` | Stable quality construct |
| `rubric_version` | Exact criteria and anchors |
| `review_assignment` | Candidate, reviewer, blindness, order, deadline |
| `human_review_session_declaration` | Reviewer competence scope, timing, interruption, fatigue, conditions, and exact packet/submission hashes |
| `human_review_session_competence` | Normalized declared competence values for one immutable human session declaration |
| `review_presentation_session` | Ordered hash-bound review session, including repeat allocation |
| `review_presentation` | One primary or hidden-repeat appearance without duplicating the review |
| `review_presentation_response` | Immutable response to one appearance |
| `review_presentation_score` | Dimension scores for a presentation response |
| `review_presentation_finding` | Evidence-grounded findings for a presentation response |
| `review` | Immutable review event |
| `review_dimension_score` | Separate score/decision per construct |
| `review_finding` | Span-grounded issue or strength |
| `reviewer_calibration_item` | Gold/contested calibration case |
| `reviewer_calibration_result` | Agreement and failure profile |
| `adjudication` | Authority decision after reviews |
| `adjudication_basis` | Which evidence/reviews support it |
| `disagreement_case` | Preserved unresolved review conflict |
| `repair_request` | Specific requested revision |
| `quality_state_transition` | Candidate lifecycle history |
| `family_synthesis_assignment` | Pass C reviewer/session/input-snapshot gate for one family |
| `family_synthesis` | Immutable comparison and diagnosis across one family's siblings |
| `family_synthesis_basis` | Exact sealed candidate reviews used by a family synthesis |
| `structural_disposition` | Separate content/schema judgment for a structurally rejected candidate |
| `structural_disposition_basis` | Exact validator failures and reviews supporting that disposition |
| `campaign_closeout_assignment` | Pass D adjudicator/session/input-snapshot workflow gate |
| `campaign_closeout` | Immutable non-binding campaign synthesis and authority acknowledgement |
| `campaign_closeout_state` | Evidence-supported D5 states recommended for later operator decision |
| `campaign_closeout_basis` | Exact adjudication, synthesis, repeat, structural, and analysis evidence |
| `campaign_failure_cluster` | Campaign-level diagnosis, repair proposal, and later-call requirement |
| `campaign_failure_cluster_member` | Exact frozen evidence assigned to a failure cluster |
| `campaign_distribution_assessment` | Required conversational-distribution finding and evidence references |

Reviews never overwrite one another. An adjudication can select an action without erasing minority analysis.
Family synthesis is not candidate adjudication and does not imply release or training approval. The current
physical schema enforces sealed Pass A and Pass B evidence before Pass C assignments may be prepared; see
[Execution 06](EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md).

### 5.9 Similarity, duplication, and contamination

| Table | Purpose |
|---|---|
| `embedding_model` | Exact embedding method/version |
| `embedding` | Vector or external artifact pointer plus input digest |
| `similarity_edge` | Lexical/semantic/structural similarity with method |
| `duplicate_cluster` | Proposed or adjudicated duplicate group |
| `cluster_member` | Candidate membership and confidence |
| `contamination_probe` | Train/eval/source overlap test definition |
| `contamination_result` | Finding and severity |
| `template_signature` | Detectable teacher/prompt/style signature |
| `lexical_holdout` | Prohibited or held-out words/phrases for a family |

Similarity proposes; it does not destructively deduplicate. Near-duplicates may be valuable controlled
paraphrases if the family says so.

The D5 physical schema now materializes `analysis_method`, `analysis_run`, `analysis_metric`,
`similarity_edge`, `template_signature`, and `analysis_run_correction`. The last table preserves corrections
to derived-run provenance without mutating the erroneous run. Embeddings, duplicate clusters, contamination
probes, and lexical holdouts remain later materializations: no empty or guessed semantic record is created
merely to make the physical schema resemble this catalog.

### 5.10 Releases, cohorts, and exports

| Table | Purpose |
|---|---|
| `cohort_definition` | Versioned query/selection policy |
| `cohort_snapshot` | Frozen evaluated result of that policy |
| `cohort_member` | Object membership and reason |
| `dataset_release` | Named immutable release |
| `release_member` | Candidate/dialogue/family membership, split and weight |
| `release_exclusion` | Explicit exclusion and reason |
| `release_manifest` | Canonical manifest digest and metadata |
| `release_validation` | Referential, policy, coverage and leakage checks |
| `render_profile` | Chat template, message selection, normalization, labels |
| `render_job` | Bounded deterministic materialization |
| `rendered_unit` | Exact bytes and logical source |
| `rendered_message_map` | Span map from rendered bytes/tokens to source messages |
| `token_sequence` | Token IDs artifact and digest |
| `loss_mask` | Token-level supervision mask artifact |
| `packed_sequence` | Packing order and boundary map |
| `export_artifact` | JSONL/Parquet/Arrow/shard/card/database deliverable |
| `export_validation` | Re-import, count, hash, schema and sample checks |

### 5.11 Training and checkpoint exposure

| Table | Purpose |
|---|---|
| `training_experiment` | Scientific comparison and predeclared hypothesis |
| `training_arm` | Data/objective condition |
| `training_run` | Seed, code, environment, GPU, configuration, start/end |
| `training_dataset_binding` | Release/cohort/render profile bound to a run |
| `sampler_config` | Scheduling, weighting, curriculum and RNG contract |
| `training_exposure` | Unit/batch exposure, step/epoch/order/weight |
| `checkpoint` | Model/optimizer/RNG/tokenizer artifact identity |
| `checkpoint_parent` | Initialization and continuation lineage |
| `training_metric` | Raw or derived measurement with step and method |
| `training_failure` | Crash, nonfinite, stall, integrity, resource, early stop |
| `compute_usage` | GPU time, energy estimate, storage, CPU/RAM, wall time |
| `run_adjudication` | Whether a run is valid for the declared experiment |

Exposure may initially be stored as deterministic sampler state plus batch manifests rather than one SQLite row
per token. It must still reconstruct the exact sequence. A later materialized `token_exposure` table is allowed
when per-token causal queries justify its size.

### 5.12 Evaluation and model behavior

| Table | Purpose |
|---|---|
| `evaluation_suite` | Stable test construct |
| `evaluation_suite_version` | Frozen items, policies, primary metrics |
| `evaluation_item` | Prompt/trajectory/probe identity |
| `evaluation_family_binding` | Independence and leakage relationships |
| `evaluation_run` | Model, checkpoint, decoder, environment |
| `evaluation_output` | Exact generated bytes/tokens/logprobs/status |
| `behavior_annotation` | Human/model/executable judgment |
| `metric_definition` | Formula, aggregation unit and version |
| `metric_observation` | Raw or aggregate result |
| `pairwise_comparison` | Blinded preference and order |
| `human_session` | Consented interaction session metadata |
| `human_turn` | Separately governed human/model messages |
| `evaluation_failure` | Empty, loop, timeout, invalid decoder, contamination |
| `statistical_analysis` | Model/specification, family clusters, intervals |

The D6 materialization contract is [PRD-13](PRD-13-EVALUATION-FIREWALL-AND-FREEZE.md). Private evaluation
payloads use public commitments and encrypted external vault objects: their metadata, lineage, hashes,
policies, runs, and safe aggregates remain visible in the public all-table ledger, but prompt and answer
plaintext never enters the publicly served SQLite file before retirement.

### 5.13 Artifact and event substrate

| Table | Purpose |
|---|---|
| `blob` | Digest, size, media type, inline/external location |
| `blob_location` | Redundant local/remote storage and verification state |
| `artifact_relation` | Derived-from, contains, renders, supersedes, validates |
| `event` | Append-only operational/scientific event |
| `event_object` | Objects affected by an event |
| `validation_run` | Named validator execution |
| `validation_finding` | Error/warning/info tied to objects |
| `schema_migration` | Ordered migration digest and application state |
| `database_snapshot` | SQLite file digest, schema and release association |
| `external_identifier` | DOI, arXiv, Hugging Face, Git commit, URL, etc. |

## 6. State machines

### 6.1 Candidate lifecycle

`proposed → generated → structurally_valid → under_review → accepted | rejected | disputed | restricted`

Accepted candidates may become `released`, `superseded`, or `retired`. Rejected candidates may become the
parent of repaired versions; the original remains rejected.

### 6.2 Family lifecycle

`draft → internally_reviewed → calibrated → frozen_for_generation → frozen_for_evaluation → retired`

A family cannot be in training and later moved into private evaluation under a new label.

### 6.3 Release lifecycle

`planned → materialized → validated → sealed → published → superseded`

Sealed releases are immutable. A fix creates a new release.

## 7. Canonical text and normalization

Raw provider bytes are retained exactly. A separate canonicalization process may create normalized Unicode,
newline, or whitespace variants. The transformation and version are stored; normalization never replaces raw
content.

Dialogue messages contain:

- participant/role reference;
- raw text artifact;
- canonical text artifact;
- language and script;
- parent message or reply relation;
- visible source attachments;
- timestamps/order;
- optional span annotations.

Chat template rendering is a pure function of a message sequence, renderer version, tokenizer version, and
configuration. Its output is content-addressed and can be reproduced byte-for-byte.

## 8. Structured model output

When the generation system needs structured data, it SHALL use the model/API's schema-constrained structured
output or required tool call. The exact schema is versioned in `tool_schema`. Free-form JSON fenced inside
assistant prose is not parsed as authoritative structure.

Failure to produce a valid structured result creates a failed attempt and may be retried or escalated. It does
not silently create partial rows.

## 9. Queries the ledger must answer

Examples include:

- Which accepted conversations teach mereology through ordinary nontechnical language and end without a
  follow-up question?
- Which family projections were generated by one worker model and reviewed by a provider-independent critic?
- Show every rejected counterexample later repaired into an accepted unit.
- Build a release with no public named entities, no source license below a threshold, and no family overlap
  with AlphaPact private evaluation.
- Compare review yield and human disagreement across prompt revisions.
- Find units where the planned speech act was `challenge` but reviewers observed `hedge`.
- Trace this token sequence back to source messages, generation calls, blueprint, reviews, renderer, and
  release membership.
- Determine every checkpoint that saw any descendant or paraphrase of a private evaluation family.
- Construct a short-answer-heavy ordinary-chat cohort with controlled category and style balance.
- Identify newly minted lenses whose families produced held-out behavioral gains.
- Re-run a historical export using the exact renderer and tokenizer.
- Compute the accepted-to-rejected ratio without losing null, timeout, or schema-failure attempts.
- Separate stated source content from teacher interpretation and from evaluator ground truth.
- Retrieve all legitimate plural-analysis cases and measure model undercoverage versus overcoverage.

## 10. Performance and physical layout

Recommended implementation properties:

- WAL mode for local concurrent readers and bounded writer transactions;
- foreign keys always enabled;
- strict tables where compatible;
- check constraints for closed operational state machines;
- FTS indexes over candidate, message, source, review, and definition text;
- ordinary indexes on family, status, digest, model, prompt, release, split, and time;
- large immutable blobs stored inline below a threshold and on the mounted data disk above it;
- relative content-addressed blob paths so a release directory is portable;
- periodic integrity checks and backup copies of sealed releases;
- no research artifacts on the root boot disk when the mounted data drive is available;
- a read-only public view/database with private and secret-bearing tables excluded.

The secret store is not SQLite. API keys, webhooks, and provider credentials are referenced by opaque secret
handles and remain in appropriate mode-restricted external storage.

## 11. Views and public interface

The implementation should provide stable views rather than require users to understand every normalized table:

- `v_training_conversation`;
- `v_message_with_categories`;
- `v_family_coverage`;
- `v_candidate_review_summary`;
- `v_rejected_candidate_lineage`;
- `v_release_unit`;
- `v_source_rights`;
- `v_generation_cost_yield`;
- `v_training_exposure_lineage`;
- `v_evaluation_family_result`;
- `v_open_lens_proposal`;
- `v_public_safe_artifact`.

Views are versioned API contracts. A change that alters meaning requires a new view version or release.

## 12. Validation suite

The ledger implementation must prove:

- fresh creation from migrations;
- migration idempotence where promised and rejection of out-of-order migrations;
- foreign-key consistency;
- no mutable update path for sealed artifacts;
- raw-to-canonical-to-rendered round trip;
- deterministic export and re-import;
- preserved rejected population;
- exact candidate ancestry;
- family-level split leakage detection;
- prompt/model/reviewer completeness;
- source/license/policy completeness for releaseable units;
- token sequence and loss-mask agreement with renderer output;
- checkpoint exposure reconstruction;
- content-address collision defense and digest verification;
- privacy/public-view exclusion;
- crash-safe batch resume without duplicate scientific objects;
- database integrity after abrupt interruption;
- snapshot manifest verification on another machine.

## 13. Public release package

A full ledger release should contain:

- sealed `.sqlite` database;
- migration history;
- schema documentation and entity-relationship overview;
- content-addressed public blob tree or archive;
- manifest with digests and sizes;
- dataset card and limitations;
- license and rights report;
- release validation report;
- example queries;
- deterministic export tool/version pointer;
- frozen train/development/public-eval partitions;
- explicit statement that private evaluation content is absent.

## 14. Acceptance criteria

PRD-02 is complete when a clean implementation can demonstrate all ten invariants, answer the required query
classes, reconstruct every released model-visible byte, retain every rejected attempt, and prove that chat
delimiters and tokenization were introduced only by an identified renderer. A compact early physical schema is
acceptable only if deferred materializations remain exactly derivable from immutable artifacts and their
derivation contracts are tested.

## 15. Current D5 physical checkpoint

Execution 05 records the first derived-evidence materialization. Execution 06 adds the fail-closed family
synthesis and structural-disposition evidence layer. Execution 07 separates review assignments from blinded
repeat presentations and adds a reviewer-stability view. Execution 08 adds the non-binding Pass D campaign
closeout, exact evidence bases, failure clusters, recommended states, and distribution assessments. The
canonical ledger now has seven migrations, 129 tables, five views, and 186 append-only triggers. The frozen
48-candidate snapshot contributes one
authoritative surface-analysis run containing 236 scoped metrics, 2,256 pair/method similarity edges, and 488
dynamic template signatures. A provenance-erroneous predecessor and its typed correction are intentionally
retained, so physical row counts include both runs.

The five new Pass C relations are empty by design while human Pass A and Pass B remain incomplete. Their
presence proves schema and workflow capability, not review completion. Their no-promotion and exact-basis
contract is recorded in [Execution 06](EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md).

The presentation relations are also empty while the original 12-item legacy Pass A packet remains open.
Future repeat responses remain separate from candidate reviews and are summarized by
`review_repeat_stability`; see [Execution 07](EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md).

The seven campaign-closeout relations are also empty by design. Pass D cannot prepare a packet until the same
human adjudicator has complete A/B reviews, the required hidden repeats, all family syntheses and structural
dispositions, and one current authoritative analysis run. Its schema forces `execution_authorized = 0` and
its write path cannot create lifecycle transitions, release members, or training exposures; see
[Execution 08](EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md).

This checkpoint satisfies only the bounded D5 surface-analysis requirements. It does not complete PRD-02's
future release, token-exposure, evaluation, embedding, contamination, or checkpoint-lineage catalog.
