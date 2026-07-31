# PRD-06 — Dataset releases, queries, and exports

## 1. Purpose

Alpha Corpus is designed as a reusable research substrate, not a single frozen JSONL file. Researchers should
be able to construct different pretraining, midtraining, SFT, preference, evaluation, or diagnostic datasets
from the same immutable ledger while knowing exactly what each unit represents and where it came from.

This PRD defines reproducible cohort selection, release sealing, model-specific rendering, and public use.

## 2. Release principles

- A release is an immutable snapshot, not a mutable “latest” query.
- Membership includes a reason, split, role, and optional weight.
- Family topology and negative/positive relationships remain available.
- Evaluation families are quarantined beyond row-level equality.
- Natural message content and model-specific chat syntax remain separate.
- Every export can be regenerated from the sealed ledger snapshot.
- Rights, provenance, limitations, and synthetic status travel with the artifact.
- Rejected and contested data may be released in distinct research tiers; they never masquerade as positives.
- Researchers can select data by semantic, linguistic, conversational, provenance, quality, and experiment fields.

## 3. Release layers

### 3.1 Ledger snapshot

The canonical normalized SQLite database plus public-safe content-addressed blobs. This is the richest artifact.

### 3.2 Family package

Blueprints, transformations, hidden contracts, reviews, and natural-language realizations for research on
relational structure.

### 3.3 Conversation package

Delimiter-independent dialogues and annotations suitable for custom rendering.

### 3.4 Rendered training package

Exact JSONL/Arrow/Parquet/binary shards for a named model/tokenizer/template/objective.

### 3.5 Preference/contrast package

Shared prefixes, chosen and rejected branches, and explicit relation/delta provenance.

### 3.6 Evaluation package

Public development/evaluation subset with executable or adjudicated contracts. Private evaluation remains
absent from public artifacts until retirement. PRD-13 specifies how the public ledger can expose all evaluation
metadata tables and cryptographic commitments without serving private prompt or answer plaintext.

### 3.7 Negative and disagreement package

Rejected candidates, critic failures, invalid counterexamples, disputed analyses, and repair lineages with
labels and use restrictions.

## 4. Quality/release tiers

- `raw-research`: every safe-to-share attempt, including failures;
- `bronze-auto`: structurally valid and automatically reviewed;
- `silver-reviewed`: calibrated batch human audit passed;
- `gold-adjudicated`: required unit/family expert review passed;
- `plural-contested`: competing analyses preserved with scope;
- `red-negatives`: verified hard negatives and failure exemplars;
- `restricted-index`: metadata only for non-redistributable artifacts;
- `frozen-eval-public`: public benchmark subset;
- `frozen-eval-private`: never shipped with training releases.

The tiers express evidence and allowed use, not a simplistic rank.

## 5. Cohort definition

A cohort is a versioned declarative selection with:

- source ledger snapshot;
- inclusion predicates;
- exclusion predicates;
- join/traversal rules;
- family independence rule;
- quality and authority requirements;
- coverage allocation;
- balancing/weighting strategy;
- random seed and stable ordering rule;
- duplicate policy;
- rights/release policy;
- renderer compatibility;
- expected counts and tolerances;
- purpose and owner.

The evaluated membership is stored as a snapshot. Re-running the same cohort against a newer ledger does not
silently change an existing release.

## 6. Selection dimensions for third parties

Users should be able to request, for example:

- linguistic phenomena and constructions;
- Donto-derived lens intersections;
- concept families and projections;
- dialogue functions and response policies;
- single-turn versus multi-turn trajectories;
- pact lifecycle stages;
- transformations and expected deltas;
- ordinary versus technical vocabulary;
- answer length and question necessity;
- ambiguity types and analysis-set size;
- source/evidence conditions;
- entity-light or fictional-only material;
- naturalness and human-review thresholds;
- teacher/provider diversity;
- license and cultural-authority constraints;
- hard-negative or preference-pair status;
- dialogue depth/context distance;
- specific languages and review status;
- release and exposure history;
- novelty/deduplication thresholds.

PRD-14 supplies the future versioned policy graph and target-versus-observed distribution records behind these
facets. Third-party selection must not treat legacy free-form policy prose as a stable categorical value.

The public interface should expose stable views and a small query builder. Users may also write SQL directly.

## 7. Example cohort specifications

### 7.1 Chat foundation

Natural ordinary dialogue, greetings, direct answers, repair, short/medium responses, answer-and-stop, varied
user styles, low technical density, strong question-rate and length controls.

### 7.2 Linguistic reasoning

Morphology, syntax, semantics, pragmatics, reference, discourse, ambiguity, translation, and metalinguistic
negotiation, with technical terminology capped and natural examples emphasized.

### 7.3 Ontology and philosophy

Mereology, roles, identity, time, events, evidence, causation, teleology, modality, social ontology,
counterexamples, and purpose-sensitive modeling.

### 7.4 AlphaPact trajectories

Adoption, delayed use, challenge, local revision, scope shift, recovery, cross-projection transfer, and matched
false bridges.

### 7.5 Evidence-first

Fictional or licensed passages, attribution, conflicting sources, valid-time/record-time, unknown versus
negative, and retrieval-required cases.

### 7.6 Short-form anti-verbosity

One- or two-sentence high-value responses, direct corrections, concise examples, and no-question closures,
balanced across conceptual lenses.

### 7.7 Red-team negatives

Invalid counterexamples, overhedging, style-only philosophy, source conflation, false analogy, collateral
revision, and canned conversation patterns.

## 8. Rendering profiles

A rendering profile declares:

- selected messages and visibility rules;
- system/developer instruction policy;
- chat template version;
- participant-to-role mapping;
- BOS/EOS behavior;
- separators and generation prompt;
- tokenizer version;
- truncation and context-window policy;
- label/loss-mask policy;
- packed-sequence policy;
- branch/preference serialization;
- Unicode and whitespace normalization;
- source attachment format;
- metadata visible to the model;
- output artifact format.

The source message text is never altered in place. A renderer creates a new content-addressed artifact and a
span/token map back to source messages.

## 9. Training modes supported

### 9.1 Causal language modeling

Render natural text or conversations with declared prediction masks.

### 9.2 Assistant-only supervised fine-tuning

Mask user/system tokens while preserving exact response-start, content, and EOS accounting.

### 9.3 Full-dialogue learning

Supervise all or selected roles only when the experiment intends the model to learn user-side generation.

### 9.4 Preference optimization

Export shared-prefix chosen/rejected pairs with hard-negative provenance. A “rejected” candidate used here must
be rejected for the intended conceptual relation, not merely uglier prose.

### 9.5 Contrastive or relation-aware objectives

Export sibling IDs, transformation labels, expected deltas, and positive/negative relation groups without
forcing a particular framework.

### 9.6 Evaluation

Export prompts, state transitions, permissible output sets, and executable/human scoring contracts separately
from training data.

## 10. Split policy

Default partitions:

- training;
- development;
- public test;
- private test;
- quarantine;
- negative-only research;
- restricted.

Split enforcement propagates through:

- family ancestry;
- projection relations;
- branches and transformations;
- paraphrases and repairs;
- source families;
- prompt/template families;
- lexical/semantic similarity clusters;
- manually declared leakage relations.

If a unit is found contaminated after sealing, record a contamination event, publish a new release, and retain
the old release with an explicit warning.

## 11. Manifest

Every release manifest contains:

- release identifier/version/date;
- parent releases;
- ledger snapshot digest and schema version;
- cohort definition digest;
- member IDs and membership reasons;
- split and family counts;
- quality-tier distribution;
- category, transformation, language, style, source, and teacher coverage;
- rights and restrictions;
- renderer/tokenizer/software digests;
- artifact paths, sizes, and hashes;
- validation results;
- known limitations;
- responsible actors and approvals;
- whether any model has been trained on it;
- whether it contains synthetic, licensed, human, or mixed content.

## 12. Dataset cards

Cards must state:

- intended and prohibited uses;
- how content was generated and reviewed;
- models/providers used by exact revision where possible;
- what “synthetic” means in each subset;
- category and language limits;
- cultural-authority limits;
- known teacher signatures;
- judge/human calibration evidence;
- privacy and licensing;
- evaluation contamination risk;
- expected failure modes;
- recommended rendering and training cautions;
- how to cite the release;
- how to report discovered errors.

## 13. Reproducibility contract

Two independent machines with the sealed public snapshot and referenced public software must be able to:

1. verify all hashes;
2. recreate cohort membership;
3. render the same model-visible bytes;
4. recreate token IDs and loss masks with the named tokenizer;
5. reproduce aggregate release statistics;
6. re-import exported files without loss;
7. trace any example to family, generation, and review lineage.

Provider calls do not need to reproduce identical prose. The original raw responses and selected artifacts are
the historical evidence.

## 14. Versioning

Use semantic intent rather than claiming compatibility from a filename:

- schema migrations have their own ordered versions;
- category definitions have valid/supersession history;
- dataset releases are immutable numbered/tagged snapshots;
- renderer changes create new versions even if outputs “look similar”;
- tokenizer changes always create a new rendered release;
- corrected examples create new candidate versions and release versions;
- aliases such as `latest` are convenience pointers, never scientific citations.

## 15. Public database safety

Before publication:

- construct a new public-safe SQLite snapshot from allowed views;
- exclude credentials, secret handles, private evaluation, reviewer private data, and restricted text;
- include metadata-only pointers where permitted;
- run privacy and rights validation;
- diff table/column coverage against the policy;
- open read-only and execute example queries;
- hash and sign/attest the final artifact where supported.

## 16. Third-party contribution path

External researchers may submit:

- new family blueprints;
- surface realizations;
- reviews or counterreviews;
- open-lens proposals;
- source annotations;
- renderer profiles;
- error reports;
- derived cohort definitions;
- model evaluation outputs.

Contributions receive immutable provenance and do not modify a sealed release. Authority, license, consent, and
review status remain explicit.

## 17. Acceptance criteria

The release system is ready when:

- third parties can build at least the example cohorts without internal knowledge;
- delimiter-independent messages render correctly for two different chat templates;
- every export is deterministic and hash-verified;
- split propagation catches related-family leakage;
- rights and quality policy can exclude records without deleting them;
- raw, positive, negative, contested, and restricted tiers cannot be confused;
- a model checkpoint can cite the exact release and render profile it saw;
- a public-safe database can be built without private-evaluation leakage;
- release cards accurately describe synthetic generation and limitations.
