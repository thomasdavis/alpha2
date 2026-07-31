# Execution 15 — D5 review evidence completeness and rubric supersession

**Executed:** 2026-07-31

**Code revision:** `2c6da67121938c8c033e7d0e30ddaefd0f0949ba`

**Public routes:**

- `https://alpha.donto.org/corpus`;
- `https://alpha.donto.org/corpus/review`;
- `https://alpha.donto.org/corpus/review/review_session_b968779b-4959-47a1-ba5a-9d64cb466f3e`;
- `https://alpha.donto.org/corpus?relation=review_comprehension_assessment`;
- `https://alpha.donto.org/corpus?relation=review_dimension_evidence`;
- `https://alpha.donto.org/corpus?relation=review_finding_explanation`; and
- `https://alpha.donto.org/corpus?relation=review_assignment_supersession`.

**Authority boundary:** review-instrument implementation and blank-packet export only. This execution created
no human judgment, model call, synthetic generation, adjudication, release member, training exposure, GPU use,
or execution authorization.

## 1. Outcome

The D5 Pass A browser worksheet, JSON packet, validator, importer, downstream evidence packets, and SQLite
ledger now capture every field required by Appendix D sections 2.1–2.6.

The preceding instrument could safely bind a response to an immutable packet and preserve reviewer-session
conditions, but it still under-recorded four authority-bearing judgments:

1. whether the first assistant sentence directly engaged the user's move;
2. whether the assistant answered before asking anything unnecessary;
3. the reviewer's one-sentence evidence for every dimension assessment; and
4. why each finding mattered and what a repair must preserve.

The audit also found that Appendix D permits `not_applicable` and `uncertain` dimension assessments, while the
deployed form forced a numeric 0–4 score. That could have manufactured false precision. Rubric v2 now preserves
these states explicitly and does not coerce them into numeric score rows.

These are changes to the review instrument, not cosmetic UI additions. The executable rubric therefore moved
from v1 to v2. Twelve unfilled v1 assignments were preserved, explicitly superseded, and replaced by twelve v2
assignments over the exact same candidate-content population.

## 2. Field-by-field audit

### 2.1 Pass A immediate comprehension

| Appendix D requirement | Pre-execution state | Rubric v2 state |
|---|---|---|
| one-sentence user aim | packet + UI + rationale JSON | unchanged |
| one-sentence assistant move | packet + UI + rationale JSON | unchanged |
| first sentence directly engages user: yes/partly/no | absent | required packet field, UI selector, validator, normalized table |
| answer before unnecessary question: yes/no/not applicable | absent | required packet field, UI selector, validator, normalized table |

### 2.2 Outcome and dimension assessments

The eight Pass A dimensions remain:

1. direct responsiveness;
2. conceptual plausibility;
3. linguistic naturalness;
4. conversational naturalness;
5. appropriate depth and length;
6. pedagogical value;
7. desire to continue; and
8. substantive value after style removal.

Each dimension now requires both:

- one assessment: integer 0–4, `not_applicable`, or `uncertain`; and
- one non-empty sentence of model-visible evidence.

`not_applicable` and `uncertain` are stored as explicit assessment states. They do not create a row in
`review_dimension_score`. Numeric judgments create both a numeric score row and an evidence-state row.

### 2.3 Question policy

The existing follow-up-question and missing-clarification judgments were already present and remain required.
They are independent of the two new immediate-comprehension fields: a response can answer first yet still end
with a ritual question, or require clarification before any responsible answer.

### 2.4 Findings

Every optional finding now requires all six Appendix D fields:

| Field | Storage |
|---|---|
| dimension | `review_finding` / `review_presentation_finding` |
| severity | same |
| exact evidence | same |
| why it matters | `review_finding_explanation` |
| smallest plausible repair | existing `recommendation` column, relabeled in the UI |
| what must be preserved | `review_finding_explanation` |

An incomplete finding prevents submission. Findings remain optional because a reviewer may legitimately have
no discrete defect beyond the required disposition, rationale, and per-dimension evidence.

### 2.5 Confidence, uncertainty, and expertise

The existing 0–4 reviewer confidence, uncertainty/admissible-alternatives note, and expertise/authority-needed
field remain unchanged. Session competence, interruption, fatigue, start/end, and conditions remain governed
by Execution 14.

## 3. Executable contract changes

`HumanReviewResponse` now contains:

```text
firstSentenceEngagement
answeredBeforeUnnecessaryQuestion
dimensionEvidence
```

`HumanReviewFinding` now additionally contains:

```text
whyItMatters
preserve
```

The browser and local importer share the same validator. A completed packet cannot be downloaded or imported
unless:

- every required response field is present;
- score and evidence maps contain exactly the dimensions in rubric v2;
- each dimension has a valid numeric or explicit non-numeric state;
- every dimension has non-empty evidence;
- Pass A comprehension selectors are complete;
- every finding contains all six fields; and
- the Execution-14 reviewer/session declaration is complete.

The immutable-envelope rule remains unchanged: only typed response fields may differ from an exact packet
previously exported for that session and pass.

## 4. First-class SQLite representation

Migration 9, `d5_review_evidence_completeness`, adds three append-only tables.

### 4.1 `review_comprehension_assessment`

Stores the two immediate-comprehension judgments. Exactly one of `review_id` and
`presentation_response_id` must be present. The presentation target is used for a hidden repeat, which must
remain a presentation response rather than inflating the candidate-review count.

### 4.2 `review_dimension_evidence`

Stores one row per rubric dimension with:

- a review or hidden-repeat presentation target;
- the exact dimension key;
- `assessment_state` = `score`, `not_applicable`, or `uncertain`; and
- the reviewer's evidence sentence.

Numeric values remain in `review_dimension_score` or `review_presentation_score`. This preserves the semantic
difference between “zero,” “not applicable,” “uncertain,” and “not answered.”

The `review_repeat_stability` view was recompiled so exact-rate comparison includes both numeric and explicit
non-numeric states. Mean absolute score difference remains defined only over dimensions that are numeric in
both presentations.

### 4.3 `review_finding_explanation`

Stores `why_it_matters` and `preserve` for either a primary `review_finding` or a hidden-repeat
`review_presentation_finding`. Exactly one target must be present.

### 4.4 Downstream propagation

Pass C family-synthesis packets and Pass D closeout packets now receive:

- the complete dimension state;
- its numeric value when applicable;
- the evidence sentence; and
- the complete finding repair contract.

This prevents later synthesis from seeing a score or short recommendation after the reviewer's evidence and
preservation constraint have been discarded.

All three tables reject update and delete operations through append-only triggers.

## 5. Rubric versioning and supersession

The executable rubric definition hash changed, so reusing rubric version 1 correctly failed closed with:

```text
Stored D5 human-review rubric differs from the executable definition
```

That failure wrote nothing. The resolution was not to weaken the hash check. Rubric version 2 was registered,
and migration 10 added `review_assignment_supersession`.

The supersession relation records:

- the preserved prior assignment;
- its replacement assignment;
- prior and replacement rubric-version identities;
- the exact reason; and
- the timestamp.

The prior assignment row remains present with status `superseded`; it is not deleted or repointed. The
replacement is a new v2 assignment with status `assigned`.

### 5.1 Population continuity proof

The sorted candidate-content-hash multiset for the final v1 packet and the v2 packet has the same SHA-256:

```text
e89ee9cbb0c9af9fdb96f14825ca0ccc209a605578174a87c09b971d59772496
```

Therefore rubric supersession did not resample the review population. Assignment IDs, opaque IDs, session ID,
and packet hash changed because they bind a new rubric and presentation session; the twelve candidate-content
hashes did not.

Current assignment state:

| Pass | Status | Count |
|---|---|---:|
| A | `superseded` under rubric v1 | 12 |
| A | `assigned` under rubric v2 | 12 |

`review_assignment_supersession` contains 12 rows.

### 5.2 Packet continuity

Preserved v1 packet blobs remain readable:

- `6d2fc108130f9918056ff44405725f9cf72d8a0e9a0b0b5636719d154687d708`;
- `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48`; and
- `95b962709e9ad77aa91f2249f0648f1ee026b5ce3d64aaff792b615f751a484a`.

The current blank rubric-v2 packet is:

```text
session: review_session_b968779b-4959-47a1-ba5a-9d64cb466f3e
SHA-256: 8c6a99c8c4dc1d74ceca0e75eb1767bb3229b9c9f2529c26d800b58e62b66f92
```

It contains twelve null outcomes, eight blank evidence fields per assignment, null comprehension judgments,
and a blank reviewer/session declaration. Exporting it created no human evidence.

The reader accepts preserved rubric versions from 1 through the current version. Missing additive fields in a
legacy packet are normalized to blank, explicitly incomplete values in memory; historical blob bytes and
hashes are not rewritten.

## 6. Backup and migration evidence

Before migration 9, the canonical database was copied to:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/
pre-d5-review-evidence-completeness-20260731T063755Z.sqlite
```

It is 6,152,192 bytes, passes `PRAGMA integrity_check`, contains eight migrations, and has SHA-256:

```text
d2ccec649ab4aaeb0aac427391de0366a2b16b852477b2aa0da1724dd2ce9d19
```

Before migration 10 and rubric-v2 assignment creation, an online SQLite backup captured the complete
nine-migration state at:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/
pre-d5-review-rubric-v2-20260731T064247Z.sqlite
```

It is 6,217,728 bytes, passes integrity and foreign-key checks, and has SHA-256:

```text
f424134a5e5212ef39d3dbe42b9cc8806c5237d2c0ffcb1236eb4b7c2a5ec3e6
```

Migration evidence:

| Version | Name | SHA-256 |
|---:|---|---|
| 9 | `d5_review_evidence_completeness` | `ec31971eb9a98b49d28d4b1fac5df4cf1b246d0c9c777a3a529b39873386e062` |
| 10 | `d5_review_rubric_supersession` | `fc7407993eba90639bc42035213a5063075fc88ef9bd9ac3d3a6b15928b2b350` |

After a successful WAL checkpoint, the canonical SQLite file is 6,254,592 bytes and has SHA-256:

```text
0695bbf651d74c227931016fbe14e617337872d05ebf494f33aa264973dd327b
```

Validation reports:

- `integrity: ok`;
- zero foreign-key violations;
- zero missing tables or views;
- zero missing or corrupt blobs; and
- ten migrations.

The live schema contains 135 tables, five views, 198 append-only triggers, and 24 indexes.

## 7. Adversarial and static verification

| Check | Result |
|---|---|
| corpus TypeScript build | pass |
| focused corpus tests | 26/26 pass, 0 fail |
| repository typecheck | pass |
| optimized web build | pass |
| commit hook's full `@alpha/web...` dependency build | pass |
| blank dimension evidence | submission rejected; zero declaration, review, evidence, or raw-submission writes |
| `uncertain` dimension | explicit state retained; zero coerced numeric rows |
| v1→v2 open assignment replacement | old rows retained as superseded; exact replacement links recorded |
| evidence-table update/delete | rejected by append-only triggers |
| legacy v1 packet | readable as explicitly incomplete |

The optimized build retains pre-existing Turbopack NFT warnings about broad filesystem tracing through
`server-state.ts`. Compilation, type checking, page collection, and route generation completed successfully.

## 8. Public browser proof

### 8.1 Exact review session

Real Chromium against the canary and final public deployment confirmed:

- one page-level `main` landmark;
- twelve assignments;
- two new Pass A immediate-comprehension selectors;
- numeric 0–4, `N/A`, and `?` controls for every dimension;
- eight evidence text areas;
- all four finding text fields after “Add finding”;
- a disabled completed-download action while fields remain blank;
- locally persisted comprehension, explicit uncertainty, evidence, and finding state;
- opaque assignment identity preserved after navigation and reload;
- no application-error overlay;
- no browser console or page errors; and
- no horizontal overflow at 390 px.

The mobile screenshot is retained at:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/outputs/
execution15-review-mobile.png
```

SHA-256:

```text
003681f4bb3c9d05da985b6468504a9cf50d172ac181720bc8f825510047a383
```

### 8.2 Public explorer

The three evidence tables render with zero rows, which is the truthful pre-review state. The supersession table
renders twelve rows and exposes its recorded reason. Every relation has one `main` landmark and no mutation
surface.

### 8.3 HTTP boundary

Public GET returned 200 for `/corpus`, `/corpus/review`, the exact v2 session, and all four new relations.
Public POST returned 405 for `/corpus`, `/corpus/review`, and the exact session.

## 9. Immutable release and service state

The active immutable release is:

```text
/home/ajax/alpha2-web-releases/2c6da67121938c8c033e7d0e30ddaefd0f0949ba
```

It contains 1,994 files and occupies approximately 61 MiB. Every manifest entry passed `sha256sum -c` before
activation. The manifest file has SHA-256:

```text
a3bdf07a655629c7d7ab1b0175e7c97f916b6d32acd00ed0c94f3da6e734f302
```

`alpha-corpus-web.service` is active with zero automatic restarts. The two-hour factual Discord timer remains
enabled.

## 10. Scientific state after execution

| Relation or state | Count |
|---|---:|
| candidates | 48 |
| structurally valid | 42 |
| structurally rejected | 6 |
| v1 assignments, superseded | 12 |
| v2 assignments, open | 12 |
| assignment supersessions | 12 |
| human session declarations | 0 |
| declared competence rows | 0 |
| human reviews | 0 |
| presentation responses | 0 |
| comprehension assessments | 0 |
| dimension-evidence rows | 0 |
| finding explanations | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |
| execution authorizations | 0 |

The project-owned Alpha Corpus artifact tree is 55,304,070 bytes, far below the 15 GiB pause threshold.

## 11. Remaining instrument boundary

This execution closes Appendix D Pass A. It does **not** claim that the generic Pass B score sheet fully
materializes Appendix D sections 3.1–3.4. Those sections require first-class, contract-indexed matrices for:

- blueprint validity questions;
- every required and prohibited commitment;
- every preserve/change instruction;
- plurality and evidence boundaries; and
- metadata-fit judgments.

Pass B remains locked and no contract-aware packet exists. Before Pass B can legitimately be revealed, the
program must either:

1. implement a separately versioned Pass B worksheet that can cite sealed Pass A rubric-v2 evidence without
   invalidating it; or
2. explicitly revise Appendix D with a reasoned equivalence proof.

The first option is preferred. A shared A/B rubric version currently couples their prerequisites; blindly
raising the shared version after Pass A would strand sealed v2 evidence. This versioning seam must be resolved
before Pass B, not after human labor has begun.

## 12. Immediate authority-bearing next action

The open human action remains real Pass A review at the v2 session URL. Implementation work may continue on a
separately versioned Pass B worksheet, public observability, and tests, but no model generation, critic call,
release promotion, training run, or GPU use is authorized by this execution.

