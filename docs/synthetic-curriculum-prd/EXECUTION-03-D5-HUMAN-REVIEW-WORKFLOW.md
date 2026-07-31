# Execution 03 — D5 human-review workflow and first blinded session

- **Date:** 2026-07-31
- **Contract:** [PRD-12](PRD-12-D5-HUMAN-ADJUDICATION.md)
- **Instrument:** [Appendix D](APPENDIX-D-D5-REVIEW-INSTRUMENT.md)
- **Scope:** implement and prove local append-only Pass A/Pass B review ingestion; prepare the first 12-item
  blinded Pass A session
- **Model calls:** none
- **Training/GPU activity:** none
- **Human judgments recorded:** zero; the session is assigned but not completed

## 1. Outcome

The D5 review contract is now an executable workflow rather than a Markdown-only intention. The corpus CLI can:

- create or recover a human reviewer actor alias;
- register the exact versioned D5 rubric;
- select candidates from both structurally valid and structurally rejected strata;
- produce a randomized Pass A packet containing only model-visible messages and kind;
- resume an open session without duplicating assignments;
- require sealed Pass A completion before any Pass B candidate becomes eligible;
- reveal contract, family, metadata, and structural status only in Pass B;
- validate every outcome, dimension, score, question-policy field, confidence, rationale, and finding;
- fail closed if the assignment, reviewer, rubric, session, pass, status, or candidate-version hash differs;
- store the exact completed form as a content-addressed artifact;
- create append-only human `review`, score, finding, event, and lineage records; and
- leave candidate status, release membership, and training exposure unchanged.

The first session contains 12 assignments under reviewer alias `ajax`. Eleven happen to be structurally valid
and one structurally rejected, but the packet does not reveal that status. All 12 response objects remain blank.
There are still zero human reviews and zero adjudications.

## 2. Implementation

### 2.1 New review module

`packages/corpus/src/review.ts` owns:

- `prepareHumanReviewPacket`;
- `submitHumanReviewPacket`;
- `humanReviewStatus`;
- executable Pass A and Pass B dimension/outcome contracts;
- versioned rubric registration;
- deterministic seeded ordering;
- blinded candidate projection;
- assignment resumption;
- exact submission validation;
- content-addressed form preservation; and
- append-only review/event insertion.

The review module deliberately does not call a model and does not reuse the existing model-critic
`recordReview` function. Human and model authority remain distinguishable in `reviewer_actor_id`,
`reviewer_model_revision_id`, and call provenance.

### 2.2 CLI

The following commands are now available:

```text
review-prepare --reviewer ALIAS [--pass A|B] [--count 12] [--seed VALUE] [--output PATH]
review-submit --file PATH
review-status [--campaign alpha-calibration-v1]
```

`review-prepare` is safe to resume. If the reviewer already has open assignments for that pass and rubric, it
re-exports that same session rather than allocating another cohort.

`review-submit` is intentionally all-or-nothing for a packet. It validates every completed response before
writing the submission artifact and review rows in one database transaction. A second submission of the same
completed assignment fails because the assignment is no longer open.

### 2.3 Public progress reporting

The factual two-hour Discord sender now reports Pass A and Pass B assigned/completed counts plus append-only
human review records. It no longer describes the entire human gate only as a generic audit-packet task.
Structural validity remains explicitly separated from training approval.

## 3. Blinding contract proven

The first packet exposes exactly two candidate fields:

```json
["kind", "messages"]
```

The packet was searched for all of the following and none occurred:

- `familySlug`;
- `structuralStatus`;
- `requiredCommitments`;
- `generatorNotes`;
- `primaryLens`; and
- `hiddenContract`.

The public SQLite ledger necessarily retains assignment-to-candidate lineage. The reviewer instruction therefore
says not to inspect the public candidate or assignment tables before completing Pass A. The packet itself does
not leak those relationships.

## 4. Canonical artifacts

### 4.1 Pre-mutation backup

Before registering the reviewer, rubric, and assignments, SQLite's online backup command created:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/pre-d5-review-prepare-20260731.sqlite
```

SHA-256:

```text
720652a93e6562a2db8b944bc94d5cc763be645879f689278945d66b80c56d39
```

The backup returned `PRAGMA integrity_check = ok` and no foreign-key findings.

### 4.2 First Pass A session

Session:

```text
review_session_1b479c00-3195-4d1f-ac69-86489019cd3e
```

Packet:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/releases/review/
  alpha-calibration-v1-a-review_session_1b479c00-3195-4d1f-ac69-86489019cd3e/
  review-packet.json
```

Human-readable companion:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/releases/review/
  alpha-calibration-v1-a-review_session_1b479c00-3195-4d1f-ac69-86489019cd3e/
  README.md
```

Content-addressed packet SHA-256:

```text
6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48
```

## 5. Current ledger state

| Measure | Count |
|---|---:|
| candidates | 48 |
| structurally valid | 42 |
| structurally rejected | 6 |
| Pass A assigned | 12 |
| Pass A completed | 0 |
| Pass B assigned/completed | 0 / 0 |
| human reviews | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |

The assignment cohort contains 11 valid and one rejected candidate. That composition was measured from the
ledger after packet creation; it is not visible in Pass A.

The canonical ledger validates with:

- integrity `ok`;
- zero foreign-key violations;
- zero missing required tables or views;
- zero missing or corrupt blobs; and
- two hash-verified migrations.

The complete Alpha Corpus tree measured 6.7 MiB including the verified backup, far below the 15 GiB resumable
soft pause.

## 6. Automated tests

The corpus package now passes 12/12 tests. The three new tests prove:

1. Pass A hides contract/family/status fields, includes rejected candidates, and resumes without duplicate
   assignments;
2. completed human forms create append-only human evidence while candidate, release, and training states remain
   unchanged, and only then unlock Pass B; and
3. a changed candidate-version hash causes submission to fail with zero reviews written.

The package TypeScript build also passes.

## 7. Public visibility

The deployed explorer reads the canonical database dynamically. Live server-rendered requests to:

- `https://alpha.donto.org/corpus?relation=review_assignment`; and
- `https://alpha.donto.org/corpus?relation=rubric_version`

returned the new session and rubric records. This is public evidence of assignment, not evidence that a human
has approved any candidate.

## 8. Current gate

The immediate next action is a real human completing the 12 blank Pass A response objects. A model must not fill
them while claiming human authority. After local `review-submit` validates and records that packet, the same
reviewer can prepare Pass B for those sealed candidates. The remaining 36 Pass A candidates follow in later
sessions, with family synthesis only after all individual passes.

No new synthetic generation, critic model, release, training, or GPU work is justified merely because the
review machinery now works.
