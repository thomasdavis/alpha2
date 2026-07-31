# Execution 07 — D5 blinded repeat presentations and reviewer-stability evidence

**Executed:** 2026-07-31

**Code revision:** `249a00f85f19d3d319bd878e0e10a8f4c26c5c89`

**Campaign:** `alpha-calibration-v1`

**Result:** the six-repeat Pass A protocol is now executable without creating duplicate candidates or duplicate
scientific reviews; the canonical repeat tables remain empty because no real Pass A response exists yet

**Authority boundary:** code, tests, ledger migration, public read-only exposure, progress reporting, and
documentation only. No human response, model call, candidate promotion, corpus expansion, release membership,
training exposure, GPU use, or live Donto mutation was created.

## 1. Outcome

PRD-12 requires six hidden repeat presentations across Pass A sessions to measure within-reviewer stability.
The original review implementation could not represent them correctly. A review assignment was identified by
candidate, reviewer, rubric, and pass, so duplicating the assignment would either violate identity or inflate
the apparent number of independently reviewed candidates.

This execution adds a separate presentation layer:

- a **review assignment** remains one candidate-level scientific review;
- a **presentation** is one occasion on which that assignment is shown in a session;
- a **primary presentation response** creates the candidate's review;
- a **hidden-repeat response** is preserved separately and compared with the original review;
- a repeated presentation never creates another candidate, assignment, or candidate review.

That separation makes the denominator honest. If two candidates are each reviewed once and one is later shown
again, the ledger contains two candidate reviews and three presentation responses—not three candidate
reviews.

## 2. Migration 6

Migration `6`, `d5_blinded_repeat_presentations`, has digest:

`3410f82088409e139be237a4f4c9935e0f01acce1a8de25e3c563acbf4cae561`

It adds five tables and one view.

| Relation | Purpose | Mutation policy |
|---|---|---|
| `review_presentation_session` | one ordered, hash-bound Pass A or B session | status may move assigned → completed |
| `review_presentation` | one primary or hidden-repeat appearance of an assignment | status may move assigned → completed |
| `review_presentation_response` | canonical structured response for that appearance | append-only |
| `review_presentation_score` | one dimension score per presentation response | append-only |
| `review_presentation_finding` | exact evidence/recommendation findings | append-only |
| `review_repeat_stability` | derived comparison between a repeat response and its source review | read-only view |

Six update/delete rejection triggers protect the three immutable response tables. Sessions and presentations
are mutable only for their bounded workflow status.

The migrated canonical ledger contains:

| Object | Count |
|---|---:|
| schema migrations | 6 |
| user tables | 122 |
| views | 5 |
| triggers | 174 |
| presentation sessions | 0 |
| presentations | 0 |
| presentation responses | 0 |
| repeat-stability rows | 0 |
| candidate reviews | 0 |
| release members | 0 |
| training exposures | 0 |

The zero presentation population is correct. The current 12-item packet predates this schema and remains open;
the workflow does not rewrite it or invent a first presentation after the fact.

## 3. Backup and migration evidence

Before applying migration 6, SQLite's online backup command created:

`backups/pre-d5-repeat-presentations-20260731T025206Z.sqlite`

Its SHA-256 is:

`c914d560405642ce641570ed8794172ec3a36edc2948ed3d27bda800059b2918`

The backup returned `PRAGMA integrity_check = ok`.

After migration, canonical validation returned:

- integrity: `ok`;
- foreign-key violations: `0`;
- missing required tables: `0`;
- missing required views: `0`;
- missing blobs: `0`;
- corrupt blobs: `0`;
- migration count: `6`;
- project-owned footprint: `29.68 MiB`.

The project tree remains far below the 15 GiB resumable-pause threshold.

## 4. Legacy packet compatibility

The first real Pass A session was prepared before the presentation layer existed. It remains at:

`releases/review/alpha-calibration-v1-a-review_session_1b479c00-3195-4d1f-ac69-86489019cd3e/review-packet.json`

The file SHA-256 remains:

`6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48`

The latest `human_review_packet_json` export row names the same digest and session. No migration or preparation
command rewrote those bytes. The local submitter continues to accept this packet's original schema and will
create ordinary candidate reviews when a real human completes it.

After that session is sealed, future Pass A sessions use first-class presentations automatically. There is no
need to convert the existing packet or ask the reviewer to start again.

## 5. Repeat scheduling policy

The executable scheduler targets six distinct hidden repeats for the campaign/reviewer/rubric combination.
For a normal later Pass A session:

1. select only prior assignments that have a completed human Pass A review;
2. exclude any assignment already used as a hidden repeat;
3. schedule at most two repeats in one session while new primary candidates remain;
4. preserve at least one new primary presentation when the requested session size permits;
5. fill the remaining session slots with new candidate assignments;
6. seed and interleave both kinds under one deterministic order;
7. give every presentation a fresh opaque item ID.

With the current intended 12-presentation sessions, the normal trajectory after the legacy first session is
ten new candidates plus two repeats across three sessions, followed by the remaining candidates. This yields
six distinct repeat cases without increasing the 48-candidate denominator.

If a nonstandard session size makes that schedule impossible, the ledger records what was actually assigned;
the status report does not claim `6/6` until six repeat responses exist.

## 6. What the reviewer sees

The JSON/browser packet includes:

- the normal assignment ID;
- a unique opaque presentation ID used only for submission integrity;
- a new opaque item label;
- the same Pass A model-visible `kind` and `messages` fields;
- the normal blank rubric response.

It does not include:

- `presentation_kind`;
- `hidden_repeat`;
- source assignment or source review ID;
- family, candidate identity, structural status, contract, lineage, or earlier response.

The test suite asserts that a mixed primary/repeat packet contains neither the string `hidden_repeat` nor a
`sourceReviewId` field. The instructions disclose that later sessions may contain consistency checks but do
not identify any item. Every presentation must be reviewed independently.

## 7. Submission semantics

Submission verifies:

- packet, rubric, reviewer, pass, and session identity;
- exact candidate content hash;
- presentation status and complete session membership;
- primary presentations point to open assignments;
- repeat presentations point to completed assignments and a frozen source review;
- no packet mixes legacy and first-class presentation modes;
- every response satisfies the same Pass A rubric.

For a primary presentation, one transaction creates:

- the normal candidate `review`;
- normal review dimension scores and findings;
- a presentation response with the same canonical response;
- presentation scores and findings;
- assignment and presentation completion events.

For a hidden repeat, the same transaction creates:

- a presentation response;
- presentation dimension scores and findings;
- an explicit `human_review_repeat_submitted` event linked to the source review and candidate version;
- no new candidate review and no assignment completion change.

The raw submitted packet remains a content-addressed artifact in both cases. The session is marked complete
only when no assigned presentation remains.

## 8. Stability measurements

`review_repeat_stability` derives one row per completed hidden repeat. It reports:

- exact outcome match;
- follow-up-question policy match;
- missing-clarification judgment match;
- absolute confidence delta;
- exact-match rate across rubric dimensions;
- mean absolute dimension-score delta.

These are descriptive reliability measurements, not proof that either answer is correct. A perfectly stable
reviewer may be consistently wrong. A thoughtful reviewer may legitimately change a judgment after noticing a
new feature. D5 must report both consistency and the substantive rationale.

Repeat responses are not shown to Pass C as additional reviews. Pass C still requires exactly one sealed A
and one sealed B candidate review. This prevents stability observations from silently adding weight to one
candidate or family.

## 9. Executable proof

The new end-to-end test performs this sequence:

1. prepare and complete one primary Pass A presentation;
2. prepare a second packet containing one new primary and one hidden repeat;
3. prove the packet hides repeat identity and source lineage;
4. submit both responses;
5. verify two candidate assignments and two candidate reviews, not three;
6. verify three presentation responses;
7. verify one derived stability row with exact agreement for the controlled fixture;
8. verify zero release members and zero training exposures;
9. prove presentation responses reject update;
10. prove the completed packet cannot be resubmitted.

The full corpus package passes 18/18 tests:

```bash
npm run build -w @alpha/corpus
npm test -w @alpha/corpus
```

## 10. Public evidence

The public explorer automatically discovered the migration. On 2026-07-31 each route returned HTTP 200:

- `https://alpha.donto.org/corpus?relation=review_presentation_session`
- `https://alpha.donto.org/corpus?relation=review_presentation`
- `https://alpha.donto.org/corpus?relation=review_presentation_response`
- `https://alpha.donto.org/corpus?relation=review_presentation_score`
- `https://alpha.donto.org/corpus?relation=review_presentation_finding`
- `https://alpha.donto.org/corpus?relation=review_repeat_stability`

The two-hour factual progress report now reports repeat presentations completed out of six, currently
assigned repeats, and available stability rows. It explicitly says repeats do not inflate candidate/review
counts.

## 11. Blindness limitation under a fully public ledger

The packet and review workspace hide repeat identity, but the canonical ledger is intentionally public and
the explorer exposes every table. A reviewer who inspects `review_presentation` during a session could discover
which item is a repeat. For the present operator-led calibration, blindness is therefore a protocol:

- do not inspect candidate or presentation lineage before sealing the packet;
- use the local-first review route, not the general relation explorer;
- preserve any accidental unblinding as a session limitation.

A later multi-reviewer benchmark that needs adversarially strong blinding should keep the repeat mapping in an
embargoed or separately encrypted evaluator artifact until submissions are sealed, then publish it with the
completed scientific record. That would be a new privacy/publication design; this execution does not weaken
the user's requirement that the present corpus tables remain publicly inspectable.

## 12. Remaining D5 gate

The protocol implementation gap is closed. The authority gate is not.

The immediate next action remains a real human completion of the existing 12-item Pass A packet. Only after
that response is sealed can the scheduler legitimately choose its first repeat. Pass B, Pass C, the six actual
stability observations, and Pass D campaign adjudication remain empty future evidence.

No semantic-quality, conversation-quality, reviewer-reliability, or training-usefulness claim follows from
this implementation alone.
