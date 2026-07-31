# Execution 06 — D5 family synthesis and structural-disposition workflow

**Executed:** 2026-07-31

**Code revision:** `66783ae7839667ae7c284d5c645a9a151fee1356`

**Campaign:** `alpha-calibration-v1`

**Result:** the Pass C and structural-rejection evidence path is implemented, migrated, tested, and publicly
browseable; it remains correctly empty because the human Pass A and Pass B prerequisites are incomplete

**Authority boundary:** implementation, ledger migration, deterministic verification, documentation, and
public read-only exposure only. No human judgment, model call, candidate promotion, release membership,
training exposure, GPU use, or live Donto mutation was created.

## 1. Outcome

D5 previously specified a family-level comparison after candidate-level review, but the executable review
system stopped at Pass A and Pass B. This execution closes that implementation gap without pretending to
close the human gate.

The corpus package now supports a third, family-level evidence phase:

1. require exactly one sealed human Pass A review and one sealed human Pass B review for every current
   candidate assigned to the same reviewer;
2. freeze the current family, candidate, structural-failure, and A/B review evidence into a hash-addressed
   Pass C packet;
3. collect one family synthesis for each eight-sibling family;
4. collect a separate semantic and schema disposition for every structurally rejected candidate;
5. verify that the submitted packet still corresponds byte-for-byte to immutable ledger evidence;
6. preserve the raw submission and every basis relation append-only;
7. leave candidate lifecycle, release membership, and training exposure untouched.

The result is an executable distinction between four things that must never be collapsed:

- a candidate passed deterministic structural validation;
- a human judged the candidate in isolation;
- a human compared the candidate with its siblings and diagnosed family-level behavior;
- an operator later authorized a repair, release, or experiment.

Only the machinery for the first three now exists. The second and third contain no completed human evidence
yet, and the fourth remains a future explicit decision.

## 2. Why Pass C needs first-class records

The scientific unit of this calibration is the concept family, not the rendered row. Eight individually
plausible candidates can still be a poor family if they paraphrase one another, share an error, omit a hard
negative, overuse one conversational policy, or teach a misleading abstraction. Candidate-level scores
cannot express those failures reliably.

Pass C therefore records:

- whether positive, negative, borderline, genuinely plural, transfer, and false-bridge pressures are
  covered;
- the strongest and weakest sibling and why;
- semantic duplicate groups, including a valid finding of no meaningful duplicates;
- shared conceptual errors, shared style or teacher signatures, response-policy imbalance, and metadata
  mismatch;
- the highest-leverage blueprint repair;
- missing hard negatives;
- unresolved uncertainty;
- a family-level disposition and confidence.

The workflow does not infer these judgments from n-grams or ask a model to manufacture them. The first-class
surface evidence in Execution 05 may guide later analysis, but Pass C is sealed human evidence grounded in
the actual family and the earlier human reviews.

## 3. Structural-rejection disposition

The current calibration retained six candidates rejected for `unknown_secondary_lens`. A deterministic
validator rejection answers only whether the declared metadata conformed to the current schema. It does not
answer whether the model-visible conversation is useful, whether the validator correctly identified the
problem, or whether the taxonomy itself is incomplete.

For each rejected sibling, Pass C therefore requires a distinct structural disposition containing:

- content utility;
- validator-finding correctness;
- the exact unknown or disputed value;
- its proposed semantic type;
- the appropriate remedy;
- the hazard of automatic acceptance;
- the hazard of automatic rejection;
- rationale and confidence.

This is deliberately not a repair-in-place. A candidate can be conceptually valuable while remaining
structurally rejected, or structurally well formed while conceptually poor. The ledger keeps both facts.

## 4. Migration 5

Migration `5`, `d5_family_synthesis_and_structural_disposition`, has digest:

`28ec75187d2dccc4518b5c43ab73a2108abf6e7d23878b7007936ef4e78baffd`

It adds five tables.

| Table | Purpose | Mutation policy |
|---|---|---|
| `family_synthesis_assignment` | reviewer/session/input-snapshot gate for one family | status may move from assigned to completed |
| `family_synthesis` | one sealed family judgment per assignment | append-only |
| `family_synthesis_basis` | exact candidate review records supporting that synthesis | append-only |
| `structural_disposition` | separate semantic/schema judgment for one rejected sibling | append-only |
| `structural_disposition_basis` | exact validator failures and reviews supporting that disposition | append-only |

Eight new rejection triggers protect the four immutable evidence tables from update and delete. Assignment
status is intentionally mutable because it represents a workflow lease rather than scientific evidence.

After migration, the canonical ledger contains:

| Object | Count |
|---|---:|
| schema migrations | 5 |
| user tables | 117 |
| views | 4 |
| triggers | 168 |
| family-synthesis assignments | 0 |
| family syntheses | 0 |
| family-synthesis basis rows | 0 |
| structural dispositions | 0 |
| structural-disposition basis rows | 0 |

The zeroes are the correct state. They show that implementation did not masquerade as human completion.

## 5. Backup and migration evidence

Before the migration, SQLite's online backup command created:

`backups/pre-d5-family-synthesis-20260731T023557Z.sqlite`

Its SHA-256 is:

`c4ae7e7b0720fd4c69f97b3331d53946e4b8dfbc9cc15869f2d0af959c58450e`

The backup returned `PRAGMA integrity_check = ok` before the live ledger was opened with the new code.

The migrated canonical ledger then returned:

- integrity: `ok`;
- foreign-key violations: `0`;
- missing required tables: `0`;
- missing required views: `0`;
- missing blobs: `0`;
- corrupt blobs: `0`;
- migration count: `5`;
- project-owned corpus tree: `24.03 MiB` at verification time.

The project-owned tree remains far below the 15 GiB resumable-pause threshold. Host disk pressure is a
separate operational concern and does not change the project threshold.

## 6. Packet preparation contract

The command is:

```bash
npm run corpus -- synthesis-prepare --reviewer REVIEWER_ALIAS
```

Preparation proceeds in this order:

1. resolve the campaign and human actor;
2. resolve the exact versioned Pass A/B rubric;
3. enumerate the current candidate version in every family;
4. require exactly one completed A review and one completed B review for every candidate;
5. load dimension scores and findings for those exact reviews;
6. load structural failures for each current candidate;
7. compute a canonical evidence payload and SHA-256 for every family;
8. compute the campaign packet snapshot from those family payloads;
9. create one assignment per family only after all prerequisites pass;
10. export a JSON packet and human-readable Markdown companion.

The order matters. The workflow loads and validates all prerequisite evidence before it creates a Pass C
rubric or assignment. A partial Pass A or Pass B state cannot leave misleading Pass C assignments behind.

Packet creation is resumable but not overwrite-friendly. If the expected content-addressed path already
exists, byte-identical content is reused. If a person edited the existing file, preparation refuses to
replace it. Human responses therefore become a distinct submission artifact rather than an invisible change
to the frozen evidence packet.

## 7. Current fail-closed proof

The canonical command was run after migration:

```text
npm run corpus -- synthesis-prepare --reviewer ajax
Error: Candidate candidatev_ce14fa164b51a123f86ce84085063e94 needs exactly one sealed Pass A and Pass B review before Pass C
```

After the expected refusal:

- `family_synthesis_assignment = 0`;
- `family_synthesis = 0`;
- `family_synthesis_basis = 0`;
- `structural_disposition = 0`;
- `structural_disposition_basis = 0`;
- `release_member = 0`;
- `training_exposure = 0`.

This is the key no-promotion proof for the current live data. The workflow is available, but human authority
has not been simulated.

## 8. Submission contract

When A and B are eventually complete, the human reviewer will fill the response objects in the frozen Pass C
packet and submit locally:

```bash
npm run corpus -- synthesis-submit --file /absolute/path/to/completed-packet.json
```

Submission verifies all of the following before a write transaction begins:

- packet format and rubric identity;
- human actor and reviewer alias;
- session and assignment identity;
- assigned rather than completed status;
- overall and per-family input snapshot hashes;
- exact family version, blueprint, current candidate version, content hash, status, structural failures,
  and sealed A/B review evidence;
- exactly one response for each assigned family;
- strongest and weakest candidates belong to that family;
- every required coverage pressure is answered exactly once;
- every semantic duplicate group contains only siblings;
- all diagnostic fields are substantive, with explicit `none observed` permitted;
- exactly one structural disposition for each rejected sibling and none for accepted siblings;
- all controlled disposition values and confidence ranges are valid.

A blank form is not a neutral answer and is rejected. A modified evidence section is rejected. A resubmission
to a completed assignment is rejected.

Within one write transaction, a valid submission stores:

- the raw submission as a content-addressed blob and artifact;
- one `family_synthesis` per family;
- two `family_synthesis_basis` rows per candidate, one for each sealed A/B review;
- one `structural_disposition` per rejected sibling;
- basis rows for every relevant validator failure and review;
- completion events and assignment status changes.

No statement in this transaction updates a candidate, adds a release member, renders a training unit, or
creates a training exposure.

## 9. Executable tests

The corpus package has 17 passing tests. The new tests prove:

1. Pass C preparation fails before every candidate has sealed A and B reviews;
2. the failed preparation creates zero synthesis assignments;
3. a complete synthetic test fixture can prepare and resume a packet deterministically;
4. modified frozen evidence is rejected;
5. blank human responses are rejected;
6. a valid family synthesis records the expected family and basis rows;
7. a rejected sibling receives a separate structural disposition and exact basis rows;
8. no release membership or training exposure is created;
9. update and delete against immutable evidence tables are rejected;
10. a completed assignment cannot be submitted again.

Verification commands:

```bash
npm run build -w @alpha/corpus
npm test -w @alpha/corpus
git diff --check
```

The build passed, all 17 tests passed, and the diff check passed before the implementation commit.

## 10. Public evidence

The read-only explorer discovers SQLite relations dynamically, so no web-server redeploy was needed. On
2026-07-31 all of the following returned HTTP 200:

- `https://alpha.donto.org/corpus?relation=family_synthesis_assignment`
- `https://alpha.donto.org/corpus?relation=family_synthesis`
- `https://alpha.donto.org/corpus?relation=structural_disposition`
- `https://alpha.donto.org/corpus/review`

The corpus web service was active with zero restarts. Public access remains read-only; the local CLI is the
only write path.

## 11. Progress reporting

The two-hour factual Discord reporter now includes:

- assigned and completed Pass C counts;
- family-synthesis count;
- structural dispositions completed out of the six retained rejections;
- an explicit statement that Pass C cannot open before every current candidate has one sealed human A and B
  review.

A dry run reported the truthful current state: Pass A `0/48`, Pass B `0/48`, Pass C `0`, family syntheses
`0`, and structural dispositions `0/6`. It also stated that generation and GPU training were inactive.

## 12. Remaining D5 gaps

This execution does not finish D5. The next authority-bearing action is still human Pass A review through the
local-first review workspace. After all A reviews are sealed, Pass B may open; only after all B reviews are
sealed may Pass C open.

At this checkpoint, one protocol feature remained unimplemented: PRD-12's six hidden repeat presentations
across Pass A sessions. [Execution 07](EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md) subsequently closed
that implementation gap with a separate presentation-event layer rather than duplicate scientific reviews.
Its contract preserves:

- a single candidate denominator;
- hidden repeat identity during review;
- presentation order and session provenance;
- a stability measurement separate from candidate outcome;
- no leakage from repeat scheduling into Pass A content.

Pass D campaign adjudication still remains a specification rather than an executable workflow. It must not be
implemented by treating family synthesis as automatic authorization.

## 13. Honest current conclusion

The project now has an auditable path from blind candidate review to contract-aware review to family-level
synthesis and structural-taxonomy diagnosis. It does not yet have the human evidence needed to use that path.

No claim about semantic quality, conversational quality, blueprint quality, or training usefulness follows
from this implementation. The only justified claim is operational: the ledger can now preserve the required
D5 family evidence without allowing incomplete human review, structural validity, or a polished generated
answer to silently become training approval.
