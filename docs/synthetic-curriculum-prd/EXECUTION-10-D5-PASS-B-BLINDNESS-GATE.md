# Execution 10 — D5 campaign-wide Pass B blindness gate

**Date:** 2026-07-31

**Status:** implemented, tested, live CLI proof passed, and pushed; human review has not begun

**Scope:** prevent contract-aware Pass B material from being exposed until the complete blind Pass A census and
hidden-repeat stability evidence are sealed for the same campaign, reviewer, and rubric

**Feature revision:** `b52792b4e0821852e500466be7f0640cf6f60b67`

## 1. Defect found

The public dashboard and PRD-12 correctly described this stage order:

```text
Pass A primaries -> hidden repeats -> Pass B -> Pass C and structural dispositions -> Pass D
```

The local packet preparer did not fully enforce that contract. Its Pass B candidate query required only a
completed Pass A assignment for the same candidate. After importing the first 12-item Pass A session, an
operator could therefore have prepared Pass B for those 12 candidates while the remaining 36 blind primaries
and all six hidden repeats were unfinished.

That would have exposed:

- concept-family identity;
- structural status;
- title and generator metadata;
- intended response policy;
- lens and transformation metadata; and
- the hidden required, prohibited, preserved, changed, and admissible commitments.

Even if the operator intended to ignore those fields later, the remaining Pass A judgments would no longer be
cleanly blind. Hidden-repeat stability could also be contaminated by remembered contracts. The resulting human
evidence would not support the D5 claims.

This was a real executable mismatch, not merely missing documentation. The individual-candidate gate was
necessary but insufficient.

## 2. Corrected gate

`prepareHumanReviewPacket(... pass: "B")` now checks campaign-wide prerequisites before it looks for an open
Pass B session or selects any Pass B candidate.

For the exact campaign, human actor, and rubric version, it requires:

1. at least one current candidate;
2. every current candidate version has one completed Pass A assignment;
3. that assignment has exactly one matching human review;
4. the review rationale binds the same assignment and Pass A;
5. all required hidden-repeat stability rows exist;
6. the repeat target is `min(6, current candidate count)` so small fixtures remain executable;
7. every counted repeat belongs to a completed Pass A presentation and session; and
8. no first-class Pass A presentation session remains assigned.

If any condition is incomplete, preparation fails before inserting a Pass B assignment, presentation session,
presentation, packet export, release member, or training exposure.

The error reports the actual gate state. On the canonical campaign it is:

```text
Pass B is locked until blinded Pass A is sealed for every current candidate and all hidden repeats:
0/48 candidate reviews, 0/6 repeat-stability rows, 0 open first-class Pass A presentation sessions
```

The first prepared session predates the first-class presentation-session layer, so its 12 open legacy
assignments appear in the `0/48 candidate reviews` term rather than the final presentation-session term.

## 3. Why all three checks matter

### 3.1 Candidate completeness

Per-candidate eligibility alone would reveal some contracts while other primary judgments remain blind. The
campaign-wide count prevents that leakage.

### 3.2 Repeat completeness

Completing all primaries is still insufficient. The six repeat observations measure stability under the same
blind condition. Revealing contracts before repeats would change the condition being measured.

### 3.3 No open first-class Pass A session

Counts can describe completed evidence while an assigned session still exists. Requiring zero open
first-class sessions prevents a partially completed or inconsistent presentation trajectory from overlapping
with Pass B.

## 4. Adversarial proof

The review test suite now exercises two distinct premature-reveal attacks.

### Attack A — individually eligible candidate

1. prepare one Pass A primary in a two-candidate fixture;
2. submit its valid human-form fixture;
3. attempt Pass B for that reviewed candidate;
4. require failure at `1/2` primary reviews and `0/2` repeats; and
5. verify zero Pass B assignments exist.

This proves that a candidate cannot cross the campaign blindness boundary merely because its own Pass A review
exists.

### Attack B — all primaries, no repeats

1. complete Pass A for both current candidates;
2. attempt Pass B;
3. require failure at `2/2` primary reviews and `0/2` repeats;
4. prepare a repeat-only blind session;
5. prove the packet contains neither `hidden_repeat` nor `sourceReviewId`;
6. submit both repeat responses;
7. verify two stability rows and zero additional candidate reviews; and
8. only then permit Pass B preparation.

The family-synthesis integration fixture was also corrected. It now completes Pass A, the required blind
repeats, Pass B, and only then Pass C. This caught and removed a test-only bypass of the real protocol.

## 5. Verification

The corpus TypeScript build passes and the full package test suite passes **21/21**.

The suite continues to prove:

- Pass A packet blindness;
- inclusion of retained structural rejections in the census;
- exact candidate-version hash binding;
- append-only submission evidence;
- no duplicate candidate reviews from repeats;
- repeat stability derivation;
- Pass C and Pass D prerequisite gates;
- no automatic release membership; and
- no training exposure from human-review operations.

The repository pre-commit gate also completed the full web dependency build before accepting the feature
commit.

## 6. Canonical live-ledger proof

The exact forbidden command was run against the canonical mounted ledger:

```text
alpha-corpus review-prepare
  --campaign alpha-calibration-v1
  --reviewer ajax
  --pass B
  --count 12
```

It exited nonzero with the `0/48` and `0/6` gate report.

The SQLite file was hashed and statted immediately before and after the attempt:

| Measurement | Before | After |
|---|---|---|
| SHA-256 | `7184a38a4213e319008d8f8f2b170f6d3c4c5d934b581c2afa9d7aad6c4847ce` | same |
| bytes | `5,836,800` | same |
| modification time | `2026-07-31 02:16:53.372268425 UTC` | same |

Independent post-attempt counts were:

| State | Count |
|---|---:|
| open Pass A assignments | 12 |
| Pass B assignments | 0 |
| human reviews | 0 |
| repeat-stability rows | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |

The failed preparation was therefore byte-for-byte non-mutating.

## 7. Public and deployment boundary

No web redeploy was required. The public service has no review mutation route and cannot invoke packet
preparation. Its existing aggregate dashboard already shows Pass B as locked behind Pass A and repeats.

The change belongs to the local, authority-bearing CLI used after a real downloaded human packet is returned.
The public contracts remain:

- `/corpus` and `/corpus/review` are read-only;
- browser drafts are local state, not evidence;
- downloaded files are not submitted evidence;
- public non-read methods return HTTP 405; and
- only local validated commands may add append-only review records.

## 8. Scientific state and next gate

The scientific state did not advance. There are still 48 candidates, 12 open Pass A assignments, and zero
human reviews, repeats, Pass B assignments, adjudications, release members, or training exposures.

The next authority-bearing action remains:

1. a real human completes the current 12 blinded Pass A responses;
2. the downloaded packet passes local validated import;
3. later Pass A sessions interleave new primaries with hidden repeats;
4. all 48 primaries and six repeats are sealed; and
5. only then may the CLI reveal Pass B contracts.

No model call, critic, synthetic generation, training run, GPU provision, Donto mutation, dataset release, or
fabricated human judgment was authorized or performed in this execution.
