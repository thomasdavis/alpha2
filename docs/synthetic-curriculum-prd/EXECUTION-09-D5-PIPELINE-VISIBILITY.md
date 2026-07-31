# Execution 09 — D5 campaign-pipeline visibility

**Date:** 2026-07-31

**Status:** implemented, tested, deployed, and publicly browser-verified; human review has not begun

**Scope:** make the complete D5 human-review sequence visible without revealing blinded candidate lineage,
adding a public mutation path, or implying that one 12-item session completes the 48-candidate census

**Feature revision:** `8f25d51c362921480d68e37a22d57a9ee39d47d9`

**Public route:** `https://alpha.donto.org/corpus/review`

## 1. Problem

The local-first review workspace correctly exposed one immutable Pass A packet with 12 open assignments. Its
session list did not show how that packet relates to the complete D5 protocol. A reviewer could reasonably
read `12 open` as the whole campaign even though PRD-12 requires:

1. blind Pass A on all 48 candidates;
2. six hidden repeat presentations;
3. contract-aware Pass B on all 48 candidates;
4. six Pass C family syntheses;
5. six separate structural dispositions; and
6. one non-binding Pass D campaign closeout.

That ambiguity was operationally important. Completing the prepared packet would be evidence for 12 of 48
Pass A candidates, not permission to reveal contracts, synthesize families, generate more material, release a
dataset, train a model, or use compute.

## 2. Outcome

The public review dashboard now derives and renders the reviewer-scoped D5 pipeline directly from the canonical
SQLite ledger. The live panel reports:

| Stage | Complete | Total | Open now |
|---|---:|---:|---:|
| Pass A — blind conversation | 0 | 48 | 12 |
| Hidden repeats — blind stability | 0 | 6 | 0 |
| Pass B — contract aware | 0 | 48 | 0 |
| Pass C — family synthesis | 0 | 6 | 0 |
| Structural — rejected cases | 0 | 6 | 0 |
| Pass D — campaign closeout | 0 | 1 | 0 |

Only Pass A is marked current. Every downstream stage is visibly locked. The current-gate sentence is explicit:

> Complete and locally import the 12 open Pass A assignments. They are one session within the 48-candidate
> census.

The panel also states that browser drafts are not evidence until a downloaded packet passes the local importer.
It displays `No execution authority` because the ledger contains no authority-bearing transition and the Pass D
schema enforces `execution_authorized = 0`.

## 3. Aggregate-only query contract

`CorpusReader.reviewCampaignProgress(campaignSlug, reviewerAlias)` performs one read-only, parameter-bound
aggregate query. It is scoped by:

- generation-campaign slug;
- a recorded human actor with the matching display name; and
- campaign, reviewer, pass, presentation, synthesis, structural, and closeout relations.

It returns only stage totals and aggregate counts. The page deliberately does not receive or render:

- candidate IDs;
- candidate-version IDs;
- concept-family labels or slugs;
- current structural status per candidate;
- hidden contracts;
- generator notes;
- transformation or lens metadata;
- hidden-repeat identity;
- another reviewer's judgments; or
- candidate-level adjudication state.

The reader still opens SQLite with `readOnly: true` and `PRAGMA query_only=ON`. The method returns `null` if a
required relation, campaign, or reviewer does not exist. It cannot silently fabricate a partial pipeline for an
older schema.

The adjudication count also guards `json_extract` with `json_valid`, so a malformed historical rationale cannot
turn a public page read into a server failure.

## 4. Stage semantics

The panel is an explanation of evidence state, not an orchestration engine. Its unlock states mirror PRD-12:

- Pass A is always the first human gate.
- Repeats unlock only after all primary Pass A reviews are sealed.
- Pass B unlocks only after Pass A and repeat evidence are complete.
- Pass C and structural disposition unlock only after Pass A, repeats, and Pass B.
- Pass D unlocks only after all candidate, repeat, family, and structural human evidence is complete.
- A complete Pass D still does not authorize generation, release, training, model calls, or compute.

The UI does not create assignments. It reports assignments and evidence already present in SQLite. Assignment
preparation and submission remain local, fail-closed CLI operations.

## 5. Automated verification

The corpus package passes **21/21** tests. The new migrated-ledger fixture proves that the reader:

- counts a two-candidate campaign and one concept family;
- separates one retained structural rejection;
- reports one open Pass A assignment;
- computes candidate-, family-, rejection-, repeat-, and closeout-level totals independently;
- returns no candidate lineage in its typed result;
- returns `null` for an unknown campaign; and
- returns `null` for an unknown reviewer.

The complete corpus suite continues to prove:

- exact public packet hashing;
- Pass A blindness;
- append-only human submissions;
- hidden repeat separation;
- Pass C and Pass D prerequisite gates;
- no automatic candidate promotion;
- no release membership or training exposure from review; and
- fail-closed storage and provenance behavior.

Both package TypeScript and the optimized Next build passed. The build emitted:

```text
ƒ /corpus
ƒ /corpus/review
ƒ /corpus/review/[sessionId]
ƒ Proxy (Middleware)
```

The pre-existing Turbopack warning about broad file tracing through the dashboard version route remains. It did
not originate in the corpus reader or review dashboard.

## 6. Release procedure

The standalone artifact was copied to a new immutable release directory:

```text
/home/ajax/alpha2-web-releases/8f25d51c362921480d68e37a22d57a9ee39d47d9
```

Before activation it was booted against the canonical ledger on separate loopback port `3115`. The canary:

- returned the pipeline page;
- contained the exact census, open-session, stage, and authority-boundary text;
- rejected `POST /corpus/review` with HTTP 405; and
- remained a live listening process until explicitly stopped.

The release contains a deterministic SHA-256 manifest and was made read-only. The
`/home/ajax/alpha2-web-current` symlink was then switched atomically, and only
`alpha-corpus-web.service` was restarted. The previous `cc9be1440ba22987124fc2388b38fc1a6e442d4e` release
remains intact as the rollback target.

After activation:

| Check | Result |
|---|---|
| service state | active / running |
| automatic restarts | 0 |
| activation | 2026-07-31 03:43:38 UTC |
| loopback `/corpus/review` | HTTP 200 |
| public `/corpus/review` | HTTP 200 |
| public `/corpus` | HTTP 200 |
| public `POST /corpus/review` | HTTP 405 |
| pushed source | `8f25d51c362921480d68e37a22d57a9ee39d47d9` |
| deployed release path | same full revision |

The ledger was neither copied nor migrated as part of this web release.

## 7. Public browser proof

Real Chromium checked the public Cloudflare URL after deployment.

### Desktop

- viewport: 1440×1100;
- all six stages visible in order;
- current and locked states legible without relying only on color;
- exact `0 / 48`, `0 / 6`, and `0 / 1` counts visible;
- `12 open`, `48-candidate census`, and `No execution authority` visible;
- no known family slug, hidden-contract field, or structural-rejection value in rendered page text;
- document width equals viewport width;
- no Next error overlay;
- no page errors; and
- no console errors.

### Mobile

- viewport: 390×844;
- stages stack in the same semantic order;
- current-gate explanation remains fully readable;
- body and document width: 375 px within a 390 px viewport;
- no horizontal overflow;
- no Next error overlay;
- no page errors; and
- no console errors.

Evidence is stored on the mounted research drive:

| Artifact | SHA-256 |
|---|---|
| `reports/d5-pipeline-visibility-20260731/public-desktop.png` | `390f4cdbf8cc40ccacc038c5f8ee49678a070dd5cacd8002003ca512856101a3` |
| `reports/d5-pipeline-visibility-20260731/public-mobile.png` | `f016e512465ec4cfe4f5264f2fa35e81160fa28f0d948325dfba3a908fd279ff` |

## 8. Scientific reconciliation

The UI release changed no scientific state:

| State | Count |
|---|---:|
| migrations | 7 |
| tables | 129 |
| views | 5 |
| triggers | 186 |
| candidates | 48 |
| structurally valid | 42 |
| retained structural rejections | 6 |
| open Pass A assignments | 12 |
| human reviews | 0 |
| repeat-stability rows | 0 |
| Pass C syntheses | 0 |
| structural dispositions | 0 |
| Pass D closeouts | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |
| execution authorizations | 0 |

Post-release validation reports:

- SQLite integrity: `ok`;
- foreign-key violations: `0`;
- missing tables or views: `0`;
- missing blobs: `0`;
- corrupt blobs: `0`; and
- complete project-owned artifact footprint: `35.65 MiB` (`37,383,662` bytes).

The footprint remains far below the resumable 15 GiB project threshold.

## 9. Next gate

The next authority-bearing action is still human work:

1. open `https://alpha.donto.org/corpus/review`;
2. complete the 12 currently blinded Pass A assignments without inspecting public candidate lineage;
3. download the completed JSON packet;
4. import it locally with the validated `review-submit` command;
5. verify the append-only review artifact and unchanged release/training counts; and
6. prepare another blinded Pass A session rather than revealing contracts early.

No GPT-5.4 or GPT-5.5 call, synthetic generation, critic call, training run, GPU provision, Donto mutation,
release promotion, or fabricated human judgment was authorized or performed in this execution.
