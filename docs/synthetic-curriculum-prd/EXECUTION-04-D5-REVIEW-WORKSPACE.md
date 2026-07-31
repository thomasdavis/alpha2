# Execution 04 — D5 local-first human-review workspace

**Date:** 2026-07-31

**Status:** implemented, tested, deployed, and publicly verified; human review has not begun

**Scope:** make the existing 12-item blinded Pass A packet practical for a real human to complete without
adding a public write surface or changing any scientific stage

**Feature revision:** `cc9be1440ba22987124fc2388b38fc1a6e442d4e`

**Public route:** `https://alpha.donto.org/corpus/review`

## 1. Outcome

Alpha Corpus now has a browser-based D5 review instrument. It reads the latest hash-verified review packet from
the public corpus ledger, renders only the fields permitted by that packet's blindness contract, autosaves a
draft in the reviewer's own browser, validates every response against the executable rubric, and downloads a
completed JSON packet.

It does **not** submit a judgment to the web server. The only path from a downloaded packet into the scientific
ledger remains the local `review-submit` command introduced in Execution 03. That importer revalidates packet,
reviewer, assignment, candidate-version hash, rubric, pass, and session before it records append-only evidence.

The workspace therefore improves usability without weakening the research boundary:

1. model-visible candidate text is public;
2. a local browser draft is private to that browser profile;
3. a downloaded packet is a human-controlled file;
4. local validated import creates a review record;
5. review still does not create an adjudication, release member, or training exposure.

## 2. Shared executable rubric

`packages/corpus/src/review-contract.ts` is now the pure, browser-safe source of truth for:

- Pass A and Pass B dimension keys, labels, and explanations;
- allowed outcomes;
- 0–4 score anchors;
- follow-up-question policies;
- missing-clarification judgments;
- blank response construction;
- response and finding validation; and
- packet schema parsing.

The local importer in `review.ts` and the browser workspace use the same contract. The existing stored rubric
version and digest remain unchanged: the refactor preserved the exact canonical definition rather than silently
creating a differently capitalized rubric version.

The executable definition and stored ledger row independently produced the same SHA-256:
`dc19f22d43420988e11d4321cb2ff41bb004624c556ee14946189116e3af99bf`.

The package exposes this pure module at `@alpha/corpus/review-contract`, so a client component does not pull
SQLite, filesystem, or other Node-only implementation into its browser bundle.

## 3. Read-only packet delivery

`CorpusReader` gained two bounded methods:

- `listReviewPackets()` returns the latest exported packet for each immutable session plus assignment status;
- `reviewPacket(sessionId)` returns one verified packet.

The reader still opens SQLite with `readOnly: true` and `PRAGMA query_only=ON`. Packet loading additionally:

- accepts only a strict `review_session_<uuid>` identifier;
- binds the session as a SQL parameter;
- resolves only a blob path beneath the corpus root;
- reads the content-addressed bytes;
- recomputes and compares SHA-256 before parsing;
- validates packet schema and supported rubric version; and
- fails closed on a modified blob.

The latest live packet remains:

```text
session: review_session_1b479c00-3195-4d1f-ac69-86489019cd3e
pass: A
assignments: 12
packet SHA-256: 6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48
```

Pass A contains `kind` and natural-language messages only. Family identity, structural status, hidden contract,
generator notes, response policy, lenses, transformations, and other reviews remain absent from the packet and
therefore absent from this page.

## 4. Review interaction

`/corpus/review` lists verified sessions. `/corpus/review/<session>` provides:

- one assignment at a time;
- visible completion state for all assignments;
- role-labelled source dialogue;
- independent summaries of user aim and assistant move;
- all rubric dimension scores;
- question-policy and missing-clarification judgments;
- optional findings with exact evidence, severity, and recommendation;
- scientific disposition, rationale, confidence, uncertainty, and expertise needs;
- per-item errors from the same validator used by local import;
- previous/next navigation;
- browser-local autosave keyed by session and packet hash;
- an always-available draft download; and
- a completed-packet download disabled until every assignment validates.

The UI explicitly says that a downloaded packet is not submitted or accepted. It does not imply that an empty
form, structural validity, or polished wording is human approval.

## 5. Public mutation boundary

The old Next middleware file was migrated to the current Next proxy convention. The proxy now rejects every
method except `GET` and `HEAD` for `/corpus` and every descendant route. This app-level rule sits behind the
existing Caddy hostname rule, so the read-only boundary is enforced twice.

Measured canary behavior:

```text
GET  /corpus/review                                            200
GET  /corpus/review/review_session_1b479c00-...                200
POST /corpus/review                                            405
POST response: Alpha Corpus is a read-only public surface.
Allow: GET, HEAD
```

Measured public behavior:

```text
GET  https://alpha.donto.org/corpus/review                     200
GET  https://alpha.donto.org/corpus/review/review_session_...  200
POST https://alpha.donto.org/corpus/review                     405
```

There is no route handler, server action, form action, fetch call, or API endpoint for review submission.

## 6. Automated verification

The corpus package passes **14/14** tests. The two new packet-reader tests prove:

1. the latest public review packet is discovered, loaded, and hash-verified; and
2. modified content-addressed bytes are rejected rather than rendered.

The existing review tests still prove:

- Pass A blindness;
- retained structural rejections in the review census;
- resumable open assignments;
- append-only exact human-submission evidence;
- no automatic candidate promotion or training exposure; and
- failure when the candidate version hash changes.

The optimized Next build passed TypeScript and emitted both dynamic routes:

```text
ƒ /corpus/review
ƒ /corpus/review/[sessionId]
```

The pre-existing Next file-tracing warning associated with the upload route remains unrelated to Alpha Corpus.

## 7. Browser verification

Real Chromium checks covered the local production build and the public Cloudflare URL.

Verified behavior:

- session discovery and navigation;
- all rubric controls available to accessibility snapshots;
- input and score persistence after a full reload;
- removal of the synthetic persistence probe after the test;
- completed download disabled while required fields are missing;
- desktop layout;
- 390×844 mobile layout;
- zero document overflow at 390 px;
- zero browser page errors;
- zero console errors; and
- network history containing only GET/RSC reads for review navigation.

Evidence is on the mounted research drive:

| Artifact | SHA-256 |
|---|---|
| `reports/review-workspace-20260731/index-desktop.png` | `a6ffe28f2d5e784032f9799fa37b056f1fa057dc8e8bb4bbf994613ad7fe7ab8` |
| `reports/review-workspace-20260731/session-mobile.png` | `761c2ba4d1d7fa1acf7740717be621d9af545531d054cdc186a1afe034401c44` |

## 8. Deployment record

- immutable release:
  `/home/ajax/alpha2-web-releases/cc9be1440ba22987124fc2388b38fc1a6e442d4e`
- current pointer: `/home/ajax/alpha2-web-current`
- prior rollback release:
  `/home/ajax/alpha2-web-releases/5a305495b329d87af1362ac09148470899c14552`
- service: `alpha-corpus-web.service`
- service user: `ajax`
- loopback: `127.0.0.1:3104`
- activation: `2026-07-31 01:57:10 UTC`
- automatic restarts after release: `0`
- measured memory current/peak after verification: approximately `73/90 MiB`
- public GET latency during release check: approximately `0.12 s`

The release was tested first on a separate loopback port, then switched through the existing immutable symlink
and the scoped service was restarted. The SQLite ledger was neither copied nor migrated.

## 9. Scientific state after release

The release changed no scientific state:

| State | Count |
|---|---:|
| candidates | 48 |
| open Pass A assignments | 12 |
| completed assignments | 0 |
| reviews | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |

Post-release ledger validation reports:

- SQLite integrity: `ok`;
- foreign-key violations: `0`;
- missing tables/views/blobs: `0`;
- corrupt blobs: `0`;
- migrations: `2`;
- validator-owned footprint: `5.51 MiB`;
- complete artifact tree including browser reports: approximately `7.1 MiB`.

This remains far below the project-local 15 GiB pause threshold.

## 10. Next gate

The next action is still a real human judgment, not more code and not more generation:

1. open `https://alpha.donto.org/corpus/review`;
2. complete the 12 blinded Pass A assignments without consulting public candidate lineage;
3. download the completed packet;
4. import it locally with `review-submit`;
5. verify the append-only review evidence and unchanged release/training counts;
6. prepare Pass B only after Pass A is sealed; and
7. continue the remaining 36-candidate census under PRD-12.

No GPT-5.5 call, critic call, synthetic expansion, evaluation construction, training run, GPU provision, or
Donto mutation was authorized or performed in this execution.
