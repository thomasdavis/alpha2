# Execution 13 — D5 human-review workspace hardening

**Date:** 2026-07-31

**Application revisions:**

- `f8e2d95596c2e68a4092b851e571aa95088707cd` (`Harden the D5 human review workspace`)
- `c4e7c4db2e007ee247113bbfe97029f5a84eec1f` (`Fix review page landmarks`)

**Production release:**
`/home/ajax/alpha2-web-releases/c4e7c4db2e007ee247113bbfe97029f5a84eec1f`

**Public routes:**

- `https://alpha.donto.org/corpus/review`
- `https://alpha.donto.org/corpus/review/review_session_1b479c00-3195-4d1f-ac69-86489019cd3e`

**Scientific authority created:** none

**Human review evidence created:** none

**Model calls, synthetic generation, training, GPU work, release promotion, and Donto mutation:** none

## 1. Outcome

The existing local-first D5 Pass A workspace is now practical to use for a sustained human review session on
desktop and mobile. It preserves the exact immutable-packet and public-read-only contracts from Executions 04
and 11 while adding the missing navigation, recovery, accessibility, and fatigue-reduction mechanics needed to
complete the first 12-assignment packet accurately.

The deployed workspace now provides:

- a visible assignment navigator at mobile widths;
- browser-local active-assignment persistence tied to the exact session and packet hash;
- `Resume first incomplete` and `Next incomplete` controls;
- local completion progress with a live status announcement;
- focus and scroll movement to the selected assignment heading;
- descriptive accessible names for every rubric score;
- safe recovery from an incompatible or corrupt local draft without hiding the canonical source packet;
- theme-aware action colors with verified contrast; and
- one correctly nested main landmark per page.

These are review-instrument improvements, not scientific observations. The system still has 12 open Pass A
assignments and zero human reviews. A real human must make the judgments.

## 2. Authority and non-goals

The operator's active goal requires a deeply categorized synthetic curriculum and public scientific ledger, but
the current D5 gate is human conceptual adjudication rather than more generation. This execution stayed within
that boundary by improving the already-authorized browser instrument.

It did not:

- fill any response field;
- infer or manufacture a human judgment;
- inspect or reveal candidate lineage in the review UI;
- expose candidate IDs, family labels, contracts, structural status, or hidden-repeat identity;
- add a public mutation or submission endpoint;
- import a packet into SQLite;
- change candidate, review, adjudication, release, or training state;
- call GPT-5.4, GPT-5.5, GPT-5.6-sol, or any other model;
- launch a training process or rent a GPU; or
- authorize D6, corpus expansion, or a training release.

The browser remains a local drafting surface. It downloads a completed JSON packet; only the local
`review-submit` CLI may validate and append a genuine human submission to the ledger.

## 3. Audit findings

The audit used the deployed public route in real Chromium at desktop and 390 px mobile widths, plus direct code
inspection of the review route and its shared packet contract.

### 3.1 Mobile navigation was unintentionally hidden

The assignment navigator used an `aside` element. A global narrow-screen rule hides all `aside` elements, so
the entire 12-item navigator disappeared on mobile. The review form remained technically reachable by scrolling,
but the reviewer could not see progress or move directly among assignments.

This was a semantic-selector collision, not a data or packet problem. The navigator is now a labeled section
whose visibility is controlled by the review component rather than the shell's generic sidebar rule.

### 3.2 Rubric radios had ambiguous accessible names

Each rubric dimension visibly included its label and anchor meanings, but individual radio controls were exposed
to assistive technology primarily as repeated numeric names such as `0`, `1`, and `2`. Across several rubric
dimensions this made the controls difficult to distinguish.

Each score now has an explicit name combining the dimension, numeric value, and anchor meaning, for example:

```text
Direct responsiveness: 0, Critical failure
```

The visible instrument and scoring semantics are unchanged.

### 3.3 Review position was not durable

The packet draft was already stored in browser-local storage, but the active assignment was not. Reloading the
page returned the reviewer to the first assignment even when later work was in progress. Moving to another
assignment from near the bottom of a long form also did not reliably bring the new prompt into view.

The workspace now stores only the active assignment's opaque presentation identity, under a key scoped to the
session and immutable packet hash. On navigation it focuses and scrolls the assignment heading. Reload restores
the position only when the same canonical packet remains active.

This position record is convenience state. It is not part of the review packet or scientific ledger.

### 3.4 Incomplete work was expensive to find

A reviewer could select assignments by number but had no direct control for finding the first or next incomplete
worksheet. For a long rubric this increases fatigue and the chance of silently missing fields.

The workspace now calculates completion locally and provides:

- a native progress element;
- a textual completed/total count;
- a `Resume first incomplete` action; and
- a `Next incomplete` action that wraps through the packet.

Completion still uses the same executable shared rubric validation as packet download and local import. The UI
does not invent a weaker visual definition of complete.

### 3.5 Invalid local state could obscure the usable workspace

The immutable-envelope boundary already rejected altered browser drafts. The hardened behavior makes recovery
explicit: a malformed, stale, or envelope-incompatible local draft is discarded, the exact server packet is
restored, and the reviewer sees a notice explaining that only the browser draft was removed.

The canonical packet and source artifact are never rewritten. Resetting the form now clears both the packet
draft and the local position record.

### 3.6 Theme contrast needed correction

The prior primary action treatment used white text on the dark accent. Its measured contrast was approximately
3.68:1 and was insufficient for normal-sized action labels. The action treatment now uses the design system's
theme-aware foreground token:

| Theme | Measured contrast |
|---|---:|
| light | 5.17:1 |
| dark | 5.38:1 |

The existing Alpha palette remains unchanged.

### 3.7 First canary exposed duplicate main landmarks

The first hardened canary rendered successfully and passed its route boundary checks, but browser inspection
found a page-level `main` nested inside the application shell's `main`. That created two main landmarks.

The page wrappers were changed to neutral `div` elements, leaving the shell as the sole owner of the main
landmark. The corrected release reports exactly one main landmark on both desktop and mobile.

The first canary was never activated in production. It remains preserved as a read-only intermediate artifact:

```text
/home/ajax/alpha2-web-releases/f8e2d95596c2e68a4092b851e571aa95088707cd
```

Its `MANIFEST.sha256` file hashes to:

```text
e10ff2620369e6b6c247a51761730050d5c616aa85dc71d31bd5af8ed5cc21cb
```

Preserving it records the correction rather than rewriting the deployment history.

## 4. Implementation

The implementation changed only the review application:

- `apps/web/src/app/corpus/review/[sessionId]/review-workspace.tsx`
- `apps/web/src/app/corpus/review/[sessionId]/page.tsx`
- `apps/web/src/app/corpus/review/page.tsx`

No corpus migration, generation task, review row, or release member was added.

### 4.1 Packet-scoped browser state

The existing draft key remains bound to the session and immutable packet export. The new position key is also
bound to that identity. A position from another session or packet cannot silently select an assignment in the
current packet.

The stored value is an opaque presentation identity rather than candidate identity. It does not weaken review
blindness and is not sent to the server.

### 4.2 Focus and navigation behavior

Every assignment heading is programmatically focusable. Selecting a numbered item, resuming incomplete work,
or moving to the next incomplete item:

1. changes the active assignment;
2. stores its opaque position locally;
3. focuses the new heading; and
4. scrolls the heading into view.

This makes the state transition perceptible to keyboard and assistive-technology users as well as sighted mouse
users.

### 4.3 Scientific boundary preservation

The following prior contracts remain in force:

- source content is loaded from an exact content-addressed exported packet;
- only typed response worksheets may differ in a completed packet;
- the browser uses the same envelope and rubric validators as the local importer;
- autosave is browser-local;
- download is the only completion action in the public UI;
- Caddy and the application return 405 to non-read methods; and
- local import remains fail-closed and append-only.

## 5. Verification

### 5.1 Static, build, and focused test gates

| Check | Result |
|---|---|
| `npm run typecheck -- --pretty false` | pass |
| `npm run build -w @alpha/corpus` | pass |
| `npm test -w @alpha/corpus` | 22/22 pass, 0 fail |
| `npm run build -w @alpha/web` | optimized build pass |
| both commit hooks | full web dependency build pass |
| `alpha-corpus validate` | integrity ok; zero FK, missing relation, missing blob, or corrupt blob failures |

The application commits were pushed to `origin/master`. No hook was bypassed.

### 5.2 Corrected immutable canary

The corrected build was materialized at:

```text
/home/ajax/alpha2-web-releases/c4e7c4db2e007ee247113bbfe97029f5a84eec1f
```

The release contains 1,994 files, occupies approximately 61 MiB, and is read-only. Its manifest file SHA-256 is:

```text
e22985fafa493c9eb815f4492337d60edaca1422f72133233d7fc02364e3275b
```

Before activation, an isolated canary on loopback port 3115 proved:

- HTTP 200 for the corpus index, review index, and exact session route;
- HTTP 405 for POST to the review index and session route;
- 375 px viewport width with no horizontal overflow;
- all 12 assignment buttons visible;
- exactly one main landmark;
- descriptive radio accessible names;
- no error overlay or console error;
- assignment navigation focuses and scrolls the selected heading; and
- reload restores the selected assignment.

One measured navigation moved from a scrolled position to assignment 2, focused its `h2`, reached scroll position
833, and restored that assignment after reload.

### 5.3 Production activation

The immutable current link was atomically changed to:

```text
/home/ajax/alpha2-web-current
  -> /home/ajax/alpha2-web-releases/c4e7c4db2e007ee247113bbfe97029f5a84eec1f
```

`alpha-corpus-web.service` was restarted at 2026-07-31 05:19:42 UTC. Post-deployment inspection showed:

| Signal | Observation |
|---|---|
| service | active/running |
| restart count | 0 |
| listener | `127.0.0.1:3104` |
| memory current | approximately 77.6 MiB at final reconciliation |
| memory peak | approximately 97.2 MiB |

Public boundary checks returned:

| Route | GET | POST |
|---|---:|---:|
| `/corpus` | 200 | 405 |
| `/corpus/review` | 200 | 405 |
| `/corpus/review/review_session_1b479c00-3195-4d1f-ac69-86489019cd3e` | 200 | 405 |

The same navigation, focus, reload-persistence, single-landmark, accessible-name, and no-overflow assertions passed
against the public route in real Chromium at 390 px.

## 6. Browser evidence

Browser captures are stored on the mounted research disk at:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/reports/d5-review-hardening-20260731/
```

| Artifact | SHA-256 |
|---|---|
| `public-desktop-light.png` | `f9c343c2fd0d27a464f2f8b6e4b41a86939a32a283543be7eb8150d735c0cc10` |
| `public-mobile-dark.png` | `0eaa184fbc4ecc2f2e0b58f3ad6cee124e178f1d3164d5c4d82f27bf7acae486` |
| `canary-desktop-light.png` | `f9c343c2fd0d27a464f2f8b6e4b41a86939a32a283543be7eb8150d735c0cc10` |
| `canary-mobile-dark.png` | `0eaa184fbc4ecc2f2e0b58f3ad6cee124e178f1d3164d5c4d82f27bf7acae486` |

The identical canary/public hashes show that the activated artifact rendered the same reviewed states; they do
not establish candidate quality.

## 7. Canonical-ledger reconciliation

The main SQLite file remains byte-identical at:

```text
7184a38a4213e319008d8f8f2b170f6d3c4c5d934b581c2afa9d7aad6c4847ce
```

The reconciled state is:

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
| Pass B reviews | 0 |
| Pass C family syntheses | 0 |
| structural dispositions | 0 |
| Pass D closeouts | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |
| execution authorizations | 0 |

The complete Alpha Corpus project tree is 37,905,554 bytes, or approximately 36.15 MiB. This is far below the
operator's 15 GiB resumable pause threshold.

## 8. Scheduled reporting

The separately authorized factual timer remains enabled. Its 2026-07-31 05:09:16 UTC run completed successfully,
and the next run was scheduled for 07:09:16 UTC. No ad hoc Discord message was sent for this execution.

The reporter continues to derive its counts from SQLite and must not describe structural validity as human
acceptance or imply that generation or training has begun.

## 9. Remaining gate

The next authority-bearing action is unchanged:

1. a real human completes the 12 open blind Pass A assignments in the hardened workspace;
2. the completed packet is downloaded;
3. the local `review-submit` CLI validates the exact envelope and appends the genuine human evidence;
4. the remaining Pass A census and hidden repeats continue under PRD-12; and
5. Pass B, Pass C, structural dispositions, and Pass D remain locked until their prerequisites are satisfied.

Do not use a model to impersonate that reviewer. Do not begin more generation, D6 construction, training, or GPU
work merely because the browser instrument is ready.
