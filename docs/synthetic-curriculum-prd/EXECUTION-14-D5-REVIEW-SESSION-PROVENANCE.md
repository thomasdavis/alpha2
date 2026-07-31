# Execution 14 — D5 reviewer-session provenance and legacy-packet continuity

**Date:** 2026-07-31

**Source revisions:**

- `8fa7e4a739fc7574593e1f6e94c5c779a06d366d` — record D5 reviewer-session conditions;
- `7cd1c365599700217dca307c6515e8f920baedef` — correct the nested corpus-explorer landmark; and
- `ccdd9e6f00129a28571fd1158b42acf508fb0991` — preserve readability of legacy D5 packets.

**Production release:**
`/home/ajax/alpha2-web-releases/ccdd9e6f00129a28571fd1158b42acf508fb0991`

**Public routes:**

- `https://alpha.donto.org/corpus`;
- `https://alpha.donto.org/corpus/review`;
- `https://alpha.donto.org/corpus/review/review_session_1b479c00-3195-4d1f-ac69-86489019cd3e`;
- `https://alpha.donto.org/corpus?relation=human_review_session_declaration`; and
- `https://alpha.donto.org/corpus?relation=human_review_session_competence`.

**Scientific authority created:** none

**Human review evidence created:** none

**Model calls, synthetic generation, training, GPU work, release promotion, and Donto mutation:** none

## 1. Outcome

The D5 review instrument now records the conditions under which a genuine human judgment was made, not only
the judgment itself. A completed A or B submission must declare reviewer competence, review start and end,
interruption status, fatigue level, and an honest scope/condition note before the local importer writes any
artifact or scientific row. The declaration and its normalized competencies are appended in the same
transaction as the review evidence.

This closes a gap between the normative instrument and its executable packet. Appendix D already required
declared competence, session start/end, and interruption/fatigue notes. Before this execution, those fields
were not present in the packet, browser workspace, importer, or first-class ledger. The omission mattered:
later researchers could inspect a score without knowing whether the reviewer claimed relevant competence or
completed the session under interruption or fatigue.

The current gate is unchanged. The packet still contains 12 open blinded Pass A assignments and no human
responses. Making the evidence contract complete is not the same as supplying the evidence.

## 2. Authority and non-goals

This execution remained inside the already authorized human-review and public-read-only boundary. It did not:

- choose any review outcome or score;
- infer reviewer competence from identity or behavior;
- represent browser automation as human authority;
- reveal candidate lineage, family, structural status, hidden contract, or repeat identity in Pass A;
- create a public mutation endpoint;
- import a completed packet;
- promote a candidate into a release or training set;
- start generation, evaluation, training, or GPU work; or
- authorize Pass B, Pass C, Pass D, D6, or production corpus expansion.

Browser verification used clearly synthetic local draft values and never downloaded or submitted a completed
packet. Those values existed only in the isolated browser profile, which was closed after verification.

## 3. Evidence-contract gap

### 3.1 Normative requirement

The D5 instrument requires each session to state:

- reviewer identity or controlled alias;
- declared areas of competence;
- the limits or scope of that competence;
- session start and end;
- interruption status;
- fatigue level; and
- any condition that could affect interpretation of the judgments.

These are provenance, not quality labels. A declared competence does not make a judgment correct, and absence
of a specialty does not automatically invalidate an ordinary conversational judgment. The fields let later
analysis stratify evidence honestly and retain expertise limits.

### 3.2 Executable omission

The pre-Execution-14 packet represented assignment responses but had no session-level response object. The
importer could therefore write human review rows without a first-class declaration. The public workspace also
could not remind the reviewer to record conditions or stop completion when the declaration was incomplete.

### 3.3 Required correction

The correction had to satisfy all of these constraints together:

1. remain additive and append-only;
2. fail before any raw submission artifact or review row is written;
3. preserve the exact immutable packet-envelope gate;
4. preserve every historical packet byte and hash;
5. keep the public server read-only;
6. avoid inferring competence through brittle rules; and
7. keep the form practical on desktop and mobile.

## 4. Migration 8

Migration 8 is named `d5_human_review_session_declarations` and has digest:

```text
0374db80ce8ff18195c7e8f1ce57b78bac6f13b9c9f92f6f23014bb93f8b0f51
```

It was applied at `2026-07-31T05:43:04.890Z` and adds two first-class tables.

### 4.1 `human_review_session_declaration`

One immutable row binds a completed session to:

- its review session, campaign, human actor, rubric version, and pass;
- start and end timestamps;
- interruption and fatigue classifications;
- competence-scope and review-condition notes;
- the declared-competency set as canonical JSON;
- the exact exported packet-envelope blob; and
- the exact submitted packet blob.

The table enforces A/B pass values, valid JSON array shape, one declaration per session, and an end time no
earlier than the start time.

### 4.2 `human_review_session_competence`

The normalized child table stores one immutable row per declared competence. It deliberately does not make
the vocabulary a closed database enum. The current instrument offers six versioned choices—conversation,
linguistics, ontology, philosophy, evidence, and other—while the declaration retains the exact versioned
packet and rubric that gave those values meaning.

Both tables have update/delete rejection triggers. Migration 8 takes the canonical ledger from seven to eight
migrations, from 129 to 131 tables, and from 186 to 190 triggers. The five public/current views are unchanged.

## 5. Packet and importer contract

### 5.1 Additive `sessionResponse`

Every newly prepared human A/B packet contains:

```json
{
  "declaredCompetencies": [],
  "competenceNote": "",
  "startedAt": "",
  "endedAt": "",
  "interruptionStatus": null,
  "fatigueLevel": null,
  "conditionsNote": ""
}
```

Preparation always emits a blank declaration. It does not guess competence, timestamps, or conditions. A
completed download stamps `endedAt`; the browser records `startedAt` locally when the verified packet is
opened.

### 5.2 Fail-before-write submission order

The local importer now performs, in order:

1. JSON and typed packet validation;
2. complete assignment-response validation;
3. complete reviewer-session declaration validation;
4. actor, rubric, campaign, candidate-version, presentation, and assignment checks;
5. exact immutable exported-envelope verification;
6. content-addressed submission-blob creation; and
7. one append-only transaction containing the raw artifact, declaration, normalized competences, reviews,
   scores, findings, presentation evidence, and events.

An incomplete declaration therefore creates no declaration, competence, raw-submission, review, score,
finding, event, or status update.

### 5.3 Current blank packet

The same first-session identity and 12 assignment identities were resumed and re-exported after migration 8:

```text
session: review_session_1b479c00-3195-4d1f-ac69-86489019cd3e
packet SHA-256: 95b962709e9ad77aa91f2249f0648f1ee026b5ce3d64aaff792b615f751a484a
```

An exact comparison against the preceding `6740d835...` packet confirmed that assignment IDs, opaque item
IDs, candidate content hashes, presentation identities, candidate surfaces, and blank assignment responses
are unchanged. The re-export adds the blank session declaration, a fresh export timestamp, and the already
approved reminder that later blinded consistency presentations must be judged independently.

All 12 outcomes and every session-declaration field remain blank. The re-export created no human evidence.

### 5.4 Legacy packet continuity

The ledger correctly preserved earlier content-addressed exports with SHA-256 values:

- `6d2fc108130f9918056ff44405725f9cf72d8a0e9a0b0b5636719d154687d708`;
- `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48`; and
- current `95b962709e9ad77aa91f2249f0648f1ee026b5ce3d64aaff792b615f751a484a`.

The first canary for the final deployment caught that the stricter reader rejected an earlier v1 packet that
had no `sessionResponse`. HTTP status checks alone had missed the embedded server-render error. The final
reader now normalizes a missing legacy declaration to an explicitly incomplete blank declaration in memory.
It does not change the historical blob, hash, or export record. Submission validation still refuses that
blank declaration until a real reviewer completes it.

This is backward-compatible interpretation, not silent historical rewriting.

## 6. Public review workspace

The reviewer-facing declaration is a labeled region above the assignments. It provides:

- six independently selectable competence categories with plain-language descriptions;
- a free-text competence scope or limitations field;
- explicit interruption and fatigue selectors;
- a free-text review-condition note;
- locally recorded start time;
- end time stamped only when a completed packet is downloaded; and
- a live list of incomplete declaration fields.

The completed-download action remains disabled until all 12 assignment worksheets and the complete session
declaration pass the shared executable validator. Draft download remains available for resumable work.

The workspace now stores the active assignment's opaque item identity rather than a numeric array position.
The key remains scoped to the exact session and packet SHA. Real pointer navigation to assignment 2 stored
`opaque_591e35b2aeda` and restored the same assignment after reload on the public site.

Execution 13 described the earlier position value as an opaque identity. Inspection during this execution
found that revision `c4e7c4d...` actually stored a numeric position. That historical statement is corrected,
not erased: Execution 14 is the revision that makes the stored value an opaque item identity.

## 7. Public explorer correction

Browser proof of the new public tables found a pre-existing nested `main` landmark in the corpus explorer.
The site shell already owns the page-level `main`; the relation detail now uses a labeled `section`. The
explorer and review routes each expose exactly one main landmark.

This semantic correction does not alter corpus data, query behavior, or public visibility. All 131 tables and
five views remain dynamically browsable at `/corpus`.

## 8. Backup and migration evidence

Before applying migration 8, the canonical database was copied to:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/
pre-d5-session-declarations-20260731T054250Z.sqlite
```

Evidence:

| Artifact or check | Result |
|---|---|
| backup SHA-256 | `ee96b8a8573df70a1ab4b0d4b56176a4b765a062cf3a82fde85d574ae73fec38` |
| backup bytes | 6,107,136 |
| backup integrity | `ok` |
| backup foreign-key violations | 0 |
| current SQLite SHA-256 | `d2ccec649ab4aaeb0aac427391de0366a2b16b852477b2aa0da1724dd2ce9d19` |
| current integrity | `ok` |
| current foreign-key violations | 0 |
| missing tables/views/blobs | 0 |
| corrupt blobs | 0 |

The backup and all older packet blobs remain preserved.

## 9. Verification

### 9.1 Static and adversarial gates

| Check | Result |
|---|---|
| `npm run build -w @alpha/corpus` | pass |
| `npm test -w @alpha/corpus` | 23/23 pass, 0 fail |
| `npm run typecheck -- --pretty false` | pass |
| `npm run build -w @alpha/web` | optimized build pass |
| commit hooks | full web dependency build pass; no bypass |
| incomplete session declaration | zero declaration/review/raw-submission writes |
| declaration/competence update and delete | rejected by append-only triggers |
| legacy v1 packet without declaration | readable as incomplete; still not submit-ready |

### 9.2 Immutable release

The final release contains 1,994 files, occupies approximately 61 MiB, and is read-only. Its
`MANIFEST.sha256` file hashes to:

```text
927f62f0240228ef5cd36016199c66a953796f6706a72f0cfbb03c0b1acdf147
```

Every listed file passed `sha256sum -c` before activation.

### 9.3 Canary and production boundaries

Both the isolated canary and public deployment returned:

| Route | GET | POST |
|---|---:|---:|
| `/corpus` | 200 | 405 |
| `/corpus/review` | 200 | 405 |
| exact review session | 200 | 405 |
| declaration relation | 200 | 405 through `/corpus` |
| competence relation | 200 | 405 through `/corpus` |

The final review index rendered one campaign session and no application error. The exact session rendered six
competence checkboxes, 12 assignments, and a disabled completed-download action. The table explorer rendered
both new relations with zero rows and exactly one main landmark.

The production service is active on `127.0.0.1:3104`, points at the exact immutable release above, and reported
zero restarts after activation.

### 9.4 Real browser evidence

Final public evidence is stored under:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/reports/
d5-session-declarations-20260731/
```

| Artifact | SHA-256 |
|---|---|
| `canary-desktop-final.png` | `a18896a1d5e1e65f6243bee1b4f2614bd89c23ee8b51a513997eacd2fe4021f7` |
| `public-desktop-final.png` | `b571f1f58436be61043a41f0606aad09212d6ea10e8a7b75cb98d21de543215b` |
| `public-mobile-final.png` | `e7fa069551fb27e8b918483dc042e1d3ed5692edf9efa51d4dcd163ef2363cef` |
| `public-ledger-table-final.png` | `4c59ae004c8fa6959bee66f251ec1c39b5acaf56fd5b8eda191d04041e43bebc` |

At a 375 px layout viewport, the review page had one main landmark and no horizontal overflow. Browser errors
and console output were empty. Browser sessions and the canary process were closed after verification.

## 10. Canonical-ledger reconciliation

| State | Count |
|---|---:|
| migrations | 8 |
| tables | 131 |
| views | 5 |
| triggers | 190 |
| candidates | 48 |
| structurally valid | 42 |
| retained structural rejections | 6 |
| open Pass A assignments | 12 |
| human-review session declarations | 0 |
| normalized human competencies | 0 |
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

The complete project-owned corpus tree is 42,336,230 bytes, or approximately 40.37 MiB. It remains far below
the operator's 15 GiB resumable pause threshold.

## 11. Remaining gate

The next authority-bearing action remains human:

1. a real reviewer declares competence scope and session conditions;
2. that reviewer completes all 12 open blinded Pass A worksheets;
3. the browser downloads the completed packet with an end timestamp;
4. the local `review-submit` command validates the exact exported envelope and appends the genuine evidence;
5. the remaining blind Pass A census and six hidden repeats continue; and
6. no contract-aware Pass B material is revealed until the complete campaign-wide blind gate passes.

Do not use a model to impersonate the reviewer. Do not begin corpus expansion, D6 population, release export,
training, or GPU work merely because the provenance contract is now executable.
