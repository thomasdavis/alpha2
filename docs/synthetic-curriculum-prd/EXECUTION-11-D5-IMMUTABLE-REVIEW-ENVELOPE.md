# Execution 11 — D5 immutable review-packet envelope

**Date:** 2026-07-31

**Status:** implemented, tested, deployed, and publicly browser-verified; human review has not begun

**Scope:** prove that a human-review submission can change reviewer-response fields only and must otherwise
match an exact packet previously exported by the ledger

**Feature revision:** `e07477b934897b71f241724a230e2ccd6320e0c9`

**Public route:**
`https://alpha.donto.org/corpus/review/review_session_1b479c00-3195-4d1f-ac69-86489019cd3e`

## 1. Problem

The D5 importer already verified assignment identity, presentation identity, reviewer, rubric, open state, and
the stored candidate-version content hash. That was necessary but not sufficient to prove what a reviewer
actually saw.

A packet could retain the original `candidateContentSha256` while changing the model-visible candidate text,
the opaque display identifier, instructions, ordering seed, or another presentation field. The old importer
would bind the resulting response to the original candidate version because it did not compare the complete
visible packet with an exported artifact. The browser's local-draft restoration had the same weakness: it
compared session, pass, rubric, assignment IDs, and candidate hashes, but not the candidate prose itself.

This was a provenance defect, not evidence of a corrupt current packet or a fabricated review. The canonical
packet remained blank and hash-valid. The defect was found before any human review, repeat response,
adjudication, release membership, or training exposure existed.

## 2. Immutable-envelope contract

The packet is divided into two regions:

1. **immutable envelope:** schema version, campaign, session, pass, reviewer alias, rubric, seed, creation
   time, instructions, assignment order and identity, presentation identity, opaque display identity,
   candidate-version hash, and the complete model-visible candidate payload;
2. **mutable response:** only each assignment's rubric-shaped `response` object.

`humanReviewPacketEnvelope(packet)` replaces every response with the rubric's exact empty response while
preserving every other field. `humanReviewPacketEnvelopeJson(packet)` then canonicalizes the result with
recursively sorted object keys. The implementation deliberately preserves unknown packet- or assignment-level
fields so an injected immutable field cannot disappear during verification and compare as equal.

The same browser-safe implementation is used in two places:

- the public workspace accepts a saved local draft only when its immutable envelope equals the exact source
  packet loaded from the content-addressed export; and
- the local importer hashes the submitted immutable envelope and requires an exact matching
  `human_review_packet_json` export artifact for the same session and pass.

The importer also requires that the matching blob has the exact canonical byte length and
`application/json` media type. The verified envelope SHA-256 is retained in review rationale and event
payloads and returned by the importer. Reviewer and rubric lookup during submission is now read-only:
submission cannot create a new actor or rubric as a side effect of a malformed packet.

The response remains ordinary natural-language human evidence. No JSON delimiter, internal identity, hash,
or envelope metadata is model-visible training content.

## 3. Code boundary

| File | Change |
|---|---|
| `packages/corpus/src/review-contract.ts` | browser-safe immutable-envelope construction, canonicalization, and comparison |
| `packages/corpus/src/review.ts` | read-only actor/rubric resolution, exact exported-envelope gate, and envelope lineage in review events |
| `apps/web/src/app/corpus/review/[sessionId]/review-workspace.tsx` | exact envelope validation before restoring a browser-local draft |
| `packages/corpus/src/review.test.ts` | response-only positive control and candidate/presentation tamper attacks |

No migration was needed. Existing export and blob relations already contain the exact content-addressed
evidence required for the gate.

## 4. Adversarial verification

The corpus suite now passes **22/22** tests. The new test begins from a real prepared packet and proves:

- filling only response fields preserves the immutable envelope;
- changing the visible assistant message while retaining the original candidate hash fails;
- changing the actual presentation ID fails against stored assignment/presentation state;
- changing the opaque presentation identifier fails the exported-envelope gate;
- all failed attacks create zero `review` rows;
- all failed attacks create zero `review_presentation_response` rows;
- all failed attacks create zero human-review submission artifacts; and
- restoring the exact exported envelope and changing only the response succeeds, with the returned envelope
  SHA equal to the original packet export SHA.

The prior changed-candidate-version test still fails with the more specific `candidate version changed`
diagnosis. Existing Pass A, hidden-repeat, Pass B, Pass C, Pass D, public-reader, append-only, storage, and
provenance tests continue to pass.

The optimized Next production build passed. Its only warning is the pre-existing broad file-tracing warning
through `server-state.ts` and the version route; this change did not add to that trace.

## 5. Canonical packet reconciliation

The canonical Pass A packet remains blank. It was not submitted, and no response was fabricated for this
proof.

| Property | Verified value |
|---|---|
| session | `review_session_1b479c00-3195-4d1f-ac69-86489019cd3e` |
| assignments | 12 |
| packet bytes | 16,067 |
| packet file SHA-256 | `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48` |
| computed immutable-envelope bytes | 16,067 |
| computed immutable-envelope SHA-256 | `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48` |
| recorded export blob SHA-256 | `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48` |
| recorded blob media type | `application/json` |

The equality proves that a future response-only completion of this exact packet has a valid exported envelope.
It does not prove the blank packet is a completed review; the rubric validator correctly refuses empty
responses.

## 6. Immutable release and live proof

The standalone web artifact was copied to:

```text
/home/ajax/alpha2-web-releases/e07477b934897b71f241724a230e2ccd6320e0c9
```

Its 1,993-entry SHA-256 manifest validates in full. Manifest SHA-256:

```text
a785c2dbe853077c5bbeb1498ed2b8c89921ab80f2b109377dc3995f12dcb055
```

The release first ran as a canary on loopback port 3115. It returned 200 for `/corpus` and
`/corpus/review`, returned 405 for `POST /corpus/review`, rendered the 12-assignment session, listened with a
bounded process, and was then explicitly stopped. An initial integrity-check invocation mistakenly passed
individual filenames to `sha256sum --check` as well as the manifest and produced parser errors; it changed no
file. The corrected manifest-only command passed before the canary and activation.

The current symlink was switched atomically and `alpha-corpus-web.service` restarted at
2026-07-31 04:13:20 UTC. Post-activation evidence:

| Check | Result |
|---|---|
| deployed release | `e07477b934897b71f241724a230e2ccd6320e0c9` |
| service | active |
| automatic restarts | 0 |
| loopback review route | HTTP 200 |
| public review route | HTTP 200 |
| public corpus route | HTTP 200 |
| public review POST | HTTP 405 |
| process listener | `127.0.0.1:3104` |

A real Chromium session then performed the browser-side adversarial check:

1. loaded the public 12-item Pass A workspace;
2. confirmed meaningful content, no framework overlay, and 12 incomplete assignments;
3. inserted `__ALPHA_TAMPER_SENTINEL__` into the first assistant message in browser local storage;
4. confirmed that the sentinel was present in storage;
5. reloaded the public page;
6. observed `bodyHasTamper=false` and `storageHasTamper=false`; and
7. observed no framework overlay and no captured console errors.

The screenshot is stored on the mounted research drive:

| Artifact | SHA-256 |
|---|---|
| `reports/d5-envelope-binding-20260731/public-review-after-tamper-rejection.png` | `c349f29ccd249bc158eda26d6e07c5183cc7a33c6b3cb54cca220c04f77aac85` |

This browser exercise changed local storage only. It entered no review judgment and called no server write.

## 7. Scientific reconciliation

The canonical SQLite main-file SHA-256 remains:

```text
7184a38a4213e319008d8f8f2b170f6d3c4c5d934b581c2afa9d7aad6c4847ce
```

The live logical state is:

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

`alpha-corpus validate` reports SQLite integrity `ok`, zero foreign-key violations, zero missing tables or
views, zero missing blobs, and zero corrupt blobs. The complete project-owned artifact footprint, including
this browser evidence, is 37,493,414 bytes (35.76 MiB), far below the resumable 15 GiB pause threshold.

## 8. Authority and next gate

This execution hardens evidence capture; it does not create scientific evidence about candidate quality.
The next authority-bearing action remains a real human completing the 12 blinded Pass A assignments in the
public local-first workspace, downloading the completed packet, and importing it locally. After that, the
workflow must continue through the remaining blind Pass A census and six hidden repeats before any Pass B
contract is revealed.

No GPT-5.4 or GPT-5.5 call, synthetic generation, critic call, training run, GPU provision, Donto mutation,
dataset release, public write endpoint, ad hoc Discord post, or fabricated human judgment was authorized or
performed in this execution.
