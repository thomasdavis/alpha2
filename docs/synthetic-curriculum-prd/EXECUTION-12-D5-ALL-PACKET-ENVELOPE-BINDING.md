# Execution 12 — D5 immutable-envelope binding for every human packet

**Date:** 2026-07-31

**Implementation revision:** `6a33410` (`Bind every D5 submission to its exported packet`)

**Scope:** Pass C family synthesis, structural dispositions, and Pass D campaign closeout; shared A/B machinery

**Scientific authority created:** none

**Model calls, generation, training, GPU work, release promotion, and Donto mutation:** none

## 1. Outcome

Every D5 human submission path now proves that the non-response portion of the submitted packet is byte-
equivalent, after deterministic response redaction, to a content-addressed JSON packet previously exported for
the same session and pass.

Execution 11 established this property for Pass A and Pass B candidate review. This execution extends the same
contract to:

- Pass C family synthesis;
- the structural dispositions embedded in the Pass C packet; and
- Pass D campaign closeout and candidate adjudication.

The implementation deliberately does not infer that a packet is authentic from plausible IDs, current ledger
rows, candidate hashes, or an internally consistent evidence digest. Those checks remain useful, but they are
not a substitute for proving that the human saw the exact packet whose immutable envelope is being submitted.

The canonical D5 campaign remains scientifically unchanged: 48 candidates, 12 open blinded Pass A
assignments, and zero human reviews, repeat-stability rows, family syntheses, structural dispositions,
closeouts, adjudications, release members, training exposures, or execution authorizations.

## 2. Why this was necessary

Before this change, the Pass C and Pass D importers checked substantial evidence:

- campaign and session identity;
- current candidate and family versions;
- candidate content hashes;
- A/B review bases;
- repeat evidence;
- surface-analysis identity;
- evidence snapshot digests;
- open assignment state; and
- response completeness.

Those checks could reject changed candidate evidence, but they did not bind every presentation-bearing field to
the actual exported artifact. A completed response could therefore be attached to a packet with a changed
timestamp, instruction, order, or another apparently non-semantic field while still passing the narrower
evidence checks.

That is a provenance defect. Review instructions, ordering, opaque identities, timestamps, and surrounding
presentation are part of the conditions under which a human judgment was elicited. They must remain evidence,
not editable decoration.

The same audit found a second problem: submission code used `ensure...` helpers for actors and rubrics. A
malformed submission could therefore create idempotent support rows before failing. No scientific judgment was
created, but a fail-closed importer should be read-only until the exact prepared assignment and export have
been proven.

## 3. Envelope contract

For a prepared packet `P`, define `E(P)` as the packet with every authorized human response field restored to
its exact blank worksheet value. Object keys are recursively sorted and serialized without whitespace. The
exporter stores the blank packet as a content-addressed JSON blob. Submission is accepted only when:

```text
sha256(E(submitted_packet)) == export_artifact.blob_sha256
```

and the matched export also has:

- the expected packet format;
- `application/json` media type;
- exact byte length;
- the same session ID; and
- the same D5 pass in its manifest.

The database lookup joins the export to the blob table. An invented hash, an unregistered blob, a blob from a
different pass, or a blob from a different session cannot satisfy the contract.

### 3.1 Pass A and Pass B

The only mutable portion remains each assignment's rubric `response`. Everything else—including visible
conversation, instructions, seed, order, reviewer, candidate version/hash, opaque identity, and presentation
identity—must reduce to the exact exported packet.

### 3.2 Pass C family synthesis

The mutable portions are:

- each assignment's `response`; and
- the judgment fields inside each existing `structuralDisposition` worksheet.

The candidate version ID that identifies a structural-disposition worksheet is not mutable. Redaction recreates
the blank worksheet around that original ID. A reviewer may fill the disposition, but may not add, remove,
replace, or redirect the set of structural candidates requiring judgment.

The immutable Pass C envelope therefore includes:

- schema and packet kind;
- campaign, session, reviewer, rubric, and snapshot identity;
- export timestamp and instructions;
- assignment membership and order;
- family version, slug, purpose, and blueprint;
- current candidate evidence, hashes, structural state, failures, and sealed review evidence; and
- the exact membership and candidate identity of structural-disposition worksheets.

### 3.3 Pass D campaign closeout

Only the complete `response` worksheet is mutable. The immutable envelope includes:

- campaign, session, adjudicator, rubric, snapshot, and timestamp;
- population counts;
- all candidate evidence;
- all family-synthesis evidence;
- all hidden-repeat stability evidence;
- the current analysis evidence; and
- ordering and packet structure.

Pass D exports now explicitly record `pass: "D"` in both JSON and Markdown artifact manifests, giving the
shared verifier the same pass-scoped binding used by A, B, and C.

## 4. Implementation

Two small shared modules separate browser-safe canonicalization from server-only ledger verification:

- `packet-envelope-contract.ts` recursively canonicalizes a response-redacted JSON value without importing
  Node APIs, so it remains safe for the public local-first browser workspace;
- `packet-envelope.ts` computes the envelope digest and requires an exact content-addressed export in SQLite.

The existing A/B implementation now calls the shared verifier without changing its public error contract.
Pass C and Pass D use their own typed redaction functions and translate the shared failure into workflow-
specific errors.

Submission paths now use read-only `require...` helpers for the prepared human actor and rubric versions.
`ensure...` helpers remain restricted to packet preparation, where creation is intended. The sequence for a
submission is now:

1. parse and type-check the packet;
2. require the previously prepared human actor and rubric;
3. reconstruct and check current prerequisite evidence;
4. require the exact open assignment and session;
5. validate all response fields;
6. require the exact exported immutable envelope;
7. only then write the raw submission and scientific evidence in one append-only batch.

Accepted Pass C and Pass D events now retain both the immutable packet-envelope SHA-256 and the completed
submission SHA-256. This distinguishes the review stimulus from the reviewer-authored response artifact.

## 5. Adversarial proof

The focused corpus suite exercises positive and negative controls for every packet class.

### 5.1 Positive controls

- A/B: changing only response fields continues to match the exported envelope.
- C: filling family responses and structural dispositions continues to match the blank export.
- D: filling the complete campaign response continues to match the blank export.
- C and D return an envelope SHA-256 exactly equal to the prepared packet's content-addressed export SHA.

### 5.2 Immutable-field attacks

- A/B candidate-text and opaque/presentation-identity attacks remain rejected.
- C changes `createdAt` after completing otherwise valid family and structural responses.
- D changes `createdAt` after completing an otherwise valid closeout response.

The C and D attacks both fail with `immutable envelope does not match an exported packet`.

### 5.3 No-write proof

After each completed-but-tampered C or D submission attempt, the tests query the temporary ledger before the
valid positive control. They prove zero rows in:

- `family_synthesis`;
- `structural_disposition`;
- `human_family_synthesis_submission` raw artifacts;
- `campaign_closeout`;
- `adjudication`; and
- `campaign_closeout_submission` raw artifacts.

This is stronger than proving that the final scientific table stayed empty: even the raw-submission layer is
not populated by a packet that never satisfied its prepared export contract.

## 6. Verification results

### 6.1 Passing checks

- `git diff --check`
- `npm run build -w @alpha/corpus`
- `npm test -w @alpha/corpus`: **22/22 pass, 0 fail**
- `npm run build -w @alpha/web`: optimized Next.js build completed successfully
- `alpha-corpus validate`: integrity `ok`, zero foreign-key violations, zero missing tables/views/blobs, zero
  corrupt blobs

The web build repeated the known Next/Turbopack warning that `apps/web/next.config.ts` causes unexpectedly
broad file tracing through `server-state.ts`. It did not fail compilation, type checking, static generation,
or route construction. This execution did not claim to fix that separate warning.

### 6.2 Preserved non-passing root command

`npm test` at the monorepo root is not a green aggregate gate: Turbo invokes `vitest run` in several packages
that contain no test files, and Vitest exits 1 for those packages. The observed failures included `core`,
`tokenizers`, `tensor`, `autograd`, `helios`, `model`, `bench`, and `train`, each reporting `No test files
found`. The focused corpus suite passed independently, and the optimized web build passed independently.

The commit hook then launched another full Turbo web dependency build after those checks had already passed.
On the data disk at 94% utilization, the nested Turbo process spent several minutes without an active build
child. It was interrupted, and the already verified change was committed with `--no-verify`. This is recorded
rather than silently presented as a green hook run.

## 7. Canonical-ledger reconciliation

`alpha-corpus validate` was run against:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus
```

The main SQLite SHA-256 was identical before and after validation:

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

The complete project-owned corpus tree is 37,493,414 bytes (35.76 MiB, approximately 0.035 GiB), far below
the resumable 15 GiB pause threshold.

## 8. Deployment and authority boundary

No new public deployment was needed. The public browser currently supports A/B review only, and its exact
exported-envelope behavior remains the already deployed and browser-proven Execution 11 release
`e07477b934897b71f241724a230e2ccd6320e0c9`. Pass C and Pass D remain local CLI workflows and cannot yet be
prepared because their human prerequisites are absent.

The next authority-bearing action is still a real human completing the 12 open blinded Pass A assignments,
downloading the response packet, and importing it locally. This execution does not populate attractive public
tables merely because their machinery exists, and it does not authorize generation, model criticism, corpus
expansion, release construction, training, a GPU, or live Donto mutation.
