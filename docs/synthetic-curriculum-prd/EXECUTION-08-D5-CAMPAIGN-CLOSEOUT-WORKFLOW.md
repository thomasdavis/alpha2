# Execution 08 — D5 campaign-closeout workflow

**Date:** 2026-07-31

**Scope:** implement the final D5 evidence-synthesis write path without performing or simulating the human
review it depends on

**Code revision:** `6cd4921f59e0fefc16dad95fe64e383928002607`

**Migration:** 7, `d5_campaign_closeout`

**Migration SHA-256:** `d64055b6b6f1572d2a3fd1bd2f57760b6b164fe3e77aaa76be04c9cb4b7ab275`

**Canonical ledger:**
`/mnt/donto-data/donto-resources/research/alpha2-corpus/alpha-corpus.sqlite`

**Public explorer:** `https://alpha.donto.org/corpus`

**Result:** implementation complete and live; human evidence absent; Pass D correctly fail-closed; no
generation, model-critic, release, training, GPU, or Donto execution authority created

## 1. Why this execution exists

PRD-12 defines four distinct D5 human-evidence layers:

1. blind conversational review in Pass A;
2. contract-aware candidate review in Pass B;
3. family synthesis and structural-rejection disposition in Pass C;
4. campaign-wide adjudication and diagnosis in Pass D.

Executions 03, 06, and 07 implemented the first three workflows and the hidden-repeat reliability substrate.
Before this execution, however, Pass D still existed only as a worksheet. That left an avoidable operational
gap: a future operator could complete A–C but then summarize the campaign in an unversioned Markdown note,
without exact links to the candidate, review, repeat, synthesis, structural, and analysis evidence on which the
summary depended.

Execution 08 closes that implementation gap. It does **not** close the human authority gate. It creates an
append-only, content-addressed campaign-closeout packet only after every prerequisite exists for the same
human adjudicator. Its submission can record scientific dispositions, failure clusters, distribution findings,
uncertainty, and recommended D5 states. It cannot change candidate lifecycle state, populate a release, create
a training exposure, authorize a model call, or start compute.

## 2. Exact authority boundary

The closeout is deliberately non-binding.

The packet requires the literal acknowledgement:

```text
non_binding_no_execution_authority
```

The database independently enforces:

```sql
execution_authorized INTEGER NOT NULL DEFAULT 0,
CHECK(execution_authorized = 0)
```

The application also writes `executionAuthority: false` into candidate-adjudication rationale and
`executionAuthorized: false` into the submission event. These are redundant controls on purpose: the human
form, executable validator, stored scientific record, and SQL schema all express the same boundary.

Pass D may recommend one or more evidence states:

- `D5_REPAIR_REQUIRED`;
- `D5_CRITIC_CALIBRATION_JUSTIFIED`;
- `D5_BATCHING_PROBE_JUSTIFIED`;
- `D5_EVALUATION_DESIGN_JUSTIFIED`;
- `D5_STOP`.

A state records what the evidence appears to justify discussing next. It is not a call budget, release policy,
training instruction, GPU authorization, or model-selection decision. A later bounded operator decision is
still required.

## 3. Fail-closed prerequisite order

`closeout-prepare` loads evidence in the order in which the human campaign must acquire it. For one declared
human adjudicator, it requires:

1. exactly one sealed Pass A review for every current candidate version;
2. exactly six completed hidden-repeat stability rows, or every candidate if the campaign has fewer than six;
3. exactly one sealed Pass B review for every current candidate version;
4. exactly one completed Pass C family synthesis for every current family version;
5. exactly one separate structural disposition for every structurally rejected candidate and none for a
   non-rejected candidate;
6. one current authoritative deterministic analysis run, excluding any run superseded by an
   `analysis_run_correction`.

The order matters operationally. A completely unreviewed campaign should report the missing Pass A gate, not
the first downstream empty table that happens to be queried. Tests now enforce this behavior.

The workflow does not accept approximate population coverage. It rejects missing or duplicate reviews,
family syntheses, structural dispositions, and repeat responses. It also rejects structural dispositions
attached to non-rejected candidates.

## 4. Migration 7 physical schema

Migration 7 adds seven relations.

### 4.1 `campaign_closeout_assignment`

This is the only mutable workflow row in the new group. It records:

- campaign;
- human adjudicator actor;
- exact rubric version;
- opaque session ID;
- frozen input-snapshot digest;
- `assigned` or `completed` status;
- creation and update times.

The unique campaign/adjudicator/rubric key permits safe packet resume. An existing open assignment can be
re-rendered only if its evidence digest is unchanged.

### 4.2 `campaign_closeout`

The append-only closeout head records:

- the exact assignment;
- campaign and adjudicator;
- recommendation summary;
- known, unknown, and proposed-next registers;
- preserved disagreements and the rationale for an empty disagreement set;
- overall rationale and confidence;
- exact submitted blob;
- the schema-enforced zero execution authority.

### 4.3 `campaign_closeout_state`

Stores each distinct recommended D5 state and its own rationale. Multiple compatible states can coexist; the
schema does not force one artificial winner.

### 4.4 `campaign_closeout_basis`

Links the closeout to exact evidence identifiers:

- candidate adjudications;
- family syntheses;
- structural dispositions;
- hidden-repeat response records;
- the authoritative analysis run.

Candidate adjudications have their own `adjudication_basis` rows for Pass A review, Pass B review, family
synthesis, and structural disposition where applicable. The two-level graph preserves campaign-level and
candidate-level provenance without copying the evidence into an unqueryable narrative.

### 4.5 `campaign_failure_cluster`

Stores the label, locus, severity, proposed repair, whether later calls might be needed, and rationale for each
diagnosed failure cluster. Allowed loci are:

- blueprint;
- realization;
- schema;
- style;
- review;
- source or authority;
- distribution.

Allowed call requirements are `no`, `possibly_later`, and `yes_if_separately_authorized`. None executes a
call.

### 4.6 `campaign_failure_cluster_member`

Links a cluster to valid frozen evidence of one of these kinds:

- candidate version;
- family version;
- review;
- family synthesis;
- structural disposition.

The executable packet validator rejects unsupported, unknown, or duplicate members.

### 4.7 `campaign_distribution_assessment`

Requires one assessment for every declared conversational-distribution dimension:

- first-sentence directness;
- question behavior;
- length appropriateness;
- lecture drift;
- canned signatures;
- multi-turn reuse;
- desire to continue;
- substantive value after style scrubbing.

Each assessment can cite exact IDs from the frozen evidence set. Evidence lists may be empty only when the
text explicitly says the available campaign cannot support a stronger finding.

## 5. Append-only and transaction behavior

The six scientific tables after the assignment table receive update and delete rejection triggers:

- `campaign_closeout`;
- `campaign_closeout_state`;
- `campaign_closeout_basis`;
- `campaign_failure_cluster`;
- `campaign_failure_cluster_member`;
- `campaign_distribution_assessment`.

That adds 12 triggers. Migration 7 takes the canonical ledger from 122 to 129 tables and from 174 to 186
triggers. The five existing public/current views are unchanged.

A valid submission is written as one database batch after validation. It creates:

- the submitted content-addressed blob and raw-artifact link;
- one campaign closeout;
- one candidate adjudication per frozen candidate;
- exact adjudication bases;
- repair requests only where the disposition or entered response requires them;
- disagreement cases where disagreement is declared;
- closeout evidence bases;
- recommended D5 states;
- failure clusters and their members;
- all eight distribution assessments;
- a completion update to the workflow assignment;
- an append-only `campaign_closeout_submitted` event.

It does not create:

- a `quality_state_transition`;
- a `release_member`;
- a `rendered_unit`;
- a `training_exposure`;
- a model-call task;
- an execution authorization.

Candidate adjudication is therefore evidence about disposition, not lifecycle mutation. Even
`accept_as_positive` remains a quarantined calibration judgment until an independently versioned release
policy acts under new authority.

## 6. Frozen packet and tamper checks

The packet contains the complete evidence needed for human synthesis:

- current candidate and family version IDs;
- candidate content hashes and structural status;
- full A and B outcome, rationale, dimension-score, finding, and timestamp evidence;
- family synthesis and family disposition;
- structural disposition for each retained rejection;
- repeat response and stability measurements;
- current analysis-run identifier and output counts;
- population totals;
- an exact input-snapshot SHA-256.

Submission recomputes the current evidence from SQLite and separately checks:

- candidate count;
- family count;
- structurally rejected count;
- completed repeat count;
- expected repeat count;
- every packet evidence object;
- the assignment session and snapshot digest;
- the rubric identity and version.

Changing a candidate status, evidence object, population count, session, or snapshot after preparation causes
submission to fail before any closeout records are written. Resuming an open packet is allowed only when the
existing file is byte-identical to the regenerated packet; an edited file is never silently overwritten.

## 7. Response validation

The executable validator requires:

- one PRD-04 disposition for every frozen candidate and no extras;
- rationale and confidence for every candidate;
- a repair request for repair-oriented outcomes;
- an explicit disagreement description for `defer_theory_disagreement`;
- unique failure-cluster keys with valid locus, severity, call requirement, and evidence members;
- an explicit rationale when no failure clusters are found;
- all eight distribution dimensions exactly once;
- valid, unique recommended D5 states with separate rationales;
- nonempty known, unknown, and proposed-next registers;
- preserved disagreements or an explicit reason none were found;
- overall rationale and confidence;
- the non-binding authority acknowledgement.

This is structural validation of a human synthesis. It does not prove the synthesis philosophically correct.
The submitted human identity, rationale, uncertainty, and disagreement remain visible evidence that later
reviewers may challenge rather than an invisible source of truth.

## 8. Commands

The local authenticated workflow is:

```bash
npm run corpus -- closeout-status
npm run corpus -- closeout-prepare --adjudicator ajax
npm run corpus -- closeout-submit --file /absolute/path/to/campaign-closeout-packet.json
```

`closeout-prepare` and `closeout-submit` operate only against the local canonical ledger. The public explorer
has no closeout POST route and remains read-only.

## 9. Test evidence

The corpus package passed 20 of 20 tests after this implementation.

The new closeout tests prove:

- an incomplete campaign reports the missing sealed Pass A review;
- no closeout assignment is created on that failed preparation;
- a fully controlled fixture can complete A, hidden repeats, B, C, structural disposition, and analysis;
- packet preparation is safely resumable;
- candidate-evidence tampering is rejected;
- population-accounting tampering is rejected;
- an incomplete response is rejected;
- a valid submission records two candidate adjudications with seven exact basis rows;
- one failure cluster, one member, one recommended state, and eight distribution assessments are preserved;
- candidate structural statuses remain unchanged;
- quality-state transitions, release members, and training exposures remain zero;
- execution authorization remains zero;
- scientific closeout records reject update and delete;
- the same packet cannot be submitted twice after its assignment closes.

The full dependent web build also passed in the commit hook before revision `6cd4921` was pushed.

## 10. Canonical migration evidence

Before applying migration 7, SQLite's online backup mechanism produced:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/
pre-d5-campaign-closeout-20260731T031620Z.sqlite
```

Backup SHA-256:

```text
ad4afb5622f30adca5c00df4b2425805bfe5235c250c8cc11f097f267002d5e3
```

The backup passed `PRAGMA integrity_check` and `PRAGMA foreign_key_check`. The migrated canonical ledger then
reported:

| Check | Result |
|---|---:|
| migrations | 7 |
| tables | 129 |
| views | 5 |
| triggers | 186 |
| integrity | `ok` |
| foreign-key violations | 0 |
| missing required tables | 0 |
| missing required views | 0 |
| missing blobs | 0 |
| corrupt blobs | 0 |
| project-owned footprint | 35.39 MiB |
| soft-pause threshold | 15 GiB |

Every earlier migration digest remained unchanged. The applied migration-7 row exactly matches the executable
digest stated at the start of this record.

## 11. Live fail-closed proof

After migration, the canonical campaign reported:

```text
campaign_closeout_assignment   0
campaign_closeout              0
campaign_closeout_state        0
campaign_closeout_basis        0
campaign_failure_cluster       0
campaign_failure_cluster_member 0
campaign_distribution_assessment 0
adjudication                   0
release_member                 0
training_exposure              0
```

Running the real preparation command for reviewer alias `ajax` failed with:

```text
Candidate candidatev_ce14fa164b51a123f86ce84085063e94 needs exactly one sealed Pass A review before Pass D
```

The failure created zero closeout assignments. This is the desired current result. There are 12 open Pass A
assignments and no human responses; manufacturing downstream records would have violated the review protocol.

## 12. Public explorer proof

The generic live-schema explorer discovered the new relations automatically. Each returned HTTP 200:

- `https://alpha.donto.org/corpus?relation=campaign_closeout_assignment`
- `https://alpha.donto.org/corpus?relation=campaign_closeout`
- `https://alpha.donto.org/corpus?relation=campaign_closeout_state`
- `https://alpha.donto.org/corpus?relation=campaign_closeout_basis`
- `https://alpha.donto.org/corpus?relation=campaign_failure_cluster`
- `https://alpha.donto.org/corpus?relation=campaign_failure_cluster_member`
- `https://alpha.donto.org/corpus?relation=campaign_distribution_assessment`

The public rows are empty because human evidence is absent, not because the routes are placeholders. Once a
real local submission occurs, the same read-only pages will expose the appended scientific record.

## 13. Recurring progress report

The authorized two-hour factual Discord reporter now reads Pass D directly from SQLite. It reports:

- assigned and completed closeout workflows;
- closeout record count;
- candidate adjudication count;
- recommended state counts;
- nonzero execution-authorization anomalies;
- the true next gate.

The reporter explicitly states that closeout is evidence, not permission to generate or train. Its current
next gate remains blinded human Pass A, not Pass D.

## 14. Limitations and remaining human gate

This workflow does not establish:

- that any of the 48 candidates is conceptually correct;
- that any structurally valid candidate is acceptable training data;
- that the human reviewer is reliable;
- that hidden-repeat stability will be high;
- that GPT-5.4 should be used for production generation;
- that a critic, batching probe, evaluation design, corpus expansion, release, or training run is justified;
- that a synthetic-only curriculum will make Alpha chatty.

Those are empirical questions the larger PRD program exists to answer.

The immediate authority-bearing action remains a real human completion of the existing 12-item blinded Pass A
packet at `https://alpha.donto.org/corpus/review`, followed by local `review-submit`. Hidden contracts and
general lineage should not be inspected by the reviewer before that packet is sealed. After Pass A, the
scheduler can create the first legitimate hidden-repeat presentations; only then may Pass B, Pass C, and Pass
D accumulate their own evidence.

The implementation is complete. The adjudication is not.
