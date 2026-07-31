# Execution 05 — D5 first-class surface and distribution evidence

**Date:** 2026-07-31

**Status:** implemented, tested, recorded in the canonical ledger, and publicly verified; human quality
review remains pending

**Scope:** turn deterministic D5 distribution, similarity, and template diagnostics into versioned,
append-only, queryable scientific records without treating them as semantic or human judgments

**Implementation revisions:**

- `1199a731fd9d3eae22284dc68c1c465786b4690e` — first-class deterministic surface analysis;
- `002e123f71cdc30ef883b6fb6222a47240578001` — verified Git provenance and append-only run corrections.

**Canonical ledger:**
`/mnt/donto-data/donto-resources/research/alpha2-corpus/alpha-corpus.sqlite`

**Public explorer:** `https://alpha.donto.org/corpus`

## 1. Outcome

The frozen 48-candidate D5 calibration now has a standalone, content-addressed analysis document and five
first-class evidence tables:

| Table | Rows in current ledger | Meaning |
|---|---:|---|
| `analysis_method` | 1 | Immutable definition and configuration of the profiler |
| `analysis_run` | 2 | One provenance-erroneous run plus its corrected replacement |
| `analysis_metric` | 472 | 236 metrics per preserved run |
| `similarity_edge` | 4,512 | 2,256 pair/method measurements per preserved run |
| `template_signature` | 976 | 488 dynamically discovered signatures per preserved run |
| `analysis_run_correction` | 1 | Typed append-only link from the erroneous run to the corrected run |

The authoritative corrected run is:

```text
run: analysis_run_a32f01644e1a96a9ea601b35b35626aa
software revision: 002e123f71cdc30ef883b6fb6222a47240578001
input snapshot: 8ce2bf64152c674f17b707137774ad7be42eb0ad5c2a51ec3953a196f75ac03e
output blob: 20f4191a9968743aa4aa329dbdce834d6c8ab6cbfb392b7168c429e459b0456e
metrics: 236
similarity edges: 2,256
template signatures: 488
```

The run is deliberately marked `surface_distribution_only`. Its stored disclaimer says that a flagged pair
is not a semantic duplicate judgment, structural validity is not human acceptance, and analysis cannot
promote candidates, create release membership, or authorize training exposure.

## 2. Version policy and corrected analysis bug

The earlier ad hoc analyzer joined every `candidate_version` row. That would have counted one logical
candidate more than once as soon as a repair created version 2. The profiler now reads
`corpus_candidate_current`, so its statistical population is exactly one latest version per candidate.

The same fix was applied to audit-packet export. Historical candidate versions remain in SQLite; they are
excluded from a current-snapshot analysis rather than deleted.

The profile input snapshot records, for every included candidate:

- stable candidate identity;
- exact candidate-version identity and version number;
- content SHA-256;
- family identity and slug; and
- current structural state.

The canonical JSON of that population is itself hashed. Any candidate repair, status change, family change,
or content change therefore creates a new input snapshot and a new analysis-run identity.

## 3. Deterministic method

Method `deterministic-surface-distribution-profile`, version 1, records:

1. campaign- and family-scoped counts, rates, word-length distributions, kinds, declared difficulties, and
   intended response policies;
2. candidate-level assistant word 3-gram Jaccard for all candidate pairs;
3. candidate-level assistant character 5-gram Jaccard for all candidate pairs; and
4. normalized assistant word n-grams of widths 2 through 6 that appear in at least two distinct candidates.

There are 48 current candidates, so each similarity method produces exactly
`48 × 47 / 2 = 1,128` edges, or 2,256 total. Every score is stored, not merely threshold hits. A score at or
above 0.70 receives `surface_review_candidate`; all others receive `not_flagged`. That label nominates human
inspection only.

Template signatures are discovered from the data rather than a hand-maintained phrase blacklist. The run
retains the 250 highest-ranked campaign signatures and up to 50 per family, ordered by distinct-candidate
frequency, longer n-gram, and lexical tie-break. This is a surface regularity instrument, not a claim that a
teacher template caused the phrase.

## 4. Measured D5 profile

The corrected run reports:

| Measurement | Value |
|---|---:|
| current candidates | 48 |
| structurally valid | 42 |
| structurally rejected | 6 |
| structural yield | 87.5% |
| assistant messages | 78 |
| multi-turn candidates | 22 |
| assistant question endings | 3 / 78 (3.846%) |
| assistant words, mean / median / p90 / max | 33.78 / 32 / 54 / 70 |
| excess normalized exact assistant duplicates | 0 |
| word 3-gram pairs at or above 0.70 | 0 / 1,128 |
| maximum word 3-gram Jaccard | 0.063492 |
| character 5-gram pairs at or above 0.70 | 0 / 1,128 |
| maximum character 5-gram Jaccard | 0.211111 |

The most frequent discovered campaign signature is the ordinary bigram `the same`, present in 21 of 48
candidates. `the same person` occurs in six. These are useful review signals because the calibration contains
several identity-through-change families; they are not evidence that those items are duplicates or poor.

The profile also exposes a design issue for later taxonomy work: nearly every generated
`intendedResponsePolicy` is free-form prose rather than a controlled operational category. The ledger keeps
that exact prose. A future approved schema revision should relate it to a separately versioned response-policy
taxonomy without replacing the raw instruction.

## 5. Provenance failure and append-only correction

The first canonical invocation supplied this plausible-looking but incorrect full revision:

```text
1199a73dc1ec3f3b415fc60dc2f7682bd6b5df42
```

It began with the correct short hash but was not the repository's actual commit. The resulting run and output
were not deleted or rewritten:

```text
erroneous run: analysis_run_e6e5e45da332d9482be1ed7ca586780f
erroneous output: aa3c14a28ed3003cd1e67fcc7da998a5395d3f601c038d511a5fc46e5e869088
correction: analysis_run_correction_d74d355ddcfc337474809c4dd1b618e0
replacement: analysis_run_a32f01644e1a96a9ea601b35b35626aa
```

Migration 4 introduced `analysis_run_correction`, and an event records the same correction. The CLI now runs
`git rev-parse HEAD` itself and rejects a supplied `--revision` unless it exactly matches the current full
HEAD. This incident is part of the scientific record and proves why append-only correction is preferable to
quiet cleanup.

## 6. Schema and immutability

Migrations 3 and 4 are additive. The already-applied migration 1 and 2 digests were compared before canonical
migration and remained exact:

```text
1 c9fc33838e1d833e8667ebaf19295b0cfadaf6faef63a63d292d719fdb0f3094
2 7f3963528b015eb9771b066ff07a5ae6cc50f1b08e0058486399e1555dcc66ce
```

All six new tables reject `UPDATE` and `DELETE`. Similarity rows constrain score and threshold to `[0,1]`,
order pair endpoints canonically, and prevent duplicate method edges. Metric rows require exactly one numeric
or textual value. Analysis runs require completed state and the explicit surface-evidence scope.

The corpus test suite now passes **15/15**. The new regression proves:

- only current candidate versions are counted;
- all pair/method edges are materialized;
- rerunning the same snapshot, method, revision, and environment is idempotent;
- a correction links two immutable runs and is itself immutable;
- neither profiling nor correction creates a release member or training exposure; and
- the ledger remains foreign-key clean.

## 7. Canonical backup and validation

Before migration, SQLite's backup API created:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/backups/
  pre-d5-surface-profile-20260731T021439Z.sqlite
SHA-256: 7211009606bc9f469ddb4d6d428c86791160ba0c29039d5c4b7239bfbc0121e5
size: 1,794,048 bytes
```

Post-execution validation reports:

- SQLite integrity: `ok`;
- foreign-key violations: `0`;
- missing tables/views/blobs: `0`;
- corrupt blobs: `0`;
- migrations: `4`;
- physical schema: 112 tables, four views, 160 append-only triggers;
- validator-owned artifact footprint: 18.46 MiB.

One validation command was initially launched concurrently with an idempotence replay and received
`SQLITE_BUSY`. The commands were then serialized; the replay resumed the same run and the isolated validation
passed. No scientific row was lost or partially applied.

The tree remains vastly below the operator's 15 GiB pause threshold.

## 8. Public proof

The existing read-only explorer discovered the additive migration without a web redeploy. Public requests
returned HTTP 200 for:

```text
https://alpha.donto.org/corpus?relation=analysis_run
https://alpha.donto.org/corpus?relation=analysis_metric
https://alpha.donto.org/corpus?relation=analysis_run_correction
```

The corrected run ID and `surface_distribution_only` scope were present in the public HTML. The service was
active with zero automatic restarts. Public access remains read-only.

## 9. Scientific state after analysis

| State | Count |
|---|---:|
| candidates | 48 |
| candidate versions | 48 |
| open Pass A assignments | 12 |
| completed assignments | 0 |
| reviews | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |

The profile makes the frozen calibration more inspectable. It does not answer whether any candidate is
conceptually correct, natural, useful, diverse in meaning, or suitable for training.

## 10. Next gate

The authority-dependent next step remains the real human Pass A session at
`https://alpha.donto.org/corpus/review`. Complete and download the 12-item blinded packet, then import it with
the local `review-submit` command. Pass B, family synthesis, adjudication, and any new generation remain behind
their existing gates.

No model call, critic call, synthetic expansion, evaluation construction, training run, GPU provision, live
Donto mutation, dataset release, or public write surface was used in this execution.
