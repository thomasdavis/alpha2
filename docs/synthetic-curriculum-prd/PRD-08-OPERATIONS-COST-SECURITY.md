# PRD-08 — Operations, cost, security, and recovery

## 1. Purpose

This PRD keeps the synthetic-data program affordable, resumable, auditable, and safe on the operator's existing
box and one-GPU workflow. It separates subscription/model-call economics from GPU training economics and
prevents “cheap generation” from becoming expensive through waste, repetition, or review backlog.

No operational action is authorized merely because it appears here.

## 2. Resource domains

### 2.1 Local research host

The Alpha repository remains at `/mnt/donto-data/workspace/alpha2`. Large generated corpora, ledger snapshots,
blobs, exports, logs, and checkpoints belong on the mounted data drive, not the small root disk.

### 2.2 Model-generation capacity

Prefer already-paid subscription capacity where allowed and reliable. The provider/model registry records
quota condition, terms, rate limits, and exact model identity. No campaign assumes an unlimited API.

### 2.3 Human attention

Human review is the scarcest high-authority resource. It receives its own budget, queue, sampling strategy,
and backlog ceiling.

### 2.4 GPU training

RunPod or equivalent GPU time is bounded separately. No persistent paid pod is kept idle. Training artifacts
must be mirrored before termination.

## 3. Cost governance

Every campaign contract declares:

- objective and release allocation;
- maximum families, calls, tokens, estimated spend, subscription quota, and wall time;
- maximum strong-orchestrator calls;
- expected worker/critic routing;
- human-review capacity;
- stop and escalation thresholds;
- storage estimate;
- named owner;
- authorization record.

Unbounded “keep generating” jobs are prohibited.

## 4. Cost optimization hierarchy

Optimize in this order:

1. eliminate repeated planning through durable blueprints and ledger state;
2. generate structure before surfaces so flawed families stop early;
3. send workers only the smallest necessary context;
4. reuse cached prompt prefixes where provider support and terms allow;
5. batch homogeneous bounded tasks;
6. route by calibrated task competence rather than prestige;
7. run deterministic filters before model critics;
8. review at family and batch level, then sample realizations;
9. repair only localized defects;
10. stop expansion when semantic novelty or quality yield falls;
11. reserve the strongest model for decisions that affect many descendants;
12. do not generate rows merely to meet a round-number target.

## 5. Unit economics dashboard

For each model/task/prompt combination report:

- attempts and successful structured responses;
- structurally valid candidates;
- accepted, repaired, rejected, disputed, and pending candidates;
- unique accepted families/edges/projections;
- input/output/cached tokens;
- monetary or quota estimate;
- strong-model calls;
- human-review minutes;
- cost per structurally valid candidate;
- cost per accepted candidate;
- cost per accepted family edge;
- duplicate-adjusted yield;
- downstream behavioral contribution when measured.

The system should surface marginal curves, not just cumulative totals.

## 6. Scheduling and concurrency

- use resumable bounded queues;
- serialize calls where one subscription/session gateway cannot safely serve producers and critics together;
- apply provider-specific concurrency limits;
- back off on empty responses, throttling, or degraded quality;
- never score an empty model response as a substantive rejection or wrong answer;
- distinguish transient call failure from valid zero-output tasks;
- prevent self-matching process monitors;
- verify real row/artifact progress over a wall-clock interval;
- monitor disk, RAM, swap, CPU load, and database contention;
- pause production before a review backlog becomes unmanageable.

## 7. Generation run directory

Each campaign has an append-only run directory on the mounted data drive containing:

- campaign contract;
- database snapshot or pointer;
- exact software revision;
- provider/model registry snapshot;
- environment summary excluding secrets;
- structured event log;
- raw call artifacts not yet committed to the ledger;
- progress and resource snapshots;
- validation reports;
- recovery instructions;
- completion/termination record.

The database remains canonical once ingestion is verified. Raw run artifacts provide independent recovery
evidence.

## 8. Crash and resume behavior

Every task has an idempotency key derived from campaign, blueprint/version, requested operation, and recipe.
Task state distinguishes planned, leased, started, response-received, validated, ledger-committed, and failed.

After interruption:

1. verify database integrity;
2. reconcile task events with raw artifacts;
3. mark expired leases;
4. ingest any response-received but uncommitted artifact exactly once;
5. resume only pending/retry-eligible tasks;
6. recheck provider/quota/model identity;
7. take a fresh real-progress sample after launch.

Never infer success solely because a process exists or a queue row says `done`.

## 9. Storage admission

Before a campaign, estimate:

- raw prompt/response bytes;
- normalized message content;
- embeddings and similarity indexes;
- SQLite indexes and WAL growth;
- rendered exports;
- token artifacts;
- snapshots;
- model checkpoints;
- temporary duplication during release build.

Require headroom for the campaign, one safe snapshot, and release materialization. Monitor the mounted disk;
past disk saturation on the box makes low-space operation unacceptable.

Content-addressed deduplication may save physical bytes without deleting separate provenance records.

## 10. Backup and preservation

- sealed ledger snapshots receive digests and at least two verified storage locations before public release;
- campaign work is recoverable from local ledger plus raw event artifacts;
- checkpoints are mirrored incrementally during paid GPU execution;
- Git tracks PRDs, schemas, migrations, prompts, tools, manifests, and small reports—not giant corpus blobs;
- external artifact locations are digest-verified;
- backup policy must align with the operator's existing OVH image backup decision and must not add an
  unauthorized heavy local dump cron;
- restoration is periodically tested on a copy.

## 11. Secret management

Provider credentials, Hugging Face tokens, SSH keys, and the Discord webhook:

- never enter Git, SQLite public snapshots, prompts, logs, screenshots, or dataset cards;
- remain in existing approved mode-restricted stores;
- are referenced by non-secret handles;
- are read only by the task that needs them;
- are redacted from exception output;
- are rotated after suspected exposure.

The Discord webhook supplied historically is considered sensitive. It is not read or used for routine
progress. A future post must pass the qualitative-improvement-only policy: same input, before/after output,
why the model improved, and an honest aggregate boundary.

## 12. Provider and model change control

A model alias changing its underlying revision triggers:

- registry update;
- small calibration batch;
- comparison of quality, style, schema compliance, and cost;
- critic calibration if used for judgment;
- decision record before production routing.

A provider returning HTTP success with empty content is a failure. Retry with bounded backoff; do not record an
empty candidate as a meaningful output.

## 13. Structured-output safety

Use provider-supported schema output or tool calling. If schema validation fails:

- retain raw response;
- record failed attempt;
- retry only under campaign policy;
- optionally escalate;
- never regex a plausible JSON object out of prose;
- never partially commit an object as if complete.

## 14. RunPod/GPU protocol

Any later paid run must:

1. cite an authorized experiment and hard budget;
2. resolve exact pod ID, GPU, price, volume billing, and termination semantics;
3. bootstrap from a pinned script/revision;
4. prove real NVIDIA execution and fail-closed no-fallback behavior;
5. run the bounded smoke gate first;
6. monitor actual steps/tokens/checkpoint growth plus CPU/RAM/swap/disk;
7. mirror and hash checkpoints and metrics while running;
8. stop on instability, futility, or budget threshold;
9. pull final evidence;
10. terminate, not merely stop, the exact billable resource;
11. verify provider state shows it gone.

The historical Alpha RunPod recipe remains in the archive. It is not authorization to create a pod.

## 15. Observability

Required live signals:

- task completion delta over time;
- accepted/rejected/pending counts;
- provider latency, empty/error/rate-limit rate;
- tokens/quota/cost estimate;
- review backlog;
- duplicate-adjusted novelty;
- SQLite WAL/file growth and write latency;
- disk free space;
- RAM, swap, load;
- process liveness and exact identity;
- training GPU utilization, step advance, nonfinite count, checkpoint mirror state;
- gate status and current authorized boundary.

Dashboards are derived and may be regenerated. Raw events remain canonical.

## 16. Data security and privacy

The primary synthetic-only training corpus should contain no private human conversations. When later human
evaluation is authorized:

- collect only necessary data;
- use explicit consent and retention terms;
- separate identity from conversation content;
- restrict raw access;
- store consent and release scope;
- allow true deletion/tombstoning where legally or ethically required;
- prevent human-eval text from silently entering training;
- review model outputs for personal-data invention.

## 17. Cultural safety

- model-generated community language is labeled synthetic;
- restricted or sacred knowledge is not fabricated as authentic;
- community authority requirements can block release/training while preserving metadata;
- disagreement about categories is stored rather than resolved by generic model vote;
- multilingual expansion requires language-competent review and appropriate governance.

## 18. Public communication

Progress reports distinguish:

- documentation complete;
- system implemented;
- pilot generated;
- batch reviewed;
- release sealed;
- training launched;
- model passed or failed;
- artifact published.

No stage is described as the next. Discord receives only meaningful qualitative improvement under its binding
contract. Repository documentation and sealed reports are the ordinary progress channel.

## 19. Incident classes

- secret exposure;
- data corruption or missing blobs;
- split contamination;
- provider model drift;
- empty-response cascade;
- judge calibration failure;
- source/license objection;
- cultural authority objection;
- disk saturation;
- stalled queue or training job;
- accidental external-data inclusion;
- runaway token/quota/GPU spend;
- mistaken publication of private evaluation.

Each incident creates an append-only record, contains the affected campaign/release, preserves evidence, and
requires revalidation before resumption.

## 20. Acceptance criteria

Operations are ready only when:

- campaigns are hard-bounded and stoppable;
- quotas and costs are visible before and during execution;
- strong-model use is auditable and amortized;
- interrupted batches resume exactly once;
- real progress and resource health are verified;
- database and artifacts survive crash/restart tests;
- secrets cannot enter public artifacts;
- storage admission protects the mounted drive;
- paid GPU lifecycle includes verified termination;
- public communication cannot mistake preparation for completion;
- no action begins without the matching dated authorization.
