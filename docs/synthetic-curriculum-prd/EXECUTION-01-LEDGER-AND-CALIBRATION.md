# Execution 01 — scientific ledger and GPT-5.4 calibration

**Executed:** 2026-07-30

**Scope:** PRD-09 stages D2–D4 plus the generation portion of D5

**Current gate:** D5 human conceptual adjudication pending

**Training status:** not authorized and not performed

**GPU status:** no Alpha GPU or RunPod action

**Donto status:** no live Donto mutation

## 1. Outcome

The first bounded Alpha Corpus vertical slice is operational. It includes a versioned SQLite scientific
ledger, content-addressed raw artifacts, an extensible Donto-derived category system, six hand-authored canary
families, schema-constrained Codex generation, deterministic validation, idempotent tasks, recovery of a
completed-but-uncommitted call, campaign analysis, and a complete human-audit packet.

The bounded calibration used GPT-5.4 for surface generation. It produced 48 candidates across six independent
families in exactly 12 calls. Forty-two candidates passed deterministic structural validation and six were
retained as structural rejections. The rejection cause was consistent: GPT-5.4 put a transformation slug into
`secondaryLenses`. No rejected record or raw response was deleted or silently repaired.

This is not a training release. It is a canary population awaiting human conceptual adjudication. The
public-training view contains zero rows.

## 2. Direct operator decisions applied

The bounded execution followed these operator amendments to the planning suite:

- GPT-5.6-sol provides high-leverage counsel and orchestration reasoning.
- GPT-5.4 performs initial synthetic sentence and dialogue generation.
- GPT-5.5 is disabled by default. It is used only if a paired, task-specific probe shows a concrete GPT-5.4
  failure that justifies escalation.
- The program tracks its own storage tree. If that tree exceeds 15 GiB, new corpus work pauses in a resumable
  state.
- The 15 GiB condition is not a global disk rule. It does not inspect, delete, move, or gate unrelated data.
- Corpus and research artifacts live on the mounted-drive research hierarchy.
- No model training or GPU spend belongs to this bounded stage.

GPT-5.6-sol counsel recommended a vertical slice in the order ledger kernel, canary families, offline mocked
orchestration, then bounded GPT-5.4 calibration. It also recommended keeping GPT-5.5 off until a specific
failure warrants a paired comparison. The implementation followed that advice.

## 3. Artifact locations

### 3.1 Repository implementation

The implementation is the TypeScript workspace package:

```text
packages/corpus/
  src/schema.ts       versioned migrations and append-only triggers
  src/db.ts           ledger writes, validation, call and candidate lineage
  src/seeds.ts        category, transformation, and canary-family seeds
  src/schemas.ts      provider-enforced JSON Schemas
  src/codex.ts        bounded Codex CLI transport and orphan recovery
  src/prompts.ts      natural-dialogue and contrast/repair recipes
  src/generate.ts     idempotent calibration campaign
  src/validate.ts     deterministic candidate validation
  src/analysis.ts     distribution and style diagnostics
  src/report.ts       human-audit packet materialization
  src/cli.ts          operator interface
  src/corpus.test.ts  offline proof suite
```

Root commands:

```bash
npm run corpus -- init
npm run corpus -- validate
npm run corpus -- plan --families 6 --items-per-call 4
npm run corpus -- generate --execute --families 6 --items-per-call 4 --model gpt-5.4
npm run corpus -- status
npm run corpus -- analyze
npm run corpus -- audit
```

`generate` is a dry-run without `--execute`.

### 3.2 Canonical mounted-drive ledger

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/
  alpha-corpus.sqlite
  blobs/sha256/
  calls/
  releases/audit/
```

The location may be overridden through `--home` or `ALPHA_CORPUS_HOME`. Corpus bytes are deliberately absent
from Git.

### 3.3 Human-audit packet

The latest complete packet at execution close is:

```text
/mnt/donto-data/donto-resources/research/alpha2-corpus/releases/audit/
  alpha-calibration-v1-2026-07-30T08-51-22-977Z/
    README.md
    audit-packet.json
```

The Markdown file is optimized for direct reading. The JSON file includes every candidate, hidden contract,
status, finding, campaign statistic, and deterministic analysis result.

## 4. Ledger architecture implemented

Migration 1 creates the scientific object graph. At execution close it contains 106 non-internal tables and
148 append-only update/delete triggers. The schema covers:

- programs, versions, decisions, gates, and actors;
- content-addressed blobs, locations, and artifact relations;
- categories, category versions, relations, annotations, disagreements, and open-lens proposals;
- concept families, projections, competency questions, scenes, trajectories, branches, transformations, and
  shortcut hazards;
- semantic states, commitments, dependencies, deltas, and admissible-analysis sets;
- sources, source versions, fragments, and evidence anchors;
- dialogues, natural messages, participants, local terms, utterance relations, response policies, and state
  links;
- providers, model aliases/revisions, prompt versions, tool schemas, recipes, and software revisions;
- campaigns, batches, tasks, attempts, calls, messages, tools, usage, routing, budgets, raw artifacts, and
  generation events;
- candidate versions, parents, failures, quality transitions, reviews, rubric versions, adjudications,
  disagreement cases, and repairs;
- cohorts, releases, exclusions, renderers, rendered units, token/loss-mask mappings, exports, validations,
  training exposure, and evaluation output;
- append-only generic events and validation runs.

Migration 2 adds four stable read surfaces:

- `corpus_candidate_current` — current version of every candidate, including quarantine states;
- `public_training_candidate` — only human-accepted candidates and no raw call internals;
- `campaign_progress` — task, call, yield, and human-acceptance reconciliation;
- `candidate_review_state` — current review/adjudication coverage.

At close, `public_training_candidate` contains zero rows. Structural validity is intentionally insufficient for
training eligibility.

## 5. Seed ontology and family population

The ledger contains 49 initial category seeds. These include Donto-style analytic lenses and dialogue-native
extensions:

- taxonomy, mereology, identity/persistence, topology, time, causation, teleology, agency, epistemology,
  deontology, axiology, modality, qualia structure, lexical semantics, social ontology, events, constitution,
  grounding, provenance, alignment, measurement, disposition, speech acts, phenomenology, and an open lens;
- pragmatics, discourse, common ground, inferential conceptual pacts, metalinguistic negotiation, intent,
  argumentation, hermeneutics, rhetoric, semiotics, narrative, translation, standpoint/authority, power,
  emotion/stance, salience, absence/negation/unknown, granularity, analogy, counterfactuals, conceptual change,
  pedagogy, conversational ethics, and answer-and-stop behavior.

The ledger contains 16 typed transformations, including paraphrase, irrelevant detail, minimal meaning change,
evidence addition/withdrawal, temporal/perspective/purpose/granularity shifts, counterexample, local repair,
delayed reuse, cross-projection, false bridge, clarification, and answer-and-stop.

The six first canary families are:

1. absence, explicit negative, unknown, and inapplicable;
2. intent, speech act, and effect;
3. part, member, material, and containment;
4. purpose, function, use, and effect;
5. role and bearer;
6. source, report, evidence, and endorsement.

All are tagged `canary` and split `quarantine`. They are not evaluation or training families.

## 6. Model routing actually used

### 6.1 Counsel

GPT-5.6-sol was used once as architectural counsel through the research-agent workflow. Its advice affected the
sequence, model split, storage behavior, preservation rules, and decision not to invoke GPT-5.5.

### 6.2 Worker

GPT-5.4 generated all 48 surface candidates. Calls used:

- the authenticated Codex CLI subscription transport;
- an explicit `gpt-5.4` model selection;
- medium reasoning effort;
- read-only sandboxing;
- ephemeral sessions;
- provider-enforced `--output-schema`;
- JSONL event output plus an exact structured last-message file;
- no free-text JSON extraction;
- serialized calls to avoid subscription-gateway contention.

Every raw prompt, schema, command, event stream, stderr stream, and structured last message exists in both the
run directory and the content-addressed ledger.

### 6.3 Critic

GPT-5.5 was not called. There is no critic result and no implied model acceptance. Human review remains the
next authority gate.

## 7. Calibration design

Each family received two prompt recipes:

1. `natural-dialogue` — answer-and-stop, compact examples, sustained local meaning, and natural variation;
2. `contrast-and-repair` — hard negatives, false bridges, minimal changes, clarification policy, and local
   rather than wholesale revision.

Each call requested four items, producing eight per family and 48 total. The requested item population mixed:

- micro-dialogues;
- multi-turn dialogues;
- linguistic pairs embedded in short natural exchanges;
- introductory, intermediate, and advanced difficulty;
- compact answers, examples, clarifications, premise challenges, plurality preservation, and answer-and-stop.

Research metadata and model-visible text are separate. Natural messages store roles in normalized columns and
contain no injected `<assistant>`-style delimiters. Export-time rendering remains a later operation.

## 8. Results

### 8.1 Population and structural yield

| Measure | Result |
|---|---:|
| Families | 6 |
| Prompt recipes per family | 2 |
| GPT-5.4 calls | 12 |
| Candidates | 48 |
| Structurally valid | 42 |
| Structurally rejected | 6 |
| Structural yield | 87.5% |
| Human accepted | 0 |
| Public training rows | 0 |
| Natural messages | 156 |
| Assistant messages | 78 |
| Multi-turn candidates | 22 |

Every family produced eight candidates. Valid/rejected counts were:

| Family | Valid | Rejected |
|---|---:|---:|
| absence-negative-unknown | 7 | 1 |
| intent-act-effect | 8 | 0 |
| part-member-material-containment | 5 | 3 |
| purpose-function-use-effect | 6 | 2 |
| role-versus-bearer | 8 | 0 |
| source-report-evidence-endorsement | 8 | 0 |

All six rejections were `unknown_secondary_lens`. Their dialogue content remains readable in the audit packet;
the status captures a generation-contract failure, not a claim that the prose is conceptually useless.

### 8.2 Conversational diagnostics

| Measure | Result |
|---|---:|
| Mean assistant words | 33.8 |
| Median assistant words | 32 |
| 90th percentile | 54 |
| Maximum | 70 |
| Assistant turns ending in `?` | 3.8% |
| Exact duplicate assistant messages | 0 |
| Three-word-shingle pairs with Jaccard ≥ 0.70 | 0 |
| Maximum observed pairwise shingle Jaccard | 0.0952 |

These metrics support the prompt-level goal of avoiding a ritual follow-up-question style and gross template
duplication. They do not establish conceptual correctness, human naturalness, or ecological validity.

The item distribution was 12 linguistic pairs, 20 dialogues, and 16 micro-dialogues; difficulty was 14
introductory, 23 intermediate, and 11 advanced.

### 8.3 Subscription/token observations

Codex reported:

| Token measure | Result |
|---|---:|
| Input tokens | 445,709 |
| Cached input tokens | 300,032 |
| Output tokens | 28,126 |

The input figure includes Codex session/instruction overhead, not just the roughly five-kilobyte worker prompt.
The 67.3% cached fraction helped, but 12 separate sessions are still expensive per 48 candidates. Before a
production pilot, compare one family per call with multi-family or higher-items-per-call batching while holding
quality and recoverability constant. Do not conclude that a cheaper model is better merely from nominal model
price; measure accepted family value per subscription/token and human-review minute.

### 8.4 Representative model-visible exchanges

These are exact calibration outputs, not hand-written examples and not human-approved training rows.

**Role versus bearer**

> **User:** I finished school last month. Am I still a student?
>
> **Assistant:** Not in the current sense. You're the same person, but the student role has ended, and it's
> still true that you were a student.

**Containment versus parthood**

> **User:** If a book is in a box, is the book part of the box?
>
> **Assistant:** No. It's in the box, but it isn't one of the box's parts.

**Use versus function**

> **User:** This hammer has lived on my desk for a year keeping invoices from blowing away. So at this point
> its function is being a paperweight, right?
>
> **Assistant:** Its current use is being a paperweight. Its designed function is still hammering. Those can
> come apart without any mystery.

**Document count versus source independence**

> **User:** A town paper prints a long letter accusing the bridge keeper of taking toll money. So now we have
> two reports: the letter and the paper. That means the charge is corroborated, right?
>
> **Assistant:** Not yet. You have two documents, but maybe only one underlying source if the paper is just
> reproducing the letter. That supports "this accusation was circulated" more than "the accusation is true."
>
> **User:** But a thermometer reading is still evidence even though it passes through an instrument. Why
> isn't a newspaper the same kind of second confirmation?
>
> **Assistant:** Because the instrument is producing its own measurement, while the paper may only be carrying
> someone else's testimony forward. If the paper also says its editor checked the accounts independently,
> that would add a new source. Without that, the claim still belongs to the letter writer.

### 8.5 Storage

At close, the complete project-owned tree was 3,387,193 bytes (about 3.23 MiB), far below the 15 GiB soft
pause. The SQLite validation command reported a logical footprint of 3.28 MiB at the final migration check.
No unrelated directory contributed to the pause calculation.

## 9. Validation evidence

The offline package test suite passes six tests:

1. fresh migrations, idempotent seeds, and complete required tables/views;
2. rejection of update/delete against append-only versioned records;
3. content-addressed blob round trip and digest verification;
4. delimiter independence and metadata separation;
5. reconstructable model-call, candidate, and audit-packet lineage;
6. resumable `paused_storage` state when a campaign-owned threshold is exceeded.

The canonical ledger passes:

- `PRAGMA integrity_check = ok`;
- zero foreign-key violations;
- zero missing required tables;
- zero missing required views;
- zero missing content-addressed blobs;
- zero corrupt blobs;
- both migration digests reconciled.

Migration digests at close:

```text
1 initial_scientific_ledger c9fc33838e1d833e8667ebaf19295b0cfadaf6faef63a63d292d719fdb0f3094
2 current_and_public_views  7f3963528b015eb9771b066ff07a5ae6cc50f1b08e0058486399e1555dcc66ce
```

## 10. Failure and recovery record

The third generation response completed successfully at the provider, but its first ledger ingestion failed on
a foreign key. The root cause was prompt-version identity: the prompt slug included the recipe but not the
family, while family-specific prompt text changed under version 1. SQLite rejected the model-call row because
the intended prompt-version ID had not been inserted after the unique-version collision.

The response was not lost and was not regenerated. Its prompt, output schema, command, JSONL event stream,
stderr, and structured last message were already in the raw call directory. The fix:

- includes family slug in prompt-template identity;
- scans the task's raw call directories for an exact prompt/schema match;
- reconstructs usage and command metadata from the preserved structured artifacts;
- records an `orphan_call_recovered` event;
- ingests the response exactly once;
- then resumes pending tasks.

The recovered campaign finished with exactly 12 provider calls, not 13. This incident is evidence that the
raw-directory/ledger dual record and fail-closed foreign keys are working as intended.

## 11. What is and is not established

Established:

- the ledger can be built from zero and migrated without drift;
- the initial category/family representation is usable for a vertical slice;
- GPT-5.4 can produce schema-valid natural conceptual dialogue from bounded family blueprints;
- failed structural assignments remain visible;
- calls and candidates survive an ingestion interruption;
- the material is concise, varied at a lexical-shingle level, and not dominated by question endings;
- no candidate can enter the public training view without human acceptance.

Not established:

- that any of the 42 structurally valid candidates is philosophically or linguistically correct;
- that synthetic users predict real human conversation;
- that hidden contracts are complete or theory-neutral;
- that the six families are statistically independent enough for research claims;
- that GPT-5.4 is the most cost-effective worker;
- that GPT-5.5 would add useful independent criticism;
- that this curriculum teaches Alpha anything;
- that 48 candidates constitute a training dataset;
- that a production allocation or large generation campaign is justified.

## 12. Current gate and next bounded decision

The program is at **D5 generated, human adjudication pending**.

The next work should be review and decision work, not more generation:

1. read the complete audit packet;
2. mark conceptual validity, conversational quality, linguistic naturalness, pedagogical value, and plurality
   calibration;
3. inspect all six rejected records as well as a stratified valid sample;
4. identify false accept/reject risk in deterministic validators;
5. record style and conceptual error clusters;
6. decide whether response-policy descriptions need controlled normalization;
7. decide whether to repair the six lens-assignment failures or merely revise the prompt;
8. design a cheaper paired batching probe before any production campaign;
9. authorize D6 evaluation construction and/or a later D7 pilot separately.

Do not start Alpha training, rent a GPU, expand the corpus, call GPT-5.5, mutate live Donto, publish a dataset,
or send ad hoc Discord messages merely because this execution record exists. The operator separately
authorized a factual two-hour progress timer after calibration close. Its tracked sender is
`scripts/post-alpha-corpus-progress.mjs`; its secret remains in ignored mode-0600 local state.
