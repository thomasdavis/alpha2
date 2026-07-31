# PRD-03 — Synthetic generation orchestration

## 1. Purpose

This PRD defines how Alpha Corpus will use a strong reasoning model for high-leverage orchestration while
routing most bounded surface generation to economical Codex 5.x-class workers. The objective is not to
maximize calls. It is to maximize reviewed conceptual and conversational value per unit of subscription,
token, human-attention, and wall-clock cost.

The orchestration system must be intelligent enough that the operator does not repeatedly hand-plan batches or
spend frontier-model context rediscovering established choices. It must also be conservative enough that cheap
bulk generation cannot silently flood the ledger with polished conceptual mistakes.

## 2. Core architecture

### 2.0 Initial model-routing decision and calibration evidence

The operator's initial route is GPT-5.6-sol for high-leverage counsel, GPT-5.4 for bounded surface generation,
and no default GPT-5.5 critic. GPT-5.5 requires a paired task-specific probe that demonstrates a concrete
GPT-5.4 failure worth escalating. This supersedes any earlier assumption that GPT-5.5 should routinely design
families or judge every batch.

Execution 01 used 12 serialized GPT-5.4 calls for 48 candidates. Codex reported 445,709 input tokens, of which
300,032 were cached, and 28,126 output tokens. The high per-session input overhead makes batching a measured
optimization target before production. Future routing should compare accepted conceptual value—not raw row
count—under one-family, multi-family, and higher-items-per-call treatments.

### 2.1 Orchestrator tier

Use the strongest available reasoning model for tasks where mistakes propagate across many units:

- program and release planning;
- concept-family design;
- taxonomy-gap discovery;
- generator prompt and tool-schema design;
- creation of hard negatives and false bridges;
- batch routing and stopping decisions;
- review-rubric calibration;
- difficult adjudication preparation;
- cross-batch diversity analysis;
- failure-cluster diagnosis;
- deciding which worker outputs deserve repair, escalation, or human attention;
- writing experimental manifests and claims.

The orchestrator does not write every sentence. A single well-designed family blueprint may enable hundreds of
economical worker generations.

### 2.2 Worker tier

Use lower-cost capable 5.x models for bounded, schema-constrained tasks such as:

- surface realizations from approved blueprints;
- paraphrases with specified invariants;
- user-turn variation;
- length and register variants;
- minimal pairs;
- ordinary conversational connective material;
- fictional entity and situation instantiation;
- straightforward source-conditioned scenes;
- candidate hard negatives generated under a precise contract;
- repairs for clearly localized review findings;
- descriptive metadata proposals later checked by reviewers.

Worker eligibility is empirical and task-specific. A cheaper model may be excellent at paraphrase but poor at
counterexamples. The registry records these profiles rather than assuming one global ranking.

### 2.3 Critic tier

Use one or more independently prompted or independently provided models for:

- conceptual boundary checks;
- source fidelity;
- unintended entailments;
- invalid counterexamples;
- style signature detection;
- user realism;
- culturally sensitive or theory-dependent warning flags;
- pairwise comparison of alternative candidates.

Claude or another provider may be useful as an independent critic when available, especially where correlated
OpenAI teacher and judge errors are a concern. Provider diversity is evidence, not automatic truth.

### 2.4 Human authority tier

Humans remain final authority for:

- contested philosophical cases in evaluation;
- cultural or community-specific language;
- whether a counterexample genuinely engages a claim;
- whether a dialogue feels worth continuing;
- whether a revision is insightful rather than an exception list;
- whether a synthetic user resembles real interaction;
- promotion of a newly minted lens into the canonical taxonomy;
- public-release risk and rights decisions.

## 3. Unit-generation workflow

### Stage A — Allocation

The orchestrator reads coverage, rejection, cost, and experiment requirements. It chooses a bounded allocation
of concept families and transformations. It states why each allocation is useful and what evidence would make
the batch stop.

### Stage B — Blueprint

For each family, the orchestrator produces a structured blueprint containing:

- purpose and competency questions;
- latent distinction or conversational operation;
- category/lens assignments;
- positive, negative, borderline, and legitimately plural cases;
- required and prohibited inferences;
- dependency graph and expected deltas;
- projections and false bridges;
- shortcut hazards and lexical holdouts;
- scene/trajectory plan;
- desired response policies and style distribution;
- evaluation probes;
- authority type and human-review needs.

Blueprints are reviewed before expensive surface expansion. A flawed blueprint can corrupt an entire family.
PRD-14 defines how desired response policies become versioned compositional targets, how those targets compile
to natural worker instructions, and how intended versus observed distributions remain separate.

### Stage C — Seed realization

Generate a small diverse seed set using at least two worker prompt variants, and optionally two worker models.
Do not expand until seed quality and family realizability are established.

### Stage D — Automated structural checks

Validate schema, required fields, content integrity, delimiter independence, length, role sequence, source
anchors, lexical holdouts, and obvious duplicates. Failures remain in the ledger and do not enter conceptual
review.

### Stage E — Independent critique

Critics assess separate dimensions rather than emit one holistic score. Each finding points to a span,
commitment, family contract, or source fragment.

### Stage F — Selective repair

Repair only candidates with a localized, repairable defect. Regenerating a fundamentally misconceived unit is
usually cheaper and cleaner than accumulating qualifications. The original and repair lineage remain stored.

### Stage G — Batch adjudication

Batch review examines candidate quality *and the batch distribution*: repetition, teacher signature,
conceptual redundancy, length, question rate, stance, and category coverage.

### Stage H — Controlled expansion

Only families with acceptable seed yield expand. Expansion budgets are adaptive: high-yield families may
receive more surface variants; low-yield or high-disagreement families receive stronger review or stop.

### Stage I — Freeze and quarantine

Accepted candidates enter a release-eligible pool. Evaluation candidates and close relatives are quarantined
at family, projection, template, lexical, and source levels before training release construction.

## 4. Campaign hierarchy

Generation operates at four scales:

1. **Probe:** a handful of calls testing whether a task/prompt/model combination is viable.
2. **Calibration batch:** enough candidates for human audit, yield estimation, and style analysis.
3. **Production batch:** bounded resumable expansion under a frozen recipe.
4. **Campaign:** multiple batches serving a declared release allocation and budget.

No campaign is “generate until 200,000.” It ends when its family allocation, marginal diversity, quality,
and cost rules say to end.

## 5. Cost-aware routing

### 5.1 Cost unit

Track at least:

- input and output tokens;
- cached input tokens where exposed;
- provider-reported or estimated monetary cost;
- subscription quota estimate;
- strong-model reasoning calls;
- human-review minutes;
- accepted units;
- accepted independent families;
- accepted transformation edges;
- semantic novelty after clustering;
- downstream behavioral lift when known.

The most important denominator is not raw rows. It is accepted useful family structure.

### 5.2 Routing policy

For every task class, maintain a measured routing profile:

- worker pass rate;
- repair rate;
- conceptual error type;
- human disagreement;
- cost per accepted result;
- novelty contribution;
- style concentration;
- latency and reliability.

Route to the cheapest tier whose calibrated performance satisfies the task's risk threshold. Escalate when:

- two worker attempts fail the same contract;
- critic disagreement crosses a threshold;
- the family is evaluation-critical;
- the case is culturally or philosophically sensitive;
- a new lens is proposed;
- an error could replicate across a large expansion;
- source fidelity cannot be established cheaply;
- the candidate would become a canonical anchor.

Do not escalate merely because prose could be prettier.

### 5.3 Strong-model amortization

The orchestrator should produce reusable high-information artifacts:

- family blueprints;
- prompt macros;
- variation plans;
- known-failure rules;
- critic checklists;
- batch summaries;
- routing recommendations;
- decision records.

Workers receive the smallest sufficient context: relevant blueprint slice, examples, and contract. They should
not repeatedly receive the entire PRD suite or Donto prompt.

### 5.4 Batch review economy

Review structurally similar candidates in calibrated batches, allowing comparison and detection of repetitive
patterns. Suggested approach:

- automated checks over all candidates;
- cheap critic over all structurally valid candidates where worthwhile;
- stratified human sample from every batch;
- full human review for evaluation anchors and high-risk classes;
- adaptive expansion or halt based on measured false-negative risk;
- periodic blind re-review of accepted and rejected samples.

Sampling never means deleting unreviewed candidates. Their status remains explicit.

## 6. Prompt architecture

Prompts are composed from versioned modules:

- role and task contract;
- relevant taxonomy definitions;
- family blueprint;
- visible source fragments;
- target scene/trajectory state;
- desired style and length;
- prohibited shortcuts;
- structured output schema;
- examples selected for diversity and non-leakage;
- self-check limited to concrete contract violations.

The full Donto extraction prompt should inform the taxonomy and blueprint generator, but should not be pasted
into every worker call. That would waste tokens and impose an extraction-report voice on conversational data.

Prompts must not request vague “deep,” “nuanced,” or “philosophical” output without operational constraints.
Those words reliably induce verbosity and stylistic imitation rather than conceptual correctness.

## 7. Structured output contract

All machine-ingested structured results use schema-constrained output or a required tool call. A generation
response should separate:

- natural model-visible messages;
- hidden family/state annotations;
- self-reported uncertainty;
- proposed categories;
- source references;
- generator notes.

The ledger retains the raw response and the validated structured object. Free-text JSON scraping is forbidden.

## 8. Generation roles

Roles are task functions, not anthropomorphic agents:

- **Allocator:** chooses under-covered high-value work.
- **Family architect:** writes semantic blueprints.
- **Scenario constructor:** creates situations and participants.
- **User simulator:** produces user turns under a constrained goal and style.
- **Assistant candidate writer:** creates natural Alpha responses.
- **Contrast constructor:** creates minimal changes, hard negatives, and false bridges.
- **Trajectory composer:** connects turns into stateful conversations.
- **Source writer:** creates fictional evidence passages where appropriate.
- **Source fidelity critic:** checks claims against visible evidence.
- **Linguistic critic:** checks naturalness and phenomenon realization.
- **Ontology critic:** checks boundaries, roles, identities, time, and entailments.
- **Conversation critic:** checks contingency, presence, momentum, length, and question necessity.
- **Adversarial critic:** searches for shortcuts and counterexamples.
- **Repairer:** addresses an identified defect without unrelated rewriting.
- **Deduplication analyst:** proposes semantic clusters and template signatures.
- **Adjudication synthesizer:** summarizes conflict for human decision without erasing minority views.

The same model may fill multiple roles in different calls, but a candidate cannot be accepted solely by the
same call or prompt lineage that produced it.

## 9. Synthetic user policy

Synthetic users are useful for breadth but are not evidence of human ecological validity. The system shall:

- generate behavior from goals, knowledge, emotional state, conversational history, and constraints—not a
  demographic stereotype label;
- include hesitations, corrections, fragments, implicit references, and changes of mind without making every
  user incoherent;
- vary how much the user knows and how accurately they express it;
- prevent the user simulator from conveniently revealing the hidden evaluation contract;
- distinguish adversarial tests from realistic ordinary users;
- use human-authored or responsibly licensed anchors for calibration where authorized;
- avoid claims that a synthetic variety represents a real community until community validation exists.

Research comparing simulated and real users shows that simulator choice can materially shift measured agent
performance and interaction style. Synthetic user results therefore remain provisional until human evaluation
([Lost in Simulation](https://arxiv.org/abs/2601.17087)).

## 10. Diversity without random noise

Variation must occur along recorded causal dimensions:

- lexical choice;
- syntax;
- discourse order;
- context distance;
- register;
- emotional stance;
- user expertise;
- response policy;
- domain projection;
- source condition;
- ambiguity structure;
- challenge type;
- answer length;
- degree of explicit metalanguage.

Uncontrolled temperature does not constitute diversity. Every generation recipe should specify what kind of
variation it seeks and later measure what occurred.

## 11. Negative generation

Hard negatives are central but dangerous. Required types include:

- same words, different structure;
- different words, same structure;
- plausible but unsupported inference;
- invalid counterexample;
- correct conclusion with collateral conceptual changes;
- overhedged response that invents alternatives;
- overconfident response that collapses a legitimate set;
- unnecessary clarification;
- canned philosophical response;
- source conflation;
- temporal flattening;
- perspective flattening;
- false analogy;
- technically correct but conversationally inert lecture;
- warm and fluent but substantively empty response.

Negative writers must not produce cartoonishly bad text. Surface quality, length, and style should be matched
closely enough that the conceptual relation is the differentiator.

## 12. Review-Instruct and chairman-style caution

Multiple candidate/reviewer/chairman roles are useful engineering precedent, but reviewer count is not
epistemic authority. The ledger preserves each independent review, and a synthesizer cannot silently average
away a principled disagreement. Human adjudicators see the underlying evidence and minority findings.

## 13. Failure clusters and automatic stopping

The orchestrator maintains failure clusters such as:

- invalid counterexamples;
- generic essay voice;
- question appended automatically;
- overlong definitional repair;
- hidden answer leakage in user turn;
- source invention;
- repetitive scenario frame;
- technical-jargon dependency;
- semantic duplicate;
- false cultural authority;
- inconsistent hidden state;
- role or delimiter contamination;
- wrong-family branch attachment.

A batch stops when:

- hard failure exceeds its threshold;
- marginal accepted semantic novelty falls below threshold;
- cost per accepted family structure exceeds authorization;
- teacher/style concentration rises;
- reviewer disagreement shows the blueprint itself is unstable;
- provider behavior changes materially;
- quota or system load becomes unsafe;
- the declared allocation is complete.

## 14. Orchestrator memory

The orchestrator reads structured ledger summaries rather than the entire corpus. Persist:

- current goal and gates;
- coverage frontier;
- recent batch outcomes;
- model/task calibration;
- known failure signatures;
- open questions;
- budget and quota state;
- decisions and supersessions;
- pending human adjudications.

Every orchestration decision that affects scope, budget, or release eligibility is stored with rationale. This
prevents expensive rediscovery across Codex sessions.

## 15. Initial model-allocation policy

At implementation time, create a runtime registry rather than hardcode current commercial names:

- `orchestrator_high_reasoning`: smartest available Codex/OpenAI reasoning model;
- `worker_general`: economical 5.x model calibrated for natural dialogue realization;
- `worker_transform`: economical model calibrated for paraphrase/minimal transformations;
- `critic_independent`: independently prompted or alternative-provider critic;
- `judge_calibrated`: model allowed to prefilter only after human calibration;
- `human_required`: route requiring human authority.

Each alias resolves to an exact dated model revision and provider. The same alias may resolve differently in a
future campaign without corrupting historical lineage.

## 16. Acceptance criteria

The generation orchestrator is ready for a pilot only when:

- a campaign can be bounded by quota, calls, tokens, cost estimate, families, and wall time;
- every call and routing decision reaches the ledger;
- interruption and resume do not duplicate accepted scientific objects;
- worker, critic, and orchestrator identities are exact;
- no candidate self-approves;
- strong-model usage is limited to declared high-leverage tasks;
- prompt contexts are minimized and cached where supported;
- batch yield and cost are calculated from accepted family value, not raw rows;
- failures and null responses remain visible;
- escalation and stopping rules work fail-closed;
- model/provider changes trigger recalibration;
- no live generation begins without a separately approved G2 campaign contract.
