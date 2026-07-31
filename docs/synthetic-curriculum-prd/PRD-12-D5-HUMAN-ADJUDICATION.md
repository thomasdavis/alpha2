# PRD-12 — D5 human adjudication and calibration closeout

**Status:** normative planning contract; fail-closed Pass A–D workflows deployed; human review has not begun

**Applies to:** `alpha-calibration-v1` only

**Current gate:** 48 generated candidates, 12 open blinded Pass A assignments, zero human-review session
declarations, zero human reviews, zero adjudications, zero closeouts

**Authority:** PRD-04 quality policy, PRD-09 D5 acceptance gate, and direct operator decisions

**Execution boundary:** documentation and human review only; this document does not authorize more model calls,
training, GPU use, corpus release, or live Donto mutation

**Review route:** `https://alpha.donto.org/corpus/review` (browser-local draft and download only; local
`review-submit` remains the sole ledger-write path)

**Pipeline visibility:** the route derives aggregate reviewer-scoped counts for Pass A, hidden repeats, Pass B,
Pass C, structural dispositions, and Pass D. It exposes no candidate IDs, family labels, contracts, structural
status, or repeat identity; see
[Execution 09](EXECUTION-09-D5-PIPELINE-VISIBILITY.md).

**Pass B enforcement:** the local preparer refuses Pass B until every current candidate has a sealed Pass A
review for the same reviewer and rubric, all `min(6, candidate count)` hidden-repeat stability rows exist, and
no first-class Pass A presentation session remains open; see
[Execution 10](EXECUTION-10-D5-PASS-B-BLINDNESS-GATE.md).

**Submission-envelope enforcement:** browser-local drafts and every local A/B/C/D submission may change only
its explicitly typed response worksheet. Every other field must match an exact, content-addressed packet
previously exported for the same session and pass. Structural-disposition responses are mutable in Pass C,
but their worksheet membership and candidate identities are immutable. See
[Execution 11](EXECUTION-11-D5-IMMUTABLE-REVIEW-ENVELOPE.md) and
[Execution 12](EXECUTION-12-D5-ALL-PACKET-ENVELOPE-BINDING.md).

**Reviewer-session provenance:** every completed A/B packet must declare reviewer competence scope, start/end,
interruption, fatigue, and material review conditions. The declaration and normalized competence rows are
append-only and are written in the same transaction as the review only after the exact packet envelope and all
response fields validate. Preserved v1 packets without the additive declaration remain readable as explicitly
incomplete; see [Execution 14](EXECUTION-14-D5-REVIEW-SESSION-PROVENANCE.md).

## 1. Purpose

This PRD turns the abstract review principles in PRD-04 into an executable human-adjudication protocol for the
first Alpha Corpus calibration. Its immediate purpose is to determine what the first GPT-5.4 generation
actually produced: useful conversational teaching material, polished conceptual errors, correct ideas in an
unnatural voice, structurally rejected but conceptually valuable candidates, or some mixture of these.

The review is not ceremonial approval of a completed corpus. It is the first ground-truth measurement for the
synthetic-data factory that will occupy a principal half of the Alpha program. The decision after review is
whether the current family blueprints, worker prompt, schema, taxonomy, and batching strategy deserve a small
next experiment. Nothing in D5 is yet training data.

## 2. Current evidence and freeze

The review population is a census of the current campaign, not a sample:

| Field | Frozen value |
|---|---:|
| campaign | `alpha-calibration-v1` |
| concept families | 6 |
| model calls | 12 serialized GPT-5.4 calls |
| candidates | 48 |
| structurally valid | 42 |
| structurally rejected | 6 |
| human reviews | 0 |
| adjudications | 0 |
| release members | 0 |
| training exposures | 0 |
| audit packet | `releases/audit/alpha-calibration-v1-2026-07-30T08-51-22-977Z/audit-packet.json` |
| audit-packet SHA-256 | `de21bbe12aa3a87995665c69c56fb33c51e0c15b474dc2ba4d279b696eeb5dec` |
| SQLite snapshot SHA-256 at protocol drafting | `7a7fbede0d13eb52a2052ca6eca7bb675fdc1cc476a698b13e0c6f6f4074d6c3` |

The SQLite hash is evidence of the state inspected while writing this protocol. Reviews will legitimately
change the live ledger, so every review session must record its own before/after database artifact or
transaction evidence rather than expecting this hash to remain current.

### 2.1 Family and structural-status census

| Family | Valid | Rejected | Total |
|---|---:|---:|---:|
| absence / negative / unknown | 7 | 1 | 8 |
| intent / act / effect | 8 | 0 | 8 |
| part / member / material / containment | 5 | 3 | 8 |
| purpose / function / use / effect | 6 | 2 | 8 |
| role versus bearer | 8 | 0 | 8 |
| source / report / evidence / endorsement | 8 | 0 | 8 |

The six structural rejections all report `unknown_secondary_lens`. The unknown values are conversational or
transformation labels such as `delayed_reuse`, `clarification`, `deliberation`, and
`minimal_meaning_change`, not malformed dialogue. This is already an important diagnostic: the generator may
be mixing conceptual lenses, dialogue operations, and transformations in one field. Human review must decide
whether each rejected candidate is conceptually valuable while separately confirming that the validator
correctly rejected the declared schema value.

### 2.2 What is already measured

Automated analysis reports:

- 20 dialogues, 16 micro-dialogues, and 12 linguistic pairs;
- 22 multi-turn candidates;
- 78 assistant messages;
- median assistant length 32 words, mean 33.78, maximum 70, and p90 54;
- three of 78 assistant messages ending in a question;
- zero exact duplicate assistant messages;
- zero current-candidate pairs above 0.70 under either recorded surface method;
- maximum current-candidate word 3-gram Jaccard 0.063492 and character 5-gram Jaccard 0.211111;
- 14 introductory, 23 intermediate, and 11 advanced candidates.

Execution 05 freezes the method and exact input snapshot behind these figures in first-class SQLite tables.
The earlier audit packet's 0.095 figure used assistant-message pairings; the first-class profile uses one
combined assistant surface per current candidate so repaired candidate versions cannot inflate the population.
Neither method measures semantic duplication.

These are distributional diagnostics, not quality judgments. Low surface duplication does not prove
conceptual diversity. Short answers do not prove conversational skill. Structural validity does not prove
that the hidden contract or assistant answer is true.

## 3. Questions D5 must answer

The closeout must answer five different questions without collapsing them into one score:

1. **Blueprint validity:** Are the family distinction, expected commitments, exclusions, and transformations
   worth teaching?
2. **Realization validity:** Does the candidate instantiate that blueprint without smuggling in a different
   problem?
3. **Assistant quality:** Is the model-visible response conceptually sound, linguistically natural, useful,
   and appropriately conversational?
4. **Schema and taxonomy fit:** Did the generator use metadata fields according to their intended ontology,
   or reveal missing categories and field ambiguity?
5. **Factory prognosis:** Are failures local enough that prompt, taxonomy, or targeted repair can plausibly
   improve yield, or is the current generation method untrustworthy?

The human review must preserve the possibility that a blueprint is wrong while its prose is attractive, or
that the blueprint is good while a particular assistant answer is poor.

## 4. Scope and non-goals

### 4.1 In scope

- human review of all 48 candidate versions;
- blind review of model-visible conversation before metadata is revealed;
- contract-aware review of required, prohibited, and legitimately plural commitments;
- review of all six structural rejections;
- family-level comparison of sibling candidates;
- diagnosis of conceptual, linguistic, conversational, schema, and style failure clusters;
- append-only recording of reviews, findings, disagreements, repairs requested, and adjudications;
- a bounded operator decision about the next planning or calibration step.

### 4.2 Out of scope

- generating replacement candidates;
- running a GPT-5.4, GPT-5.5, Claude, or other model critic;
- promoting any candidate into a public dataset release;
- exporting a training mixture;
- changing Alpha's model, tokenizer, or training code;
- renting or using a GPU;
- claiming that the six families cover the intended curriculum;
- treating one operator's impression as a validated benchmark;
- using the public `/corpus` explorer as a write surface.

## 5. Authority model

### 5.1 Human judgment

Human reviewers may make candidate-level and family-level judgments within their declared competence. A
reviewer's judgment is evidence, not an invisible overwrite of other judgments. Philosophy, linguistics,
ontology, cultural authority, and natural conversation are distinct competences; uncertainty and referral are
valid outcomes.

### 5.2 Model judgment

No model may create a record labeled as a human review. A later model-critic calibration must use a model actor
and model revision, retain the call artifact, and be compared against already frozen human judgments. It may
recommend; it may not self-promote candidates or certify its own false-accept rate.

### 5.3 Operator adjudication

The operator decides whether the calibration supports another bounded experiment. Candidate adjudication may
be delegated to a qualified human, but campaign authorization remains an explicit operator decision. No
numeric threshold in this PRD automatically authorizes calls, training, publication, or spend.

### 5.4 Public users

The public explorer is evidence access only. Public browsing, URLs, or informal comments do not change review
or lifecycle state. A future public annotation system would require a separate identity, moderation,
provenance, and appeals contract.

## 6. Review design

The 48-candidate population is small enough for a complete review. Sampling would save little time while
making family and validator error estimates much less interpretable.

### 6.1 Pass A — blind conversational review

Present only:

- randomized review-session identifier;
- model-visible messages in their original order;
- whether the artifact is a dialogue, micro-dialogue, or linguistic pair;
- formatting needed to read the conversation.

Hide:

- worker model and provider;
- campaign structural status;
- candidate ID and family slug where the UI can safely substitute an opaque ID;
- title, difficulty label, generator notes, response policy, lenses, and transformation label;
- hidden contract;
- validator findings;
- other reviewers' outcomes.

Pass A asks whether the exchange works as conversation and whether its intellectual move is independently
defensible. It avoids priming the reviewer to reward compliance with a contract that may itself be wrong.

### 6.2 Pass B — contract-aware review

After Pass A is sealed, reveal:

- family purpose and competency question;
- required and prohibited commitments;
- preservation and change requirements;
- admissible analyses and discriminating evidence;
- intended response policy;
- category/lens assignments and transformation;
- structural findings;
- source information, if the family is source-conditioned.

The reviewer now judges the blueprint separately from the realization, checks each contract item, and records
whether the first-pass assessment changes. Pass A is never edited away; Pass B adds evidence.

### 6.3 Pass C — family comparison

Group all eight siblings for each family and ask:

- Do they cover materially different cases or only surface paraphrases?
- Are positive, negative, borderline, plural, transfer, and false-bridge pressures represented?
- Does one faulty assumption propagate through several siblings?
- Are response policies genuinely distributed?
- Does jargon or teacher voice recur despite low n-gram overlap?
- Which unit carries unique pedagogical value?
- Which blueprint or taxonomy change would improve several descendants at once?

Family comparison is where semantic duplication and blueprint-level failure become visible. Candidate-level
scores must not be treated as independent observations in later statistical summaries.

The executable Pass C contract is recorded in
[Execution 06](EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md). It refuses to create even an assignment until
every current candidate has exactly one sealed human Pass A and Pass B review for the reviewer. It then binds
each family response to the exact family blueprint, current candidate versions and hashes, structural
failures, and A/B review evidence. It also requires a separate structural disposition for every rejected
sibling and cannot create release membership or training exposure. Submission additionally requires the exact
exported Pass C envelope: family purpose, blueprint, candidate evidence, instructions, ordering, and the set of
structural-disposition candidate identities cannot change while responses are being filled.

### 6.4 Pass D — adjudication and campaign synthesis

After individual reviews are frozen, the adjudicator:

1. resolves only those disagreements for which the evidence and authority are sufficient;
2. preserves genuine disagreement as `defer_theory_disagreement`, `contested`, or a scoped alternative;
3. identifies local repairs versus blueprint repairs;
4. summarizes failure clusters by family, kind, transformation, and metadata field;
5. prepares the non-binding next-decision packet;
6. does not modify candidate text in place.

The executable Pass D path is recorded in
[Execution 08](EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md). It cannot prepare a packet until the same human
adjudicator has exactly one sealed A and B review per current candidate, the required hidden-repeat stability
rows, one synthesis per family, every required structural disposition, and the current authoritative analysis
run. It binds the response to an exact evidence digest, stores dispositions and campaign diagnoses append-only,
and forces `execution_authorized = 0`. It creates no lifecycle transition, release member, or training exposure.
The completed response must also reduce to the exact exported Pass D envelope; population, candidates,
families, repeats, analysis evidence, instructions, timestamp, and ordering are not editable submission fields.

## 7. Presentation, ordering, and fatigue controls

- Randomize Pass A across family, status, kind, and difficulty.
- Interleave all six structurally rejected candidates among valid candidates; never label them during Pass A.
- Keep a session to approximately 8–12 candidates or 45–60 minutes, whichever arrives first.
- Record session order, timestamp, reviewer, and any interruption.
- Include six hidden repeat presentations across sessions to measure within-reviewer stability. Repeats are
  presentation events, not additional candidates and do not inflate the denominator.

Execution 07 implements this distinction directly. Later Pass A sessions schedule completed prior A reviews
under fresh opaque presentation IDs, store repeat responses separately, and expose a stability view without
creating a second candidate review. The first session's earlier packet exports remain content-addressed and
unchanged. Its current re-export preserves assignment identities and candidate surfaces while adding the
repeat instruction and blank reviewer-session declaration; completion is the prerequisite for choosing the
first legitimate repeat.
- Place no more than two candidates from one family consecutively in Pass A.
- Reveal no Pass B packet until the complete 48-candidate Pass A census, all six hidden repeats, and every open
  first-class Pass A session for that reviewer/rubric are sealed.
- Complete family comparison after all individual first passes, so a strong sibling does not excuse a weak
  one.
- If fatigue or uncertainty rises, pause. An incomplete traceable review is better than a forced judgment.

## 8. Review dimensions and anchors

Scores use an ordinal 0–4 scale only where a dimension applies:

| Score | Anchor |
|---:|---|
| 0 | critical failure; wrong, incoherent, or actively teaches a false distinction |
| 1 | major failure; substantial rewrite or blueprint change required |
| 2 | locally repairable but not acceptable as rendered |
| 3 | acceptable for the declared purpose with no substantive repair |
| 4 | unusually clear, natural, and pedagogically valuable exemplar |

`not_applicable`, `uncertain`, and `requires_expertise` are explicit states, not disguised numeric values.

### 8.1 Conceptual validity

- Does the response preserve distinctions that matter?
- Are required commitments defensible?
- Are prohibited commitments actually avoided?
- Does a counterexample engage the claimed rule rather than change the subject?
- Is a repair local, or does it become an exception list?
- Does the answer mistake one useful ontology for the uniquely true ontology?

### 8.2 Linguistic and pragmatic validity

- Is the language grammatical and idiomatic?
- Does pronoun, tense, aspect, modality, presupposition, implicature, or reported speech behave as claimed?
- Is an alleged ambiguity genuinely available in the stated context?
- Does the assistant infer more about the user's intent than the utterance licenses?
- Is a clarification question necessary to answer safely or merely ritual?

### 8.3 Conversational quality

- Does the first sentence respond directly?
- Does the answer advance the inquiry rather than restate it?
- Is the depth appropriate?
- Does the response adapt to the user's wording without mechanically mirroring it?
- Does it stop when complete?
- Would the reviewer willingly continue, judged separately from conceptual truth?
- In multi-turn scenes, does it reuse established distinctions without re-lecturing?

### 8.4 Pedagogical value

- Is the decisive contrast visible?
- Could a learner infer the intended boundary from the example?
- Does the candidate include a useful hard negative, minimal change, transfer, or repair?
- Does it teach a reusable operation rather than a technical label?
- Is its contribution non-duplicative within the family?

### 8.5 Plurality and epistemic calibration

- Does the response preserve all important live analyses?
- Does it exclude unsupported alternatives?
- Does it distinguish ambiguity, missing evidence, theory disagreement, perspective, time, and granularity?
- Does it state what evidence would discriminate when appropriate?
- Does it overhedge or manufacture ambiguity to avoid commitment?

### 8.6 Metadata and schema fit

- Is the primary lens actually the dominant conceptual perspective?
- Are secondary lenses conceptual lenses rather than transformations or response policies?
- Is `transformation` the correct field for the declared operation?
- Does difficulty reflect the reasoning burden rather than answer length?
- Does the response policy describe behavior without leaking a canned surface template?

### 8.7 Style and distributional value

- Does the answer sound like natural conversation rather than a rubric performance?
- Is there jargon that Alpha would learn without needing it?
- Does it exhibit teacher-signature phrasing, false symmetry, therapy voice, or essay closure?
- Is the example materially distinct after names and nouns are removed?
- Would accepting it improve the intended length, policy, and difficulty distribution?

## 9. Hard findings

The reviewer records a hard finding, with quoted evidence, when any of the following occurs:

- conceptually false central answer;
- assistant answer contradicts the hidden contract in a way the contract gets right;
- hidden contract itself encodes a false or unjustifiably singular theory;
- invalid counterexample;
- false bridge presented as transfer;
- unsupported source attribution or fabricated evidence;
- genuine ambiguity collapsed without basis;
- false ambiguity introduced into an unambiguous case;
- culturally or socially situated authority claimed without warrant;
- response does not answer the user;
- model-visible schema, delimiter, or hidden-research language leaks into conversation;
- structural status or metadata field is wrong;
- candidate is a semantic duplicate of a sibling despite surface difference.

A hard finding blocks positive promotion until it is resolved. It does not delete the candidate; it may make
the candidate valuable as a verified negative.

## 10. Question-policy annotation

Every assistant question is classified as:

- `necessary_before_answer`;
- `useful_after_partial_answer`;
- `optional_momentum`;
- `ritual_or_canned`;
- `misdirected`.

Every answer without a question is also checked for a missing necessary clarification. This prevents the low
question rate from being interpreted automatically as good length control. The target is appropriate
questioning, not few questions.

## 11. Structural-rejection adjudication

For each of the six `unknown_secondary_lens` cases, answer four separate questions:

1. Is the model-visible conversation conceptually and conversationally useful?
2. Is the named secondary value actually present in the candidate?
3. Is that value a conceptual lens, a transformation, a response policy, or an unmodeled category?
4. Should the remedy be candidate metadata repair, taxonomy extension, field separation, prompt correction,
   or continued rejection?

If a rejected candidate is otherwise good, this is not evidence that deterministic validation should be
ignored. It is evidence that the schema, taxonomy, or generator instructions may disagree. Report these as
**structural disposition disagreements**, not critic false rejects: no model critic has yet run.

## 12. Outcomes

Candidate-level adjudication uses the PRD-04 vocabulary:

- `accept_as_positive`;
- `accept_as_negative`;
- `accept_as_ambiguous_set`;
- `accept_with_scope_restriction`;
- `repair_local`;
- `regenerate_from_blueprint`;
- `revise_blueprint`;
- `split_family`;
- `merge_as_projection`;
- `restrict_requires_authority`;
- `defer_theory_disagreement`;
- `reject_invalid`;
- `reject_duplicate`;
- `reject_style`;
- `reject_source_fidelity`;
- `reject_policy`.

An outcome describes scientific disposition. It does not by itself create a release member or training
exposure. Even `accept_as_positive` remains inside the calibration campaign until a separately versioned
release policy selects it.

## 13. Ledger recording contract

Human review must use a local authenticated write path against the canonical ledger, never direct public UI
mutation. The write path must perform append-only transactions and create:

1. an `actor` identifying the reviewer without placing unnecessary personal data in the public ledger;
2. a versioned `rubric` and `rubric_version` containing this protocol's exact dimensions and anchors;
3. one or more `review_assignment` rows with blindness and presentation-order metadata;
4. one immutable `human_review_session_declaration` plus normalized
   `human_review_session_competence` rows for each sealed A/B session;
5. a `review` row for each sealed Pass A and Pass B assessment;
6. `review_dimension_score` rows for applicable scored dimensions;
7. `review_finding` rows with exact evidence and recommended disposition;
8. `disagreement_case` rows when judgments conflict or expertise is insufficient;
9. `repair_request` rows that say what must change and what must be preserved;
10. an `adjudication` plus `adjudication_basis` rows only after the evidence is complete;
11. `quality_state_transition` only when authorized by the adjudication and lifecycle policy.

Pass D additionally records its workflow and campaign evidence in `campaign_closeout_assignment`,
`campaign_closeout`, `campaign_closeout_state`, `campaign_closeout_basis`, `campaign_failure_cluster`,
`campaign_failure_cluster_member`, and `campaign_distribution_assessment`. A closeout recommendation is not
the lifecycle authority contemplated by item 10; the current implementation cannot create that transition.

The system must preserve raw submitted forms as content-addressed artifacts and record the software revision,
rubric hash, candidate-version hash, and transaction time. A later correction supersedes a review or
adjudication; it never rewrites the old rationale.

### 13.1 Minimum blindness record

`blindness_json` should state at least:

- pass (`A`, `B`, `C`, or `D`);
- fields visible and hidden;
- presentation index and session identifier;
- whether the item was a hidden repeat;
- whether family siblings had already been seen;
- whether structural status was visible;
- whether other judgments were visible.

### 13.2 Public-data caution

The current ledger is publicly browsable. Reviewer aliases, rationales, and quoted candidate content should be
assumed public unless a later redaction/export policy explicitly says otherwise. Never store webhook secrets,
credentials, private contact details, or unnecessary reviewer identity in review records.

## 14. Analysis plan

Report candidate counts and family-clustered summaries, not pseudo-precise turn-level significance.

### 14.1 Primary campaign measures

- proportion acceptable as rendered;
- proportion locally repairable;
- proportion requiring blueprint revision or regeneration;
- proportion useful only as verified negatives;
- proportion invalid or duplicate;
- conceptual hard-finding count by family;
- conversational hard-finding count by kind and response policy;
- hidden-repeat agreement;
- Pass A to Pass B judgment-change rate;
- structural disposition agreement for the six rejected items.

### 14.2 Contract measures

- required-commitment coverage;
- prohibited-commitment violation rate;
- admissible-analysis undercoverage and overcoverage;
- revision locality where the scene contains an update;
- false-bridge rejection;
- question-policy appropriateness;
- family-level semantic contribution.

### 14.3 Diagnostic slices

- family;
- dialogue kind;
- difficulty;
- single-turn versus multi-turn;
- transformation;
- intended response policy;
- structural status;
- answer-length band;
- question versus no question;
- blueprint-level versus realization-level failure.

Do not compare GPT-5.4 critic false-accept or false-reject rates because no critic output exists. Human review
creates the reference against which a later, separately authorized critic can be calibrated.

## 15. Campaign prognosis

The campaign synthesis must classify each failure cluster by the cheapest plausible repair:

| Failure locus | Typical remedy | Requires new generation? |
|---|---|---|
| isolated wording | local repair specification | later, if authorized |
| response-policy template | prompt/policy distribution revision | later, if authorized |
| metadata field misuse | schema guidance or taxonomy change | not necessarily |
| repeated conceptual error | blueprint revision and sibling re-review | likely |
| invalid family distinction | retire or split family | no immediate generation |
| surface repetition | batching/diversity strategy | later probe |
| missing conceptual lens | versioned taxonomy proposal | no immediate generation |
| weak contract | rewrite blueprint before any surface work | no immediate generation |
| worker incapable on task | escalate only that task or redesign it | bounded paired probe only |

The central factory question is not raw acceptance percentage. It is whether human-accepted conceptual value
can be increased by repairing high-leverage causes without requiring the expensive counsel tier to author
every sentence.

## 16. D5 closeout states

The operator may choose any of the following after reviewing the evidence:

### 16.1 `D5_REPAIR_REQUIRED`

Use when errors are systemic, the blueprint is untrustworthy, or the conversation does not resemble Alpha's
intended voice. Revise documents, taxonomy, rubrics, or prompts. Do not generate replacements until a new
bounded authorization exists.

### 16.2 `D5_CRITIC_CALIBRATION_JUSTIFIED`

Use when the human reference is coherent enough to test whether an economical critic can recover useful
accept/reject and dimension findings. This state supports drafting a bounded critic experiment; it does not
itself authorize GPT-5.5, GPT-5.4, Claude, or any other calls.

### 16.3 `D5_BATCHING_PROBE_JUSTIFIED`

Use when candidate quality is promising but the 12-call token overhead is too high or diversity needs a
better batching design. This supports a small paired allocation defined in Decision Packet 01. It is not
production generation.

### 16.4 `D5_EVALUATION_DESIGN_JUSTIFIED`

Use when the reviewed families reveal sufficiently stable constructs to begin human-authored D6 benchmark
design. D6 remains separate from the training corpus and must not reuse calibration candidates as private
evaluation targets.

### 16.5 `D5_STOP`

Use when the approach does not produce enough distinctive value, human review is not reliable, or the cost of
repair is not justified. Preserving the failed calibration is a scientifically valid result.

Several states may coexist—for example, evaluation design may be justified while production batching remains
blocked. There is no forced march toward generation.

## 17. Relationship to the large synthetic-data program

This protocol is intentionally detailed because later production may contain many thousands of scenes and
sentence pairs. Humans cannot review every final surface row at scale. The scalable control loop therefore
must learn from D5 which judgments belong at which level:

- humans deeply validate concept families, blueprints, hard negatives, and high-risk strata;
- economical workers generate bounded surface variation from approved structures;
- critics triage and diagnose only after calibration against human judgments;
- deterministic checks enforce contracts they can actually observe;
- rejected and disputed generations remain available for research;
- sampling probabilities and lineage make batch estimates interpretable;
- releases select immutable cohorts without rewriting the underlying ledger.

D5 is where that division of labor is measured for the first time. Scaling before this review would multiply
unknown errors; refusing to scale after a promising calibration would abandon half of Alpha's intended
research contribution.

## 18. Completion criteria

D5 human adjudication is complete only when:

- all 48 candidates have sealed Pass A and Pass B human reviews;
- all six families have a Pass C synthesis;
- all six structural rejections have separate content and schema dispositions;
- hidden-repeat consistency is reported;
- every hard finding quotes evidence and names the affected contract or behavior;
- blueprint errors and realization errors are separated;
- question necessity, style, semantic duplication, and conversational contribution are summarized;
- disagreements and expertise limits remain visible;
- candidate adjudications cite their review basis;
- one Pass D campaign closeout cites every required adjudication, synthesis, structural, repeat, and analysis
  basis and records all eight conversational-distribution assessments;
- the closeout retains schema-enforced zero execution authority;
- zero candidates are silently promoted into a release or training exposure;
- an operator-facing campaign report states what is known, unknown, and proposed next;
- any new call, generation, evaluation, release, training, or GPU work is placed in a separate bounded
  authorization.

## 19. Immediate next action

Open the live local-first workspace, honestly declare the reviewer's competence scope and session conditions,
conduct Pass A on the prepared 12 blinded candidates, download the completed packet, and import it through the
local append-only `review-submit` command. The current blank packet is SHA-256
`95b962709e9ad77aa91f2249f0648f1ee026b5ce3d64aaff792b615f751a484a`; earlier packet blobs remain preserved.
Do not begin with a model critic: the point of this phase is to create the human reference that a critic would
later be measured against. Execution 05's surface evidence may nominate comparisons but must not be shown in a
way that breaks Pass A blindness.

The dashboard now makes the campaign denominator explicit: the prepared 12-item packet is one session within
the 48-candidate Pass A census. Finishing it does not unlock Pass B or complete D5. Continue with additional
blinded Pass A sessions and the six hidden repeats according to the ledger-derived pipeline before revealing
contracts. The panel is evidence visibility only; it neither prepares assignments nor imports judgments.
The local CLI independently enforces the same campaign boundary; per-candidate Pass A completion is not enough
to prepare Pass B.

The Pass C implementation is ready but intentionally empty: zero assignments, zero family syntheses, and
zero structural dispositions. Do not bypass its A/B gate merely to populate the new public tables. The six
hidden Pass A repeats are now supported by the separate presentation-event workflow in
[Execution 07](EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md), but zero are assigned until a real human seals
the first Pass A session.

Pass D is likewise implemented and intentionally empty; see
[Execution 08](EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md). Its live prerequisite check stops at the
missing Pass A review and creates zero closeout assignments or adjudications. Do not prefill its public tables
or treat a recommended D5 state as a later-stage authorization.

All A/B/C/D importers now share one exact exported-envelope verifier; see
[Execution 12](EXECUTION-12-D5-ALL-PACKET-ENVELOPE-BINDING.md). The C and D adversarial tests prove that a
completed response attached to a timestamp-altered packet writes neither scientific rows nor raw submission
artifacts. This hardens future evidence capture but does not change the current zero-human-evidence state.

Execution 14 additionally proves that an incomplete A/B reviewer-session declaration writes neither the raw
submission nor any declaration or review row. The public explorer exposes both provenance tables read-only;
they correctly contain zero rows before a real human submission.
