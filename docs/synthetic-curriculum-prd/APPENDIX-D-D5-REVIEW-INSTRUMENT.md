# Appendix D — D5 human review instrument

**Use with:** PRD-12

**Population:** all 48 `alpha-calibration-v1` candidates

**Rule:** complete and seal the blind form before viewing the contract-aware form

**Executable provenance:** Execution 14 makes the reviewer/session header a required A/B packet response and
stores it append-only in `human_review_session_declaration` plus normalized competence rows. A missing or
incomplete header causes the local importer to write nothing.

## 1. Reviewer and session header

| Field | Entry |
|---|---|
| reviewer actor/alias | |
| declared competence | conversation / linguistics / ontology / philosophy / evidence / other |
| session ID | |
| rubric version/hash | |
| session start/end | |
| presentation index | |
| opaque item ID | |
| hidden repeat? | recorded by system, hidden from reviewer |
| interruption/fatigue note | |

Do not put private contact details or credentials in the ledger. Use an operator-controlled actor alias.

## 2. Pass A — blind model-visible review

### 2.1 Immediate comprehension

1. In one sentence, what is the user trying to resolve?
2. In one sentence, what intellectual move does the assistant make?
3. Did the first assistant sentence directly engage the user's move?
   - `yes`
   - `partly`
   - `no`
4. Did the assistant answer before asking for anything unnecessary?
   - `yes`
   - `no`
   - `not_applicable`

### 2.2 Blind outcome

Choose one provisional outcome:

- `acceptable_as_rendered`;
- `locally_repairable`;
- `major_rewrite_needed`;
- `conceptually_invalid`;
- `conversationally_invalid`;
- `valuable_as_negative`;
- `uncertain`;
- `requires_expertise`.

### 2.3 Blind dimension scores

Use 0–4 or `not_applicable`/`uncertain`:

| Dimension | Score | One-sentence evidence |
|---|---:|---|
| direct responsiveness | | |
| conceptual plausibility | | |
| linguistic naturalness | | |
| conversational naturalness | | |
| appropriate depth/length | | |
| pedagogical value | | |
| desire to continue | | |
| substantive value after style is removed | | |

### 2.4 Question policy

If the assistant asks a question, classify it:

- `necessary_before_answer`;
- `useful_after_partial_answer`;
- `optional_momentum`;
- `ritual_or_canned`;
- `misdirected`.

If it asks no question, was a clarification necessary before answering?

- `no`;
- `yes_missing_clarification`;
- `uncertain`.

### 2.5 Blind findings

For each finding:

| Field | Entry |
|---|---|
| dimension | |
| severity | observation / minor / major / critical |
| exact quoted evidence | |
| why it matters | |
| smallest plausible repair | |
| what must be preserved | |

### 2.6 Blind confidence

- confidence: 0–4;
- reason for uncertainty;
- expertise needed, if any.

Seal Pass A before continuing.

## 3. Pass B — contract-aware review

The system now reveals family, title, hidden contract, intended response policy, difficulty, lenses,
transformation, generator notes, and deterministic findings.

### 3.1 Blueprint validity

| Question | Answer | Evidence/rationale |
|---|---|---|
| Is the underlying distinction worth teaching? | yes / scoped / no / disputed | |
| Are required commitments defensible? | all / some / none / disputed | |
| Are prohibited commitments truly prohibited? | all / some / none / disputed | |
| Are admissible analyses complete enough? | yes / undercoverage / overcoverage / disputed | |
| Is discriminating evidence actually discriminating? | yes / partly / no | |
| Is the competency question clear and useful? | yes / repair / no | |
| Does the contract impose one theory where plurality is legitimate? | no / yes / uncertain | |

Record blueprint findings independently of the assistant prose.

### 3.2 Realization-to-contract checks

For every required commitment:

| Commitment | Expressed | Correct | Natural | Notes |
|---|---|---|---|---|
| | yes / implicit / no | yes / no / disputed | yes / no | |

For every prohibited commitment:

| Commitment | Violated | Exact evidence | Severity |
|---|---|---|---|
| | yes / no / uncertain | | |

For every preserve/change instruction:

| Instruction | Satisfied | Evidence |
|---|---|---|
| | yes / partly / no | |

### 3.3 Plurality and evidence

- Important analyses retained:
- Unsupported analyses introduced:
- Analysis unjustifiably collapsed:
- Missing evidence correctly named:
- Source/report/endorsement boundaries preserved:
- Clarification would resolve the issue: `yes` / `partly` / `no` / `not_applicable`.

### 3.4 Metadata fit

| Field | Judgment | Proposed repair |
|---|---|---|
| primary lens | correct / alternative / invalid | |
| secondary lenses | correct / mixed field types / missing category / invalid | |
| transformation | correct / alternative / invalid | |
| difficulty | correct / too low / too high | |
| response policy | useful / overprescriptive / canned / mismatched | |
| kind | correct / alternative | |

### 3.5 Pass B outcome

Choose the recommended scientific disposition:

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

Then record:

- concise rationale;
- exact review IDs or evidence spans supporting it;
- confidence 0–4;
- whether Pass B changed Pass A;
- why the hidden contract changed or did not change the judgment.

## 4. Structural-rejection addendum

Complete this for any item later revealed as `structurally_rejected`:

| Question | Entry |
|---|---|
| Is the content independently useful? | yes / repairable / no / uncertain |
| Was the validator finding factually correct? | yes / no / partly |
| Unknown value | |
| Correct semantic type | conceptual lens / transformation / response policy / discourse operation / other |
| Correct remedy | metadata repair / taxonomy proposal / field split / prompt repair / keep rejected |
| Would automatic acceptance have hidden a schema problem? | |
| Would automatic rejection discard valuable content? | |

Do not label this a critic false reject. It is a structural-disposition comparison until an actual critic has
made a prediction.

## 5. Pass C — family synthesis

Complete once for each eight-candidate family.

### 5.1 Family identity

| Field | Entry |
|---|---|
| family slug/version | |
| reviewer/adjudicator | |
| candidate review IDs | |
| family purpose | |
| central distinction in plain language | |

### 5.2 Coverage matrix

Mark which candidates genuinely realize each pressure:

| Pressure | Candidate IDs | Adequacy | Missing work |
|---|---|---|---|
| simple positive | | | |
| hard negative | | | |
| borderline/plural case | | | |
| minimal meaning change | | | |
| local repair | | | |
| delayed reuse | | | |
| cross-domain projection | | | |
| false bridge | | | |
| answer-and-stop | | | |
| necessary clarification | | | |

Not every pilot family must contain every pressure. Empty cells reveal coverage, not automatic failure.

### 5.3 Family diagnosis

- strongest candidate and why;
- weakest candidate and why;
- semantic duplicates after nouns/names are removed;
- shared conceptual error;
- shared style signature;
- response-policy imbalance;
- metadata/taxonomy mismatch;
- blueprint repair with the highest descendant leverage;
- candidate(s) worth preserving only as negatives;
- family-level uncertainty or theory disagreement.

### 5.4 Family disposition

- `retain_blueprint`;
- `retain_with_local_repairs`;
- `revise_blueprint`;
- `split_family`;
- `merge_or_reframe`;
- `restrict_requires_expert`;
- `retire_family`;
- `contested`.

This is a recommendation. Campaign authorization remains an operator decision.

## 6. Pass D — campaign synthesis worksheet

### 6.1 Population accounting

| Metric | Count/rate | Family-clustered interpretation |
|---|---:|---|
| reviewed candidates | | |
| acceptable as rendered | | |
| locally repairable | | |
| blueprint revision needed | | |
| useful negatives | | |
| invalid | | |
| disputed/requires expertise | | |
| structural disposition disagreements | | |
| hidden-repeat agreement | | |
| Pass A/B changes | | |

### 6.2 Failure-cluster register

| Cluster | Families/candidates | Locus | Severity | Proposed repair | New calls needed? |
|---|---|---|---|---|---|
| | | blueprint / realization / schema / style / review | | | |

### 6.3 Conversational-distribution register

- first-sentence directness;
- answer-before-question behavior;
- necessary/useful/ritual question distribution;
- length appropriateness, not merely raw length;
- lecture drift;
- canned distinctions or closers;
- multi-turn reuse versus re-explanation;
- desire-to-continue distribution;
- substantive value after style scrubbing.

### 6.4 Recommended D5 states

Check and justify any that apply:

- `D5_REPAIR_REQUIRED`;
- `D5_CRITIC_CALIBRATION_JUSTIFIED`;
- `D5_BATCHING_PROBE_JUSTIFIED`;
- `D5_EVALUATION_DESIGN_JUSTIFIED`;
- `D5_STOP`.

No checked state is authorization. Attach Decision Packet 01 and obtain a separate operator decision.

Execution 08 turns this worksheet into a content-addressed local packet only after A, hidden repeats, B, C,
structural disposition, and authoritative surface evidence are complete for the adjudicator. The packet
requires every distribution dimension and recommended state to be justified. Its SQL schema fixes execution
authority at zero, so completing the worksheet still cannot start generation, release, or training.

## 7. Suggested ledger mapping

| Form element | Ledger record |
|---|---|
| reviewer/session identity | `actor`, raw form artifact |
| reviewer competence scope and session conditions | `human_review_session_declaration` |
| normalized declared competences | `human_review_session_competence` |
| protocol version | `rubric`, `rubric_version` |
| presentation and blinding | `review_assignment.blindness_json` |
| Pass A or Pass B outcome | `review` |
| numeric dimension | `review_dimension_score` |
| quoted problem/repair | `review_finding` |
| conflicting judgments | `disagreement_case` |
| requested local change | `repair_request` |
| final candidate disposition | `adjudication`, `adjudication_basis` |
| Pass D workflow and submitted synthesis | `campaign_closeout_assignment`, `campaign_closeout` |
| recommended D5 state | `campaign_closeout_state` |
| campaign-level evidence lineage | `campaign_closeout_basis` |
| failure cluster and cited members | `campaign_failure_cluster`, `campaign_failure_cluster_member` |
| conversational-distribution assessment | `campaign_distribution_assessment` |
| lifecycle promotion/restriction | `quality_state_transition` |
| original submitted form | content-addressed `raw_artifact`/`blob` |

Each Pass A and Pass B should be a distinct review record so the effect of revealing the hidden contract can
be measured. The reviewer must never overwrite the Pass A rationale after seeing Pass B.

## 8. Review completion checklist

- [ ] Reviewer identity, competence scope, session start/end, interruption, fatigue, and material conditions declared.
- [ ] Pass A completed without forbidden metadata.
- [ ] Pass A sealed before contract reveal.
- [ ] Pass B checks every required/prohibited commitment.
- [ ] Blueprint and realization judged separately.
- [ ] Questions classified by necessity.
- [ ] Findings quote exact evidence.
- [ ] Repair states what must remain invariant.
- [ ] Confidence and expertise limits recorded.
- [ ] Structural addendum completed if applicable.
- [ ] Raw form stored and hashed.
- [ ] Ledger rows point to candidate and rubric versions.
- [ ] No release or training state changed automatically.
