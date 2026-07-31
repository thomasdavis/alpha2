import type {
  HumanReviewFinding,
  HumanReviewPacket,
  HumanReviewPass,
  HumanReviewResponse,
  HumanReviewSessionResponse,
  JsonValue
} from "./types.js";
import { canonicalPacketEnvelopeJson } from "./packet-envelope-contract.js";

export const HUMAN_REVIEW_RUBRIC_SLUG = "d5-human-adjudication";
export const HUMAN_REVIEW_RUBRIC_VERSION = 2;

export interface HumanReviewChoice {
  value: string;
  label: string;
  description: string;
}

export interface HumanReviewDimension {
  key: string;
  label: string;
  description: string;
}

export const HUMAN_REVIEW_SCORE_ANCHORS = [
  { value: 0, label: "Critical failure" },
  { value: 1, label: "Major failure" },
  { value: 2, label: "Locally repairable" },
  { value: 3, label: "Acceptable" },
  { value: 4, label: "Exemplar" }
] as const;

export const HUMAN_REVIEW_PASS_A_DIMENSIONS: readonly HumanReviewDimension[] = [
  { key: "direct_responsiveness", label: "Direct responsiveness", description: "Answers the move the user actually made." },
  { key: "conceptual_plausibility", label: "Conceptual plausibility", description: "Makes a defensible intellectual move without bluffing." },
  { key: "linguistic_naturalness", label: "Linguistic naturalness", description: "Reads as fluent, idiomatic language rather than generated scaffolding." },
  { key: "conversational_naturalness", label: "Conversational naturalness", description: "Feels situated in this exchange rather than like a detachable essay." },
  { key: "appropriate_depth_length", label: "Depth and length", description: "Uses enough detail for the moment without lecturing or truncating." },
  { key: "pedagogical_value", label: "Pedagogical value", description: "Leaves the distinction clearer or more usable." },
  { key: "desire_to_continue", label: "Desire to continue", description: "Contributes momentum without relying on a canned follow-up question." },
  { key: "substantive_value_after_style_removed", label: "Substance after style removal", description: "Still contributes a useful intellectual move when friendly phrasing is ignored." }
] as const;

export const HUMAN_REVIEW_PASS_B_DIMENSIONS: readonly HumanReviewDimension[] = [
  { key: "blueprint_validity", label: "Blueprint validity", description: "The underlying task and distinction are coherent." },
  { key: "required_commitment_coverage", label: "Required commitments", description: "The realization covers what the contract requires." },
  { key: "prohibited_commitment_safety", label: "Prohibited commitments", description: "The realization avoids the contract's forbidden implications." },
  { key: "plurality_calibration", label: "Plurality calibration", description: "Preserves genuine alternatives without inventing ambiguity." },
  { key: "linguistic_pragmatic_validity", label: "Linguistic and pragmatic validity", description: "The scenario, wording, implicature, and response policy are defensible." },
  { key: "conversational_quality", label: "Conversational quality", description: "The rendered interaction works as dialogue." },
  { key: "pedagogical_value", label: "Pedagogical value", description: "The example teaches the intended distinction efficiently." },
  { key: "metadata_schema_fit", label: "Metadata and schema fit", description: "The recorded structure accurately describes the model-visible item." },
  { key: "style_distributional_value", label: "Distributional value", description: "The item adds useful stylistic variation rather than a repeated synthetic signature." }
] as const;

export const HUMAN_REVIEW_PASS_A_OUTCOMES: readonly HumanReviewChoice[] = [
  { value: "acceptable_as_rendered", label: "Acceptable as rendered", description: "Usable without a substantive change." },
  { value: "locally_repairable", label: "Locally repairable", description: "A bounded edit could make it usable." },
  { value: "major_rewrite_needed", label: "Major rewrite needed", description: "The idea may survive, but the response should be rebuilt." },
  { value: "conceptually_invalid", label: "Conceptually invalid", description: "The intellectual move is wrong or unsupported." },
  { value: "conversationally_invalid", label: "Conversationally invalid", description: "The response fails as an interaction even if some content is sound." },
  { value: "valuable_as_negative", label: "Valuable as a negative", description: "Preserve it as a useful failure or contrast." },
  { value: "uncertain", label: "Uncertain", description: "The reviewer cannot resolve the judgment from this pass." },
  { value: "requires_expertise", label: "Requires expertise", description: "A domain-qualified reviewer is needed." }
] as const;

export const HUMAN_REVIEW_PASS_B_OUTCOMES: readonly HumanReviewChoice[] = [
  { value: "accept_as_positive", label: "Accept as positive", description: "Eligible for a positive cohort after all gates." },
  { value: "accept_as_negative", label: "Accept as negative", description: "Useful as an explicit failure or rejected continuation." },
  { value: "accept_as_ambiguous_set", label: "Accept as ambiguous set", description: "Several analyses are legitimately admissible." },
  { value: "accept_with_scope_restriction", label: "Accept with scope restriction", description: "Valid only under a recorded purpose, theory, culture, or domain." },
  { value: "repair_local", label: "Repair locally", description: "Retain the blueprint and make a bounded realization edit." },
  { value: "regenerate_from_blueprint", label: "Regenerate from blueprint", description: "The contract stands, but this realization does not." },
  { value: "revise_blueprint", label: "Revise blueprint", description: "The family contract itself needs correction." },
  { value: "split_family", label: "Split family", description: "The family conflates distinctions that should be separate." },
  { value: "merge_as_projection", label: "Merge as projection", description: "Treat this as another projection of an existing family." },
  { value: "restrict_requires_authority", label: "Restrict: authority required", description: "Do not generalize without appropriate cultural or domain authority." },
  { value: "defer_theory_disagreement", label: "Defer theory disagreement", description: "Preserve competing defensible analyses without forced resolution." },
  { value: "reject_invalid", label: "Reject: invalid", description: "Neither the item nor its proposed use is defensible." },
  { value: "reject_duplicate", label: "Reject: duplicate", description: "Adds no meaningful conceptual or surface variation." },
  { value: "reject_style", label: "Reject: style", description: "Carries an undesirable synthetic or interaction pattern." },
  { value: "reject_source_fidelity", label: "Reject: source fidelity", description: "Misstates or outruns its source evidence." },
  { value: "reject_policy", label: "Reject: policy", description: "Cannot be retained under the corpus policy." }
] as const;

export const HUMAN_REVIEW_QUESTION_POLICIES: readonly HumanReviewChoice[] = [
  { value: "necessary_before_answer", label: "Necessary before answer", description: "A responsible answer depends on clarification." },
  { value: "useful_after_partial_answer", label: "Useful after a partial answer", description: "The model should contribute first, then clarify." },
  { value: "optional_momentum", label: "Optional momentum", description: "A question could productively continue the exchange." },
  { value: "ritual_or_canned", label: "Ritual or canned", description: "The question is a learned conversational tic rather than a useful move." },
  { value: "misdirected", label: "Misdirected", description: "The question steers away from the user's actual concern." },
  { value: "not_applicable", label: "Not applicable", description: "There is no relevant follow-up-question decision here." }
] as const;

export const HUMAN_REVIEW_MISSING_CLARIFICATION: readonly HumanReviewChoice[] = [
  { value: "no", label: "No", description: "The response did not omit a necessary clarification." },
  { value: "yes_missing_clarification", label: "Yes", description: "The response answered past an ambiguity that had to be resolved." },
  { value: "uncertain", label: "Uncertain", description: "Whether clarification was required is itself unclear." },
  { value: "not_applicable", label: "Not applicable", description: "The item does not present a clarification decision." }
] as const;

export const HUMAN_REVIEW_FIRST_SENTENCE_ENGAGEMENT: readonly HumanReviewChoice[] = [
  { value: "yes", label: "Yes", description: "The first assistant sentence directly engages the user's actual move." },
  { value: "partly", label: "Partly", description: "The opening is relevant, but indirect, incomplete, or partly generic." },
  { value: "no", label: "No", description: "The opening does not directly engage the user's actual move." }
] as const;

export const HUMAN_REVIEW_ANSWERED_BEFORE_UNNECESSARY_QUESTION: readonly HumanReviewChoice[] = [
  { value: "yes", label: "Yes", description: "The assistant contributes an answer before asking anything unnecessary." },
  { value: "no", label: "No", description: "An unnecessary question precedes or replaces a useful answer." },
  { value: "not_applicable", label: "Not applicable", description: "The exchange contains no such ordering decision." }
] as const;

export const HUMAN_REVIEW_COMPETENCIES: readonly HumanReviewChoice[] = [
  { value: "conversation", label: "Conversation", description: "Natural interaction, responsiveness, momentum, and appropriateness." },
  { value: "linguistics", label: "Linguistics", description: "Language form, meaning, pragmatics, discourse, or metalinguistic analysis." },
  { value: "ontology", label: "Ontology", description: "Categories, identity, dependence, parthood, roles, events, or representation choices." },
  { value: "philosophy", label: "Philosophy", description: "Conceptual analysis, argument, counterexample, inference, or theory comparison." },
  { value: "evidence", label: "Evidence", description: "Source, testimony, attribution, uncertainty, provenance, or belief revision." },
  { value: "other", label: "Other", description: "Another relevant competence described in the accompanying note." }
] as const;

export const HUMAN_REVIEW_INTERRUPTION_STATUSES: readonly HumanReviewChoice[] = [
  { value: "none", label: "No interruption", description: "The session was completed without a meaningful interruption." },
  { value: "paused_once", label: "Paused once", description: "The reviewer took one meaningful break and then resumed." },
  { value: "paused_multiple", label: "Paused multiple times", description: "The reviewer resumed after more than one break." },
  { value: "technical_disruption", label: "Technical disruption", description: "A browser, network, device, or tooling problem interrupted review." },
  { value: "other", label: "Other interruption", description: "Another interruption described in the conditions note." }
] as const;

export const HUMAN_REVIEW_FATIGUE_LEVELS: readonly HumanReviewChoice[] = [
  { value: "none", label: "None", description: "No noticeable fatigue affected the session." },
  { value: "mild", label: "Mild", description: "Some fatigue was present but did not noticeably impair judgment." },
  { value: "moderate", label: "Moderate", description: "Fatigue may have affected attention or consistency." },
  { value: "high", label: "High", description: "Fatigue materially threatens the reliability of later judgments." },
  { value: "stopped_early", label: "Stopped early", description: "The reviewer stopped rather than forcing judgments under fatigue." }
] as const;

export function humanReviewDimensions(pass: HumanReviewPass): readonly HumanReviewDimension[] {
  return pass === "A" ? HUMAN_REVIEW_PASS_A_DIMENSIONS : HUMAN_REVIEW_PASS_B_DIMENSIONS;
}

export function humanReviewOutcomes(pass: HumanReviewPass): readonly HumanReviewChoice[] {
  return pass === "A" ? HUMAN_REVIEW_PASS_A_OUTCOMES : HUMAN_REVIEW_PASS_B_OUTCOMES;
}

export function emptyHumanReviewResponse(pass: HumanReviewPass): HumanReviewResponse {
  return {
    outcome: null,
    summaryUserAim: "",
    summaryAssistantMove: "",
    firstSentenceEngagement: null,
    answeredBeforeUnnecessaryQuestion: null,
    scores: Object.fromEntries(humanReviewDimensions(pass).map((dimension) => [dimension.key, null])),
    dimensionEvidence: Object.fromEntries(humanReviewDimensions(pass).map((dimension) => [dimension.key, ""])),
    questionPolicy: null,
    missingClarification: null,
    findings: [],
    rationale: "",
    confidence: null,
    uncertainty: "",
    expertiseNeeded: ""
  };
}

export function emptyHumanReviewSessionResponse(): HumanReviewSessionResponse {
  return {
    declaredCompetencies: [],
    competenceNote: "",
    startedAt: "",
    endedAt: "",
    interruptionStatus: null,
    fatigueLevel: null,
    conditionsNote: ""
  };
}

/**
 * Return the immutable, model-visible packet envelope. Reviewer responses are
 * the only mutable portion of a submitted packet, so they are reset to the
 * rubric's exact empty shape before comparison or hashing. Object spreads
 * deliberately preserve unknown packet/assignment fields so an injected field
 * cannot disappear during verification and accidentally compare as equal.
 */
export function humanReviewPacketEnvelope(packet: HumanReviewPacket): HumanReviewPacket {
  return {
    ...packet,
    instructions: [...packet.instructions],
    sessionResponse: emptyHumanReviewSessionResponse(),
    assignments: packet.assignments.map((assignment) => ({
      ...assignment,
      response: emptyHumanReviewResponse(packet.pass)
    }))
  };
}

function validIsoTimestamp(value: string): boolean {
  return typeof value === "string" && value.trim().length > 0 && !Number.isNaN(Date.parse(value));
}

export function humanReviewSessionResponseErrors(
  response: HumanReviewSessionResponse,
  options: { requireEndedAt?: boolean } = {}
): string[] {
  const errors: string[] = [];
  if (!isRecord(response)) return ["Session declaration is missing."];
  const allowedCompetencies = new Set(HUMAN_REVIEW_COMPETENCIES.map((choice) => choice.value));
  if (!Array.isArray(response.declaredCompetencies) || response.declaredCompetencies.length < 1
    || response.declaredCompetencies.some((value) => typeof value !== "string" || !allowedCompetencies.has(value))
    || new Set(response.declaredCompetencies).size !== response.declaredCompetencies.length) {
    errors.push("Declare at least one valid reviewer competence without duplicates.");
  }
  if (response.declaredCompetencies?.includes("other")
    && (typeof response.competenceNote !== "string" || response.competenceNote.trim().length < 1)) {
    errors.push("Describe the other declared competence.");
  }
  if (!validIsoTimestamp(response.startedAt)) errors.push("Record a valid session start time.");
  if (options.requireEndedAt && !validIsoTimestamp(response.endedAt)) {
    errors.push("Record a valid session end time.");
  }
  if (validIsoTimestamp(response.startedAt) && validIsoTimestamp(response.endedAt)
    && Date.parse(response.endedAt) < Date.parse(response.startedAt)) {
    errors.push("Session end time cannot precede its start time.");
  }
  if (!allowedValue(response.interruptionStatus, HUMAN_REVIEW_INTERRUPTION_STATUSES)) {
    errors.push("Record the session interruption status.");
  }
  if (!allowedValue(response.fatigueLevel, HUMAN_REVIEW_FATIGUE_LEVELS)) {
    errors.push("Record the session fatigue level.");
  }
  if (typeof response.competenceNote !== "string" || typeof response.conditionsNote !== "string") {
    errors.push("Session notes must be text.");
  }
  return errors;
}

/** Browser-safe canonical form; this module intentionally has no Node imports. */
export function humanReviewPacketEnvelopeJson(packet: HumanReviewPacket): string {
  return canonicalPacketEnvelopeJson(humanReviewPacketEnvelope(packet) as unknown as JsonValue);
}

export function humanReviewPacketMatchesEnvelope(
  candidate: HumanReviewPacket,
  exported: HumanReviewPacket
): boolean {
  return humanReviewPacketEnvelopeJson(candidate) === humanReviewPacketEnvelopeJson(exported);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function allowedValue(value: string | null, choices: readonly HumanReviewChoice[]): boolean {
  return value !== null && choices.some((choice) => choice.value === value);
}

function findingErrors(finding: HumanReviewFinding, label: string): string[] {
  const errors: string[] = [];
  if (!isRecord(finding) || typeof finding.dimension !== "string" || finding.dimension.trim().length < 1) {
    errors.push(`${label} needs a dimension.`);
  }
  if (!isRecord(finding) || !["observation", "minor", "major", "critical"].includes(finding.severity)) {
    errors.push(`${label} has an invalid severity.`);
  }
  if (!isRecord(finding) || typeof finding.evidence !== "string" || finding.evidence.trim().length < 1) {
    errors.push(`${label} needs exact evidence.`);
  }
  if (!isRecord(finding) || typeof finding.whyItMatters !== "string" || finding.whyItMatters.trim().length < 1) {
    errors.push(`${label} needs an explanation of why it matters.`);
  }
  if (!isRecord(finding) || typeof finding.recommendation !== "string" || finding.recommendation.trim().length < 1) {
    errors.push(`${label} needs the smallest plausible repair.`);
  }
  if (!isRecord(finding) || typeof finding.preserve !== "string" || finding.preserve.trim().length < 1) {
    errors.push(`${label} needs what the repair must preserve.`);
  }
  return errors;
}

export function humanReviewResponseErrors(
  pass: HumanReviewPass,
  response: HumanReviewResponse,
  opaqueItemId = "Item"
): string[] {
  const errors: string[] = [];
  if (!isRecord(response)) return [`${opaqueItemId} response is missing.`];
  if (!allowedValue(response.outcome, humanReviewOutcomes(pass))) errors.push(`${opaqueItemId} needs an outcome.`);

  const expectedDimensions = humanReviewDimensions(pass).map((dimension) => dimension.key).sort();
  const actualDimensions = isRecord(response.scores) ? Object.keys(response.scores).sort() : [];
  if (actualDimensions.join("\0") !== expectedDimensions.join("\0")) {
    errors.push(`${opaqueItemId} score dimensions differ from rubric version ${HUMAN_REVIEW_RUBRIC_VERSION}.`);
  } else {
    for (const dimension of expectedDimensions) {
      const score = response.scores[dimension];
      const validNumeric = typeof score === "number" && Number.isInteger(score) && score >= 0 && score <= 4;
      if (!validNumeric && score !== "not_applicable" && score !== "uncertain") {
        errors.push(`${opaqueItemId} needs a 0–4, not-applicable, or uncertain assessment for ${dimension}.`);
      }
    }
  }
  const actualEvidenceDimensions = isRecord(response.dimensionEvidence)
    ? Object.keys(response.dimensionEvidence).sort()
    : [];
  if (actualEvidenceDimensions.join("\0") !== expectedDimensions.join("\0")) {
    errors.push(`${opaqueItemId} evidence dimensions differ from rubric version ${HUMAN_REVIEW_RUBRIC_VERSION}.`);
  } else {
    for (const dimension of expectedDimensions) {
      const evidence = response.dimensionEvidence[dimension];
      if (typeof evidence !== "string" || evidence.trim().length < 1) {
        errors.push(`${opaqueItemId} needs one-sentence evidence for ${dimension}.`);
      }
    }
  }
  if (!allowedValue(response.questionPolicy, HUMAN_REVIEW_QUESTION_POLICIES)) {
    errors.push(`${opaqueItemId} needs a follow-up question judgment.`);
  }
  if (!allowedValue(response.missingClarification, HUMAN_REVIEW_MISSING_CLARIFICATION)) {
    errors.push(`${opaqueItemId} needs a missing-clarification judgment.`);
  }
  if (typeof response.rationale !== "string" || response.rationale.trim().length < 1) {
    errors.push(`${opaqueItemId} needs a rationale.`);
  }
  if (response.confidence === null || !Number.isInteger(response.confidence)
    || response.confidence < 0 || response.confidence > 4) {
    errors.push(`${opaqueItemId} needs a 0–4 confidence score.`);
  }
  if (pass === "A") {
    if (typeof response.summaryUserAim !== "string" || response.summaryUserAim.trim().length < 1) {
      errors.push(`${opaqueItemId} needs a summary of the user's aim.`);
    }
    if (typeof response.summaryAssistantMove !== "string" || response.summaryAssistantMove.trim().length < 1) {
      errors.push(`${opaqueItemId} needs a summary of the assistant's move.`);
    }
    if (!allowedValue(response.firstSentenceEngagement, HUMAN_REVIEW_FIRST_SENTENCE_ENGAGEMENT)) {
      errors.push(`${opaqueItemId} needs a first-sentence engagement judgment.`);
    }
    if (!allowedValue(
      response.answeredBeforeUnnecessaryQuestion,
      HUMAN_REVIEW_ANSWERED_BEFORE_UNNECESSARY_QUESTION
    )) {
      errors.push(`${opaqueItemId} needs an answer-before-unnecessary-question judgment.`);
    }
  } else if (response.firstSentenceEngagement !== null || response.answeredBeforeUnnecessaryQuestion !== null) {
    errors.push(`${opaqueItemId} contains Pass A comprehension judgments in Pass B.`);
  }
  if (!Array.isArray(response.findings)) {
    errors.push(`${opaqueItemId} findings must be a list.`);
  } else {
    response.findings.forEach((finding, index) => errors.push(...findingErrors(finding, `${opaqueItemId} finding ${index + 1}`)));
  }
  return errors;
}

export function parseHumanReviewPacketText(text: string): HumanReviewPacket {
  const parsed = JSON.parse(text) as unknown;
  if (!isRecord(parsed) || parsed.schemaVersion !== 1 || typeof parsed.campaignSlug !== "string"
    || typeof parsed.sessionId !== "string" || (parsed.pass !== "A" && parsed.pass !== "B")
    || typeof parsed.reviewerAlias !== "string" || typeof parsed.rubricSlug !== "string"
    || typeof parsed.rubricVersion !== "number" || typeof parsed.seed !== "string"
    || typeof parsed.createdAt !== "string" || !Array.isArray(parsed.instructions)
    || !parsed.instructions.every((instruction) => typeof instruction === "string")
    || !Array.isArray(parsed.assignments)) {
    throw new Error("Human-review submission does not match packet schema version 1");
  }
  const sessionResponse = parsed["sessionResponse"];
  if (sessionResponse === undefined) {
    // Early v1 review packets predate the additive reviewer-session declaration.
    // Preserve their exact artifact bytes while normalizing them to an explicitly
    // incomplete declaration in memory. Submission validation still requires the
    // reviewer to complete every declaration field before any evidence is written.
    parsed["sessionResponse"] = emptyHumanReviewSessionResponse();
  } else if (!isRecord(sessionResponse)
    || !Array.isArray(sessionResponse["declaredCompetencies"])
    || !sessionResponse["declaredCompetencies"].every((value) => typeof value === "string")
    || typeof sessionResponse["competenceNote"] !== "string"
    || typeof sessionResponse["startedAt"] !== "string"
    || typeof sessionResponse["endedAt"] !== "string"
    || !(sessionResponse["interruptionStatus"] === null
      || typeof sessionResponse["interruptionStatus"] === "string")
    || !(sessionResponse["fatigueLevel"] === null
      || typeof sessionResponse["fatigueLevel"] === "string")
    || typeof sessionResponse["conditionsNote"] !== "string") {
    throw new Error("Human-review submission does not match packet schema version 1");
  }
  for (const assignment of parsed.assignments) {
    if (!isRecord(assignment) || typeof assignment.assignmentId !== "string"
      || ("presentationId" in assignment && typeof assignment.presentationId !== "string")
      || typeof assignment.opaqueItemId !== "string" || typeof assignment.candidateContentSha256 !== "string"
      || !("candidate" in assignment) || !isRecord(assignment.response)) {
      throw new Error("Human-review submission contains an invalid assignment");
    }
    const response = assignment.response;
    const empty = emptyHumanReviewResponse(parsed.pass);
    if (!("firstSentenceEngagement" in response)) response["firstSentenceEngagement"] = null;
    if (!("answeredBeforeUnnecessaryQuestion" in response)) {
      response["answeredBeforeUnnecessaryQuestion"] = null;
    }
    if (!("dimensionEvidence" in response)) response["dimensionEvidence"] = empty.dimensionEvidence;
    if (Array.isArray(response["findings"])) {
      for (const finding of response["findings"]) {
        if (!isRecord(finding)) continue;
        if (!("whyItMatters" in finding)) finding["whyItMatters"] = "";
        if (!("preserve" in finding)) finding["preserve"] = "";
      }
    }
  }
  return parsed as unknown as HumanReviewPacket;
}
