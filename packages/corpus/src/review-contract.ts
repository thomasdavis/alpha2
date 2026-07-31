import type {
  HumanReviewFinding,
  HumanReviewPacket,
  HumanReviewPass,
  HumanReviewResponse,
  JsonValue
} from "./types.js";
import { canonicalPacketEnvelopeJson } from "./packet-envelope-contract.js";

export const HUMAN_REVIEW_RUBRIC_SLUG = "d5-human-adjudication";
export const HUMAN_REVIEW_RUBRIC_VERSION = 1;

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
    scores: Object.fromEntries(humanReviewDimensions(pass).map((dimension) => [dimension.key, null])),
    questionPolicy: null,
    missingClarification: null,
    findings: [],
    rationale: "",
    confidence: null,
    uncertainty: "",
    expertiseNeeded: ""
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
    assignments: packet.assignments.map((assignment) => ({
      ...assignment,
      response: emptyHumanReviewResponse(packet.pass)
    }))
  };
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
  if (!isRecord(finding) || typeof finding.recommendation !== "string" || finding.recommendation.trim().length < 1) {
    errors.push(`${label} needs a recommendation.`);
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
      if (score === null || !Number.isInteger(score) || score < 0 || score > 4) {
        errors.push(`${opaqueItemId} needs a 0–4 score for ${dimension}.`);
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
  for (const assignment of parsed.assignments) {
    if (!isRecord(assignment) || typeof assignment.assignmentId !== "string"
      || ("presentationId" in assignment && typeof assignment.presentationId !== "string")
      || typeof assignment.opaqueItemId !== "string" || typeof assignment.candidateContentSha256 !== "string"
      || !("candidate" in assignment) || !isRecord(assignment.response)) {
      throw new Error("Human-review submission contains an invalid assignment");
    }
  }
  return parsed as unknown as HumanReviewPacket;
}
