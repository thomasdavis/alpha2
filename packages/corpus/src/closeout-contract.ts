import { HUMAN_REVIEW_PASS_B_OUTCOMES } from "./review-contract.js";
import type { JsonValue } from "./types.js";

export const CAMPAIGN_CLOSEOUT_RUBRIC_SLUG = "d5-campaign-closeout";
export const CAMPAIGN_CLOSEOUT_RUBRIC_VERSION = 1;

export const D5_CLOSEOUT_STATES = [
  "D5_REPAIR_REQUIRED",
  "D5_CRITIC_CALIBRATION_JUSTIFIED",
  "D5_BATCHING_PROBE_JUSTIFIED",
  "D5_EVALUATION_DESIGN_JUSTIFIED",
  "D5_STOP"
] as const;

export const FAILURE_LOCI = [
  "blueprint",
  "realization",
  "schema",
  "style",
  "review",
  "source_or_authority",
  "distribution"
] as const;

export const FAILURE_SEVERITIES = ["observation", "minor", "major", "critical"] as const;
export const NEW_CALL_REQUIREMENTS = ["no", "possibly_later", "yes_if_separately_authorized"] as const;

export const CAMPAIGN_DISTRIBUTION_DIMENSIONS = [
  "first_sentence_directness",
  "question_behavior",
  "length_appropriateness",
  "lecture_drift",
  "canned_signatures",
  "multi_turn_reuse",
  "desire_to_continue",
  "substantive_value_after_style_scrub"
] as const;

export type D5CloseoutState = typeof D5_CLOSEOUT_STATES[number];
export type FailureLocus = typeof FAILURE_LOCI[number];
export type FailureSeverity = typeof FAILURE_SEVERITIES[number];
export type NewCallRequirement = typeof NEW_CALL_REQUIREMENTS[number];
export type DistributionDimension = typeof CAMPAIGN_DISTRIBUTION_DIMENSIONS[number];

export interface CampaignCloseoutCandidateEvidence {
  candidateId: string;
  candidateVersionId: string;
  familyVersionId: string;
  familySlug: string;
  status: string;
  contentSha256: string;
  passAReviewId: string;
  passBReviewId: string;
  passAReview: JsonValue;
  passBReview: JsonValue;
  familySynthesisId: string;
  structuralDispositionId: string | null;
  structuralDisposition: JsonValue | null;
}

export interface CampaignCloseoutFamilyEvidence {
  familyVersionId: string;
  familySlug: string;
  familySynthesisId: string;
  disposition: string;
  synthesis: JsonValue;
}

export interface CampaignCloseoutRepeatEvidence {
  presentationId: string;
  repeatResponseId: string;
  sourceReviewId: string;
  candidateVersionId: string;
  outcomeMatch: number;
  questionPolicyMatch: number;
  missingClarificationMatch: number;
  confidenceDelta: number;
  dimensionExactRate: number;
  meanAbsoluteScoreDelta: number;
}

export interface CampaignCloseoutAnalysisEvidence {
  analysisRunId: string;
  inputSnapshotSha256: string;
  metricCount: number;
  similarityEdgeCount: number;
  templateSignatureCount: number;
}

export interface CandidateAdjudicationResponse {
  candidateVersionId: string;
  outcome: string | null;
  rationale: string;
  confidence: number | null;
  uncertainty: string;
  repairRequest: string;
  preserve: string[];
  disagreementDescription: string;
}

export interface CampaignFailureClusterResponse {
  clusterKey: string;
  label: string;
  locus: FailureLocus | null;
  severity: FailureSeverity | null;
  proposedRepair: string;
  newCallsNeeded: NewCallRequirement | null;
  rationale: string;
  members: Array<{ memberKind: string; memberId: string }>;
}

export interface CampaignDistributionResponse {
  dimension: DistributionDimension;
  assessment: string;
  evidenceIds: string[];
}

export interface CampaignStateResponse {
  state: D5CloseoutState | null;
  rationale: string;
}

export interface CampaignCloseoutResponse {
  recommendationSummary: string;
  candidateDispositions: CandidateAdjudicationResponse[];
  failureClusters: CampaignFailureClusterResponse[];
  noFailureClustersRationale: string;
  distributionAssessments: CampaignDistributionResponse[];
  recommendedStates: CampaignStateResponse[];
  known: string[];
  unknown: string[];
  proposedNext: string[];
  disagreements: string[];
  noDisagreementRationale: string;
  overallRationale: string;
  confidence: number | null;
  executionAuthorizationAcknowledgement: string;
}

export interface CampaignCloseoutPacket {
  schemaVersion: 1;
  campaignSlug: string;
  sessionId: string;
  adjudicatorAlias: string;
  rubricSlug: string;
  rubricVersion: number;
  inputSnapshotSha256: string;
  createdAt: string;
  population: {
    candidates: number;
    families: number;
    structurallyRejected: number;
    completedRepeatPresentations: number;
    expectedRepeatPresentations: number;
  };
  candidates: CampaignCloseoutCandidateEvidence[];
  families: CampaignCloseoutFamilyEvidence[];
  repeats: CampaignCloseoutRepeatEvidence[];
  analysis: CampaignCloseoutAnalysisEvidence;
  response: CampaignCloseoutResponse;
}

export function emptyCampaignCloseoutResponse(
  candidates: CampaignCloseoutCandidateEvidence[]
): CampaignCloseoutResponse {
  return {
    recommendationSummary: "",
    candidateDispositions: candidates.map((candidate) => ({
      candidateVersionId: candidate.candidateVersionId,
      outcome: null,
      rationale: "",
      confidence: null,
      uncertainty: "",
      repairRequest: "",
      preserve: [],
      disagreementDescription: ""
    })),
    failureClusters: [],
    noFailureClustersRationale: "",
    distributionAssessments: CAMPAIGN_DISTRIBUTION_DIMENSIONS.map((dimension) => ({
      dimension,
      assessment: "",
      evidenceIds: []
    })),
    recommendedStates: [{ state: null, rationale: "" }],
    known: [],
    unknown: [],
    proposedNext: [],
    disagreements: [],
    noDisagreementRationale: "",
    overallRationale: "",
    confidence: null,
    executionAuthorizationAcknowledgement: "non_binding_no_execution_authority"
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function stringListErrors(value: unknown, label: string, allowEmpty = false): string[] {
  if (!Array.isArray(value) || !value.every(nonempty)) return [`${label} must contain only nonempty strings.`];
  if (!allowEmpty && value.length === 0) return [`${label} must not be empty.`];
  return [];
}

export function campaignCloseoutResponseErrors(packet: CampaignCloseoutPacket): string[] {
  const errors: string[] = [];
  const response = packet.response;
  if (!nonempty(response.recommendationSummary)) errors.push("Closeout needs a recommendation summary.");
  const expectedCandidates = packet.candidates.map((candidate) => candidate.candidateVersionId).sort();
  const actualCandidates = response.candidateDispositions.map((candidate) => candidate.candidateVersionId).sort();
  if (expectedCandidates.join("\0") !== actualCandidates.join("\0")
    || new Set(actualCandidates).size !== actualCandidates.length) {
    errors.push("Closeout needs exactly one disposition for every frozen candidate version.");
  }
  const allowedOutcomes = new Set(HUMAN_REVIEW_PASS_B_OUTCOMES.map((outcome) => outcome.value));
  const repairOutcomes = new Set(["repair_local", "regenerate_from_blueprint", "revise_blueprint", "split_family"]);
  for (const candidate of response.candidateDispositions) {
    if (candidate.outcome === null || !allowedOutcomes.has(candidate.outcome)) {
      errors.push(`Candidate ${candidate.candidateVersionId} needs a PRD-04 disposition.`);
    }
    if (!nonempty(candidate.rationale)) errors.push(`Candidate ${candidate.candidateVersionId} needs a rationale.`);
    if (candidate.confidence === null || !Number.isInteger(candidate.confidence)
      || candidate.confidence < 0 || candidate.confidence > 4) {
      errors.push(`Candidate ${candidate.candidateVersionId} needs confidence 0-4.`);
    }
    if (candidate.outcome !== null && repairOutcomes.has(candidate.outcome)
      && !nonempty(candidate.repairRequest)) {
      errors.push(`Candidate ${candidate.candidateVersionId} needs a repair request for ${candidate.outcome}.`);
    }
    if (candidate.outcome === "defer_theory_disagreement" && !nonempty(candidate.disagreementDescription)) {
      errors.push(`Candidate ${candidate.candidateVersionId} needs a disagreement description.`);
    }
    errors.push(...stringListErrors(candidate.preserve, `Candidate ${candidate.candidateVersionId} preserve list`, true));
  }
  const evidenceIds = new Set<string>([
    ...packet.candidates.map((candidate) => candidate.candidateVersionId),
    ...packet.candidates.flatMap((candidate) => [candidate.passAReviewId, candidate.passBReviewId]),
    ...packet.families.map((family) => family.familyVersionId),
    ...packet.families.map((family) => family.familySynthesisId),
    ...packet.candidates.flatMap((candidate) => candidate.structuralDispositionId === null
      ? [] : [candidate.structuralDispositionId]),
    ...packet.repeats.flatMap((repeat) => [repeat.presentationId, repeat.repeatResponseId, repeat.sourceReviewId]),
    packet.analysis.analysisRunId
  ]);
  const clusterKeys = new Set<string>();
  for (const cluster of response.failureClusters) {
    if (!nonempty(cluster.clusterKey) || clusterKeys.has(cluster.clusterKey)) {
      errors.push("Every failure cluster needs a unique nonempty key.");
    }
    clusterKeys.add(cluster.clusterKey);
    if (!nonempty(cluster.label) || !nonempty(cluster.proposedRepair) || !nonempty(cluster.rationale)) {
      errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} needs label, repair, and rationale.`);
    }
    if (cluster.locus === null || !FAILURE_LOCI.includes(cluster.locus)) {
      errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} needs a valid locus.`);
    }
    if (cluster.severity === null || !FAILURE_SEVERITIES.includes(cluster.severity)) {
      errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} needs a valid severity.`);
    }
    if (cluster.newCallsNeeded === null || !NEW_CALL_REQUIREMENTS.includes(cluster.newCallsNeeded)) {
      errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} needs a call requirement.`);
    }
    if (!Array.isArray(cluster.members) || cluster.members.length === 0) {
      errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} needs evidence members.`);
    } else {
      const memberKeys = new Set<string>();
      for (const member of cluster.members) {
        const key = `${member.memberKind}:${member.memberId}`;
        if (!["candidate_version", "family_version", "review", "family_synthesis", "structural_disposition"]
          .includes(member.memberKind) || !evidenceIds.has(member.memberId) || memberKeys.has(key)) {
          errors.push(`Failure cluster ${cluster.clusterKey || "<blank>"} has an invalid or duplicate member ${key}.`);
        }
        memberKeys.add(key);
      }
    }
  }
  if (response.failureClusters.length === 0 && !nonempty(response.noFailureClustersRationale)) {
    errors.push("An empty failure-cluster register needs an explicit rationale.");
  }
  const expectedDimensions = [...CAMPAIGN_DISTRIBUTION_DIMENSIONS].sort();
  const actualDimensions = response.distributionAssessments.map((entry) => entry.dimension).sort();
  if (expectedDimensions.join("\0") !== actualDimensions.join("\0")
    || new Set(actualDimensions).size !== actualDimensions.length) {
    errors.push("Closeout needs exactly one assessment for every conversational-distribution dimension.");
  }
  for (const assessment of response.distributionAssessments) {
    if (!nonempty(assessment.assessment)) errors.push(`Distribution ${assessment.dimension} needs an assessment.`);
    if (!Array.isArray(assessment.evidenceIds)
      || assessment.evidenceIds.some((id) => !evidenceIds.has(id))) {
      errors.push(`Distribution ${assessment.dimension} has invalid evidence IDs.`);
    }
  }
  if (response.recommendedStates.length === 0) errors.push("Closeout needs at least one recommended D5 state.");
  const states = new Set<string>();
  for (const state of response.recommendedStates) {
    if (state.state === null || !D5_CLOSEOUT_STATES.includes(state.state) || states.has(state.state)) {
      errors.push("Recommended D5 states must be valid and unique.");
    }
    if (!nonempty(state.rationale)) errors.push(`Recommended state ${state.state ?? "<blank>"} needs a rationale.`);
    if (state.state !== null) states.add(state.state);
  }
  errors.push(...stringListErrors(response.known, "Known findings"));
  errors.push(...stringListErrors(response.unknown, "Unknown findings"));
  errors.push(...stringListErrors(response.proposedNext, "Proposed next actions"));
  errors.push(...stringListErrors(response.disagreements, "Disagreements", true));
  if (response.disagreements.length === 0 && !nonempty(response.noDisagreementRationale)) {
    errors.push("No disagreements requires an explicit rationale.");
  }
  if (!nonempty(response.overallRationale)) errors.push("Closeout needs an overall rationale.");
  if (response.confidence === null || !Number.isInteger(response.confidence)
    || response.confidence < 0 || response.confidence > 4) {
    errors.push("Closeout needs confidence 0-4.");
  }
  if (response.executionAuthorizationAcknowledgement !== "non_binding_no_execution_authority") {
    errors.push("Closeout must acknowledge that it grants no execution authority.");
  }
  return errors;
}

export function parseCampaignCloseoutPacketText(text: string): CampaignCloseoutPacket {
  const parsed = JSON.parse(text) as unknown;
  if (!isRecord(parsed) || parsed.schemaVersion !== 1 || !nonempty(parsed.campaignSlug)
    || !nonempty(parsed.sessionId) || !nonempty(parsed.adjudicatorAlias)
    || !nonempty(parsed.rubricSlug) || typeof parsed.rubricVersion !== "number"
    || !nonempty(parsed.inputSnapshotSha256) || !nonempty(parsed.createdAt)
    || !isRecord(parsed.population) || !Array.isArray(parsed.candidates)
    || !Array.isArray(parsed.families) || !Array.isArray(parsed.repeats)
    || !isRecord(parsed.analysis) || !isRecord(parsed.response)) {
    throw new Error("Campaign-closeout submission does not match packet schema version 1");
  }
  return parsed as unknown as CampaignCloseoutPacket;
}

export function closeoutContractDefinition(): JsonValue {
  return {
    slug: CAMPAIGN_CLOSEOUT_RUBRIC_SLUG,
    version: CAMPAIGN_CLOSEOUT_RUBRIC_VERSION,
    candidateOutcomes: HUMAN_REVIEW_PASS_B_OUTCOMES.map((outcome) => outcome.value),
    closeoutStates: D5_CLOSEOUT_STATES,
    failureLoci: FAILURE_LOCI,
    failureSeverities: FAILURE_SEVERITIES,
    newCallRequirements: NEW_CALL_REQUIREMENTS,
    distributionDimensions: CAMPAIGN_DISTRIBUTION_DIMENSIONS,
    authority: "non_binding_no_execution_authority"
  } as unknown as JsonValue;
}
