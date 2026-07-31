import type { JsonValue } from "./types.js";
import { canonicalPacketEnvelopeJson } from "./packet-envelope-contract.js";

export const FAMILY_SYNTHESIS_RUBRIC_SLUG = "d5-family-synthesis";
export const FAMILY_SYNTHESIS_RUBRIC_VERSION = 1;

export const FAMILY_COVERAGE_PRESSURES = [
  "simple_positive",
  "hard_negative",
  "borderline_or_plural_case",
  "minimal_meaning_change",
  "local_repair",
  "delayed_reuse",
  "cross_domain_projection",
  "false_bridge",
  "answer_and_stop",
  "necessary_clarification"
] as const;

export const FAMILY_SYNTHESIS_DISPOSITIONS = [
  "retain_blueprint",
  "retain_with_local_repairs",
  "revise_blueprint",
  "split_family",
  "merge_or_reframe",
  "restrict_requires_expert",
  "retire_family",
  "contested"
] as const;

export const COVERAGE_ADEQUACY = ["adequate", "partial", "missing", "not_applicable"] as const;
export const STRUCTURAL_CONTENT_UTILITY = ["useful", "repairable", "not_useful", "uncertain"] as const;
export const VALIDATOR_FINDING_CORRECTNESS = ["yes", "no", "partly", "uncertain"] as const;
export const STRUCTURAL_SEMANTIC_TYPES = [
  "conceptual_lens",
  "transformation",
  "response_policy",
  "discourse_operation",
  "unmodeled_category",
  "other"
] as const;
export const STRUCTURAL_REMEDIES = [
  "metadata_repair",
  "taxonomy_proposal",
  "field_split",
  "prompt_repair",
  "keep_rejected",
  "other"
] as const;

export interface FamilyCoverageResponse {
  pressure: string;
  candidateVersionIds: string[];
  adequacy: string | null;
  missingWork: string;
}

export interface SemanticDuplicateGroup {
  candidateVersionIds: string[];
  rationale: string;
}

export interface FamilySynthesisResponse {
  disposition: string | null;
  centralDistinction: string;
  coverage: FamilyCoverageResponse[];
  strongestCandidateVersionId: string | null;
  strongestCandidateRationale: string;
  weakestCandidateVersionId: string | null;
  weakestCandidateRationale: string;
  semanticDuplicateGroups: SemanticDuplicateGroup[];
  sharedConceptualError: string;
  sharedStyleSignature: string;
  responsePolicyImbalance: string;
  metadataTaxonomyMismatch: string;
  highestLeverageBlueprintRepair: string;
  negativeCandidateVersionIds: string[];
  uncertaintyOrTheoryDisagreement: string;
  rationale: string;
  confidence: number | null;
}

export interface StructuralDispositionResponse {
  candidateVersionId: string;
  contentUtility: string | null;
  validatorFindingCorrectness: string | null;
  identifiedValue: string;
  semanticType: string | null;
  remedy: string | null;
  automaticAcceptanceHazard: string;
  automaticRejectionHazard: string;
  rationale: string;
  confidence: number | null;
}

export interface FamilySynthesisReviewEvidence {
  reviewId: string;
  pass: "A" | "B";
  outcome: string;
  rationale: JsonValue;
  scores: Record<string, number>;
  findings: Array<{
    dimension: string;
    severity: string;
    evidence: string;
    recommendation: string;
  }>;
}

export interface FamilySynthesisCandidate {
  candidateVersionId: string;
  candidateContentSha256: string;
  structuralStatus: string;
  item: JsonValue;
  failures: Array<{ failureId: string; code: string; detail: string }>;
  reviews: FamilySynthesisReviewEvidence[];
}

export interface FamilySynthesisPacketAssignment {
  assignmentId: string;
  familyVersionId: string;
  familySlug: string;
  familyVersion: number;
  familyInputSnapshotSha256: string;
  familyPurpose: string;
  familyBlueprint: JsonValue;
  candidates: FamilySynthesisCandidate[];
  response: FamilySynthesisResponse;
  structuralDispositions: StructuralDispositionResponse[];
}

export interface FamilySynthesisPacket {
  schemaVersion: 1;
  kind: "d5_family_synthesis_packet";
  campaignSlug: string;
  sessionId: string;
  reviewerAlias: string;
  rubricSlug: string;
  rubricVersion: number;
  inputSnapshotSha256: string;
  createdAt: string;
  instructions: string[];
  assignments: FamilySynthesisPacketAssignment[];
}

export function emptyFamilySynthesisResponse(): FamilySynthesisResponse {
  return {
    disposition: null,
    centralDistinction: "",
    coverage: FAMILY_COVERAGE_PRESSURES.map((pressure) => ({
      pressure,
      candidateVersionIds: [],
      adequacy: null,
      missingWork: ""
    })),
    strongestCandidateVersionId: null,
    strongestCandidateRationale: "",
    weakestCandidateVersionId: null,
    weakestCandidateRationale: "",
    semanticDuplicateGroups: [],
    sharedConceptualError: "",
    sharedStyleSignature: "",
    responsePolicyImbalance: "",
    metadataTaxonomyMismatch: "",
    highestLeverageBlueprintRepair: "",
    negativeCandidateVersionIds: [],
    uncertaintyOrTheoryDisagreement: "",
    rationale: "",
    confidence: null
  };
}

export function emptyStructuralDisposition(candidateVersionId: string): StructuralDispositionResponse {
  return {
    candidateVersionId,
    contentUtility: null,
    validatorFindingCorrectness: null,
    identifiedValue: "",
    semanticType: null,
    remedy: null,
    automaticAcceptanceHazard: "",
    automaticRejectionHazard: "",
    rationale: "",
    confidence: null
  };
}

export function familySynthesisPacketEnvelope(packet: FamilySynthesisPacket): FamilySynthesisPacket {
  return {
    ...packet,
    instructions: [...packet.instructions],
    assignments: packet.assignments.map((assignment) => ({
      ...assignment,
      response: emptyFamilySynthesisResponse(),
      structuralDispositions: assignment.structuralDispositions.map((disposition) =>
        emptyStructuralDisposition(disposition.candidateVersionId))
    }))
  };
}

export function familySynthesisPacketEnvelopeJson(packet: FamilySynthesisPacket): string {
  return canonicalPacketEnvelopeJson(familySynthesisPacketEnvelope(packet) as unknown as JsonValue);
}

export function familySynthesisPacketMatchesEnvelope(
  candidate: FamilySynthesisPacket,
  exported: FamilySynthesisPacket
): boolean {
  return familySynthesisPacketEnvelopeJson(candidate) === familySynthesisPacketEnvelopeJson(exported);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonEmpty(value: unknown): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

function allowed(value: unknown, values: readonly string[]): boolean {
  return typeof value === "string" && values.includes(value);
}

function validReferenceList(value: unknown, allowedIds: Set<string>, minimum = 0): value is string[] {
  return Array.isArray(value) && value.length >= minimum
    && value.every((entry) => typeof entry === "string" && allowedIds.has(entry))
    && new Set(value).size === value.length;
}

export function familySynthesisAssignmentErrors(
  assignment: FamilySynthesisPacketAssignment
): string[] {
  const label = assignment.familySlug || assignment.assignmentId || "Family";
  const errors: string[] = [];
  const candidateIds = new Set(assignment.candidates.map((candidate) => candidate.candidateVersionId));
  const rejectedIds = new Set(
    assignment.candidates
      .filter((candidate) => candidate.structuralStatus === "structurally_rejected")
      .map((candidate) => candidate.candidateVersionId)
  );
  const response = assignment.response;
  if (!isRecord(response)) return [`${label} synthesis response is missing.`];
  if (!allowed(response.disposition, FAMILY_SYNTHESIS_DISPOSITIONS)) {
    errors.push(`${label} needs a family disposition.`);
  }
  if (!nonEmpty(response.centralDistinction)) errors.push(`${label} needs a plain-language central distinction.`);
  if (!Array.isArray(response.coverage)
    || response.coverage.map((entry) => entry.pressure).join("\0") !== FAMILY_COVERAGE_PRESSURES.join("\0")) {
    errors.push(`${label} coverage rows differ from rubric version ${FAMILY_SYNTHESIS_RUBRIC_VERSION}.`);
  } else {
    for (const entry of response.coverage) {
      if (!validReferenceList(entry.candidateVersionIds, candidateIds)) {
        errors.push(`${label} coverage ${entry.pressure} cites a candidate outside the family or repeats one.`);
      }
      if (!allowed(entry.adequacy, COVERAGE_ADEQUACY)) {
        errors.push(`${label} coverage ${entry.pressure} needs an adequacy judgment.`);
      }
      if ((entry.adequacy === "partial" || entry.adequacy === "missing") && !nonEmpty(entry.missingWork)) {
        errors.push(`${label} coverage ${entry.pressure} needs missing-work detail.`);
      }
    }
  }
  if (typeof response.strongestCandidateVersionId !== "string"
    || !candidateIds.has(response.strongestCandidateVersionId)) {
    errors.push(`${label} needs a strongest candidate from this family.`);
  }
  if (!nonEmpty(response.strongestCandidateRationale)) errors.push(`${label} needs a strongest-candidate rationale.`);
  if (typeof response.weakestCandidateVersionId !== "string"
    || !candidateIds.has(response.weakestCandidateVersionId)) {
    errors.push(`${label} needs a weakest candidate from this family.`);
  }
  if (!nonEmpty(response.weakestCandidateRationale)) errors.push(`${label} needs a weakest-candidate rationale.`);
  if (assignment.candidates.length > 1
    && response.strongestCandidateVersionId === response.weakestCandidateVersionId) {
    errors.push(`${label} strongest and weakest candidates must differ.`);
  }
  if (!Array.isArray(response.semanticDuplicateGroups)) {
    errors.push(`${label} semantic duplicate groups must be a list.`);
  } else {
    for (const [index, group] of response.semanticDuplicateGroups.entries()) {
      if (!isRecord(group) || !validReferenceList(group.candidateVersionIds, candidateIds, 2)
        || !nonEmpty(group.rationale)) {
        errors.push(`${label} semantic duplicate group ${index + 1} needs at least two family candidates and a rationale.`);
      }
    }
  }
  for (const [field, description] of [
    [response.sharedConceptualError, "shared conceptual-error diagnosis"],
    [response.sharedStyleSignature, "shared style-signature diagnosis"],
    [response.responsePolicyImbalance, "response-policy balance diagnosis"],
    [response.metadataTaxonomyMismatch, "metadata/taxonomy diagnosis"],
    [response.highestLeverageBlueprintRepair, "highest-leverage blueprint repair"],
    [response.uncertaintyOrTheoryDisagreement, "uncertainty or theory-disagreement statement"],
    [response.rationale, "family rationale"]
  ] as const) {
    if (!nonEmpty(field)) errors.push(`${label} needs a ${description}; use an explicit 'none observed' when applicable.`);
  }
  if (!validReferenceList(response.negativeCandidateVersionIds, candidateIds)) {
    errors.push(`${label} negative candidates must be unique members of this family.`);
  }
  if (response.confidence === null || !Number.isInteger(response.confidence)
    || response.confidence < 0 || response.confidence > 4) {
    errors.push(`${label} needs a 0–4 confidence score.`);
  }

  if (!Array.isArray(assignment.structuralDispositions)) {
    errors.push(`${label} structural dispositions must be a list.`);
  } else {
    const dispositionIds = assignment.structuralDispositions.map((entry) => entry.candidateVersionId);
    if (new Set(dispositionIds).size !== dispositionIds.length
      || dispositionIds.length !== rejectedIds.size
      || dispositionIds.some((id) => !rejectedIds.has(id))) {
      errors.push(`${label} must contain exactly one structural disposition for every rejected family candidate.`);
    }
    for (const disposition of assignment.structuralDispositions) {
      const itemLabel = `${label} structural disposition ${disposition.candidateVersionId}`;
      if (!allowed(disposition.contentUtility, STRUCTURAL_CONTENT_UTILITY)) {
        errors.push(`${itemLabel} needs a content-utility judgment.`);
      }
      if (!allowed(disposition.validatorFindingCorrectness, VALIDATOR_FINDING_CORRECTNESS)) {
        errors.push(`${itemLabel} needs a validator-finding judgment.`);
      }
      if (!nonEmpty(disposition.identifiedValue)) errors.push(`${itemLabel} needs the value being classified.`);
      if (!allowed(disposition.semanticType, STRUCTURAL_SEMANTIC_TYPES)) {
        errors.push(`${itemLabel} needs a semantic type.`);
      }
      if (!allowed(disposition.remedy, STRUCTURAL_REMEDIES)) errors.push(`${itemLabel} needs a remedy.`);
      if (!nonEmpty(disposition.automaticAcceptanceHazard)) errors.push(`${itemLabel} needs an automatic-acceptance hazard.`);
      if (!nonEmpty(disposition.automaticRejectionHazard)) errors.push(`${itemLabel} needs an automatic-rejection hazard.`);
      if (!nonEmpty(disposition.rationale)) errors.push(`${itemLabel} needs a rationale.`);
      if (disposition.confidence === null || !Number.isInteger(disposition.confidence)
        || disposition.confidence < 0 || disposition.confidence > 4) {
        errors.push(`${itemLabel} needs a 0–4 confidence score.`);
      }
    }
  }
  return errors;
}

export function parseFamilySynthesisPacketText(text: string): FamilySynthesisPacket {
  const parsed = JSON.parse(text) as unknown;
  if (!isRecord(parsed) || parsed.schemaVersion !== 1 || parsed.kind !== "d5_family_synthesis_packet"
    || typeof parsed.campaignSlug !== "string" || typeof parsed.sessionId !== "string"
    || typeof parsed.reviewerAlias !== "string" || typeof parsed.rubricSlug !== "string"
    || typeof parsed.rubricVersion !== "number" || typeof parsed.inputSnapshotSha256 !== "string"
    || typeof parsed.createdAt !== "string" || !Array.isArray(parsed.instructions)
    || !parsed.instructions.every((instruction) => typeof instruction === "string")
    || !Array.isArray(parsed.assignments)) {
    throw new Error("Family-synthesis submission does not match packet schema version 1");
  }
  for (const assignment of parsed.assignments) {
    if (!isRecord(assignment) || typeof assignment.assignmentId !== "string"
      || typeof assignment.familyVersionId !== "string" || typeof assignment.familySlug !== "string"
      || typeof assignment.familyVersion !== "number" || typeof assignment.familyInputSnapshotSha256 !== "string"
      || typeof assignment.familyPurpose !== "string" || !("familyBlueprint" in assignment)
      || !Array.isArray(assignment.candidates) || !isRecord(assignment.response)
      || !Array.isArray(assignment.structuralDispositions)) {
      throw new Error("Family-synthesis submission contains an invalid assignment");
    }
  }
  return parsed as unknown as FamilySynthesisPacket;
}
