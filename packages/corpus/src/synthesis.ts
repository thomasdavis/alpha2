import type { InValue } from "@libsql/client";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { putBlob, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import {
  ensureHumanActor,
  ensureHumanReviewRubric,
  requireHumanActor,
  requireHumanReviewRubric
} from "./review.js";
import { requireExportedPacketEnvelope } from "./packet-envelope.js";
import { writeAtomic } from "./storage.js";
import {
  COVERAGE_ADEQUACY,
  emptyFamilySynthesisResponse,
  emptyStructuralDisposition,
  FAMILY_COVERAGE_PRESSURES,
  FAMILY_SYNTHESIS_DISPOSITIONS,
  FAMILY_SYNTHESIS_RUBRIC_SLUG,
  FAMILY_SYNTHESIS_RUBRIC_VERSION,
  familySynthesisPacketEnvelopeJson,
  familySynthesisAssignmentErrors,
  parseFamilySynthesisPacketText,
  STRUCTURAL_CONTENT_UTILITY,
  STRUCTURAL_REMEDIES,
  STRUCTURAL_SEMANTIC_TYPES,
  VALIDATOR_FINDING_CORRECTNESS,
  type FamilySynthesisCandidate,
  type FamilySynthesisPacket,
  type FamilySynthesisPacketAssignment,
  type FamilySynthesisReviewEvidence
} from "./synthesis-contract.js";
import type { JsonValue } from "./types.js";

interface FamilyEvidence {
  familyVersionId: string;
  familySlug: string;
  familyVersion: number;
  familyPurpose: string;
  familyBlueprint: JsonValue;
  candidates: FamilySynthesisCandidate[];
  familyInputSnapshotSha256: string;
}

interface SynthesisAssignmentRow {
  assignmentId: string;
  familyVersionId: string;
  sessionId: string;
  inputSnapshotSha256: string;
  createdAt: string;
}

export interface PreparedFamilySynthesis {
  packetPath: string;
  markdownPath: string;
  sessionId: string;
  familyCount: number;
  candidateCount: number;
  structuralDispositionCount: number;
  packetSha256: string;
  resumed: boolean;
}

export interface SubmittedFamilySynthesis {
  sessionId: string;
  familySyntheses: number;
  structuralDispositions: number;
  packetEnvelopeSha256: string;
  submissionSha256: string;
}

export interface FamilySynthesisStatus {
  campaignSlug: string;
  assignments: Record<string, number>;
  familySyntheses: number;
  structuralDispositions: number;
  familySynthesisArtifacts: number;
  releaseMembers: number;
  trainingExposures: number;
}

const FAMILY_SYNTHESIS_RUBRIC_DEFINITION = {
  slug: FAMILY_SYNTHESIS_RUBRIC_SLUG,
  version: FAMILY_SYNTHESIS_RUBRIC_VERSION,
  pass: "C",
  pressures: FAMILY_COVERAGE_PRESSURES,
  coverageAdequacy: COVERAGE_ADEQUACY,
  familyDispositions: FAMILY_SYNTHESIS_DISPOSITIONS,
  structuralContentUtility: STRUCTURAL_CONTENT_UTILITY,
  validatorFindingCorrectness: VALIDATOR_FINDING_CORRECTNESS,
  structuralSemanticTypes: STRUCTURAL_SEMANTIC_TYPES,
  structuralRemedies: STRUCTURAL_REMEDIES,
  rule: "Family synthesis is human evidence after sealed Pass A and Pass B. It cannot promote a candidate, create a release, or authorize training."
} as const;

function now(): string {
  return new Date().toISOString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseJsonValue(text: string, label: string): JsonValue {
  const value = JSON.parse(text) as unknown;
  if (value === undefined) throw new Error(`${label} is not JSON`);
  return value as JsonValue;
}

function parseJsonRecord(text: string, label: string): Record<string, JsonValue> {
  const value = parseJsonValue(text, label);
  if (!isRecord(value)) throw new Error(`${label} must be a JSON object`);
  return value as Record<string, JsonValue>;
}

function exactWrite(path: string, bytes: string): void {
  if (existsSync(path)) {
    const existing = readFileSync(path);
    if (sha256Bytes(existing) !== sha256Bytes(bytes)) {
      throw new Error(`Refusing to replace an edited or non-identical synthesis packet at ${path}`);
    }
    return;
  }
  writeAtomic(path, bytes);
}

async function campaignId(ledger: Ledger, slug: string): Promise<string> {
  const result = await ledger.client.execute({
    sql: "SELECT id FROM generation_campaign WHERE slug = ?",
    args: [slug]
  });
  if (result.rows.length !== 1) throw new Error(`Unknown campaign ${slug}`);
  return String(result.rows[0]!["id"]);
}

async function ensureFamilySynthesisRubric(ledger: Ledger): Promise<string> {
  const rubricId = stableId("rubric", FAMILY_SYNTHESIS_RUBRIC_SLUG);
  const definitionJson = canonicalJson(FAMILY_SYNTHESIS_RUBRIC_DEFINITION as unknown as JsonValue);
  const digest = sha256Bytes(definitionJson);
  const versionId = stableId(
    "rubricv",
    `${FAMILY_SYNTHESIS_RUBRIC_SLUG}:${FAMILY_SYNTHESIS_RUBRIC_VERSION}:${digest}`
  );
  const ts = now();
  await ledger.client.batch([
    {
      sql: "INSERT OR IGNORE INTO rubric(id, slug, created_at) VALUES (?, ?, ?)",
      args: [rubricId, FAMILY_SYNTHESIS_RUBRIC_SLUG, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO rubric_version
            (id, rubric_id, version, definition_json, content_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [versionId, rubricId, FAMILY_SYNTHESIS_RUBRIC_VERSION, definitionJson, digest, ts]
    }
  ], "write");
  const stored = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM rubric_version WHERE id = ?",
    args: [versionId]
  });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["content_sha256"]) !== digest) {
    throw new Error("Stored D5 family-synthesis rubric differs from the executable definition");
  }
  return versionId;
}

async function requireFamilySynthesisRubric(ledger: Ledger): Promise<string> {
  const definitionJson = canonicalJson(FAMILY_SYNTHESIS_RUBRIC_DEFINITION as unknown as JsonValue);
  const digest = sha256Bytes(definitionJson);
  const versionId = stableId(
    "rubricv",
    `${FAMILY_SYNTHESIS_RUBRIC_SLUG}:${FAMILY_SYNTHESIS_RUBRIC_VERSION}:${digest}`
  );
  const stored = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM rubric_version WHERE id = ?",
    args: [versionId]
  });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["content_sha256"]) !== digest) {
    throw new Error("D5 family-synthesis rubric was not registered by packet preparation");
  }
  return versionId;
}

async function reviewEvidence(
  ledger: Ledger,
  candidateVersionId: string,
  actorId: string,
  candidateRubricVersionId: string
): Promise<FamilySynthesisReviewEvidence[]> {
  const reviews = await ledger.client.execute({
    sql: `SELECT r.id AS review_id, r.outcome, r.rationale,
                 json_extract(ra.blindness_json, '$.pass') AS review_pass
          FROM review_assignment ra
          JOIN review r ON json_extract(r.rationale, '$.assignmentId') = ra.id
          WHERE ra.candidate_version_id = ? AND ra.reviewer_actor_id = ?
            AND ra.rubric_version_id = ? AND ra.status = 'completed'
            AND json_extract(ra.blindness_json, '$.pass') IN ('A', 'B')
          ORDER BY review_pass`,
    args: [candidateVersionId, actorId, candidateRubricVersionId]
  });
  const result: FamilySynthesisReviewEvidence[] = [];
  for (const row of reviews.rows) {
    const reviewId = String(row["review_id"]);
    const scores = await ledger.client.execute({
      sql: `SELECT dimension, score FROM review_dimension_score
            WHERE review_id = ? ORDER BY dimension`,
      args: [reviewId]
    });
    const findings = await ledger.client.execute({
      sql: `SELECT dimension, severity, evidence, recommendation FROM review_finding
            WHERE review_id = ? ORDER BY id`,
      args: [reviewId]
    });
    result.push({
      reviewId,
      pass: String(row["review_pass"]) as "A" | "B",
      outcome: String(row["outcome"]),
      rationale: parseJsonValue(String(row["rationale"]), `review ${reviewId} rationale`),
      scores: Object.fromEntries(scores.rows.map((score) => [String(score["dimension"]), Number(score["score"])])),
      findings: findings.rows.map((finding) => ({
        dimension: String(finding["dimension"]),
        severity: String(finding["severity"]),
        evidence: String(finding["evidence"]),
        recommendation: String(finding["recommendation"])
      }))
    });
  }
  const passes = result.map((review) => review.pass).sort().join("");
  if (result.length !== 2 || passes !== "AB") {
    throw new Error(`Candidate ${candidateVersionId} needs exactly one sealed Pass A and Pass B review before Pass C`);
  }
  return result;
}

function evidencePayload(evidence: Omit<FamilyEvidence, "familyInputSnapshotSha256">): JsonValue {
  return {
    familyVersionId: evidence.familyVersionId,
    familySlug: evidence.familySlug,
    familyVersion: evidence.familyVersion,
    familyPurpose: evidence.familyPurpose,
    familyBlueprint: evidence.familyBlueprint,
    candidates: evidence.candidates
  } as unknown as JsonValue;
}

function assignmentEvidencePayload(assignment: FamilySynthesisPacketAssignment): JsonValue {
  return {
    familyVersionId: assignment.familyVersionId,
    familySlug: assignment.familySlug,
    familyVersion: assignment.familyVersion,
    familyPurpose: assignment.familyPurpose,
    familyBlueprint: assignment.familyBlueprint,
    candidates: assignment.candidates
  } as unknown as JsonValue;
}

async function loadFamilyEvidence(
  ledger: Ledger,
  campaignSlug: string,
  actorId: string,
  candidateRubricVersionId: string
): Promise<FamilyEvidence[]> {
  const rows = await ledger.client.execute({
    sql: `SELECT cc.candidate_version_id, cc.content_sha256, cc.status, cc.content_json,
                 cc.hidden_contract_json, cc.family_id, cc.family_slug,
                 fv.id AS family_version_id, fv.version AS family_version, fv.blueprint_json
          FROM corpus_candidate_current cc
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          JOIN family_version fv ON fv.family_id = cc.family_id
            AND fv.version = (SELECT MAX(latest.version) FROM family_version latest WHERE latest.family_id = cc.family_id)
          WHERE gc.slug = ?
          ORDER BY cc.family_slug, cc.candidate_id`,
    args: [campaignSlug]
  });
  if (rows.rows.length === 0) throw new Error(`Campaign ${campaignSlug} has no current candidates`);
  const grouped = new Map<string, Omit<FamilyEvidence, "familyInputSnapshotSha256">>();
  for (const row of rows.rows) {
    const candidateVersionId = String(row["candidate_version_id"]);
    const content = parseJsonRecord(String(row["content_json"]), `candidate ${candidateVersionId} content`);
    const hiddenContract = parseJsonValue(
      String(row["hidden_contract_json"]),
      `candidate ${candidateVersionId} hidden contract`
    );
    const failureRows = await ledger.client.execute({
      sql: `SELECT id, code, detail FROM candidate_failure
            WHERE candidate_id = (SELECT candidate_id FROM candidate_version WHERE id = ?)
            ORDER BY id`,
      args: [candidateVersionId]
    });
    const candidate: FamilySynthesisCandidate = {
      candidateVersionId,
      candidateContentSha256: String(row["content_sha256"]),
      structuralStatus: String(row["status"]),
      item: { ...content, hiddenContract } as JsonValue,
      failures: failureRows.rows.map((failure) => ({
        failureId: String(failure["id"]),
        code: String(failure["code"]),
        detail: String(failure["detail"])
      })),
      reviews: await reviewEvidence(ledger, candidateVersionId, actorId, candidateRubricVersionId)
    };
    const familyVersionId = String(row["family_version_id"]);
    let family = grouped.get(familyVersionId);
    if (!family) {
      const blueprint = parseJsonValue(String(row["blueprint_json"]), `family ${familyVersionId} blueprint`);
      const purpose = isRecord(blueprint) && typeof blueprint["purpose"] === "string"
        ? blueprint["purpose"]
        : "Purpose absent from this family blueprint.";
      family = {
        familyVersionId,
        familySlug: String(row["family_slug"]),
        familyVersion: Number(row["family_version"]),
        familyPurpose: purpose,
        familyBlueprint: blueprint,
        candidates: []
      };
      grouped.set(familyVersionId, family);
    }
    family.candidates.push(candidate);
  }
  return [...grouped.values()].map((family) => ({
    ...family,
    familyInputSnapshotSha256: sha256Bytes(canonicalJson(evidencePayload(family)))
  }));
}

function overallSnapshotSha256(
  campaignSlug: string,
  reviewerAlias: string,
  evidence: FamilyEvidence[]
): string {
  return sha256Bytes(canonicalJson({
    schemaVersion: 1,
    campaignSlug,
    reviewerAlias,
    families: evidence.map((family) => ({
      familyVersionId: family.familyVersionId,
      familyInputSnapshotSha256: family.familyInputSnapshotSha256
    }))
  } as unknown as JsonValue));
}

async function openSynthesisAssignments(
  ledger: Ledger,
  campaignIdValue: string,
  actorId: string,
  rubricVersionId: string
): Promise<SynthesisAssignmentRow[]> {
  const rows = await ledger.client.execute({
    sql: `SELECT id, family_version_id, session_id, input_snapshot_sha256, created_at
          FROM family_synthesis_assignment
          WHERE campaign_id = ? AND reviewer_actor_id = ? AND rubric_version_id = ? AND status = 'assigned'
          ORDER BY family_version_id`,
    args: [campaignIdValue, actorId, rubricVersionId]
  });
  return rows.rows.map((row) => ({
    assignmentId: String(row["id"]),
    familyVersionId: String(row["family_version_id"]),
    sessionId: String(row["session_id"]),
    inputSnapshotSha256: String(row["input_snapshot_sha256"]),
    createdAt: String(row["created_at"])
  }));
}

function packetInstructions(): string[] {
  return [
    "Pass A and Pass B are sealed. Compare all siblings at family level; do not edit prior reviews.",
    "Judge semantic contribution after names and nouns are removed. Surface similarity is diagnostic evidence, not a duplicate oracle.",
    "Use explicit 'none observed' when a required diagnosis has no finding; do not leave required fields blank.",
    "Complete one structural disposition for every structurally rejected candidate. This is not critic calibration.",
    "A family disposition is a recommendation only and cannot create a dataset release or training exposure."
  ];
}

function renderMarkdown(packet: FamilySynthesisPacket): string {
  const lines = [
    "# Alpha D5 human family synthesis — Pass C",
    "",
    `- Campaign: \`${packet.campaignSlug}\``,
    `- Session: \`${packet.sessionId}\``,
    `- Reviewer: \`${packet.reviewerAlias}\``,
    `- Families: ${packet.assignments.length}`,
    `- Input snapshot: \`${packet.inputSnapshotSha256}\``,
    "",
    "> This packet records no synthesis until the completed JSON is validated and submitted locally.",
    "",
    "## Instructions",
    "",
    ...packet.instructions.map((instruction) => `- ${instruction}`)
  ];
  for (const assignment of packet.assignments) {
    lines.push("", `## ${assignment.familySlug}`, "", assignment.familyPurpose, "");
    lines.push(`Candidates: ${assignment.candidates.length}; structural addenda: ${assignment.structuralDispositions.length}.`, "");
    for (const candidate of assignment.candidates) {
      lines.push(`### ${candidate.candidateVersionId}`, "", `Status: ${candidate.structuralStatus}`, "");
      const item = candidate.item as Record<string, unknown>;
      const messages = item["messages"];
      if (Array.isArray(messages)) {
        for (const message of messages) {
          if (isRecord(message)) lines.push(`**${String(message["role"])}:** ${String(message["content"])}`, "");
        }
      }
      lines.push(`Sealed reviews: ${candidate.reviews.map((review) => `${review.pass}:${review.outcome}`).join("; ")}`, "");
    }
    lines.push(`Complete the matching response for \`${assignment.assignmentId}\` in the JSON packet.`);
  }
  return `${lines.join("\n")}\n`;
}

async function recordPacketExport(
  ledger: Ledger,
  packet: FamilySynthesisPacket,
  packetPath: string,
  markdownPath: string
): Promise<string> {
  const packetText = canonicalJson(packet as unknown as JsonValue);
  const markdown = renderMarkdown(packet);
  exactWrite(packetPath, packetText);
  exactWrite(markdownPath, markdown);
  const packetSha = await putBlob(ledger, packetText, "application/json");
  const markdownSha = await putBlob(ledger, markdown, "text/markdown");
  const ts = now();
  await ledger.client.batch([
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'human_family_synthesis_packet_json', ?, ?, ?)`,
      args: [stableId("export", `family-synthesis:${packet.sessionId}:json:${packetSha}`), packetSha,
        canonicalJson({ path: packetPath, pass: "C", sessionId: packet.sessionId } as JsonValue), ts]
    },
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'human_family_synthesis_packet_markdown', ?, ?, ?)`,
      args: [stableId("export", `family-synthesis:${packet.sessionId}:markdown:${markdownSha}`), markdownSha,
        canonicalJson({ path: markdownPath, pass: "C", sessionId: packet.sessionId } as JsonValue), ts]
    }
  ], "write");
  return packetSha;
}

export async function prepareFamilySynthesisPacket(
  ledger: Ledger,
  options: { campaignSlug: string; reviewerAlias: string; outputDirectory?: string }
): Promise<PreparedFamilySynthesis> {
  const campaignIdValue = await campaignId(ledger, options.campaignSlug);
  const actorId = await ensureHumanActor(ledger, options.reviewerAlias);
  const candidateRubricVersionId = await ensureHumanReviewRubric(ledger);
  const evidence = await loadFamilyEvidence(
    ledger,
    options.campaignSlug,
    actorId,
    candidateRubricVersionId
  );
  const rubricVersionId = await ensureFamilySynthesisRubric(ledger);
  let assignments = await openSynthesisAssignments(
    ledger,
    campaignIdValue,
    actorId,
    rubricVersionId
  );
  const resumed = assignments.length > 0;
  if (resumed && assignments.length !== evidence.length) {
    throw new Error("Open Pass C assignments do not cover every current family");
  }
  let sessionId: string;
  let createdAt: string;
  if (resumed) {
    sessionId = assignments[0]!.sessionId;
    createdAt = assignments[0]!.createdAt;
    if (assignments.some((assignment) => assignment.sessionId !== sessionId)) {
      throw new Error("Open Pass C assignments span multiple sessions");
    }
    for (const assignment of assignments) {
      const family = evidence.find((entry) => entry.familyVersionId === assignment.familyVersionId);
      if (!family || family.familyInputSnapshotSha256 !== assignment.inputSnapshotSha256) {
        throw new Error(`Pass C input changed for family version ${assignment.familyVersionId}`);
      }
    }
  } else {
    const completed = await ledger.client.execute({
      sql: `SELECT COUNT(*) AS count FROM family_synthesis_assignment
            WHERE campaign_id = ? AND reviewer_actor_id = ? AND rubric_version_id = ? AND status = 'completed'`,
      args: [campaignIdValue, actorId, rubricVersionId]
    });
    if (Number(completed.rows[0]!["count"]) > 0) {
      throw new Error("Pass C is already completed for this reviewer and campaign");
    }
    sessionId = `family_synthesis_session_${randomUUID()}`;
    createdAt = now();
    assignments = evidence.map((family) => ({
      assignmentId: stableId(
        "family_synthesis_assignment",
        `${campaignIdValue}:${family.familyVersionId}:${actorId}:${rubricVersionId}`
      ),
      familyVersionId: family.familyVersionId,
      sessionId,
      inputSnapshotSha256: family.familyInputSnapshotSha256,
      createdAt
    }));
    await ledger.client.batch(assignments.map((assignment) => ({
      sql: `INSERT INTO family_synthesis_assignment
            (id, campaign_id, family_version_id, reviewer_actor_id, rubric_version_id, session_id,
             input_snapshot_sha256, blindness_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'assigned', ?, ?)`,
      args: [assignment.assignmentId, campaignIdValue, assignment.familyVersionId, actorId, rubricVersionId,
        sessionId, assignment.inputSnapshotSha256, canonicalJson({
          pass: "C",
          sessionId,
          visibleFields: ["family_blueprint", "candidate_content", "structural_status", "sealed_pass_a", "sealed_pass_b", "validator_findings", "surface_diagnostics"],
          hiddenFields: ["other_family_syntheses", "campaign_adjudication"]
        } as JsonValue), createdAt, createdAt]
    })), "write");
  }

  const packetAssignments: FamilySynthesisPacketAssignment[] = assignments.map((assignment) => {
    const family = evidence.find((entry) => entry.familyVersionId === assignment.familyVersionId)!;
    return {
      assignmentId: assignment.assignmentId,
      familyVersionId: family.familyVersionId,
      familySlug: family.familySlug,
      familyVersion: family.familyVersion,
      familyInputSnapshotSha256: family.familyInputSnapshotSha256,
      familyPurpose: family.familyPurpose,
      familyBlueprint: family.familyBlueprint,
      candidates: family.candidates,
      response: emptyFamilySynthesisResponse(),
      structuralDispositions: family.candidates
        .filter((candidate) => candidate.structuralStatus === "structurally_rejected")
        .map((candidate) => emptyStructuralDisposition(candidate.candidateVersionId))
    };
  });
  const inputSnapshotSha256 = overallSnapshotSha256(
    options.campaignSlug,
    options.reviewerAlias.trim(),
    evidence
  );
  const packet: FamilySynthesisPacket = {
    schemaVersion: 1,
    kind: "d5_family_synthesis_packet",
    campaignSlug: options.campaignSlug,
    sessionId,
    reviewerAlias: options.reviewerAlias.trim(),
    rubricSlug: FAMILY_SYNTHESIS_RUBRIC_SLUG,
    rubricVersion: FAMILY_SYNTHESIS_RUBRIC_VERSION,
    inputSnapshotSha256,
    createdAt,
    instructions: packetInstructions(),
    assignments: packetAssignments
  };
  const directory = resolve(options.outputDirectory
    ?? join(ledger.paths.releases, "review", `${options.campaignSlug}-c-${sessionId}`));
  const packetPath = join(directory, "family-synthesis-packet.json");
  const markdownPath = join(directory, "README.md");
  const packetSha256 = await recordPacketExport(ledger, packet, packetPath, markdownPath);
  return {
    packetPath,
    markdownPath,
    sessionId,
    familyCount: packet.assignments.length,
    candidateCount: packet.assignments.reduce((sum, assignment) => sum + assignment.candidates.length, 0),
    structuralDispositionCount: packet.assignments.reduce(
      (sum, assignment) => sum + assignment.structuralDispositions.length,
      0
    ),
    packetSha256,
    resumed
  };
}

export async function submitFamilySynthesisPacket(
  ledger: Ledger,
  path: string
): Promise<SubmittedFamilySynthesis> {
  const submissionBytes = readFileSync(resolve(path));
  const packet = parseFamilySynthesisPacketText(submissionBytes.toString("utf8"));
  if (packet.rubricSlug !== FAMILY_SYNTHESIS_RUBRIC_SLUG
    || packet.rubricVersion !== FAMILY_SYNTHESIS_RUBRIC_VERSION) {
    throw new Error("Family-synthesis packet uses an unsupported rubric version");
  }
  const campaignIdValue = await campaignId(ledger, packet.campaignSlug);
  const actorId = await requireHumanActor(ledger, packet.reviewerAlias);
  const candidateRubricVersionId = await requireHumanReviewRubric(ledger);
  const rubricVersionId = await requireFamilySynthesisRubric(ledger);
  const evidence = await loadFamilyEvidence(
    ledger,
    packet.campaignSlug,
    actorId,
    candidateRubricVersionId
  );
  const expectedGlobalSnapshot = overallSnapshotSha256(packet.campaignSlug, packet.reviewerAlias, evidence);
  if (packet.inputSnapshotSha256 !== expectedGlobalSnapshot) {
    throw new Error("Family-synthesis campaign input changed since packet preparation");
  }
  if (packet.assignments.length !== evidence.length) {
    throw new Error("Family-synthesis packet does not cover every current family");
  }
  const validated: Array<{
    packetAssignment: FamilySynthesisPacketAssignment;
    evidence: FamilyEvidence;
  }> = [];
  for (const packetAssignment of packet.assignments) {
    const family = evidence.find((entry) => entry.familyVersionId === packetAssignment.familyVersionId);
    if (!family) throw new Error(`Unknown family version ${packetAssignment.familyVersionId}`);
    const packetEvidenceSha = sha256Bytes(canonicalJson(assignmentEvidencePayload(packetAssignment)));
    if (packetEvidenceSha !== family.familyInputSnapshotSha256
      || packetAssignment.familyInputSnapshotSha256 !== family.familyInputSnapshotSha256) {
      throw new Error(`Family evidence changed for ${packetAssignment.familySlug}`);
    }
    const assignmentRow = await ledger.client.execute({
      sql: `SELECT status, campaign_id, reviewer_actor_id, rubric_version_id, session_id, input_snapshot_sha256
            FROM family_synthesis_assignment WHERE id = ? AND family_version_id = ?`,
      args: [packetAssignment.assignmentId, packetAssignment.familyVersionId]
    });
    if (assignmentRow.rows.length !== 1) throw new Error(`Unknown Pass C assignment ${packetAssignment.assignmentId}`);
    const row = assignmentRow.rows[0]!;
    if (String(row["status"]) !== "assigned" || String(row["campaign_id"]) !== campaignIdValue
      || String(row["reviewer_actor_id"]) !== actorId || String(row["rubric_version_id"]) !== rubricVersionId
      || String(row["session_id"]) !== packet.sessionId
      || String(row["input_snapshot_sha256"]) !== family.familyInputSnapshotSha256) {
      throw new Error(`Pass C assignment ${packetAssignment.assignmentId} is not open under this packet contract`);
    }
    const errors = familySynthesisAssignmentErrors(packetAssignment);
    if (errors.length > 0) throw new Error(errors.join("\n"));
    validated.push({ packetAssignment, evidence: family });
  }

  const packetEnvelopeSha256 = await requireExportedPacketEnvelope(ledger, {
    format: "human_family_synthesis_packet_json",
    sessionId: packet.sessionId,
    pass: "C",
    envelopeJson: familySynthesisPacketEnvelopeJson(packet)
  }).catch((error: unknown) => {
    if (error instanceof Error && error.message === "Submission immutable envelope does not match an exported packet") {
      throw new Error("Family-synthesis submission immutable envelope does not match an exported packet");
    }
    throw error;
  });

  const submissionSha256 = await putBlob(ledger, submissionBytes, "application/json");
  const ts = now();
  const statements: Array<{ sql: string; args: InValue[] }> = [{
    sql: `INSERT OR IGNORE INTO raw_artifact(id, task_id, kind, blob_sha256, created_at)
          VALUES (?, NULL, 'human_family_synthesis_submission', ?, ?)`,
    args: [stableId("artifact", `human-family-synthesis:${packet.sessionId}:${submissionSha256}`), submissionSha256, ts]
  }];
  let structuralDispositionCount = 0;
  for (const entry of validated) {
    const assignment = entry.packetAssignment;
    const response = assignment.response;
    const familySynthesisId = stableId(
      "family_synthesis",
      `${assignment.assignmentId}:${submissionSha256}`
    );
    const diagnosis = {
      strongestCandidateVersionId: response.strongestCandidateVersionId,
      strongestCandidateRationale: response.strongestCandidateRationale,
      weakestCandidateVersionId: response.weakestCandidateVersionId,
      weakestCandidateRationale: response.weakestCandidateRationale,
      semanticDuplicateGroups: response.semanticDuplicateGroups,
      sharedConceptualError: response.sharedConceptualError,
      sharedStyleSignature: response.sharedStyleSignature,
      responsePolicyImbalance: response.responsePolicyImbalance,
      metadataTaxonomyMismatch: response.metadataTaxonomyMismatch,
      highestLeverageBlueprintRepair: response.highestLeverageBlueprintRepair,
      negativeCandidateVersionIds: response.negativeCandidateVersionIds,
      uncertaintyOrTheoryDisagreement: response.uncertaintyOrTheoryDisagreement
    };
    statements.push({
      sql: `INSERT INTO family_synthesis
            (id, assignment_id, family_version_id, reviewer_actor_id, disposition, central_distinction,
             coverage_json, diagnosis_json, rationale, confidence, submission_blob_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [familySynthesisId, assignment.assignmentId, assignment.familyVersionId, actorId,
        response.disposition!, response.centralDistinction, canonicalJson(response.coverage as unknown as JsonValue),
        canonicalJson(diagnosis as unknown as JsonValue), response.rationale, response.confidence!,
        submissionSha256, ts]
    });
    for (const candidate of entry.evidence.candidates) {
      for (const review of candidate.reviews) {
        statements.push({
          sql: `INSERT INTO family_synthesis_basis
                (id, family_synthesis_id, candidate_version_id, review_id, review_pass, created_at)
                VALUES (?, ?, ?, ?, ?, ?)`,
          args: [stableId("family_synthesis_basis", `${familySynthesisId}:${review.reviewId}`),
            familySynthesisId, candidate.candidateVersionId, review.reviewId, review.pass, ts]
        });
      }
    }
    for (const disposition of assignment.structuralDispositions) {
      structuralDispositionCount++;
      const dispositionId = stableId(
        "structural_disposition",
        `${familySynthesisId}:${disposition.candidateVersionId}`
      );
      statements.push({
        sql: `INSERT INTO structural_disposition
              (id, family_synthesis_id, candidate_version_id, reviewer_actor_id, content_utility,
               validator_finding_correctness, identified_value, semantic_type, remedy,
               automatic_acceptance_hazard, automatic_rejection_hazard, rationale, confidence, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [dispositionId, familySynthesisId, disposition.candidateVersionId, actorId,
          disposition.contentUtility!, disposition.validatorFindingCorrectness!, disposition.identifiedValue,
          disposition.semanticType!, disposition.remedy!, disposition.automaticAcceptanceHazard,
          disposition.automaticRejectionHazard, disposition.rationale, disposition.confidence!, ts]
      });
      const candidate = entry.evidence.candidates.find(
        (item) => item.candidateVersionId === disposition.candidateVersionId
      )!;
      for (const failure of candidate.failures) {
        statements.push({
          sql: `INSERT INTO structural_disposition_basis
                (id, structural_disposition_id, basis_kind, basis_id, created_at)
                VALUES (?, ?, 'candidate_failure', ?, ?)`,
          args: [stableId("structural_disposition_basis", `${dispositionId}:failure:${failure.failureId}`),
            dispositionId, failure.failureId, ts]
        });
      }
      for (const review of candidate.reviews) {
        statements.push({
          sql: `INSERT INTO structural_disposition_basis
                (id, structural_disposition_id, basis_kind, basis_id, created_at)
                VALUES (?, ?, 'review', ?, ?)`,
          args: [stableId("structural_disposition_basis", `${dispositionId}:review:${review.reviewId}`),
            dispositionId, review.reviewId, ts]
        });
      }
    }
    statements.push({
      sql: `UPDATE family_synthesis_assignment SET status = 'completed', updated_at = ?
            WHERE id = ? AND status = 'assigned'`,
      args: [ts, assignment.assignmentId]
    });
    const eventId = stableId("event", `human-family-synthesis-submitted:${familySynthesisId}`);
    statements.push({
      sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
            VALUES (?, 'human_family_synthesis_submitted', 'family_synthesis', ?, ?, ?)`,
      args: [eventId, familySynthesisId, canonicalJson({
        assignmentId: assignment.assignmentId,
        familyVersionId: assignment.familyVersionId,
        sessionId: packet.sessionId,
        packetEnvelopeSha256,
        submissionSha256
      } as JsonValue), ts]
    });
  }
  await ledger.client.batch(statements, "write");
  return {
    sessionId: packet.sessionId,
    familySyntheses: validated.length,
    structuralDispositions: structuralDispositionCount,
    packetEnvelopeSha256,
    submissionSha256
  };
}

export async function familySynthesisStatus(
  ledger: Ledger,
  campaignSlug: string
): Promise<FamilySynthesisStatus> {
  const campaignIdValue = await campaignId(ledger, campaignSlug);
  const assignments = await ledger.client.execute({
    sql: `SELECT status AS key, COUNT(*) AS count FROM family_synthesis_assignment
          WHERE campaign_id = ? GROUP BY status`,
    args: [campaignIdValue]
  });
  const synthesisCount = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM family_synthesis fs
          JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
          WHERE fsa.campaign_id = ?`,
    args: [campaignIdValue]
  });
  const dispositionCount = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM structural_disposition sd
          JOIN family_synthesis fs ON fs.id = sd.family_synthesis_id
          JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
          WHERE fsa.campaign_id = ?`,
    args: [campaignIdValue]
  });
  const artifacts = await ledger.client.execute(
    "SELECT COUNT(*) AS count FROM raw_artifact WHERE kind = 'human_family_synthesis_submission'"
  );
  const releaseMembers = await ledger.client.execute("SELECT COUNT(*) AS count FROM release_member");
  const trainingExposures = await ledger.client.execute("SELECT COUNT(*) AS count FROM training_exposure");
  return {
    campaignSlug,
    assignments: Object.fromEntries(assignments.rows.map((row) => [String(row["key"]), Number(row["count"])])),
    familySyntheses: Number(synthesisCount.rows[0]!["count"]),
    structuralDispositions: Number(dispositionCount.rows[0]!["count"]),
    familySynthesisArtifacts: Number(artifacts.rows[0]!["count"]),
    releaseMembers: Number(releaseMembers.rows[0]!["count"]),
    trainingExposures: Number(trainingExposures.rows[0]!["count"])
  };
}
